"""In-process worker tests: the same Worker class both containers run."""
from __future__ import annotations

import threading
import time

import numpy as np

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db
from app.worker import Worker


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.02,
        batch_delay_seconds=0.01,
    )
    base.update(overrides)
    return Settings(**base)


def sample(records: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def run_worker(settings: Settings, worker_id: str) -> tuple[Worker, threading.Thread]:
    worker = Worker(settings, worker_id)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    return worker, thread


def stop(worker: Worker, thread: threading.Thread) -> None:
    worker.request_stop()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_worker_trains_job_to_completion(tmp_path):
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(20), 2)
    worker, thread = run_worker(settings, "w1")
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
    finally:
        stop(worker, thread)
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"] == 6
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == [1, 2, 3, 4, 5, 6]
    assert {c["worker_id"] for c in checkpoints} == {"w1"}
    assert all(c["lease_epoch"] == 1 for c in checkpoints)
    assert lm.holder(settings.db_path, job["job_id"]) is None  # released on completion


def test_two_workers_never_train_the_same_job_concurrently(tmp_path):
    settings = make_settings(tmp_path, batch_delay_seconds=0.05)
    init_db(settings.db_path)
    jobs = [store.create_job(settings.db_path, *sample(16, seed=i), 2) for i in range(2)]
    workers = [run_worker(settings, f"w{i}") for i in (1, 2)]
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if all(
                store.get_job(settings.db_path, job["job_id"])["status"] == "completed"
                for job in jobs
            ):
                break
            time.sleep(0.02)
    finally:
        for worker, thread in workers:
            stop(worker, thread)

    for job in jobs:
        final = store.get_job(settings.db_path, job["job_id"])
        assert final["status"] == "completed"
        assert final["completed_steps"] == final["total_steps"]
        checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
        steps = [c["step"] for c in checkpoints]
        assert steps == list(range(1, final["total_steps"] + 1))
        # Exactly one lease epoch may write a given run of consecutive steps
        # per owner; a takeover would be visible as an epoch increase.
        epochs = [c["lease_epoch"] for c in checkpoints]
        assert epochs == sorted(epochs)


def test_worker_survives_a_crash_preventing_commit(tmp_path):
    """A worker that loses its lease midway leaves a clean resumable checkpoint."""
    settings = make_settings(tmp_path, batch_delay_seconds=0.0)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(24, seed=3), 3)

    lm.register_worker(settings.db_path, "first")
    first = lm.acquire(settings.db_path, job["job_id"], "first", settings.lease_seconds)
    for _ in range(2):
        store.advance_job(settings.db_path, first)
    # "first" dies here; a new worker takes over after expiry.
    time.sleep(settings.lease_seconds + 0.05)

    second_worker, thread = run_worker(settings, "second")
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
    finally:
        stop(second_worker, thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == list(range(1, 10))
    owners = [c["worker_id"] for c in checkpoints]
    assert owners[:2] == ["first", "first"]
    assert set(owners[2:]) == {"second"}
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs[:2] == [1, 1]
    assert epochs[2:] == [2] * 7


def test_worker_skips_a_job_completed_between_scan_and_claim(tmp_path):
    """B's candidate list is stale: A already completed the job.

    B must not record a start event for it, must not fail, and must still be
    able to pick up a newly submitted pending job afterwards.
    """
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    features, labels = sample(16, seed=7)
    finished = store.create_job(settings.db_path, features, labels, 2)

    lm.register_worker(settings.db_path, "A")
    lease = lm.acquire(settings.db_path, finished["job_id"], "A", settings.lease_seconds)
    while store.advance_job(settings.db_path, lease)["status"] != "completed":
        pass
    lm.release(settings.db_path, lease)
    events_before = store.list_events(settings.db_path, finished["job_id"])

    # B scanned the directory before A finished and still holds the id.
    worker = Worker(settings, "B")
    worker.lease = lm.Lease(finished["job_id"], "B", 99)
    worker._train()  # must not raise and must not log a start
    assert store.list_events(settings.db_path, finished["job_id"]) == events_before
    worker.lease = None

    pending = store.create_job(settings.db_path, features, labels, 2)
    worker, thread = run_worker(settings, "B")
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, pending["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
    finally:
        stop(worker, thread)
    final = store.get_job(settings.db_path, pending["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]


def test_poll_loop_drops_a_stale_completed_candidate_and_keeps_serving(tmp_path, monkeypatch):
    """Reproduces the reported interleaving end to end at the worker level.

    A completes J1 and releases it while B is working from a scan that still
    listed J1 as pending.  B must not claim J1, must not log a start event for
    it, and must keep polling - so a newly submitted J2 still gets trained.
    """
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    features, labels = sample(16, seed=9)
    finished = store.create_job(settings.db_path, features, labels, 2)

    # Worker A finishes and releases the job before B tries to claim it.
    lm.register_worker(settings.db_path, "A")
    lease_a = lm.acquire(settings.db_path, finished["job_id"], "A", settings.lease_seconds)
    while store.advance_job(settings.db_path, lease_a)["status"] != "completed":
        pass
    lm.release(settings.db_path, lease_a)
    events_before = store.list_events(settings.db_path, finished["job_id"])
    completed_before = store.get_job(settings.db_path, finished["job_id"])

    real_claimable = store.claimable_jobs
    scans = {"count": 0}

    def stale_first_scans(db_path, limit=20):
        scans["count"] += 1
        if scans["count"] <= 3:
            return [finished["job_id"]]  # B's stale snapshot
        return real_claimable(db_path, limit)

    monkeypatch.setattr(store, "claimable_jobs", stale_first_scans)

    worker, thread = run_worker(settings, "B")
    try:
        deadline = time.time() + 10
        while time.time() < deadline and scans["count"] < 2:
            time.sleep(0.01)
        pending = store.create_job(settings.db_path, features, labels, 2)
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, pending["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
    finally:
        stop(worker, thread)

    assert scans["count"] > 3, "B never got past its stale candidate list"
    # J1 untouched: same events, same completion state, no B start event.
    assert store.list_events(settings.db_path, finished["job_id"]) == events_before
    assert store.get_job(settings.db_path, finished["job_id"]) == completed_before
    started = [
        e
        for e in store.list_events(settings.db_path, pending["job_id"])
        if e["event"] == "training_started"
    ]
    assert [e["detail"]["worker_id"] for e in started] == ["B"]
    final = store.get_job(settings.db_path, pending["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]


def test_one_failing_job_does_not_take_the_worker_out_of_service(tmp_path, monkeypatch):
    """An unexpected job-level error is logged, released, and the poll loop
    continues with the next pending job (the reported 'new jobs stay queued'
    symptom)."""
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    features, labels = sample(16, seed=8)
    first = store.create_job(settings.db_path, features, labels, 2)
    second = store.create_job(settings.db_path, features, labels, 2)

    real_advance = store.advance_job
    failed_once = {"done": False}

    def flaky(db_path, lease, batch_delay=0.0):
        if lease.job_id == first["job_id"] and not failed_once["done"]:
            failed_once["done"] = True
            raise RuntimeError("boom")
        return real_advance(db_path, lease, batch_delay)

    monkeypatch.setattr(store, "advance_job", flaky)

    worker, thread = run_worker(settings, "w1")
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            jobs = [store.get_job(settings.db_path, job["job_id"]) for job in (first, second)]
            if all(job["status"] == "completed" for job in jobs):
                break
            time.sleep(0.02)
    finally:
        stop(worker, thread)

    assert failed_once["done"] is True
    for job in (first, second):
        final = store.get_job(settings.db_path, job["job_id"])
        assert final["status"] == "completed"
        assert final["completed_steps"] == final["total_steps"]
        steps = [c["step"] for c in store.list_checkpoints(settings.db_path, job["job_id"])]
        assert steps == list(range(1, final["total_steps"] + 1))
