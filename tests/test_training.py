"""Deterministic training tests: maths, checkpoints and crash resumption."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db
from app.features import normalise
from app.lease import LeaseLost
from app.model import batch_loss, sgd_step


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    return s


def sample(records: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def reference_train(features, labels, epochs, lr=0.1, momentum=0.9, batch_size=8):
    """Independent replay of the training loop used as the oracle."""
    design, targets = normalise(features, labels)
    weights = np.zeros(design.shape[1])
    velocity = np.zeros(design.shape[1])
    losses = []
    steps = 0
    for _ in range(epochs):
        for start in range(0, design.shape[0], batch_size):
            weights, velocity, loss = sgd_step(
                weights,
                velocity,
                design[start : start + batch_size],
                targets[start : start + batch_size],
                lr,
                momentum,
            )
            losses.append(loss)
            steps += 1
    return weights, velocity, steps, losses


def drain(settings, job_id, lease, limit=1000):
    """Advance a job to completion as ``lease``'s owner; returns steps."""
    seen = []
    while True:
        result = store.advance_job(settings.db_path, lease)
        seen.append(result["step"])
        if result["status"] == "completed":
            return seen
        assert len(seen) < limit


def test_math_matches_reference(settings):
    features, labels = sample(11)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, job["job_id"], "w", settings.lease_seconds)
    assert drain(settings, job["job_id"], lease) == list(range(1, 5))
    final = store.get_job(settings.db_path, job["job_id"])
    w, v, steps, losses = reference_train(features, labels, 2)
    assert steps == 4
    assert np.allclose(final["weights"], w, rtol=0, atol=1e-12)
    assert np.allclose(final["momentum_velocity"], v, rtol=0, atol=1e-12)
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == [1, 2, 3, 4]
    assert np.allclose([c["batch_loss"] for c in checkpoints], losses, atol=1e-12)


def test_hyperparameters_and_batching(settings):
    features, labels = sample(18)
    job = store.create_job(settings.db_path, features, labels, 3)
    assert job["batch_size"] == 8
    assert job["learning_rate"] == 0.1
    assert job["momentum"] == 0.9
    assert job["batches_per_epoch"] == math.ceil(18 / 8) == 3
    assert job["total_steps"] == 9


def test_each_batch_is_committed_with_weights_momentum_loss_progress(settings):
    features, labels = sample(10)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, job["job_id"], "w", settings.lease_seconds)
    store.advance_job(settings.db_path, lease)
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    latest = checkpoints[-1]
    current = store.get_job(settings.db_path, job["job_id"])
    assert set(latest) >= {
        "weights",
        "momentum_velocity",
        "batch_loss",
        "loss",
        "step",
        "committed_at",
        "worker_id",
        "lease_epoch",
    }
    assert latest["step"] == current["completed_steps"]
    assert latest["weights"] == current["weights"]
    assert latest["loss"] == pytest.approx(current["loss"], abs=1e-12)
    assert current["progress"] == pytest.approx(0.25)


def test_resume_from_committed_checkpoint_matches_uninterrupted(settings):
    features, labels = sample(19, seed=4)
    uninterrupted = store.create_job(settings.db_path, features, labels, 3)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, uninterrupted["job_id"], "w", settings.lease_seconds)
    drain(settings, uninterrupted["job_id"], lease)
    expected = store.get_job(settings.db_path, uninterrupted["job_id"])

    interrupted = store.create_job(settings.db_path, features, labels, 3)
    lease = lm.acquire(settings.db_path, interrupted["job_id"], "w", settings.lease_seconds)
    for _ in range(3):
        store.advance_job(settings.db_path, lease)
    # Simulate the process dying: lease is abandoned, then expires.
    lm.release(settings.db_path, lease)
    time.sleep(settings.lease_seconds + 0.05)
    resumed_lease = lm.acquire(
        settings.db_path, interrupted["job_id"], "w2", settings.lease_seconds
    )
    assert resumed_lease is not None
    drain(settings, interrupted["job_id"], resumed_lease)
    resumed = store.get_job(settings.db_path, interrupted["job_id"])

    assert resumed["completed_steps"] == expected["completed_steps"]
    assert resumed["final_weights"] == expected["final_weights"]
    assert resumed["momentum_velocity"] == expected["momentum_velocity"]
    assert resumed["loss"] == expected["loss"]
    steps = [c["step"] for c in store.list_checkpoints(settings.db_path, interrupted["job_id"])]
    assert steps == list(range(1, expected["total_steps"] + 1))


def test_stale_worker_cannot_write_after_handover(settings):
    features, labels = sample(20, seed=9)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    store.advance_job(settings.db_path, old)
    assert lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds) is None
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert new is not None and new.epoch == old.epoch + 1
    store.advance_job(settings.db_path, new)
    before = store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["completed_steps"] == before["completed_steps"]
    assert after["weights"] == before["weights"]


def test_stale_worker_cannot_report_completion(settings):
    features, labels = sample(8, seed=2)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert new is not None
    while True:
        result = store.advance_job(settings.db_path, new)
        if result["status"] == "completed":
            break
    completed = store.get_job(settings.db_path, job["job_id"])
    assert completed["status"] == "completed"
    assert completed["completed_steps"] == completed["total_steps"]
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == completed["completed_steps"]
    assert final["finished_at"] == completed["finished_at"]


def test_late_worker_cannot_roll_back_checkpoint(settings):
    features, labels = sample(16, seed=11)
    job = store.create_job(settings.db_path, features, labels, 3)
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    store.advance_job(settings.db_path, old)
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    for _ in range(3):
        store.advance_job(settings.db_path, new)
    snapshot = store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["completed_steps"] == snapshot["completed_steps"]
    assert after["weights"] == snapshot["weights"]


def test_input_validation(settings):
    from app.features import InputError

    features, labels = sample(7)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels, 2)
    features, labels = sample(8)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels, 1)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels, 6)
    bad_labels = list(labels)
    bad_labels[0] = 2
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, bad_labels, 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels[:-1], 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, [[0.1, 0.2]] * 7 + [["x", 0.2]], labels, 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, [[0.1, 0.2]] * 7 + [[0.3]], labels, 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, [[0.1, 0.2]] * 7 + [[float("nan"), 0.2]], labels, 2)


def test_single_batch_loss_is_cross_entropy_of_zero_weights():
    features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.zeros(3)
    design = np.concatenate([features, np.ones((4, 1))], axis=1)
    assert batch_loss(weights, design, labels) == pytest.approx(math.log(2))


def test_reacquired_lease_never_reuses_an_epoch(settings):
    """Releasing a lease must not let a new owner inherit an old epoch.

    Otherwise an old worker still holding an expired Lease(job, w, 1) could
    write after a *new* owner was also given epoch 1.
    """
    features, labels = sample(16, seed=6)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "old")
    first = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    assert first.epoch == 1
    lm.release(settings.db_path, first)
    second = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert second is not None and second.epoch == 2
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, first)
    assert store.get_job(settings.db_path, job["job_id"])["completed_steps"] == 0
    store.advance_job(settings.db_path, second)
    assert store.get_job(settings.db_path, job["job_id"])["completed_steps"] == 1


def test_completed_job_cannot_be_claimed_from_a_stale_candidate_list(settings):
    """B scanned while the job was pending; A then completed and released it.

    B must not be able to acquire the finished job, so it never reaches the
    point of recording a start event for it.
    """
    features, labels = sample(16, seed=13)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "A")
    lm.register_worker(settings.db_path, "B")
    # B's candidate scan: the job is still unfinished here.
    stale_candidates = store.claimable_jobs(settings.db_path)
    assert job["job_id"] in stale_candidates

    lease_a = lm.acquire(settings.db_path, job["job_id"], "A", settings.lease_seconds)
    while True:
        if store.advance_job(settings.db_path, lease_a)["status"] == "completed":
            break
    lm.release(settings.db_path, lease_a)
    done = store.get_job(settings.db_path, job["job_id"])
    events_before = store.list_events(settings.db_path, job["job_id"])

    assert lm.acquire(settings.db_path, job["job_id"], "B", settings.lease_seconds) is None
    assert lm.holder(settings.db_path, job["job_id"]) is None
    assert store.start_training(
        settings.db_path, lm.Lease(job["job_id"], "B", 99)
    ) is None
    assert store.list_events(settings.db_path, job["job_id"]) == events_before
    assert store.get_job(settings.db_path, job["job_id"]) == done


def test_late_worker_cannot_record_start_after_lease_transfer(settings):
    """B holds a lease, A takes it over; B's start event must be rejected."""
    features, labels = sample(16, seed=14)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "A")
    lm.register_worker(settings.db_path, "B")

    b = lm.acquire(settings.db_path, job["job_id"], "B", settings.lease_seconds)
    assert b is not None
    time.sleep(settings.lease_seconds + 0.05)
    a = lm.acquire(settings.db_path, job["job_id"], "A", settings.lease_seconds)
    assert a is not None and a.epoch == b.epoch + 1

    events_before = store.list_events(settings.db_path, job["job_id"])
    assert store.start_training(settings.db_path, b) is None
    assert store.list_events(settings.db_path, job["job_id"]) == events_before
    # B must not release or otherwise disturb A's lease.
    assert lm.holder(settings.db_path, job["job_id"])[:2] == ("A", a.epoch)
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, b)

    started = store.start_training(settings.db_path, a)
    assert started == {
        "job_id": job["job_id"],
        "completed_steps": 0,
        "total_steps": job["total_steps"],
        "status": "queued",
    }


def test_start_event_is_not_duplicated_for_a_completed_job(settings):
    """Once a job is completed, no further start event is accepted, and the
    completion state itself is untouched."""
    features, labels = sample(8, seed=15)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, job["job_id"], "w", settings.lease_seconds)
    assert store.start_training(settings.db_path, lease) is not None
    while True:
        if store.advance_job(settings.db_path, lease)["status"] == "completed":
            break
    completed = store.get_job(settings.db_path, job["job_id"])
    events = store.list_events(settings.db_path, job["job_id"])
    assert sum(e["event"] == "training_started" for e in events) == 1
    assert sum(e["event"] == "job_completed" for e in events) == 1

    assert store.start_training(settings.db_path, lease) is None
    assert store.list_events(settings.db_path, job["job_id"]) == events
    assert store.get_job(settings.db_path, job["job_id"]) == completed


def test_advance_on_an_already_completed_job_is_a_clean_noop(settings):
    """The losing side of the completion race gets a normal result, not a
    KeyError that would kill the worker process, and writes nothing."""
    features, labels = sample(8, seed=16)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "owner")
    lease = lm.acquire(settings.db_path, job["job_id"], "owner", settings.lease_seconds)
    while True:
        if store.advance_job(settings.db_path, lease)["status"] == "completed":
            break
    completed = store.get_job(settings.db_path, job["job_id"])
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    events = store.list_events(settings.db_path, job["job_id"])

    result = store.advance_job(settings.db_path, lease)
    assert result["status"] == "completed"
    assert result["step"] == completed["total_steps"]
    assert result["loss"] == completed["loss"]
    assert result["total_steps"] == completed["total_steps"]
    assert store.list_checkpoints(settings.db_path, job["job_id"]) == checkpoints
    assert store.list_events(settings.db_path, job["job_id"]) == events
    assert store.get_job(settings.db_path, job["job_id"]) == completed

