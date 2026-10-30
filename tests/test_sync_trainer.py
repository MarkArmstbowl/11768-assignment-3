"""Offline checks for the synchronous trainer's sampling and update lifecycle.

The concurrency check uses an event barrier rather than timing sleeps. All
sampling clients, updates, and evaluations are local fakes: no backend starts.
"""

import asyncio
import random
from dataclasses import dataclass

import pytest

import scottyrl.train.sync as sync_module
from scottyrl.config import ModelConfig, TrainConfig
from scottyrl.train.sync import SyncTrainer
from scottyrl.types import LLMCall, Task, TokenSequence, Trajectory


@dataclass(frozen=True)
class SamplingClient:
    name: str
    version: int


@dataclass(frozen=True)
class WrappedPolicy:
    client: SamplingClient

    @property
    def name(self):
        return self.client.name

    @property
    def version(self):
        return self.client.version


class FakeSession:
    def __init__(self):
        self.version = 0
        self.clients = []
        self.policies = []
        self.updates = []
        self.update_metrics = {"loss": 0.125}

    async def new_sampling_client(self, name):
        client = SamplingClient(name, self.version)
        self.clients.append(client)
        return client

    async def train_step(self, seqs, *, loss_fn, lr, grad_clip_norm):
        self.updates.append((seqs, loss_fn, lr, grad_clip_norm))
        self.version += 1
        return self.update_metrics


class FakeLogger:
    def __init__(self, step_offset=0):
        self.step_offset = step_offset
        self.records = []

    def log(self, step, metrics):
        # Match the real logger's ownership of absolute step numbering.
        self.records.append((step + self.step_offset, dict(metrics)))


def make_sequence():
    return TokenSequence([1, 2], [1], [-0.5], [0.5])


def make_trajectory(task, reward):
    return Trajectory(task, [LLMCall([1, 2], [3], [-0.5])], reward, 1)


def make_trainer(*, step_offset=0, **overrides):
    settings = {
        "group_size": 2,
        "tasks_per_batch": 3,
        "num_steps": 3,
        "eval_every": 0,
        "eval_tasks": 2,
    }
    settings.update(overrides)
    session = FakeSession()
    logger = FakeLogger(step_offset)
    trainer = SyncTrainer(
        session=session,
        env_factory=lambda task: None,
        train_tasks=[Task("find", f"train-{i}", 1) for i in range(7)],
        val_tasks=[Task("find", f"val-{i}", 1) for i in range(5)],
        model_cfg=ModelConfig(),
        train_cfg=TrainConfig(**settings),
        logger=logger,
    )

    def make_policy(client):
        policy = WrappedPolicy(client)
        session.policies.append(policy)
        return policy

    trainer.make_policy = make_policy
    return trainer, session, logger


def records_with_prefix(logger, prefix):
    return [
        (step, metrics)
        for step, metrics in logger.records
        if any(key.startswith(prefix) for key in metrics)
    ]


def install_scripted_batches(trainer, nonempty_steps):
    """Replace rollout work, retaining the real update and logging helpers."""
    collected_policies = []
    evaluated_policies = []
    prepared_batches = {}

    async def collect_batch(policy):
        assert isinstance(policy, WrappedPolicy), "wrap each sampling client with make_policy"
        has_sequences = nonempty_steps[len(collected_policies)]
        collected_policies.append(policy)
        task = trainer.train_tasks[0]
        groups = [[make_trajectory(task, 1.0), make_trajectory(task, 0.0 if has_sequences else 1.0)]]
        seqs = [make_sequence()] if has_sequences else []
        stats = {
            "num_groups": 1,
            "num_degenerate_groups": int(not has_sequences),
            "frac_degenerate_groups": float(not has_sequences),
            "mean_reward": 0.5 if has_sequences else 1.0,
            "num_sequences": len(seqs),
            "num_action_tokens": len(seqs),
        }
        prepared_batches[id(groups)] = (seqs, stats)
        return groups

    def build_batch(groups):
        return prepared_batches.pop(id(groups))

    async def evaluate_now(policy):
        assert isinstance(policy, WrappedPolicy)
        evaluated_policies.append(policy)
        return {"sampler": policy.name, "version": policy.version, "success_rate": 0.5}

    trainer.collect_batch = collect_batch
    trainer.build_batch = build_batch
    trainer.evaluate_now = evaluate_now
    return collected_policies, evaluated_policies


def test_collect_batch_samples_distinct_tasks_reproducibly_and_collects_them_concurrently(monkeypatch):
    trainer, _, _ = make_trainer(tasks_per_batch=3, group_size=4, seed=321)
    original_tasks = list(trainer.train_tasks)
    expected_rng = random.Random(trainer.cfg.seed)
    expected_batches = [
        expected_rng.sample(trainer.train_tasks, trainer.cfg.tasks_per_batch)
        for _ in range(2)
    ]
    policy = object()
    agent_cls = object()
    trainer.agent_cls = lambda: agent_cls
    calls = []

    async def scenario():
        # All task groups must start before any one of them can finish.
        # A sequential implementation hits wait_for instead of hanging.
        barriers = [asyncio.Event(), asyncio.Event()]

        async def collect_group(policy, task, env_factory, group_size, agent_cls):
            batch_index = len(calls) // trainer.cfg.tasks_per_batch
            calls.append((policy, task, env_factory, group_size, agent_cls))
            if len(calls) % trainer.cfg.tasks_per_batch == 0:
                barriers[batch_index].set()
            await barriers[batch_index].wait()
            return [make_trajectory(task, 0.0) for _ in range(group_size)]

        monkeypatch.setattr(sync_module, "collect_group", collect_group)
        return [
            await asyncio.wait_for(trainer.collect_batch(policy), timeout=2.0)
            for _ in range(2)
        ]

    batches = asyncio.run(scenario())

    assert [[group[0].task for group in batch] for batch in batches] == expected_batches
    for batch in batches:
        assert len({group[0].task.id for group in batch}) == trainer.cfg.tasks_per_batch
        assert all(len(group) == trainer.cfg.group_size for group in batch)
    assert trainer.train_tasks == original_tasks
    assert len(calls) == 2 * trainer.cfg.tasks_per_batch
    for given_policy, _, env_factory, group_size, given_agent in calls:
        assert given_policy is policy
        assert env_factory is trainer.env_factory
        assert group_size == 4
        assert given_agent is agent_cls


def test_collect_batch_failure_cancels_and_awaits_sibling_groups(monkeypatch):
    async def scenario():
        trainer, _, _ = make_trainer(tasks_per_batch=3)
        failure = RuntimeError("one task group failed")
        siblings_started = asyncio.Event()
        blocked = asyncio.Event()
        groups = []
        cleaned = set()

        async def collect_group(*args, **kwargs):
            index = len(groups)
            groups.append(asyncio.current_task())
            if index == 0:
                await siblings_started.wait()
                raise failure
            try:
                if len(groups) == 3:
                    siblings_started.set()
                await blocked.wait()
            finally:
                await asyncio.sleep(0)
                cleaned.add(index)

        monkeypatch.setattr(sync_module, "collect_group", collect_group)
        try:
            with pytest.raises(RuntimeError) as caught:
                await asyncio.wait_for(trainer.collect_batch(object()), timeout=2.0)
            assert caught.value is failure
            assert cleaned == {1, 2}
            assert len(groups) == 3 and all(task.done() for task in groups)
        finally:
            for task in groups:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*groups, return_exceptions=True)

    asyncio.run(scenario())


def test_collect_batch_outer_cancellation_awaits_all_group_cleanup(monkeypatch):
    async def scenario():
        trainer, _, _ = make_trainer(tasks_per_batch=3)
        all_started = asyncio.Event()
        blocked = asyncio.Event()
        groups = []
        cleaned = set()

        async def collect_group(*args, **kwargs):
            index = len(groups)
            groups.append(asyncio.current_task())
            try:
                if len(groups) == 3:
                    all_started.set()
                await blocked.wait()
            finally:
                await asyncio.sleep(0)
                cleaned.add(index)

        monkeypatch.setattr(sync_module, "collect_group", collect_group)
        collection = asyncio.create_task(trainer.collect_batch(object()))
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2.0)
            collection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await collection
            assert cleaned == {0, 1, 2}
            assert all(task.done() for task in groups)
        finally:
            collection.cancel()
            await asyncio.gather(collection, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["grpo", "reinforce"])
def test_build_batch_delegates_the_selected_advantage_rule_and_preserves_results(monkeypatch, method):
    trainer, _, logger = make_trainer(algo=method, group_size=1 if method == "reinforce" else 2)
    rewards = [1.0] if method == "reinforce" else [1.0, 0.0]
    groups = [[make_trajectory(trainer.train_tasks[0], reward) for reward in rewards]]
    sequences = [make_sequence()]
    stats = {"num_groups": 1, "custom_stat": 17}
    calls = []

    def groups_to_batch(groups, method):
        calls.append((groups, method))
        return sequences, stats

    monkeypatch.setattr(sync_module, "groups_to_batch", groups_to_batch)
    actual_sequences, actual_stats = trainer.build_batch(groups)

    assert len(calls) == 1
    assert calls[0][0] is groups
    assert calls[0][1] == method
    assert actual_sequences is sequences
    assert actual_stats is stats
    assert logger.records == []


def test_update_forwards_optimizer_settings_and_returns_metrics_without_logging():
    trainer, session, logger = make_trainer(loss_fn="cross_entropy", lr=3e-5, grad_clip_norm=0.25)
    sequences = [make_sequence(), make_sequence()]

    metrics = asyncio.run(trainer.update(sequences))

    assert len(session.updates) == 1
    given_sequences, loss_fn, lr, grad_clip_norm = session.updates[0]
    assert given_sequences is sequences
    assert (loss_fn, lr, grad_clip_norm) == ("cross_entropy", 3e-5, 0.25)
    assert metrics is session.update_metrics
    assert logger.records == []


def test_update_propagates_session_failure_without_replacing_it():
    trainer, session, logger = make_trainer()
    failure = RuntimeError("fake optimizer failed")

    async def fail_step(*args, **kwargs):
        raise failure

    session.train_step = fail_step
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(trainer.update([make_sequence()]))

    assert raised.value is failure
    assert logger.records == []


def test_evaluate_now_uses_validation_prefix_and_configured_agent_and_k(monkeypatch):
    trainer, _, logger = make_trainer(eval_tasks=3, eval_k=5)
    original_tasks = list(trainer.val_tasks)
    policy = object()
    agent_cls = object()
    trainer.agent_cls = lambda: agent_cls
    expected_metrics = {"success_rate": 0.4, "pass@5": 0.8}
    calls = []

    async def evaluate_policy(policy, tasks, env_factory, *, k, agent_cls):
        calls.append((policy, tasks, env_factory, k, agent_cls))
        return expected_metrics

    monkeypatch.setattr(sync_module, "evaluate_policy", evaluate_policy)
    result = asyncio.run(trainer.evaluate_now(policy))

    assert len(calls) == 1
    given_policy, tasks, env_factory, k, given_agent = calls[0]
    assert given_policy is policy
    assert tasks == original_tasks[:3]
    assert trainer.val_tasks == original_tasks
    assert env_factory is trainer.env_factory
    assert k == 5
    assert given_agent is agent_cls
    assert result is expected_metrics
    assert logger.records == []


def test_train_uses_refreshed_policy_after_every_update_and_only_periodic_evaluation():
    trainer, session, logger = make_trainer(num_steps=3, eval_every=2)
    collected, evaluated = install_scripted_batches(trainer, [True, True, True])

    result = asyncio.run(trainer.train())

    assert result is None
    assert [(policy.name, policy.version) for policy in collected] == [
        ("step_000000", 0), ("step_000001", 1), ("step_000002", 2)
    ]
    assert [(client.name, client.version) for client in session.clients] == [
        ("step_000000", 0), ("step_000001", 1), ("step_000002", 2), ("step_000003", 3)
    ]
    assert len(session.policies) == 4
    assert len(session.updates) == 3
    assert [step for step, _ in records_with_prefix(logger, "train/")] == [1, 2, 3]
    assert [(policy.name, policy.version) for policy in evaluated] == [("step_000002", 2)]
    eval_records = records_with_prefix(logger, "eval/")
    assert [(step, metrics["eval/sampler"]) for step, metrics in eval_records] == [(2, "step_000002")]
    assert all(step != 0 for step, _ in logger.records), "the entry script owns the baseline"


def test_train_skipped_step_reuses_policy_but_still_records_and_evaluates():
    trainer, session, logger = make_trainer(num_steps=3, eval_every=1)
    collected, evaluated = install_scripted_batches(trainer, [True, False, True])

    asyncio.run(trainer.train())

    assert [(policy.name, policy.version) for policy in collected] == [
        ("step_000000", 0), ("step_000001", 1), ("step_000001", 1)
    ]
    assert [(client.name, client.version) for client in session.clients] == [
        ("step_000000", 0), ("step_000001", 1), ("step_000003", 2)
    ]
    assert len(session.policies) == 3
    assert len(session.updates) == 2
    records = records_with_prefix(logger, "train/")
    assert [step for step, _ in records] == [1, 2, 3]
    assert records[1][1]["train/skipped"] == 1
    assert "train/loss" not in records[1][1]
    assert "train/skipped" not in records[0][1]
    assert "train/skipped" not in records[2][1]
    assert [(policy.name, policy.version) for policy in evaluated] == [
        ("step_000001", 1), ("step_000001", 1), ("step_000003", 2)
    ]
    assert [step for step, _ in records_with_prefix(logger, "eval/")] == [1, 2, 3]


def test_train_resumed_evaluation_uses_absolute_step_cadence_without_repeating_baseline():
    trainer, session, logger = make_trainer(step_offset=45, num_steps=7, eval_every=10)
    collected, evaluated = install_scripted_batches(trainer, [True] * 7)

    asyncio.run(trainer.train())

    assert len(collected) == len(session.updates) == 7
    assert [step for step, _ in records_with_prefix(logger, "train/")] == list(range(46, 53))
    assert [(policy.name, policy.version) for policy in evaluated] == [("step_000005", 5)]
    eval_records = records_with_prefix(logger, "eval/")
    assert [(step, metrics["eval/version"]) for step, metrics in eval_records] == [(50, 5)]
    # TinkerSession, rather than the trainer, converts relative snapshot names
    # to resumed absolute names; the fake keeps the submitted names visible.
    assert session.clients[0].name == "step_000000"
