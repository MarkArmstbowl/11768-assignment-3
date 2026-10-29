"""Offline Part 3 contract checks, with event-controlled concurrency.

All services, sampling and evaluations are local fakes. Timeouts only prevent
deadlocks; no assertion depends on an operation being faster than another.
"""

import asyncio
from dataclasses import dataclass

import pytest

from scottyrl.config import ModelConfig, TrainConfig
from scottyrl.train import async_grpo as async_module
from scottyrl.train.async_grpo import AsyncGRPOTrainer, GroupResult
from scottyrl.types import LLMCall, Task, TokenSequence, Trajectory


@dataclass
class Policy:
    name: str
    version: int = 0


class FakeSession:
    tokenizer = None

    def __init__(self):
        self.snapshots = []
        self.updates = []
        self.metrics = {"loss": 0.25}

    async def new_sampling_client(self, name):
        self.snapshots.append(name)
        return name

    async def train_step(self, sequences, *, loss_fn, lr, grad_clip_norm):
        self.updates.append((sequences, loss_fn, lr, grad_clip_norm))
        return self.metrics


class FakeLogger:
    def __init__(self):
        self.records = []
        self.step_offset = 0

    def log(self, step, metrics):
        self.records.append((step + self.step_offset, dict(metrics)))


class TrainingFailure(Exception):
    pass


def make_trainer(**overrides):
    settings = {
        "group_size": 2,
        "groups_per_batch": 1,
        "num_workers": 1,
        "queue_capacity": 1,
        "num_steps": 2,
        "eval_every": 0,
        "eval_tasks": 2,
        "max_staleness": 100,
    }
    settings.update(overrides)
    session = FakeSession()
    logger = FakeLogger()
    trainer = AsyncGRPOTrainer(
        session=session,
        env_factory=lambda task: None,
        train_tasks=[Task("train", "a", 1), Task("train", "b", 1)],
        val_tasks=[Task("val", f"val-{i}", 1) for i in range(4)],
        model_cfg=ModelConfig(),
        train_cfg=TrainConfig(**settings),
        logger=logger,
    )
    trainer.make_policy = lambda client: Policy(client)
    return trainer, session, logger


def sequence(marker=1):
    return TokenSequence([marker, marker + 1], [1], [-0.5], [0.5])


def group(version=0, *, task_id="a", degenerate=False):
    return GroupResult(
        task_id,
        version,
        [] if degenerate else [sequence()],
        [1.0, 1.0] if degenerate else [1.0, 0.0],
    )


def trajectory(task, reward, marker):
    return Trajectory(task, [LLMCall([marker], [marker + 1], [-0.25])], reward, 1)


def run(coroutine):
    async def bounded():
        return await asyncio.wait_for(coroutine, timeout=2.0)

    return asyncio.run(bounded())


async def assert_no_new_tasks(baseline):
    # Let completion callbacks run without waiting on a wall-clock interval.
    await asyncio.sleep(0)
    assert not [task for task in asyncio.all_tasks() if task not in baseline and not task.done()]


def install_queue_worker(trainer, closed):
    async def worker(results, task_iter):
        try:
            for task_id in task_iter:
                await results.put(group(trainer.version, task_id=task_id))
        finally:
            closed.set()

    trainer.rollout_worker = worker


def test_generate_group_rejects_unknown_task_before_starting_rollouts(monkeypatch):
    trainer, _, _ = make_trainer()

    async def forbidden(*args, **kwargs):
        raise AssertionError("unknown task must be rejected before collection")

    monkeypatch.setattr(async_module, "collect_group", forbidden)
    with pytest.raises((KeyError, ValueError)):
        run(trainer.generate_group("missing", Policy("captured"), 7))


@pytest.mark.parametrize("rewards", [[1.0, 0.0], [1.0, 1.0]])
def test_generate_group_keeps_supplied_policy_version_and_all_rewards(monkeypatch, rewards):
    async def scenario():
        trainer, _, _ = make_trainer()
        captured = Policy("captured", 7)
        trainer.policy, trainer.version = captured, 7
        agent_cls = object()
        trainer.agent_cls = lambda: agent_cls
        entered, release = asyncio.Event(), asyncio.Event()
        expected_task = trainer.tasks_by_id["a"]

        async def collect(policy, task, env_factory, group_size, *, agent_cls):
            assert policy is captured
            assert task is expected_task
            assert env_factory is trainer.env_factory
            assert group_size == 2
            assert agent_cls is trainer.agent_cls()
            entered.set()
            await release.wait()
            return [trajectory(task, reward, 10 + i * 10) for i, reward in enumerate(rewards)]

        monkeypatch.setattr(async_module, "collect_group", collect)
        pending = asyncio.create_task(trainer.generate_group("a", captured, 7))
        await entered.wait()
        trainer.policy, trainer.version = Policy("newer", 8), 8
        release.set()
        result = await pending

        assert isinstance(result, GroupResult)
        assert result.task_id == "a"
        assert result.policy_version == 7
        assert result.rewards == rewards
        if rewards[0] == rewards[1]:
            assert result.sequences == []
        else:
            assert [item.tokens for item in result.sequences] == [[10, 11], [20, 21]]
            assert [item.advantages for item in result.sequences] == [[0.5], [-0.5]]

    run(scenario())


def test_worker_captures_one_pair_per_group_and_finishes_a_finite_iterator():
    async def scenario():
        trainer, _, _ = make_trainer()
        old, new = Policy("old", 3), Policy("new", 4)
        trainer.policy, trainer.version = old, 3
        calls = []
        results = asyncio.Queue(maxsize=2)

        async def generate(task_id, policy, version):
            calls.append((task_id, policy, version))
            trainer.policy, trainer.version = new, 4
            await asyncio.sleep(0)
            return group(version, task_id=task_id, degenerate=True)

        trainer.generate_group = generate
        await trainer.rollout_worker(results, iter(["a", "b"]))

        assert calls == [("a", old, 3), ("b", new, 4)]
        returned = [results.get_nowait(), results.get_nowait()]
        assert [(item.task_id, item.policy_version) for item in returned] == [("a", 3), ("b", 4)]
        assert all(item.sequences == [] and item.rewards == [1.0, 1.0] for item in returned)

    run(scenario())


def test_worker_waits_for_queue_capacity_before_generating_another_group():
    async def scenario():
        trainer, _, _ = make_trainer()
        trainer.policy = Policy("v0")
        putting = asyncio.Event()

        class ObservableQueue(asyncio.Queue):
            async def put(self, item):
                putting.set()
                await super().put(item)

        results = ObservableQueue(maxsize=1)
        filler = group()
        results.put_nowait(filler)
        calls = []

        async def generate(task_id, policy, version):
            calls.append(task_id)
            return group(version, task_id=task_id)

        trainer.generate_group = generate
        pending = asyncio.create_task(trainer.rollout_worker(results, iter(["a", "b"])))
        await putting.wait()
        assert calls == ["a"], "a full queue must suspend the producer's put"
        assert results.get_nowait() is filler
        assert (await results.get()).task_id == "a"
        assert (await results.get()).task_id == "b"
        await pending
        assert calls == ["a", "b"]

    run(scenario())


def test_worker_propagates_the_original_generation_failure():
    async def scenario():
        trainer, _, _ = make_trainer()
        trainer.policy = Policy("v0")
        failure = TrainingFailure("generation")

        async def generate(*args):
            raise failure

        trainer.generate_group = generate
        with pytest.raises(TrainingFailure) as caught:
            await trainer.rollout_worker(asyncio.Queue(1), iter(["a"]))
        assert caught.value is failure

    run(scenario())


def test_collect_fresh_batch_accepts_boundary_and_degenerate_groups_and_accumulates_drops():
    async def scenario():
        trainer, _, _ = make_trainer(groups_per_batch=2, max_staleness=2)
        trainer.version = 5
        results = asyncio.Queue()
        stale = group(2)
        boundary = group(3, degenerate=True)
        current = group(5)
        for item in (stale, boundary, current):
            results.put_nowait(item)
        stats = {"groups_dropped_stale": 4, "custom": 17}

        batch = await trainer.collect_fresh_batch(results, stats)

        assert len(batch) == 2 and batch[0] is boundary and batch[1] is current
        assert stats["groups_dropped_stale"] == 5
        assert stats["custom"] == 17

    run(scenario())


def test_collect_fresh_batch_checks_version_after_awaiting_each_queue_entry():
    async def scenario():
        trainer, _, _ = make_trainer(max_staleness=1)
        trainer.version = 1
        waiting = asyncio.Event()

        class ObservableQueue(asyncio.Queue):
            async def get(self):
                waiting.set()
                return await super().get()

        results = ObservableQueue()
        stats = {}
        pending = asyncio.create_task(trainer.collect_fresh_batch(results, stats))
        await waiting.wait()
        trainer.version = 4
        results.put_nowait(group(2))  # Fresh at entry, stale when consumed.
        accepted = group(3)
        results.put_nowait(accepted)

        assert (await pending) == [accepted]
        assert stats["groups_dropped_stale"] == 1

    run(scenario())


def test_update_flattens_groups_in_order_and_forwards_configuration_unchanged():
    trainer, session, logger = make_trainer(loss_fn="cross_entropy", lr=3e-5, grad_clip_norm=0.25)
    first, second, third = sequence(10), sequence(20), sequence(30)
    batch = [GroupResult("a", 0, [first, second]), GroupResult("b", 0, []),
             GroupResult("a", 0, [third])]

    metrics = run(trainer.update(batch))

    assert metrics is session.metrics
    assert len(session.updates) == 1
    sequences, loss_fn, lr, clip = session.updates[0]
    assert len(sequences) == 3
    assert all(actual is expected for actual, expected in zip(sequences, [first, second, third]))
    assert (loss_fn, lr, clip) == ("cross_entropy", 3e-5, 0.25)
    assert session.snapshots == [] and logger.records == []


def test_evaluate_now_uses_module_helper_validation_prefix_and_configured_agent(monkeypatch):
    trainer, _, logger = make_trainer(eval_tasks=3, eval_k=5)
    captured = Policy("captured")
    agent_cls = object()
    trainer.agent_cls = lambda: agent_cls
    expected = {"success_rate": 0.5, "pass@5": 1.0}
    calls = []

    async def evaluate(policy, tasks, env_factory, *, k, agent_cls):
        calls.append((policy, tasks, env_factory, k, agent_cls))
        return expected

    monkeypatch.setattr(async_module, "evaluate_policy", evaluate)
    assert run(trainer.evaluate_now(captured)) is expected
    assert len(calls) == 1
    policy, tasks, factory, k, actual_agent = calls[0]
    assert policy is captured and tasks == trainer.val_tasks[:3]
    assert factory is trainer.env_factory and k == 5 and actual_agent is agent_cls
    assert logger.records == []


def test_train_shares_one_cyclic_stream_and_closes_workers_that_return_on_cancellation():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, session, logger = make_trainer(num_workers=2, groups_per_batch=2, num_steps=3)
        seen_ids, queues, closed = [], [], set()

        async def worker(results, task_iter):
            queues.append(results)
            try:
                for task_id in task_iter:
                    seen_ids.append(task_id)
                    await results.put(group(trainer.version, task_id=task_id))
            except asyncio.CancelledError:
                # A worker may finish normally during successful shutdown.
                return
            finally:
                closed.add(asyncio.current_task())

        async def update(batch):
            await asyncio.sleep(0)
            return {}

        trainer.rollout_worker, trainer.update = worker, update
        stats = await trainer.train()

        assert stats == {"train_steps": 3, "groups_consumed": 6, "groups_dropped_stale": 0}
        assert session.snapshots == ["v0", "v1", "v2", "v3"]
        assert len(queues) == 2 and queues[0] is queues[1] and queues[0].maxsize == 1
        assert len(closed) == 2
        assert seen_ids and seen_ids == ["a" if index % 2 == 0 else "b" for index in range(len(seen_ids))]
        assert [step for step, metrics in logger.records if "train/policy_version" in metrics] == [1, 2, 3]
        await assert_no_new_tasks(baseline)

    run(scenario())


def test_train_zero_steps_returns_exact_zero_statistics_without_leaking_tasks():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, session, logger = make_trainer(num_steps=0, eval_every=1)

        async def worker(results, task_iter):
            await asyncio.Event().wait()

        trainer.rollout_worker = worker
        stats = await trainer.train()

        assert stats == {"train_steps": 0, "groups_consumed": 0, "groups_dropped_stale": 0}
        assert session.snapshots == ["v0"]
        assert session.updates == [] and logger.records == []
        await assert_no_new_tasks(baseline)

    run(scenario())


@pytest.mark.parametrize("source", ["worker", "evaluation"])
@pytest.mark.parametrize("phase", ["update", "publish"])
def test_background_failure_interrupts_a_blocked_learner_phase(source, phase):
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, session, _ = make_trainer(eval_every=1 if source == "evaluation" else 0)
        phase_started, phase_closed = asyncio.Event(), asyncio.Event()
        worker_closed = asyncio.Event()
        failure = TrainingFailure(f"{source} during {phase}")
        target = 2 if source == "evaluation" else 1
        updates = 0
        original_snapshot = session.new_sampling_client

        async def blocked_phase():
            phase_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                phase_closed.set()

        async def update(batch):
            nonlocal updates
            updates += 1
            if phase == "update" and updates == target:
                await blocked_phase()
            return {}

        async def snapshot(name):
            if phase == "publish" and name == f"v{target}":
                await blocked_phase()
            return await original_snapshot(name)

        async def worker(results, task_iter):
            try:
                for task_id in task_iter:
                    await results.put(group(trainer.version, task_id=task_id))
                    if source == "worker":
                        await phase_started.wait()
                        raise failure
            finally:
                worker_closed.set()

        async def evaluate(policy):
            await phase_started.wait()
            raise failure

        trainer.update, trainer.rollout_worker = update, worker
        trainer.evaluate_now = evaluate
        session.new_sampling_client = snapshot
        with pytest.raises(TrainingFailure) as caught:
            await trainer.train()

        assert caught.value is failure
        assert phase_started.is_set() and phase_closed.is_set()
        assert worker_closed.is_set()
        await assert_no_new_tasks(baseline)

    run(scenario())


@pytest.mark.parametrize("phase", ["update", "publish"])
def test_learner_failure_cleans_up_workers_and_an_existing_blocked_evaluation(phase):
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, session, _ = make_trainer(eval_every=1)
        eval_started, eval_closed, worker_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        failure = TrainingFailure(f"learner {phase}")
        updates = 0
        original_snapshot = session.new_sampling_client

        async def update(batch):
            nonlocal updates
            updates += 1
            if phase == "update" and updates == 2:
                await eval_started.wait()
                raise failure
            return {}

        async def snapshot(name):
            if phase == "publish" and name == "v2":
                await eval_started.wait()
                raise failure
            return await original_snapshot(name)

        async def evaluate(policy):
            eval_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                eval_closed.set()

        install_queue_worker(trainer, worker_closed)
        trainer.update, trainer.evaluate_now = update, evaluate
        session.new_sampling_client = snapshot
        with pytest.raises(TrainingFailure) as caught:
            await trainer.train()

        assert caught.value is failure
        assert eval_started.is_set() and eval_closed.is_set() and worker_closed.is_set()
        await assert_no_new_tasks(baseline)

    run(scenario())


def test_failure_while_draining_final_evaluations_cancels_the_other_evaluation():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, _, _ = make_trainer(eval_every=1)
        second_started, second_closed, worker_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        failure = TrainingFailure("final evaluation")

        async def evaluate(policy):
            if policy.name == "v1":
                await second_started.wait()
                raise failure
            assert policy.name == "v2"
            second_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                second_closed.set()

        install_queue_worker(trainer, worker_closed)
        trainer.evaluate_now = evaluate
        with pytest.raises(TrainingFailure) as caught:
            await trainer.train()

        assert caught.value is failure
        assert second_started.is_set() and second_closed.is_set() and worker_closed.is_set()
        await assert_no_new_tasks(baseline)

    run(scenario())


def test_external_cancellation_closes_learner_workers_and_pending_evaluations():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, _, _ = make_trainer(eval_every=1)
        eval_started, eval_closed = asyncio.Event(), asyncio.Event()
        learner_started, learner_closed, worker_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        updates = 0

        async def update(batch):
            nonlocal updates
            updates += 1
            if updates == 2:
                await eval_started.wait()
                learner_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    learner_closed.set()
            return {}

        async def evaluate(policy):
            eval_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                eval_closed.set()

        install_queue_worker(trainer, worker_closed)
        trainer.update, trainer.evaluate_now = update, evaluate
        pending = asyncio.create_task(trainer.train())
        await learner_started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        assert learner_closed.is_set() and worker_closed.is_set() and eval_closed.is_set()
        await assert_no_new_tasks(baseline)

    run(scenario())


def test_worker_return_before_training_finishes_fails_instead_of_hanging():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, _, _ = make_trainer()

        async def stopped_worker(results, task_iter):
            return

        trainer.rollout_worker = stopped_worker
        with pytest.raises(RuntimeError):
            await trainer.train()
        await assert_no_new_tasks(baseline)

    run(scenario())


def test_worker_failure_queued_with_last_batch_is_not_hidden_by_successful_shutdown():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, _, _ = make_trainer(num_steps=1)
        failure = TrainingFailure("failed immediately after producing the final batch")

        async def worker(results, task_iter):
            # Queue.put on an empty queue need not yield. It wakes the learner
            # before the worker's completion callbacks are scheduled, so a
            # final no-I/O update may finish before those callbacks execute.
            await results.put(group(0))
            raise failure

        async def update(batch):
            return {}

        trainer.rollout_worker, trainer.update = worker, update
        with pytest.raises(TrainingFailure) as caught:
            await trainer.train()
        assert caught.value is failure
        await assert_no_new_tasks(baseline)

    run(scenario())


def test_resumed_mixed_skip_update_keeps_absolute_eval_cadence_and_captured_version():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        trainer, session, logger = make_trainer(num_steps=2, eval_every=10)
        logger.step_offset = 9
        session.version_offset = 30
        worker_closed, update_finished = asyncio.Event(), asyncio.Event()
        batches = iter([[group(0, degenerate=True)], [group(0)]])
        evaluated = []
        original_update = trainer.update

        async def collect(results, stats):
            # Keep real workers alive while isolating the driver's two steps.
            await asyncio.sleep(0)
            return next(batches)

        async def update(batch):
            metrics = await original_update(batch)
            update_finished.set()
            return metrics

        async def evaluate(policy):
            # The skipped-step evaluation completes after the next update,
            # so its snapshot/version cannot be read from the current state.
            await update_finished.wait()
            evaluated.append(policy)
            return {"sampler": policy.name, "stamp": policy.version}

        install_queue_worker(trainer, worker_closed)
        trainer.collect_fresh_batch, trainer.update, trainer.evaluate_now = collect, update, evaluate
        stats = await trainer.train()

        assert stats == {"train_steps": 2, "groups_consumed": 2, "groups_dropped_stale": 0}
        assert session.snapshots == ["v0", "v1"]
        assert len(session.updates) == 1 and trainer.version == 1
        training = [(step, metrics) for step, metrics in logger.records
                    if "train/policy_version" in metrics]
        assert [(step, metrics["train/policy_version"]) for step, metrics in training] == [(10, 30), (11, 31)]
        assert training[0][1]["train/skipped"] == 1
        assert "train/skipped" not in training[1][1]
        evaluations = [(step, metrics) for step, metrics in logger.records
                       if "eval/policy_version" in metrics]
        assert evaluations == [(10, {"eval/sampler": "v0", "eval/stamp": 30,
                                     "eval/policy_version": 30})]
        assert len(evaluated) == 1 and evaluated[0].name == "v0" and evaluated[0].version == 30
        assert worker_closed.is_set()
        await assert_no_new_tasks(baseline)

    run(scenario())
