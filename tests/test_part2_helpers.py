"""Offline correctness checks for rollout collection and batch assembly."""

import asyncio
from copy import deepcopy

import pytest

from scottyrl.train import common
from scottyrl.types import LLMCall, Task, Trajectory


class DummyAgent:
    def __init__(self, policy):
        self.policy = policy
        self.events = []


class DummyEnv:
    def __init__(self, task):
        self.task = task
        self.events = []


def make_trajectory(reward, marker, calls=None):
    if calls is None:
        calls = [LLMCall([marker], [marker + 1], [-0.25])]
    return Trajectory(
        task=Task(goal="test task", id=f"task-{marker}", max_steps=8),
        calls=calls,
        reward=reward,
        num_steps=len(calls),
    )


def assert_stats(stats, **expected):
    # Other optional diagnostic fields are allowed by the helper's contract.
    for name, value in expected.items():
        assert stats[name] == value, name


def test_collect_group_gives_each_rollout_fresh_agent_and_environment(monkeypatch):
    task = Task(goal="shared task", id="shared", max_steps=1)
    policy = object()
    agents = []
    environments = []
    returned_trajectories = []

    async def fake_episode(agent, env):
        assert agent.events == []
        assert env.events == []
        agent.events.append("used")
        env.events.append("used")
        agents.append(agent)
        environments.append(env)
        trajectory = make_trajectory(0.0, len(agents))
        returned_trajectories.append(trajectory)
        return trajectory

    monkeypatch.setattr(common, "run_episode", fake_episode)

    trajectories = asyncio.run(
        common.collect_group(policy, task, DummyEnv, 3, agent_cls=DummyAgent)
    )

    assert len(agents) == len(environments) == len(trajectories) == 3
    assert len({id(agent) for agent in agents}) == 3
    assert len({id(env) for env in environments}) == 3
    assert all(agent.policy is policy for agent in agents)
    assert all(env.task is task for env in environments)
    assert all(
        actual is expected
        for actual, expected in zip(trajectories, returned_trajectories)
    )


def test_collect_group_rollouts_overlap_instead_of_running_sequentially(monkeypatch):
    async def run():
        all_started = asyncio.Event()
        started = 0

        async def fake_episode(agent, env):
            nonlocal started
            started += 1
            if started == 3:
                all_started.set()
            # A sequential collector cannot start the other two episodes.
            await all_started.wait()
            return make_trajectory(0.0, started)

        monkeypatch.setattr(common, "run_episode", fake_episode)
        trajectories = await asyncio.wait_for(
            common.collect_group(
                object(),
                Task(goal="g", id="concurrent", max_steps=1),
                DummyEnv,
                3,
                agent_cls=DummyAgent,
            ),
            timeout=2.0,
        )
        assert started == 3
        assert len(trajectories) == 3

    asyncio.run(run())


def test_collect_group_preserves_rollout_order_when_completions_arrive_reversed(
    monkeypatch,
):
    async def run():
        all_started = asyncio.Event()
        gates = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
        gates[2].set()
        environments = []
        completion_order = []
        outputs = [make_trajectory(0.0, 10), make_trajectory(1.0, 20), make_trajectory(0.0, 30)]
        started = 0

        def env_factory(task):
            env = DummyEnv(task)
            env.rollout_index = len(environments)
            environments.append(env)
            return env

        async def fake_episode(agent, env):
            nonlocal started
            index = env.rollout_index
            started += 1
            if started == 3:
                all_started.set()
            await all_started.wait()
            await gates[index].wait()
            completion_order.append(index)
            if index:
                gates[index - 1].set()
            return outputs[index]

        monkeypatch.setattr(common, "run_episode", fake_episode)
        trajectories = await asyncio.wait_for(
            common.collect_group(
                object(),
                Task(goal="g", id="ordered", max_steps=1),
                env_factory,
                3,
                agent_cls=DummyAgent,
            ),
            timeout=2.0,
        )

        assert completion_order == [2, 1, 0]
        assert len(trajectories) == 3
        assert trajectories[0] is outputs[0]
        assert trajectories[1] is outputs[1]
        assert trajectories[2] is outputs[2]

    asyncio.run(run())


def test_collect_group_propagates_the_original_rollout_error(monkeypatch):
    class RolloutFailure(Exception):
        pass

    failure = RolloutFailure("episode failed")

    async def failing_episode(agent, env):
        raise failure

    monkeypatch.setattr(common, "run_episode", failing_episode)

    with pytest.raises(RolloutFailure) as raised:
        asyncio.run(
            common.collect_group(
                object(),
                Task(goal="g", id="failure", max_steps=1),
                DummyEnv,
                1,
                agent_cls=DummyAgent,
            )
        )
    assert raised.value is failure


def test_collect_group_failure_cancels_and_awaits_sibling_episodes(monkeypatch):
    async def scenario():
        failure = RuntimeError("fatal rollout")
        siblings_started = asyncio.Event()
        blocked = asyncio.Event()
        episodes = []
        cleaned = set()

        async def fake_episode(agent, env):
            index = len(episodes)
            episodes.append(asyncio.current_task())
            if index == 0:
                await siblings_started.wait()
                raise failure
            try:
                if len(episodes) == 3:
                    siblings_started.set()
                await blocked.wait()
            finally:
                # Cancellation alone is insufficient: the collector must
                # await asynchronous cleanup before it propagates the error.
                await asyncio.sleep(0)
                cleaned.add(index)

        monkeypatch.setattr(common, "run_episode", fake_episode)
        try:
            with pytest.raises(RuntimeError) as caught:
                await asyncio.wait_for(
                    common.collect_group(
                        object(), Task("goal", "failed-group", 1), DummyEnv, 3,
                        agent_cls=DummyAgent,
                    ),
                    timeout=2.0,
                )
            assert caught.value is failure
            assert cleaned == {1, 2}
            assert len(episodes) == 3 and all(task.done() for task in episodes)
        finally:
            for task in episodes:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*episodes, return_exceptions=True)

    asyncio.run(scenario())


def test_collect_group_outer_cancellation_awaits_all_episode_cleanup(monkeypatch):
    async def scenario():
        all_started = asyncio.Event()
        blocked = asyncio.Event()
        episodes = []
        cleaned = set()

        async def fake_episode(agent, env):
            index = len(episodes)
            episodes.append(asyncio.current_task())
            try:
                if len(episodes) == 3:
                    all_started.set()
                await blocked.wait()
            finally:
                await asyncio.sleep(0)
                cleaned.add(index)

        monkeypatch.setattr(common, "run_episode", fake_episode)
        collection = asyncio.create_task(
            common.collect_group(
                object(), Task("goal", "cancelled-group", 1), DummyEnv, 3,
                agent_cls=DummyAgent,
            )
        )
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2.0)
            collection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await collection
            assert cleaned == {0, 1, 2}
            assert all(task.done() for task in episodes)
        finally:
            collection.cancel()
            await asyncio.gather(collection, return_exceptions=True)

    asyncio.run(scenario())


def test_async_trainer_fatal_episode_leaves_no_nested_rollouts(monkeypatch):
    from types import SimpleNamespace

    from scottyrl.config import ModelConfig, TrainConfig
    from scottyrl.train.async_grpo import AsyncGRPOTrainer

    async def scenario():
        failure = RuntimeError("worker's nested episode failed")
        siblings_started = asyncio.Event()
        blocked = asyncio.Event()
        episodes = []
        cleaned = set()

        async def fake_episode(agent, env):
            index = len(episodes)
            episodes.append(asyncio.current_task())
            if index == 0:
                await siblings_started.wait()
                raise failure
            try:
                if len(episodes) == 3:
                    siblings_started.set()
                await blocked.wait()
            finally:
                await asyncio.sleep(0)
                cleaned.add(index)

        async def sampling_client(name):
            return name

        session = SimpleNamespace(new_sampling_client=sampling_client)
        trainer = AsyncGRPOTrainer(
            session, DummyEnv, [Task("goal", "a", 1)], [], ModelConfig(),
            TrainConfig(
                group_size=3, groups_per_batch=1, num_workers=1,
                queue_capacity=1, num_steps=1, eval_every=0,
            ),
            SimpleNamespace(step_offset=0, log=lambda *args: None),
        )
        trainer.make_policy = lambda client: SimpleNamespace(name=client)
        trainer.agent_cls = lambda: DummyAgent
        monkeypatch.setattr(common, "run_episode", fake_episode)
        try:
            with pytest.raises(RuntimeError) as caught:
                await asyncio.wait_for(trainer.train(), timeout=2.0)
            assert caught.value is failure
            assert cleaned == {1, 2}
            assert len(episodes) == 3 and all(task.done() for task in episodes)
        finally:
            for task in episodes:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*episodes, return_exceptions=True)

    asyncio.run(scenario())


def test_groups_to_batch_empty_input_has_zero_counts_and_reward():
    sequences, stats = common.groups_to_batch([])

    assert sequences == []
    assert_stats(
        stats,
        num_groups=0,
        num_degenerate_groups=0,
        frac_degenerate_groups=0.0,
        mean_reward=0.0,
        num_sequences=0,
        num_action_tokens=0,
    )


def test_groups_to_batch_skips_degenerate_groups_but_counts_all_episode_rewards():
    groups = [
        [make_trajectory(1.0, 10), make_trajectory(0.0, 20)],
        [make_trajectory(1.0, 30), make_trajectory(1.0, 40), make_trajectory(1.0, 50)],
        [make_trajectory(0.0, 60), make_trajectory(0.0, 70)],
    ]

    sequences, stats = common.groups_to_batch(groups)

    assert [sequence.tokens for sequence in sequences] == [[10, 11], [20, 21]]
    assert [sequence.advantages for sequence in sequences] == [[0.5], [-0.5]]
    assert_stats(
        stats,
        num_groups=3,
        num_degenerate_groups=2,
        frac_degenerate_groups=pytest.approx(2 / 3),
        # Four successes in seven episodes, rather than averaging group means.
        mean_reward=pytest.approx(4 / 7),
        num_sequences=2,
        num_action_tokens=2,
    )


def test_groups_to_batch_all_degenerate_groups_produce_no_sequences():
    groups = [
        [make_trajectory(1.0, 10), make_trajectory(1.0, 20)],
        [make_trajectory(0.0, 30), make_trajectory(0.0, 40)],
    ]

    sequences, stats = common.groups_to_batch(groups)

    assert sequences == []
    assert_stats(
        stats,
        num_groups=2,
        num_degenerate_groups=2,
        frac_degenerate_groups=1.0,
        mean_reward=0.5,
        num_sequences=0,
        num_action_tokens=0,
    )


def test_groups_to_batch_keeps_zero_advantages_in_a_nondegenerate_grpo_group():
    groups = [[make_trajectory(0.0, 10), make_trajectory(1.0, 20), make_trajectory(2.0, 30)]]

    sequences, stats = common.groups_to_batch(groups, method="grpo")

    assert [sequence.tokens for sequence in sequences] == [[10, 11], [20, 21], [30, 31]]
    assert [sequence.mask for sequence in sequences] == [[1], [1], [1]]
    assert [sequence.advantages for sequence in sequences] == [[-1.0], [0.0], [1.0]]
    assert_stats(
        stats,
        num_groups=1,
        num_degenerate_groups=0,
        frac_degenerate_groups=0.0,
        mean_reward=1.0,
        num_sequences=3,
        num_action_tokens=3,
    )


def test_groups_to_batch_reinforce_keeps_failed_singleton_rollouts():
    groups = [[make_trajectory(0.0, 10)], [make_trajectory(1.0, 20)]]

    sequences, stats = common.groups_to_batch(groups, method="reinforce")

    assert [sequence.tokens for sequence in sequences] == [[10, 11], [20, 21]]
    assert [sequence.advantages for sequence in sequences] == [[0.0], [1.0]]
    assert_stats(
        stats,
        num_groups=2,
        num_degenerate_groups=0,
        frac_degenerate_groups=0.0,
        mean_reward=0.5,
        num_sequences=2,
        num_action_tokens=2,
    )


def test_groups_to_batch_flattens_in_group_trajectory_and_sequence_order():
    first = make_trajectory(
        1.0,
        10,
        calls=[
            LLMCall([10], [11, 12], [-0.1, -0.2]),
            LLMCall([10, 99], [13], [-0.3]),
        ],
    )
    groups = [
        [first, make_trajectory(0.0, 20)],
        [make_trajectory(0.0, 30), make_trajectory(1.0, 40)],
    ]
    original = deepcopy(groups)

    sequences, stats = common.groups_to_batch(groups)

    assert [sequence.tokens for sequence in sequences] == [
        [10, 11, 12],
        [10, 99, 13],
        [20, 21],
        [30, 31],
        [40, 41],
    ]
    assert [sequence.advantages for sequence in sequences] == [
        [0.5, 0.5],
        [0.0, 0.5],
        [-0.5],
        [-0.5],
        [0.5],
    ]
    assert_stats(
        stats,
        num_groups=2,
        num_degenerate_groups=0,
        frac_degenerate_groups=0.0,
        mean_reward=0.5,
        num_sequences=5,
        num_action_tokens=6,
    )
    assert groups == original


def test_groups_to_batch_does_not_confuse_missing_actions_with_degenerate_rewards():
    groups = [[make_trajectory(1.0, 10, calls=[]), make_trajectory(0.0, 20)]]

    sequences, stats = common.groups_to_batch(groups)

    assert [sequence.tokens for sequence in sequences] == [[20, 21]]
    assert sequences[0].advantages == [-0.5]
    assert_stats(
        stats,
        num_groups=1,
        num_degenerate_groups=0,
        frac_degenerate_groups=0.0,
        mean_reward=0.5,
        num_sequences=1,
        num_action_tokens=1,
    )
