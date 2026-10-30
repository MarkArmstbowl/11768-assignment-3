"""Offline checks for progress-only observation; no HTTP or model requests."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from scottyrl.types import Task, Trajectory


def _load_progress_helper():
    # The project installs scottyrl, not scripts. Pytest's console entry
    # point therefore must not rely on the repo root being on sys.path.
    name = "_scottyrl_evaluation_progress_for_tests"
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_with_progress.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the progress helper: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


progress = _load_progress_helper()


def test_progress_preserves_given_evaluator_results_and_episode_count(monkeypatch, capsys):
    tasks = [Task("goal", "a", 5), Task("goal", "b", 5)]
    calls = []
    per_task = {"a": 0, "b": 0}
    policy = object()
    env_factory = lambda task: task
    agent_cls = lambda policy: object()
    original_evaluator = progress.given_evaluation.evaluate_policy
    observed = {}

    async def fake_episode(agent, task):
        index = per_task[task.id]
        per_task[task.id] += 1
        calls.append((task.id, index))
        await asyncio.sleep(0)
        reward = float(task.id == "a" and index == 0)
        info = {"stop_reason": "simulated failure"} if task.id == "b" and index == 1 else {}
        return Trajectory(task, [], reward, index + 1, info=info)

    async def capture_evaluator(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        result = await original_evaluator(*args, **kwargs)
        observed["result"] = result
        return result

    async def fake_cli_main():
        result = await progress.given_cli.evaluate_policy(
            policy, tasks, env_factory, k=2, concurrency=2, agent_cls=agent_cls
        )
        assert result is observed["result"]
        print(json.dumps(result))

    original_cli_evaluator = progress.given_cli.evaluate_policy
    monkeypatch.setattr(progress.given_evaluation, "run_episode", fake_episode)
    monkeypatch.setattr(progress.given_evaluation, "evaluate_policy", capture_evaluator)
    monkeypatch.setattr(progress.given_cli, "main", fake_cli_main)
    asyncio.run(progress.main())

    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result == {
        "success_rate": 0.25, "pass@2": 0.5, "avg_steps": 1.5,
        "abandoned_episodes": 1, "num_tasks": 2, "episodes_per_task": 2,
    }
    assert sorted(calls) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    assert observed["args"] == (policy, tasks, env_factory)
    assert observed["kwargs"] == {"k": 2, "concurrency": 2, "agent_cls": agent_cls}
    assert "0/4 episodes" in out.err and "4/4 episodes (100.0%)" in out.err
    assert "successes=1 | abandoned=1" in out.err
    assert progress.given_evaluation.run_episode is fake_episode
    assert progress.given_cli.evaluate_policy is original_cli_evaluator


def test_progress_propagates_original_error_without_retry(monkeypatch, capsys):
    failure = RuntimeError("simulated first-turn failure")
    calls = []

    async def fake_episode(agent, task):
        calls.append(task.id)
        raise failure

    async def fake_cli_main():
        await progress.given_cli.evaluate_policy(
            object(), [Task("goal", "a", 5)], lambda task: task,
            k=1, agent_cls=lambda policy: object(),
        )

    original_cli_evaluator = progress.given_cli.evaluate_policy
    monkeypatch.setattr(progress.given_evaluation, "run_episode", fake_episode)
    monkeypatch.setattr(progress.given_cli, "main", fake_cli_main)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(progress.main())
    assert caught.value is failure
    assert calls == ["a"]
    assert progress.given_evaluation.run_episode is fake_episode
    assert progress.given_cli.evaluate_policy is original_cli_evaluator
    out = capsys.readouterr()
    assert "0/1 episodes" in out.err and "1/1 episodes" not in out.err
    assert "interrupted; no final result" in out.err
    assert out.out == ""


def test_progress_preserves_cancellation_and_restores_hooks(monkeypatch, capsys):
    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()

        async def fake_episode(agent, task):
            started.set()
            await never.wait()

        async def fake_cli_main():
            await progress.given_cli.evaluate_policy(
                object(), [Task("goal", "a", 5)], lambda task: task,
                k=1, agent_cls=lambda policy: object(),
            )

        original_cli_evaluator = progress.given_cli.evaluate_policy
        monkeypatch.setattr(progress.given_evaluation, "run_episode", fake_episode)
        monkeypatch.setattr(progress.given_cli, "main", fake_cli_main)
        task = asyncio.create_task(progress.main())
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert progress.given_evaluation.run_episode is fake_episode
        assert progress.given_cli.evaluate_policy is original_cli_evaluator

    asyncio.run(scenario())
    out = capsys.readouterr()
    assert "0/1 episodes" in out.err and "1/1 episodes" not in out.err
    assert "interrupted; no final result" in out.err
