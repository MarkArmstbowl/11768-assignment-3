"""Run the given checkpoint evaluator with local episode-progress messages.

Use the same arguments as scripts/evaluate_checkpoint.py. The given entry
point still loads the config, selects tasks and runs/scores every episode.
This wrapper only observes returned trajectories, writes progress to stderr,
and leaves the final result JSON unchanged. It makes no additional model
requests and adds no retries. Use it for a NEW evaluation process only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import time

from scottyrl import evaluate as given_evaluation


def _load_given_cli():
    """Load the supplied file without requiring an installed scripts package."""
    name = "_scottyrl_given_checkpoint_cli"
    path = Path(__file__).resolve().with_name("evaluate_checkpoint.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the supplied evaluator: {path}")
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


given_cli = _load_given_cli()


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class _Progress:
    total: int
    success_threshold: float
    completed: int = 0
    active: int = 0
    successes: int = 0
    abandoned: int = 0
    started: float = field(default_factory=time.monotonic)
    last_finished: float | None = None

    def emit(self, *, status: str = "running") -> None:
        now = time.monotonic()
        percent = 100 * self.completed / self.total if self.total else 0.0
        last = (
            _duration(now - self.last_finished)
            if self.last_finished is not None
            else "none yet"
        )
        # Observability must not turn an otherwise valid evaluation into a
        # failure if its progress-output stream has been closed.
        try:
            print(
                f"[progress] {self.completed}/{self.total} episodes ({percent:.1f}%)"
                f" | active={self.active} | successes={self.successes}"
                f" | abandoned={self.abandoned} | elapsed={_duration(now - self.started)}"
                f" | since_last_finish={last} | {status}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    def record(self, trajectory) -> None:
        self.completed += 1
        self.successes += int(trajectory.reward >= self.success_threshold)
        self.abandoned += int("stop_reason" in trajectory.info)
        self.last_finished = time.monotonic()
        if self.completed == 1 or self.completed % 10 == 0 or self.completed == self.total:
            self.emit(status="complete" if self.completed == self.total else "running")


async def evaluate_with_progress(policy, tasks, env_factory, **kwargs) -> dict:
    """Observe the original evaluator without replacing its scheduling/scoring."""
    progress = _Progress(
        total=len(tasks) * kwargs.get("k", 1),
        success_threshold=kwargs.get("success_threshold", 1.0),
    )
    original_episode = given_evaluation.run_episode

    async def observed_episode(*args, **episode_kwargs):
        progress.active += 1
        try:
            trajectory = await original_episode(*args, **episode_kwargs)
        finally:
            progress.active -= 1
        # Exceptions/cancellation propagate unchanged and do not reach this
        # line. Returned abandoned episodes DO count as completed failures.
        progress.record(trajectory)
        return trajectory

    loop = asyncio.get_running_loop()
    timer: asyncio.TimerHandle | None = None

    def tick() -> None:
        nonlocal timer
        progress.emit()
        timer = loop.call_later(30.0, tick)

    progress.emit()
    given_evaluation.run_episode = observed_episode
    timer = loop.call_later(30.0, tick)
    try:
        return await given_evaluation.evaluate_policy(policy, tasks, env_factory, **kwargs)
    except BaseException:
        progress.emit(status="interrupted; no final result")
        raise
    finally:
        timer.cancel()
        given_evaluation.run_episode = original_episode


async def main() -> None:
    original_evaluator = given_cli.evaluate_policy
    given_cli.evaluate_policy = evaluate_with_progress
    try:
        await given_cli.main()
    finally:
        given_cli.evaluate_policy = original_evaluator


if __name__ == "__main__":
    asyncio.run(main())
