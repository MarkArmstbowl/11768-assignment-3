"""Local correctness checks for Part 1, with hand-computed expectations.

These checks use token IDs directly and need no tokenizer or training service.
"""

from copy import deepcopy

import pytest

from scottyrl.advantages import compute_group_advantages
from scottyrl.datagen import (
    extension_suffix,
    stitch_trajectory,
    trajectory_to_training_sequences,
)
from scottyrl.types import LLMCall, Task, Trajectory


def make_trajectory(calls):
    return Trajectory(
        task=Task(goal="test task", id="part1", max_steps=8),
        calls=calls,
        reward=1.0,
        num_steps=len(calls),
    )


def test_reinforce_returns_raw_rewards_as_floats_in_order():
    rewards = (2, 0, -1, 0.5)
    advantages = compute_group_advantages(rewards, method="reinforce")

    assert type(advantages) is list
    assert advantages == [2.0, 0.0, -1.0, 0.5]
    assert all(type(value) is float for value in advantages)
    assert rewards == (2, 0, -1, 0.5)


def test_reinforce_keeps_zero_rewards_and_accepts_empty_input():
    assert compute_group_advantages([0, 0], method="reinforce") == [0.0, 0.0]
    assert compute_group_advantages([], method="reinforce") == []


def test_grpo_centers_rewards_without_standard_deviation_normalization():
    rewards = [1, 0, 0, 0]
    advantages = compute_group_advantages(rewards, method="grpo")

    assert advantages == [0.75, -0.25, -0.25, -0.25]
    assert all(type(value) is float for value in advantages)
    assert rewards == [1, 0, 0, 0]


def test_grpo_retains_a_zero_advantage_inside_a_nondegenerate_group():
    assert compute_group_advantages([0, 1, 2]) == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize("rewards", [[1.0, 1.0], [0.0, 0.0, 0.0]])
def test_grpo_identical_rewards_are_degenerate(rewards):
    assert compute_group_advantages(rewards, method="grpo") is None


def test_grpo_degeneracy_uses_exact_equality():
    # Binary fractions keep the tiny, nonzero expected advantages exact.
    rewards = [1.0, 1.0 + 2.0**-40]
    assert compute_group_advantages(rewards) == [-2.0**-41, 2.0**-41]


@pytest.mark.parametrize("rewards", [[], [1.0]])
def test_grpo_requires_at_least_two_rewards(rewards):
    with pytest.raises(ValueError):
        compute_group_advantages(rewards, method="grpo")


def test_unknown_advantage_method_raises_even_for_empty_input():
    with pytest.raises(ValueError):
        compute_group_advantages([1.0, 0.0], method="ppo")
    with pytest.raises(ValueError):
        compute_group_advantages([], method="unknown")


@pytest.mark.parametrize(
    ("accumulated", "prompt", "expected"),
    [
        ([1, 2, 3], [1, 2, 3, 9, 9], [9, 9]),
        ([1, 2, 3], [1, 2, 3], []),
        ([1, 2, 3], [1, 9, 3, 4], None),
        ([1, 2, 3], [1, 2], None),
        ([], [1, 2], [1, 2]),
        ([], [], []),
    ],
)
def test_extension_suffix_distinguishes_empty_extension_from_break(
    accumulated, prompt, expected
):
    original_accumulated = list(accumulated)
    original_prompt = list(prompt)

    assert extension_suffix(accumulated, prompt) == expected
    assert accumulated == original_accumulated
    assert prompt == original_prompt


def test_three_extending_calls_preserve_next_token_alignment():
    traj = make_trajectory(
        [
            LLMCall([10, 11], [20, 21], [-0.2, -0.3]),
            LLMCall([10, 11, 20, 21, 30], [40, 41], [-0.4, -0.5]),
            LLMCall([10, 11, 20, 21, 30, 40, 41, 50, 51], [60], [-0.6]),
        ]
    )

    sequences = stitch_trajectory(traj)

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.tokens == [10, 11, 20, 21, 30, 40, 41, 50, 51, 60]
    assert sequence.mask == [0, 1, 1, 0, 1, 1, 0, 0, 1]
    assert sequence.logprobs == [0.0, -0.2, -0.3, 0.0, -0.4, -0.5, 0.0, 0.0, -0.6]
    assert sequence.advantages == [0.0] * 9
    assert sequence.num_action_tokens == 5


def test_equal_next_prompt_continues_the_sequence_without_a_context_suffix():
    traj = make_trajectory(
        [
            LLMCall([1, 2], [3], [-0.3]),
            LLMCall([1, 2, 3], [4], [-0.4]),
        ]
    )

    sequences = stitch_trajectory(traj)

    assert len(sequences) == 1
    assert sequences[0].tokens == [1, 2, 3, 4]
    assert sequences[0].mask == [0, 1, 1]
    assert sequences[0].logprobs == [0.0, -0.3, -0.4]
    assert sequences[0].advantages == [0.0, 0.0, 0.0]


def test_rewritten_history_starts_a_new_sequence_then_can_extend_again():
    traj = make_trajectory(
        [
            LLMCall([1, 2], [3, 4], [-0.3, -0.4]),
            # Token 3 remains in rewritten history, but is now prompt context.
            LLMCall([1, 2, 3, 9], [5], [-0.5]),
            LLMCall([1, 2, 3, 9, 5, 6], [7], [-0.7]),
        ]
    )

    sequences = stitch_trajectory(traj)

    assert len(sequences) == 2
    first, second = sequences
    assert first.tokens == [1, 2, 3, 4]
    assert first.mask == [0, 1, 1]
    assert first.logprobs == [0.0, -0.3, -0.4]
    assert first.advantages == [0.0, 0.0, 0.0]
    assert second.tokens == [1, 2, 3, 9, 5, 6, 7]
    assert second.mask == [0, 0, 0, 1, 0, 1]
    assert second.logprobs == [0.0, 0.0, 0.0, -0.5, 0.0, -0.7]
    assert second.advantages == [0.0] * 6


@pytest.mark.parametrize(
    ("advantage", "expected_first", "expected_second"),
    [
        (-0.5, [0.0, -0.5, -0.5], [0.0, 0.0, -0.5]),
        (0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_advantage_is_broadcast_only_to_actions_in_every_sequence(
    advantage, expected_first, expected_second
):
    traj = make_trajectory(
        [
            LLMCall([10, 11], [20, 21], [-0.2, -0.3]),
            LLMCall([10, 11, 30], [40], [-0.4]),
        ]
    )

    sequences = trajectory_to_training_sequences(traj, advantage)

    # A zero advantage changes neither sequence membership nor action masks.
    assert len(sequences) == 2
    assert sequences[0].mask == [0, 1, 1]
    assert sequences[1].mask == [0, 0, 1]
    assert sequences[0].logprobs == [0.0, -0.2, -0.3]
    assert sequences[1].logprobs == [0.0, 0.0, -0.4]
    assert sequences[0].advantages == expected_first
    assert sequences[1].advantages == expected_second


def test_sequences_without_trainable_action_targets_are_filtered():
    assert stitch_trajectory(make_trajectory([])) == []
    assert stitch_trajectory(make_trajectory([LLMCall([1, 2], [], [])])) == []
    assert stitch_trajectory(make_trajectory([LLMCall([], [], [])])) == []
    # The first token has no preceding input token, so it is not a target.
    assert stitch_trajectory(make_trajectory([LLMCall([], [1], [-0.1])])) == []


def test_empty_prompt_trains_only_completion_tokens_after_the_first_token():
    traj = make_trajectory([LLMCall([], [10, 11, 12], [-0.1, -0.2, -0.3])])

    sequences = stitch_trajectory(traj)

    assert len(sequences) == 1
    assert sequences[0].tokens == [10, 11, 12]
    assert sequences[0].mask == [1, 1]
    assert sequences[0].logprobs == [-0.2, -0.3]
    assert sequences[0].advantages == [0.0, 0.0]


def test_empty_completions_keep_context_without_losing_later_actions():
    traj = make_trajectory(
        [
            LLMCall([1, 2], [], []),
            LLMCall([1, 2, 9], [3], [-0.3]),
            LLMCall([1, 2, 9, 3, 4], [], []),
        ]
    )

    sequences = stitch_trajectory(traj)

    assert len(sequences) == 1
    assert sequences[0].tokens == [1, 2, 9, 3, 4]
    assert sequences[0].mask == [0, 0, 1, 0]
    assert sequences[0].logprobs == [0.0, 0.0, -0.3, 0.0]
    assert sequences[0].advantages == [0.0, 0.0, 0.0, 0.0]


def test_stitching_rejects_missing_completion_logprobs():
    traj = make_trajectory([LLMCall([1], [2, 3], [-0.2])])

    with pytest.raises(ValueError, match="same length"):
        stitch_trajectory(traj)


def test_stitching_and_advantage_broadcast_do_not_mutate_the_trajectory():
    traj = make_trajectory(
        [
            LLMCall([1, 2], [3], [-0.3]),
            LLMCall([1, 2, 3, 4], [5], [-0.5]),
        ]
    )
    original = deepcopy(traj)

    stitched = stitch_trajectory(traj)
    training = trajectory_to_training_sequences(traj, advantage=0.75)

    assert traj == original
    assert stitched[0].advantages == [0.0, 0.0, 0.0, 0.0]
    assert training[0].advantages == [0.0, 0.75, 0.0, 0.75]
    # Returned buffers must not alias the lists stored in the episode.
    training[0].tokens[0] = 999
    training[0].logprobs[1] = -999.0
    assert traj == original
