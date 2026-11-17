"""Shared fast Viterbi shortest-path dynamic programming solver."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def solve_viterbi[S](
    states_per_step: Sequence[Sequence[S]],
    unary_cost: Callable[[int, S], float],
    transition_cost: Callable[[int, S, S], float],
) -> list[S]:
    """Find minimum-cost state sequence across T steps in O(sum(S_t * S_{t-1}))."""
    n_steps = len(states_per_step)
    if n_steps == 0:
        return []
    for step_idx, candidates in enumerate(states_per_step):
        if not candidates:
            raise ValueError(f"Empty state candidates at step {step_idx}")

    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []

    first_states = states_per_step[0]
    dp0 = np.array([unary_cost(0, s) for s in first_states], dtype=np.float64)
    dp.append(dp0)
    back.append(np.zeros(len(first_states), dtype=np.int32))

    for t in range(1, n_steps):
        prev_states = states_per_step[t - 1]
        cur_states = states_per_step[t]
        prev_dp = dp[t - 1]
        cur_dp = np.empty(len(cur_states), dtype=np.float64)
        cur_back = np.empty(len(cur_states), dtype=np.int32)

        for j, cur_s in enumerate(cur_states):
            u = unary_cost(t, cur_s)
            best_cost = float("inf")
            best_i = 0
            for i, prev_s in enumerate(prev_states):
                c = float(prev_dp[i]) + transition_cost(t, prev_s, cur_s) + u
                if c < best_cost:
                    best_cost = c
                    best_i = i
            cur_dp[j] = best_cost
            cur_back[j] = best_i

        dp.append(cur_dp)
        back.append(cur_back)

    chosen_idx = [0] * n_steps
    chosen_idx[-1] = int(np.argmin(dp[-1]))
    for t in range(n_steps - 1, 0, -1):
        chosen_idx[t - 1] = int(back[t][chosen_idx[t]])

    return [states_per_step[t][chosen_idx[t]] for t in range(n_steps)]
