"""Shared Viterbi dynamic programming solver and string articulation helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import music21


class _HasNotesPose(Protocol):
    notes_pose: tuple[tuple[int, int, int | str], ...]


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

    prev_dp = [unary_cost(0, s) for s in states_per_step[0]]
    back: list[list[int]] = [[0] * len(states_per_step[0])]

    for t in range(1, n_steps):
        prev_states = states_per_step[t - 1]
        cur_states = states_per_step[t]
        cur_dp: list[float] = []
        cur_back: list[int] = []

        for cur_s in cur_states:
            u = unary_cost(t, cur_s)
            best_cost = float("inf")
            best_i = 0
            for i, prev_s in enumerate(prev_states):
                c = prev_dp[i] + transition_cost(t, prev_s, cur_s) + u
                if c < best_cost:
                    best_cost = c
                    best_i = i
            cur_dp.append(best_cost)
            cur_back.append(best_i)

        prev_dp = cur_dp
        back.append(cur_back)

    chosen_idx = [0] * n_steps
    chosen_idx[-1] = min(range(len(prev_dp)), key=prev_dp.__getitem__)
    for t in range(n_steps - 1, 0, -1):
        chosen_idx[t - 1] = back[t][chosen_idx[t]]

    return [states_per_step[t][chosen_idx[t]] for t in range(n_steps)]


def pick_unique_string_candidates[T](
    per_note_opts: Sequence[Sequence[T]],
    string_of: Callable[[T], int],
) -> list[T]:
    """Greedily pick one candidate per chord note preferring distinct strings."""
    used: set[int] = set()
    picked: list[T] = []
    for opts in per_note_opts:
        choice = next((o for o in opts if string_of(o) not in used), opts[0])
        used.add(string_of(choice))
        picked.append(choice)
    return picked


def apply_string_articulations(
    elements: Sequence[music21.note.Note | music21.chord.Chord],
    poses: Sequence[_HasNotesPose],
    *,
    include_fret: bool = False,
) -> None:
    """Attach Fingering, StringIndication, and optional FretIndication tags to elements."""
    for el, pose in zip(elements, poses, strict=True):
        # Fingerings first, then String/Fret so MusicXML chord export pairs them 1-to-1
        for _, _, finger in pose.notes_pose:
            el.articulations.append(music21.articulations.Fingering(finger))
        for s_num, fret, _ in pose.notes_pose:
            el.articulations.append(music21.articulations.StringIndication(s_num))
            if include_fret:
                el.articulations.append(music21.articulations.FretIndication(fret))
