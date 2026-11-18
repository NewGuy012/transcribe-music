"""Shared helpers for fretted and bowed string fingering solvers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import music21


class _HasNotesPose(Protocol):
    notes_pose: tuple[tuple[int, int, int | str], ...]


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
