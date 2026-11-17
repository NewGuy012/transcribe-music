"""O(N) Viterbi piano fingering solver for right- and left-hand Grand Staff parts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import music21

from transcribe_music.enrich.fingering.viterbi import solve_viterbi
from transcribe_music.export.musicxml import sorted_playable_notes

# White-key x coordinates in white-key units (~2.35 cm); black keys sit at half-steps.
_KEY_X = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
_BLACK_PCS = frozenset({1, 3, 6, 8, 10})
_FINGER_BIAS = {1: 0.0, 2: 0.05, 3: 0.05, 4: 0.2, 5: 0.3}


@dataclass(frozen=True)
class _PianoPose:
    xs: tuple[float, ...]
    is_black: tuple[bool, ...]
    fingers: tuple[int, ...]


def _key_geometry(midi: int) -> tuple[float, bool]:
    octave, pc = divmod(midi, 12)
    return octave * 7.0 + _KEY_X[pc], pc in _BLACK_PCS


def _chord_poses(midis: tuple[int, ...], side: str) -> list[_PianoPose]:
    geom = [_key_geometry(m) for m in midis]
    xs = tuple(g[0] for g in geom)
    is_black = tuple(g[1] for g in geom)
    k = len(midis)
    pool = (1, 2, 3, 4, 5) if side == "right" else (5, 4, 3, 2, 1)

    if k == 1:
        return [_PianoPose(xs=xs, is_black=is_black, fingers=(f,)) for f in pool]
    if k > 5:
        spread = tuple(pool[round(i * 4 / (k - 1))] for i in range(k))
        return [_PianoPose(xs=xs, is_black=is_black, fingers=spread)]

    return [_PianoPose(xs=xs, is_black=is_black, fingers=combo) for combo in combinations(pool, k)]


def _annotate_single_part(part: music21.stream.Part, side: str) -> None:
    elements = sorted_playable_notes(part)
    if not elements:
        return

    for el in elements:
        el.articulations = [
            a for a in el.articulations if not isinstance(a, music21.articulations.Fingering)
        ]

    ordered_indices: list[list[int]] = []
    steps: list[list[_PianoPose]] = []
    for el in elements:
        if isinstance(el, music21.note.Note):
            ordered_indices.append([0])
            steps.append(_chord_poses((int(el.pitch.midi),), side))
        elif isinstance(el, music21.chord.Chord):
            pairs = sorted(enumerate(el.notes), key=lambda p: int(p[1].pitch.midi))
            ordered_indices.append([idx for idx, _ in pairs])
            steps.append(_chord_poses(tuple(int(n.pitch.midi) for _, n in pairs), side))

    sign = 1.0 if side == "right" else -1.0

    def _unary(_t: int, pose: _PianoPose) -> float:
        cost = 0.0
        for f, blk in zip(pose.fingers, pose.is_black, strict=True):
            cost += _FINGER_BIAS[f]
            if blk and f == 1:
                cost += 0.9
            elif blk and f == 5:
                cost += 0.5
        for i in range(len(pose.fingers) - 1):
            dx = pose.xs[i + 1] - pose.xs[i]
            df = abs(pose.fingers[i + 1] - pose.fingers[i])
            max_span = df * (1.8 if 1 in (pose.fingers[i], pose.fingers[i + 1]) else 1.25)
            if dx > max_span:
                cost += 1.5 * (dx - max_span)
            elif dx < df * 0.45:
                cost += 0.6 * (df * 0.45 - dx)
        return cost

    def _transition(_t: int, prev: _PianoPose, cur: _PianoPose) -> float:
        prev_x = sum(prev.xs) / len(prev.xs)
        cur_x = sum(cur.xs) / len(cur.xs)
        prev_f = sum(prev.fingers) / len(prev.fingers)
        cur_f = sum(cur.fingers) / len(cur.fingers)
        dx = cur_x - prev_x
        if dx == 0.0:
            return 0.0 if prev.fingers == cur.fingers else 0.4

        dir_dx = dx * sign
        df = (cur_f - prev_f) * sign
        # Thumb-under or finger-over crossing in scale motion
        if len(prev.fingers) == 1 and len(cur.fingers) == 1:
            pf, cf = prev.fingers[0], cur.fingers[0]
            if cf == 1 and pf in (3, 4) and 0.0 < dir_dx <= 2.5:
                return 0.9
            if pf == 1 and cf in (3, 4) and -2.5 <= dir_dx < 0.0:
                return 0.9

        stretch = abs(dir_dx - df * 1.15)
        if (dir_dx > 0.0 and df <= 0.0) or (dir_dx < 0.0 and df >= 0.0):
            return 1.8 + 0.8 * stretch
        return 0.55 * stretch

    best = solve_viterbi(steps, _unary, _transition)
    for el, orig_order, pose in zip(elements, ordered_indices, best, strict=True):
        if isinstance(el, music21.note.Note):
            el.articulations.append(music21.articulations.Fingering(pose.fingers[0]))
        elif isinstance(el, music21.chord.Chord):
            by_orig = dict(zip(orig_order, pose.fingers, strict=True))
            for idx in range(len(el.notes)):
                el.articulations.append(music21.articulations.Fingering(by_orig[idx]))


def annotate_piano_fingering(score: music21.stream.Score) -> music21.stream.Score:
    """Assign ergonomic 1-5 fingerings for piano RH (part 0) and LH (part 1)."""
    parts = list(score.parts)
    if not parts:
        return score
    _annotate_single_part(parts[0], side="right")
    if len(parts) >= 2:
        _annotate_single_part(parts[1], side="left")
    return score
