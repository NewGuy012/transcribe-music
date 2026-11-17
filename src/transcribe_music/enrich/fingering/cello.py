"""Biomechanical 4-string Cello Viterbi solver (closed, extended & thumb positions)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import music21

from transcribe_music.enrich.fingering.viterbi import solve_viterbi
from transcribe_music.export.musicxml import sorted_playable_notes
from transcribe_music.models import INSTRUMENT_SPECS, Instrument

_CELLO_OPEN = INSTRUMENT_SPECS[Instrument.CELLO].open_string_midi  # A3(1), D3(2), G2(3), C2(4)
_MAX_SEMITONES_ON_STRING = 19  # High A3+19 = E5 comfortably


@dataclass(frozen=True)
class _CelloPose:
    notes_pose: tuple[tuple[int, int, str], ...]  # ((string 1..4, semitone, '0'..'4'|'T'), ...)
    position_base: int
    extended: bool
    thumb_mode: bool


def _single_candidates(midi: int) -> list[tuple[int, int, str, int, bool, bool]]:
    """Return playable cello (string, semitone, finger_label, base_pos, extended, thumb)."""
    out: list[tuple[int, int, str, int, bool, bool]] = []
    for s_idx, open_midi in enumerate(_CELLO_OPEN, start=1):
        semi = midi - open_midi
        if semi == 0:
            for pos in (1, 2, 4):
                out.append((s_idx, 0, "0", pos, False, False))
        elif 1 <= semi <= 10:
            # Closed neck hand frame: 1=base, 2=base+1, 3=base+2, 4=base+3
            for finger_idx in (1, 2, 3, 4):
                pos = semi - (finger_idx - 1)
                if 1 <= pos <= 7:
                    out.append((s_idx, semi, str(finger_idx), pos, False, False))
            # Extended neck hand frame: 1=base, 2=base+2, 3=base+3, 4=base+4
            for finger_idx, offset in ((1, 0), (2, 2), (3, 3), (4, 4)):
                pos = semi - offset
                if 1 <= pos <= 6:
                    out.append((s_idx, semi, str(finger_idx), pos, True, False))
        if 7 <= semi <= _MAX_SEMITONES_ON_STRING:
            # Thumb position above 4th position: T=base, 1=base+2, 2=base+4, 3=base+5
            for finger_label, offset in (("T", 0), ("1", 2), ("2", 3), ("2", 4), ("3", 5), ("3", 6)):
                pos = semi - offset
                if 7 <= pos <= 14:
                    out.append((s_idx, semi, finger_label, pos, False, True))

    if not out:
        clamped = min(max(midi, _CELLO_OPEN[-1]), _CELLO_OPEN[0] + _MAX_SEMITONES_ON_STRING)
        return _single_candidates(clamped)
    return sorted(set(out))


def _single_poses(midi: int) -> list[_CelloPose]:
    return [
        _CelloPose(
            notes_pose=((s, semi, f),),
            position_base=pos,
            extended=ext,
            thumb_mode=thumb,
        )
        for s, semi, f, pos, ext, thumb in _single_candidates(midi)
    ]


def _chord_poses(midis: list[int]) -> list[_CelloPose]:
    per_note = [_single_candidates(m) for m in midis]
    valid: list[_CelloPose] = []
    for combo in product(*per_note):
        strings = [c[0] for c in combo]
        if len(set(strings)) != len(strings):
            continue
        positions = [c[3] for c in combo]
        if max(positions) - min(positions) > 1:
            continue
        fingers = [c[2] for c in combo if c[2] != "0"]
        if len(set(fingers)) != len(fingers):
            continue
        valid.append(
            _CelloPose(
                notes_pose=tuple((s, semi, f) for s, semi, f, _, _, _ in combo),
                position_base=round(sum(positions) / len(positions)),
                extended=any(c[4] for c in combo),
                thumb_mode=any(c[5] for c in combo),
            )
        )
        if len(valid) >= 16:
            break
    if valid:
        return valid

    used: set[int] = set()
    fb: list[tuple[int, int, str]] = []
    pos = 1
    for opts in per_note:
        pick = next((o for o in opts if o[0] not in used), opts[0])
        used.add(pick[0])
        fb.append((pick[0], pick[1], pick[2]))
        pos = pick[3]
    return [_CelloPose(notes_pose=tuple(fb), position_base=pos, extended=False, thumb_mode=pos >= 7)]


def annotate_cello_fingering(score: music21.stream.Score) -> music21.stream.Score:
    """Assign ergonomic 4-string cello fingerings (0-4 and T) and string numbers (1-4)."""
    if not score.parts:
        return score
    part = score.parts[0]
    elements = sorted_playable_notes(part)
    if not elements:
        return score

    for el in elements:
        el.articulations = [
            a
            for a in el.articulations
            if not isinstance(
                a, (music21.articulations.Fingering, music21.articulations.StringIndication)
            )
        ]

    steps: list[list[_CelloPose]] = []
    for el in elements:
        if isinstance(el, music21.note.Note):
            steps.append(_single_poses(int(el.pitch.midi)))
        elif isinstance(el, music21.chord.Chord):
            steps.append(_chord_poses([int(n.pitch.midi) for n in el.notes]))

    def _unary(_t: int, pose: _CelloPose) -> float:
        cost = 0.25 * (pose.position_base - 1)
        if pose.extended:
            cost += 0.45  # Slight preference for relaxed closed frame unless shift avoided
        if pose.thumb_mode:
            cost += 0.65
        for s_num, semi, f in pose.notes_pose:
            if f == "0" and pose.position_base >= 5:
                cost += 1.3
            elif f == "0":
                cost -= 0.15
            if f == "4" and semi > 7:
                cost += 0.8  # Cello pinky rarely used in thumb position
            if s_num > 1 and semi > 12:
                cost += 0.5
        return cost

    def _transition(_t: int, prev: _CelloPose, cur: _CelloPose) -> float:
        shift = abs(cur.position_base - prev.position_base)
        shift_cost = 0.0 if shift == 0 else (1.9 + 0.35 * shift)
        thumb_toggle = 1.1 if cur.thumb_mode != prev.thumb_mode else 0.0
        string_cross = 0.3 * abs(cur.notes_pose[0][0] - prev.notes_pose[0][0])
        same_finger_cost = 0.0
        if (
            len(prev.notes_pose) == 1
            and len(cur.notes_pose) == 1
            and prev.notes_pose[0][2] != "0"
            and prev.notes_pose[0][2] == cur.notes_pose[0][2]
            and prev.notes_pose[0][:2] != cur.notes_pose[0][:2]
        ):
            same_finger_cost = 1.1
        return shift_cost + thumb_toggle + string_cross + same_finger_cost

    best = solve_viterbi(steps, _unary, _transition)
    for el, pose in zip(elements, best, strict=True):
        for _, _, finger_label in pose.notes_pose:
            el.articulations.append(music21.articulations.Fingering(finger_label))
        for s_num, _, _ in pose.notes_pose:
            el.articulations.append(music21.articulations.StringIndication(s_num))

    return score
