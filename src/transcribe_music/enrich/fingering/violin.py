"""Biomechanical 4-string Violin Viterbi position & fingering solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import music21

from transcribe_music.enrich.fingering.viterbi import solve_viterbi
from transcribe_music.models import INSTRUMENT_SPECS, Instrument

_VIOLIN_OPEN = INSTRUMENT_SPECS[Instrument.VIOLIN].open_string_midi  # E5(1), A4(2), D4(3), G3(4)
_MAX_SEMITONES_ON_STRING = 17  # Up to ~7th+ position comfortably
_MAX_POSITION_BASE = 12


@dataclass(frozen=True)
class _ViolinPose:
    notes_pose: tuple[tuple[int, int, int], ...]  # ((string 1..4, semitone_offset, finger 0..4), ...)
    position_base: int  # Semitone offset where finger 1 sits (1=I, 3=II, 5=III, 7=IV, ...)


def _single_candidates(midi: int) -> list[tuple[int, int, int, int]]:
    """Return playable (string 1..4, semitone_above_open, finger 0..4, position_base)."""
    out: list[tuple[int, int, int, int]] = []
    for s_idx, open_midi in enumerate(_VIOLIN_OPEN, start=1):
        semi = midi - open_midi
        if semi == 0:
            # Open string works seamlessly from low positions (I-III)
            for pos in (1, 3, 5):
                out.append((s_idx, 0, 0, pos))
        elif 1 <= semi <= _MAX_SEMITONES_ON_STRING:
            # Diatonic/chromatic hand frame: finger 1 at `pos`, finger 4 reaches up to `pos + 6`
            for finger, nominal_offset in ((1, 0), (2, 2), (3, 4), (4, 5)):
                for jitter in (0, -1, 1):
                    pos = semi - (nominal_offset + jitter)
                    if 1 <= pos <= _MAX_POSITION_BASE:
                        out.append((s_idx, semi, finger, pos))
    if not out:
        clamped = min(max(midi, _VIOLIN_OPEN[-1]), _VIOLIN_OPEN[0] + _MAX_SEMITONES_ON_STRING)
        return _single_candidates(clamped)
    # Deduplicate keeping distinct (string, finger, position) states
    return sorted(set(out))


def _single_poses(midi: int) -> list[_ViolinPose]:
    return [
        _ViolinPose(notes_pose=((s, semi, f),), position_base=pos)
        for s, semi, f, pos in _single_candidates(midi)
    ]


def _chord_poses(midis: list[int]) -> list[_ViolinPose]:
    per_note = [_single_candidates(m) for m in midis]
    valid: list[_ViolinPose] = []
    for combo in product(*per_note):
        strings = [c[0] for c in combo]
        if len(set(strings)) != len(strings):
            continue
        positions = [c[3] for c in combo]
        if max(positions) - min(positions) > 1:
            continue
        fingers = [c[2] for c in combo if c[2] > 0]
        if len(set(fingers)) != len(fingers):
            continue
        pos = round(sum(positions) / len(positions))
        valid.append(
            _ViolinPose(
                notes_pose=tuple((s, semi, f) for s, semi, f, _ in combo),
                position_base=pos,
            )
        )
        if len(valid) >= 16:
            break
    if valid:
        return valid
    # Fallback: distinct strings without position constraint
    used: set[int] = set()
    fb: list[tuple[int, int, int]] = []
    pos = 1
    for opts in per_note:
        pick = next((o for o in opts if o[0] not in used), opts[0])
        used.add(pick[0])
        fb.append((pick[0], pick[1], pick[2]))
        pos = pick[3]
    return [_ViolinPose(notes_pose=tuple(fb), position_base=pos)]


def annotate_violin_fingering(score: music21.stream.Score) -> music21.stream.Score:
    """Assign ergonomic 4-string violin fingerings (0-4) and string numbers (1-4)."""
    if not score.parts:
        return score
    part = score.parts[0]
    elements = sorted(part.recurse().notes, key=lambda el: float(el.getOffsetInHierarchy(part)))
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

    steps: list[list[_ViolinPose]] = []
    for el in elements:
        if isinstance(el, music21.note.Note):
            steps.append(_single_poses(int(el.pitch.midi)))
        elif isinstance(el, music21.chord.Chord):
            steps.append(_chord_poses([int(n.pitch.midi) for n in el.notes]))

    def _unary(_t: int, pose: _ViolinPose) -> float:
        cost = 0.28 * (pose.position_base - 1)
        for s_num, semi, finger in pose.notes_pose:
            # Prefer natural 1st/3rd positions and avoid awkward extensions
            if finger == 4:
                cost += 0.35
            if finger == 0 and pose.position_base >= 5:
                cost += 1.4  # Avoid open-string timbre break in high positions
            elif finger == 0:
                cost -= 0.15
            # Natural frame distance penalty
            if finger > 0:
                expected_semi = pose.position_base + (finger - 1) * 1.75
                cost += 0.2 * abs(semi - expected_semi)
            # Mild preference for avoiding extreme E/G high positions when middle strings fit
            if semi > 12 and s_num > 1:
                cost += 0.4
        return cost

    def _transition(_t: int, prev: _ViolinPose, cur: _ViolinPose) -> float:
        pos_shift = abs(cur.position_base - prev.position_base)
        shift_cost = 0.0 if pos_shift == 0 else (1.8 + 0.35 * pos_shift)

        prev_s = prev.notes_pose[0][0]
        cur_s = cur.notes_pose[0][0]
        string_cross = 0.3 * abs(cur_s - prev_s)

        same_finger_cost = 0.0
        if len(prev.notes_pose) == 1 and len(cur.notes_pose) == 1:
            if (
                prev.notes_pose[0][2] != 0
                and prev.notes_pose[0][2] == cur.notes_pose[0][2]
                and prev.notes_pose[0][:2] != cur.notes_pose[0][:2]
            ):
                same_finger_cost = 1.2

        return shift_cost + string_cross + same_finger_cost

    best = solve_viterbi(steps, _unary, _transition)
    for el, pose in zip(elements, best, strict=True):
        for _, _, finger in pose.notes_pose:
            el.articulations.append(music21.articulations.Fingering(finger))
        for s_num, _, _ in pose.notes_pose:
            el.articulations.append(music21.articulations.StringIndication(s_num))

    return score
