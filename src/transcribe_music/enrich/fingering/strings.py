"""Unified biomechanical 4-string bowed Viterbi solver for Violin and Cello."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import music21

from transcribe_music.enrich.fingering.viterbi import solve_viterbi
from transcribe_music.export.musicxml import clear_technical_articulations, sorted_playable_notes
from transcribe_music.models import INSTRUMENT_SPECS, Instrument


@dataclass(frozen=True)
class _StringPose:
    notes_pose: tuple[tuple[int, int, str], ...]  # ((string 1..4, semitone_offset, finger), ...)
    position_base: int
    extended: bool = False
    thumb_mode: bool = False


def _single_candidates(
    midi: int,
    open_strings: tuple[int, ...],
    is_cello: bool,
) -> list[tuple[int, int, str, int, bool, bool]]:
    """Return playable (string 1..4, semitone, finger_label, base_pos, extended, thumb)."""
    max_semi = 19 if is_cello else 17
    out: list[tuple[int, int, str, int, bool, bool]] = []

    for s_idx, open_midi in enumerate(open_strings, start=1):
        semi = midi - open_midi
        if semi == 0:
            open_positions = (1, 2, 4) if is_cello else (1, 3, 5)
            for pos in open_positions:
                out.append((s_idx, 0, "0", pos, False, False))
            continue

        if is_cello:
            if 1 <= semi <= 10:
                for finger_idx in (1, 2, 3, 4):
                    pos = semi - (finger_idx - 1)
                    if 1 <= pos <= 7:
                        out.append((s_idx, semi, str(finger_idx), pos, False, False))
                for finger_idx, offset in ((1, 0), (2, 2), (3, 3), (4, 4)):
                    pos = semi - offset
                    if 1 <= pos <= 6:
                        out.append((s_idx, semi, str(finger_idx), pos, True, False))
            if 7 <= semi <= max_semi:
                for label, offset in (("T", 0), ("1", 2), ("2", 3), ("2", 4), ("3", 5), ("3", 6)):
                    pos = semi - offset
                    if 7 <= pos <= 14:
                        out.append((s_idx, semi, label, pos, False, True))
        elif 1 <= semi <= max_semi:
            for finger_idx, nominal in ((1, 0), (2, 2), (3, 4), (4, 5)):
                for jitter in (0, -1, 1):
                    pos = semi - (nominal + jitter)
                    if 1 <= pos <= 12:
                        out.append((s_idx, semi, str(finger_idx), pos, False, False))

    if not out:
        clamped = min(max(midi, open_strings[-1]), open_strings[0] + max_semi)
        return _single_candidates(clamped, open_strings, is_cello)
    return sorted(set(out))


def _chord_poses(
    midis: list[int],
    open_strings: tuple[int, ...],
    is_cello: bool,
) -> list[_StringPose]:
    if len(midis) == 1:
        return [
            _StringPose(notes_pose=((s, semi, f),), position_base=pos, extended=ext, thumb_mode=th)
            for s, semi, f, pos, ext, th in _single_candidates(midis[0], open_strings, is_cello)
        ]

    per_note = [_single_candidates(m, open_strings, is_cello) for m in midis]
    valid: list[_StringPose] = []
    for combo in product(*per_note):
        if len({c[0] for c in combo}) != len(combo):
            continue
        positions = [c[3] for c in combo]
        if max(positions) - min(positions) > 1:
            continue
        fingers = [c[2] for c in combo if c[2] != "0"]
        if len(set(fingers)) != len(fingers):
            continue
        valid.append(
            _StringPose(
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
    return [_StringPose(notes_pose=tuple(fb), position_base=pos, thumb_mode=is_cello and pos >= 7)]


def annotate_bowed_string_fingering(
    score: music21.stream.Score,
    instrument: Instrument,
) -> music21.stream.Score:
    """Assign ergonomic 4-string fingerings and string numbers for Violin or Cello."""
    if instrument not in (Instrument.VIOLIN, Instrument.CELLO):
        raise ValueError(f"Unsupported bowed string instrument: {instrument}")
    if not score.parts:
        return score

    part = score.parts[0]
    elements = sorted_playable_notes(part)
    if not elements:
        return score

    clear_technical_articulations(
        elements, music21.articulations.Fingering, music21.articulations.StringIndication
    )

    is_cello = instrument == Instrument.CELLO
    open_strings = INSTRUMENT_SPECS[instrument].open_string_midi
    steps = [
        _chord_poses(
            [int(el.pitch.midi)]
            if isinstance(el, music21.note.Note)
            else [int(n.pitch.midi) for n in el.notes],
            open_strings,
            is_cello,
        )
        for el in elements
    ]

    pos_weight = 0.25 if is_cello else 0.28
    shift_base = 1.9 if is_cello else 1.8
    open_high_pen = 1.3 if is_cello else 1.4
    same_finger_pen = 1.1 if is_cello else 1.2

    def _unary(_t: int, pose: _StringPose) -> float:
        cost = pos_weight * (pose.position_base - 1)
        if pose.extended:
            cost += 0.45
        if pose.thumb_mode:
            cost += 0.65
        for s_num, semi, f in pose.notes_pose:
            if f == "0":
                cost += open_high_pen if pose.position_base >= 5 else -0.15
            elif not is_cello:
                if f == "4":
                    cost += 0.35
                cost += 0.2 * abs(semi - (pose.position_base + (int(f) - 1) * 1.75))
            elif f == "4" and semi > 7:
                cost += 0.8
            if s_num > 1 and semi > 12:
                cost += 0.5 if is_cello else 0.4
        return cost

    def _transition(_t: int, prev: _StringPose, cur: _StringPose) -> float:
        shift = abs(cur.position_base - prev.position_base)
        cost = (0.0 if shift == 0 else shift_base + 0.35 * shift) + 0.3 * abs(
            cur.notes_pose[0][0] - prev.notes_pose[0][0]
        )
        if is_cello and cur.thumb_mode != prev.thumb_mode:
            cost += 1.1
        if (
            len(prev.notes_pose) == 1
            and len(cur.notes_pose) == 1
            and prev.notes_pose[0][2] != "0"
            and prev.notes_pose[0][2] == cur.notes_pose[0][2]
            and prev.notes_pose[0][:2] != cur.notes_pose[0][:2]
        ):
            cost += same_finger_pen
        return cost

    for el, pose in zip(elements, solve_viterbi(steps, _unary, _transition), strict=True):
        for _, _, finger_label in pose.notes_pose:
            el.articulations.append(music21.articulations.Fingering(finger_label))
        for s_num, _, _ in pose.notes_pose:
            el.articulations.append(music21.articulations.StringIndication(s_num))

    return score
