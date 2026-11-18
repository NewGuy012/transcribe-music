"""Biomechanical 6-string Guitar Viterbi fingering + TAB (string/fret) solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import music21

from transcribe_music.enrich.fingering._shared import (
    apply_string_articulations,
    pick_unique_string_candidates,
)
from transcribe_music.enrich.fingering.viterbi import solve_viterbi
from transcribe_music.models import INSTRUMENT_SPECS, Instrument
from transcribe_music.score_utils import element_midis, prepare_playable_notes

_GUITAR_OPEN_MIDI = INSTRUMENT_SPECS[Instrument.GUITAR].open_string_midi  # Strings 1..6 (E4..E2)
_MAX_FRET = 19


@dataclass(frozen=True)
class _GuitarPose:
    """Playable single-note or multi-note chord voicing on 6 strings."""

    notes_pose: tuple[tuple[int, int, int], ...]  # ((string_1_to_6, fret_0_to_19, finger_0_to_4), ...)
    anchor_fret: float
    span: int


def _single_note_candidates(sounding_midi: int) -> list[tuple[int, int]]:
    """Return all valid (string 1..6, fret 0..19) placements for one sounding MIDI pitch."""
    out: list[tuple[int, int]] = []
    for s_idx, open_midi in enumerate(_GUITAR_OPEN_MIDI, start=1):
        fret = sounding_midi - open_midi
        if 0 <= fret <= _MAX_FRET:
            out.append((s_idx, fret))
    return out


def _single_note_poses(sounding_midi: int) -> list[_GuitarPose]:
    poses: list[_GuitarPose] = []
    for s_num, fret in _single_note_candidates(sounding_midi):
        if fret == 0:
            poses.append(_GuitarPose(notes_pose=((s_num, 0, 0),), anchor_fret=1.0, span=0))
        else:
            for finger in (1, 2, 3, 4):
                anchor = max(1.0, float(fret - (finger - 1)))
                poses.append(_GuitarPose(notes_pose=((s_num, fret, finger),), anchor_fret=anchor, span=0))
    if not poses:
        clamped = min(max(sounding_midi, _GUITAR_OPEN_MIDI[-1]), _GUITAR_OPEN_MIDI[0] + _MAX_FRET)
        return _single_note_poses(clamped)
    return poses


def _chord_poses(sounding_midis: list[int]) -> list[_GuitarPose]:
    """Generate playable 6-string chord voicings (distinct strings, <= 4 fret span)."""
    per_note_opts = [_single_note_candidates(m) for m in sounding_midis]
    if any(not opts for opts in per_note_opts):
        fallback = [_single_note_poses(m)[0].notes_pose[0] for m in sounding_midis]
        frets = [f for _, f, _ in fallback if f > 0]
        return [
            _GuitarPose(
                notes_pose=tuple(fallback),
                anchor_fret=float(min(frets)) if frets else 1.0,
                span=(max(frets) - min(frets)) if frets else 0,
            )
        ]

    valid: list[_GuitarPose] = []
    for combo in product(*per_note_opts):
        strings = [s for s, _ in combo]
        if len(set(strings)) != len(strings):
            continue
        fretted = [(idx, s, f) for idx, (s, f) in enumerate(combo) if f > 0]
        if not fretted:
            poses = tuple((s, 0, 0) for s, _ in combo)
            valid.append(_GuitarPose(notes_pose=poses, anchor_fret=1.0, span=0))
            continue

        fret_vals = [f for _, _, f in fretted]
        min_f, max_f = min(fret_vals), max(fret_vals)
        span = max_f - min_f
        if span > 4:
            continue

        ordered_fretted = sorted(fretted, key=lambda item: (item[2], -item[1]))
        fingers_by_idx: dict[int, int] = {}
        barre_fret = min_f if sum(1 for f in fret_vals if f == min_f) >= 2 else None
        next_finger = 2 if barre_fret is not None else 1
        for orig_idx, _, f in ordered_fretted:
            if barre_fret is not None and f == barre_fret:
                fingers_by_idx[orig_idx] = 1
            else:
                offset_f = f - min_f + 1
                assigned = max(next_finger, min(4, offset_f))
                fingers_by_idx[orig_idx] = min(4, assigned)
                next_finger = min(4, fingers_by_idx[orig_idx] + 1)

        poses = tuple(
            (s, f, 0 if f == 0 else fingers_by_idx[idx]) for idx, (s, f) in enumerate(combo)
        )
        valid.append(_GuitarPose(notes_pose=poses, anchor_fret=float(min_f), span=span))
        if len(valid) >= 24:
            break

    if not valid:
        picked = pick_unique_string_candidates(per_note_opts, lambda c: c[0])
        fallback_list = [(s, f, 0 if f == 0 else 1) for s, f in picked]
        frets = [f for _, f, _ in fallback_list if f > 0]
        return [
            _GuitarPose(
                notes_pose=tuple(fallback_list),
                anchor_fret=float(min(frets)) if frets else 1.0,
                span=(max(frets) - min(frets)) if frets else 0,
            )
        ]

    valid.sort(key=lambda p: (p.span, p.anchor_fret))
    return valid[:12]


def annotate_guitar_fingering(score: music21.stream.Score) -> music21.stream.Score:
    """Assign optimal 6-string guitar TAB (<string>, <fret>) and left-hand <fingering>."""
    if not score.parts:
        return score
    part = score.parts[0]
    elements = prepare_playable_notes(
        part,
        music21.articulations.Fingering,
        music21.articulations.StringIndication,
        music21.articulations.FretIndication,
    )
    if not elements:
        return score

    first_clef = part.recurse().getElementsByClass(music21.clef.Clef).first()
    is_written_8vb = part.atSoundingPitch is False or isinstance(
        first_clef, music21.clef.Treble8vbClef
    )
    if is_written_8vb:
        transpose = INSTRUMENT_SPECS[Instrument.GUITAR].transpose_semitones
    elif part.atSoundingPitch is True or isinstance(first_clef, music21.clef.BassClef):
        transpose = 0
    else:
        pitches = [m for el in elements for m in element_midis(el)]
        transpose = (
            0
            if pitches and min(pitches) < 52
            else INSTRUMENT_SPECS[Instrument.GUITAR].transpose_semitones
        )

    states_per_step = [
        _single_note_poses(element_midis(el)[0] + transpose)
        if isinstance(el, music21.note.Note)
        else _chord_poses([m + transpose for m in element_midis(el)])
        for el in elements
    ]

    def _unary(_step: int, pose: _GuitarPose) -> float:
        cost = 0.0
        for s_num, fret, finger in pose.notes_pose:
            cost += 0.35 * fret + (0.45 if finger == 4 else 0.0)
            if fret == 0:
                cost -= 0.4
            if 2 <= s_num <= 5:
                cost -= 0.1
        return cost + 0.5 * pose.span

    def _transition(_step: int, prev: _GuitarPose, cur: _GuitarPose) -> float:
        shift = abs(cur.anchor_fret - prev.anchor_fret)
        prev_avg_s = sum(p[0] for p in prev.notes_pose) / len(prev.notes_pose)
        cur_avg_s = sum(p[0] for p in cur.notes_pose) / len(cur.notes_pose)
        string_jump = abs(cur_avg_s - prev_avg_s)

        same_finger_penalty = 0.0
        if len(prev.notes_pose) == 1 and len(cur.notes_pose) == 1:
            ps, pf, pfin = prev.notes_pose[0]
            cs, cf, cfin = cur.notes_pose[0]
            if pfin != 0 and pfin == cfin and (ps != cs or pf != cf):
                same_finger_penalty = 0.9

        return 1.6 * shift + 0.25 * string_jump + same_finger_penalty

    best_poses = solve_viterbi(states_per_step, _unary, _transition)
    apply_string_articulations(elements, best_poses, include_fret=True)
    return score
