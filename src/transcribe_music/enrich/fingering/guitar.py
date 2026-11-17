"""Biomechanical 6-string Guitar Viterbi fingering + TAB (string/fret) solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import music21

from transcribe_music.enrich.fingering.viterbi import solve_viterbi
from transcribe_music.export.musicxml import clear_technical_articulations, sorted_playable_notes
from transcribe_music.models import INSTRUMENT_SPECS, Instrument

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
            # Allow fingers 1..4 in local hand-position window
            for finger in (1, 2, 3, 4):
                anchor = max(1.0, float(fret - (finger - 1)))
                poses.append(_GuitarPose(notes_pose=((s_num, fret, finger),), anchor_fret=anchor, span=0))
    if not poses:
        # Clamp extreme out-of-range note to nearest string edge gracefully
        clamped = min(max(sounding_midi, _GUITAR_OPEN_MIDI[-1]), _GUITAR_OPEN_MIDI[0] + _MAX_FRET)
        return _single_note_poses(clamped)
    return poses


def _chord_poses(sounding_midis: list[int]) -> list[_GuitarPose]:
    """Generate playable 6-string chord voicings (distinct strings, <= 4 fret span)."""
    # Keep up to 6 unique pitches ordered low -> high
    per_note_opts = [_single_note_candidates(m) for m in sounding_midis]
    if any(not opts for opts in per_note_opts):
        # Fallback: solve each note independently when chord contains extreme pitch
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
            continue  # Two chord notes cannot occupy the same string
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

        # Assign fingers 1..4 by fret offset from min_f (and string order for ties/barres)
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
        # Relaxed fallback if chord is wider than standard 4-fret grip
        used_strings: set[int] = set()
        fallback_list: list[tuple[int, int, int]] = []
        for opts in per_note_opts:
            choice = next(((s, f) for s, f in opts if s not in used_strings), opts[0])
            used_strings.add(choice[0])
            f = choice[1]
            fallback_list.append((choice[0], f, 0 if f == 0 else 1))
        frets = [f for _, f, _ in fallback_list if f > 0]
        return [
            _GuitarPose(
                notes_pose=tuple(fallback_list),
                anchor_fret=float(min(frets)) if frets else 1.0,
                span=(max(frets) - min(frets)) if frets else 0,
            )
        ]

    # Keep top-12 cleanest low-position compact voicings to keep Viterbi O(1) fast
    valid.sort(key=lambda p: (p.span, p.anchor_fret))
    return valid[:12]


def annotate_guitar_fingering(score: music21.stream.Score) -> music21.stream.Score:
    """Assign optimal 6-string guitar TAB (<string>, <fret>) and left-hand <fingering>."""
    if not score.parts:
        return score
    part = score.parts[0]
    transpose = INSTRUMENT_SPECS[Instrument.GUITAR].transpose_semitones  # -12

    elements = sorted_playable_notes(part)
    if not elements:
        return score

    clear_technical_articulations(
        elements,
        music21.articulations.Fingering,
        music21.articulations.StringIndication,
        music21.articulations.FretIndication,
    )

    states_per_step: list[list[_GuitarPose]] = []
    for el in elements:
        if isinstance(el, music21.note.Note):
            sounding = int(el.pitch.midi) + transpose
            states_per_step.append(_single_note_poses(sounding))
        elif isinstance(el, music21.chord.Chord):
            sounding_list = [int(n.pitch.midi) + transpose for n in el.notes]
            states_per_step.append(_chord_poses(sounding_list))

    def _unary(_step: int, pose: _GuitarPose) -> float:
        cost = 0.0
        for s_num, fret, finger in pose.notes_pose:
            # Prefer lower positions (< fret 7) and stronger fingers 1-3 over pinky stretch
            cost += 0.35 * fret + (0.45 if finger == 4 else 0.0)
            # Slight bonus for open strings in first position
            if fret == 0:
                cost -= 0.4
            # Slight middle-string preference for warm classical tone
            if 2 <= s_num <= 5:
                cost -= 0.1
        return cost + 0.5 * pose.span

    def _transition(_step: int, prev: _GuitarPose, cur: _GuitarPose) -> float:
        shift = abs(cur.anchor_fret - prev.anchor_fret)
        # Penalize hand-position jumps and unnecessary string hops
        prev_avg_s = sum(p[0] for p in prev.notes_pose) / len(prev.notes_pose)
        cur_avg_s = sum(p[0] for p in cur.notes_pose) / len(cur.notes_pose)
        string_jump = abs(cur_avg_s - prev_avg_s)

        # Penalize using the exact same fretted finger across wide pitch jumps
        same_finger_penalty = 0.0
        if len(prev.notes_pose) == 1 and len(cur.notes_pose) == 1:
            ps, pf, pfin = prev.notes_pose[0]
            cs, cf, cfin = cur.notes_pose[0]
            if pfin != 0 and pfin == cfin and (ps != cs or pf != cf):
                same_finger_penalty = 0.9

        return 1.6 * shift + 0.25 * string_jump + same_finger_penalty

    best_poses = solve_viterbi(states_per_step, _unary, _transition)

    for el, pose in zip(elements, best_poses, strict=True):
        # MusicXML Note: attach Finger, String, Fret directly.
        # MusicXML Chord: attach Fingerings first, then String+Fret (redistributed 1-to-1 on export).
        for _, _, finger in pose.notes_pose:
            el.articulations.append(music21.articulations.Fingering(finger))
        for s_num, fret, _ in pose.notes_pose:
            el.articulations.append(music21.articulations.StringIndication(s_num))
            el.articulations.append(music21.articulations.FretIndication(fret))

    return score
