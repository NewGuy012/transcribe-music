"""In-memory piano fingering optimizer using pianoplayer.hand.Hand + chord fallback."""

from __future__ import annotations

import music21
from pianoplayer.hand import Hand
from pianoplayer.models import INote, keypos_midi

from transcribe_music.export.musicxml import sorted_playable_notes


def _clear_fingerings(part: music21.stream.Part) -> None:
    for el in sorted_playable_notes(part):
        el.articulations = [
            a for a in el.articulations if not isinstance(a, music21.articulations.Fingering)
        ]


def _step_fallback_finger(prev_midi: int | None, cur_midi: int, prev_finger: int, side: str) -> int:
    if prev_midi is None:
        return 3
    diff = cur_midi - prev_midi
    if diff == 0:
        return prev_finger
    direction = 1 if side == "right" else -1
    step = diff * direction
    candidate = prev_finger + (1 if step > 0 else -1)
    if candidate > 5:
        return 1
    if candidate < 1:
        return 5
    return candidate


def _annotate_single_part(part: music21.stream.Part, side: str) -> None:
    _clear_fingerings(part)

    # Flatten notes/chords in chronological order; keep mapping back to music21 objects
    elements = sorted_playable_notes(part)
    if not elements:
        return

    inotes: list[INote] = []
    target_slots: list[tuple[music21.note.NotRest, int]] = []
    time_cursor = 0.0
    note_id = 0

    for el in elements:
        onset_beat = float(el.getOffsetInHierarchy(part))
        dur_s = max(0.08, float(el.duration.quarterLength) * 0.5)
        if isinstance(el, music21.note.Note):
            an = INote(
                name=el.pitch.name,
                isChord=False,
                isBlack=bool(el.pitch.alter != 0),
                pitch=int(el.pitch.midi),
                octave=int(el.pitch.octave or 4),
                x=float(keypos_midi(type("_P", (), {"pitch": int(el.pitch.midi)})())),
                time=max(time_cursor, onset_beat * 0.5),
                duration=dur_s,
                noteID=note_id,
            )
            inotes.append(an)
            target_slots.append((el, 0))
            note_id += 1
            time_cursor = an.time + 0.01
        elif isinstance(el, music21.chord.Chord):
            # Order chord notes ascending for RH, descending for LH ergonomic arpeggiation
            ordered = sorted(
                enumerate(el.notes),
                key=lambda pair: pair[1].pitch.midi,
                reverse=(side == "left"),
            )
            for stagger_idx, (orig_idx, n) in enumerate(ordered):
                an = INote(
                    name=n.pitch.name,
                    isChord=True,
                    isBlack=bool(n.pitch.alter != 0),
                    pitch=int(n.pitch.midi),
                    octave=int(n.pitch.octave or 4),
                    x=float(keypos_midi(type("_P", (), {"pitch": int(n.pitch.midi)})())),
                    time=max(time_cursor, onset_beat * 0.5 + stagger_idx * 0.04),
                    duration=dur_s,
                    noteID=note_id,
                    chordID=id(el),
                )
                inotes.append(an)
                target_slots.append((el, orig_idx))
                note_id += 1
            time_cursor = onset_beat * 0.5 + len(ordered) * 0.04

    if not inotes:
        return

    try:
        hand = Hand(inotes, side=side, size="M")
        hand.verbose = False
        hand.autodepth = True
        hand.generate(nmeasures=10000)
    except Exception as exc:  # noqa: BLE001 — fallback to monotonic finger assignment
        import logging

        logging.getLogger(__name__).debug("pianoplayer fallback: %s", exc)

    # Map results back onto music21 Notes & Chords (ensuring valid 1..5 on every slot)
    chord_fingers: dict[int, dict[int, int]] = {}
    chord_refs: dict[int, music21.chord.Chord] = {}
    prev_midi: int | None = None
    prev_finger = 3

    for an, (el, orig_idx) in zip(inotes, target_slots, strict=True):
        raw_f = int(an.fingering) if isinstance(an.fingering, (int, float)) else 0
        finger = raw_f if 1 <= raw_f <= 5 else _step_fallback_finger(prev_midi, int(an.pitch), prev_finger, side)
        prev_midi = int(an.pitch)
        prev_finger = finger

        if isinstance(el, music21.note.Note):
            el.articulations.append(music21.articulations.Fingering(finger))
        elif isinstance(el, music21.chord.Chord):
            cid = id(el)
            chord_refs[cid] = el
            chord_fingers.setdefault(cid, {})[orig_idx] = finger

    for cid, el in chord_refs.items():
        slot_map = chord_fingers[cid]
        # Attach in el.notes order so music21 distributes 1-to-1 across chord <note>s
        used: set[int] = set()
        for idx in range(len(el.notes)):
            f = slot_map.get(idx, 1)
            # Avoid identical finger on two simultaneous chord tones when 1..5 available
            if f in used and len(used) < 5:
                for alt in (1, 2, 3, 4, 5):
                    if alt not in used:
                        f = alt
                        break
            used.add(f)
            el.articulations.append(music21.articulations.Fingering(f))


def annotate_piano_fingering(score: music21.stream.Score) -> music21.stream.Score:
    """Assign ergonomic 1-5 fingerings for piano RH (part 0) and LH (part 1)."""
    parts = list(score.parts)
    if not parts:
        return score

    _annotate_single_part(parts[0], side="right")
    if len(parts) >= 2:
        _annotate_single_part(parts[1], side="left")
    return score
