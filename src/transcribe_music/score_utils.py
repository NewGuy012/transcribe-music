"""Shared music21 stream, clef, duration, and playable-note utilities."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from fractions import Fraction

import music21


def is_playable_chord(el: object) -> bool:
    """Return True for performed Chord objects, excluding Harmony ChordSymbols."""
    return isinstance(el, music21.chord.Chord) and not isinstance(
        el, music21.harmony.ChordSymbol
    )


def iter_playable_notes(
    stream: music21.stream.Stream,
) -> Iterator[music21.note.Note | music21.chord.Chord]:
    """Yield performed Notes and Chords (excluding ChordSymbols) from a stream."""
    for el in stream.recurse().notes:
        if isinstance(el, music21.note.Note) or is_playable_chord(el):
            yield el


def element_midis(el: music21.note.Note | music21.chord.Chord) -> list[int]:
    """Return integer MIDI pitches for a Note or playable Chord."""
    if isinstance(el, music21.note.Note):
        return [int(el.pitch.midi)]
    return [int(p.midi) for p in el.pitches]


def prepare_playable_notes(
    part: music21.stream.Stream,
    *articulation_types: type[music21.articulations.Articulation],
) -> list[music21.note.Note | music21.chord.Chord]:
    """Return offset-sorted playable notes after clearing target technical articulations."""
    items = sorted(iter_playable_notes(part), key=lambda el: float(el.getOffsetInHierarchy(part)))
    if articulation_types:
        for el in items:
            el.articulations = [a for a in el.articulations if not isinstance(a, articulation_types)]
    return items


def coerce_to_score(
    parsed: music21.stream.Stream,
    default_title: str | None = None,
    *,
    ensure_part: bool = False,
) -> tuple[music21.stream.Score, list[str]]:
    """Normalize a music21 Stream (Score, Part, Opus, or flat stream) into a Score."""
    warnings: list[str] = []
    if isinstance(parsed, music21.stream.Opus):
        scores = list(parsed.scores)
        if not scores:
            raise ValueError("Opus file contains zero scores.")
        if len(scores) > 1:
            warnings.append(
                f"Input Opus contains {len(scores)} movements/tunes; loaded the first score only."
            )
        score = scores[0]
    elif isinstance(parsed, music21.stream.Score):
        if list(parsed.parts):
            score = parsed
        else:
            loose = [
                el
                for el in list(parsed)
                if not isinstance(el, (music21.metadata.Metadata, music21.layout.LayoutBase))
            ]
            if loose or ensure_part:
                score = music21.stream.Score()
                if parsed.metadata is not None:
                    score.metadata = copy.deepcopy(parsed.metadata)
                part = music21.stream.Part()
                for el in loose:
                    part.append(el)
                score.insert(0, part)
            else:
                score = parsed
    elif isinstance(parsed, music21.stream.Part):
        score = music21.stream.Score()
        score.insert(0, parsed)
    else:
        score = music21.stream.Score()
        part = music21.stream.Part()
        for el in list(parsed):
            if isinstance(el, music21.metadata.Metadata):
                score.metadata = el
            elif not isinstance(el, music21.layout.LayoutBase):
                part.append(el)
        score.insert(0, part)

    if score.metadata is None:
        score.metadata = music21.metadata.Metadata()
    resolved_title = (score.metadata.title or score.metadata.movementName or "").strip()
    if resolved_title and resolved_title != "Music21 Fragment":
        score.metadata.title = resolved_title
    elif default_title:
        score.metadata.title = default_title
    return score, warnings


def make_clef(spec_clef: str, staff_idx: int = 0) -> music21.clef.Clef:
    """Create the initial music21 Clef matching an InstrumentSpec clef_mode."""
    if spec_clef == "grand":
        return music21.clef.TrebleClef() if staff_idx == 0 else music21.clef.BassClef()
    if spec_clef == "treble_8vb":
        return music21.clef.Treble8vbClef()
    if spec_clef == "bass":
        return music21.clef.BassClef()
    return music21.clef.TrebleClef()


def replace_or_insert_initial_clef(
    part: music21.stream.Part,
    new_clef: music21.clef.Clef,
    *,
    force_replace: bool,
) -> None:
    """Insert or replace the opening clef of a Part (in Measure 1 when measured)."""
    measures = list(part.getElementsByClass(music21.stream.Measure))
    target: music21.stream.Stream = measures[0] if measures else part
    existing = list(target.getElementsByClass(music21.clef.Clef))
    if not existing:
        if measures:
            for c in list(part.getElementsByClass(music21.clef.Clef)):
                part.remove(c)
        target.insert(0, new_clef)
        return
    if force_replace:
        for c in existing:
            target.remove(c)
        target.insert(0, new_clef)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def split_tuplet_interval(onset: Fraction, end: Fraction) -> list[tuple[Fraction, Fraction]]:
    """Keep tuplet spans beat-local by splitting non-binary endpoints at integer beats."""
    if onset >= end:
        return []
    if _is_pow2(onset.denominator) and _is_pow2(end.denominator):
        return [(onset, end)]

    segments: list[tuple[Fraction, Fraction]] = []
    cur = onset
    if not _is_pow2(cur.denominator):
        first_beat_end = Fraction(math.floor(cur) + 1, 1)
        if end <= first_beat_end:
            return [(cur, end)]
        segments.append((cur, first_beat_end))
        cur = first_beat_end

    if not _is_pow2(end.denominator):
        last_beat_start = Fraction(math.floor(end), 1)
        if cur < last_beat_start:
            segments.append((cur, last_beat_start))
            cur = last_beat_start
    if cur < end:
        segments.append((cur, end))
    return segments


def fill_stream_rest_gaps(
    st: music21.stream.Stream,
    bar_ql: Fraction,
    notes_only: list[music21.note.Note | music21.chord.Chord] | None = None,
) -> None:
    """Fill uncovered beat intervals in a single-voice Measure/Voice with non-overlapping Rests."""
    if notes_only is None:
        notes_only = sorted(
            [
                el
                for el in list(st)
                if (isinstance(el, music21.note.Note) or is_playable_chord(el))
                and not el.duration.isGrace
            ],
            key=lambda e: (float(e.offset), id(e)),
        )
    if not notes_only:
        st.insert(Fraction(0, 1), music21.note.Rest(quarterLength=bar_ql))
        return

    cursor = Fraction(0, 1)
    for el in notes_only:
        off = Fraction(el.offset).limit_denominator(24)
        end = off + Fraction(el.duration.quarterLength).limit_denominator(24)
        if off > cursor:
            for seg_start, seg_end in split_tuplet_interval(cursor, min(off, bar_ql)):
                st.insert(seg_start, music21.note.Rest(quarterLength=seg_end - seg_start))
        cursor = max(cursor, min(end, bar_ql))
    if cursor < bar_ql:
        for seg_start, seg_end in split_tuplet_interval(cursor, bar_ql):
            st.insert(seg_start, music21.note.Rest(quarterLength=seg_end - seg_start))
