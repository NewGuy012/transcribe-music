"""Guaranteed-baseline MusicXML exporter, score normalizer, and XML validator."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import music21

from transcribe_music.models import INSTRUMENT_SPECS, Instrument


@dataclass(frozen=True)
class ExportMetrics:
    """Structural counts extracted from a validated MusicXML score."""

    measure_count: int
    note_count: int
    chord_count: int
    fingering_count: int


def is_playable_chord(el: object) -> bool:
    """Return True for performed Chord objects, excluding Harmony ChordSymbols."""
    return isinstance(el, music21.chord.Chord) and not isinstance(
        el, music21.harmony.ChordSymbol
    )


def sorted_playable_notes(
    part: music21.stream.Stream,
) -> list[music21.note.Note | music21.chord.Chord]:
    """Return all performed Notes and Chords (excluding ChordSymbol) ordered by part offset."""
    items = [
        el
        for el in part.recurse().notes
        if isinstance(el, music21.note.Note) or is_playable_chord(el)
    ]
    return sorted(items, key=lambda el: float(el.getOffsetInHierarchy(part)))


def clear_technical_articulations(
    elements: list[music21.note.Note | music21.chord.Chord],
    *articulation_types: type[music21.articulations.Articulation],
) -> None:
    """Remove any existing technical articulations of the given types before re-annotating."""
    for el in elements:
        el.articulations = [a for a in el.articulations if not isinstance(a, articulation_types)]


def _make_music21_instrument(instrument: Instrument) -> music21.instrument.Instrument:
    if instrument == Instrument.PIANO:
        return music21.instrument.Piano()
    if instrument == Instrument.GUITAR:
        return music21.instrument.AcousticGuitar()
    if instrument == Instrument.VIOLIN:
        return music21.instrument.Violin()
    return music21.instrument.Violoncello()


def _make_clef(spec_clef: str, staff_idx: int = 0) -> music21.clef.Clef:
    if spec_clef == "grand":
        return music21.clef.TrebleClef() if staff_idx == 0 else music21.clef.BassClef()
    if spec_clef == "treble_8vb":
        return music21.clef.Treble8vbClef()
    if spec_clef == "bass":
        return music21.clef.BassClef()
    return music21.clef.TrebleClef()


_UNEXPORTABLE_DURATION_TYPES = frozenset({"inexpressible", "zero", "2048th", "duplex-maxima"})


def _is_unexportable_element(el: object) -> bool:
    """Return True if el is a non-grace Note/Rest/Chord with an unexportable MusicXML duration."""
    return (
        isinstance(el, (music21.note.Note, music21.note.Rest, music21.chord.Chord))
        and not isinstance(el, music21.harmony.ChordSymbol)
        and not el.duration.isGrace
        and el.duration.type in _UNEXPORTABLE_DURATION_TYPES
    )


def _is_playable_or_rest(el: object) -> bool:
    return isinstance(el, (music21.note.Note, music21.note.Rest)) or is_playable_chord(el)


def _snap_clean_quarter_length(
    value: float,
    *,
    allow_zero: bool = False,
    min_ql: Fraction = Fraction(1, 8),
) -> Fraction:
    """Snap an arbitrary float duration/offset to the closest standard binary/triplet Fraction."""
    if value <= 0.0:
        return Fraction(0, 1) if allow_zero else min_ql
    candidates: list[tuple[float, int, Fraction]] = []
    for div in (4, 3, 8, 6):
        num = round(value * div)
        if num == 0 and not allow_zero:
            continue
        candidates.append((abs(value - num / div), div, Fraction(num, div)))
    if not candidates:
        return Fraction(0, 1) if allow_zero else min_ql
    candidates.sort(key=lambda item: (round(item[0], 6), item[1]))
    snapped = candidates[0][2]
    if not allow_zero and snapped <= 0:
        return min_ql
    return snapped


def _exact_fraction_or_none(value: float | Fraction) -> Fraction | None:
    if isinstance(value, Fraction):
        return value
    return Fraction(float(value)).limit_denominator(10000)


def _repair_stream_elements(st: music21.stream.Stream, bar_ql: Fraction | None) -> None:
    """Repair unexportable durations and float-rounded offsets/gaps in a Measure, Voice, or Part."""
    items = [el for el in list(st) if _is_playable_or_rest(el) and not el.duration.isGrace]
    repaired_ids: set[int] = set()

    for el in items:
        ql_float = float(el.duration.quarterLength)
        if isinstance(el, music21.note.Rest) and ql_float < 1 / 32:
            st.remove(el)
            continue

        off_frac = _exact_fraction_or_none(el.offset)
        if off_frac is not None and off_frac.denominator > 24:
            new_off = _snap_clean_quarter_length(float(el.offset), allow_zero=True)
            st.setElementOffset(el, new_off)
            repaired_ids.add(id(el))

        ql_frac = _exact_fraction_or_none(el.duration.quarterLength)
        needs_dur_fix = _is_unexportable_element(el) or (
            ql_frac is not None and ql_frac.denominator > 24
        )
        if needs_dur_fix:
            new_ql = _snap_clean_quarter_length(ql_float, allow_zero=False)
            if bar_ql is not None:
                cur_off = Fraction(el.offset).limit_denominator(24)
                if cur_off < bar_ql:
                    new_ql = min(new_ql, bar_ql - cur_off)
                else:
                    new_ql = min(new_ql, bar_ql)
                if new_ql <= 0:
                    new_ql = Fraction(1, 8)
            el.duration = music21.duration.Duration(new_ql)
            repaired_ids.add(id(el))

    if bar_ql is None or not repaired_ids:
        return

    # Rebuild rests cleanly in the repaired measure/voice so gaps never carry rounding artifacts
    for r in list(st.getElementsByClass(music21.note.Rest)):
        if not r.duration.isGrace:
            st.remove(r)

    notes_only = sorted(
        [
            el
            for el in st
            if _is_playable_or_rest(el)
            and not isinstance(el, music21.note.Rest)
            and not el.duration.isGrace
        ],
        key=lambda e: (float(e.offset), id(e)),
    )
    for idx, n in enumerate(notes_only):
        if id(n) not in repaired_ids:
            continue
        n_off = Fraction(n.offset).limit_denominator(24)
        if n_off >= bar_ql:
            continue
        next_off = (
            Fraction(notes_only[idx + 1].offset).limit_denominator(24)
            if idx + 1 < len(notes_only)
            else bar_ql
        )
        if next_off > n_off:
            max_dur = min(bar_ql, next_off) - n_off
            cur_ql = Fraction(n.duration.quarterLength).limit_denominator(24)
            if cur_ql > max_dur > 0:
                n.duration = music21.duration.Duration(max_dur)

    # Fill gaps with beat-split rests when non-binary offsets exist
    cursor = Fraction(0, 1)
    for n in notes_only:
        n_off = Fraction(n.offset).limit_denominator(24)
        n_end = n_off + Fraction(n.duration.quarterLength).limit_denominator(24)
        if n_off > cursor:
            _insert_split_rests(st, cursor, min(n_off, bar_ql))
        cursor = max(cursor, min(n_end, bar_ql))
    if cursor < bar_ql:
        _insert_split_rests(st, cursor, bar_ql)


def _insert_split_rests(st: music21.stream.Stream, start: Fraction, end: Fraction) -> None:
    for seg_start, seg_end in split_tuplet_interval(start, end):
        st.insert(seg_start, music21.note.Rest(quarterLength=seg_end - seg_start))


def split_tuplet_interval(onset: Fraction, end: Fraction) -> list[tuple[Fraction, Fraction]]:
    """Keep tuplet spans beat-local by splitting non-binary endpoints at integer beat boundaries."""
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


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _repair_unexportable_durations(part: music21.stream.Part) -> None:
    """Repair any unexportable Note/Chord/Rest durations so MusicXML export never fails."""
    if not any(_is_unexportable_element(el) for el in part.recurse().notesAndRests):
        return

    measures = list(part.getElementsByClass(music21.stream.Measure))
    if not measures:
        _repair_stream_elements(part, bar_ql=None)
        return

    for m in measures:
        if not any(_is_unexportable_element(el) for el in m.recurse().notesAndRests):
            continue
        bar_ql = (
            Fraction(m.barDuration.quarterLength).limit_denominator(24)
            if m.barDuration is not None
            else Fraction(4, 1)
        )
        voices = list(m.getElementsByClass(music21.stream.Voice))
        for st in voices or [m]:
            _repair_stream_elements(st, bar_ql=bar_ql)


def normalize_score_structure(
    score: music21.stream.Score,
    instrument: Instrument,
    title: str | None = None,
) -> music21.stream.Score:
    """Ensure parts have valid instrument metadata, clefs, time signatures, and complete measures."""
    spec = INSTRUMENT_SPECS[instrument]

    if score.metadata is None:
        score.metadata = music21.metadata.Metadata()
    if title:
        score.metadata.title = title

    parts = list(score.parts)
    if not parts:
        part = music21.stream.Part()
        for el in list(score):
            if not isinstance(el, (music21.metadata.Metadata, music21.layout.LayoutBase)):
                part.append(el)
        score = music21.stream.Score(id=f"{instrument.value}_score")
        if title:
            score.metadata = music21.metadata.Metadata(title=title)
        score.insert(0, part)
        parts = [part]

    for idx, part in enumerate(parts):
        for existing_inst in list(part.recurse().getElementsByClass(music21.instrument.Instrument)):
            part.remove(existing_inst, recurse=True)
        m21_inst = _make_music21_instrument(instrument)
        if spec.transpose_semitones:
            m21_inst.transposition = music21.interval.Interval(spec.transpose_semitones)
        part.insert(0, m21_inst)

        measures = list(part.getElementsByClass(music21.stream.Measure))
        if not measures:
            if not part.getElementsByClass(music21.meter.TimeSignature):
                part.insert(0, music21.meter.TimeSignature("4/4"))
            _repair_unexportable_durations(part)
            part.makeMeasures(inPlace=True)
            measures = list(part.getElementsByClass(music21.stream.Measure))

        if measures:
            first_m = measures[0]
            if not first_m.getElementsByClass(music21.clef.Clef):
                first_m.insert(0, _make_clef(spec.clef_mode, idx))
            if not first_m.getElementsByClass(music21.meter.TimeSignature):
                ts = part.recurse().getElementsByClass(music21.meter.TimeSignature).first()
                first_m.insert(0, ts if ts is not None else music21.meter.TimeSignature("4/4"))
            if not first_m.getElementsByClass(music21.key.KeySignature):
                ks = part.recurse().getElementsByClass(music21.key.KeySignature).first()
                if ks is not None:
                    first_m.insert(0, ks)

        _repair_unexportable_durations(part)
        part.makeNotation(inPlace=True)
        # Final guard in case makeNotation introduced any unexportable duration/rest
        _repair_unexportable_durations(part)

    if instrument == Instrument.PIANO and len(parts) == 2:
        has_brace = any(
            getattr(sg, "symbol", None) == "brace"
            for sg in score.getElementsByClass(music21.layout.StaffGroup)
        )
        if not has_brace:
            score.insert(0, music21.layout.StaffGroup(parts, name="Piano", symbol="brace"))

    return score


def _strip_non_note_articulations(score: music21.stream.Score) -> None:
    """Fallback cleanup if an exotic articulation/technical tag prevents XML export."""
    for n in score.recurse().notes:
        n.articulations = []
        for expr in list(n.expressions):
            if not isinstance(expr, music21.expressions.TextExpression):
                n.expressions.remove(expr)
    for h in list(score.recurse().getElementsByClass(music21.harmony.ChordSymbol)):
        if h.activeSite is not None:
            h.activeSite.remove(h)


def _redistribute_chord_technical_tags(xml_path: Path) -> None:
    """Fix music21 Chord serialization by pairing <string>/<fret> 1-to-1 across chord notes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = False

    for measure in root.findall(".//measure"):
        notes = measure.findall("note")
        i = 0
        while i < len(notes):
            group = [notes[i]]
            j = i + 1
            while j < len(notes) and notes[j].find("chord") is not None:
                group.append(notes[j])
                j += 1

            if len(group) > 1:
                head_tech = group[0].find("notations/technical")
                if head_tech is not None:
                    strings = head_tech.findall("string")
                    frets = head_tech.findall("fret")
                    if len(strings) == len(group) or len(frets) == len(group):
                        # Remove pooled string/fret tags from head note, keep only index 0
                        for idx in range(1, len(group)):
                            target_notations = group[idx].find("notations")
                            if target_notations is None:
                                target_notations = ET.SubElement(group[idx], "notations")
                            target_tech = target_notations.find("technical")
                            if target_tech is None:
                                target_tech = ET.SubElement(target_notations, "technical")
                            if idx < len(strings):
                                head_tech.remove(strings[idx])
                                target_tech.append(strings[idx])
                                changed = True
                            if idx < len(frets):
                                head_tech.remove(frets[idx])
                                target_tech.append(frets[idx])
                                changed = True
            i = j

    if changed:
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _extract_metrics_and_validate_xml(xml_path: Path) -> ExportMetrics:
    """Validate serialized MusicXML structure and extract structural counts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag not in ("score-partwise", "score-timewise"):
        raise ValueError(f"Invalid MusicXML root element: {root.tag}")

    xml_measures = root.findall(".//measure")
    if not xml_measures:
        raise ValueError("Exported MusicXML contains zero <measure> elements")

    xml_notes = root.findall(".//note")
    for note_el in xml_notes:
        if note_el.find("grace") is None and note_el.find("duration") is None:
            raise ValueError("Exported MusicXML contains a <note> missing <duration>")

    parts = root.findall("part")
    measure_count = len(parts[0].findall("measure")) if parts else len(xml_measures)
    note_count = sum(1 for n in xml_notes if n.find("rest") is None)
    return ExportMetrics(
        measure_count=measure_count,
        note_count=note_count,
        chord_count=len(root.findall(".//harmony")),
        fingering_count=len(root.findall(".//technical/fingering")),
    )


def export_musicxml(
    score: music21.stream.Score,
    output_path: Path,
    instrument: Instrument,
    title: str | None = None,
) -> tuple[ExportMetrics, list[str]]:
    """Normalize, serialize, and validate a MusicXML score with graceful fallback."""
    warnings: list[str] = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_score_structure(score, instrument=instrument, title=title)

    try:
        normalized.write("musicxml", fp=str(output_path))
    except Exception as exc:  # noqa: BLE001 — guarantee valid baseline MusicXML output
        warnings.append(f"Enrichment tag fallback triggered during MusicXML write: {exc}")
        _strip_non_note_articulations(normalized)
        normalized.write("musicxml", fp=str(output_path))

    _redistribute_chord_technical_tags(output_path)
    metrics = _extract_metrics_and_validate_xml(output_path)
    return metrics, warnings
