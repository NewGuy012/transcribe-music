"""Guaranteed-baseline MusicXML exporter, score normalizer, and XML validator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import music21

from transcribe_music.models import INSTRUMENT_SPECS, Instrument
from transcribe_music.storage.catalog import ChordRow, MeasureRow, NoteRow


@dataclass(frozen=True)
class ExportMetrics:
    """Extracted structural counts and DuckDB rows from an exported MusicXML score."""

    measure_count: int
    note_count: int
    chord_count: int
    fingering_count: int
    measures: list[MeasureRow]
    notes: list[NoteRow]
    chords: list[ChordRow]


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
        # Replace or insert clean instrument metadata
        for existing_inst in list(part.recurse().getElementsByClass(music21.instrument.Instrument)):
            part.remove(existing_inst, recurse=True)
        m21_inst = _make_music21_instrument(instrument)
        if spec.transpose_semitones:
            m21_inst.transposition = music21.interval.Interval(spec.transpose_semitones)
        part.insert(0, m21_inst)

        # Ensure measures exist and are properly formed
        measures = list(part.getElementsByClass(music21.stream.Measure))
        if not measures:
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

        # Fill gaps with valid rests and ensure ties/beams are notation-ready
        part.makeNotation(inPlace=True)

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


def _extract_metrics_and_validate_xml(
    score: music21.stream.Score, xml_path: Path
) -> ExportMetrics:
    """Validate serialized MusicXML structure and extract DuckDB rows."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag not in ("score-partwise", "score-timewise"):
        raise ValueError(f"Invalid MusicXML root element: {root.tag}")

    xml_measures = root.findall(".//measure")
    if not xml_measures:
        raise ValueError("Exported MusicXML contains zero <measure> elements")

    xml_notes = root.findall(".//note")
    xml_harmonies = root.findall(".//harmony")
    xml_fingerings = root.findall(".//technical/fingering")

    # Ensure every <note> element has valid <duration>
    for note_el in xml_notes:
        if note_el.find("grace") is None and note_el.find("duration") is None:
            raise ValueError("Exported MusicXML contains a <note> missing <duration>")

    measures: list[MeasureRow] = []
    notes: list[NoteRow] = []
    chords: list[ChordRow] = []

    seen_measures: set[int] = set()
    for part_idx, part in enumerate(score.parts):
        cur_ts: str | None = None
        cur_ks: str | None = None
        for m in part.getElementsByClass(music21.stream.Measure):
            m_num = int(m.number) if m.number is not None else 0
            seen_measures.add(m_num)
            ts_el = m.getElementsByClass(music21.meter.TimeSignature).first()
            if ts_el is not None:
                cur_ts = ts_el.ratioString
            ks_el = m.getElementsByClass(music21.key.KeySignature).first()
            if ks_el is not None:
                cur_ks = str(ks_el)
            measures.append(
                MeasureRow(
                    part_idx=part_idx,
                    measure_number=m_num,
                    time_signature=cur_ts,
                    key_signature=cur_ks,
                    duration_ql=float(m.duration.quarterLength),
                )
            )

            for el in m.notesAndRests:
                if isinstance(el, music21.note.Note):
                    finger, string_num, fret_num = _extract_technical_from_note(el)
                    notes.append(
                        NoteRow(
                            part_idx=part_idx,
                            measure_number=m_num,
                            offset_ql=float(el.offset),
                            duration_ql=float(el.duration.quarterLength),
                            midi=int(el.pitch.midi),
                            pitch_name=el.pitch.nameWithOctave,
                            is_chord=False,
                            finger=finger,
                            string_num=string_num,
                            fret_num=fret_num,
                        )
                    )
                elif isinstance(el, music21.chord.Chord):
                    # Check note-level articulations on chord or sub-notes
                    for sub_idx, n in enumerate(el.notes):
                        finger, string_num, fret_num = _extract_technical_from_note(n)
                        if finger is None and sub_idx == 0:
                            finger, string_num, fret_num = _extract_technical_from_note(el)
                        notes.append(
                            NoteRow(
                                part_idx=part_idx,
                                measure_number=m_num,
                                offset_ql=float(el.offset),
                                duration_ql=float(el.duration.quarterLength),
                                midi=int(n.pitch.midi),
                                pitch_name=n.pitch.nameWithOctave,
                                is_chord=True,
                                finger=finger,
                                string_num=string_num,
                                fret_num=fret_num,
                            )
                        )

            if part_idx == 0:
                for h in m.getElementsByClass(music21.harmony.ChordSymbol):
                    root_name = h.root().name if h.root() is not None else ""
                    chords.append(
                        ChordRow(
                            measure_number=m_num,
                            offset_ql=float(h.offset),
                            symbol=h.figure,
                            root=root_name,
                            kind=str(h.chordKind or ""),
                        )
                    )

    return ExportMetrics(
        measure_count=len(seen_measures) if seen_measures else len(xml_measures),
        note_count=len(notes),
        chord_count=len(xml_harmonies),
        fingering_count=len(xml_fingerings),
        measures=measures,
        notes=notes,
        chords=chords,
    )


def _extract_technical_from_note(
    el: music21.note.NotRest,
) -> tuple[str | None, int | None, int | None]:
    finger: str | None = None
    string_num: int | None = None
    fret_num: int | None = None
    for art in getattr(el, "articulations", ()):
        if isinstance(art, music21.articulations.Fingering):
            finger = str(art.fingerNumber)
        elif isinstance(art, music21.articulations.StringIndication):
            try:
                string_num = int(art.number)
            except (TypeError, ValueError):
                pass
        elif isinstance(art, music21.articulations.FretIndication):
            try:
                fret_num = int(art.number)
            except (TypeError, ValueError):
                pass
    return finger, string_num, fret_num


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
    except Exception as exc:
        warnings.append(f"Enrichment tag fallback triggered during MusicXML write: {exc}")
        _strip_non_note_articulations(normalized)
        normalized.write("musicxml", fp=str(output_path))

    metrics = _extract_metrics_and_validate_xml(normalized, output_path)
    return metrics, warnings
