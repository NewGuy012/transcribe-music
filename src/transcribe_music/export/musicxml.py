"""Guaranteed-baseline MusicXML exporter, score normalizer, and XML validator."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
