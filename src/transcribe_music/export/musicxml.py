"""Guaranteed-baseline MusicXML exporter, score normalizer, and XML validator."""

from __future__ import annotations

import copy
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import music21

from transcribe_music.models import INSTRUMENT_SPECS, GuitarStaffMode, Instrument
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    coerce_to_score,
    fill_stream_rest_gaps,
    get_guitar_track_meta,
    is_playable_chord,
    make_clef,
    replace_or_insert_initial_clef,
    try_int,
)


@dataclass(frozen=True)
class ExportMetrics:
    """Structural counts extracted from a validated MusicXML score."""

    measure_count: int
    note_count: int
    chord_count: int
    fingering_count: int


def _make_music21_instrument(instrument: Instrument) -> music21.instrument.Instrument:
    if instrument == Instrument.PIANO:
        return music21.instrument.Piano()
    if instrument == Instrument.GUITAR:
        return music21.instrument.AcousticGuitar()
    if instrument == Instrument.VIOLIN:
        return music21.instrument.Violin()
    return music21.instrument.Violoncello()


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


def _to_fraction(value: float | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(float(value)).limit_denominator(10000)


def _repair_stream_elements(st: music21.stream.Stream, bar_ql: Fraction | None) -> None:
    """Repair unexportable durations and float-rounded offsets/gaps in a Measure, Voice, or Part."""
    items = [el for el in list(st) if _is_playable_or_rest(el) and not el.duration.isGrace]
    repaired_ids: set[int] = set()

    for el in items:
        ql_float = float(el.duration.quarterLength)
        if isinstance(el, music21.note.Rest) and ql_float < 1 / 32:
            st.remove(el)
            continue

        off_frac = _to_fraction(el.offset)
        if off_frac.denominator > 24:
            new_off = _snap_clean_quarter_length(float(el.offset), allow_zero=True)
            st.setElementOffset(el, new_off)
            repaired_ids.add(id(el))

        ql_frac = _to_fraction(el.duration.quarterLength)
        needs_dur_fix = _is_unexportable_element(el) or ql_frac.denominator > 24
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

    fill_stream_rest_gaps(st, bar_ql, notes_only=notes_only)


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

    if not list(score.parts):
        score, _ = coerce_to_score(score, default_title=title, ensure_part=True)
    parts = list(score.parts)

    for idx, part in enumerate(parts):
        for existing_inst in list(part.recurse().getElementsByClass(music21.instrument.Instrument)):
            part.remove(existing_inst, recurse=True)
        m21_inst = _make_music21_instrument(instrument)
        if spec.transpose_semitones:
            m21_inst.transposition = music21.interval.Interval(spec.transpose_semitones)
            part.atSoundingPitch = False
        part.insert(0, m21_inst)

        measures = list(part.getElementsByClass(music21.stream.Measure))
        if not measures:
            if not part.getElementsByClass(music21.meter.TimeSignature):
                part.insert(0, music21.meter.TimeSignature("4/4"))
            if not part.getElementsByClass(music21.clef.Clef):
                part.insert(0, make_clef(spec.clef_mode, idx))
            _repair_unexportable_durations(part)
            part.makeMeasures(inPlace=True)
            part.makeNotation(inPlace=True)
            part.makeRests(fillGaps=True, inPlace=True)
            measures = list(part.getElementsByClass(music21.stream.Measure))

        if measures:
            first_m = measures[0]
            first_clef = first_m.getElementsByClass(music21.clef.Clef).first()
            force_clef = instrument == Instrument.GUITAR and not isinstance(
                first_clef, music21.clef.Treble8vbClef
            )
            replace_or_insert_initial_clef(
                part, make_clef(spec.clef_mode, idx), force_replace=force_clef
            )

            if not first_m.getElementsByClass(music21.meter.TimeSignature):
                ts = part.recurse().getElementsByClass(music21.meter.TimeSignature).first()
                first_m.insert(
                    0, copy.deepcopy(ts) if ts is not None else music21.meter.TimeSignature("4/4")
                )
            if not first_m.getElementsByClass(music21.key.KeySignature):
                ks = part.recurse().getElementsByClass(music21.key.KeySignature).first()
                if ks is not None:
                    first_m.insert(0, copy.deepcopy(ks))

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


def _redistribute_chord_technical_tags(root: ET.Element) -> bool:
    """Fix music21 Chord serialization by pairing <string>/<fret> 1-to-1 across chord notes."""
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
    return changed


def _normalize_power_chord_harmonies(root: ET.Element) -> bool:
    """Convert music21's `<kind>none</kind>` + `<degree>5</degree>` into standard `<kind text='5'>power</kind>`."""
    changed = False
    for harmony in root.findall(".//harmony"):
        kind_el = harmony.find("kind")
        degrees = harmony.findall("degree")
        if kind_el is None or kind_el.text != "none" or len(degrees) != 1:
            continue
        deg = degrees[0]
        if (
            deg.findtext("degree-value") == "5"
            and (deg.findtext("degree-alter") or "0") == "0"
            and deg.findtext("degree-type") == "add"
        ):
            kind_el.text = "power"
            kind_el.set("text", "5")
            harmony.remove(deg)
            changed = True
    return changed


_ATTR_TAG_ORDER: dict[str, int] = {
    "divisions": 0,
    "key": 1,
    "time": 2,
    "staves": 3,
    "part-symbol": 4,
    "instruments": 5,
    "clef": 6,
    "staff-details": 7,
    "transpose": 8,
    "directive": 9,
    "measure-style": 10,
}

_NOTE_TAG_ORDER: dict[str, int] = {
    "grace": 0,
    "chord": 1,
    "pitch": 2,
    "unpitched": 2,
    "rest": 2,
    "duration": 3,
    "tie": 4,
    "instrument": 5,
    "voice": 6,
    "type": 7,
    "dot": 8,
    "accidental": 9,
    "time-modification": 10,
    "stem": 11,
    "notehead": 12,
    "notehead-text": 13,
    "staff": 14,
    "beam": 15,
    "notations": 16,
    "lyric": 17,
}


def _sort_children_by_order(parent: ET.Element, order_map: dict[str, int]) -> None:
    children = list(parent)
    children.sort(key=lambda el: order_map.get(el.tag, 99))
    parent[:] = children


def _build_staff_details_element(
    meta: GuitarTrackMeta,
    *,
    staff_number: int | None = None,
) -> ET.Element:
    attrib = {"number": str(staff_number)} if staff_number is not None else {}
    details = ET.Element("staff-details", attrib)
    ET.SubElement(details, "staff-lines").text = str(len(meta.open_string_midi))
    for line_idx, midi_val in enumerate(reversed(meta.open_string_midi), start=1):
        pitch = music21.pitch.Pitch(midi=midi_val)
        tuning_el = ET.SubElement(details, "staff-tuning", {"line": str(line_idx)})
        ET.SubElement(tuning_el, "tuning-step").text = pitch.step
        alter_int = round(pitch.alter) if pitch.alter else 0
        if alter_int != 0:
            ET.SubElement(tuning_el, "tuning-alter").text = str(alter_int)
        ET.SubElement(tuning_el, "tuning-octave").text = str(int(pitch.octave or 0))
    if meta.capo_fret > 0:
        ET.SubElement(details, "capo").text = str(meta.capo_fret)
    return details


def _shift_note_octave(note_el: ET.Element, delta: int) -> None:
    oct_el = note_el.find("./pitch/octave")
    if oct_el is not None and (oct_val := try_int(oct_el.text)) is not None:
        oct_el.text = str(oct_val + delta)


def _set_or_replace_staff(note_el: ET.Element, staff_num: int) -> None:
    staff_el = note_el.find("staff")
    if staff_el is None:
        staff_el = ET.Element("staff")
        note_el.append(staff_el)
    staff_el.text = str(staff_num)
    _sort_children_by_order(note_el, _NOTE_TAG_ORDER)


def _timeline_delta(child: ET.Element) -> int | None:
    if child.tag == "note":
        if child.find("grace") is not None or child.find("chord") is not None:
            return None
        return try_int(child.findtext("duration"))
    if child.tag == "forward":
        return try_int(child.findtext("duration"))
    if child.tag == "backup":
        dur = try_int(child.findtext("duration"))
        return -dur if dur is not None else None
    return None


def _strip_technical_tags(note_el: ET.Element, *tag_names: str) -> None:
    tech = note_el.find("notations/technical")
    if tech is None:
        return
    for tag in tag_names:
        for child in list(tech.findall(tag)):
            tech.remove(child)
    if not list(tech):
        notations = note_el.find("notations")
        if notations is not None:
            notations.remove(tech)
            if not list(notations):
                note_el.remove(notations)


def _apply_guitar_staff_layout(
    root: ET.Element,
    guitar_staff: GuitarStaffMode,
    guitar_meta: GuitarTrackMeta,
) -> bool:
    parts = root.findall("part")
    if not parts:
        return False
    part = parts[0]
    measures = part.findall("measure")
    if not measures:
        return False

    is_nonstandard = (
        guitar_meta.open_string_midi != INSTRUMENT_SPECS[Instrument.GUITAR].open_string_midi
        or guitar_meta.capo_fret > 0
    )
    if guitar_staff == GuitarStaffMode.NOTATION and not is_nonstandard:
        return False

    first_attrs = measures[0].find("attributes")
    if first_attrs is None:
        first_attrs = ET.Element("attributes")
        measures[0].insert(0, first_attrs)

    if guitar_staff == GuitarStaffMode.NOTATION:
        if not first_attrs.findall("staff-details"):
            first_attrs.append(_build_staff_details_element(guitar_meta))
            _sort_children_by_order(first_attrs, _ATTR_TAG_ORDER)
        return True

    if guitar_staff == GuitarStaffMode.TAB:
        for clef_el in list(first_attrs.findall("clef")):
            first_attrs.remove(clef_el)
        tab_clef = ET.SubElement(first_attrs, "clef")
        ET.SubElement(tab_clef, "sign").text = "TAB"
        ET.SubElement(tab_clef, "line").text = "5"
        transposes = list(first_attrs.findall("transpose"))
        had_8vb_transpose = any(try_int(tr.findtext("octave-change")) == -1 for tr in transposes)
        for tr_el in transposes:
            first_attrs.remove(tr_el)
        for sd_el in list(first_attrs.findall("staff-details")):
            first_attrs.remove(sd_el)
        first_attrs.append(_build_staff_details_element(guitar_meta))
        _sort_children_by_order(first_attrs, _ATTR_TAG_ORDER)
        if had_8vb_transpose:
            for note_el in part.findall(".//note"):
                _shift_note_octave(note_el, -1)
        return True

    # GuitarStaffMode.BOTH: Staff 1 Notation (written 8vb) + Staff 2 TAB (sounding)
    staves_el = first_attrs.find("staves")
    if staves_el is None:
        staves_el = ET.SubElement(first_attrs, "staves")
    staves_el.text = "2"

    clefs = first_attrs.findall("clef")
    if clefs:
        clefs[0].set("number", "1")
    else:
        c1 = ET.SubElement(first_attrs, "clef", {"number": "1"})
        ET.SubElement(c1, "sign").text = "G"
        ET.SubElement(c1, "line").text = "2"
        ET.SubElement(c1, "clef-octave-change").text = "-1"
    c2 = ET.SubElement(first_attrs, "clef", {"number": "2"})
    ET.SubElement(c2, "sign").text = "TAB"
    ET.SubElement(c2, "line").text = "5"

    transposes = list(first_attrs.findall("transpose"))
    had_8vb_transpose = any(try_int(tr.findtext("octave-change")) == -1 for tr in transposes)
    for tr_el in transposes:
        tr_el.set("number", "1")
    for sd_el in list(first_attrs.findall("staff-details")):
        first_attrs.remove(sd_el)
    first_attrs.append(_build_staff_details_element(guitar_meta, staff_number=2))
    _sort_children_by_order(first_attrs, _ATTR_TAG_ORDER)

    voice_shift = max(
        4,
        max(
            (
                v
                for m in measures
                for ch in m
                if ch.tag == "note" and (v := try_int(ch.findtext("voice"))) is not None
            ),
            default=0,
        ),
    )

    for measure in measures:
        orig_children = list(measure)
        timeline_elems: list[ET.Element] = []
        prefix_elems: list[ET.Element] = []
        prefix_delta = 0
        seen_note = False
        staff1_cursor = 0
        for child in orig_children:
            if child.tag not in {"note", "forward", "backup"}:
                continue
            delta = _timeline_delta(child)
            if not seen_note:
                if child.tag == "note":
                    seen_note = True
                    if prefix_delta != 0:
                        timeline_elems.extend(prefix_elems)
                    timeline_elems.append(child)
                else:
                    prefix_elems.append(child)
                    if delta is not None:
                        prefix_delta += delta
            else:
                timeline_elems.append(child)
            if delta is not None:
                staff1_cursor += delta

        if not timeline_elems:
            continue

        cloned_staff2: list[ET.Element] = []
        for elem in timeline_elems:
            clone = copy.deepcopy(elem)
            if elem.tag == "note":
                _strip_technical_tags(elem, "string", "fret")
                _set_or_replace_staff(elem, 1)

                if had_8vb_transpose:
                    _shift_note_octave(clone, -1)
                voice_el = clone.find("voice")
                if voice_el is not None and (v_num := try_int(voice_el.text)) is not None:
                    voice_el.text = str(v_num + voice_shift)
                for drop_tag in ("stem", "beam", "accidental", "notehead", "lyric"):
                    for d_el in list(clone.findall(drop_tag)):
                        clone.remove(d_el)
                _strip_technical_tags(clone, "fingering")
                _set_or_replace_staff(clone, 2)
            cloned_staff2.append(clone)

        insert_pos = len(orig_children)
        right_bar = measure.find("barline[@location='right']")
        if right_bar is not None:
            insert_pos = list(measure).index(right_bar)

        extras: list[ET.Element] = []
        if staff1_cursor > 0:
            backup_el = ET.Element("backup")
            ET.SubElement(backup_el, "duration").text = str(staff1_cursor)
            extras.append(backup_el)
        extras.extend(cloned_staff2)
        for offset_i, new_el in enumerate(extras):
            measure.insert(insert_pos + offset_i, new_el)

    return True


def _validate_part_measures(parts: list[ET.Element]) -> None:
    """Ensure no measure contains duplicate whole-bar rests per staff or overflows its time signature."""
    for part in parts:
        divisions: int | None = None
        beats: int | None = None
        beat_type: int | None = None
        senza_misura = False

        for measure in part.findall("measure"):
            m_num = measure.attrib.get("number", "?")
            rests_by_staff: dict[str, int] = {}
            for note_el in measure.findall("./note"):
                rest_el = note_el.find("rest")
                if rest_el is not None and rest_el.attrib.get("measure") == "yes":
                    staff_id = note_el.findtext("staff") or "1"
                    rests_by_staff[staff_id] = rests_by_staff.get(staff_id, 0) + 1
            if any(cnt > 1 for cnt in rests_by_staff.values()):
                raise ValueError(f"Measure {m_num} contains duplicate whole-measure rests")

            for attrs in measure.findall("attributes"):
                if (div_val := try_int(attrs.findtext("divisions"))) is not None:
                    divisions = div_val
                time_el = attrs.find("time")
                if time_el is not None:
                    if time_el.find("senza-misura") is not None:
                        senza_misura = True
                    else:
                        senza_misura = False
                        b_val = try_int(time_el.findtext("beats"))
                        bt_val = try_int(time_el.findtext("beat-type"))
                        if b_val and bt_val and b_val > 0 and bt_val > 0:
                            beats, beat_type = b_val, bt_val

            if senza_misura or divisions is None or beats is None or beat_type is None:
                continue
            if measure.attrib.get("implicit") == "yes":
                continue

            expected_divs = Fraction(divisions * beats * 4, beat_type)
            cursor = 0
            max_cursor = 0
            for child in measure:
                if (delta := _timeline_delta(child)) is not None:
                    cursor += delta
                    max_cursor = max(max_cursor, cursor)

            if max_cursor > expected_divs + 1:
                raise ValueError(
                    f"Measure {m_num} overflows active {beats}/{beat_type} meter "
                    f"({max_cursor} > {expected_divs} divisions)"
                )


def _postprocess_and_validate_xml(
    xml_path: Path,
    *,
    guitar_staff: GuitarStaffMode = GuitarStaffMode.NOTATION,
    guitar_meta: GuitarTrackMeta | None = None,
) -> ExportMetrics:
    """Redistribute chord technical tags, apply guitar staff layout, validate, and count."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag not in ("score-partwise", "score-timewise"):
        raise ValueError(f"Invalid MusicXML root element: {root.tag}")

    changed = _redistribute_chord_technical_tags(root)
    changed = _normalize_power_chord_harmonies(root) or changed
    if guitar_meta is not None:
        changed = _apply_guitar_staff_layout(root, guitar_staff, guitar_meta) or changed
    if changed:
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    xml_measures = root.findall(".//measure")
    if not xml_measures:
        raise ValueError("Exported MusicXML contains zero <measure> elements")

    xml_notes = root.findall(".//note")
    for note_el in xml_notes:
        if note_el.find("grace") is None and note_el.find("duration") is None:
            raise ValueError("Exported MusicXML contains a <note> missing <duration>")

    parts = root.findall("part")
    if parts:
        _validate_part_measures(parts)

    has_linked_tab_staff = (
        len(parts) == 1
        and any(c.findtext("sign") == "TAB" for c in parts[0].findall("./measure[1]/attributes/clef[@number='2']"))
    )
    measure_count = len(parts[0].findall("measure")) if parts else len(xml_measures)
    note_count = sum(
        1
        for n in xml_notes
        if n.find("rest") is None
        and not (has_linked_tab_staff and n.findtext("staff") == "2")
    )
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
    *,
    guitar_staff: GuitarStaffMode = GuitarStaffMode.NOTATION,
) -> tuple[ExportMetrics, list[str]]:
    """Normalize, serialize, and validate a MusicXML score with graceful fallback."""
    warnings: list[str] = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    guitar_meta: GuitarTrackMeta | None = None
    if instrument == Instrument.GUITAR:
        part0 = score.parts[0] if list(score.parts) else None
        guitar_meta = (
            (get_guitar_track_meta(part0) if part0 is not None else None)
            or GuitarTrackMeta(open_string_midi=INSTRUMENT_SPECS[Instrument.GUITAR].open_string_midi)
        )

    normalized = normalize_score_structure(score, instrument=instrument, title=title)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}_",
            suffix=".musicxml",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
        try:
            normalized.write("musicxml", fp=str(tmp_path))
        except (music21.exceptions21.Music21Exception, OSError, ValueError, TypeError) as exc:
            warnings.append(f"Enrichment tag fallback triggered during MusicXML write: {exc}")
            _strip_non_note_articulations(normalized)
            normalized.write("musicxml", fp=str(tmp_path))

        metrics = _postprocess_and_validate_xml(
            tmp_path,
            guitar_staff=guitar_staff,
            guitar_meta=guitar_meta,
        )
        os.replace(tmp_path, output_path)
        tmp_path = None
        return metrics, warnings
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
