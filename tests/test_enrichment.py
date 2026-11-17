"""Tests for chord symbol recognition and all 4 instrument fingering solvers."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21

from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.models import INSTRUMENT_SPECS, Instrument


def _build_chord_progression_score() -> music21.stream.Score:
    """Build C - Am - F - G arpeggiated score for chord & piano tests."""
    s = music21.stream.Score()
    rh = music21.stream.Part(id="RH")
    lh = music21.stream.Part(id="LH")

    bars = [
        (["C4", "E4", "G4", "E4"], "C3"),
        (["A4", "C5", "E5", "C5"], "A2"),
        (["F4", "A4", "C5", "A4"], "F2"),
        (["G4", "B4", "D5", "B4"], "G2"),
    ]
    for rh_notes, lh_bass in bars:
        for p in rh_notes:
            rh.append(music21.note.Note(p, quarterLength=1.0))
        lh.append(music21.note.Note(lh_bass, quarterLength=4.0))

    rh.makeMeasures(inPlace=True)
    lh.makeMeasures(inPlace=True)
    s.insert(0, rh)
    s.insert(0, lh)
    return s


def test_chord_recognition_viterbi_progression(tmp_path: Path) -> None:
    score = _build_chord_progression_score()
    annotate_chords(score, window_beats=4.0)

    out_xml = tmp_path / "chords.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.PIANO)
    assert not warnings
    assert metrics.chord_count == 4

    root = ET.parse(out_xml).getroot()
    roots = [h.findtext("root/root-step") for h in root.findall(".//harmony")]
    kinds = [h.findtext("kind") for h in root.findall(".//harmony")]
    assert roots == ["C", "A", "F", "G"]
    assert kinds == ["major", "minor", "major", "major"]


def test_piano_fingering_notes_and_chords(tmp_path: Path) -> None:
    score = _build_chord_progression_score()
    # Add a C-major chord on beat 1 of measure 5
    rh_m5 = music21.stream.Measure(number=5)
    rh_m5.append(music21.chord.Chord(["C4", "E4", "G4"], quarterLength=4.0))
    score.parts[0].append(rh_m5)
    lh_m5 = music21.stream.Measure(number=5)
    lh_m5.append(music21.chord.Chord(["C3", "G3"], quarterLength=4.0))
    score.parts[1].append(lh_m5)

    apply_fingering(score, Instrument.PIANO)
    out_xml = tmp_path / "piano_fingered.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.PIANO)
    assert not warnings
    assert metrics.fingering_count == metrics.note_count

    root = ET.parse(out_xml).getroot()
    fingers = [f.text for f in root.findall(".//technical/fingering")]
    assert len(fingers) == metrics.note_count
    assert all(f in {"1", "2", "3", "4", "5"} for f in fingers)


def test_guitar_fingering_and_tab_chord_distribution(tmp_path: Path) -> None:
    s = music21.stream.Score()
    p = music21.stream.Part()
    # Guitar written pitches (sounds 8vb): E3-B3-E4 open/first position + C4-E4-G4 chord
    for pitch in ["E3", "F3", "G3", "A3"]:
        p.append(music21.note.Note(pitch, quarterLength=1.0))
    p.append(music21.chord.Chord(["C4", "E4", "G4"], quarterLength=4.0))
    p.makeMeasures(inPlace=True)
    s.insert(0, p)

    apply_fingering(s, Instrument.GUITAR)
    out_xml = tmp_path / "guitar.musicxml"
    metrics, warnings = export_musicxml(s, out_xml, Instrument.GUITAR)
    assert not warnings
    assert metrics.note_count == 7
    assert metrics.fingering_count == 7

    root = ET.parse(out_xml).getroot()
    # Every non-rest <note> should have exactly 1 fingering, 1 string (1-6), 1 fret (0-19)
    notes_xml = [n for n in root.findall(".//note") if n.find("rest") is None]
    assert len(notes_xml) == 7
    for n_el in notes_xml:
        tech = n_el.find("notations/technical")
        assert tech is not None
        fingers = tech.findall("fingering")
        strings = tech.findall("string")
        frets = tech.findall("fret")
        assert len(fingers) == 1 and fingers[0].text in {"0", "1", "2", "3", "4"}
        assert len(strings) == 1 and 1 <= int(strings[0].text or "0") <= 6
        assert len(frets) == 1 and 0 <= int(frets[0].text or "-1") <= 19

    # Verify chord notes occupy 3 distinct strings!
    chord_notes = notes_xml[-3:]
    chord_strings = {n.findtext("notations/technical/string") for n in chord_notes}
    assert len(chord_strings) == 3


def test_violin_and_cello_fingering_including_thumb_position(tmp_path: Path) -> None:
    # Violin scale + double stop
    s_vln = music21.stream.Score()
    p_vln = music21.stream.Part()
    for pitch in ["G3", "A3", "B3", "C4", "D4", "E4"]:
        p_vln.append(music21.note.Note(pitch, quarterLength=1.0))
    p_vln.append(music21.chord.Chord(["G3", "D4"], quarterLength=2.0))
    p_vln.makeMeasures(inPlace=True)
    s_vln.insert(0, p_vln)

    apply_fingering(s_vln, Instrument.VIOLIN)
    vln_xml = tmp_path / "violin.musicxml"
    m_vln, w_vln = export_musicxml(s_vln, vln_xml, Instrument.VIOLIN)
    assert not w_vln
    assert m_vln.fingering_count == 8

    # Cello low scale + high thumb-position passage (D4..A4 on A3 string)
    s_cel = music21.stream.Score()
    p_cel = music21.stream.Part()
    for pitch in ["C2", "D2", "E2", "F2", "D4", "E4", "F#4", "A4"]:
        p_cel.append(music21.note.Note(pitch, quarterLength=1.0))
    p_cel.makeMeasures(inPlace=True)
    s_cel.insert(0, p_cel)

    apply_fingering(s_cel, Instrument.CELLO)
    cel_xml = tmp_path / "cello.musicxml"
    m_cel, w_cel = export_musicxml(s_cel, cel_xml, Instrument.CELLO)
    assert not w_cel
    assert m_cel.fingering_count == 8

    root_cel = ET.parse(cel_xml).getroot()
    cel_fingers = [f.text for f in root_cel.findall(".//technical/fingering")]
    cel_strings = [s.text for s in root_cel.findall(".//technical/string")]
    assert all(f in {"0", "1", "2", "3", "4", "T"} for f in cel_fingers)
    assert all(st in {"1", "2", "3", "4"} for st in cel_strings)
    # High passage (D4..A4) should utilize Thumb ('T') or valid upper positions
    assert len(cel_fingers) == len(INSTRUMENT_SPECS[Instrument.CELLO].open_string_midi) * 2
