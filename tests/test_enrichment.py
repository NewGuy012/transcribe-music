"""Tests for symbolic chord recognition, preservation flags, and all 4 fingering solvers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
from conftest import make_progression_score

from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.models import Instrument


def test_chord_recognition_viterbi_and_ensemble_reference(tmp_path: Path) -> None:
    score = make_progression_score()
    annotate_chords(score, window_beats=4.0)

    out_xml = tmp_path / "chords.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.PIANO)
    assert not warnings
    assert metrics.chord_count == 4
    root = ET.parse(out_xml).getroot()
    assert [h.findtext("root/root-step") for h in root.findall(".//harmony")] == ["C", "A", "F", "G"]
    assert [h.findtext("kind") for h in root.findall(".//harmony")] == [
        "major",
        "minor",
        "major",
        "major",
    ]

    full_score = make_progression_score()
    solo = music21.stream.Part(id="Cello")
    for _ in range(2):
        solo.append(music21.note.Rest(quarterLength=4.0))
    solo.append(music21.note.Note("F3", quarterLength=4.0))
    solo.append(music21.note.Note("G3", quarterLength=4.0))
    solo.makeMeasures(inPlace=True)
    full_score.insert(0, solo)

    solo_only = music21.stream.Score([solo])
    annotate_chords(solo_only, harmonic_reference_score=full_score, window_beats=4.0)
    solo_xml = tmp_path / "solo_chords.musicxml"
    solo_metrics, solo_warnings = export_musicxml(solo_only, solo_xml, Instrument.CELLO)
    assert not solo_warnings
    assert solo_metrics.chord_count == 4
    solo_root = ET.parse(solo_xml).getroot()
    assert [h.findtext("root/root-step") for h in solo_root.findall(".//harmony")] == [
        "C",
        "A",
        "F",
        "G",
    ]
    assert not solo_root.findall(".//harmony/offset")

    custom_m = music21.stream.Measure(number=1)
    custom_m.insert(0.0, music21.harmony.ChordSymbol("Dmaj7"))
    custom_m.insert(0.0, music21.note.Note("D3", quarterLength=4.0))
    custom = music21.stream.Score([music21.stream.Part([custom_m])])
    annotate_chords(custom, preserve_existing=True)
    assert [s.figure for s in custom.recurse().getElementsByClass(music21.harmony.ChordSymbol)] == [
        "Dmaj7"
    ]


def test_piano_fingering_notes_and_chords(tmp_path: Path) -> None:
    score = make_progression_score()
    rh_m5 = music21.stream.Measure(number=5)
    rh_m5.append(music21.chord.Chord(["C4", "E4", "G4"], quarterLength=4.0))
    score.parts[0].append(rh_m5)
    lh_m5 = music21.stream.Measure(number=5)
    lh_m5.append(music21.chord.Chord(["C3", "G3"], quarterLength=4.0))
    score.parts[1].append(lh_m5)

    apply_fingering(score, Instrument.PIANO)
    out_xml = tmp_path / "piano.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.PIANO)
    assert not warnings
    fingers = [f.text for f in ET.parse(out_xml).findall(".//technical/fingering")]
    assert len(fingers) == metrics.note_count
    assert all(f in {"1", "2", "3", "4", "5"} for f in fingers)


def test_guitar_fingering_tab_and_sounding_pitch_handling(tmp_path: Path) -> None:
    s = music21.stream.Score()
    p = music21.stream.Part()
    p.insert(0, music21.clef.Treble8vbClef())
    p.atSoundingPitch = True
    for pitch in ("E3", "F3", "G3", "A3"):
        p.append(music21.note.Note(pitch, quarterLength=1.0))
    p.append(music21.chord.Chord(["C4", "E4", "G4"], quarterLength=4.0))
    p.makeMeasures(inPlace=True)
    s.insert(0, p)

    apply_fingering(s, Instrument.GUITAR)
    out_xml = tmp_path / "guitar.musicxml"
    metrics, warnings = export_musicxml(s, out_xml, Instrument.GUITAR)
    assert not warnings
    assert metrics.fingering_count == 7

    notes_xml = [n for n in ET.parse(out_xml).findall(".//note") if n.find("rest") is None]
    assert notes_xml[0].findtext("notations/technical/string") == "6"
    assert notes_xml[0].findtext("notations/technical/fret") == "0"
    assert len({n.findtext("notations/technical/string") for n in notes_xml[-3:]}) == 3

    # Fresh sounding-pitch Part (atSoundingPitch='unknown', no clef) must not shift down -12
    raw_s = music21.stream.Score([music21.stream.Part([music21.note.Note("E2", quarterLength=4.0)])])
    apply_fingering(raw_s, Instrument.GUITAR)
    raw_arts = list(next(raw_s.recurse().notes).articulations)
    assert any(isinstance(a, music21.articulations.StringIndication) and a.number == 6 for a in raw_arts)
    assert any(isinstance(a, music21.articulations.FretIndication) and a.number == 0 for a in raw_arts)

    s_pres = music21.stream.Score()
    p_pres = music21.stream.Part()
    n_pres = music21.note.Note("E3", quarterLength=4.0)
    n_pres.articulations.append(music21.articulations.Fingering("4"))
    p_pres.append(n_pres)
    p_pres.makeMeasures(inPlace=True)
    s_pres.insert(0, p_pres)
    apply_fingering(s_pres, Instrument.GUITAR, preserve_existing=True)
    assert [a.fingerNumber for a in next(s_pres.recurse().notes).articulations] == ["4"]


def test_violin_and_cello_fingering_including_thumb_position(tmp_path: Path) -> None:
    s_vln = music21.stream.Score()
    p_vln = music21.stream.Part()
    for pitch in ("G3", "A3", "B3", "C4", "D4", "E4"):
        p_vln.append(music21.note.Note(pitch, quarterLength=1.0))
    p_vln.append(music21.chord.Chord(["G3", "D4"], quarterLength=2.0))
    p_vln.makeMeasures(inPlace=True)
    s_vln.insert(0, p_vln)

    apply_fingering(s_vln, Instrument.VIOLIN)
    m_vln, w_vln = export_musicxml(s_vln, tmp_path / "violin.musicxml", Instrument.VIOLIN)
    assert not w_vln
    assert m_vln.fingering_count == 8

    s_cel = music21.stream.Score()
    p_cel = music21.stream.Part()
    for pitch in ("C2", "D2", "E2", "F2", "D4", "E4", "F#4", "A4"):
        p_cel.append(music21.note.Note(pitch, quarterLength=1.0))
    p_cel.makeMeasures(inPlace=True)
    s_cel.insert(0, p_cel)

    apply_fingering(s_cel, Instrument.CELLO)
    cel_xml = tmp_path / "cello.musicxml"
    m_cel, w_cel = export_musicxml(s_cel, cel_xml, Instrument.CELLO)
    assert not w_cel
    assert m_cel.fingering_count == 8
    root_cel = ET.parse(cel_xml).getroot()
    assert all(f.text in {"0", "1", "2", "3", "4", "T"} for f in root_cel.findall(".//technical/fingering"))
    assert all(st.text in {"1", "2", "3", "4"} for st in root_cel.findall(".//technical/string"))
