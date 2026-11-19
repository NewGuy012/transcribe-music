"""Tests for symbolic chord recognition, preservation flags, and all 4 fingering solvers."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import music21
from conftest import make_progression_score, note_tab_tags

from transcribe_music.cli import main
from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.models import GuitarStaffMode, Instrument
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    build_fretted_element,
    set_guitar_track_meta,
)


def test_chord_recognition_viterbi_and_ensemble_reference(tmp_path: Path) -> None:
    score = make_progression_score()
    annotate_chords(score, window_beats=4.0)

    out_xml = tmp_path / "chords.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.PIANO)
    assert not warnings
    assert metrics.chord_count == 4
    root = ET.parse(out_xml).getroot()
    assert [h.findtext("root/root-step") for h in root.findall(".//harmony")] == ["C", "A", "F", "G"]
    assert [h.findtext("kind") for h in root.findall(".//harmony")] == ["major", "minor", "major", "major"]

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
    assert [h.findtext("root/root-step") for h in solo_root.findall(".//harmony")] == ["C", "A", "F", "G"]
    assert not solo_root.findall(".//harmony/offset")

    custom_m = music21.stream.Measure(number=1)
    custom_m.insert(0.0, music21.harmony.ChordSymbol("Dmaj7"))
    custom_m.insert(0.0, music21.note.Note("D3", quarterLength=4.0))
    custom = music21.stream.Score([music21.stream.Part([custom_m])])
    annotate_chords(custom, preserve_existing=True)
    assert [s.figure for s in custom.recurse().getElementsByClass(music21.harmony.ChordSymbol)] == ["Dmaj7"]


def test_chords_key_enharmonics_slash_inversions_window_and_roman_numerals(tmp_path: Path) -> None:
    # Flat key (-3 flats, C minor): Db chord spells D- (<root-step>D</root-step><root-alter>-1</root-alter>)
    s_flat = music21.stream.Score()
    p_flat = music21.stream.Part()
    m1 = music21.stream.Measure(number=1)
    m1.insert(0.0, music21.key.KeySignature(-3))
    m1.insert(0.0, music21.chord.Chord(["C3", "E-3", "G3", "C4"], quarterLength=2.0))
    m1.insert(2.0, music21.chord.Chord(["D-3", "F3", "A-3", "D-4"], quarterLength=2.0))
    p_flat.append(m1)
    s_flat.insert(0, p_flat)
    flat_in = tmp_path / "flat_in.musicxml"
    s_flat.write("musicxml", fp=str(flat_in))

    flat_out = tmp_path / "flat_out.musicxml"
    assert (
        main(
            [
                "convert",
                str(flat_in),
                "-i",
                "piano",
                "--chord-window",
                "2.0",
                "--roman-numerals",
                "--no-cache",
                "-o",
                str(flat_out),
            ]
        )
        == 0
    )
    flat_root = ET.parse(flat_out).getroot()
    harms_flat = [
        (h.findtext("root/root-step"), h.findtext("root/root-alter"))
        for h in flat_root.findall(".//harmony")
    ]
    assert harms_flat == [("C", None), ("D", "-1")]
    words_flat = [w.text for w in flat_root.findall(".//direction/direction-type/words")]
    assert len(words_flat) == 2

    # Sharp key (+4 sharps, E major) spells C# minor rather than Db
    s_sharp = music21.stream.Score()
    p_sharp = music21.stream.Part()
    ms1 = music21.stream.Measure(number=1)
    ms1.insert(0.0, music21.key.KeySignature(4))
    ms1.insert(0.0, music21.chord.Chord(["C#3", "E3", "G#3", "C#4"], quarterLength=4.0))
    p_sharp.append(ms1)
    s_sharp.insert(0, p_sharp)
    annotate_chords(s_sharp, window_beats=4.0)
    assert [
        cs.figure for cs in s_sharp.recurse().getElementsByClass(music21.harmony.ChordSymbol)
    ] == ["C#m"]

    # Slash-chord inversions: C -> Dm7/C -> G7/B -> C, plus --no-slash-chords omitting <bass>
    s_inv = music21.stream.Score()
    p_inv = music21.stream.Part()
    for chord_tones in (
        ["C3", "E4", "G4", "C5"],
        ["C3", "D4", "F4", "A4"],
        ["B2", "D4", "F4", "G4"],
        ["C3", "E4", "G4", "C5"],
    ):
        p_inv.append(music21.chord.Chord(chord_tones, quarterLength=4.0))
    p_inv.makeMeasures(inPlace=True)
    s_inv.insert(0, p_inv)
    inv_in = tmp_path / "inv_in.musicxml"
    s_inv.write("musicxml", fp=str(inv_in))

    inv_with_slash = tmp_path / "inv_slash.musicxml"
    assert (
        main(
            [
                "convert",
                str(inv_in),
                "-i",
                "piano",
                "--chord-window",
                "4.0",
                "--no-cache",
                "-o",
                str(inv_with_slash),
            ]
        )
        == 0
    )
    inv_root = ET.parse(inv_with_slash).getroot()
    assert [
        (h.findtext("root/root-step"), h.findtext("kind"), h.findtext("bass/bass-step"))
        for h in inv_root.findall(".//harmony")
    ] == [
        ("C", "major", None),
        ("D", "minor-seventh", "C"),
        ("G", "dominant", "B"),
        ("C", "major", None),
    ]

    inv_no_slash = tmp_path / "inv_noslash.musicxml"
    assert (
        main(
            [
                "convert",
                str(inv_in),
                "-i",
                "piano",
                "--chord-window",
                "4.0",
                "--no-slash-chords",
                "--no-cache",
                "-o",
                str(inv_no_slash),
            ]
        )
        == 0
    )
    assert not ET.parse(inv_no_slash).findall(".//harmony/bass")

    # Custom chord_window_beats (1.0 vs 4.0) produces finer vs coarser harmony timeline
    s_beat = copy.deepcopy(s_flat)
    annotate_chords(s_beat, window_beats=1.0)
    cs_1 = list(s_beat.recurse().getElementsByClass(music21.harmony.ChordSymbol))
    s_bar = copy.deepcopy(s_flat)
    annotate_chords(s_bar, window_beats=4.0)
    cs_4 = list(s_bar.recurse().getElementsByClass(music21.harmony.ChordSymbol))
    assert len(cs_1) == 2 and len(cs_4) == 1


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

    # Left-hand ascending scale decreases 5->1, and bass-to-chord ostinato uses 5 on bass / 3-1 above
    def _run_lh(elements: list[music21.note.NotRest]) -> list[list[int]]:
        s_lh = music21.stream.Score()
        rh = music21.stream.Part(id="RH")
        rh.append(music21.note.Note("C5", quarterLength=4.0))
        rh.makeMeasures(inPlace=True)
        lh = music21.stream.Part(id="LH")
        for el in elements:
            lh.append(el)
        lh.makeMeasures(inPlace=True)
        s_lh.insert(0, rh)
        s_lh.insert(0, lh)
        apply_fingering(s_lh, Instrument.PIANO)
        return [
            [
                int(a.fingerNumber)
                for a in el.articulations
                if isinstance(a, music21.articulations.Fingering)
            ]
            for el in s_lh.parts[1].recurse().notes
        ]

    assert _run_lh([music21.note.Note(p, quarterLength=1.0) for p in ("C3", "D3", "E3", "F3", "G3")]) == [
        [5],
        [4],
        [3],
        [2],
        [1],
    ]
    assert _run_lh([music21.note.Note(p, quarterLength=1.0) for p in ("G3", "F3", "E3", "D3", "C3")]) == [
        [1],
        [2],
        [3],
        [4],
        [5],
    ]
    ostinato_fings = _run_lh(
        [
            music21.note.Note("F3", quarterLength=0.5),
            music21.chord.Chord(["C4", "F4"], quarterLength=0.5),
            music21.note.Note("F3", quarterLength=0.5),
            music21.chord.Chord(["C4", "F4"], quarterLength=0.5),
        ]
    )
    assert ostinato_fings[0] == [5]
    assert ostinato_fings[1][0] > ostinato_fings[1][1] and ostinato_fings[1][1] == 1


def test_guitar_fingering_tab_locked_tuning_and_chords(tmp_path: Path) -> None:
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

    # Sounding-pitch Part without clef must not shift down -12
    raw_s = music21.stream.Score([music21.stream.Part([music21.note.Note("E2", quarterLength=4.0)])])
    apply_fingering(raw_s, Instrument.GUITAR)
    assert note_tab_tags(next(raw_s.recurse().notes)) == ([6], [0])

    s_pres = music21.stream.Score()
    p_pres = music21.stream.Part()
    n_pres = music21.note.Note("E3", quarterLength=4.0)
    n_pres.articulations.append(music21.articulations.Fingering("4"))
    p_pres.append(n_pres)
    p_pres.makeMeasures(inPlace=True)
    s_pres.insert(0, p_pres)
    apply_fingering(s_pres, Instrument.GUITAR, preserve_existing=True)
    assert [a.fingerNumber for a in next(s_pres.recurse().notes).articulations] == ["4"]

    # Locked TAB preserves authored positions while recompute moves to open/2nd fret
    s_locked = music21.stream.Score()
    p_locked = music21.stream.Part()
    p_locked.insert(0, music21.clef.Treble8vbClef())
    p_locked.atSoundingPitch = False
    for written_pitch, fret in (("A3", 5), ("B3", 7)):
        n = music21.note.Note(written_pitch, quarterLength=2.0)
        n.articulations.append(music21.articulations.StringIndication(6))
        n.articulations.append(music21.articulations.FretIndication(fret))
        p_locked.append(n)
    p_locked.makeMeasures(inPlace=True)
    s_locked.insert(0, p_locked)

    s_recomputed = copy.deepcopy(s_locked)
    apply_fingering(s_locked, Instrument.GUITAR, preserve_existing_tab=True)
    assert [note_tab_tags(n) for n in s_locked.parts[0].recurse().notes] == [([6], [5]), ([6], [7])]

    apply_fingering(s_recomputed, Instrument.GUITAR, preserve_existing_tab=False)
    assert [note_tab_tags(n)[0][0] for n in s_recomputed.parts[0].recurse().notes] == [5, 5]

    # Drop-D + 24th fret E6 (= written E7)
    s_drop = music21.stream.Score()
    p_drop = music21.stream.Part()
    p_drop.insert(0, music21.clef.Treble8vbClef())
    p_drop.atSoundingPitch = False
    set_guitar_track_meta(p_drop, GuitarTrackMeta(open_string_midi=(64, 59, 55, 50, 45, 38)))
    p_drop.append(music21.note.Note("D3", quarterLength=2.0))
    p_drop.append(music21.note.Note("E7", quarterLength=2.0))
    p_drop.makeMeasures(inPlace=True)
    s_drop.insert(0, p_drop)
    apply_fingering(s_drop, Instrument.GUITAR, preserve_existing_tab=False)
    assert [note_tab_tags(n) for n in s_drop.parts[0].recurse().notes] == [([6], [0]), ([1], [24])]

    # Power/sus/add9 chords + standard triads
    s_gc = music21.stream.Score()
    p_gc = music21.stream.Part()
    p_gc.append(music21.chord.Chord(["E2", "B2", "E3"], quarterLength=4.0))
    p_gc.append(music21.chord.Chord(["D3", "A3", "D4", "G4"], quarterLength=4.0))
    p_gc.append(music21.chord.Chord(["C3", "E3", "G3", "D4"], quarterLength=4.0))
    p_gc.makeMeasures(inPlace=True)
    s_gc.insert(0, p_gc)
    annotate_chords(s_gc, window_beats=4.0, instrument=Instrument.GUITAR)
    assert [cs.figure for cs in s_gc.recurse().getElementsByClass(music21.harmony.ChordSymbol)] == [
        "E5",
        "Dsus4",
        "Cadd9",
    ]
    gc_xml = tmp_path / "guitar_chords.musicxml"
    _, gc_warnings = export_musicxml(s_gc, gc_xml, Instrument.GUITAR)
    assert not gc_warnings
    gc_harmonies = ET.parse(gc_xml).findall(".//harmony")
    assert gc_harmonies[0].findtext("kind") == "power"
    kind0 = gc_harmonies[0].find("kind")
    assert kind0 is not None and kind0.attrib.get("text") == "5"

    # Multi-voice measure with mid-bar harmony should not clone harmony cursor forwards to staff 2
    s_mv = music21.stream.Score()
    p_mv = music21.stream.Part()
    m_mv = music21.stream.Measure(number=1)
    m_mv.insert(0.0, music21.meter.TimeSignature("4/4"))
    m_mv.insert(0.0, music21.harmony.ChordSymbol("Am"))
    m_mv.insert(2.0, music21.harmony.ChordSymbol("E7"))
    v1 = music21.stream.Voice(id="1")
    v1.insert(0.0, music21.note.Note("C5", quarterLength=4.0))
    v2 = music21.stream.Voice(id="2")
    v2.insert(0.0, music21.note.Note("A3", quarterLength=4.0))
    m_mv.insert(0.0, v1)
    m_mv.insert(0.0, v2)
    p_mv.append(m_mv)
    s_mv.insert(0, p_mv)
    mv_xml = tmp_path / "mv_both.musicxml"
    _, mv_warnings = export_musicxml(
        s_mv, mv_xml, Instrument.GUITAR, guitar_staff=GuitarStaffMode.BOTH
    )
    assert not mv_warnings
    assert len(ET.parse(mv_xml).findall(".//measure[@number='1']/forward")) == 1

    # Unison double-stop on distinct strings + natural harmonic chord + multi-voice fingering placement
    s_engr = music21.stream.Score()
    p_engr = music21.stream.Part()
    p_engr.insert(0, music21.clef.Treble8vbClef())
    p_engr.atSoundingPitch = False
    m_engr = music21.stream.Measure(number=1)
    m_engr.insert(0.0, music21.meter.TimeSignature("4/4"))
    ev1 = music21.stream.Voice(id="1")
    unison_chord = build_fretted_element(
        [(64 + 12, 1, 0), (64 + 12, 2, 5)], music21.duration.Duration(2.0)
    )
    harm_chord = build_fretted_element(
        [(62 + 12, 4, 12), (67 + 12, 3, 12), (71 + 12, 2, 12)],
        music21.duration.Duration(2.0),
    )
    harm_chord.articulations.append(music21.articulations.Harmonic())
    ev1.insert(0.0, unison_chord)
    ev1.insert(2.0, harm_chord)
    ev2 = music21.stream.Voice(id="2")
    ev2.insert(0.0, music21.note.Note("E3", quarterLength=4.0))
    m_engr.insert(0.0, ev1)
    m_engr.insert(0.0, ev2)
    p_engr.append(m_engr)
    s_engr.insert(0, p_engr)

    apply_fingering(s_engr, Instrument.GUITAR, preserve_existing_tab=True)
    engr_xml = tmp_path / "guitar_engraving.musicxml"
    _, engr_warnings = export_musicxml(
        s_engr, engr_xml, Instrument.GUITAR, guitar_staff=GuitarStaffMode.BOTH
    )
    assert not engr_warnings
    engr_root = ET.parse(engr_xml).getroot()
    s2_unison = [
        (n.findtext("notations/technical/string"), n.findtext("notations/technical/fret"))
        for n in engr_root.findall(".//measure[@number='1']/note")
        if n.findtext("staff") == "2" and n.findtext("voice") == "5"
    ][:2]
    assert s2_unison == [("2", "5"), ("1", "0")]

    s1_notes = [
        n
        for n in engr_root.findall(".//measure[@number='1']/note")
        if n.findtext("staff") == "1"
    ]
    v1_fings = [
        f.get("placement")
        for n in s1_notes
        if n.findtext("voice") == "1"
        for f in n.findall("./notations/technical/fingering")
    ]
    v2_fings = [
        f.get("placement")
        for n in s1_notes
        if n.findtext("voice") == "2"
        for f in n.findall("./notations/technical/fingering")
    ]
    # Unison (2 notes) gets fingerings above; 3-note harmonic omits barre <fingering> clutter; voice 2 gets below
    assert v1_fings == ["above", "above"]
    assert v2_fings == ["below"]
    s2_harm_frets = [
        n.findtext("notations/technical/fret")
        for n in engr_root.findall(".//measure[@number='1']/note")
        if n.findtext("staff") == "2" and n.findtext("voice") == "5"
    ][2:]
    assert s2_harm_frets == ["12", "12", "12"]


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
