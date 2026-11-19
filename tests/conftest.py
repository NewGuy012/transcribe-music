"""Shared synthetic score builders and helpers for tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import music21
import pytest


def note_tab_tags(el: music21.note.Note | music21.chord.Chord) -> tuple[list[int], list[int]]:
    """Return (string_numbers, fret_numbers) attached to a Note or Chord."""
    strings = [
        int(a.number)
        for a in el.articulations
        if isinstance(a, music21.articulations.StringIndication) and a.number is not None
    ]
    frets = [
        int(a.number)
        for a in el.articulations
        if isinstance(a, music21.articulations.FretIndication) and a.number is not None
    ]
    return strings, frets


def make_simple_cello_score(title: str = "Cello Fixture") -> music21.stream.Score:
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title)
    part = music21.stream.Part(id="SoloCello")
    part.partName = "Solo Violoncello"
    part.insert(0, music21.instrument.Violoncello())
    m = music21.stream.Measure(number=1)
    m.insert(0, music21.clef.BassClef())
    m.insert(0, music21.meter.TimeSignature("4/4"))
    m.insert(0, music21.key.KeySignature(2))
    for p in ("D3", "F#3", "A3", "D4"):
        m.append(music21.note.Note(p, quarterLength=1.0))
    part.append(m)
    score.insert(0, part)
    return score


def make_concerto_score() -> music21.stream.Score:
    """Create a multi-part score with Orchestra Strings and Solo Cello (Bass->Tenor clef)."""
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title="Haydn D Major Sample")

    orch = music21.stream.Part(id="Strings")
    orch.partName = "Orchestra Strings"
    orch.insert(0, music21.instrument.StringInstrument())
    m1_o = music21.stream.Measure(number=1)
    m1_o.insert(0, music21.meter.TimeSignature("4/4"))
    m1_o.insert(0, music21.key.KeySignature(2))
    m1_o.insert(0, music21.chord.Chord(["D4", "F#4", "A4"], quarterLength=4.0))
    m2_o = music21.stream.Measure(number=2)
    m2_o.insert(0, music21.chord.Chord(["A3", "C#4", "E4", "G4"], quarterLength=4.0))
    orch.append(m1_o)
    orch.append(m2_o)

    cello = music21.stream.Part(id="SoloCello")
    cello.partName = "Solo Violoncello"
    cello.insert(0, music21.instrument.Violoncello())
    m1_c = music21.stream.Measure(number=1)
    m1_c.insert(0, music21.clef.BassClef())
    m1_c.insert(0, music21.meter.TimeSignature("4/4"))
    m1_c.insert(0, music21.key.KeySignature(2))
    m1_c.insert(0, music21.note.Rest(quarterLength=4.0))
    m2_c = music21.stream.Measure(number=2)
    m2_c.insert(0, music21.clef.TenorClef())
    for p in ("A3", "C#4", "E4", "A4"):
        m2_c.append(music21.note.Note(p, quarterLength=1.0))
    cello.append(m1_c)
    cello.append(m2_c)

    tutti = music21.stream.Part(id="TuttiCello")
    tutti.partName = "Violoncello Tutti"
    tutti.insert(0, music21.instrument.Violoncello())
    for p in ("D2", "D2", "A2", "D2", "D2", "D2", "A2", "D2"):
        tutti.append(music21.note.Note(p, quarterLength=1.0))
    tutti.makeMeasures(inPlace=True)

    score.insert(0, orch)
    score.insert(0, cello)
    score.insert(0, tutti)
    return score


def make_progression_score() -> music21.stream.Score:
    """Build C - Am - F - G 2-staff score for chord & piano tests."""
    s = music21.stream.Score()
    rh = music21.stream.Part(id="RH")
    lh = music21.stream.Part(id="LH")
    for rh_notes, lh_bass in (
        (["C4", "E4", "G4", "E4"], "C3"),
        (["A4", "C5", "E5", "C5"], "A2"),
        (["F4", "A4", "C5", "A4"], "F2"),
        (["G4", "B4", "D5", "B4"], "G2"),
    ):
        for p in rh_notes:
            rh.append(music21.note.Note(p, quarterLength=1.0))
        lh.append(music21.note.Note(lh_bass, quarterLength=4.0))
    rh.makeMeasures(inPlace=True)
    lh.makeMeasures(inPlace=True)
    s.insert(0, rh)
    s.insert(0, lh)
    return s


def write_mxl_bytes(score: music21.stream.Score, tmp_path: Path) -> bytes:
    out = tmp_path / "_fixture.mxl"
    score.write("mxl", fp=str(out))
    return out.read_bytes()


def write_pdf_file(path: Path, labels: tuple[str, ...] = ("Page 1", "Page 2")) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for label in labels:
        doc.new_page().insert_text((72, 72), label)
    doc.save(str(path))
    doc.close()
    return path


def write_gp5_file(
    path: Path,
    *,
    notes: tuple[tuple[int, int], ...] = ((6, 0), (5, 0), (4, 0), (1, 0)),
    chord_beats: tuple[tuple[tuple[int, int], ...], ...] = (),
    with_drums: bool = False,
    tuning: tuple[int, ...] | None = None,
    capo: int = 0,
    with_effects: bool = False,
) -> Path:
    guitarpro = pytest.importorskip("guitarpro")
    gm = pytest.importorskip("guitarpro.models")

    song = gm.Song(title="GP5 Study", artist="Test Composer", tempo=108)
    gtr_track = song.tracks[0]
    gtr_track.name = "Nylon Guitar"
    gtr_track.channel.instrument = 24
    gtr_track.offset = capo
    if tuning is not None:
        gtr_track.strings = [gm.GuitarString(i + 1, val) for i, val in enumerate(tuning)]
    if with_drums:
        drum_track = gm.Track(song, number=2, name="Drums", isPercussionTrack=True)
        drum_track.channel.channel = 9
        drum_track.measures.append(gm.Measure(drum_track, song.measureHeaders[0]))
        song.tracks.append(drum_track)

    if with_effects:
        song.measureHeaders[0].isRepeatOpen = True
        song.measureHeaders[0].repeatClose = 2

    v0 = gtr_track.measures[0].voices[0]
    beat_groups = list(chord_beats) if chord_beats else [(nf,) for nf in notes]
    for b_idx, group in enumerate(beat_groups):
        beat = gm.Beat(v0, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
        for string_num, fret in group:
            note = gm.Note(beat, value=fret, string=string_num, type=gm.NoteType.normal)
            if with_effects:
                if b_idx == 0:
                    note.effect.hammer = True
                    note.effect.palmMute = True
                elif b_idx == 1:
                    note.effect.slides = [gm.SlideType.shiftSlideTo]
                    note.effect.harmonic = gm.NaturalHarmonic()
                elif b_idx == 2:
                    note.effect.bend = gm.BendEffect(
                        type=gm.BendType.bend,
                        value=100,
                        points=[gm.BendPoint(0, 0), gm.BendPoint(6, 4), gm.BendPoint(12, 4)],
                    )
            beat.notes.append(note)
        v0.beats.append(beat)
    guitarpro.write(song, str(path))
    return path


def write_gpif_file(
    path: Path,
    *,
    title: str = "GPIF Study",
    artist: str = "GPIF Artist",
    tuning_low_to_high: str = "38 45 50 55 59 64",
    capo: int = 2,
    doctype: str = "",
) -> Path:
    """Create a synthetic Guitar Pro 7/8 `.gp` archive with `Content/score.gpif`."""
    gpif_xml = (
        f'<?xml version="1.0" encoding="utf-8"?>{doctype}<GPIF>'
        f"<Score><Title>{title}</Title><Artist>{artist}</Artist></Score>"
        "<MasterTrack><Automations><Automation><Type>Tempo</Type><Value>112 2</Value></Automation></Automations></MasterTrack>"
        "<Tracks>"
        '<Track id="0"><Name>Lead Guitar</Name>'
        "<GeneralMidi><Program>27</Program><PrimaryChannel>0</PrimaryChannel></GeneralMidi>"
        f'<Staves><Staff><Properties><Property name="Tuning"><Pitches>{tuning_low_to_high}</Pitches></Property>'
        f'<Property name="CapoFret"><Fret>{capo}</Fret></Property></Properties></Staff></Staves></Track>'
        '<Track id="1"><Name>Drumkit</Name><InstrumentSet><Type>drumKit</Type></InstrumentSet>'
        "<GeneralMidi><Program>0</Program><PrimaryChannel>9</PrimaryChannel></GeneralMidi></Track>"
        "</Tracks>"
        '<MasterBars><MasterBar><Time>4/4</Time><Key><AccidentalCount>1</AccidentalCount></Key>'
        '<Repeat start="true" end="true" count="2"/><Bars>0 1</Bars></MasterBar></MasterBars>'
        '<Bars><Bar id="0"><Voices>0 -1</Voices></Bar><Bar id="1"><Voices>1 -1</Voices></Bar></Bars>'
        '<Voices><Voice id="0"><Beats>0 1 2</Beats></Voice><Voice id="1"><Beats>3</Beats></Voice></Voices>'
        '<Beats><Beat id="0"><Rhythm ref="0"/><Notes>0 1</Notes></Beat>'
        '<Beat id="1"><Rhythm ref="0"/><Notes>2</Notes></Beat>'
        '<Beat id="2"><Rhythm ref="1"/><Notes>3</Notes></Beat>'
        '<Beat id="3"><Rhythm ref="2"/><Notes></Notes></Beat></Beats>'
        "<Notes>"
        '<Note id="0"><Properties><Property name="String"><String>0</String></Property>'
        '<Property name="Fret"><Fret>0</Fret></Property><Property name="PalmMuted"><Enable/></Property></Properties></Note>'
        '<Note id="1"><Properties><Property name="String"><String>5</String></Property>'
        '<Property name="Fret"><Fret>3</Fret></Property><Property name="HopOOrigin"><Enable/></Property></Properties></Note>'
        '<Note id="2"><Properties><Property name="String"><String>5</String></Property>'
        '<Property name="Fret"><Fret>5</Fret></Property><Property name="Bended"><Enable/></Property>'
        '<Property name="BendDestinationValue"><Float>100</Float></Property></Properties></Note>'
        '<Note id="3"><Properties><Property name="String"><String>4</String></Property>'
        '<Property name="Fret"><Fret>12</Fret></Property><Property name="Harmonic"><Enable/></Property></Properties></Note>'
        "</Notes>"
        '<Rhythms><Rhythm id="0"><NoteValue>Quarter</NoteValue></Rhythm>'
        '<Rhythm id="1"><NoteValue>Half</NoteValue></Rhythm>'
        '<Rhythm id="2"><NoteValue>Whole</NoteValue></Rhythm></Rhythms></GPIF>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Content/score.gpif", gpif_xml)
    return path


def write_ascii_tab_file(path: Path, content: str | None = None) -> Path:
    default_tab = (
        "Title: Riff Study\nArtist: Tab Author\nTime: 4/4\nCapo: 2\n\n"
        "e|---0---3h5---7\\5---12b--|\nB|---1--------------------|\n"
        "G|---0--------------------|\nD|---2--------------------|\n"
        "A|---3--------------------|\nE|------------------------|\n"
    )
    path.write_text(content if content is not None else default_tab, encoding="utf-8")
    return path
