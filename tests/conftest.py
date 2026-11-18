"""Shared synthetic music score and fixture builders for tests."""

from __future__ import annotations

from pathlib import Path

import music21
import pytest


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
    with_drums: bool = False,
) -> Path:
    guitarpro = pytest.importorskip("guitarpro")
    gm = pytest.importorskip("guitarpro.models")

    song = gm.Song(title="GP5 Study", artist="Test Composer", tempo=108)
    gtr_track = song.tracks[0]
    gtr_track.name = "Nylon Guitar"
    gtr_track.channel.instrument = 24
    if with_drums:
        drum_track = gm.Track(song, number=2, name="Drums", isPercussionTrack=True)
        drum_track.channel.channel = 9
        drum_track.measures.append(gm.Measure(drum_track, song.measureHeaders[0]))
        song.tracks.append(drum_track)

    v0 = gtr_track.measures[0].voices[0]
    for string_num, fret in notes:
        beat = gm.Beat(v0, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
        beat.notes.append(gm.Note(beat, value=fret, string=string_num, type=gm.NoteType.normal))
        v0.beats.append(beat)
    guitarpro.write(song, str(path))
    return path
