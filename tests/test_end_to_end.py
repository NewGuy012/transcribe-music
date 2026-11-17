"""End-to-end integration tests across pipeline, CLI, Verovio, and DuckDB cache."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import numpy as np
import pretty_midi
import pymupdf
import pytest
import soundfile as sf
from typer.testing import CliRunner

from transcribe_music import (
    Instrument,
    TranscriptionConfig,
    transcribe_audio,
    transcribe_pdf,
    transcribe_youtube,
)
from transcribe_music.cli import app
from transcribe_music.storage.catalog import CatalogStore


def _write_synthetic_piano_wav(path: Path, sr: int = 44100, seconds: float = 2.0) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = np.zeros_like(t)
    for start_s, freqs in [(0.1, [261.63, 329.63, 392.0]), (1.0, [392.0, 493.88, 587.33])]:
        mask = t >= start_s
        dt = t[mask] - start_s
        env = np.exp(-4.0 * dt)
        chord = sum(0.25 * np.sin(2 * np.pi * f * dt) for f in freqs)
        wave[mask] += env * chord
    sf.write(path, np.stack([wave, wave], axis=1).astype(np.float32), sr)


def _stub_piano_amt(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_transkun(wav_path: Path, output_midi_path: Path, device: str = "auto"):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        for start, pitches in [(0.0, (48, 60, 64, 67)), (1.0, (55, 67, 71, 74))]:
            for pitch in pitches:
                inst.notes.append(pretty_midi.Note(85, pitch, start, start + 0.5))
        midi.instruments.append(inst)
        return midi

    monkeypatch.setattr("transcribe_music.pipeline.transcribe_piano_transkun", _fake_transkun)


def test_end_to_end_audio_piano_and_duckdb_cache(tmp_path: Path) -> None:
    wav_path = tmp_path / "piano_test.wav"
    _write_synthetic_piano_wav(wav_path)

    out_xml = tmp_path / "out_piano.musicxml"
    cfg = TranscriptionConfig(
        instrument=Instrument.PIANO,
        concerto_mode="solo",
        device="cpu",
        tempo_bpm=120.0,
        cache_dir=tmp_path / ".cache",
    )

    res1 = transcribe_audio(wav_path, Instrument.PIANO, output_path=out_xml, config=cfg)
    assert out_xml.exists()
    assert res1.cached is False
    assert res1.measure_count >= 1
    assert res1.fingering_count == res1.note_count

    root = ET.parse(out_xml).getroot()
    assert any(el.text == "brace" for el in root.findall(".//group-symbol"))

    verovio = pytest.importorskip("verovio")
    tk = verovio.toolkit()
    assert tk.loadData(out_xml.read_text(encoding="utf-8"))
    assert "<svg" in tk.renderToSVG(1)

    out_xml_2 = tmp_path / "out_piano_cached.musicxml"
    res2 = transcribe_audio(wav_path, Instrument.PIANO, output_path=out_xml_2, config=cfg)
    assert res2.cached is True
    assert res2.job_hash == res1.job_hash
    assert out_xml_2.exists()


@pytest.mark.parametrize(
    ("start", "end", "expected_range", "expected_name"),
    [
        (None, None, None, "yt_dQw4w9WgXcQ_piano.musicxml"),
        ("", "   ", None, "yt_dQw4w9WgXcQ_piano.musicxml"),
        ("0:10", "0:12", (10.0, 12.0), "yt_dQw4w9WgXcQ_10_12_piano.musicxml"),
        ("0:10", None, (10.0, None), "yt_dQw4w9WgXcQ_10_end_piano.musicxml"),
        (None, "0:02", (0.0, 2.0), "yt_dQw4w9WgXcQ_0_2_piano.musicxml"),
    ],
)
def test_end_to_end_youtube_ranges_mocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    start: str | None,
    end: str | None,
    expected_range: tuple[float, float | None] | None,
    expected_name: str,
) -> None:
    _stub_piano_amt(monkeypatch)
    seen_ranges = []

    def _fake_download_slice(url, time_range, output_wav, sample_rate=44100):
        seen_ranges.append(time_range)
        seconds = time_range.duration_sec if (time_range and time_range.duration_sec) else 2.0
        _write_synthetic_piano_wav(output_wav, sr=sample_rate, seconds=seconds)
        return output_wav

    monkeypatch.setattr("transcribe_music.pipeline.download_youtube_slice", _fake_download_slice)

    cfg = TranscriptionConfig(
        instrument=Instrument.PIANO,
        concerto_mode="solo",
        device="cpu",
        tempo_bpm=120.0,
        cache_dir=tmp_path / ".cache",
    )
    res = transcribe_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        Instrument.PIANO,
        start=start,
        end=end,
        config=cfg,
    )
    assert res.musicxml_path.exists()
    assert res.musicxml_path.name == expected_name
    assert res.measure_count >= 1

    tr = seen_ranges[-1]
    actual_range = None if tr is None else (tr.start_sec, tr.end_sec)
    assert actual_range == expected_range

    jobs = CatalogStore(cfg.cache_dir).list_recent_jobs()
    assert len(jobs) == 1
    assert (jobs[0]["start_sec"], jobs[0]["end_sec"]) == (
        (None, None) if expected_range is None else expected_range
    )


def test_end_to_end_pdf_guitar_omr_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "guitar_piece.pdf"
    doc = pymupdf.open()
    for label in ("Page 1", "Page 2"):
        doc.new_page().insert_text((72, 72), label)
    doc.save(str(pdf_path))
    doc.close()

    def _fake_homr(image_path: Path) -> Path:
        s = music21.stream.Score()
        p = music21.stream.Part()
        for pitch in ["C4", "E4", "G4", "C5"]:
            p.append(music21.note.Note(pitch, quarterLength=1.0))
        p.makeMeasures(inPlace=True)
        s.insert(0, p)
        xml_out = image_path.with_suffix(".musicxml")
        s.write("musicxml", fp=str(xml_out))
        return xml_out

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._run_homr_on_image", _fake_homr)

    out_xml = tmp_path / "guitar_pdf.musicxml"
    cfg = TranscriptionConfig(
        instrument=Instrument.GUITAR,
        cache_dir=tmp_path / ".cache",
        pdf_dpi=150,
    )
    res = transcribe_pdf(pdf_path, Instrument.GUITAR, output_path=out_xml, config=cfg)
    assert out_xml.exists()
    assert res.measure_count == 2
    assert res.note_count == 8
    assert res.chord_count >= 1
    assert res.fingering_count == 8

    root = ET.parse(out_xml).getroot()
    assert len(root.findall(".//technical/string")) == 8
    assert len(root.findall(".//technical/fret")) == 8


def test_cli_youtube_audio_and_cache_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_piano_amt(monkeypatch)
    monkeypatch.setattr(
        "transcribe_music.pipeline.download_youtube_slice",
        lambda url, time_range, output_wav, sample_rate=44100: (
            _write_synthetic_piano_wav(output_wav, sr=sample_rate, seconds=2.0) or output_wav
        ),
    )

    wav_path = tmp_path / "cli_piano.wav"
    _write_synthetic_piano_wav(wav_path)
    cache_dir = tmp_path / ".cli_cache"
    runner = CliRunner()

    # --no-cache leaves no DuckDB file behind
    nocache_out = tmp_path / "nocache.musicxml"
    nc_res = runner.invoke(
        app,
        [
            "audio",
            str(wav_path),
            "--instrument",
            "piano",
            "--mode",
            "solo",
            "--device",
            "cpu",
            "--tempo",
            "120",
            "--no-cache",
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(nocache_out),
        ],
    )
    assert nc_res.exit_code == 0, nc_res.output
    assert nocache_out.exists()
    assert not (cache_dir / "catalog.duckdb").exists()

    # YouTube CLI defaults to whole video when --start/--end are omitted
    yt_out = tmp_path / "cli_yt.musicxml"
    yt_res = runner.invoke(
        app,
        [
            "youtube",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--instrument",
            "piano",
            "--mode",
            "solo",
            "--device",
            "cpu",
            "--tempo",
            "120",
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(yt_out),
        ],
    )
    assert yt_res.exit_code == 0, yt_res.output
    assert yt_out.exists()

    list_res = runner.invoke(app, ["cache", "list", "--cache-dir", str(cache_dir)])
    assert list_res.exit_code == 0, list_res.output
    assert "piano" in list_res.output

    clear_res = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir)])
    assert clear_res.exit_code == 0, clear_res.output
    assert "Cleared 1 job records" in clear_res.output
