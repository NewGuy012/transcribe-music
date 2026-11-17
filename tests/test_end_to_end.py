"""End-to-end integration tests across pipeline, CLI, Verovio, and DuckDB cache."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import numpy as np
import soundfile as sf
import verovio
from reportlab.pdfgen import canvas
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
    # Two distinct percussive piano-like attacks: C4+E4+G4 at 0.1s, G4+B4+D5 at 1.0s
    wave = np.zeros_like(t)
    for start_s, freqs in [(0.1, [261.63, 329.63, 392.0]), (1.0, [392.0, 493.88, 587.33])]:
        mask = t >= start_s
        dt = t[mask] - start_s
        env = np.exp(-4.0 * dt)
        chord = sum(0.25 * np.sin(2 * np.pi * f * dt) for f in freqs)
        wave[mask] += env * chord
    sf.write(path, np.stack([wave, wave], axis=1).astype(np.float32), sr)


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

    # Verify Verovio can engrave our generated MusicXML into SVG
    tk = verovio.toolkit()
    assert tk.loadData(out_xml.read_text(encoding="utf-8"))
    svg = tk.renderToSVG(1)
    assert "<svg" in svg

    # Second run hits DuckDB cache immediately
    out_xml_2 = tmp_path / "out_piano_cached.musicxml"
    res2 = transcribe_audio(wav_path, Instrument.PIANO, output_path=out_xml_2, config=cfg)
    assert res2.cached is True
    assert res2.job_hash == res1.job_hash
    assert out_xml_2.exists()


def test_end_to_end_youtube_slice_mocked(monkeypatch, tmp_path: Path) -> None:
    def _fake_download_slice(url, time_range, output_wav, sample_rate=44100):
        _write_synthetic_piano_wav(output_wav, sr=sample_rate, seconds=time_range.duration_sec)
        return output_wav

    monkeypatch.setattr("transcribe_music.pipeline.download_youtube_slice", _fake_download_slice)

    out_xml = tmp_path / "yt_piano.musicxml"
    cfg = TranscriptionConfig(
        instrument=Instrument.PIANO,
        concerto_mode="solo",
        device="cpu",
        tempo_bpm=120.0,
        cache_dir=tmp_path / ".cache",
    )
    res = transcribe_youtube(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        start="0:10",
        end="0:12",
        instrument=Instrument.PIANO,
        output_path=out_xml,
        config=cfg,
    )
    assert out_xml.exists()
    assert res.measure_count >= 1

    store = CatalogStore(cfg.cache_dir)
    jobs = store.list_recent_jobs()
    assert len(jobs) == 1
    assert jobs[0]["start_sec"] == 10.0
    assert jobs[0]["end_sec"] == 12.0


def test_end_to_end_pdf_guitar_omr_mocked(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "guitar_piece.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 700, "Page 1")
    c.showPage()
    c.drawString(100, 700, "Page 2")
    c.showPage()
    c.save()

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


def test_cli_audio_and_cache_commands(tmp_path: Path) -> None:
    wav_path = tmp_path / "cli_piano.wav"
    _write_synthetic_piano_wav(wav_path)
    out_xml = tmp_path / "cli_out.musicxml"
    cache_dir = tmp_path / ".cli_cache"

    runner = CliRunner()
    res = runner.invoke(
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
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(out_xml),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_xml.exists()

    list_res = runner.invoke(app, ["cache", "list", "--cache-dir", str(cache_dir)])
    assert list_res.exit_code == 0, list_res.output
    assert "piano" in list_res.output

    clear_res = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir)])
    assert clear_res.exit_code == 0, clear_res.output
    assert "Cleared 1 job records" in clear_res.output
