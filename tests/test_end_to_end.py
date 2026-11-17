"""End-to-end integration tests across pipeline, CLI, Verovio, and DuckDB cache."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import numpy as np
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

    # Verify Verovio can engrave our generated MusicXML into SVG when ui extra is installed
    verovio = pytest.importorskip("verovio")
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


def test_end_to_end_youtube_defaults_to_whole_video_and_partial_ranges(
    monkeypatch, tmp_path: Path
) -> None:
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

    # 1. Neither start nor end provided -> defaults to whole video (time_range=None)
    res_full = transcribe_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        Instrument.PIANO,
        config=cfg,
    )
    assert seen_ranges[-1] is None
    assert res_full.musicxml_path.exists()
    assert res_full.musicxml_path.name == "yt_dQw4w9WgXcQ_piano.musicxml"
    assert res_full.measure_count >= 1

    # 2. Only start provided -> start_sec to end of video
    res_start_only = transcribe_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        Instrument.PIANO,
        start="0:10",
        config=cfg,
    )
    assert seen_ranges[-1] is not None
    assert seen_ranges[-1].start_sec == 10.0
    assert seen_ranges[-1].end_sec is None
    assert res_start_only.musicxml_path.name == "yt_dQw4w9WgXcQ_10_end_piano.musicxml"

    # 3. Only end provided -> 0.0 to end_sec
    res_end_only = transcribe_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        Instrument.PIANO,
        end="0:02",
        config=cfg,
    )
    assert seen_ranges[-1] is not None
    assert seen_ranges[-1].start_sec == 0.0
    assert seen_ranges[-1].end_sec == 2.0
    assert res_end_only.musicxml_path.name == "yt_dQw4w9WgXcQ_0_2_piano.musicxml"

    assert len({res_full.job_hash, res_start_only.job_hash, res_end_only.job_hash}) == 3

    store = CatalogStore(cfg.cache_dir)
    jobs_by_hash = {j["job_hash"]: j for j in store.list_recent_jobs()}
    assert jobs_by_hash[res_full.job_hash]["start_sec"] is None
    assert jobs_by_hash[res_full.job_hash]["end_sec"] is None
    assert jobs_by_hash[res_start_only.job_hash]["start_sec"] == 10.0
    assert jobs_by_hash[res_start_only.job_hash]["end_sec"] is None
    assert jobs_by_hash[res_end_only.job_hash]["start_sec"] == 0.0
    assert jobs_by_hash[res_end_only.job_hash]["end_sec"] == 2.0

    # 4. CLI youtube command without --start / --end
    cli_out = tmp_path / "cli_yt_full.musicxml"
    runner = CliRunner()
    cli_res = runner.invoke(
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
            str(cfg.cache_dir),
            "-o",
            str(cli_out),
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert cli_out.exists()
    assert "Cache Hit" in cli_res.output
    assert "Yes" in cli_res.output

    # 5. Blank whitespace strings behave as omitted (hit whole-video cache)
    res_blank = transcribe_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        Instrument.PIANO,
        start="",
        end="   ",
        config=cfg,
    )
    assert res_blank.cached is True
    assert res_blank.job_hash == res_full.job_hash


def test_end_to_end_pdf_guitar_omr_mocked(monkeypatch, tmp_path: Path) -> None:
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
