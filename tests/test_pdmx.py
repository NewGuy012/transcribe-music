"""Tests for PDMX DuckDB indexing, offline search, tar.gz extraction, and CLI workflow."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import music21
import pytest
from typer.testing import CliRunner

from transcribe_music import (
    Instrument,
    convert_music,
    download_score_result,
    get_pdmx_status,
    search_digital_scores,
    sync_pdmx,
)
from transcribe_music.cli import app


def _make_cello_mxl_bytes(tmp_path: Path) -> bytes:
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title="Haydn Cello Concerto in D Major")
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
    out = tmp_path / "fixture.mxl"
    score.write("mxl", fp=str(out))
    return out.read_bytes()


def test_pdmx_duckdb_sync_search_extract_and_cli(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    pdmx_dir = tmp_path / "pdmx_store"
    pdmx_dir.mkdir(parents=True)

    mxl_rel = "./mxl/h/ha/QmHaydnCelloD.mxl"
    conflict_rel = "./mxl/c/co/QmConflict.mxl"
    dvorak_rel = "./mxl/d/dv/QmDvorakCello.mxl"

    csv_path = pdmx_dir / "PDMX.csv"
    csv_path.write_text(
        "\n".join(
            [
                'mxl,mid,pdf,title,song_name,composer_name,artist_name,genres,tags,n_tracks,complexity,rating,n_ratings,is_best_unique_arrangement,"subset:no_license_conflict"',
                f'{mxl_rel},NA,NA,"Cello Concerto No. 2 in D major (Hob. VIIb:2)","Cello Concerto in D","Joseph Haydn",NA,classical,cello,2,3.0,4.85,12,True,True',
                f'{conflict_rel},NA,NA,"Cello Concerto in D major (Conflict)","Cello Concerto",NA,NA,classical,NA,NA,NA,NA,NA,False,False',
                f'{dvorak_rel},NA,NA,"Cello Concerto in B minor","Cello Concerto in B minor","Antonín Dvořák",NA,classical,cello,2,4.0,4.7,9,True,True',
            ]
        ),
        encoding="utf-8",
    )

    # 1. Sync metadata-only first and verify CLI status + helpful missing-archive error
    status_meta = sync_pdmx(cache_dir=cache_dir, pdmx_dir=pdmx_dir, download_csv=False)
    assert status_meta.csv_exists is True
    assert status_meta.db_exists is True
    assert status_meta.score_count == 2  # license_conflict row excluded by default
    assert status_meta.mxl_available is False

    # Accent-insensitive query works offline against DuckDB
    dvorak_hits = search_digital_scores(
        "Dvorak Cello Concerto",
        limit=5,
        include_remote=False,
        include_local=False,
        cache_dir=cache_dir,
    )
    assert len(dvorak_hits) == 1
    assert dvorak_hits[0].source == "PDMX"
    assert dvorak_hits[0].composer == "Antonín Dvořák"
    assert "needs `pdmx sync --with-mxl`" in (dvorak_hits[0].edition_note or "")

    with pytest.raises(FileNotFoundError, match="pdmx sync --with-mxl"):
        download_score_result(dvorak_hits[0], download_dir=tmp_path / "dl")

    # 2. Add mxl.tar.gz archive and extract + convert Haydn Cello Concerto end-to-end
    mxl_bytes = _make_cello_mxl_bytes(tmp_path)
    tar_path = pdmx_dir / "mxl.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name=mxl_rel)
        info.size = len(mxl_bytes)
        tf.addfile(info, io.BytesIO(mxl_bytes))

    assert get_pdmx_status(cache_dir).mxl_available is True

    haydn_hits = search_digital_scores(
        "Haydn Cello Concerto",
        limit=5,
        include_remote=False,
        include_local=False,
        cache_dir=cache_dir,
    )
    assert len(haydn_hits) == 1
    assert haydn_hits[0].format_ext == ".mxl"
    assert haydn_hits[0].pdmx_member == "mxl/h/ha/QmHaydnCelloD.mxl"

    saved_mxl = download_score_result(haydn_hits[0], download_dir=tmp_path / "dl")
    assert saved_mxl.exists()
    assert saved_mxl.read_bytes() == mxl_bytes

    converted = convert_music(saved_mxl, Instrument.CELLO, output_path=tmp_path / "haydn_out.musicxml")
    assert converted.note_count == 4
    assert converted.fingering_count == 4

    # 3. CLI pdmx status & pdmx sync --no-download
    runner = CliRunner()
    st_res = runner.invoke(app, ["pdmx", "status", "--cache-dir", str(cache_dir)])
    assert st_res.exit_code == 0, st_res.stdout
    assert "Indexed Scores" in st_res.stdout
    assert "Available" in st_res.stdout

    sync_res = runner.invoke(
        app,
        [
            "pdmx",
            "sync",
            "--pdmx-dir",
            str(pdmx_dir),
            "--no-download",
            "--include-license-conflicts",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert sync_res.exit_code == 0, sync_res.stdout
    assert get_pdmx_status(cache_dir).score_count == 3
