"""Tests for PDMX DuckDB indexing, offline search, tar.gz extraction, and security bounds."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from conftest import make_simple_cello_score, write_mxl_bytes
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
from transcribe_music.ingest.pdmx import extract_pdmx_member


def test_pdmx_duckdb_sync_search_extract_and_security(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    pdmx_dir = tmp_path / "pdmx_store"
    pdmx_dir.mkdir(parents=True)

    mxl_rel = "./mxl/h/ha/QmHaydnCelloD.mxl"
    conflict_rel = "./mxl/c/co/QmConflict.mxl"
    dvorak_rel = "./mxl/d/dv/QmDvorakCello.mxl"

    (pdmx_dir / "PDMX.csv").write_text(
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

    status_meta = sync_pdmx(cache_dir=cache_dir, pdmx_dir=pdmx_dir, download_csv=False)
    assert status_meta.score_count == 2
    assert status_meta.mxl_available is False

    dvorak_hits = search_digital_scores(
        "Dvorak Cello Concerto",
        include_remote=False,
        include_local=False,
        cache_dir=cache_dir,
    )
    assert len(dvorak_hits) == 1
    assert dvorak_hits[0].composer == "Antonín Dvořák"
    with pytest.raises(FileNotFoundError, match="pdmx sync --with-mxl"):
        download_score_result(dvorak_hits[0], download_dir=tmp_path / "dl")

    mxl_bytes = write_mxl_bytes(make_simple_cello_score("Haydn Cello Concerto"), tmp_path)
    with tarfile.open(pdmx_dir / "mxl.tar.gz", "w:gz") as tf:
        info = tarfile.TarInfo(name=mxl_rel)
        info.size = len(mxl_bytes)
        tf.addfile(info, io.BytesIO(mxl_bytes))
        sym = tarfile.TarInfo(name="./mxl/evil_link.mxl")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "/etc/passwd"
        tf.addfile(sym)

    assert get_pdmx_status(cache_dir).mxl_available is True
    haydn_hits = search_digital_scores(
        "Haydn Cello Concerto", include_remote=False, include_local=False, cache_dir=cache_dir
    )
    saved_mxl = download_score_result(haydn_hits[0], download_dir=tmp_path / "dl")
    assert saved_mxl.read_bytes() == mxl_bytes

    converted = convert_music(saved_mxl, Instrument.CELLO, output_path=tmp_path / "haydn_out.musicxml")
    assert converted.note_count == 4
    assert converted.fingering_count == 4

    runner = CliRunner()
    assert runner.invoke(app, ["pdmx", "status", "--cache-dir", str(cache_dir)]).exit_code == 0
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
    assert sync_res.exit_code == 0
    assert get_pdmx_status(cache_dir).score_count == 3

    for unsafe in ("../../etc/passwd", "/etc/passwd", "mxl/../../etc/passwd"):
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            extract_pdmx_member(pdmx_dir, unsafe, tmp_path / "out.mxl")

    with pytest.raises(ValueError, match="not a regular file"):
        extract_pdmx_member(pdmx_dir, "mxl/evil_link.mxl", tmp_path / "out.mxl")


def test_pdmx_tracks_column_and_instrument_search_boost(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache_gtr"
    pdmx_dir = tmp_path / "pdmx_gtr"
    pdmx_dir.mkdir(parents=True)
    (pdmx_dir / "PDMX.csv").write_text(
        'mxl,mid,pdf,title,song_name,composer_name,artist_name,genres,tracks,tags,n_tracks,complexity,rating,n_ratings,is_best_unique_arrangement,"subset:no_license_conflict"\n'
        './mxl/g/gu/QmPiano.mxl,NA,NA,"Capricho","Capricho","Composer A",NA,classical,"Piano",piano,1,2.0,4.9,10,True,True\n'
        './mxl/g/gu/QmGuitar.mxl,NA,NA,"Capricho","Capricho","Composer B",NA,classical,"Classical Guitar",guitar,1,2.0,3.5,2,False,True\n',
        encoding="utf-8",
    )
    status = sync_pdmx(cache_dir=cache_dir, pdmx_dir=pdmx_dir, download_csv=False)
    assert status.score_count == 2

    hits = search_digital_scores(
        "Capricho",
        instrument=Instrument.GUITAR,
        include_remote=False,
        include_local=False,
        cache_dir=cache_dir,
    )
    assert len(hits) == 2
    assert hits[0].filename == "QmGuitar.mxl"
    assert "Classical Guitar" in (hits[0].edition_note or "")
