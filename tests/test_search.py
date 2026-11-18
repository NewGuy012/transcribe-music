"""Tests for digital score search, MusicXML ranking preference, download, and CLI workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import music21
import pytest
from typer.testing import CliRunner

from transcribe_music import download_score_result, search_digital_scores
from transcribe_music.cli import app
from transcribe_music.ingest import search as search_mod


def _make_sample_mxl_bytes(tmp_path: Path, title: str = "Moonlight Study") -> bytes:
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title)
    part = music21.stream.Part(id="P1")
    m = music21.stream.Measure(number=1)
    m.insert(0, music21.meter.TimeSignature("4/4"))
    for p in ("C4", "E4", "G4", "C5"):
        m.append(music21.note.Note(p, quarterLength=1.0))
    part.append(m)
    score.insert(0, part)
    mxl_path = tmp_path / "sample_fixture.mxl"
    score.write("mxl", fp=str(mxl_path))
    return mxl_path.read_bytes()


def _stub_remote_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    orchestra: Callable[[], list[dict[str, Any]]] = list,
    lieder: Callable[[], list[dict[str, Any]]] = list,
    quartets: Callable[[], list[dict[str, Any]]] = list,
    when_in_rome: Callable[[], list[dict[str, Any]]] = list,
) -> None:
    monkeypatch.setattr(search_mod, "_fetch_openscore_orchestra_index", orchestra)
    monkeypatch.setattr(search_mod, "_fetch_openscore_lieder_index", lieder)
    monkeypatch.setattr(search_mod, "_fetch_openscore_sq_index", quartets)
    monkeypatch.setattr(search_mod, "_fetch_when_in_rome_index", when_in_rome)


def test_search_prefers_musicxml_over_other_formats_and_caps_top_5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_remote_sources(
        monkeypatch,
        when_in_rome=lambda: [
            {
                "title": "Sonate No 14 Moonlight 1st Movement",
                "composer": "Beethoven",
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": "Sonate_No._14_Moonlight_1st_Movement.mxl",
                "download_url": "https://example.com/moonlight_1.mxl",
            },
            {
                "title": "Moonlight Sonata 3rd Movement",
                "composer": "Beethoven",
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": "moonlight_sonata_3rd_movement.mxl",
                "download_url": "https://example.com/moonlight_3.mxl",
            },
        ],
        orchestra=lambda: [
            {
                "title": "Moonlight Sonata",
                "composer": "Ludwig van Beethoven",
                "format_ext": ".krn",
                "source": "OpenScore Orchestra",
                "filename": "moonlight.krn",
                "download_url": "https://example.com/moonlight.krn",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".mid",
                "source": "OpenScore Orchestra",
                "filename": "moonlight.mid",
                "download_url": "https://example.com/moonlight.mid",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".gp5",
                "source": "OpenScore Orchestra",
                "filename": "moonlight.gp5",
                "download_url": "https://example.com/moonlight.gp5",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".mscz",
                "source": "OpenScore Orchestra",
                "filename": "moonlight.mscz",
                "download_url": "https://example.com/moonlight.mscz",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".musicxml",
                "source": "OpenScore Orchestra",
                "filename": "moonlight.musicxml",
                "download_url": "https://example.com/moonlight.musicxml",
            },
        ],
    )

    results = search_digital_scores(
        "Moonlight Sonata",
        limit=5,
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert len(results) == 5
    # Top 3 must all be native MusicXML (.musicxml / .mxl) before .krn / .mscz / .gp5 / .mid
    assert all(r.format_ext in {".musicxml", ".mxl"} for r in results[:3])

    mxl_only = search_digital_scores(
        "Moonlight Sonata",
        limit=5,
        musicxml_only=True,
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert len(mxl_only) == 3
    assert all(r.format_ext in {".musicxml", ".mxl", ".xml"} for r in mxl_only)


def test_search_diacritics_and_remote_resilience(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom() -> list[dict[str, object]]:
        raise RuntimeError("network down")

    _stub_remote_sources(
        monkeypatch,
        orchestra=_boom,
        lieder=lambda: [
            {
                "title": "Der Erlkönig",
                "composer": "Emilie Mayer",
                "format_ext": ".mxl",
                "source": "OpenScore Lieder",
                "filename": "Der_Erlkonig.mxl",
                "download_url": "https://example.com/erlkonig.mxl",
            }
        ],
    )

    results = search_digital_scores(
        "Erlkonig",
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert len(results) == 1
    assert results[0].title == "Der Erlkönig"
    assert results[0].format_ext == ".mxl"


def test_offline_music21_corpus_search_and_download(tmp_path: Path) -> None:
    results = search_digital_scores(
        "bwv1.6",
        limit=5,
        musicxml_only=True,
        include_remote=False,
        cache_dir=tmp_path / "cache",
    )
    assert results
    assert results[0].source == "music21 Corpus"
    assert results[0].format_ext in {".mxl", ".xml", ".musicxml"}

    saved = download_score_result(results[0], download_dir=tmp_path / "downloads")
    assert saved.exists()
    assert saved.stat().st_size > 0
    assert saved.suffix.lower() == results[0].format_ext


def test_search_supports_all_digital_format_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample_records = [
        {
            "title": "Study Piece",
            "composer": "Anon",
            "format_ext": ext,
            "source": "Archive.org",
            "filename": f"study{ext}",
            "download_url": f"https://example.com/study{ext}",
        }
        for ext in (".gp5", ".mscz", ".mei", ".abc", ".krn", ".mid", ".mxl")
    ]
    _stub_remote_sources(monkeypatch, orchestra=lambda: sample_records)

    gp_only = search_digital_scores(
        "Study Piece",
        formats=["gp5"],
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert [r.format_ext for r in gp_only] == [".gp5"]

    mscz_or_mei = search_digital_scores(
        "Study Piece",
        formats=["mscz", "mei"],
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert {r.format_ext for r in mscz_or_mei} == {".mscz", ".mei"}


def test_cli_search_interactive_select_cancel_and_convert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mxl_bytes = _make_sample_mxl_bytes(tmp_path)

    _stub_remote_sources(
        monkeypatch,
        when_in_rome=lambda: [
            {
                "title": "Moonlight Sonata 1st Movement",
                "composer": "Ludwig van Beethoven",
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": "../../evil_path/moonlight.mxl",
                "download_url": "https://example.com/moonlight.mxl",
            },
            {
                "title": "Moonlight Sonata 3rd Movement",
                "composer": "Ludwig van Beethoven",
                "format_ext": ".mid",
                "source": "Archive.org",
                "filename": "moonlight_3.mid",
                "download_url": "https://example.com/moonlight_3.mid",
            },
        ],
    )

    def _fake_get(url: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            content=mxl_bytes if url.endswith(".mxl") else b"MThd",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(search_mod.requests, "get", _fake_get)

    runner = CliRunner()
    dl_dir = tmp_path / "dl"
    cache_dir = tmp_path / "cache"

    # 1. Interactive cancel does not download
    cancel_res = runner.invoke(
        app,
        ["search", "Moonlight", "--dir", str(dl_dir), "--cache-dir", str(cache_dir)],
        input="\n",
    )
    assert cancel_res.exit_code == 0, cancel_res.stdout
    assert "Download cancelled." in cancel_res.stdout
    assert not dl_dir.exists() or not any(dl_dir.iterdir())

    # 2. Interactive selection "1" downloads safely inside dl_dir (no path traversal)
    pick_res = runner.invoke(
        app,
        ["search", "Moonlight", "--dir", str(dl_dir), "--cache-dir", str(cache_dir)],
        input="1\n",
    )
    assert pick_res.exit_code == 0, pick_res.stdout
    saved = dl_dir / "moonlight.mxl"
    assert saved.exists()
    assert saved.read_bytes() == mxl_bytes

    # 3. Non-interactive --select 1 --instrument piano downloads AND produces enriched MusicXML
    conv_dir = tmp_path / "dl_conv"
    conv_res = runner.invoke(
        app,
        [
            "search",
            "Moonlight",
            "--select",
            "1",
            "--dir",
            str(conv_dir),
            "--instrument",
            "piano",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert conv_res.exit_code == 0, conv_res.stdout
    assert "MusicXML Ready" in conv_res.stdout
    assert (conv_dir / "moonlight.mxl").exists()


def test_github_tree_catalog_parsers_orchestra_and_when_in_rome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_mod,
        "_fetch_github_tree",
        lambda url: (
            [
                {"path": "data/Haydn,_Franz/Symphony_No.104/01/Haydn_104_01.mxl"},
                {"path": "data/Haydn,_Franz/Symphony_No.104/01/Haydn_104_01_melody.mxl"},
            ]
            if "Hauptstimme" in url
            else [
                {"path": "Corpus/Piano_Sonatas/Mozart,_Wolfgang_Amadeus/K331/01/score.mxl"},
                {"path": "Corpus/Textbooks/Reger,_Max/ex01/score.mxl"},
                {"path": "Corpus/OpenScore-LiederCorpus/Schubert,_Franz/song/score.mxl"},
            ]
        ),
    )
    orch = search_mod._fetch_openscore_orchestra_index()
    assert len(orch) == 1
    assert orch[0]["composer"] == "Franz Haydn"
    assert orch[0]["title"] == "Symphony No.104 (01)"

    wir = search_mod._fetch_when_in_rome_index()
    assert len(wir) == 1
    assert wir[0]["composer"] == "Wolfgang Amadeus Mozart"
    assert wir[0]["title"] == "K331 (01)"


def test_download_enforces_https_and_size_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    insecure = search_mod.ScoreSearchResult(
        title="Plain HTTP",
        format_ext=".mxl",
        source="PDMX",
        filename="http_score.mxl",
        download_url="http://example.com/score.mxl",
    )
    with pytest.raises(ValueError, match="Only HTTPS"):
        download_score_result(insecure, download_dir=tmp_path)

    valid = insecure.model_copy(update={"download_url": "https://example.com/score.mxl"})
    monkeypatch.setattr(
        search_mod.requests,
        "get",
        lambda *_a, **_kw: SimpleNamespace(
            url="http://downgraded.example.com/score.mxl",
            headers={},
            content=b"PK",
            raise_for_status=lambda: None,
        ),
    )
    with pytest.raises(ValueError, match="Only HTTPS"):
        download_score_result(valid, download_dir=tmp_path)

    monkeypatch.setattr(
        search_mod.requests,
        "get",
        lambda *_a, **_kw: SimpleNamespace(
            url="https://example.com/score.mxl",
            headers={"content-length": str(search_mod._MAX_DOWNLOAD_BYTES + 1)},
            content=b"PK",
            raise_for_status=lambda: None,
        ),
    )
    with pytest.raises(ValueError, match="safety limit"):
        download_score_result(valid, download_dir=tmp_path)
