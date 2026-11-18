"""Tests for digital score search, ranking, download safety, and CLI search workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_simple_cello_score, write_mxl_bytes
from typer.testing import CliRunner

from transcribe_music import download_score_result, search_digital_scores
from transcribe_music.cli import app
from transcribe_music.ingest import search as search_mod


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


def test_search_ranking_diacritics_resilience_and_format_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom() -> list[dict[str, Any]]:
        raise RuntimeError("network down")

    _stub_remote_sources(
        monkeypatch,
        lieder=_boom,
        when_in_rome=lambda: [
            {
                "title": "Der Erlkönig",
                "composer": "Emilie Mayer",
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": "Der_Erlkonig.mxl",
                "download_url": "https://example.com/erlkonig.mxl",
            },
            {
                "title": "Moonlight Sonata 1st Movement",
                "composer": "Beethoven",
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": "moonlight_1.mxl",
                "download_url": "https://example.com/moonlight_1.mxl",
            },
            {
                "title": "Moonlight Sonata 3rd Movement",
                "composer": "Beethoven",
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": "moonlight_3.mxl",
                "download_url": "https://example.com/moonlight_3.mxl",
            },
        ],
        orchestra=lambda: [
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ext,
                "source": "OpenScore Orchestra",
                "filename": f"moonlight{ext}",
                "download_url": f"https://example.com/moonlight{ext}",
            }
            for ext in (".krn", ".mid", ".gp5", ".mscz", ".musicxml")
        ],
    )

    results = search_digital_scores(
        "Moonlight Sonata", limit=5, include_local=False, cache_dir=tmp_path / "cache"
    )
    assert len(results) == 5
    assert all(r.format_ext in {".musicxml", ".mxl"} for r in results[:3])

    mxl_only = search_digital_scores(
        "Moonlight Sonata",
        limit=5,
        musicxml_only=True,
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert [r.format_ext for r in mxl_only] == [".musicxml", ".mxl", ".mxl"]

    krn_only = search_digital_scores(
        "Moonlight Sonata",
        formats=["krn", "mid"],
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert {r.format_ext for r in krn_only} == {".krn", ".mid"}

    diacritics = search_digital_scores(
        "Erlkonig", include_local=False, cache_dir=tmp_path / "cache"
    )
    assert len(diacritics) == 1 and diacritics[0].title == "Der Erlkönig"


def test_offline_music21_corpus_and_github_tree_parsers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = search_digital_scores(
        "bwv1.6", limit=5, musicxml_only=True, include_remote=False, cache_dir=tmp_path / "cache"
    )
    assert results and results[0].source == "music21 Corpus"
    saved = download_score_result(results[0], download_dir=tmp_path / "downloads")
    assert saved.exists() and saved.stat().st_size > 0

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
    assert len(orch) == 1 and orch[0]["composer"] == "Franz Haydn"
    wir = search_mod._fetch_when_in_rome_index()
    assert len(wir) == 1 and wir[0]["composer"] == "Wolfgang Amadeus Mozart"


def test_cli_search_and_download_security_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mxl_bytes = write_mxl_bytes(make_simple_cello_score("Moonlight Study"), tmp_path)
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
            }
        ],
    )
    monkeypatch.setattr(
        search_mod.requests,
        "get",
        lambda *_a, **_kw: SimpleNamespace(
            url="https://example.com/moonlight.mxl",
            headers={},
            content=mxl_bytes,
            raise_for_status=lambda: None,
        ),
    )

    runner = CliRunner()
    dl_dir = tmp_path / "dl"
    cache_dir = tmp_path / "cache"

    cancel_res = runner.invoke(
        app, ["search", "Moonlight", "--dir", str(dl_dir), "--cache-dir", str(cache_dir)], input="\n"
    )
    assert cancel_res.exit_code == 0 and "Download cancelled." in cancel_res.stdout

    conv_res = runner.invoke(
        app,
        [
            "search",
            "Moonlight",
            "--select",
            "1",
            "--dir",
            str(dl_dir),
            "--instrument",
            "piano",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert conv_res.exit_code == 0, conv_res.stdout
    assert "MusicXML Ready" in conv_res.stdout
    assert (dl_dir / "moonlight.mxl").read_bytes() == mxl_bytes

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
