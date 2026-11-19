"""Tests for digital score search, ranking, download safety, and CLI search workflow."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_simple_cello_score, write_mxl_bytes

from transcribe_music import (
    Instrument,
    download_score_result,
    get_gp_catalog_status,
    search_digital_scores,
    sync_gp_catalog,
)
from transcribe_music.cli import main
from transcribe_music.ingest import _io as io_mod
from transcribe_music.ingest import search as search_mod


def _fake_opener(payload: bytes, url: str, headers: dict[str, str] | None = None) -> Callable[..., Any]:
    class _FakeResp(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(payload)
            self.headers = headers or {}

        def geturl(self) -> str:
            return url

    return lambda *_handlers: SimpleNamespace(open=lambda *_a, **_kw: _FakeResp())


def _stub_remote_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    orchestra: Callable[[], list[dict[str, Any]]] = list,
    lieder: Callable[[], list[dict[str, Any]]] = list,
    quartets: Callable[[], list[dict[str, Any]]] = list,
    when_in_rome: Callable[[], list[dict[str, Any]]] = list,
    gp_classical: Callable[[], list[dict[str, Any]]] = list,
) -> None:
    monkeypatch.setattr(search_mod, "_fetch_openscore_orchestra_index", orchestra)
    monkeypatch.setattr(search_mod, "_fetch_openscore_lieder_index", lieder)
    monkeypatch.setattr(search_mod, "_fetch_openscore_sq_index", quartets)
    monkeypatch.setattr(search_mod, "_fetch_when_in_rome_index", when_in_rome)
    monkeypatch.setattr(search_mod, "_fetch_gp_classical_guitar_index", gp_classical)


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
            else (
                [{"path": "Brouwer/Estudios_Sencillos/Estudio_01.gp5"}]
                if "gp-classical-guitar" in url
                else [
                    {"path": "Corpus/Piano_Sonatas/Mozart,_Wolfgang_Amadeus/K331/01/score.mxl"},
                    {"path": "Corpus/Textbooks/Reger,_Max/ex01/score.mxl"},
                    {"path": "Corpus/OpenScore-LiederCorpus/Schubert,_Franz/song/score.mxl"},
                ]
            )
        ),
    )
    orch = search_mod._fetch_openscore_orchestra_index()
    assert len(orch) == 1 and orch[0]["composer"] == "Franz Haydn"
    wir = search_mod._fetch_when_in_rome_index()
    assert len(wir) == 1 and wir[0]["composer"] == "Wolfgang Amadeus Mozart"
    gp_cls = search_mod._fetch_gp_classical_guitar_index()
    assert len(gp_cls) == 1
    assert gp_cls[0]["composer"] == "Brouwer"
    assert gp_cls[0]["format_ext"] == ".gp5"
    assert gp_cls[0]["source"] == "GP Classical Guitar"


def test_cli_search_and_download_security_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
        urllib.request,
        "build_opener",
        _fake_opener(mxl_bytes, "https://example.com/moonlight.mxl"),
    )

    dl_dir = tmp_path / "dl"
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert main(["search", "Moonlight", "--dir", str(dl_dir), "--cache-dir", str(cache_dir)]) == 0
    assert "Download cancelled." in capsys.readouterr().out

    assert (
        main(
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
            ]
        )
        == 0
    )
    assert "MusicXML Ready" in capsys.readouterr().out
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

    valid = replace(insecure, download_url="https://example.com/score.mxl")
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        _fake_opener(b"PK", "http://downgraded.example.com/score.mxl"),
    )
    with pytest.raises(ValueError, match="Only HTTPS"):
        download_score_result(valid, download_dir=tmp_path)

    # Pre-hop redirect guard blocks http:// downgrade and strips Authorization across hosts
    handler = io_mod._SafeHttpsRedirectHandler()
    auth_req = urllib.request.Request(
        "https://api.example.com/data", headers={"Authorization": "Bearer secret"}
    )
    with pytest.raises(ValueError, match="Only HTTPS"):
        handler.redirect_request(auth_req, None, 302, "Found", {}, "http://evil.example.com/x")
    redir_req = handler.redirect_request(
        auth_req, None, 302, "Found", {}, "https://cdn.other.com/x"
    )
    assert redir_req is not None
    assert "Authorization" not in redir_req.headers

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        _fake_opener(
            b"PK",
            "https://example.com/score.mxl",
            headers={"Content-Length": str(search_mod._MAX_DOWNLOAD_BYTES + 1)},
        ),
    )
    with pytest.raises(ValueError, match="safety limit"):
        download_score_result(valid, download_dir=tmp_path)


def test_guitar_instrument_search_and_dadagp_duckdb_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import tarfile
    import zipfile

    from transcribe_music.ingest import gp_catalog as gp_mod

    _stub_remote_sources(
        monkeypatch,
        quartets=lambda: [
            {
                "title": "Asturias Quartet Transcription",
                "composer": "Isaac Albeniz",
                "format_ext": ".mxl",
                "source": "OpenScore StringQuartets",
                "filename": "asturias_q.mxl",
                "download_url": "https://example.com/asturias_q.mxl",
            }
        ],
        gp_classical=lambda: [
            {
                "title": "Asturias (Leyenda)",
                "composer": "Isaac Albeniz",
                "format_ext": ".gp5",
                "source": "GP Classical Guitar",
                "filename": "asturias.gp5",
                "edition_note": "Classical Guitar TAB",
                "download_url": "https://example.com/asturias.gp5",
            }
        ],
    )
    gtr_hits = search_digital_scores(
        "Asturias Albeniz",
        instrument=Instrument.GUITAR,
        include_local=False,
        cache_dir=tmp_path / "search_cache",
    )
    assert gtr_hits[0].format_ext == ".gp5"
    assert gtr_hits[0].source == "GP Classical Guitar"

    # DadaGP / local GP catalog indexing across direct file, .tar.gz, and .zip
    gp_dir = tmp_path / "gp_library"
    (gp_dir / "Rock" / "Metallica").mkdir(parents=True)
    (gp_dir / "Rock" / "Metallica" / "Nothing_Else_Matters.gp5").write_bytes(b"GP5RAW")

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
        payload = b"TAR_GP4_BYTES"
        ti = tarfile.TarInfo("DadaGP-v1.1/A/Albeniz/Capricho_Arabe.gp4")
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
    (gp_dir / "bundle.tar.gz").write_bytes(tar_buf.getvalue())

    with zipfile.ZipFile(gp_dir / "pack.zip", "w") as zf:
        zf.writestr("Bach/Bourree_in_E_minor.tab", "e|---0---|\n")

    cache_dir = tmp_path / "gp_cache"
    status = sync_gp_catalog(cache_dir=cache_dir, gp_dir=gp_dir)
    assert status.db_exists is True
    assert status.score_count == 3
    assert get_gp_catalog_status(cache_dir).score_count == 3

    for q, expected_ext, expected_bytes in (
        ("Metallica Nothing Else", ".gp5", b"GP5RAW"),
        ("Capricho Arabe", ".gp4", b"TAR_GP4_BYTES"),
        ("Bourree", ".tab", b"e|---0---|\n"),
    ):
        hits = search_digital_scores(
            q,
            instrument=Instrument.GUITAR,
            include_remote=False,
            include_local=False,
            cache_dir=cache_dir,
        )
        assert hits and hits[0].source == "DadaGP" and hits[0].format_ext == expected_ext
        out = download_score_result(hits[0], download_dir=tmp_path / "dl_gp")
        assert out.read_bytes() == expected_bytes

    with pytest.raises(ValueError, match="Unsafe archive member"):
        gp_mod.extract_gp_catalog_member(gp_dir, "file::../../etc/passwd", tmp_path / "x.gp5")

    def _raise_403(*_a: object, **_kw: object) -> object:
        raise urllib.error.HTTPError("https://zenodo.org/api/records/4789123", 403, "Forbidden", {}, None)

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_h: SimpleNamespace(open=_raise_403),
    )
    with pytest.raises(RuntimeError, match="approved researcher access"):
        sync_gp_catalog(cache_dir=cache_dir, zenodo_record="4789123")

    assert main(["dadagp", "status", "--cache-dir", str(cache_dir)]) == 0
    assert "3" in capsys.readouterr().out
