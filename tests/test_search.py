"""Tests for score search, PDMX/DadaGP SQLite catalogs, archive extraction, and HTTP guards."""

from __future__ import annotations

import io
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_simple_cello_score, write_gp5_file, write_mxl_bytes

from transcribe_music import (
    Instrument,
    convert_music,
    download_score_result,
    get_gp_catalog_status,
    get_pdmx_status,
    search_digital_scores,
    sync_gp_catalog,
    sync_pdmx,
)
from transcribe_music.cli import main
from transcribe_music.ingest import _io as io_mod
from transcribe_music.ingest import gp_catalog as gp_mod
from transcribe_music.ingest import search as search_mod
from transcribe_music.ingest.gp_catalog import extract_gp_catalog_member
from transcribe_music.ingest.pdmx import extract_pdmx_member


def _rec(title: str, ext: str, source: str, composer: str, filename: str, url: str, **extra: Any) -> dict[str, Any]:
    return {
        "title": title,
        "composer": composer,
        "format_ext": ext,
        "source": source,
        "filename": filename,
        "download_url": url,
        **extra,
    }


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
            _rec("Der Erlkönig", ".mxl", "When-in-Rome", "Emilie Mayer", "Der_Erlkonig.mxl", "https://example.com/erlkonig.mxl"),
            _rec("Moonlight Sonata 1st Movement", ".mxl", "When-in-Rome", "Beethoven", "moonlight_1.mxl", "https://example.com/moonlight_1.mxl"),
            _rec("Moonlight Sonata 3rd Movement", ".mxl", "When-in-Rome", "Beethoven", "moonlight_3.mxl", "https://example.com/moonlight_3.mxl"),
        ],
        orchestra=lambda: [
            _rec("Moonlight Sonata", ext, "OpenScore Orchestra", "Beethoven", f"moonlight{ext}", f"https://example.com/moonlight{ext}")
            for ext in (".krn", ".mid", ".gp5", ".mscz", ".musicxml")
        ],
    )

    results = search_digital_scores(
        "Moonlight Sonata", limit=5, include_local=False, cache_dir=tmp_path / "cache"
    )
    assert len(results) == 5
    assert all(r.format_ext in {".musicxml", ".mxl"} for r in results[:3])

    mxl_only = search_digital_scores(
        "Moonlight Sonata", limit=5, musicxml_only=True, include_local=False, cache_dir=tmp_path / "cache"
    )
    assert [r.format_ext for r in mxl_only] == [".musicxml", ".mxl", ".mxl"]

    krn_only = search_digital_scores(
        "Moonlight Sonata", formats=["krn", "mid"], include_local=False, cache_dir=tmp_path / "cache"
    )
    assert {r.format_ext for r in krn_only} == {".krn", ".mid"}

    diacritics = search_digital_scores("Erlkonig", include_local=False, cache_dir=tmp_path / "cache")
    assert len(diacritics) == 1 and diacritics[0].title == "Der Erlkönig"

    # Stale index cache survives transient network outage
    monkeypatch.setattr(search_mod, "_fetch_when_in_rome_index", _boom)
    assert search_mod._load_or_fetch_index(
        tmp_path / "cache", "when_in_rome", _boom, allow_network=True, ttl_sec=-1
    )

    # Alpha-digit boundary tokenization matches "etude 1" against "Etude1" while keeping catalog tokens
    assert search_mod._normalize_text("Villa-Lobos_Etude1") == "villa lobos etude 1"
    assert search_mod._normalize_text("Mozart K156 movement2") == "mozart k156 movement 2"


def test_offline_music21_corpus_and_github_tree_parsers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = search_digital_scores(
        "bwv1.6", limit=5, musicxml_only=True, include_remote=False, cache_dir=tmp_path / "cache"
    )
    assert results and results[0].source == "music21 Corpus"
    saved = download_score_result(results[0], download_dir=tmp_path / "downloads")
    assert saved.exists() and saved.stat().st_size > 0

    k156_hits = search_digital_scores(
        "mozart k156 movement2",
        limit=5,
        include_remote=False,
        cache_dir=tmp_path / "cache",
    )
    assert k156_hits and k156_hits[0].local_corpus_path == "mozart/k156/movement2.mxl"

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
                [
                    {"path": "Brouwer/Estudios_Sencillos/Estudio_01.gp5"},
                    {"path": "Brouwer/dada/b1.gp5"},
                ]
                if "gp-classical-guitar/git/trees/master" in url
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
    assert gp_cls[0]["format_ext"] == ".gp5"
    assert gp_cls[0]["download_url"].startswith(
        "https://raw.githubusercontent.com/otnemrasordep/gp-classical-guitar/master/"
    )


def test_cli_search_and_http_redirect_security_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mxl_bytes = write_mxl_bytes(make_simple_cello_score("Moonlight Study"), tmp_path)
    _stub_remote_sources(
        monkeypatch,
        when_in_rome=lambda: [
            _rec(
                "Moonlight Sonata 1st Movement",
                ".mxl",
                "When-in-Rome",
                "Ludwig van Beethoven",
                "../../evil_path/moonlight.mxl",
                "https://example.com/moonlight.mxl",
            )
        ],
    )
    monkeypatch.setattr(
        urllib.request, "build_opener", _fake_opener(mxl_bytes, "https://example.com/moonlight.mxl")
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

    for bad_url in (
        "http://example.com/score.mxl",
        "https://user:pass@example.com/score.mxl",
        "https://localhost/score.mxl",
        "https://127.0.0.1/score.mxl",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/score.mxl",
        "https://example.com/score.mxl\n",
    ):
        with pytest.raises(ValueError, match="permitted|Only HTTPS"):
            io_mod.validate_https_url(bad_url)

    valid = replace(insecure, download_url="https://example.com/score.mxl")
    monkeypatch.setattr(
        urllib.request, "build_opener", _fake_opener(b"PK", "http://downgraded.example.com/score.mxl")
    )
    with pytest.raises(ValueError, match="Only HTTPS"):
        download_score_result(valid, download_dir=tmp_path)

    handler = io_mod._SafeHttpsRedirectHandler()
    auth_req = urllib.request.Request("https://api.example.com/data", headers={"Authorization": "Bearer secret"})
    with pytest.raises(ValueError, match="Only HTTPS"):
        handler.redirect_request(auth_req, None, 302, "Found", {}, "http://evil.example.com/x")
    redir_req = handler.redirect_request(auth_req, None, 302, "Found", {}, "https://cdn.other.com/x")
    assert redir_req is not None and "Authorization" not in redir_req.headers

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


def test_pdmx_sqlite_sync_search_extract_and_security(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    pdmx_dir = tmp_path / "pdmx_store"
    pdmx_dir.mkdir(parents=True)

    mxl_rel = "./mxl/h/ha/QmHaydnCelloD.mxl"
    (pdmx_dir / "PDMX.csv").write_text(
        "\n".join(
            [
                'mxl,mid,pdf,title,song_name,composer_name,artist_name,genres,tracks,tags,n_tracks,complexity,rating,n_ratings,is_best_unique_arrangement,"subset:no_license_conflict"',
                f'{mxl_rel},NA,NA,"Cello Concerto No. 2","Cello Concerto","Joseph Haydn",NA,classical,Cello,cello,2,3.0,4.85,12,True,True',
                './mxl/c/co/QmConflict.mxl,NA,NA,"Cello Conflict","Cello",NA,NA,classical,Cello,NA,NA,NA,NA,NA,False,False',
                './mxl/d/dv/QmDvorak.mxl,NA,NA,"Cello Concerto in B minor","Cello Concerto","Antonín Dvořák",NA,classical,Cello,cello,2,4.0,4.7,9,True,True',
                './mxl/g/gu/QmGuitar.mxl,NA,NA,"Capricho","Capricho","Composer B",NA,classical,"Classical Guitar",guitar,1,2.0,4.5,5,False,True',
            ]
        ),
        encoding="utf-8",
    )

    status_meta = sync_pdmx(cache_dir=cache_dir, pdmx_dir=pdmx_dir, download_csv=False)
    assert status_meta.score_count == 3 and status_meta.mxl_available is False

    # Un-synced PDMX rows are hidden by default and returned when include_unsynced_pdmx=True
    assert (
        search_digital_scores(
            "Dvorak Cello Concerto", include_remote=False, include_local=False, cache_dir=cache_dir
        )
        == []
    )
    dvorak_hits = search_digital_scores(
        "Dvorak Cello Concerto",
        include_unsynced_pdmx=True,
        include_remote=False,
        include_local=False,
        cache_dir=cache_dir,
    )
    assert len(dvorak_hits) == 1 and dvorak_hits[0].composer == "Antonín Dvořák"
    assert search_digital_scores(
        "%", include_unsynced_pdmx=True, include_remote=False, include_local=False, cache_dir=cache_dir
    ) == []
    with pytest.raises(FileNotFoundError, match="pdmx sync --with-mxl"):
        download_score_result(dvorak_hits[0], download_dir=tmp_path / "dl")

    gtr_hits = search_digital_scores(
        "Capricho",
        instrument=Instrument.GUITAR,
        include_unsynced_pdmx=True,
        include_remote=False,
        include_local=False,
        cache_dir=cache_dir,
    )
    assert gtr_hits[0].filename == "QmGuitar.mxl" and "Classical Guitar" in (gtr_hits[0].edition_note or "")

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
    assert convert_music(saved_mxl, Instrument.CELLO, output_path=tmp_path / "haydn.musicxml").fingering_count == 4

    assert main(["pdmx", "status", "--cache-dir", str(cache_dir)]) == 0
    assert (
        main(
            [
                "pdmx",
                "sync",
                "--pdmx-dir",
                str(pdmx_dir),
                "--no-download",
                "--include-license-conflicts",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    assert get_pdmx_status(cache_dir).score_count == 4

    for unsafe in ("../../etc/passwd", "/etc/passwd", "mxl/../../etc/passwd"):
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            extract_pdmx_member(pdmx_dir, unsafe, tmp_path / "out.mxl")
    with pytest.raises(ValueError, match="not a regular file"):
        extract_pdmx_member(pdmx_dir, "mxl/evil_link.mxl", tmp_path / "out.mxl")


def test_guitar_search_and_dadagp_sqlite_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_remote_sources(
        monkeypatch,
        quartets=lambda: [
            _rec("Asturias Quartet Transcription", ".mxl", "OpenScore StringQuartets", "Isaac Albeniz", "asturias_q.mxl", "https://example.com/asturias_q.mxl")
        ],
        gp_classical=lambda: [
            _rec("Asturias (Leyenda)", ".gp5", "GP Classical Guitar", "Isaac Albeniz", "asturias.gp5", "https://example.com/asturias.gp5", edition_note="Classical Guitar TAB")
        ],
    )
    gtr_hits = search_digital_scores(
        "Asturias Albeniz",
        instrument=Instrument.GUITAR,
        include_local=False,
        cache_dir=tmp_path / "search_cache",
    )
    assert gtr_hits[0].format_ext == ".gp5" and gtr_hits[0].source == "GP Classical Guitar"

    gp_dir = tmp_path / "gp_library"
    (gp_dir / "Rock" / "Metallica").mkdir(parents=True)
    (gp_dir / "Rock" / "Metallica" / "Nothing_Else_Matters.gp5").write_bytes(b"GP5RAW")
    outside_gp = tmp_path / "secret.gp5"
    outside_gp.write_bytes(b"SECRET")
    (gp_dir / "Rock" / "escape.gp5").symlink_to(outside_gp)

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
        for payload in (b"OLD_BYTES", b"TAR_GP4_BYTES"):
            ti = tarfile.TarInfo("DadaGP-v1.1/A/Albeniz/Capricho_Arabe.gp4")
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
    (gp_dir / "bundle.tar.gz").write_bytes(tar_buf.getvalue())

    with zipfile.ZipFile(gp_dir / "pack.zip", "w") as zf:
        zf.writestr("Bach/Bourree_in_E_minor.tab", "e|---0---|\n")
        sym_zi = zipfile.ZipInfo("Bach/evil_link.tab")
        sym_zi.external_attr = 0o120777 << 16
        zf.writestr(sym_zi, "/etc/passwd")

    cache_dir = tmp_path / "gp_cache"
    status = sync_gp_catalog(cache_dir=cache_dir, gp_dir=gp_dir)
    assert status.db_exists is True and status.score_count == 3
    assert get_gp_catalog_status(cache_dir).score_count == 3

    for q, expected_ext, expected_bytes in (
        ("Metallica Nothing-Else", ".gp5", b"GP5RAW"),
        ("Capricho Arabe", ".gp4", b"TAR_GP4_BYTES"),
        ("Bourree", ".tab", b"e|---0---|\n"),
    ):
        hits = search_digital_scores(
            q, instrument=Instrument.GUITAR, include_remote=False, include_local=False, cache_dir=cache_dir
        )
        assert hits and hits[0].source == "DadaGP" and hits[0].format_ext == expected_ext
        out = download_score_result(hits[0], download_dir=tmp_path / "dl_gp")
        assert out.read_bytes() == expected_bytes

    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_gp_catalog_member(gp_dir, "file::../../etc/passwd", tmp_path / "x.gp5")
    with pytest.raises(ValueError, match="symlink"):
        extract_gp_catalog_member(gp_dir, "zip::pack.zip::Bach/evil_link.tab", tmp_path / "x.tab")

    # Zenodo Bearer token is never sent to external mirror hosts
    seen_auth: list[tuple[str, str | None]] = []
    with monkeypatch.context() as mp:
        mp.setattr(
            gp_mod,
            "http_get_json",
            lambda url, **_kw: {
                "files": [{"key": "tabs.zip", "links": {"self": "https://cdn.other.example/tabs.zip"}}]
            },
        )
        mp.setattr(
            gp_mod,
            "download_https_stream",
            lambda url, dest, *, headers=None, **_kw: (
                seen_auth.append((url, (headers or {}).get("Authorization"))),
                dest,
            )[1],
        )
        gp_mod._download_zenodo_record_archives(
            "12345", tmp_path / "z_dl", token="secret_tok", force=True
        )
    assert seen_auth == [("https://cdn.other.example/tabs.zip", None)]

    def _raise_403(*_a: object, **_kw: object) -> object:
        raise urllib.error.HTTPError("https://zenodo.org/api/records/4789123", 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_h: SimpleNamespace(open=_raise_403))
    with pytest.raises(RuntimeError, match="approved researcher access"):
        sync_gp_catalog(cache_dir=cache_dir, zenodo_record="4789123")

    assert main(["dadagp", "status", "--cache-dir", str(cache_dir)]) == 0
    assert "3" in capsys.readouterr().out

    # CLI search --guitar-staff both forwards staff mode to conversion
    gp5_bytes = write_gp5_file(tmp_path / "Etude1.gp5").read_bytes()
    _stub_remote_sources(
        monkeypatch,
        gp_classical=lambda: [
            _rec("Etude1", ".gp5", "GP Classical Guitar", "Heitor Villa-Lobos", "Villa-Lobos_Etude1.gp5", "https://example.com/Villa-Lobos_Etude1.gp5", edition_note="Classical Guitar TAB")
        ],
    )
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        _fake_opener(gp5_bytes, "https://example.com/Villa-Lobos_Etude1.gp5"),
    )
    cli_cache = tmp_path / "cli_gtr_cache"
    assert (
        main(
            [
                "search",
                "villa-lobos etude 1",
                "--select",
                "1",
                "-i",
                "guitar",
                "--guitar-staff",
                "both",
                "--dir",
                str(tmp_path / "cli_dl"),
                "--cache-dir",
                str(cli_cache),
            ]
        )
        == 0
    )
    out_text = capsys.readouterr().out
    out_line = next(line for line in out_text.splitlines() if "Output MusicXML" in line)
    exported_xml = Path(out_line.split("Output MusicXML", 1)[1].strip())
    assert ET.parse(exported_xml).findtext(".//attributes/staves") == "2"
