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
    humdrum: Callable[[], list[dict[str, Any]]] = list,
    mutopia: Callable[..., list[dict[str, Any]]] | None = None,
    imslp: Callable[..., list[dict[str, Any]]] | None = None,
    archive: Callable[..., list[dict[str, Any]]] | None = None,
) -> None:
    monkeypatch.setattr(search_mod, "_fetch_openscore_orchestra_index", orchestra)
    monkeypatch.setattr(search_mod, "_fetch_openscore_lieder_index", lieder)
    monkeypatch.setattr(search_mod, "_fetch_openscore_sq_index", quartets)
    monkeypatch.setattr(search_mod, "_fetch_when_in_rome_index", when_in_rome)
    monkeypatch.setattr(search_mod, "_fetch_humdrum_github_index", humdrum)
    monkeypatch.setattr(
        search_mod,
        "_search_mutopia",
        mutopia if mutopia is not None else (lambda *_a, **_kw: []),
    )
    monkeypatch.setattr(
        search_mod,
        "_search_imslp",
        imslp if imslp is not None else (lambda *_a, **_kw: []),
    )
    monkeypatch.setattr(
        search_mod,
        "_search_archive_org",
        archive if archive is not None else (lambda *_a, **_kw: []),
    )


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
        humdrum=lambda: [
            {
                "title": "Moonlight Sonata",
                "composer": "Ludwig van Beethoven",
                "format_ext": ".krn",
                "source": "Humdrum GitHub",
                "filename": "moonlight.krn",
                "download_url": "https://example.com/moonlight.krn",
            }
        ],
        archive=lambda *_a, **_kw: [
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".mid",
                "source": "Archive.org",
                "filename": "moonlight.mid",
                "download_url": "https://example.com/moonlight.mid",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".gp5",
                "source": "Archive.org",
                "filename": "moonlight.gp5",
                "download_url": "https://example.com/moonlight.gp5",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".mscz",
                "source": "Archive.org",
                "filename": "moonlight.mscz",
                "download_url": "https://example.com/moonlight.mscz",
            },
            {
                "title": "Moonlight Sonata",
                "composer": "Beethoven",
                "format_ext": ".musicxml",
                "source": "Archive.org",
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


def test_new_catalog_parsers_orchestra_when_in_rome_humdrum_and_mutopia(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import io
    import zipfile

    # 1. OpenScore Orchestra (Hauptstimme) skips _melody.mxl and parses full scores
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

    # 2. Humdrum manifest parser & tree-based classical .krn fallback
    repos = search_mod._parse_humdrum_list_repos(
        "!! comment\n**ghname\t**ghrepo\nmusedata\tvivaldi\tfoo\tbar\nmusedata\tvivaldi\tbaz\tqux\n"
    )
    assert repos == ["musedata/vivaldi"]

    monkeypatch.setattr(
        search_mod,
        "_http_get_text",
        lambda url: (_ for _ in ()).throw(RuntimeError("404")) if "index.hmd" in url else "",
    )
    monkeypatch.setattr(
        search_mod,
        "_fetch_github_tree",
        lambda _url: [{"path": "op01/kern/vivaldi-op01n01m01.krn"}],
    )
    vivaldi_recs = search_mod._fetch_single_humdrum_repo("musedata/vivaldi")
    assert len(vivaldi_recs) == 1
    assert vivaldi_recs[0]["composer"] == "Antonio Vivaldi"
    assert vivaldi_recs[0]["format_ext"] == ".krn"

    # 3. Mutopia HTML table parser (with UTF-8 bytes) & MIDI zip unpacking
    mutopia_html = """
    <table class="table-bordered result-table">
      <tr><td>Cello Concerto Cadenza</td><td>by C. Brey (1954–)</td><td>Hob. VIIb:2</td><td></td></tr>
      <tr><td>for Cello</td><td>2001</td><td>Classical</td><td></td></tr>
      <tr><td>Ed.</td><td>Public Domain</td><td><a href="piece-info.cgi?id=10">More Information</a></td><td></td></tr>
      <tr>
        <td class="zipped"><a href="https://www.mutopiaproject.org/ftp/BreyC/cadenza/cadenza-mids.zip">.mid files (zipped)</a></td>
        <td><a href="https://www.mutopiaproject.org/ftp/BreyC/cadenza/cadenza.ly">.ly file</a></td>
      </tr>
    </table>
    """.encode()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("01_allegro.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60")
        zf.writestr("02_adagio.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60")

    def _fake_mutopia_get(url: str, **_kwargs: object) -> SimpleNamespace:
        if "make-table.cgi" in url:
            return SimpleNamespace(content=mutopia_html, raise_for_status=lambda: None)
        return SimpleNamespace(content=zip_buf.getvalue(), headers={}, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(search_mod.requests, "get", _fake_mutopia_get)
    mut_recs = search_mod._search_mutopia("haydn cello", frozenset({".mid", ".ly"}))
    assert len(mut_recs) == 2
    mid_rec = next(r for r in mut_recs if r["format_ext"] == ".mid")
    assert mid_rec["composer"] == "C. Brey"
    assert "Hob. VIIb:2" in mid_rec["title"]

    hit = search_mod.ScoreSearchResult(**mid_rec, score=20.0)
    saved_mid = download_score_result(hit, download_dir=tmp_path / "mutopia_dl")
    assert saved_mid.exists()
    assert saved_mid.read_bytes().startswith(b"MThd")
    assert (tmp_path / "mutopia_dl" / "02_adagio.mid").exists()


def test_imslp_solo_typeset_ranking_archive_pdf_and_gate_handling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from transcribe_music.ingest import imslp as imslp_mod

    wikitext = """
{{#fte:imslpfile
|File Name 1=PMLP18850-Haydn_Manuscript.pdf
|File Description 1=Complete Score
|Image Type=Manuscript Scan
}}
{{#fte:imslpfile
|File Name 1=PMLP18850-Haydn_Oboe1.pdf
|File Description 1=Oboe 1 Part
|Image Type=Normal Scan
}}
{{#fte:imslpfile
|File Name 1=PMLP18850-Haydn_Cello_Solo_Mandozzi.pdf
|File Description 1=Complete Part
|Editor={{LinkEd|Orfeo|Mandozzi|1968|}}
|Image Type=Typeset
}}
{{#fte:imslpfile
|File Name 1=PMLP18850-Haydn_Cadenza_Popper.pdf
|File Description 1=Cadenza
|Image Type=Normal Scan
}}
"""

    def _fake_imslp_get(url: str, params=None, **_kwargs):
        action = (params or {}).get("list") or (params or {}).get("prop")
        if action == "search":
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "query": {
                        "search": [
                            {"title": "Cello Concerto No.2 in D major, Hob.VIIb:2 (Haydn, Joseph)"}
                        ]
                    }
                },
            )
        if action == "revisions":
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "query": {
                        "pages": {
                            "18850": {
                                "title": "Cello Concerto No.2 in D major, Hob.VIIb:2 (Haydn, Joseph)",
                                "revisions": [{"*": wikitext}],
                            }
                        }
                    }
                },
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "query": {
                    "pages": {
                        "1": {
                            "title": "File:PMLP18850-Haydn_Cello_Solo_Mandozzi.pdf",
                            "imageinfo": [
                                {"url": "//imslp.org/images/c/cf/PMLP18850-Haydn_Cello_Solo_Mandozzi.pdf"}
                            ],
                        }
                    }
                }
            },
        )

    monkeypatch.setattr(imslp_mod.requests, "get", _fake_imslp_get)
    imslp_hits = imslp_mod.search_imslp_works(
        "Haydn Cello Concerto D major",
        frozenset({".pdf", ".mxl", ".mid"}),
        user_agent="test",
    )
    assert len(imslp_hits) == 4
    # Solo Typeset edition must rank first above Manuscript, Oboe, and Cadenza
    assert imslp_hits[0]["filename"] == "PMLP18850-Haydn_Cello_Solo_Mandozzi.pdf"
    assert imslp_hits[0]["composer"] == "Joseph Haydn"
    assert "Typeset" in (imslp_hits[0]["edition_note"] or "")
    assert imslp_hits[0]["download_url"] == "https://imslp.org/images/c/cf/PMLP18850-Haydn_Cello_Solo_Mandozzi.pdf"

    # Digital results still outrank IMSLP PDFs, and include_pdf=False excludes PDFs
    _stub_remote_sources(
        monkeypatch,
        imslp=lambda *_a, **_kw: imslp_hits,
        mutopia=lambda *_a, **_kw: [
            {
                "title": "Haydn Cello Concerto in D major (MIDI)",
                "composer": "Joseph Haydn",
                "format_ext": ".mid",
                "source": "Mutopia Project",
                "filename": "haydn_cello.mid",
                "download_url": "https://example.com/haydn_cello.mid",
            }
        ],
    )
    mixed = search_digital_scores(
        "Haydn Cello Concerto",
        limit=5,
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert mixed[0].format_ext == ".mid"
    assert any(r.format_ext == ".pdf" and r.source == "IMSLP" for r in mixed)

    no_pdf = search_digital_scores(
        "Haydn Cello Concerto",
        limit=5,
        include_pdf=False,
        include_local=False,
        cache_dir=tmp_path / "cache",
    )
    assert all(r.format_ext != ".pdf" for r in no_pdf)

    # Anti-bot friendlytest.html gate raises ManualDownloadRequiredError & CLI handles it gracefully
    monkeypatch.setattr(
        search_mod.requests,
        "get",
        lambda *_a, **_kw: SimpleNamespace(
            content=b"<!DOCTYPE html><html><body>Verify</body></html>",
            url="https://imslp.org/friendlytest.html",
            headers={"content-type": "text/html"},
            raise_for_status=lambda: None,
        ),
    )
    pdf_hit = next(r for r in mixed if r.source == "IMSLP")
    with pytest.raises(search_mod.ManualDownloadRequiredError) as exc_info:
        download_score_result(pdf_hit, download_dir=tmp_path / "dl_gate")
    assert "imslp.org/wiki/File:" in exc_info.value.url

    runner = CliRunner()
    cli_res = runner.invoke(
        app,
        [
            "search",
            "Haydn Cello Concerto",
            "--format",
            "pdf",
            "--select",
            "1",
            "--instrument",
            "cello",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert cli_res.exit_code == 0, cli_res.stdout
    assert "browser verification" in cli_res.stdout
    assert "transcribe-music convert" in cli_res.stdout

    # Archive.org query excludes audio/lp_ items and picks solo PDF from collection:imslp
    picked = search_mod._pick_best_archive_file(
        [
            {"name": "item_meta.xml"},
            {"name": "PMLP_piano.pdf"},
            {"name": "PMLP_cello_solo.pdf"},
        ],
        frozenset({".pdf", ".mid"}),
        query_tokens=("haydn", "cello"),
    )
    assert picked == ("PMLP_cello_solo.pdf", ".pdf")
