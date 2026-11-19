"""Stdlib argparse CLI entrypoint for transcribe-music."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from transcribe_music.ingest.digital import inspect_digital_score
from transcribe_music.ingest.gp_catalog import (
    GpCatalogStatus,
    get_gp_catalog_status,
    sync_gp_catalog,
)
from transcribe_music.ingest.pdmx import PdmxStatus, get_pdmx_status, sync_pdmx
from transcribe_music.ingest.search import download_score_result, search_digital_scores
from transcribe_music.models import (
    GuitarStaffMode,
    Instrument,
    OmrBackend,
    ScoreSearchResult,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.pipeline import convert_music
from transcribe_music.storage.catalog import CatalogStore


def _print_table(
    title: str,
    headers: Sequence[str] | None,
    rows: Sequence[Sequence[str]],
) -> None:
    print(title)
    all_rows = [headers, *rows] if headers else list(rows)
    if not all_rows:
        return
    widths = [max(len(str(r[c])) for r in all_rows) for c in range(len(all_rows[0]))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    if headers:
        print(fmt.format(*headers).rstrip())
        print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*(str(c) for c in row)).rstrip())


def _print_result(res: TranscriptionResult) -> None:
    rows: list[tuple[str, str]] = [
        ("Output MusicXML", str(res.musicxml_path)),
        ("Instrument", res.instrument.value),
        ("Source", f"{res.source_type.value}: {res.source_ref}"),
    ]
    if res.backend:
        rows.append(("Backend", res.backend))
    if res.selected_part:
        rows.append(("Selected Part", res.selected_part))
    rows.extend(
        [
            ("Cache Hit", "Yes" if res.cached else "No"),
            ("Elapsed", f"{res.elapsed_sec:.2f}s"),
            ("Measures", str(res.measure_count)),
            ("Notes", str(res.note_count)),
            ("Chords (<harmony>)", str(res.chord_count)),
            ("Fingerings", str(res.fingering_count)),
        ]
    )
    _print_table("MusicXML Ready", None, rows)
    for warn in res.warnings:
        print(f"Warning: {warn}")


def _print_search_results(query: str, results: list[ScoreSearchResult]) -> None:
    show_details = any(item.edition_note for item in results)
    headers = ["#", "Format", "Title", "Composer", "Source"]
    if show_details:
        headers.append("Details")
    rows: list[list[str]] = []
    for idx, item in enumerate(results, start=1):
        row = [
            str(idx),
            item.format_ext.lstrip(".").upper(),
            item.title,
            item.composer or "-",
            item.source,
        ]
        if show_details:
            row.append(item.edition_note or "-")
        rows.append(row)
    _print_table(f"Top {len(results)} Scores for {query!r}", headers, rows)


def _print_pdmx_status(status: PdmxStatus) -> None:
    _print_table(
        "PDMX Local Catalog",
        ["Field", "Value"],
        [
            ("PDMX Directory", str(status.pdmx_dir)),
            ("PDMX.csv Present", "Yes" if status.csv_exists else "No"),
            ("SQLite Index", "Yes" if status.db_exists else "No"),
            ("Indexed Scores", f"{status.score_count:,}"),
            (
                "MXL Archive/Dir",
                "Available" if status.mxl_available else "Not synced (--with-mxl)",
            ),
            (
                "MID Archive/Dir",
                "Available" if status.mid_available else "Not synced (--with-mid)",
            ),
        ],
    )


def _print_gp_status(status: GpCatalogStatus) -> None:
    _print_table(
        "DadaGP / Guitar Pro Local Catalog",
        ["Field", "Value"],
        [
            ("Catalog Directory", str(status.gp_dir)),
            ("SQLite Index", "Yes" if status.db_exists else "No"),
            ("Indexed Scores", f"{status.score_count:,}"),
            ("Files / Archives", "Available" if status.archive_available else "Empty"),
        ],
    )


def _existing_file(raw: str) -> Path:
    path = Path(raw)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {raw}")
    return path


def _bounded_int(lo: int, hi: int | None = None):
    def _parse(raw: str) -> int:
        val = int(raw)
        if val < lo or (hi is not None and val > hi):
            bound = f"[{lo}..{hi}]" if hi is not None else f">= {lo}"
            raise argparse.ArgumentTypeError(f"expected integer {bound}, got {val}")
        return val

    return _parse


def _positive_float(raw: str) -> float:
    try:
        val = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected positive number, got {raw!r}") from exc
    if not math.isfinite(val) or val <= 0.0:
        raise argparse.ArgumentTypeError(f"expected positive finite number, got {val}")
    return val


def _enum_choices(enum_cls: type[Enum]) -> list[str]:
    return [m.value for m in enum_cls]


def _add_cache_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".transcribe_cache"),
        help="Cache directory.",
    )


def _cmd_convert(args: argparse.Namespace) -> int:
    inst = Instrument(args.instrument)
    cfg = TranscriptionConfig(
        instrument=inst,
        part=args.part,
        title=args.title,
        composer=args.composer,
        enrich_chords=args.chords,
        chord_window_beats=args.chord_window,
        include_slash_chords=args.slash_chords,
        roman_numerals=args.roman_numerals,
        enrich_fingering=args.fingering,
        preserve_existing_chords=args.preserve_chords,
        preserve_existing_fingering=args.preserve_fingering,
        preserve_existing_tab=args.preserve_tab,
        guitar_staff=GuitarStaffMode(args.guitar_staff),
        use_cache=args.cache,
        cache_dir=args.cache_dir,
        pdf_dpi=args.dpi,
        omr_backend=OmrBackend(args.omr_backend),
    )
    res = convert_music(input_path=args.input_file, instrument=inst, output_path=args.output, config=cfg)
    _print_result(res)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_digital_score(args.score_file)
    show_guitar_meta = any(p.tuning or p.capo for p in info.parts)
    headers = ["#", "Part", "Instrument", "GM", "Clef", "Bars", "Notes", "Range"]
    if show_guitar_meta:
        headers.extend(["Tuning", "Capo"])
    rows: list[list[str]] = []
    for p in info.parts:
        span = (
            f"{p.pitch_min}..{p.pitch_max}"
            if p.pitch_min is not None and p.pitch_max is not None
            else "-"
        )
        row = [
            str(p.index),
            p.name,
            p.instrument_name or "-",
            str(p.midi_program) if p.midi_program is not None else "-",
            p.clef or "-",
            str(p.measure_count),
            str(p.note_count),
            span,
        ]
        if show_guitar_meta:
            row.append(p.tuning or "-")
            row.append(str(p.capo) if p.capo else "-")
        rows.append(row)
    _print_table(f"{info.title or info.source_path.name} ({info.format_name})", headers, rows)
    return 0


def _cmd_search(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    inst = Instrument(args.instrument) if args.instrument is not None else None
    parsed_formats = (
        [tok.strip() for tok in args.formats.split(",") if tok.strip()] if args.formats else None
    )
    results = search_digital_scores(
        args.query,
        limit=args.limit,
        instrument=inst,
        musicxml_only=args.musicxml_only,
        formats=parsed_formats,
        include_unsynced_pdmx=args.include_unsynced,
        include_remote=not args.offline,
        cache_dir=args.cache_dir,
    )
    if not results:
        print(f"No matching digital scores found for {args.query!r}.")
        return 0

    _print_search_results(args.query, results)

    if args.select is not None:
        if not (1 <= args.select <= len(results)):
            parser.error(f"--select must be between 1 and {len(results)}, got {args.select}.")
        chosen_idx = args.select
    else:
        try:
            raw_choice = input(
                f"Select result to download [1-{len(results)}] (Enter or 'q' to cancel): "
            ).strip()
        except EOFError:
            raw_choice = ""
        if not raw_choice or raw_choice.lower() in {"q", "quit", "exit", "0", "n", "no"}:
            print("Download cancelled.")
            return 0
        if not raw_choice.isdigit() or not (1 <= int(raw_choice) <= len(results)):
            parser.error(f"Invalid selection {raw_choice!r}. Enter a number from 1 to {len(results)}.")
        chosen_idx = int(raw_choice)

    chosen = results[chosen_idx - 1]
    saved_path = download_score_result(chosen, output_path=args.output, download_dir=args.download_dir)
    print(f"Downloaded: {saved_path} ({chosen.format_ext.lstrip('.').upper()} from {chosen.source})")

    if inst is not None:
        cfg = TranscriptionConfig(
            instrument=inst,
            guitar_staff=GuitarStaffMode(args.guitar_staff),
            cache_dir=args.cache_dir,
        )
        res = convert_music(input_path=saved_path, instrument=inst, config=cfg)
        _print_result(res)
    else:
        suggested = (
            "guitar"
            if chosen.format_ext in {".gp", ".gpx", ".gp5", ".gp4", ".gp3", ".tab"}
            else "piano"
        )
        print(f'Next: transcribe-music convert "{saved_path}" --instrument {suggested}')
    return 0


def _cmd_cache_list(args: argparse.Namespace) -> int:
    rows = CatalogStore(args.cache_dir).list_recent_jobs(limit=args.limit)
    if not rows:
        print("No cached jobs in SQLite.")
        return 0
    headers = [
        "job_hash",
        "source_type",
        "instrument",
        "backend",
        "part",
        "measures",
        "notes",
        "chords",
        "fingerings",
        "elapsed_s",
    ]
    table_rows = [
        [
            str(r["job_hash"])[:10],
            str(r["source_type"]),
            str(r["instrument"]),
            str(r["backend"] or "-"),
            str(r["selected_part"] or "-"),
            str(r["measure_count"]),
            str(r["note_count"]),
            str(r["chord_count"]),
            str(r["fingering_count"]),
            f"{r['elapsed_sec']:.2f}",
        ]
        for r in rows
    ]
    _print_table("SQLite MusicXML Catalog", headers, table_rows)
    return 0


def _cmd_cache_clear(args: argparse.Namespace) -> int:
    store = CatalogStore(args.cache_dir)
    removed = store.clear_cache()
    print(f"Cleared {removed} job records from {store.db_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-music",
        description="Convert digital symbolic scores (primary) or sheet music PDFs (fallback) into enriched MusicXML.",
    )
    sub = parser.add_subparsers(dest="command")

    p_conv = sub.add_parser("convert", help="Convert any digital score or fallback PDF to MusicXML.")
    p_conv.add_argument(
        "input_file",
        type=_existing_file,
        help="Digital score (.musicxml/.mxl/.mid/.krn/.mei/.abc/.gp*/.tab/.mscz/.capx/.ly) or fallback .pdf.",
    )
    p_conv.add_argument(
        "--instrument", "-i", required=True, choices=_enum_choices(Instrument), help="Target instrument."
    )
    p_conv.add_argument("--output", "-o", type=Path, default=None, help="Output .musicxml path.")
    p_conv.add_argument(
        "--part", "-p", default=None, help="Part number (1-based) or name filter (e.g. 'cello' or '1,2')."
    )
    p_conv.add_argument("--title", default=None, help="Override MusicXML work/movement title.")
    p_conv.add_argument("--composer", default=None, help="Override MusicXML composer metadata.")
    p_conv.add_argument(
        "--dpi", type=_bounded_int(150, 600), default=300, help="PDF rasterization DPI (150-600)."
    )
    p_conv.add_argument(
        "--omr-backend",
        choices=_enum_choices(OmrBackend),
        default=OmrBackend.AUTO.value,
        help="PDF OMR backend (auto prefers Audiveris if installed, else homr).",
    )
    p_conv.add_argument(
        "--chords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate chord symbols.",
    )
    p_conv.add_argument(
        "--chord-window",
        type=_positive_float,
        default=2.0,
        help="Chord recognition window in quarter-note beats (default: 2.0).",
    )
    p_conv.add_argument(
        "--slash-chords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit slash-chord bass inversions like Dm7/C (disable with --no-slash-chords).",
    )
    p_conv.add_argument(
        "--roman-numerals",
        action="store_true",
        default=False,
        help="Annotate Roman numeral harmonic analysis below chord symbols.",
    )
    p_conv.add_argument(
        "--fingering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate fingerings/TAB.",
    )
    p_conv.add_argument(
        "--preserve-chords", action="store_true", help="Keep authored <harmony> chords when present."
    )
    p_conv.add_argument(
        "--preserve-fingering", action="store_true", help="Keep authored fingerings when present."
    )
    tab_grp = p_conv.add_mutually_exclusive_group()
    tab_grp.add_argument(
        "--preserve-tab",
        dest="preserve_tab",
        action="store_true",
        help="Keep authored guitar TAB placements when present (default).",
    )
    tab_grp.add_argument(
        "--recompute-tab",
        dest="preserve_tab",
        action="store_false",
        help="Recompute guitar TAB placements via Viterbi solver.",
    )
    p_conv.set_defaults(preserve_tab=True)
    p_conv.add_argument(
        "--guitar-staff",
        choices=_enum_choices(GuitarStaffMode),
        default=GuitarStaffMode.NOTATION.value,
        help="Guitar MusicXML staff layout: notation (Treble 8vb), tab (6-line TAB), or both.",
    )
    p_conv.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SQLite cache.",
    )
    _add_cache_dir_arg(p_conv)

    p_insp = sub.add_parser("inspect", help="List parts, instruments, clefs, and note counts.")
    p_insp.add_argument("score_file", type=_existing_file, help="Digital score file to inspect.")

    p_search = sub.add_parser(
        "search", help="Search symbolic score catalogs and download a pick."
    )
    p_search.add_argument("query", help="Title, composer, or keywords to search.")
    p_search.add_argument(
        "--limit", "-n", type=_bounded_int(1, 25), default=5, help="Maximum top results (1-25)."
    )
    p_search.add_argument(
        "--select", "-s", type=_bounded_int(1), default=None, help="1-based index to download."
    )
    p_search.add_argument("--output", "-o", type=Path, default=None, help="Output file or dir.")
    p_search.add_argument(
        "--dir", "-d", dest="download_dir", type=Path, default=Path("downloads"), help="Download dir."
    )
    p_search.add_argument(
        "--musicxml-only", action="store_true", help="Only return native MusicXML results."
    )
    p_search.add_argument(
        "--format", "-f", dest="formats", default=None, help="Comma-separated extension filter."
    )
    p_search.add_argument(
        "--instrument",
        "-i",
        choices=_enum_choices(Instrument),
        default=None,
        help="Prioritize matching scores and convert download immediately.",
    )
    p_search.add_argument(
        "--guitar-staff",
        choices=_enum_choices(GuitarStaffMode),
        default=GuitarStaffMode.NOTATION.value,
        help="Guitar MusicXML staff layout when converting with -i guitar (default: notation).",
    )
    p_search.add_argument(
        "--include-unsynced",
        action="store_true",
        default=False,
        help="Include local PDMX catalog rows whose score archives are not downloaded yet.",
    )
    p_search.add_argument(
        "--offline", action="store_true", help="Search local corpora/indexes without network calls."
    )
    _add_cache_dir_arg(p_search)

    p_cache = sub.add_parser("cache", help="Inspect or manage the SQLite conversion cache.")
    cache_sub = p_cache.add_subparsers(dest="subcommand")
    p_clist = cache_sub.add_parser("list", help="List cached MusicXML jobs stored in SQLite.")
    p_clist.add_argument("--limit", "-n", type=_bounded_int(1), default=20, help="Max jobs to show.")
    _add_cache_dir_arg(p_clist)
    p_cclear = cache_sub.add_parser("clear", help="Clear cached entries in SQLite.")
    _add_cache_dir_arg(p_cclear)

    p_pdmx = sub.add_parser("pdmx", help="Manage the optional local PDMX catalog.")
    pdmx_sub = p_pdmx.add_subparsers(dest="subcommand")
    p_pdmx_st = pdmx_sub.add_parser("status", help="Inspect local PDMX catalog status.")
    _add_cache_dir_arg(p_pdmx_st)
    p_pdmx_sync = pdmx_sub.add_parser("sync", help="Sync PDMX metadata/archives into SQLite.")
    p_pdmx_sync.add_argument("--pdmx-dir", type=Path, default=None, help="Custom PDMX directory.")
    p_pdmx_sync.add_argument("--with-mxl", action="store_true", help="Download mxl.tar.gz.")
    p_pdmx_sync.add_argument("--with-mid", action="store_true", help="Download mid.tar.gz.")
    p_pdmx_sync.add_argument("--extract", action="store_true", help="Extract .tar.gz archives.")
    p_pdmx_sync.add_argument(
        "--no-download", action="store_true", help="Index local PDMX.csv without downloading."
    )
    p_pdmx_sync.add_argument(
        "--include-license-conflicts", action="store_true", help="Include license-flagged rows."
    )
    p_pdmx_sync.add_argument("--force", action="store_true", help="Re-download cached archives.")
    _add_cache_dir_arg(p_pdmx_sync)

    p_gp = sub.add_parser("dadagp", help="Manage the optional local DadaGP / Guitar Pro catalog.")
    gp_sub = p_gp.add_subparsers(dest="subcommand")
    p_gp_st = gp_sub.add_parser("status", help="Inspect local DadaGP / Guitar Pro catalog status.")
    _add_cache_dir_arg(p_gp_st)
    p_gp_sync = gp_sub.add_parser(
        "sync", help="Index local Guitar Pro files or download a Zenodo record into SQLite."
    )
    p_gp_sync.add_argument("--gp-dir", type=Path, default=None, help="Local Guitar Pro directory.")
    p_gp_sync.add_argument(
        "--zenodo-record",
        default=None,
        help="Optional Zenodo record ID (set ZENODO_TOKEN env var if restricted).",
    )
    p_gp_sync.add_argument("--force", action="store_true", help="Re-download Zenodo archives.")
    _add_cache_dir_arg(p_gp_sync)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "convert":
        return _cmd_convert(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "search":
        return _cmd_search(args, parser)
    if args.command == "cache":
        if args.subcommand == "list":
            return _cmd_cache_list(args)
        if args.subcommand == "clear":
            return _cmd_cache_clear(args)
        parser.parse_args(["cache", "--help"])
        return 0
    if args.command == "pdmx":
        if args.subcommand == "status":
            _print_pdmx_status(get_pdmx_status(args.cache_dir))
            return 0
        if args.subcommand == "sync":
            _print_pdmx_status(
                sync_pdmx(
                    cache_dir=args.cache_dir,
                    pdmx_dir=args.pdmx_dir,
                    download_csv=not args.no_download,
                    download_mxl=args.with_mxl,
                    download_mid=args.with_mid,
                    extract_archives=args.extract,
                    include_license_conflicts=args.include_license_conflicts,
                    force=args.force,
                )
            )
            return 0
        parser.parse_args(["pdmx", "--help"])
        return 0
    if args.command == "dadagp":
        if args.subcommand == "status":
            _print_gp_status(get_gp_catalog_status(args.cache_dir))
            return 0
        if args.subcommand == "sync":
            _print_gp_status(
                sync_gp_catalog(
                    cache_dir=args.cache_dir,
                    gp_dir=args.gp_dir,
                    zenodo_record=args.zenodo_record,
                    force=args.force,
                )
            )
            return 0
        parser.parse_args(["dadagp", "--help"])
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
