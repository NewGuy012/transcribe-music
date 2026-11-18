"""Typer CLI entrypoint for transcribe-music."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

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

app = typer.Typer(
    name="transcribe-music",
    help="Convert digital symbolic scores (primary) or sheet music PDFs (fallback) into enriched MusicXML.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Inspect or manage the SQLite conversion cache.")
pdmx_app = typer.Typer(help="Manage the optional local PDMX (254k+ public domain MusicXML) catalog.")
dadagp_app = typer.Typer(help="Manage the optional local DadaGP / Guitar Pro catalog.")
app.add_typer(cache_app, name="cache")
app.add_typer(pdmx_app, name="pdmx")
app.add_typer(dadagp_app, name="dadagp")

console = Console()

InstrumentOpt = Annotated[Instrument, typer.Option(..., "--instrument", "-i", help="Target instrument.")]
OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Output .musicxml path.")]
PartOpt = Annotated[
    str | None,
    typer.Option("--part", "-p", help="Part number (1-based) or name filter (e.g. 'cello' or '1,2')."),
]
ChordsOpt = Annotated[bool, typer.Option("--chords/--no-chords", help="Annotate chord symbols.")]
FingeringOpt = Annotated[bool, typer.Option("--fingering/--no-fingering", help="Annotate fingerings/TAB.")]
PreserveChordsOpt = Annotated[
    bool,
    typer.Option("--preserve-chords", help="Keep authored <harmony> chords when present."),
]
PreserveFingeringOpt = Annotated[
    bool,
    typer.Option("--preserve-fingering", help="Keep authored fingerings when present."),
]
PreserveTabOpt = Annotated[
    bool,
    typer.Option(
        "--preserve-tab/--recompute-tab",
        help="Keep authored guitar TAB string/fret placements when present (default: preserve).",
    ),
]
GuitarStaffOpt = Annotated[
    GuitarStaffMode,
    typer.Option(
        "--guitar-staff",
        help="Guitar MusicXML staff layout: notation (Treble 8vb), tab (6-line TAB), or both.",
    ),
]
CacheOpt = Annotated[bool, typer.Option("--cache/--no-cache", help="Use SQLite cache.")]
CacheDirOpt = Annotated[Path, typer.Option("--cache-dir", help="Cache directory.")]
OmrBackendOpt = Annotated[
    OmrBackend,
    typer.Option("--omr-backend", help="PDF OMR backend (auto prefers Audiveris if installed, else homr)."),
]


def _print_result(res: TranscriptionResult) -> None:
    table = Table(title="MusicXML Ready", show_header=False)
    table.add_row("Output MusicXML", str(res.musicxml_path))
    table.add_row("Instrument", res.instrument.value)
    table.add_row("Source", f"{res.source_type.value}: {res.source_ref}")
    if res.backend:
        table.add_row("Backend", res.backend)
    if res.selected_part:
        table.add_row("Selected Part", res.selected_part)
    table.add_row("Cache Hit", "Yes" if res.cached else "No")
    table.add_row("Elapsed", f"{res.elapsed_sec:.2f}s")
    table.add_row("Measures", str(res.measure_count))
    table.add_row("Notes", str(res.note_count))
    table.add_row("Chords (<harmony>)", str(res.chord_count))
    table.add_row("Fingerings", str(res.fingering_count))
    console.print(table)
    for warn in res.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warn}")


def _print_search_results(query: str, results: list[ScoreSearchResult]) -> None:
    show_details = any(item.edition_note for item in results)
    table = Table(title=f"Top {len(results)} Scores for {query!r}")
    table.add_column("#", no_wrap=True)
    table.add_column("Format", no_wrap=True)
    table.add_column("Title")
    table.add_column("Composer")
    table.add_column("Source", no_wrap=True)
    if show_details:
        table.add_column("Details")
    for idx, item in enumerate(results, start=1):
        fmt_label = item.format_ext.lstrip(".").upper()
        row = [
            str(idx),
            fmt_label,
            item.title,
            item.composer or "-",
            item.source,
        ]
        if show_details:
            row.append(item.edition_note or "-")
        table.add_row(*row)
    console.print(table)


@app.command()
def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Digital score (.musicxml/.mxl/.mid/.krn/.mei/.abc/.gp*/.tab/.mscz/.capx/.ly) or fallback .pdf.",
        ),
    ],
    instrument: InstrumentOpt,
    output: OutputOpt = None,
    part: PartOpt = None,
    dpi: Annotated[int, typer.Option("--dpi", help="PDF rasterization DPI (for .pdf fallback only).")] = 300,
    omr_backend: OmrBackendOpt = OmrBackend.AUTO,
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    preserve_chords: PreserveChordsOpt = False,
    preserve_fingering: PreserveFingeringOpt = False,
    preserve_tab: PreserveTabOpt = True,
    guitar_staff: GuitarStaffOpt = GuitarStaffMode.NOTATION,
    cache: CacheOpt = True,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Convert any supported digital score (primary) or sheet music PDF (fallback) to MusicXML."""
    cfg = TranscriptionConfig(
        instrument=instrument,
        part=part,
        enrich_chords=chords,
        enrich_fingering=fingering,
        preserve_existing_chords=preserve_chords,
        preserve_existing_fingering=preserve_fingering,
        preserve_existing_tab=preserve_tab,
        guitar_staff=guitar_staff,
        use_cache=cache,
        cache_dir=cache_dir,
        pdf_dpi=dpi,
        omr_backend=omr_backend,
    )
    res = convert_music(input_path=input_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def inspect(
    score_file: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Digital score file to inspect."),
    ],
) -> None:
    """List all parts, instruments, clefs, and note counts in a digital score file."""
    info = inspect_digital_score(score_file)
    show_guitar_meta = any(p.tuning or p.capo for p in info.parts)
    table = Table(title=f"{info.title or info.source_path.name} ({info.format_name})")
    table.add_column("#", no_wrap=True)
    table.add_column("Part", no_wrap=True)
    table.add_column("Instrument")
    table.add_column("GM", no_wrap=True)
    table.add_column("Clef")
    table.add_column("Bars", no_wrap=True)
    table.add_column("Notes", no_wrap=True)
    table.add_column("Range", no_wrap=True)
    if show_guitar_meta:
        table.add_column("Tuning", no_wrap=True)
        table.add_column("Capo", no_wrap=True)
    for p in info.parts:
        pitch_span = (
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
            pitch_span,
        ]
        if show_guitar_meta:
            row.append(p.tuning or "-")
            row.append(str(p.capo) if p.capo else "-")
        table.add_row(*row)
    console.print(table)


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(
            help="Title, composer, or keywords to search (e.g. 'Moonlight Sonata', 'Bach Cello')."
        ),
    ],
    limit: Annotated[
        int, typer.Option("--limit", "-n", min=1, max=25, help="Maximum number of top results.")
    ] = 5,
    select: Annotated[
        int | None,
        typer.Option("--select", "-s", min=1, help="1-based result index to download immediately."),
    ] = None,
    output: OutputOpt = None,
    download_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Directory for downloaded file when --output is omitted."),
    ] = Path("downloads"),
    musicxml_only: Annotated[
        bool,
        typer.Option(
            "--musicxml-only",
            help="Only return native MusicXML (.musicxml/.mxl/.xml) results.",
        ),
    ] = False,
    formats: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Comma-separated extension filter (e.g. 'mxl,musicxml' or 'mid').",
        ),
    ] = None,
    instrument: Annotated[
        Instrument | None,
        typer.Option(
            "--instrument",
            "-i",
            help="Prioritize matching scores for this instrument and convert the download immediately.",
        ),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Search local corpus and cached indices without network calls."),
    ] = False,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Search symbolic score catalogs (preferring MusicXML or GP TAB) and download a pick."""
    parsed_formats = (
        [tok.strip() for tok in formats.split(",") if tok.strip()] if formats else None
    )
    results = search_digital_scores(
        query,
        limit=limit,
        instrument=instrument,
        musicxml_only=musicxml_only,
        formats=parsed_formats,
        include_remote=not offline,
        cache_dir=cache_dir,
    )
    if not results:
        console.print(f"[yellow]No matching digital scores found for {query!r}.[/yellow]")
        return

    _print_search_results(query, results)

    chosen_idx: int
    if select is not None:
        if not (1 <= select <= len(results)):
            raise typer.BadParameter(f"--select must be between 1 and {len(results)}, got {select}.")
        chosen_idx = select
    else:
        raw_choice = typer.prompt(
            f"Select result to download [1-{len(results)}] (Enter or 'q' to cancel)",
            default="",
            show_default=False,
        ).strip()
        if not raw_choice or raw_choice.lower() in {"q", "quit", "exit", "0", "n", "no"}:
            console.print("Download cancelled.")
            return
        if not raw_choice.isdigit() or not (1 <= int(raw_choice) <= len(results)):
            raise typer.BadParameter(
                f"Invalid selection {raw_choice!r}. Enter a number from 1 to {len(results)}."
            )
        chosen_idx = int(raw_choice)

    chosen = results[chosen_idx - 1]
    saved_path = download_score_result(
        chosen,
        output_path=output,
        download_dir=download_dir,
    )
    console.print(
        f"[green]Downloaded:[/green] {saved_path} "
        f"({chosen.format_ext.lstrip('.').upper()} from {chosen.source})"
    )

    if instrument is not None:
        cfg = TranscriptionConfig(instrument=instrument, cache_dir=cache_dir)
        res = convert_music(input_path=saved_path, instrument=instrument, config=cfg)
        _print_result(res)
    else:
        suggested = (
            "guitar"
            if chosen.format_ext in {".gp", ".gpx", ".gp5", ".gp4", ".gp3", ".tab"}
            else "piano"
        )
        console.print(
            f"Next: [bold]transcribe-music convert \"{saved_path}\" --instrument {suggested}[/bold]"
        )


@cache_app.command("list")
def cache_list(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max jobs to show.")] = 20,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """List cached MusicXML jobs stored in SQLite."""
    store = CatalogStore(cache_dir)
    rows = store.list_recent_jobs(limit=limit)
    if not rows:
        console.print("No cached jobs in SQLite.")
        return
    table = Table(title="SQLite MusicXML Catalog")
    for col in (
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
    ):
        table.add_column(col)
    for r in rows:
        table.add_row(
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
        )
    console.print(table)


@cache_app.command("clear")
def cache_clear(cache_dir: CacheDirOpt = Path(".transcribe_cache")) -> None:
    """Clear cached entries in SQLite."""
    store = CatalogStore(cache_dir)
    removed = store.clear_cache()
    console.print(f"Cleared {removed} job records from {store.db_path}")


def _print_pdmx_status(status: PdmxStatus) -> None:
    table = Table(title="PDMX Local Catalog")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value")
    table.add_row("PDMX Directory", str(status.pdmx_dir))
    table.add_row("PDMX.csv Present", "Yes" if status.csv_exists else "No")
    table.add_row("SQLite Index", "Yes" if status.db_exists else "No")
    table.add_row("Indexed Scores", f"{status.score_count:,}")
    table.add_row("MXL Archive/Dir", "Available" if status.mxl_available else "Not synced (--with-mxl)")
    table.add_row("MID Archive/Dir", "Available" if status.mid_available else "Not synced (--with-mid)")
    console.print(table)


@pdmx_app.command("status")
def pdmx_status(cache_dir: CacheDirOpt = Path(".transcribe_cache")) -> None:
    """Inspect local PDMX catalog and score archive status."""
    _print_pdmx_status(get_pdmx_status(cache_dir))


@pdmx_app.command("sync")
def pdmx_sync(
    pdmx_dir: Annotated[
        Path | None,
        typer.Option("--pdmx-dir", help="Custom directory containing or receiving PDMX files."),
    ] = None,
    with_mxl: Annotated[
        bool,
        typer.Option("--with-mxl", help="Download mxl.tar.gz (~1.89 GB) for offline .mxl extraction."),
    ] = False,
    with_mid: Annotated[
        bool,
        typer.Option("--with-mid", help="Download mid.tar.gz (~214 MB) for offline .mid extraction."),
    ] = False,
    extract: Annotated[
        bool,
        typer.Option("--extract", help="Unpack downloaded .tar.gz archives into PDMX directory."),
    ] = False,
    no_download: Annotated[
        bool,
        typer.Option("--no-download", help="Index an existing local PDMX.csv without downloading."),
    ] = False,
    include_license_conflicts: Annotated[
        bool,
        typer.Option("--include-license-conflicts", help="Include rows flagged with license conflicts."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download PDMX files even if already cached locally."),
    ] = False,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Sync PDMX metadata (Zenodo 15571083) into SQLite and optionally fetch score archives."""
    status = sync_pdmx(
        cache_dir=cache_dir,
        pdmx_dir=pdmx_dir,
        download_csv=not no_download,
        download_mxl=with_mxl,
        download_mid=with_mid,
        extract_archives=extract,
        include_license_conflicts=include_license_conflicts,
        force=force,
    )
    _print_pdmx_status(status)


def _print_gp_status(status: GpCatalogStatus) -> None:
    table = Table(title="DadaGP / Guitar Pro Local Catalog")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value")
    table.add_row("Catalog Directory", str(status.gp_dir))
    table.add_row("SQLite Index", "Yes" if status.db_exists else "No")
    table.add_row("Indexed Scores", f"{status.score_count:,}")
    table.add_row("Files / Archives", "Available" if status.archive_available else "Empty")
    console.print(table)


@dadagp_app.command("status")
def dadagp_status(cache_dir: CacheDirOpt = Path(".transcribe_cache")) -> None:
    """Inspect local DadaGP / Guitar Pro catalog status."""
    _print_gp_status(get_gp_catalog_status(cache_dir))


@dadagp_app.command("sync")
def dadagp_sync(
    gp_dir: Annotated[
        Path | None,
        typer.Option("--gp-dir", help="Local directory of .gp*/.tab files or .tar.gz/.zip archives."),
    ] = None,
    zenodo_record: Annotated[
        str | None,
        typer.Option("--zenodo-record", help="Optional Zenodo record ID to download archives from."),
    ] = None,
    zenodo_token: Annotated[
        str | None,
        typer.Option("--zenodo-token", help="Optional Zenodo API token (or set ZENODO_TOKEN)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download Zenodo archives even if cached locally."),
    ] = False,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Index local Guitar Pro / DadaGP files or download from Zenodo into SQLite."""
    status = sync_gp_catalog(
        cache_dir=cache_dir,
        gp_dir=gp_dir,
        zenodo_record=zenodo_record,
        zenodo_token=zenodo_token,
        force=force,
    )
    _print_gp_status(status)
