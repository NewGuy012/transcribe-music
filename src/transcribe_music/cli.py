"""Typer CLI entrypoint for transcribe-music."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from transcribe_music.models import Instrument, TranscriptionConfig, TranscriptionResult
from transcribe_music.pipeline import (
    convert_music,
    inspect_score,
    transcribe_pdf,
    transcribe_score,
)
from transcribe_music.storage.catalog import CatalogStore

app = typer.Typer(
    name="transcribe-music",
    help="Convert digital symbolic scores (primary) or sheet music PDFs (fallback) into enriched MusicXML.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Inspect or manage the DuckDB conversion cache.")
app.add_typer(cache_app, name="cache")

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
    typer.Option("--preserve-fingering", help="Keep authored fingerings/TAB when present."),
]
CacheOpt = Annotated[bool, typer.Option("--cache/--no-cache", help="Use DuckDB cache.")]
CacheDirOpt = Annotated[Path, typer.Option("--cache-dir", help="Cache directory.")]


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


def _build_config(
    *,
    instrument: Instrument,
    part: str | None,
    chords: bool,
    fingering: bool,
    preserve_chords: bool,
    preserve_fingering: bool,
    cache: bool,
    cache_dir: Path,
    dpi: int = 300,
) -> TranscriptionConfig:
    return TranscriptionConfig(
        instrument=instrument,
        part=part,
        enrich_chords=chords,
        enrich_fingering=fingering,
        preserve_existing_chords=preserve_chords,
        preserve_existing_fingering=preserve_fingering,
        use_cache=cache,
        cache_dir=cache_dir,
        pdf_dpi=dpi,
    )


@app.command()
def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Digital score (.musicxml/.mxl/.mid/.krn/.mei/.abc/.gp3-.gp5/.mscz) or fallback .pdf.",
        ),
    ],
    instrument: InstrumentOpt,
    output: OutputOpt = None,
    part: PartOpt = None,
    dpi: Annotated[int, typer.Option("--dpi", help="PDF rasterization DPI (for .pdf fallback only).")] = 300,
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    preserve_chords: PreserveChordsOpt = False,
    preserve_fingering: PreserveFingeringOpt = False,
    cache: CacheOpt = True,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Convert any supported digital score (primary) or sheet music PDF (fallback) to MusicXML."""
    cfg = _build_config(
        instrument=instrument,
        part=part,
        chords=chords,
        fingering=fingering,
        preserve_chords=preserve_chords,
        preserve_fingering=preserve_fingering,
        cache=cache,
        cache_dir=cache_dir,
        dpi=dpi,
    )
    res = convert_music(input_path=input_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def score(
    score_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Digital score (.musicxml/.xml/.mxl/.mid/.midi/.krn/.mei/.abc/.gp3-.gp5/.mscz/.mscx).",
        ),
    ],
    instrument: InstrumentOpt,
    output: OutputOpt = None,
    part: PartOpt = None,
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    preserve_chords: PreserveChordsOpt = False,
    preserve_fingering: PreserveFingeringOpt = False,
    cache: CacheOpt = True,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Convert and enrich a digital symbolic score file into MusicXML."""
    cfg = _build_config(
        instrument=instrument,
        part=part,
        chords=chords,
        fingering=fingering,
        preserve_chords=preserve_chords,
        preserve_fingering=preserve_fingering,
        cache=cache,
        cache_dir=cache_dir,
    )
    res = transcribe_score(score_path=score_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def pdf(
    pdf_file: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Sheet music .pdf file (OMR fallback).")
    ],
    instrument: InstrumentOpt,
    output: OutputOpt = None,
    part: PartOpt = None,
    dpi: Annotated[int, typer.Option("--dpi", help="PDF rasterization DPI for OMR.")] = 300,
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    preserve_chords: PreserveChordsOpt = False,
    preserve_fingering: PreserveFingeringOpt = False,
    cache: CacheOpt = True,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Run fallback OMR on a sheet music PDF and enrich with chords and fingerings."""
    cfg = _build_config(
        instrument=instrument,
        part=part,
        chords=chords,
        fingering=fingering,
        preserve_chords=preserve_chords,
        preserve_fingering=preserve_fingering,
        cache=cache,
        cache_dir=cache_dir,
        dpi=dpi,
    )
    res = transcribe_pdf(pdf_path=pdf_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def inspect(
    score_file: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Digital score file to inspect."),
    ],
) -> None:
    """List all parts, instruments, clefs, and note counts in a digital score file."""
    info = inspect_score(score_file)
    table = Table(title=f"{info.title or info.source_path.name} ({info.format_name})")
    table.add_column("#", no_wrap=True)
    table.add_column("Part", no_wrap=True)
    table.add_column("Instrument")
    table.add_column("GM", no_wrap=True)
    table.add_column("Clef")
    table.add_column("Bars", no_wrap=True)
    table.add_column("Notes", no_wrap=True)
    table.add_column("Range", no_wrap=True)
    for p in info.parts:
        pitch_span = (
            f"{p.pitch_min}..{p.pitch_max}"
            if p.pitch_min is not None and p.pitch_max is not None
            else "-"
        )
        table.add_row(
            str(p.index),
            p.name,
            p.instrument_name or "-",
            str(p.midi_program) if p.midi_program is not None else "-",
            p.clef or "-",
            str(p.measure_count),
            str(p.note_count),
            pitch_span,
        )
    console.print(table)


@app.command()
def ui(
    port: Annotated[int, typer.Option("--port", help="Port for Marimo workbench.")] = 2718,
    edit: Annotated[bool, typer.Option("--edit", help="Launch in Marimo notebook edit mode.")] = False,
) -> None:
    """Launch the interactive Marimo + Verovio + DuckDB workbench."""
    missing = [pkg for pkg in ("marimo", "verovio") if importlib.util.find_spec(pkg) is None]
    if missing:
        console.print(
            f"[red]Missing UI dependencies ({', '.join(missing)}).[/red] "
            "Install them with: [bold]uv sync --extra ui[/bold]"
        )
        raise typer.Exit(code=1)
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "workbench.py"
    if not notebook.exists():
        notebook = Path.cwd() / "notebooks" / "workbench.py"
    if not notebook.exists():
        raise typer.BadParameter(f"Workbench notebook not found at {notebook}")
    mode_cmd = "edit" if edit else "run"
    try:
        subprocess.run(
            [sys.executable, "-m", "marimo", mode_cmd, str(notebook), "--port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        pass


@cache_app.command("list")
def cache_list(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max jobs to show.")] = 20,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """List cached MusicXML jobs stored in DuckDB."""
    store = CatalogStore(cache_dir)
    rows = store.list_recent_jobs(limit=limit)
    if not rows:
        console.print("No cached jobs in DuckDB.")
        return
    table = Table(title="DuckDB MusicXML Catalog")
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
    """Clear cached entries in DuckDB."""
    store = CatalogStore(cache_dir)
    removed = store.clear_cache()
    console.print(f"Cleared {removed} job records from {store.db_path}")
