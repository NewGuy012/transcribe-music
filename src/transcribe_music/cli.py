"""Typer CLI entrypoint for transcribe-music."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from transcribe_music.models import Instrument, TranscriptionConfig, TranscriptionResult
from transcribe_music.pipeline import transcribe_audio, transcribe_pdf, transcribe_youtube
from transcribe_music.storage.catalog import CatalogStore

app = typer.Typer(
    name="transcribe-music",
    help="Transcribe YouTube videos, audio files, or sheet music PDFs into enriched MusicXML.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Inspect or manage the DuckDB transcription cache.")
app.add_typer(cache_app, name="cache")

console = Console()

InstrumentOpt = Annotated[Instrument, typer.Option(..., "--instrument", "-i", help="Target instrument.")]
OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Output .musicxml path.")]
ModeOpt = Annotated[Literal["auto", "solo", "concerto"], typer.Option("--mode", help="Solo vs concerto handling.")]
DeviceOpt = Annotated[Literal["auto", "mps", "cpu"], typer.Option("--device", help="Torch compute device.")]
ChordsOpt = Annotated[bool, typer.Option("--chords/--no-chords", help="Annotate chord symbols.")]
FingeringOpt = Annotated[bool, typer.Option("--fingering/--no-fingering", help="Annotate fingerings/TAB.")]
CacheOpt = Annotated[bool, typer.Option("--cache/--no-cache", help="Use DuckDB cache.")]
TempoOpt = Annotated[float | None, typer.Option("--tempo", help="Override BPM instead of auto beat tracking.")]
CacheDirOpt = Annotated[Path, typer.Option("--cache-dir", help="Cache directory.")]


def _format_backend(backend: str | None, device_used: str | None) -> str:
    if not backend:
        return "-"
    return f"{backend} ({device_used})" if device_used else backend


def _print_result(res: TranscriptionResult) -> None:
    table = Table(title="Transcription Complete", show_header=False)
    table.add_row("Output MusicXML", str(res.musicxml_path))
    table.add_row("Instrument", res.instrument.value)
    table.add_row("Source", f"{res.source_type.value}: {res.source_ref}")
    if res.backend:
        table.add_row("Backend", _format_backend(res.backend, res.device_used))
    table.add_row("Cache Hit", "Yes" if res.cached else "No")
    table.add_row("Elapsed", f"{res.elapsed_sec:.2f}s")
    table.add_row("Measures", str(res.measure_count))
    table.add_row("Notes", str(res.note_count))
    table.add_row("Chords (<harmony>)", str(res.chord_count))
    table.add_row("Fingerings", str(res.fingering_count))
    console.print(table)
    for warn in res.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warn}")


@app.command()
def youtube(
    url: Annotated[str, typer.Argument(help="YouTube video URL.")],
    instrument: InstrumentOpt,
    start: Annotated[
        str | None,
        typer.Option("--start", "-s", help="Optional start timestamp (e.g. '0:42' or '42')."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", "-e", help="Optional end timestamp (e.g. '2:15' or '135')."),
    ] = None,
    output: OutputOpt = None,
    mode: ModeOpt = "auto",
    device: DeviceOpt = "auto",
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    cache: CacheOpt = True,
    tempo: TempoOpt = None,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Download a YouTube video (or optional start/end slice) and transcribe it to MusicXML."""
    cfg = TranscriptionConfig(
        instrument=instrument,
        concerto_mode=mode,
        device=device,
        enrich_chords=chords,
        enrich_fingering=fingering,
        use_cache=cache,
        tempo_bpm=tempo,
        cache_dir=cache_dir,
    )
    res = transcribe_youtube(url=url, instrument=instrument, start=start, end=end, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def audio(
    audio_file: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Local audio file (WAV/MP3/FLAC/M4A).")
    ],
    instrument: InstrumentOpt,
    output: OutputOpt = None,
    mode: ModeOpt = "auto",
    device: DeviceOpt = "auto",
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    cache: CacheOpt = True,
    tempo: TempoOpt = None,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Transcribe a full local audio file into MusicXML."""
    cfg = TranscriptionConfig(
        instrument=instrument,
        concerto_mode=mode,
        device=device,
        enrich_chords=chords,
        enrich_fingering=fingering,
        use_cache=cache,
        tempo_bpm=tempo,
        cache_dir=cache_dir,
    )
    res = transcribe_audio(audio_path=audio_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def pdf(
    pdf_file: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Sheet music .pdf file.")
    ],
    instrument: InstrumentOpt,
    output: OutputOpt = None,
    dpi: Annotated[int, typer.Option("--dpi", help="PDF rasterization DPI for OMR.")] = 300,
    chords: ChordsOpt = True,
    fingering: FingeringOpt = True,
    cache: CacheOpt = True,
    cache_dir: CacheDirOpt = Path(".transcribe_cache"),
) -> None:
    """Run OMR on a sheet music PDF and enrich with chords and fingerings."""
    cfg = TranscriptionConfig(
        instrument=instrument,
        enrich_chords=chords,
        enrich_fingering=fingering,
        use_cache=cache,
        cache_dir=cache_dir,
        pdf_dpi=dpi,
    )
    res = transcribe_pdf(pdf_path=pdf_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


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
    """List cached transcription jobs stored in DuckDB."""
    store = CatalogStore(cache_dir)
    rows = store.list_recent_jobs(limit=limit)
    if not rows:
        console.print("No cached jobs in DuckDB.")
        return
    table = Table(title="DuckDB Transcription Catalog")
    for col in ("job_hash", "source_type", "instrument", "measures", "notes", "chords", "fingerings", "elapsed_s"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["job_hash"])[:10],
            str(r["source_type"]),
            str(r["instrument"]),
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
