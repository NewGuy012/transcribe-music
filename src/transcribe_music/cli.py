"""Typer CLI entrypoint for transcribe-music."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

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


def _print_result(res: TranscriptionResult) -> None:
    table = Table(title="Transcription Complete", show_header=False)
    table.add_row("Output MusicXML", str(res.musicxml_path))
    table.add_row("Instrument", res.instrument.value)
    table.add_row("Source", f"{res.source_type.value}: {res.source_ref}")
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
    instrument: Instrument,
    mode: Literal["auto", "solo", "concerto"],
    device: Literal["auto", "mps", "cpu"],
    chords: bool,
    fingering: bool,
    cache: bool,
    tempo: float | None,
    cache_dir: Path,
    dpi: int = 300,
) -> TranscriptionConfig:
    return TranscriptionConfig(
        instrument=instrument,
        concerto_mode=mode,
        device=device,
        enrich_chords=chords,
        enrich_fingering=fingering,
        use_cache=cache,
        tempo_bpm=tempo,
        cache_dir=cache_dir,
        pdf_dpi=dpi,
    )


@app.command()
def youtube(
    url: str = typer.Argument(..., help="YouTube video URL."),
    start: str | None = typer.Option(
        None,
        "--start",
        "-s",
        help="Optional start timestamp (e.g. '0:42' or '42'). Defaults to start of video.",
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        "-e",
        help="Optional end timestamp (e.g. '2:15' or '135'). Defaults to end of video.",
    ),
    instrument: Instrument = typer.Option(..., "--instrument", "-i", help="Target instrument."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output .musicxml path."),
    mode: Literal["auto", "solo", "concerto"] = typer.Option("auto", "--mode", help="Solo vs concerto handling."),
    device: Literal["auto", "mps", "cpu"] = typer.Option("auto", "--device", help="Torch compute device."),
    chords: bool = typer.Option(True, "--chords/--no-chords", help="Annotate chord symbols."),
    fingering: bool = typer.Option(True, "--fingering/--no-fingering", help="Annotate fingerings/TAB."),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use DuckDB cache."),
    tempo: float | None = typer.Option(None, "--tempo", help="Override BPM instead of auto beat tracking."),
    cache_dir: Path = typer.Option(Path(".transcribe_cache"), "--cache-dir", help="Cache directory."),
) -> None:
    """Download a YouTube video (or optional start/end slice) and transcribe it to MusicXML."""
    cfg = _build_config(instrument, mode, device, chords, fingering, cache, tempo, cache_dir)
    res = transcribe_youtube(url=url, instrument=instrument, start=start, end=end, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def audio(
    audio_file: Path = typer.Argument(..., exists=True, readable=True, help="Local audio file (WAV/MP3/FLAC/M4A)."),
    instrument: Instrument = typer.Option(..., "--instrument", "-i", help="Target instrument."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output .musicxml path."),
    mode: Literal["auto", "solo", "concerto"] = typer.Option("auto", "--mode", help="Solo vs concerto handling."),
    device: Literal["auto", "mps", "cpu"] = typer.Option("auto", "--device", help="Torch compute device."),
    chords: bool = typer.Option(True, "--chords/--no-chords", help="Annotate chord symbols."),
    fingering: bool = typer.Option(True, "--fingering/--no-fingering", help="Annotate fingerings/TAB."),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use DuckDB cache."),
    tempo: float | None = typer.Option(None, "--tempo", help="Override BPM instead of auto beat tracking."),
    cache_dir: Path = typer.Option(Path(".transcribe_cache"), "--cache-dir", help="Cache directory."),
) -> None:
    """Transcribe a full local audio file into MusicXML."""
    cfg = _build_config(instrument, mode, device, chords, fingering, cache, tempo, cache_dir)
    res = transcribe_audio(audio_path=audio_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def pdf(
    pdf_file: Path = typer.Argument(..., exists=True, readable=True, help="Sheet music .pdf file."),
    instrument: Instrument = typer.Option(..., "--instrument", "-i", help="Target instrument."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output .musicxml path."),
    dpi: int = typer.Option(300, "--dpi", help="PDF rasterization DPI for OMR."),
    chords: bool = typer.Option(True, "--chords/--no-chords", help="Annotate chord symbols."),
    fingering: bool = typer.Option(True, "--fingering/--no-fingering", help="Annotate fingerings/TAB."),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use DuckDB cache."),
    cache_dir: Path = typer.Option(Path(".transcribe_cache"), "--cache-dir", help="Cache directory."),
) -> None:
    """Run OMR on a sheet music PDF and enrich with chords and fingerings."""
    cfg = _build_config(instrument, "auto", "auto", chords, fingering, cache, None, cache_dir, dpi=dpi)
    res = transcribe_pdf(pdf_path=pdf_file, instrument=instrument, output_path=output, config=cfg)
    _print_result(res)


@app.command()
def ui(
    port: int = typer.Option(2718, "--port", help="Port for Marimo workbench."),
    edit: bool = typer.Option(False, "--edit", help="Launch in Marimo notebook edit mode."),
) -> None:
    """Launch the interactive Marimo + Verovio + DuckDB workbench."""
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "workbench.py"
    if not notebook.exists():
        notebook = Path.cwd() / "notebooks" / "workbench.py"
    if not notebook.exists():
        raise typer.BadParameter(f"Workbench notebook not found at {notebook}")
    mode_cmd = "edit" if edit else "run"
    subprocess.run([sys.executable, "-m", "marimo", mode_cmd, str(notebook), "--port", str(port)], check=True)


@cache_app.command("list")
def cache_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max jobs to show."),
    cache_dir: Path = typer.Option(Path(".transcribe_cache"), "--cache-dir", help="Cache directory."),
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
def cache_clear(
    cache_dir: Path = typer.Option(Path(".transcribe_cache"), "--cache-dir", help="Cache directory."),
) -> None:
    """Clear cached entries in DuckDB."""
    store = CatalogStore(cache_dir)
    removed = store.clear_cache()
    console.print(f"Cleared {removed} job records from {store.db_path}")
