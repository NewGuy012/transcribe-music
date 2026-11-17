import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="transcribe-music Workbench")


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import duckdb
    import marimo as mo
    import verovio

    from transcribe_music import (
        Instrument,
        TranscriptionConfig,
        transcribe_audio,
        transcribe_pdf,
        transcribe_youtube,
    )
    from transcribe_music.ingest.youtube import extract_youtube_id
    from transcribe_music.models import parse_timestamp

    mo.md(
        """
        # transcribe-music Workbench
        Convert **YouTube videos**, **audio files**, or **PDF sheet music** into validated **MusicXML**
        with **notes, chord symbols, guitar TAB, and ergonomic fingerings** for Piano, Guitar, Violin, and Cello.
        """
    )
    return (
        Instrument,
        Path,
        TranscriptionConfig,
        duckdb,
        extract_youtube_id,
        mo,
        parse_timestamp,
        tempfile,
        transcribe_audio,
        transcribe_pdf,
        transcribe_youtube,
        verovio,
    )


@app.cell
def _(Instrument, mo):
    instrument_sel = mo.ui.dropdown(
        options=[i.value for i in Instrument],
        value=Instrument.PIANO.value,
        label="Instrument",
    )
    mode_sel = mo.ui.dropdown(
        options=["auto", "solo", "concerto"],
        value="auto",
        label="Ensemble Mode",
    )
    enrich_chords = mo.ui.checkbox(value=True, label="Chord Symbols (<harmony>)")
    enrich_fingering = mo.ui.checkbox(value=True, label="Fingerings / Guitar TAB")
    use_cache = mo.ui.checkbox(value=True, label="Use DuckDB Cache")
    cache_dir_input = mo.ui.text(value=".transcribe_cache", label="Cache Directory")

    yt_url = mo.ui.text(
        value="",
        placeholder="https://www.youtube.com/watch?v=...",
        label="YouTube URL",
        full_width=True,
    )
    yt_start = mo.ui.text(
        value="",
        placeholder="0:00 (optional)",
        label="Start Time (MM:SS or sec, optional)",
    )
    yt_end = mo.ui.text(
        value="",
        placeholder="End of video (optional)",
        label="End Time (MM:SS or sec, optional)",
    )

    audio_upload = mo.ui.file(
        filetypes=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        multiple=False,
        label="Upload Audio File",
    )
    pdf_upload = mo.ui.file(
        filetypes=[".pdf"],
        multiple=False,
        label="Upload Sheet Music PDF",
    )

    source_tabs = mo.ui.tabs(
        {
            "YouTube": mo.vstack([yt_url, mo.hstack([yt_start, yt_end])]),
            "Audio File": audio_upload,
            "Sheet Music PDF": pdf_upload,
        }
    )
    run_btn = mo.ui.run_button(label="Transcribe to MusicXML", kind="success")

    mo.vstack(
        [
            mo.hstack([instrument_sel, mode_sel, cache_dir_input]),
            mo.hstack([enrich_chords, enrich_fingering, use_cache]),
            source_tabs,
            run_btn,
        ]
    )
    return (
        audio_upload,
        cache_dir_input,
        enrich_chords,
        enrich_fingering,
        instrument_sel,
        mode_sel,
        pdf_upload,
        run_btn,
        source_tabs,
        use_cache,
        yt_end,
        yt_start,
        yt_url,
    )


@app.cell
def _(extract_youtube_id, mo, parse_timestamp, source_tabs, yt_start, yt_url):
    preview = None
    if source_tabs.value == "YouTube" and yt_url.value.strip():
        vid = extract_youtube_id(yt_url.value)
        start_s = 0
        if yt_start.value.strip():
            try:
                start_s = int(parse_timestamp(yt_start.value))
            except ValueError:
                start_s = 0
        if len(vid) == 11:
            preview = mo.Html(
                f'<iframe width="560" height="315" '
                f'src="https://www.youtube.com/embed/{vid}?start={start_s}" '
                f'frameborder="0" allowfullscreen></iframe>'
            )
    preview
    return (preview,)


@app.cell
def _(
    Instrument,
    Path,
    TranscriptionConfig,
    audio_upload,
    cache_dir_input,
    enrich_chords,
    enrich_fingering,
    instrument_sel,
    mo,
    mode_sel,
    pdf_upload,
    run_btn,
    source_tabs,
    tempfile,
    transcribe_audio,
    transcribe_pdf,
    transcribe_youtube,
    use_cache,
    yt_end,
    yt_start,
    yt_url,
):
    mo.stop(not run_btn.value)

    cfg = TranscriptionConfig(
        instrument=Instrument(instrument_sel.value),
        concerto_mode=mode_sel.value,
        enrich_chords=enrich_chords.value,
        enrich_fingering=enrich_fingering.value,
        use_cache=use_cache.value,
        cache_dir=Path(cache_dir_input.value or ".transcribe_cache"),
    )

    tab_name = source_tabs.value
    result = None
    if tab_name == "YouTube":
        if not yt_url.value.strip():
            raise ValueError("Please enter a YouTube URL.")
        result = transcribe_youtube(
            url=yt_url.value.strip(),
            instrument=cfg.instrument,
            start=yt_start.value.strip() or None,
            end=yt_end.value.strip() or None,
            config=cfg,
        )
    elif tab_name == "Audio File":
        if not audio_upload.value:
            raise ValueError("Please upload an audio file.")
        uploaded = audio_upload.value[0]
        raw_suffix = Path(uploaded.name).suffix.lower()
        suffix = raw_suffix if raw_suffix in {".wav", ".mp3", ".flac", ".m4a", ".ogg"} else ".wav"
        with tempfile.TemporaryDirectory(prefix="tm_ui_") as tmp_dir_str:
            tmp_file = Path(tmp_dir_str) / f"input{suffix}"
            tmp_file.write_bytes(uploaded.contents)
            result = transcribe_audio(tmp_file, instrument=cfg.instrument, config=cfg)
    else:
        if not pdf_upload.value:
            raise ValueError("Please upload a PDF file.")
        uploaded = pdf_upload.value[0]
        with tempfile.TemporaryDirectory(prefix="tm_ui_") as tmp_dir_str:
            tmp_file = Path(tmp_dir_str) / "input.pdf"
            tmp_file.write_bytes(uploaded.contents)
            result = transcribe_pdf(tmp_file, instrument=cfg.instrument, config=cfg)

    result
    return (result,)


@app.cell
def _(mo, result, verovio):
    xml_bytes = result.musicxml_path.read_bytes()
    tk = verovio.toolkit()
    tk.setOptions('{"pageWidth": 1800, "scale": 45, "adjustPageHeight": true}')
    tk.loadData(xml_bytes.decode("utf-8"))
    pages_svg = [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]

    backend_label = (
        f"{result.backend} ({result.device_used})"
        if result.backend and result.device_used
        else (result.backend or "-")
    )
    summary_md = mo.md(
        f"""
        ### Export Ready (`{result.musicxml_path.name}`)
        - **Backend:** `{backend_label}` | **Measures:** {result.measure_count} | **Notes:** {result.note_count} | **Chords:** {result.chord_count} | **Fingerings:** {result.fingering_count}
        - **Cached:** `{result.cached}` | **Elapsed:** `{result.elapsed_sec:.2f}s`
        """
    )
    download_btn = mo.download(
        data=xml_bytes,
        filename=result.musicxml_path.name,
        mimetype="application/vnd.recordare.musicxml+xml",
        label="Download MusicXML",
    )
    svg_stack = mo.vstack([mo.Html(svg) for svg in pages_svg])
    mo.vstack([summary_md, download_btn, svg_stack])
    return (download_btn, pages_svg, summary_md, svg_stack, xml_bytes)


@app.cell
def _(Path, cache_dir_input, duckdb, mo):
    db_path = Path(cache_dir_input.value or ".transcribe_cache") / "catalog.duckdb"
    catalog_view = mo.md("*No DuckDB catalog recorded yet.*")
    if db_path.exists():
        cols = (
            "job_hash",
            "source_type",
            "instrument",
            "measure_count",
            "note_count",
            "chord_count",
            "fingering_count",
            "elapsed_sec",
            "created_at",
        )
        with duckdb.connect(str(db_path), read_only=True) as con:
            rows = con.execute(
                """
                SELECT job_hash, source_type, instrument, measure_count, note_count,
                       chord_count, fingering_count, elapsed_sec, created_at
                FROM jobs
                ORDER BY created_at DESC
                LIMIT 25
                """
            ).fetchall()
        jobs_rows = [dict(zip(cols, row, strict=True)) for row in rows]
        catalog_view = mo.vstack([mo.md("### DuckDB Catalog History"), mo.ui.table(jobs_rows)])
    catalog_view
    return (catalog_view, db_path)


if __name__ == "__main__":
    app.run()
