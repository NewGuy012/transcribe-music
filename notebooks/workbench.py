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
        inspect_score,
        transcribe_pdf,
        transcribe_score,
    )
    from transcribe_music.ingest.digital import DIGITAL_EXTENSIONS

    mo.md(
        """
        # transcribe-music Workbench
        Convert **digital symbolic scores** (`.musicxml`, `.mxl`, `.mid`, `.krn`, `.mei`, `.abc`, `.gp*`, `.mscz`, `.capx`, `.ly`)
        — or **PDF sheet music** as an OMR fallback — into validated **MusicXML** with **chord symbols**,
        **guitar TAB**, and **ergonomic fingerings** for Piano, Guitar, Violin, and Cello.
        """
    )
    return (
        DIGITAL_EXTENSIONS,
        Instrument,
        Path,
        TranscriptionConfig,
        duckdb,
        inspect_score,
        mo,
        tempfile,
        transcribe_pdf,
        transcribe_score,
        verovio,
    )


@app.cell
def _(DIGITAL_EXTENSIONS, Instrument, mo):
    instrument_sel = mo.ui.dropdown(
        options=[i.value for i in Instrument],
        value=Instrument.CELLO.value,
        label="Target Instrument",
    )
    part_input = mo.ui.text(
        value="",
        placeholder="Auto (or e.g. '1', '1,2', 'cello')",
        label="Part Override (optional)",
    )
    enrich_chords = mo.ui.checkbox(value=True, label="Chord Symbols (<harmony>)")
    enrich_fingering = mo.ui.checkbox(value=True, label="Fingerings / Guitar TAB")
    preserve_chords = mo.ui.checkbox(value=False, label="Preserve Existing Chords")
    preserve_fingering = mo.ui.checkbox(value=False, label="Preserve Existing Fingerings")
    use_cache = mo.ui.checkbox(value=True, label="Use DuckDB Cache")
    cache_dir_input = mo.ui.text(value=".transcribe_cache", label="Cache Directory")

    digital_upload = mo.ui.file(
        filetypes=sorted(DIGITAL_EXTENSIONS),
        multiple=False,
        label="Upload Digital Score (.mxl, .musicxml, .mid, .krn, .mei, .abc, .gp*, .mscz, .capx, .ly)",
    )
    digital_path_input = mo.ui.text(
        value="",
        placeholder="/path/to/score.mxl (optional alternative to upload)",
        label="Or Local Digital File Path",
        full_width=True,
    )
    pdf_upload = mo.ui.file(
        filetypes=[".pdf"],
        multiple=False,
        label="Upload Sheet Music PDF (OMR Fallback)",
    )
    pdf_dpi_slider = mo.ui.slider(start=150, stop=600, step=50, value=300, label="PDF DPI")

    source_tabs = mo.ui.tabs(
        {
            "Digital Score (Primary)": mo.vstack([digital_upload, digital_path_input]),
            "Sheet Music PDF (Fallback OMR)": mo.vstack(
                [
                    mo.md(
                        "*Tip: Prefer digital `.mxl`/`.musicxml`/`.mid`/`.krn`/`.gp5` files whenever available for 100% notation accuracy.*"
                    ),
                    pdf_upload,
                    pdf_dpi_slider,
                ]
            ),
        }
    )
    run_btn = mo.ui.run_button(label="Convert to Enriched MusicXML", kind="success")

    mo.vstack(
        [
            mo.hstack([instrument_sel, part_input, cache_dir_input]),
            mo.hstack([enrich_chords, enrich_fingering, preserve_chords, preserve_fingering, use_cache]),
            source_tabs,
            run_btn,
        ]
    )
    return (
        cache_dir_input,
        digital_path_input,
        digital_upload,
        enrich_chords,
        enrich_fingering,
        instrument_sel,
        part_input,
        pdf_dpi_slider,
        pdf_upload,
        preserve_chords,
        preserve_fingering,
        run_btn,
        source_tabs,
        use_cache,
    )


@app.cell
def _(
    Path,
    digital_path_input,
    digital_upload,
    inspect_score,
    mo,
    source_tabs,
    tempfile,
):
    inspection_view = None
    if source_tabs.value == "Digital Score (Primary)":
        _inspection = None
        if digital_upload.value:
            _uploaded = digital_upload.value[0]
            _suffix = Path(_uploaded.name).suffix.lower() or ".musicxml"
            with tempfile.TemporaryDirectory(prefix="tm_insp_") as _tmp_dir_str:
                _tmp_file = Path(_tmp_dir_str) / f"preview{_suffix}"
                _tmp_file.write_bytes(_uploaded.contents)
                _inspection = inspect_score(_tmp_file)
        elif digital_path_input.value.strip():
            _p = Path(digital_path_input.value.strip())
            if _p.exists():
                _inspection = inspect_score(_p)
        if _inspection is not None:
            _part_rows = [
                {
                    "#": part.index,
                    "name": part.name,
                    "instrument": part.instrument_name or "-",
                    "midi_program": part.midi_program if part.midi_program is not None else "-",
                    "clef": part.clef or "-",
                    "measures": part.measure_count,
                    "notes": part.note_count,
                }
                for part in _inspection.parts
            ]
            inspection_view = mo.vstack(
                [
                    mo.md(f"### Detected Parts (`{_inspection.format_name}`)"),
                    mo.ui.table(_part_rows),
                ]
            )
    inspection_view
    return (inspection_view,)


@app.cell
def _(
    Instrument,
    Path,
    TranscriptionConfig,
    cache_dir_input,
    digital_path_input,
    digital_upload,
    enrich_chords,
    enrich_fingering,
    instrument_sel,
    mo,
    part_input,
    pdf_dpi_slider,
    pdf_upload,
    preserve_chords,
    preserve_fingering,
    run_btn,
    source_tabs,
    tempfile,
    transcribe_pdf,
    transcribe_score,
    use_cache,
):
    mo.stop(not run_btn.value)

    cfg = TranscriptionConfig(
        instrument=Instrument(instrument_sel.value),
        part=part_input.value.strip() or None,
        enrich_chords=enrich_chords.value,
        enrich_fingering=enrich_fingering.value,
        preserve_existing_chords=preserve_chords.value,
        preserve_existing_fingering=preserve_fingering.value,
        use_cache=use_cache.value,
        cache_dir=Path(cache_dir_input.value or ".transcribe_cache"),
        pdf_dpi=int(pdf_dpi_slider.value),
    )

    if source_tabs.value == "Digital Score (Primary)":
        if digital_upload.value:
            _uploaded = digital_upload.value[0]
            _suffix = Path(_uploaded.name).suffix.lower() or ".musicxml"
            with tempfile.TemporaryDirectory(prefix="tm_ui_") as _tmp_dir_str:
                _tmp_file = Path(_tmp_dir_str) / f"{Path(_uploaded.name).stem}{_suffix}"
                _tmp_file.write_bytes(_uploaded.contents)
                result = transcribe_score(_tmp_file, instrument=cfg.instrument, config=cfg)
        elif digital_path_input.value.strip():
            result = transcribe_score(
                Path(digital_path_input.value.strip()), instrument=cfg.instrument, config=cfg
            )
        else:
            raise ValueError("Please upload a digital score file or enter a local file path.")
    else:
        if not pdf_upload.value:
            raise ValueError("Please upload a PDF file.")
        _uploaded = pdf_upload.value[0]
        with tempfile.TemporaryDirectory(prefix="tm_ui_") as _tmp_dir_str:
            _tmp_file = Path(_tmp_dir_str) / f"{Path(_uploaded.name).stem}.pdf"
            _tmp_file.write_bytes(_uploaded.contents)
            result = transcribe_pdf(_tmp_file, instrument=cfg.instrument, config=cfg)

    result
    return (result,)


@app.cell
def _(mo, result, verovio):
    xml_bytes = result.musicxml_path.read_bytes()
    tk = verovio.toolkit()
    tk.setOptions('{"pageWidth": 1800, "scale": 45, "adjustPageHeight": true}')
    tk.loadData(xml_bytes.decode("utf-8"))
    pages_svg = [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]

    part_label = result.selected_part or "-"
    summary_md = mo.md(
        f"""
        ### Export Ready (`{result.musicxml_path.name}`)
        - **Backend:** `{result.backend or "-"}` | **Part:** `{part_label}` | **Measures:** {result.measure_count} | **Notes:** {result.note_count} | **Chords:** {result.chord_count} | **Fingerings:** {result.fingering_count}
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
            "backend",
            "selected_part",
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
                SELECT job_hash, source_type, instrument, backend, selected_part,
                       measure_count, note_count, chord_count, fingering_count,
                       elapsed_sec, created_at
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
