"""End-to-end digital score conversion, PDF fallback OMR, enrichment, caching, and MusicXML export."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.digital import load_digital_score
from transcribe_music.models import (
    Instrument,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.storage.catalog import CatalogStore, compute_job_hash, hash_file
from transcribe_music.transcribe.omr_pdf import run_pdf_omr
from transcribe_music.transcribe.part_selector import select_and_adapt_parts


def _default_output_path(default_stem: str, instrument: Instrument, artifact_dir: Path) -> Path:
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in default_stem)
    return artifact_dir / f"{safe_stem}_{instrument.value}.musicxml"


def _copy_if_needed(src: Path, dst: Path) -> Path:
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def _produce_enrich_and_export(
    input_path: Path,
    source_type: SourceType,
    job_hash: str,
    cfg: TranscriptionConfig,
    work_dir: Path,
    export_path: Path,
    store: CatalogStore | None,
    t_start: float,
) -> TranscriptionResult:
    if source_type == SourceType.PDF:
        raw_score, backend, load_warnings = run_pdf_omr(
            input_path,
            work_dir,
            instrument=cfg.instrument,
            dpi=cfg.pdf_dpi,
            omr_backend=cfg.omr_backend,
        )
        is_sounding = False
    else:
        loaded = load_digital_score(input_path, quantize_divisors=cfg.quantize_divisors)
        raw_score = loaded.score
        backend = loaded.backend
        load_warnings = loaded.warnings
        is_sounding = loaded.is_sounding_pitch_source

    selected = select_and_adapt_parts(
        raw_score,
        instrument=cfg.instrument,
        part_selector=cfg.part,
        is_sounding_pitch_source=is_sounding,
    )
    score = selected.score
    warnings = [*load_warnings, *selected.warnings]

    if cfg.enrich_chords:
        try:
            annotate_chords(
                score,
                harmonic_reference_score=raw_score,
                preserve_existing=cfg.preserve_existing_chords,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment fallback
            warnings.append(f"Chord enrichment skipped safely: {exc}")

    if cfg.enrich_fingering:
        try:
            apply_fingering(
                score,
                cfg.instrument,
                preserve_existing=cfg.preserve_existing_fingering,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment fallback
            warnings.append(f"Fingering enrichment skipped safely: {exc}")

    existing_title = score.metadata.title.strip() if score.metadata and score.metadata.title else ""
    metrics, export_warnings = export_musicxml(
        score,
        output_path=export_path,
        instrument=cfg.instrument,
        title=existing_title or input_path.stem,
    )
    warnings.extend(export_warnings)

    result = TranscriptionResult(
        job_hash=job_hash,
        source_type=source_type,
        source_ref=str(input_path),
        instrument=cfg.instrument,
        musicxml_path=export_path,
        cached=False,
        elapsed_sec=round(time.perf_counter() - t_start, 3),
        measure_count=metrics.measure_count,
        note_count=metrics.note_count,
        chord_count=metrics.chord_count,
        fingering_count=metrics.fingering_count,
        backend=backend,
        selected_part=selected.selected_part,
        warnings=warnings,
    )
    if store is not None:
        store.record_job(result=result, config=cfg)
    return result


def convert_music(
    input_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Convert a digital symbolic score (primary) or PDF sheet scan (fallback) to MusicXML."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    t_start = time.perf_counter()
    source_type = SourceType.PDF if path.suffix.lower() == ".pdf" else SourceType.DIGITAL
    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = cfg.model_copy(update={"instrument": inst})

    job_hash = compute_job_hash(source_type, hash_file(path), cfg)

    if cfg.use_cache:
        store = CatalogStore(cfg.cache_dir)
        artifact_dir = store.artifact_dir(job_hash)
        canonical_out = _default_output_path(path.stem, inst, artifact_dir)
        final_out = Path(output_path) if output_path is not None else canonical_out

        cached = store.get_cached_job(job_hash)
        if cached is not None:
            _copy_if_needed(cached.musicxml_path, final_out)
            return cached.model_copy(
                update={
                    "musicxml_path": final_out,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                }
            )

        result = _produce_enrich_and_export(
            path, source_type, job_hash, cfg, artifact_dir, canonical_out, store, t_start
        )
        _copy_if_needed(canonical_out, final_out)
        return result.model_copy(update={"musicxml_path": final_out})

    if output_path is not None:
        with tempfile.TemporaryDirectory(prefix="tm_work_") as tmp_dir:
            return _produce_enrich_and_export(
                path, source_type, job_hash, cfg, Path(tmp_dir), Path(output_path), None, t_start
            )

    artifact_dir = cfg.cache_dir / "artifacts" / job_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return _produce_enrich_and_export(
        path,
        source_type,
        job_hash,
        cfg,
        artifact_dir,
        _default_output_path(path.stem, inst, artifact_dir),
        None,
        t_start,
    )
