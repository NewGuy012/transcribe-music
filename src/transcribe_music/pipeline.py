"""End-to-end digital score conversion, PDF fallback OMR, enrichment, caching, and MusicXML export."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import music21

from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.digital import inspect_digital_score, load_digital_score
from transcribe_music.models import (
    Instrument,
    ScoreInspection,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.storage.catalog import CatalogStore, compute_job_hash, hash_file
from transcribe_music.transcribe.omr_pdf import transcribe_pdf_omr
from transcribe_music.transcribe.part_selector import select_and_adapt_parts


def _resolve_output_path(
    output_path: Path | None,
    default_stem: str,
    instrument: Instrument,
    artifact_dir: Path,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in default_stem)
    return artifact_dir / f"{safe_stem}_{instrument.value}.musicxml"


def _copy_if_needed(src: Path, dst: Path) -> Path:
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


@dataclass(frozen=True)
class _ProducedScore:
    score: music21.stream.Score
    harmonic_reference_score: music21.stream.Score | None
    backend: str
    selected_part: str | None
    warnings: list[str] = field(default_factory=list)


def _enrich_and_export(
    produced: _ProducedScore,
    config: TranscriptionConfig,
    source_type: SourceType,
    source_ref: str,
    job_hash: str,
    output_path: Path,
    store: CatalogStore | None,
    t_start: float,
    fallback_title: str,
) -> TranscriptionResult:
    score = produced.score
    warnings = list(produced.warnings)

    if config.enrich_chords:
        try:
            annotate_chords(
                score,
                harmonic_reference_score=produced.harmonic_reference_score,
                preserve_existing=config.preserve_existing_chords,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment fallback
            warnings.append(f"Chord enrichment skipped safely: {exc}")

    if config.enrich_fingering:
        try:
            apply_fingering(
                score,
                config.instrument,
                preserve_existing=config.preserve_existing_fingering,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment fallback
            warnings.append(f"Fingering enrichment skipped safely: {exc}")

    existing_title = score.metadata.title.strip() if score.metadata and score.metadata.title else ""
    final_title = existing_title or fallback_title

    metrics, export_warnings = export_musicxml(
        score,
        output_path=output_path,
        instrument=config.instrument,
        title=final_title,
    )
    warnings.extend(export_warnings)

    elapsed = round(time.perf_counter() - t_start, 3)
    result = TranscriptionResult(
        job_hash=job_hash,
        source_type=source_type,
        source_ref=source_ref,
        instrument=config.instrument,
        musicxml_path=output_path,
        cached=False,
        elapsed_sec=elapsed,
        measure_count=metrics.measure_count,
        note_count=metrics.note_count,
        chord_count=metrics.chord_count,
        fingering_count=metrics.fingering_count,
        backend=produced.backend,
        selected_part=produced.selected_part,
        warnings=warnings,
    )

    if store is not None:
        store.record_job(result=result, config=config)
    return result


def _require_existing_file(path: Path, expected_suffix: str | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if expected_suffix and path.suffix.lower() != expected_suffix:
        raise ValueError(f"Expected a {expected_suffix} file, got: {path.suffix}")


def _run_cached_job(
    source_type: SourceType,
    source_identity: str,
    source_ref: str,
    instrument: Instrument | str,
    default_stem: str,
    output_path: Path | None,
    config: TranscriptionConfig | None,
    produce_score: Callable[[Path, TranscriptionConfig], _ProducedScore],
    *,
    fallback_title: str,
) -> TranscriptionResult:
    t_start = time.perf_counter()
    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = cfg.model_copy(update={"instrument": inst})

    job_hash = compute_job_hash(source_type, source_identity, cfg)

    if cfg.use_cache:
        store = CatalogStore(cfg.cache_dir)
        artifact_dir = store.artifact_dir(job_hash)
        final_out = _resolve_output_path(output_path, default_stem, inst, artifact_dir)
        cached = store.get_cached_job(job_hash)
        if cached is not None:
            _copy_if_needed(cached.musicxml_path, final_out)
            return cached.model_copy(
                update={
                    "musicxml_path": final_out,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                }
            )
        produced = produce_score(artifact_dir, cfg)
        return _enrich_and_export(
            produced=produced,
            config=cfg,
            source_type=source_type,
            source_ref=source_ref,
            job_hash=job_hash,
            output_path=final_out,
            store=store,
            t_start=t_start,
            fallback_title=fallback_title,
        )

    if output_path is not None:
        with tempfile.TemporaryDirectory(prefix="tm_work_") as tmp_dir_str:
            work_dir = Path(tmp_dir_str)
            produced = produce_score(work_dir, cfg)
            return _enrich_and_export(
                produced=produced,
                config=cfg,
                source_type=source_type,
                source_ref=source_ref,
                job_hash=job_hash,
                output_path=Path(output_path),
                store=None,
                t_start=t_start,
                fallback_title=fallback_title,
            )

    artifact_dir = cfg.cache_dir / "artifacts" / job_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_out = _resolve_output_path(None, default_stem, inst, artifact_dir)
    produced = produce_score(artifact_dir, cfg)
    return _enrich_and_export(
        produced=produced,
        config=cfg,
        source_type=source_type,
        source_ref=source_ref,
        job_hash=job_hash,
        output_path=final_out,
        store=None,
        t_start=t_start,
        fallback_title=fallback_title,
    )


def inspect_score(score_path: Path | str) -> ScoreInspection:
    """Inspect parts, instruments, clefs, and pitch ranges inside a digital score file."""
    return inspect_digital_score(score_path)


def transcribe_score(
    score_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Convert and enrich a digital symbolic score (.musicxml, .mxl, .mid, .krn, .mei, .abc, .gp3-.gp5, .mscz)."""
    path = Path(score_path)
    _require_existing_file(path)

    def _produce(_artifact_dir: Path, cfg: TranscriptionConfig) -> _ProducedScore:
        loaded = load_digital_score(path, quantize_divisors=cfg.quantize_divisors)
        selected = select_and_adapt_parts(
            loaded.score,
            instrument=cfg.instrument,
            part_selector=cfg.part,
            is_sounding_pitch_source=loaded.is_sounding_pitch_source,
        )
        return _ProducedScore(
            score=selected.score,
            harmonic_reference_score=loaded.score,
            backend=loaded.backend,
            selected_part=selected.selected_part,
            warnings=[*loaded.warnings, *selected.warnings],
        )

    return _run_cached_job(
        SourceType.DIGITAL,
        hash_file(path),
        str(path),
        instrument,
        path.stem,
        output_path,
        config,
        _produce,
        fallback_title=path.stem,
    )


def transcribe_pdf(
    pdf_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Fallback OMR on a sheet music PDF followed by part selection, chords, and fingerings."""
    path = Path(pdf_path)
    _require_existing_file(path, expected_suffix=".pdf")

    def _produce(artifact_dir: Path, cfg: TranscriptionConfig) -> _ProducedScore:
        raw_score = transcribe_pdf_omr(
            path, artifact_dir, instrument=cfg.instrument, dpi=cfg.pdf_dpi
        )
        selected = select_and_adapt_parts(
            raw_score,
            instrument=cfg.instrument,
            part_selector=cfg.part,
            is_sounding_pitch_source=False,
        )
        return _ProducedScore(
            score=selected.score,
            harmonic_reference_score=raw_score,
            backend="homr",
            selected_part=selected.selected_part,
            warnings=selected.warnings,
        )

    return _run_cached_job(
        SourceType.PDF,
        hash_file(path),
        str(path),
        instrument,
        path.stem,
        output_path,
        config,
        _produce,
        fallback_title=path.stem,
    )


def convert_music(
    input_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Auto-route a digital symbolic file (primary) or PDF sheet scan (fallback) to MusicXML."""
    path = Path(input_path)
    _require_existing_file(path)
    if path.suffix.lower() == ".pdf":
        return transcribe_pdf(path, instrument=instrument, output_path=output_path, config=config)
    return transcribe_score(path, instrument=instrument, output_path=output_path, config=config)
