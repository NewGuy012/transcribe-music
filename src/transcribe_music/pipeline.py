"""End-to-end digital score conversion, PDF fallback OMR, enrichment, caching, and MusicXML export."""

from __future__ import annotations

import copy
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import music21

from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import (
    export_musicxml,
    format_score_title,
    looks_like_raw_filename,
)
from transcribe_music.ingest._io import copy_file_safe
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


def _opening_container(part: music21.stream.Part) -> music21.stream.Stream:
    return part.getElementsByClass(music21.stream.Measure).first() or part


def _refine_initial_key_mode(
    score: music21.stream.Score,
    reference_score: music21.stream.Score,
    *,
    is_midi: bool,
) -> None:
    """Disambiguate relative major/minor <mode> when untyped or MIDI-defaulted, preserving fifths."""
    parts = list(score.parts)
    if not parts:
        return

    first_ks = next(
        (
            ks
            for part in parts
            if (ks := _opening_container(part).getElementsByClass(music21.key.KeySignature).first())
            is not None
        ),
        None,
    )
    if first_ks is None or (isinstance(first_ks, music21.key.Key) and not is_midi):
        return

    try:
        analyzed = reference_score.analyze("key")
    except Exception:  # noqa: BLE001 — non-fatal heuristic
        return
    if (
        not isinstance(analyzed, music21.key.Key)
        or analyzed.sharps != first_ks.sharps
        or float(getattr(analyzed, "correlationCoefficient", 0.0) or 0.0) < 0.70
        or (isinstance(first_ks, music21.key.Key) and first_ks.mode == analyzed.mode)
    ):
        return

    for part in parts:
        container = _opening_container(part)
        for ks in list(container.getElementsByClass(music21.key.KeySignature)):
            if ks.sharps == analyzed.sharps:
                off = ks.offset
                container.remove(ks)
                container.insert(off, copy.deepcopy(analyzed))


def _resolve_export_title(
    score: music21.stream.Score,
    input_path: Path,
    override_title: str | None,
) -> str:
    if override_title and override_title.strip():
        return override_title.strip()
    existing = score.metadata.title.strip() if score.metadata and score.metadata.title else ""
    if existing and existing not in {input_path.name, input_path.stem} and not looks_like_raw_filename(existing):
        return existing
    return format_score_title(existing or input_path.stem)


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
                window_beats=cfg.chord_window_beats,
                preserve_existing=cfg.preserve_existing_chords,
                instrument=cfg.instrument,
                include_slash_chords=cfg.include_slash_chords,
                roman_numerals=cfg.roman_numerals,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal fallback
            warnings.append(f"Chord enrichment skipped safely: {exc}")

    if cfg.enrich_fingering:
        try:
            apply_fingering(
                score,
                cfg.instrument,
                preserve_existing=cfg.preserve_existing_fingering,
                preserve_existing_tab=cfg.preserve_existing_tab,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal fallback
            warnings.append(f"Fingering enrichment skipped safely: {exc}")

    _refine_initial_key_mode(score, raw_score, is_midi=(backend == "midi"))

    resolved_title = _resolve_export_title(score, input_path, cfg.title)
    resolved_composer = cfg.composer.strip() if cfg.composer and cfg.composer.strip() else None
    metrics, export_warnings = export_musicxml(
        score,
        output_path=export_path,
        instrument=cfg.instrument,
        title=resolved_title,
        composer=resolved_composer,
        guitar_staff=cfg.guitar_staff,
        repair_overflows=(source_type == SourceType.PDF),
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
    if not path.is_file():
        raise IsADirectoryError(f"Input path is not a file: {path}")

    t_start = time.perf_counter()
    source_type = SourceType.PDF if path.suffix.lower() == ".pdf" else SourceType.DIGITAL
    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = replace(cfg, instrument=inst)

    job_hash = compute_job_hash(source_type, hash_file(path), cfg)

    store = CatalogStore(cfg.cache_dir) if cfg.use_cache else None
    if store is not None and (cached := store.get_cached_job(job_hash)) is not None:
        final_out = (
            Path(output_path)
            if output_path is not None
            else _default_output_path(path.stem, inst, store.artifact_dir(job_hash))
        )
        copy_file_safe(cached.musicxml_path, final_out)
        return replace(
            cached,
            musicxml_path=final_out,
            elapsed_sec=round(time.perf_counter() - t_start, 3),
        )

    if store is None and output_path is not None:
        with tempfile.TemporaryDirectory(prefix="tm_work_") as tmp_dir:
            return _produce_enrich_and_export(
                path, source_type, job_hash, cfg, Path(tmp_dir), Path(output_path), None, t_start
            )

    artifact_dir = (
        store.artifact_dir(job_hash)
        if store is not None
        else (cfg.cache_dir / "artifacts" / job_hash)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    canonical_out = _default_output_path(path.stem, inst, artifact_dir)
    result = _produce_enrich_and_export(
        path, source_type, job_hash, cfg, artifact_dir, canonical_out, store, t_start
    )
    final_out = Path(output_path) if output_path is not None else canonical_out
    copy_file_safe(canonical_out, final_out)
    return replace(result, musicxml_path=final_out)
