"""End-to-end transcription, enrichment, caching, and MusicXML export pipeline."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import music21

from transcribe_music.enrich.chords import annotate_chords
from transcribe_music.enrich.fingering import apply_fingering
from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.audio import normalize_audio_to_wav
from transcribe_music.ingest.youtube import download_youtube_slice, extract_youtube_id
from transcribe_music.models import (
    Instrument,
    SourceType,
    TimeRange,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.storage.catalog import CatalogStore, compute_job_hash, hash_file
from transcribe_music.transcribe.amt_general import transcribe_with_mt3
from transcribe_music.transcribe.amt_piano import transcribe_piano_transkun
from transcribe_music.transcribe.concerto import (
    detect_ensemble_accompaniment,
    prune_concerto_notes,
)
from transcribe_music.transcribe.omr_pdf import transcribe_pdf_omr
from transcribe_music.transcribe.quantize import midi_to_quantized_score


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


def _enrich_and_export(
    score: music21.stream.Score,
    config: TranscriptionConfig,
    source_type: SourceType,
    source_ref: str,
    job_hash: str,
    output_path: Path,
    store: CatalogStore,
    t_start: float,
    warnings: list[str],
    audio_wav_path: Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    title: str | None = None,
) -> TranscriptionResult:
    if config.enrich_chords:
        try:
            annotate_chords(
                score,
                audio_wav_path=audio_wav_path,
                tempo_bpm=config.tempo_bpm,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment fallback
            warnings.append(f"Chord enrichment skipped safely: {exc}")

    if config.enrich_fingering:
        try:
            apply_fingering(score, config.instrument)
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment fallback
            warnings.append(f"Fingering enrichment skipped safely: {exc}")

    metrics, export_warnings = export_musicxml(
        score,
        output_path=output_path,
        instrument=config.instrument,
        title=title,
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
        warnings=warnings,
    )

    store.record_job(
        result=result,
        config=config,
        measures=metrics.measures,
        notes=metrics.notes,
        chords=metrics.chords,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    return result


def _run_audio_pipeline(
    wav_path: Path,
    artifact_dir: Path,
    config: TranscriptionConfig,
    title: str,
) -> tuple[music21.stream.Score, list[str]]:
    """Run optimal solo vs concerto transcription path and return quantized Score."""
    warnings: list[str] = []
    instrument = config.instrument

    is_concerto = config.concerto_mode == "concerto"
    if config.concerto_mode == "auto":
        is_concerto = detect_ensemble_accompaniment(wav_path)

    raw_midi_path = artifact_dir / "transcribed.mid"

    if instrument == Instrument.PIANO:
        if is_concerto:
            solo_midi, _, mt3_warn = transcribe_with_mt3(
                wav_path,
                instrument=Instrument.PIANO,
                use_multi_track=True,
                cache_dir=config.cache_dir,
                output_midi_path=raw_midi_path,
                device=config.device,
            )
            warnings.extend(mt3_warn)
            if not any(inst.notes for inst in solo_midi.instruments):
                warnings.append("Multi-track piano stem empty; falling back to Transkun.")
                solo_midi = transcribe_piano_transkun(
                    wav_path, raw_midi_path, device=config.device
                )
        else:
            solo_midi = transcribe_piano_transkun(wav_path, raw_midi_path, device=config.device)
    else:
        # Guitar solo uses ultra-fast MR-MT3; concerto/strings with accompaniment use YourMT3
        use_multi = is_concerto
        solo_midi, _, mt3_warn = transcribe_with_mt3(
            wav_path,
            instrument=instrument,
            use_multi_track=use_multi,
            cache_dir=config.cache_dir,
            output_midi_path=raw_midi_path,
            device=config.device,
        )
        warnings.extend(mt3_warn)

    cleaned_midi = prune_concerto_notes(solo_midi, instrument=instrument)
    score = midi_to_quantized_score(
        cleaned_midi,
        instrument=instrument,
        audio_wav_path=wav_path,
        tempo_bpm=config.tempo_bpm,
        quantize_divisors=config.quantize_divisors,
        concerto_cues=is_concerto and instrument in (Instrument.VIOLIN, Instrument.CELLO),
        title=title,
    )
    return score, warnings


def _normalize_timestamp_input(value: str | float | None) -> str | float | None:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def transcribe_youtube(
    url: str,
    instrument: Instrument | str,
    start: str | float | None = None,
    end: str | float | None = None,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Download a YouTube video (or optional [start, end] slice) and transcribe it to MusicXML."""
    t_start = time.perf_counter()
    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = cfg.model_copy(update={"instrument": inst})

    start_val = _normalize_timestamp_input(start)
    end_val = _normalize_timestamp_input(end)
    time_range = (
        None
        if (start_val is None and end_val is None)
        else TimeRange.from_inputs(start_val, end_val)
    )
    start_sec = time_range.start_sec if time_range is not None else None
    end_sec = time_range.end_sec if time_range is not None else None

    video_id = extract_youtube_id(url)
    store = CatalogStore(cfg.cache_dir)
    job_hash = compute_job_hash(
        SourceType.YOUTUBE,
        video_id,
        cfg,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    artifact_dir = store.artifact_dir(job_hash)
    if time_range is None:
        default_stem = f"yt_{video_id}"
        pipeline_title = f"YouTube {video_id}"
    elif time_range.end_sec is None:
        default_stem = f"yt_{video_id}_{int(time_range.start_sec)}_end"
        pipeline_title = f"YouTube {video_id} ({int(time_range.start_sec)}s-end)"
    else:
        default_stem = f"yt_{video_id}_{int(time_range.start_sec)}_{int(time_range.end_sec)}"
        pipeline_title = f"YouTube {video_id} ({int(time_range.start_sec)}s-{int(time_range.end_sec)}s)"

    final_out = _resolve_output_path(output_path, default_stem, inst, artifact_dir)

    if cfg.use_cache:
        cached = store.get_cached_job(job_hash)
        if cached is not None:
            _copy_if_needed(cached.musicxml_path, final_out)
            return cached.model_copy(
                update={
                    "musicxml_path": final_out,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                }
            )

    wav_path = artifact_dir / ("yt_audio.wav" if time_range is None else "yt_slice.wav")
    download_youtube_slice(url, time_range, wav_path, sample_rate=44100)
    score, warnings = _run_audio_pipeline(wav_path, artifact_dir, cfg, title=pipeline_title)

    return _enrich_and_export(
        score=score,
        config=cfg,
        source_type=SourceType.YOUTUBE,
        source_ref=url,
        job_hash=job_hash,
        output_path=final_out,
        store=store,
        t_start=t_start,
        warnings=warnings,
        audio_wav_path=wav_path,
        start_sec=start_sec,
        end_sec=end_sec,
        title=f"YouTube {video_id}",
    )


def transcribe_audio(
    audio_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Transcribe a full local audio file into enriched MusicXML."""
    t_start = time.perf_counter()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = cfg.model_copy(update={"instrument": inst})

    store = CatalogStore(cfg.cache_dir)
    digest = hash_file(audio_path)
    job_hash = compute_job_hash(SourceType.AUDIO, digest, cfg)
    artifact_dir = store.artifact_dir(job_hash)
    final_out = _resolve_output_path(output_path, audio_path.stem, inst, artifact_dir)

    if cfg.use_cache:
        cached = store.get_cached_job(job_hash)
        if cached is not None:
            _copy_if_needed(cached.musicxml_path, final_out)
            return cached.model_copy(
                update={
                    "musicxml_path": final_out,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                }
            )

    wav_path = artifact_dir / "audio.wav"
    normalize_audio_to_wav(audio_path, wav_path, sample_rate=44100)
    score, warnings = _run_audio_pipeline(wav_path, artifact_dir, cfg, title=audio_path.stem)

    return _enrich_and_export(
        score=score,
        config=cfg,
        source_type=SourceType.AUDIO,
        source_ref=str(audio_path),
        job_hash=job_hash,
        output_path=final_out,
        store=store,
        t_start=t_start,
        warnings=warnings,
        audio_wav_path=wav_path,
        title=audio_path.stem,
    )


def transcribe_pdf(
    pdf_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Run OMR on a sheet music PDF and enrich with chords and ergonomic fingerings."""
    t_start = time.perf_counter()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = cfg.model_copy(update={"instrument": inst})

    store = CatalogStore(cfg.cache_dir)
    digest = hash_file(pdf_path)
    job_hash = compute_job_hash(SourceType.PDF, digest, cfg)
    artifact_dir = store.artifact_dir(job_hash)
    final_out = _resolve_output_path(output_path, pdf_path.stem, inst, artifact_dir)

    if cfg.use_cache:
        cached = store.get_cached_job(job_hash)
        if cached is not None:
            _copy_if_needed(cached.musicxml_path, final_out)
            return cached.model_copy(
                update={
                    "musicxml_path": final_out,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                }
            )

    score = transcribe_pdf_omr(pdf_path, artifact_dir, instrument=inst, dpi=cfg.pdf_dpi)
    return _enrich_and_export(
        score=score,
        config=cfg,
        source_type=SourceType.PDF,
        source_ref=str(pdf_path),
        job_hash=job_hash,
        output_path=final_out,
        store=store,
        t_start=t_start,
        warnings=[],
        audio_wav_path=None,
        title=pdf_path.stem,
    )
