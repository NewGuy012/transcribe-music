"""End-to-end transcription, enrichment, caching, and MusicXML export pipeline."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
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
from transcribe_music.transcribe.amt_piano import (
    consume_last_torch_device,
    resolve_torch_device,
    transcribe_piano_transkun,
)
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


@dataclass(frozen=True)
class _ProducedScore:
    score: music21.stream.Score
    warnings: list[str]
    audio_wav_path: Path | None
    backend: str
    device_used: str | None


def _enrich_and_export(
    produced: _ProducedScore,
    config: TranscriptionConfig,
    source_type: SourceType,
    source_ref: str,
    job_hash: str,
    output_path: Path,
    store: CatalogStore | None,
    t_start: float,
    start_sec: float | None = None,
    end_sec: float | None = None,
    title: str | None = None,
) -> TranscriptionResult:
    score = produced.score
    warnings = list(produced.warnings)
    if config.enrich_chords:
        try:
            annotate_chords(
                score,
                audio_wav_path=produced.audio_wav_path,
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
        backend=produced.backend,
        device_used=produced.device_used,
        warnings=warnings,
    )

    if store is not None:
        store.record_job(
            result=result,
            config=config,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    return result


def _run_audio_pipeline(
    wav_path: Path,
    artifact_dir: Path,
    config: TranscriptionConfig,
    title: str,
) -> _ProducedScore:
    """Run optimal solo vs concerto transcription path and return quantized Score."""
    warnings: list[str] = []
    instrument = config.instrument

    is_concerto = config.concerto_mode == "concerto"
    if config.concerto_mode == "auto":
        is_concerto = detect_ensemble_accompaniment(wav_path)

    raw_midi_path = artifact_dir / "transcribed.mid"
    consume_last_torch_device()

    if instrument == Instrument.PIANO:
        if is_concerto:
            backend = "yourmt3"
            solo_midi, mt3_warn = transcribe_with_mt3(
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
                backend = "transkun"
                consume_last_torch_device()
                solo_midi = transcribe_piano_transkun(
                    wav_path, raw_midi_path, device=config.device
                )
        else:
            backend = "transkun"
            solo_midi = transcribe_piano_transkun(wav_path, raw_midi_path, device=config.device)
    else:
        # Guitar solo uses ultra-fast MR-MT3; concerto/strings with accompaniment use YourMT3
        use_multi = is_concerto
        backend = "yourmt3" if use_multi else "mr_mt3"
        solo_midi, mt3_warn = transcribe_with_mt3(
            wav_path,
            instrument=instrument,
            use_multi_track=use_multi,
            cache_dir=config.cache_dir,
            output_midi_path=raw_midi_path,
            device=config.device,
        )
        warnings.extend(mt3_warn)

    device_used = consume_last_torch_device() or resolve_torch_device(config.device)
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
    return _ProducedScore(
        score=score,
        warnings=warnings,
        audio_wav_path=wav_path,
        backend=backend,
        device_used=device_used,
    )


def _normalize_timestamp_input(value: str | float | None) -> str | float | None:
    if isinstance(value, str) and not value.strip():
        return None
    return value


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
    title: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> TranscriptionResult:
    t_start = time.perf_counter()
    inst = Instrument(instrument)
    cfg = config or TranscriptionConfig(instrument=inst)
    if cfg.instrument != inst:
        cfg = cfg.model_copy(update={"instrument": inst})

    job_hash = compute_job_hash(
        source_type,
        source_identity,
        cfg,
        start_sec=start_sec,
        end_sec=end_sec,
    )

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
            start_sec=start_sec,
            end_sec=end_sec,
            title=title,
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
                start_sec=start_sec,
                end_sec=end_sec,
                title=title,
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
        start_sec=start_sec,
        end_sec=end_sec,
        title=title,
    )


def transcribe_youtube(
    url: str,
    instrument: Instrument | str,
    start: str | float | None = None,
    end: str | float | None = None,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Download a YouTube video (or optional [start, end] slice) and transcribe it to MusicXML."""
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

    if time_range is None:
        default_stem = f"yt_{video_id}"
        pipeline_title = f"YouTube {video_id}"
    elif time_range.end_sec is None:
        default_stem = f"yt_{video_id}_{int(time_range.start_sec)}_end"
        pipeline_title = f"YouTube {video_id} ({int(time_range.start_sec)}s-end)"
    else:
        default_stem = f"yt_{video_id}_{int(time_range.start_sec)}_{int(time_range.end_sec)}"
        pipeline_title = f"YouTube {video_id} ({int(time_range.start_sec)}s-{int(time_range.end_sec)}s)"

    def _produce(artifact_dir: Path, cfg: TranscriptionConfig) -> _ProducedScore:
        wav_path = artifact_dir / ("yt_audio.wav" if time_range is None else "yt_slice.wav")
        download_youtube_slice(url, time_range, wav_path, sample_rate=44100)
        return _run_audio_pipeline(wav_path, artifact_dir, cfg, title=pipeline_title)

    return _run_cached_job(
        SourceType.YOUTUBE,
        video_id,
        url,
        instrument,
        default_stem,
        output_path,
        config,
        _produce,
        title=f"YouTube {video_id}",
        start_sec=start_sec,
        end_sec=end_sec,
    )


def transcribe_audio(
    audio_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Transcribe a full local audio file into enriched MusicXML."""
    path = Path(audio_path)
    _require_existing_file(path)

    def _produce(artifact_dir: Path, cfg: TranscriptionConfig) -> _ProducedScore:
        wav_path = artifact_dir / "audio.wav"
        normalize_audio_to_wav(path, wav_path, sample_rate=44100)
        return _run_audio_pipeline(wav_path, artifact_dir, cfg, title=path.stem)

    return _run_cached_job(
        SourceType.AUDIO,
        hash_file(path),
        str(path),
        instrument,
        path.stem,
        output_path,
        config,
        _produce,
        title=path.stem,
    )


def transcribe_pdf(
    pdf_path: Path | str,
    instrument: Instrument | str,
    output_path: Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Run OMR on a sheet music PDF and enrich with chords and ergonomic fingerings."""
    path = Path(pdf_path)
    _require_existing_file(path, expected_suffix=".pdf")

    def _produce(artifact_dir: Path, cfg: TranscriptionConfig) -> _ProducedScore:
        score = transcribe_pdf_omr(path, artifact_dir, instrument=cfg.instrument, dpi=cfg.pdf_dpi)
        return _ProducedScore(
            score=score,
            warnings=[],
            audio_wav_path=None,
            backend="homr",
            device_used=None,
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
        title=path.stem,
    )
