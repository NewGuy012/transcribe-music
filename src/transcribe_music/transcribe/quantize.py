"""Beat tracking, rhythmic quantization, Grand Staff piano split, and Tutti marking."""

from __future__ import annotations

from pathlib import Path

import librosa
import music21
import numpy as np
import pretty_midi

from transcribe_music.ingest.audio import load_mono_audio
from transcribe_music.models import INSTRUMENT_SPECS, Instrument


def estimate_tempo_and_beats(
    wav_path: Path | None,
    fallback_bpm: float = 120.0,
) -> tuple[float, np.ndarray]:
    """Estimate BPM and beat timestamps from audio using librosa."""
    if wav_path is None or not Path(wav_path).exists():
        return fallback_bpm, np.array([], dtype=np.float64)

    y, sr = load_mono_audio(wav_path, target_sr=22050)
    if len(y) < sr * 1.5 or float(np.max(np.abs(y))) < 1e-5:
        return fallback_bpm, np.array([], dtype=np.float64)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo_val = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else fallback_bpm
    if not (35.0 <= tempo_val <= 240.0):
        tempo_val = fallback_bpm
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return tempo_val, beat_times


def _seconds_to_beats(t_sec: float, tempo_bpm: float, beat_times: np.ndarray) -> float:
    """Convert audio seconds into beat offsets using tracked beats with linear fallback."""
    if len(beat_times) >= 4:
        # Piecewise linear map over tracked beat grid, extrapolated linearly outside bounds
        if t_sec <= beat_times[0]:
            dt = (t_sec - beat_times[0]) * (tempo_bpm / 60.0)
            return max(0.0, dt)
        if t_sec >= beat_times[-1]:
            return float(len(beat_times) - 1) + (t_sec - beat_times[-1]) * (tempo_bpm / 60.0)
        beat_idx = np.interp(t_sec, beat_times, np.arange(len(beat_times), dtype=np.float64))
        return float(beat_idx)
    return max(0.0, t_sec * (tempo_bpm / 60.0))


def _quantize_ql(value: float, divisors: tuple[int, ...] = (4, 3)) -> float:
    """Snap quarterLength to nearest allowed rhythmic subdivision grid."""
    candidates: list[float] = []
    for div in divisors:
        step = 1.0 / div
        snapped = round(value / step) * step
        candidates.append(snapped)
    best = min(candidates, key=lambda c: abs(c - value))
    return float(round(best, 4))


def _quantize_duration_ql(dur_ql: float, divisors: tuple[int, ...] = (4, 3)) -> float:
    min_step = 1.0 / max(divisors)
    snapped = _quantize_ql(max(min_step, dur_ql), divisors=divisors)
    return max(min_step, snapped)


def _split_piano_rh_lh(
    events: list[tuple[float, float, int, int]],
) -> tuple[list[tuple[float, float, int, int]], list[tuple[float, float, int, int]]]:
    """Split piano notes into right-hand (Treble) and left-hand (Bass) streams."""
    rh: list[tuple[float, float, int, int]] = []
    lh: list[tuple[float, float, int, int]] = []
    # Group simultaneous onsets so chords stay ergonomically coherent
    by_onset: dict[float, list[tuple[float, float, int, int]]] = {}
    for ev in events:
        by_onset.setdefault(ev[0], []).append(ev)

    split_pitch = 60  # Middle C4 baseline
    for onset in sorted(by_onset):
        group = sorted(by_onset[onset], key=lambda x: x[2])
        if len(group) == 1:
            ev = group[0]
            if ev[2] >= split_pitch:
                rh.append(ev)
            else:
                lh.append(ev)
        else:
            # Split wide chords across hands if span > octave, otherwise use C4 split
            low = group[0][2]
            high = group[-1][2]
            if high - low > 12:
                mid = 0.5 * (low + high)
                for ev in group:
                    (rh if ev[2] >= mid else lh).append(ev)
            else:
                for ev in group:
                    (rh if ev[2] >= split_pitch else lh).append(ev)
    return rh, lh


def _populate_part_from_events(
    part: music21.stream.Part,
    events: list[tuple[float, float, int, int]],
    total_beats: float,
    transpose_semitones: int = 0,
) -> None:
    """Insert note/chord events cleanly by beat offset without colliding voices."""
    by_onset: dict[float, list[tuple[float, float, int, int]]] = {}
    for onset, dur, midi_pitch, vel in events:
        by_onset.setdefault(onset, []).append((onset, dur, midi_pitch, vel))

    sorted_onsets = sorted(by_onset)
    for idx, onset in enumerate(sorted_onsets):
        group = sorted(by_onset[onset], key=lambda x: x[2])
        # Trim duration if next onset on same staff overlaps excessively to avoid messy voices
        next_onset = sorted_onsets[idx + 1] if idx + 1 < len(sorted_onsets) else total_beats
        max_non_overlap = max(0.125, next_onset - onset)

        if len(group) == 1:
            _, dur, midi_pitch, vel = group[0]
            written_pitch = midi_pitch - transpose_semitones
            n = music21.note.Note(written_pitch)
            n.duration = music21.duration.Duration(min(dur, max_non_overlap))
            n.volume.velocity = vel
            part.insert(onset, n)
        else:
            # Deduplicate duplicate pitches in same onset bucket
            unique_pitches: dict[int, tuple[float, int]] = {}
            for _, dur, midi_pitch, vel in group:
                if midi_pitch not in unique_pitches:
                    unique_pitches[midi_pitch] = (dur, vel)
            pitches = sorted(unique_pitches)
            durs = [unique_pitches[p][0] for p in pitches]
            chord_dur = min(max(durs), max_non_overlap)
            written_pitches = [p - transpose_semitones for p in pitches]
            ch = music21.chord.Chord(written_pitches)
            ch.duration = music21.duration.Duration(chord_dur)
            ch.volume.velocity = int(np.mean([unique_pitches[p][1] for p in pitches]))
            part.insert(onset, ch)


def _mark_tutti_rests(part: music21.stream.Part) -> None:
    """Label extended empty concerto spans (>= 2 full measures) with a Tutti rehearsal cue."""
    measures = list(part.getElementsByClass(music21.stream.Measure))
    run_start: int | None = None
    run_len = 0

    def _flush(start_idx: int | None, length: int) -> None:
        if start_idx is not None and length >= 2:
            measures[start_idx].insert(0, music21.expressions.RehearsalMark("Tutti"))

    for i, m in enumerate(measures):
        has_notes = any(
            not isinstance(n, music21.harmony.ChordSymbol) for n in m.notes
        )
        if not has_notes:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
        else:
            _flush(run_start, run_len)
            run_start = None
            run_len = 0
    _flush(run_start, run_len)


def _build_part(
    part_id: str,
    events: list[tuple[float, float, int, int]],
    total_beats: float,
    bpm: float,
    clef: music21.clef.Clef | None = None,
    include_tempo: bool = True,
    transpose_semitones: int = 0,
) -> music21.stream.Part:
    part = music21.stream.Part(id=part_id)
    if clef is not None:
        part.insert(0, clef)
    part.insert(0, music21.meter.TimeSignature("4/4"))
    if include_tempo:
        part.insert(0, music21.tempo.MetronomeMark(number=round(bpm, 1)))
    _populate_part_from_events(part, events, total_beats, transpose_semitones=transpose_semitones)
    if part.highestTime < total_beats:
        part.insert(
            part.highestTime,
            music21.note.Rest(quarterLength=max(0.25, total_beats - part.highestTime)),
        )
    part.makeMeasures(inPlace=True)
    part.makeRests(fillGaps=True, inPlace=True)
    return part


def midi_to_quantized_score(
    midi: pretty_midi.PrettyMIDI,
    instrument: Instrument,
    audio_wav_path: Path | None = None,
    tempo_bpm: float | None = None,
    quantize_divisors: tuple[int, ...] = (4, 3),
    concerto_cues: bool = False,
    title: str | None = None,
) -> music21.stream.Score:
    """Convert raw MIDI events into a beat-quantized, bar-aligned music21 Score."""
    spec = INSTRUMENT_SPECS[instrument]

    if tempo_bpm is not None:
        bpm = float(tempo_bpm)
        beat_times = np.array([], dtype=np.float64)
    else:
        bpm, beat_times = estimate_tempo_and_beats(audio_wav_path, fallback_bpm=120.0)

    raw_notes = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    events: list[tuple[float, float, int, int]] = []
    max_beat = 4.0
    for n in raw_notes:
        if not (spec.midi_min <= n.pitch <= spec.midi_max):
            continue
        onset_b = _quantize_ql(
            _seconds_to_beats(n.start, bpm, beat_times), divisors=quantize_divisors
        )
        end_b = _seconds_to_beats(n.end, bpm, beat_times)
        dur_b = _quantize_duration_ql(max(0.125, end_b - onset_b), divisors=quantize_divisors)
        events.append((onset_b, dur_b, int(n.pitch), int(n.velocity)))
        max_beat = max(max_beat, onset_b + dur_b)

    total_measures = max(1, int(np.ceil(max_beat / 4.0)))
    total_beats = float(total_measures * 4)

    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title or f"{spec.display_name} Transcription")

    if instrument == Instrument.PIANO:
        rh_events, lh_events = _split_piano_rh_lh(events)
        score.insert(0, _build_part("RH", rh_events, total_beats, bpm, music21.clef.TrebleClef()))
        score.insert(
            0, _build_part("LH", lh_events, total_beats, bpm, music21.clef.BassClef(), include_tempo=False)
        )
    else:
        part = _build_part(
            instrument.value, events, total_beats, bpm, transpose_semitones=spec.transpose_semitones
        )
        if concerto_cues:
            _mark_tutti_rests(part)
        score.insert(0, part)

    # Estimate global key signature for accurate enharmonic pitch spelling (F# vs Gb)
    try:
        estimated_key = score.analyze("key")
        if estimated_key is not None:
            for part in score.parts:
                first_m = part.getElementsByClass(music21.stream.Measure).first()
                if first_m is not None:
                    first_m.insert(0, estimated_key)
                for n in part.recurse().notes:
                    if isinstance(n, music21.note.Note):
                        n.pitch.simplifyEnharmonic(inPlace=True)
                    elif isinstance(n, music21.chord.Chord):
                        for p in n.pitches:
                            p.simplifyEnharmonic(inPlace=True)
    except Exception as exc:  # noqa: BLE001 — optional key estimation fallback
        import logging

        logging.getLogger(__name__).debug("Key estimation skipped: %s", exc)

    return score
