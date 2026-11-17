"""Beat tracking, rhythmic quantization, Grand Staff piano split, and Tutti marking."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import librosa
import music21
import numpy as np
import pretty_midi

from transcribe_music.export.musicxml import split_tuplet_interval
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


def _snap_to_divisor(value: float, divisor: int) -> Fraction:
    return Fraction(round(value * divisor), divisor)


def _select_beat_divisors(
    raw_beats: list[tuple[float, float, int, int]],
    divisors: tuple[int, ...],
) -> dict[int, int]:
    """Choose one consistent subdivision divisor per beat to avoid mixed-grid tuplets."""
    divs = divisors or (4,)
    onsets_by_beat: dict[int, list[float]] = {}
    for sb, _, _, _ in raw_beats:
        onsets_by_beat.setdefault(max(0, math.floor(sb)), []).append(sb)

    beat_div: dict[int, int] = {}
    for beat_idx, onsets in onsets_by_beat.items():
        best_div = divs[0]
        best_score = (float("inf"), 0)
        for idx, div in enumerate(divs):
            err = sum(abs(float(_snap_to_divisor(o, div)) - o) for o in onsets)
            score = (round(err, 9), idx)
            if score < best_score:
                best_score = score
                best_div = div
        beat_div[beat_idx] = best_div
    return beat_div


def _split_piano_rh_lh(
    events: list[tuple[Fraction, Fraction, int, int]],
) -> tuple[list[tuple[Fraction, Fraction, int, int]], list[tuple[Fraction, Fraction, int, int]]]:
    """Split piano notes into right-hand (Treble) and left-hand (Bass) streams."""
    rh: list[tuple[Fraction, Fraction, int, int]] = []
    lh: list[tuple[Fraction, Fraction, int, int]] = []
    # Group simultaneous onsets so chords stay ergonomically coherent
    by_onset: dict[Fraction, list[tuple[Fraction, Fraction, int, int]]] = {}
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
    events: list[tuple[Fraction, Fraction, int, int]],
    total_beats: Fraction,
    transpose_semitones: int = 0,
) -> Fraction:
    """Insert note/chord events cleanly by beat offset without colliding voices or bad tuplets."""
    by_onset: dict[Fraction, list[tuple[Fraction, Fraction, int, int]]] = {}
    for onset, dur, midi_pitch, vel in events:
        onset_f = Fraction(onset)
        dur_f = Fraction(dur)
        by_onset.setdefault(onset_f, []).append((onset_f, dur_f, midi_pitch, vel))

    sorted_onsets = sorted(by_onset)
    cursor = Fraction(0, 1)
    for idx, onset in enumerate(sorted_onsets):
        if onset > cursor:
            for r_start, r_end in split_tuplet_interval(cursor, onset):
                part.insert(r_start, music21.note.Rest(quarterLength=r_end - r_start))

        group = sorted(by_onset[onset], key=lambda x: x[2])
        # Trim duration to next onset on same staff so voices never collide
        next_onset = sorted_onsets[idx + 1] if idx + 1 < len(sorted_onsets) else total_beats

        unique_pitches: dict[int, tuple[Fraction, int]] = {}
        for _, dur, midi_pitch, vel in group:
            if midi_pitch not in unique_pitches:
                unique_pitches[midi_pitch] = (dur, vel)
            else:
                prev_dur, prev_vel = unique_pitches[midi_pitch]
                unique_pitches[midi_pitch] = (max(prev_dur, dur), max(prev_vel, vel))

        pitches = sorted(unique_pitches)
        raw_end = onset + max(unique_pitches[p][0] for p in pitches)
        clipped_end = min(raw_end, next_onset)
        if clipped_end <= onset:
            clipped_end = next_onset

        vel_avg = int(np.mean([unique_pitches[p][1] for p in pitches]))
        written_pitches = [p - transpose_semitones for p in pitches]

        segments = split_tuplet_interval(onset, clipped_end)
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_dur = seg_end - seg_start
            if len(written_pitches) == 1:
                el: music21.note.Note | music21.chord.Chord = music21.note.Note(written_pitches[0])
            else:
                el = music21.chord.Chord(written_pitches)
            el.duration = music21.duration.Duration(seg_dur)
            el.volume.velocity = vel_avg
            if len(segments) > 1:
                tie_type = (
                    "start"
                    if seg_idx == 0
                    else ("stop" if seg_idx == len(segments) - 1 else "continue")
                )
                el.tie = music21.tie.Tie(tie_type)
            part.insert(seg_start, el)

        cursor = max(cursor, clipped_end)
    return cursor


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
    events: list[tuple[Fraction, Fraction, int, int]],
    total_beats: Fraction,
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
    cursor = _populate_part_from_events(
        part, events, total_beats, transpose_semitones=transpose_semitones
    )
    if cursor < total_beats:
        for r_start, r_end in split_tuplet_interval(cursor, total_beats):
            part.insert(r_start, music21.note.Rest(quarterLength=r_end - r_start))
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
    divs = quantize_divisors or (4,)

    if tempo_bpm is not None:
        bpm = float(tempo_bpm)
        beat_times = np.array([], dtype=np.float64)
    else:
        bpm, beat_times = estimate_tempo_and_beats(audio_wav_path, fallback_bpm=120.0)

    raw_notes = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    raw_beats = [
        (
            _seconds_to_beats(n.start, bpm, beat_times),
            _seconds_to_beats(n.end, bpm, beat_times),
            int(n.pitch),
            int(n.velocity),
        )
        for n in raw_notes
        if spec.midi_min <= n.pitch <= spec.midi_max
    ]
    beat_div = _select_beat_divisors(raw_beats, divs)

    events: list[tuple[Fraction, Fraction, int, int]] = []
    max_beat = Fraction(4, 1)
    for sb, eb, pitch, vel in raw_beats:
        d_start = beat_div.get(max(0, math.floor(sb)), divs[0])
        onset_q = max(Fraction(0, 1), _snap_to_divisor(sb, d_start))
        d_onset = beat_div.get(math.floor(onset_q), divs[0])

        d_end = beat_div.get(max(0, math.floor(eb)), divs[0])
        end_q = _snap_to_divisor(eb, d_end)
        min_dur = Fraction(1, d_onset)
        if end_q <= onset_q:
            end_q = onset_q + min_dur
        events.append((onset_q, end_q - onset_q, pitch, vel))
        max_beat = max(max_beat, end_q)

    total_measures = max(1, math.ceil(max_beat / 4))
    total_beats = Fraction(total_measures * 4, 1)

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
