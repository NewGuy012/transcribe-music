"""Concerto accompaniment detection and soloist salience / polyphony filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from transcribe_music.models import INSTRUMENT_SPECS, Instrument


def detect_ensemble_accompaniment(wav_path: Path) -> bool:
    """Fast heuristic check for wide/sustained orchestral backing vs dry solo audio."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if data.shape[0] < sr:
        return False

    # Check stereo side-channel energy (solo close-mic stays center; concertos have wide hall/orchestra)
    if data.shape[1] >= 2:
        mid = 0.5 * (data[:, 0] + data[:, 1])
        side = 0.5 * (data[:, 0] - data[:, 1])
        mid_rms = float(np.sqrt(np.mean(mid**2)) + 1e-8)
        side_rms = float(np.sqrt(np.mean(side**2)))
        side_ratio = side_rms / mid_rms
        if side_ratio > 0.35:
            return True

    return False


def prune_concerto_notes(
    midi: pretty_midi.PrettyMIDI,
    instrument: Instrument,
    min_duration_sec: float = 0.045,
    min_velocity: int = 20,
    onset_cluster_sec: float = 0.04,
) -> pretty_midi.PrettyMIDI:
    """Prune background orchestral bleed, micro-blips, and excess polyphony per instrument."""
    spec = INSTRUMENT_SPECS[instrument]
    cleaned = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    all_notes: list[pretty_midi.Note] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            dur = note.end - note.start
            if (
                dur >= min_duration_sec
                and note.velocity >= min_velocity
                and spec.midi_min <= note.pitch <= spec.midi_max
            ):
                all_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    )
                )

    all_notes.sort(key=lambda n: (n.start, -n.velocity, -n.pitch))

    # Deduplicate near-identical pitch repetitions within 35ms
    deduped: list[pretty_midi.Note] = []
    last_onset_by_pitch: dict[int, float] = {}
    for n in all_notes:
        prev = last_onset_by_pitch.get(n.pitch)
        if prev is not None and (n.start - prev) < 0.035:
            continue
        deduped.append(n)
        last_onset_by_pitch[n.pitch] = n.start

    # Enforce instrument max polyphony across onset clusters (keeps top-salience soloist voices)
    pruned: list[pretty_midi.Note] = []
    idx = 0
    while idx < len(deduped):
        anchor_t = deduped[idx].start
        cluster = [deduped[idx]]
        j = idx + 1
        while j < len(deduped) and (deduped[j].start - anchor_t) <= onset_cluster_sec:
            cluster.append(deduped[j])
            j += 1

        if len(cluster) > spec.max_polyphony:
            # For solo violin/cello, prefer higher-velocity foreground melody notes
            cluster.sort(key=lambda n: (n.velocity, n.pitch), reverse=True)
            cluster = cluster[: spec.max_polyphony]
            cluster.sort(key=lambda n: n.pitch)

        pruned.extend(cluster)
        idx = j

    # Clamp notes longer than 4.5s on bowed strings (typical AMT bleed artifact)
    if spec.max_polyphony == 2 and len(pruned) > 1:
        for n in pruned:
            if n.end - n.start > 4.5:
                n.end = n.start + 4.5

    out_inst = pretty_midi.Instrument(program=spec.gm_programs[0], name=spec.display_name)
    out_inst.notes = sorted(pruned, key=lambda n: (n.start, n.pitch))
    cleaned.instruments.append(out_inst)
    return cleaned
