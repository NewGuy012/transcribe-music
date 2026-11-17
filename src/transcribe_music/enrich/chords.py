"""Hybrid acoustic chroma + symbolic Viterbi HMM chord recognizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import music21
import numpy as np

from transcribe_music.ingest.audio import load_mono_audio

_ROOT_NAMES = ("C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")

_QUALITIES: list[tuple[str, tuple[int, ...], float]] = [
    ("", (0, 4, 7), 1.0),  # Major triad
    ("m", (0, 3, 7), 1.0),  # Minor triad
    ("7", (0, 4, 7, 10), 0.92),  # Dominant 7
    ("maj7", (0, 4, 7, 11), 0.90),  # Major 7
    ("m7", (0, 3, 7, 10), 0.92),  # Minor 7
    ("dim", (0, 3, 6), 0.85),  # Diminished
]


@dataclass(frozen=True)
class _ChordTemplate:
    symbol: str
    root_pc: int
    vector: np.ndarray


def _build_templates() -> list[_ChordTemplate]:
    templates: list[_ChordTemplate] = []
    for root_pc, root_name in enumerate(_ROOT_NAMES):
        for suffix, intervals, prior_weight in _QUALITIES:
            vec = np.zeros(12, dtype=np.float64)
            for idx, interval in enumerate(intervals):
                pc = (root_pc + interval) % 12
                # Root and 3rd carry highest harmonic weight
                vec[pc] = 1.0 if idx == 0 else (0.85 if idx == 1 else 0.7)
            vec = vec / (np.linalg.norm(vec) + 1e-8) * prior_weight
            templates.append(_ChordTemplate(symbol=f"{root_name}{suffix}", root_pc=root_pc, vector=vec))
    return templates


_TEMPLATES = _build_templates()
_TEMPLATE_MATRIX = np.stack([t.vector for t in _TEMPLATES], axis=0)  # (72, 12)


def _compute_audio_chroma_by_window(
    wav_path: Path,
    tempo_bpm: float,
    total_beats: float,
    window_beats: float,
) -> np.ndarray | None:
    """Compute normalized 12-bin CQT chroma averaged over each beat window."""
    if not wav_path.exists():
        return None
    y, sr = load_mono_audio(wav_path, target_sr=22050)
    if y.size < sr // 2 or float(np.max(np.abs(y))) < 1e-5:
        return None

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    beats_per_sec = tempo_bpm / 60.0
    frame_beats = times * beats_per_sec

    n_windows = max(1, int(np.ceil(total_beats / window_beats)))
    out = np.zeros((n_windows, 12), dtype=np.float64)
    for w in range(n_windows):
        b0 = w * window_beats
        b1 = (w + 1) * window_beats
        mask = (frame_beats >= b0) & (frame_beats < b1)
        if np.any(mask):
            vec = np.mean(chroma[:, mask], axis=1)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                out[w] = vec / norm
    return out


def _compute_symbolic_chroma_by_window(
    score: music21.stream.Score,
    total_beats: float,
    window_beats: float,
) -> tuple[np.ndarray, list[int | None]]:
    """Accumulate pitch-class weights and lowest bass pitch class per beat window."""
    n_windows = max(1, int(np.ceil(total_beats / window_beats)))
    out = np.zeros((n_windows, 12), dtype=np.float64)
    lowest_midi: list[int | None] = [None] * n_windows

    flat = score.flatten()
    for el in flat.notes:
        if isinstance(el, music21.harmony.ChordSymbol):
            continue
        onset = float(el.offset)
        dur = max(0.125, float(el.duration.quarterLength))
        end = onset + dur
        pitches = [el.pitch] if isinstance(el, music21.note.Note) else list(el.pitches)

        w_start = max(0, int(onset // window_beats))
        w_end = min(n_windows - 1, int(max(onset, end - 1e-6) // window_beats))
        for w in range(w_start, w_end + 1):
            overlap = min(end, (w + 1) * window_beats) - max(onset, w * window_beats)
            if overlap <= 0:
                continue
            # Sublinear duration weighting so sustained bass pedals don't erase 3rds/5ths
            weight = float(np.sqrt(overlap))
            for p in pitches:
                midi_val = int(p.midi)
                pc = midi_val % 12
                out[w, pc] += weight
                cur_low = lowest_midi[w]
                if cur_low is None or midi_val < cur_low:
                    lowest_midi[w] = midi_val

    for w in range(n_windows):
        norm = float(np.linalg.norm(out[w]))
        if norm > 1e-8:
            out[w] /= norm
    bass_pcs = [m % 12 if m is not None else None for m in lowest_midi]
    return out, bass_pcs


def _viterbi_chords(
    chroma_windows: np.ndarray,
    bass_pcs: list[int | None] | None = None,
    switch_penalty: float = 0.14,
) -> list[str | None]:
    """Decode smoothest chord sequence via Viterbi over cosine similarity + bass/root priors."""
    n_win = chroma_windows.shape[0]
    n_states = len(_TEMPLATES)
    sims = chroma_windows @ _TEMPLATE_MATRIX.T  # (W, S)

    if bass_pcs is not None:
        root_pcs = np.array([t.root_pc for t in _TEMPLATES], dtype=np.intp)
        bass_arr = np.array([-1 if b is None else b for b in bass_pcs], dtype=np.intp)
        sims += (bass_arr[:, None] == root_pcs[None, :]) * 0.12
        sims -= (chroma_windows[:, root_pcs] < 1e-3) * 0.20

    energies = np.linalg.norm(chroma_windows, axis=1)
    dp = np.zeros((n_win, n_states), dtype=np.float64)
    back = np.zeros((n_win, n_states), dtype=np.int32)
    dp[0] = sims[0]
    state_ids = np.arange(n_states, dtype=np.int32)

    for t in range(1, n_win):
        prev_best_idx = int(np.argmax(dp[t - 1]))
        switch_val = dp[t - 1, prev_best_idx] - switch_penalty
        stay_mask = dp[t - 1] >= switch_val
        dp[t] = np.where(stay_mask, dp[t - 1], switch_val) + sims[t]
        back[t] = np.where(stay_mask, state_ids, prev_best_idx)

    best_states = np.zeros(n_win, dtype=np.int32)
    best_states[-1] = int(np.argmax(dp[-1]))
    for t in range(n_win - 1, 0, -1):
        best_states[t - 1] = back[t, best_states[t]]

    result: list[str | None] = []
    for t in range(n_win):
        if energies[t] < 1e-6:
            result.append(None)
        else:
            result.append(_TEMPLATES[int(best_states[t])].symbol)
    return result


def annotate_chords(
    score: music21.stream.Score,
    audio_wav_path: Path | None = None,
    tempo_bpm: float | None = None,
    window_beats: float = 2.0,
) -> music21.stream.Score:
    """Annotate score with smoothed <harmony> ChordSymbol objects at beat/half-bar offsets."""
    if not score.parts:
        return score

    target_part = score.parts[0]
    # Remove any existing ChordSymbols before re-annotating cleanly
    for existing in list(target_part.recurse().getElementsByClass(music21.harmony.ChordSymbol)):
        if existing.activeSite is not None:
            existing.activeSite.remove(existing)

    total_beats = float(target_part.highestTime)
    if total_beats <= 0:
        return score

    if tempo_bpm is None:
        mm = target_part.recurse().getElementsByClass(music21.tempo.MetronomeMark).first()
        tempo_bpm = float(mm.number) if mm is not None and mm.number else 120.0

    sym_chroma, bass_pcs = _compute_symbolic_chroma_by_window(score, total_beats, window_beats)
    aud_chroma = (
        _compute_audio_chroma_by_window(audio_wav_path, tempo_bpm, total_beats, window_beats)
        if audio_wav_path is not None
        else None
    )

    if aud_chroma is not None and aud_chroma.shape == sym_chroma.shape:
        # Fuse symbolic clarity + full-mix acoustic chroma (great for concerto Tutti!)
        combined = 0.65 * sym_chroma + 0.35 * aud_chroma
        for i in range(combined.shape[0]):
            # When soloist rests (Tutti), lean fully on acoustic orchestra chroma
            if float(np.linalg.norm(sym_chroma[i])) < 1e-6:
                combined[i] = aud_chroma[i]
    else:
        combined = sym_chroma

    labels = _viterbi_chords(combined, bass_pcs=bass_pcs)
    measures = list(target_part.getElementsByClass(music21.stream.Measure))

    last_symbol: str | None = None
    for w_idx, label in enumerate(labels):
        if label is None or label == last_symbol:
            continue
        beat_offset = w_idx * window_beats
        try:
            cs = music21.harmony.ChordSymbol(label)
        except (music21.harmony.ChordSymbolException, ValueError):
            continue

        if measures:
            # Place directly inside the owning measure at local measure offset
            placed = False
            for m in measures:
                m_start = float(m.offset)
                m_end = m_start + float(m.duration.quarterLength)
                if m_start - 1e-6 <= beat_offset < m_end - 1e-6:
                    m.insert(max(0.0, beat_offset - m_start), cs)
                    placed = True
                    break
            if not placed:
                target_part.insert(beat_offset, cs)
        else:
            target_part.insert(beat_offset, cs)
        last_symbol = label

    return score
