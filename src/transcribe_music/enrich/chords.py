"""Pure symbolic Viterbi HMM chord recognizer with optional full-score harmonic context."""

from __future__ import annotations

import math
from dataclasses import dataclass

import music21

from transcribe_music.export.musicxml import is_playable_chord

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
    vector: tuple[float, ...]


def _build_templates() -> list[_ChordTemplate]:
    templates: list[_ChordTemplate] = []
    for root_pc, root_name in enumerate(_ROOT_NAMES):
        for suffix, intervals, prior_weight in _QUALITIES:
            vec = [0.0] * 12
            for idx, interval in enumerate(intervals):
                pc = (root_pc + interval) % 12
                vec[pc] = 1.0 if idx == 0 else (0.85 if idx == 1 else 0.7)
            scale = prior_weight / (math.hypot(*vec) + 1e-8)
            templates.append(
                _ChordTemplate(
                    symbol=f"{root_name}{suffix}",
                    root_pc=root_pc,
                    vector=tuple(v * scale for v in vec),
                )
            )
    return templates


_TEMPLATES = _build_templates()


def _compute_symbolic_chroma_by_window(
    score: music21.stream.Score,
    total_beats: float,
    window_beats: float,
) -> tuple[list[list[float]], list[int | None]]:
    """Accumulate pitch-class weights and lowest bass pitch class per beat window."""
    n_windows = max(1, math.ceil(total_beats / window_beats))
    out = [[0.0] * 12 for _ in range(n_windows)]
    lowest_midi: list[int | None] = [None] * n_windows

    try:
        sounding_score = score.toSoundingPitch()
    except Exception:  # noqa: BLE001 — fallback when transposition metadata is incomplete
        sounding_score = score

    flat = sounding_score.flatten()
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
            # Sublinear weighting prevents sustained bass notes from drowning 3rds/5ths
            weight = math.sqrt(overlap)
            row = out[w]
            for p in pitches:
                midi_val = int(p.midi)
                row[midi_val % 12] += weight
                cur_low = lowest_midi[w]
                if cur_low is None or midi_val < cur_low:
                    lowest_midi[w] = midi_val

    for row in out:
        norm = math.hypot(*row)
        if norm > 1e-8:
            inv = 1.0 / norm
            for i in range(12):
                row[i] *= inv
    bass_pcs = [m % 12 if m is not None else None for m in lowest_midi]
    return out, bass_pcs


def _viterbi_chords(
    chroma_windows: list[list[float]],
    bass_pcs: list[int | None] | None = None,
    switch_penalty: float = 0.14,
) -> list[str | None]:
    """Decode smoothest chord sequence via Viterbi over cosine similarity + bass/root priors."""
    n_win = len(chroma_windows)
    n_states = len(_TEMPLATES)

    sims: list[list[float]] = []
    energies: list[float] = []
    for t, chroma in enumerate(chroma_windows):
        energies.append(math.hypot(*chroma))
        bass = bass_pcs[t] if bass_pcs is not None else None
        row: list[float] = []
        for tmpl in _TEMPLATES:
            sim = sum(c * tv for c, tv in zip(chroma, tmpl.vector, strict=True))
            if bass_pcs is not None:
                if bass == tmpl.root_pc:
                    sim += 0.12
                if chroma[tmpl.root_pc] < 1e-3:
                    sim -= 0.20
            row.append(sim)
        sims.append(row)

    dp = list(sims[0])
    back: list[list[int]] = []
    for t in range(1, n_win):
        prev_best_idx = max(range(n_states), key=dp.__getitem__)
        switch_val = dp[prev_best_idx] - switch_penalty
        cur_sims = sims[t]
        cur_dp: list[float] = []
        cur_back: list[int] = []
        for s in range(n_states):
            stay = dp[s] >= switch_val
            cur_dp.append((dp[s] if stay else switch_val) + cur_sims[s])
            cur_back.append(s if stay else prev_best_idx)
        dp = cur_dp
        back.append(cur_back)

    best_states = [0] * n_win
    best_states[-1] = max(range(n_states), key=dp.__getitem__)
    for t in range(n_win - 1, 0, -1):
        best_states[t - 1] = back[t - 1][best_states[t]]

    return [
        None if energies[t] < 1e-6 else _TEMPLATES[best_states[t]].symbol
        for t in range(n_win)
    ]


def annotate_chords(
    score: music21.stream.Score,
    harmonic_reference_score: music21.stream.Score | None = None,
    window_beats: float = 2.0,
    preserve_existing: bool = False,
) -> music21.stream.Score:
    """Annotate score with smoothed <harmony> ChordSymbol objects at beat/half-bar offsets."""
    if not score.parts:
        return score

    target_part = score.parts[0]
    existing_chords = list(target_part.recurse().getElementsByClass(music21.harmony.ChordSymbol))
    if preserve_existing and existing_chords:
        return score

    for existing in existing_chords:
        if existing.activeSite is not None:
            existing.activeSite.remove(existing)

    total_beats = float(target_part.highestTime)
    if total_beats <= 0:
        return score

    ref_score = harmonic_reference_score if harmonic_reference_score is not None else score
    chroma, bass_pcs = _compute_symbolic_chroma_by_window(ref_score, total_beats, window_beats)
    labels = _viterbi_chords(chroma, bass_pcs=bass_pcs)
    measures = list(target_part.getElementsByClass(music21.stream.Measure))
    measure_has_notes = {
        id(m): any(
            isinstance(el, music21.note.Note) or is_playable_chord(el)
            for el in m.recurse().notes
        )
        for m in measures
    }

    last_symbol: str | None = None
    placed_rest_measures: set[int] = set()
    for w_idx, label in enumerate(labels):
        if label is None or label == last_symbol:
            continue
        beat_offset = w_idx * window_beats
        try:
            cs = music21.harmony.ChordSymbol(label)
            cs.writeAsChord = False
        except (music21.harmony.ChordSymbolException, ValueError):
            continue

        if measures:
            placed = False
            for m in measures:
                m_start = float(m.offset)
                m_end = m_start + float(m.duration.quarterLength)
                if m_start - 1e-6 <= beat_offset < m_end - 1e-6:
                    if not measure_has_notes[id(m)]:
                        if id(m) in placed_rest_measures:
                            placed = True
                            break
                        m.insert(0.0, cs)
                        placed_rest_measures.add(id(m))
                    else:
                        m.insert(max(0.0, beat_offset - m_start), cs)
                    placed = True
                    break
            if not placed:
                target_part.insert(beat_offset, cs)
        else:
            target_part.insert(beat_offset, cs)
        last_symbol = label

    return score
