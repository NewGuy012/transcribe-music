"""Pure symbolic Viterbi HMM chord recognizer with optional full-score harmonic context."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from operator import itemgetter

import music21

from transcribe_music.models import Instrument
from transcribe_music.score_utils import element_midis, is_playable_note, iter_playable_notes

_SHARP_ROOT_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_ROOT_NAMES = ("C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B")
_NEUTRAL_ROOT_NAMES = ("C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")

_BASE_QUALITIES: list[tuple[str, tuple[int, ...], float]] = [
    ("", (0, 4, 7), 1.0),  # Major triad
    ("m", (0, 3, 7), 1.0),  # Minor triad
    ("7", (0, 4, 7, 10), 0.96),  # Dominant 7
    ("maj7", (0, 4, 7, 11), 0.92),  # Major 7
    ("m7", (0, 3, 7, 10), 0.96),  # Minor 7
    ("dim", (0, 3, 6), 0.84),  # Diminished
]
_GUITAR_QUALITIES: list[tuple[str, tuple[int, ...], float]] = [
    ("5", (0, 7), 0.90),  # Power chord
    ("sus4", (0, 5, 7), 0.84),
    ("sus2", (0, 2, 7), 0.84),
    ("add9", (0, 4, 7, 2), 0.91),
]

_BASS_REGISTER_CUTOFF_MIDI = 53  # Below F3: boosts bass/Travis-picking fundamentals
_BASS_REGISTER_WEIGHT = 1.25


@dataclass(frozen=True)
class _ChordTemplate:
    suffix: str
    root_pc: int
    chord_pcs: frozenset[int]
    vector: tuple[float, ...]


def _build_templates(qualities: list[tuple[str, tuple[int, ...], float]]) -> list[_ChordTemplate]:
    templates: list[_ChordTemplate] = []
    for root_pc in range(12):
        for suffix, intervals, prior_weight in qualities:
            vec = [0.0] * 12
            for idx, interval in enumerate(intervals):
                pc = (root_pc + interval) % 12
                vec[pc] = 1.0 if idx == 0 else (0.85 if idx == 1 else 0.7)
            scale = prior_weight / (math.hypot(*vec) + 1e-8)
            templates.append(
                _ChordTemplate(
                    suffix=suffix,
                    root_pc=root_pc,
                    chord_pcs=frozenset((root_pc + iv) % 12 for iv in intervals),
                    vector=tuple(v * scale for v in vec),
                )
            )
    return templates


_BASE_TEMPLATES = _build_templates(_BASE_QUALITIES)
_GUITAR_TEMPLATES = _build_templates(_BASE_QUALITIES + _GUITAR_QUALITIES)


def _root_name(pc: int, sharps: int) -> str:
    if sharps < 0:
        return _FLAT_ROOT_NAMES[pc % 12]
    if sharps > 0:
        return _SHARP_ROOT_NAMES[pc % 12]
    return _NEUTRAL_ROOT_NAMES[pc % 12]


def _resolve_key_context(
    ref_score: music21.stream.Score,
    *,
    need_roman_key: bool,
) -> tuple[list[tuple[float, int]], int, music21.key.Key | None]:
    """Extract sorted KeySignature offsets/sharps and optional Key object for Roman numerals."""
    anchor: music21.stream.Stream = ref_score.parts[0] if ref_score.parts else ref_score
    ks_events: list[tuple[float, int, music21.key.Key | None]] = []
    for ks in anchor.recurse().getElementsByClass(music21.key.KeySignature):
        try:
            off = float(ks.getOffsetInHierarchy(anchor))
        except (music21.exceptions21.Music21Exception, ValueError, TypeError):
            off = 0.0
        explicit_key = ks if isinstance(ks, music21.key.Key) else None
        ks_events.append((off, int(ks.sharps), explicit_key))
    ks_events.sort(key=lambda item: item[0])

    analyzed_key: music21.key.Key | None = None
    if not ks_events or need_roman_key:
        try:
            analyzed = ref_score.analyze("key")
            if isinstance(analyzed, music21.key.Key):
                analyzed_key = analyzed
        except (music21.exceptions21.Music21Exception, ValueError, TypeError):
            analyzed_key = None

    if not ks_events:
        fallback_sharps = int(analyzed_key.sharps) if analyzed_key is not None else 0
        return [], fallback_sharps, analyzed_key

    sig_timeline = [(off, sharps) for off, sharps, _ in ks_events]
    roman_key: music21.key.Key | None = None
    if need_roman_key:
        first_sharps = ks_events[0][1]
        explicit_first = ks_events[0][2]
        if explicit_first is not None:
            roman_key = explicit_first
        elif analyzed_key is not None and int(analyzed_key.sharps) == first_sharps:
            roman_key = analyzed_key
        else:
            roman_key = music21.key.KeySignature(first_sharps).asKey()

    return sig_timeline, sig_timeline[0][1], roman_key


def _sharps_at_offset(
    sig_timeline: list[tuple[float, int]],
    fallback_sharps: int,
    beat_offset: float,
) -> int:
    if not sig_timeline:
        return fallback_sharps
    idx = bisect_right(sig_timeline, beat_offset + 1e-6, key=itemgetter(0)) - 1
    return sig_timeline[max(0, idx)][1]


def _compute_symbolic_chroma_by_window(
    score: music21.stream.Score,
    total_beats: float,
    window_beats: float,
) -> tuple[list[list[float]], list[int | None], list[float]]:
    """Accumulate pitch-class weights, lowest bass pitch class, and lowest bass duration per window."""
    n_windows = max(1, math.ceil(total_beats / window_beats))
    raw_chroma = [[0.0] * 12 for _ in range(n_windows)]
    lowest_midi: list[int | None] = [None] * n_windows
    lowest_dur: list[float] = [0.0] * n_windows

    try:
        sounding_score = score.toSoundingPitch()
    except (music21.exceptions21.Music21Exception, ValueError, TypeError):
        sounding_score = score

    flat = sounding_score.flatten()
    for el in iter_playable_notes(flat):
        onset = float(el.offset)
        dur = max(0.125, float(el.duration.quarterLength))
        end = onset + dur
        midis = element_midis(el)

        w_start = max(0, int(onset // window_beats))
        w_end = min(n_windows - 1, int(max(onset, end - 1e-6) // window_beats))
        for w in range(w_start, w_end + 1):
            overlap = min(end, (w + 1) * window_beats) - max(onset, w * window_beats)
            if overlap <= 0:
                continue
            row = raw_chroma[w]
            for midi_val in midis:
                bass_weight = _BASS_REGISTER_WEIGHT if midi_val < _BASS_REGISTER_CUTOFF_MIDI else 1.0
                row[midi_val % 12] += overlap * bass_weight
                cur_low = lowest_midi[w]
                if cur_low is None or midi_val < cur_low:
                    lowest_midi[w] = midi_val
                    lowest_dur[w] = overlap
                elif midi_val == cur_low:
                    lowest_dur[w] += overlap

    out: list[list[float]] = []
    for row in raw_chroma:
        # Sublinear sqrt per pitch class prevents repeated fast passing notes from overpowering chords
        weighted = [math.sqrt(v) if v > 0.0 else 0.0 for v in row]
        norm = math.hypot(*weighted)
        if norm > 1e-8:
            inv = 1.0 / norm
            weighted = [v * inv for v in weighted]
        out.append(weighted)
    bass_pcs = [m % 12 if m is not None else None for m in lowest_midi]
    return out, bass_pcs, lowest_dur


def _viterbi_chords(
    chroma_windows: list[list[float]],
    bass_pcs: list[int | None],
    switch_penalty: float = 0.14,
    templates: list[_ChordTemplate] = _BASE_TEMPLATES,
) -> list[_ChordTemplate | None]:
    """Decode smoothest chord template sequence via Viterbi over cosine similarity + bass priors."""
    n_win = len(chroma_windows)
    n_states = len(templates)

    sims: list[list[float]] = []
    energies: list[float] = []
    for t, chroma in enumerate(chroma_windows):
        energies.append(math.hypot(*chroma))
        bass = bass_pcs[t]
        row: list[float] = []
        for tmpl in templates:
            sim = sum(c * tv for c, tv in zip(chroma, tmpl.vector, strict=True))
            if bass == tmpl.root_pc:
                sim += 0.10
            elif bass is not None and bass in tmpl.chord_pcs:
                sim += 0.07
            if chroma[tmpl.root_pc] < 1e-3:
                sim -= 0.20
            if tmpl.suffix == "5" and (
                chroma[(tmpl.root_pc + 3) % 12] >= 0.15 or chroma[(tmpl.root_pc + 4) % 12] >= 0.15
            ):
                sim -= 0.12
            if tmpl.suffix in ("sus2", "sus4") and (
                chroma[(tmpl.root_pc + 3) % 12] >= 0.20
                or chroma[(tmpl.root_pc + 4) % 12] >= 0.20
                or chroma[(tmpl.root_pc + 7) % 12] < 1e-3
            ):
                sim -= 0.12
            if tmpl.suffix == "add9" and (
                chroma[(tmpl.root_pc + 7) % 12] < 1e-3 or chroma[(tmpl.root_pc + 2) % 12] < 1e-3
            ):
                sim -= 0.10
            if tmpl.suffix == "dim" and any(
                chroma[pc] >= 0.15 and pc not in tmpl.chord_pcs for pc in range(12)
            ):
                sim -= 0.08
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
        None if energies[t] < 1e-6 else templates[best_states[t]]
        for t in range(n_win)
    ]


def _format_window_label(
    tmpl: _ChordTemplate | None,
    bass_pc: int | None,
    bass_dur: float,
    window_beats: float,
    sharps: int,
    *,
    include_slash_chords: bool,
) -> str | None:
    if tmpl is None:
        return None
    root_str = _root_name(tmpl.root_pc, sharps)
    base_symbol = f"{root_str}{tmpl.suffix}"
    if (
        include_slash_chords
        and bass_pc is not None
        and bass_pc != tmpl.root_pc
        and (bass_pc in tmpl.chord_pcs or bass_dur >= 0.5 * window_beats)
    ):
        return f"{base_symbol}/{_root_name(bass_pc, sharps)}"
    return base_symbol


def annotate_chords(
    score: music21.stream.Score,
    harmonic_reference_score: music21.stream.Score | None = None,
    window_beats: float = 2.0,
    preserve_existing: bool = False,
    instrument: Instrument | None = None,
    include_slash_chords: bool = True,
    roman_numerals: bool = False,
) -> music21.stream.Score:
    """Annotate score with smoothed <harmony> ChordSymbol objects and optional Roman numerals."""
    if not math.isfinite(window_beats) or window_beats <= 0.0:
        raise ValueError(f"window_beats must be a positive finite number, got {window_beats}")
    if not score.parts:
        return score

    target_part = score.parts[0]
    existing_chords = list(target_part.recurse().getElementsByClass(music21.harmony.ChordSymbol))
    if preserve_existing and existing_chords:
        return score

    for existing in existing_chords:
        if existing.activeSite is not None:
            existing.activeSite.remove(existing)
    for existing_expr in list(target_part.recurse().getElementsByClass(music21.expressions.TextExpression)):
        if existing_expr.editorial.get("roman_numeral") and existing_expr.activeSite is not None:
            existing_expr.activeSite.remove(existing_expr)

    total_beats = float(target_part.highestTime)
    if total_beats <= 0:
        return score

    ref_score = harmonic_reference_score if harmonic_reference_score is not None else score
    sig_timeline, fallback_sharps, roman_key = _resolve_key_context(
        ref_score, need_roman_key=roman_numerals
    )

    chroma, bass_pcs, bass_durs = _compute_symbolic_chroma_by_window(
        ref_score, total_beats, window_beats
    )
    templates = _GUITAR_TEMPLATES if instrument == Instrument.GUITAR else _BASE_TEMPLATES
    decoded = _viterbi_chords(chroma, bass_pcs=bass_pcs, templates=templates)
    measures = list(target_part.getElementsByClass(music21.stream.Measure))
    measure_has_notes = {
        id(m): any(is_playable_note(el) for el in m.recurse().notes)
        for m in measures
    }

    last_symbol: str | None = None
    placed_rest_measures: set[int] = set()
    m_idx = 0
    n_measures = len(measures)
    for w_idx, tmpl in enumerate(decoded):
        beat_offset = w_idx * window_beats
        sharps = _sharps_at_offset(sig_timeline, fallback_sharps, beat_offset)
        label = _format_window_label(
            tmpl,
            bass_pcs[w_idx],
            bass_durs[w_idx],
            window_beats,
            sharps,
            include_slash_chords=include_slash_chords,
        )
        if label is None or label == last_symbol:
            continue
        try:
            cs = music21.harmony.ChordSymbol(label)
            cs.writeAsChord = False
        except (music21.harmony.ChordSymbolException, ValueError):
            continue

        rn_expr: music21.expressions.TextExpression | None = None
        if roman_numerals and roman_key is not None:
            active_rn_key = roman_key
            if sig_timeline and sharps != sig_timeline[0][1]:
                active_rn_key = music21.key.KeySignature(sharps).asKey()
            try:
                rn = music21.roman.romanNumeralFromChord(cs, active_rn_key)
                rn_expr = music21.expressions.TextExpression(rn.figure)
                rn_expr.placement = "below"
                rn_expr.editorial["roman_numeral"] = True
            except (music21.exceptions21.Music21Exception, ValueError, TypeError):
                rn_expr = None

        target: music21.stream.Stream = target_part
        insert_off = beat_offset
        should_insert = True
        if measures:
            while (
                m_idx < n_measures
                and float(measures[m_idx].offset) + float(measures[m_idx].duration.quarterLength) - 1e-6
                <= beat_offset
            ):
                m_idx += 1
            if m_idx < n_measures and float(measures[m_idx].offset) - 1e-6 <= beat_offset:
                m = measures[m_idx]
                target = m
                if not measure_has_notes[id(m)]:
                    should_insert = id(m) not in placed_rest_measures
                    placed_rest_measures.add(id(m))
                    insert_off = 0.0
                else:
                    insert_off = max(0.0, beat_offset - float(m.offset))
        if should_insert:
            target.insert(insert_off, cs)
            if rn_expr is not None:
                target.insert(insert_off, rn_expr)
        last_symbol = label

    return score
