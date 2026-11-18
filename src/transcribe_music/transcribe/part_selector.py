"""Part selection, Piano Grand Staff splitting, and instrument notation adaptation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from fractions import Fraction

import music21

from transcribe_music.ingest.digital import summarize_score_parts
from transcribe_music.models import INSTRUMENT_SPECS, Instrument, PartInfo
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    element_midis,
    fill_stream_rest_gaps,
    get_guitar_track_meta,
    iter_playable_notes,
    make_clef,
    replace_or_insert_initial_clef,
)

_INSTRUMENT_KEYWORDS: dict[Instrument, tuple[str, ...]] = {
    Instrument.PIANO: ("piano", "klavier", "pno", "keyboard", "rh", "lh", "right", "left"),
    Instrument.GUITAR: (
        "guitar",
        "chitarra",
        "gitarre",
        "guitare",
        "gtr",
        "lead",
        "rhythm",
        "acoustic",
        "electric",
        "nylon",
        "steel",
    ),
    Instrument.VIOLIN: ("violin", "violino", "violine", "violon", "vln", "vl"),
    Instrument.CELLO: ("cello", "violoncello", "violoncelle", "vc"),
}

_ENSEMBLE_PENALTY_TOKENS = ("tutti", "ripieno", "orchestra", "continuo", "bassi", "rip.")


@dataclass(frozen=True)
class SelectedScore:
    """Target-instrument score isolated and adapted for enrichment/export."""

    score: music21.stream.Score
    selected_part: str
    warnings: list[str] = field(default_factory=list)


def _format_part_choices(infos: list[PartInfo]) -> str:
    return ", ".join(f"{p.index}: {p.name}" for p in infos)


def _match_by_explicit_selector(
    parts: list[music21.stream.Part],
    infos: list[PartInfo],
    instrument: Instrument,
    selector: str,
) -> tuple[list[int], list[str]]:
    raw = selector.strip()
    if not raw:
        raise ValueError("Part selector cannot be empty.")

    warnings: list[str] = []
    # Support explicit 1-based index or comma-separated indices (e.g. "1" or "1,2")
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if tokens and all(tok.isdigit() for tok in tokens):
        indices: list[int] = []
        for tok in tokens:
            idx = int(tok) - 1
            if not (0 <= idx < len(parts)):
                raise ValueError(
                    f"Part index {tok} is out of range for score with {len(parts)} parts "
                    f"({_format_part_choices(infos)})."
                )
            if idx not in indices:
                indices.append(idx)
        return indices, warnings

    needle = raw.lower()
    matched = [
        i
        for i, info in enumerate(infos)
        if needle in info.name.lower()
        or (info.instrument_name and needle in info.instrument_name.lower())
        or needle in info.part_id.lower()
    ]
    if not matched:
        raise ValueError(
            f"No part matching {selector!r}. Available parts: {_format_part_choices(infos)}."
        )

    if instrument == Instrument.PIANO:
        if len(matched) > 2:
            warnings.append(
                f"Part selector {selector!r} matched {len(matched)} parts; kept the first two for Grand Staff."
            )
            matched = matched[:2]
        return matched, warnings

    if len(matched) > 1:
        matched_infos = [infos[i] for i in matched]
        best = max(
            matched_infos,
            key=lambda p: (
                "solo" in p.name.lower()
                or (instrument == Instrument.GUITAR and "lead" in p.name.lower()),
                p.note_count,
            ),
        )
        warnings.append(
            f"Part selector {selector!r} matched multiple parts "
            f"({', '.join(p.name for p in matched_infos)}); selected {best.index}: {best.name}."
        )
        return [best.index - 1], warnings

    return matched, warnings


def _score_part_for_instrument(
    info: PartInfo,
    instrument: Instrument,
    *,
    is_sounding_pitch_source: bool,
    guitar_meta: GuitarTrackMeta | None = None,
) -> tuple[float, bool]:
    spec = INSTRUMENT_SPECS[instrument]
    keywords = _INSTRUMENT_KEYWORDS[instrument]
    haystack = " ".join(
        part for part in (info.name, info.instrument_name or "", info.part_id) if part
    ).lower()

    score = 0.0
    exact_match = False

    if info.midi_program is not None and info.midi_program in spec.gm_programs:
        score += 100.0
        exact_match = True

    if any(kw in haystack for kw in keywords):
        score += 80.0
        exact_match = True

    if "solo" in haystack:
        score += 25.0
    if instrument == Instrument.GUITAR:
        if "lead" in haystack:
            score += 15.0
        if "rhythm" in haystack or "backing" in haystack:
            score -= 5.0
    if any(tok in haystack for tok in _ENSEMBLE_PENALTY_TOKENS):
        score -= 20.0

    if info.pitch_min is not None and info.pitch_max is not None:
        if instrument == Instrument.GUITAR and guitar_meta is not None:
            lo = min(guitar_meta.open_string_midi) + guitar_meta.capo_fret
            hi = max(guitar_meta.open_string_midi) + guitar_meta.capo_fret + 24
        else:
            lo, hi = spec.midi_min, spec.midi_max
        if instrument == Instrument.GUITAR and not is_sounding_pitch_source:
            lo, hi = lo + 12, hi + 12
        if info.pitch_min >= lo - 2 and info.pitch_max <= hi + 2:
            score += 30.0
        else:
            overlap = max(0, min(info.pitch_max, hi) - max(info.pitch_min, lo))
            span = max(1, info.pitch_max - info.pitch_min)
            score += 20.0 * (overlap / span)

    if info.clef:
        clef_l = info.clef.lower()
        if (
            (instrument == Instrument.CELLO and ("bass" in clef_l or "tenor" in clef_l))
            or (instrument == Instrument.VIOLIN and "treble" in clef_l and "8vb" not in clef_l)
            or (instrument == Instrument.GUITAR and "treble8vb" in clef_l)
        ):
            score += 8.0

    score += min(info.note_count, 5000) * 0.001
    if info.note_count == 0:
        score -= 50.0

    return score, exact_match


def _auto_select_parts(
    parts: list[music21.stream.Part],
    infos: list[PartInfo],
    instrument: Instrument,
    *,
    is_sounding_pitch_source: bool,
) -> tuple[list[int], list[str]]:
    if len(parts) == 1:
        return [0], []

    warnings: list[str] = []
    scored = [
        (
            idx,
            *_score_part_for_instrument(
                info,
                instrument,
                is_sounding_pitch_source=is_sounding_pitch_source,
                guitar_meta=get_guitar_track_meta(parts[idx]),
            ),
        )
        for idx, info in enumerate(infos)
    ]

    if instrument == Instrument.PIANO:
        piano_matches = [idx for idx, sc, exact in scored if exact and sc > 0]
        if len(piano_matches) >= 2:
            piano_matches.sort(key=lambda i: (-scored[i][1], i))
            chosen = sorted(piano_matches[:2])
            return chosen, warnings
        if len(parts) == 2 and all(
            any(k in (infos[i].clef or "").lower() for k in ("treble", "bass")) for i in (0, 1)
        ):
            return [0, 1], warnings

    best_idx, _, best_exact = max(scored, key=lambda item: (item[1], -item[0]))
    if not best_exact:
        warnings.append(
            f"No part explicitly matched {INSTRUMENT_SPECS[instrument].display_name}; "
            f"auto-selected part {infos[best_idx].index} ({infos[best_idx].name}). "
            f"Available parts: {_format_part_choices(infos)}."
        )
    return [best_idx], warnings


def _insert_grouped_events(
    measure: music21.stream.Measure,
    events: list[tuple[Fraction, Fraction, list[int], music21.tie.Tie | None]],
    bar_ql: Fraction,
) -> None:
    by_offset: dict[Fraction, list[tuple[Fraction, int, music21.tie.Tie | None]]] = {}
    for off, dur, pitches, tie_obj in events:
        if off >= bar_ql:
            continue
        clipped_dur = min(dur, bar_ql - off)
        if clipped_dur <= 0:
            continue
        for p in pitches:
            by_offset.setdefault(off, []).append((clipped_dur, p, tie_obj))

    sorted_offsets = sorted(by_offset)
    for idx, off in enumerate(sorted_offsets):
        group = by_offset[off]
        next_off = sorted_offsets[idx + 1] if idx + 1 < len(sorted_offsets) else bar_ql
        max_allowed = next_off - off
        if max_allowed <= 0:
            continue
        dur = min(max(item[0] for item in group), max_allowed)
        pitches = sorted({item[1] for item in group})
        ties = [item[2] for item in group if item[2] is not None]
        if len(pitches) == 1:
            el: music21.note.Note | music21.chord.Chord = music21.note.Note(
                pitches[0], quarterLength=dur
            )
        else:
            el = music21.chord.Chord(pitches, quarterLength=dur)
        if ties:
            el.tie = copy.deepcopy(ties[0])
        measure.insert(off, el)

    fill_stream_rest_gaps(measure, bar_ql)


def _split_single_part_piano(src_part: music21.stream.Part) -> tuple[music21.stream.Part, music21.stream.Part]:
    """Split a single-part piano stream into Grand Staff RH (Treble) and LH (Bass)."""
    working = copy.deepcopy(src_part)
    if not list(working.getElementsByClass(music21.stream.Measure)):
        working.makeMeasures(inPlace=True)

    rh = music21.stream.Part(id="RH")
    lh = music21.stream.Part(id="LH")
    rh.partName = "Piano RH"
    lh.partName = "Piano LH"

    active_bar_ql = Fraction(4, 1)
    for m_idx, m in enumerate(working.getElementsByClass(music21.stream.Measure)):
        ts = m.getElementsByClass(music21.meter.TimeSignature).first()
        if ts is not None:
            active_bar_ql = Fraction(ts.barDuration.quarterLength).limit_denominator(24)

        rh_m = music21.stream.Measure(number=m.number)
        lh_m = music21.stream.Measure(number=m.number)

        if m_idx == 0:
            rh_m.insert(0, music21.clef.TrebleClef())
            lh_m.insert(0, music21.clef.BassClef())

        for meta_cls in (
            music21.meter.TimeSignature,
            music21.key.KeySignature,
            music21.tempo.MetronomeMark,
            music21.expressions.RehearsalMark,
            music21.bar.Barline,
        ):
            for el in m.getElementsByClass(meta_cls):
                if isinstance(
                    el, (music21.tempo.MetronomeMark, music21.expressions.RehearsalMark)
                ):
                    rh_m.insert(el.offset, copy.deepcopy(el))
                else:
                    rh_m.insert(el.offset, copy.deepcopy(el))
                    lh_m.insert(el.offset, copy.deepcopy(el))

        for h in m.getElementsByClass(music21.harmony.ChordSymbol):
            rh_m.insert(h.offset, copy.deepcopy(h))

        rh_events: list[tuple[Fraction, Fraction, list[int], music21.tie.Tie | None]] = []
        lh_events: list[tuple[Fraction, Fraction, list[int], music21.tie.Tie | None]] = []

        for el in iter_playable_notes(m):
            off = Fraction(float(el.getOffsetInHierarchy(m))).limit_denominator(24)
            dur = max(Fraction(1, 8), Fraction(float(el.duration.quarterLength)).limit_denominator(24))
            pitches = sorted(element_midis(el))
            if not pitches:
                continue
            if len(pitches) == 1:
                (rh_events if pitches[0] >= 60 else lh_events).append((off, dur, pitches, el.tie))
            else:
                split_pt = 0.5 * (pitches[0] + pitches[-1]) if pitches[-1] - pitches[0] > 12 else 60
                r_p = [p for p in pitches if p >= split_pt]
                l_p = [p for p in pitches if p < split_pt]
                if r_p:
                    rh_events.append((off, dur, r_p, el.tie))
                if l_p:
                    lh_events.append((off, dur, l_p, el.tie))

        _insert_grouped_events(rh_m, rh_events, active_bar_ql)
        _insert_grouped_events(lh_m, lh_events, active_bar_ql)
        rh.append(rh_m)
        lh.append(lh_m)

    return rh, lh


def _adapt_guitar_part(
    part: music21.stream.Part,
    *,
    is_sounding_pitch_source: bool,
) -> None:
    """Ensure Guitar part uses written 8vb pitches (E3=52 -> sounds E2=40) and Treble8vbClef."""
    inst = part.recurse().getElementsByClass(music21.instrument.Instrument).first()
    existing_transposition = getattr(inst, "transposition", None) if inst is not None else None
    first_clef = part.recurse().getElementsByClass(music21.clef.Clef).first()

    pitches = [m for el in iter_playable_notes(part) for m in element_midis(el)]
    min_pitch = min(pitches) if pitches else None
    meta = get_guitar_track_meta(part)
    written_floor = (
        (min(meta.open_string_midi) + meta.capo_fret + 12) if meta is not None else 52
    )

    needs_octave_up = is_sounding_pitch_source or (
        existing_transposition is None
        and not isinstance(first_clef, music21.clef.Treble8vbClef)
        and min_pitch is not None
        and min_pitch < written_floor
    )
    if needs_octave_up:
        for el in iter_playable_notes(part):
            if isinstance(el, music21.note.Note):
                el.pitch.midi += 12
            else:
                for p in el.pitches:
                    p.midi += 12

    replace_or_insert_initial_clef(
        part,
        music21.clef.Treble8vbClef(),
        force_replace=(
            is_sounding_pitch_source
            or isinstance(first_clef, music21.clef.BassClef)
            or not isinstance(first_clef, music21.clef.Treble8vbClef)
        ),
    )
    part.atSoundingPitch = False


def select_and_adapt_parts(
    score: music21.stream.Score,
    instrument: Instrument,
    part_selector: str | None = None,
    *,
    is_sounding_pitch_source: bool = False,
) -> SelectedScore:
    """Isolate target part(s) from a Score and adapt clef/transposition/Grand Staff layout."""
    parts = list(score.parts)
    if not parts:
        raise ValueError("Input score contains zero parts.")

    infos = summarize_score_parts(score)
    if part_selector is not None:
        chosen_indices, warnings = _match_by_explicit_selector(
            parts, infos, instrument, part_selector
        )
    else:
        chosen_indices, warnings = _auto_select_parts(
            parts,
            infos,
            instrument,
            is_sounding_pitch_source=is_sounding_pitch_source,
        )

    out_score = music21.stream.Score()
    if score.metadata is not None:
        out_score.metadata = copy.deepcopy(score.metadata)

    selected_labels = ", ".join(f"{infos[i].index}: {infos[i].name}" for i in chosen_indices)

    if instrument == Instrument.PIANO:
        if len(chosen_indices) == 1:
            rh, lh = _split_single_part_piano(parts[chosen_indices[0]])
            out_score.insert(0, rh)
            out_score.insert(0, lh)
        else:
            for staff_idx, idx in enumerate(chosen_indices[:2]):
                p_copy = copy.deepcopy(parts[idx])
                replace_or_insert_initial_clef(
                    p_copy,
                    make_clef("grand", staff_idx),
                    force_replace=is_sounding_pitch_source,
                )
                out_score.insert(0, p_copy)
        return SelectedScore(
            score=out_score,
            selected_part=selected_labels,
            warnings=warnings,
        )

    part_copy = copy.deepcopy(parts[chosen_indices[0]])
    if instrument == Instrument.GUITAR:
        _adapt_guitar_part(part_copy, is_sounding_pitch_source=is_sounding_pitch_source)
    else:
        replace_or_insert_initial_clef(
            part_copy,
            make_clef(INSTRUMENT_SPECS[instrument].clef_mode),
            force_replace=is_sounding_pitch_source,
        )

    out_score.insert(0, part_copy)
    return SelectedScore(
        score=out_score,
        selected_part=selected_labels,
        warnings=warnings,
    )
