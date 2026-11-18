"""Plain-text ASCII guitar tablature (.tab / .txt) parser."""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import music21

from transcribe_music.ingest._io import MAX_MEMBER_BYTES
from transcribe_music.models import INSTRUMENT_SPECS, Instrument
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    attach_guitar_techniques,
    build_fretted_element,
    fill_stream_rest_gaps,
    set_guitar_track_meta,
)

_TAB_LINE_RE = re.compile(
    r"^\s*(?P<label>[A-Ga-g][#b]?\d?)\s*[:|]\s*(?P<body>[-0-9hHpPbBrRxX/\\~^()*.\s|]+)\s*$"
)
_HEADER_RE = re.compile(r"^\s*(?P<key>[A-Za-z]+)\s*:\s*(?P<val>.+?)\s*$")
_DEFAULT_GUITAR_TUNING = INSTRUMENT_SPECS[Instrument.GUITAR].open_string_midi

_PC_BY_NAME: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}


def _parse_pitch_label(token: str) -> tuple[int, int | None]:
    clean = token.strip()
    m = re.match(r"^([A-Ga-g])([#b]?)(-?\d+)?$", clean)
    if not m:
        raise ValueError(f"Unrecognized string tuning label: {token!r}")
    name = (m.group(1).upper() + m.group(2).upper()).replace("B", "B")
    # Preserve flat 'b' vs note 'B'
    note_key = m.group(1).upper() + ("#" if m.group(2) == "#" else ("B" if m.group(2).lower() == "b" else ""))
    if note_key not in _PC_BY_NAME:
        raise ValueError(f"Unrecognized pitch name: {name!r}")
    pc = _PC_BY_NAME[note_key]
    octave = int(m.group(3)) if m.group(3) is not None else None
    return pc, octave


def _resolve_tuning(labels: list[str]) -> tuple[int, ...]:
    """Resolve top-to-bottom tab string labels into descending MIDI pitches (string 1..N)."""
    resolved: list[int] = []
    for idx, lbl in enumerate(labels):
        pc, octave = _parse_pitch_label(lbl)
        if octave is not None:
            resolved.append(12 * (octave + 1) + pc)
            continue
        if idx == 0:
            # Place top string in standard treble guitar range [59, 67] (B3..G4)
            cand = 60 + pc  # C4..B4
            if cand > 67:
                cand -= 12
            if cand < 58:
                cand += 12
            resolved.append(cand)
        else:
            prev = resolved[-1]
            # Find highest octave of `pc` strictly below `prev`
            k = (prev - 1 - pc) // 12
            resolved.append(12 * k + pc)
    return tuple(resolved)


def _split_bar_segments(body: str) -> list[str]:
    stripped = body.strip()
    if "|" not in stripped:
        return [stripped] if stripped else []
    raw_parts = stripped.split("|")
    if stripped.startswith("|"):
        raw_parts = raw_parts[1:]
    if stripped.endswith("|"):
        raw_parts = raw_parts[:-1]
    return [seg for seg in raw_parts if seg]


def _snap_offset(raw: float, bar_ql: Fraction, step: Fraction = Fraction(1, 4)) -> Fraction:
    units = round(raw / float(step))
    snapped = units * step
    if snapped < 0:
        return Fraction(0, 1)
    if snapped >= bar_ql:
        return max(Fraction(0, 1), bar_ql - step)
    return snapped


def parse_ascii_tab_file(path: Path) -> tuple[music21.stream.Score, list[str]]:
    """Parse a plain-text ASCII guitar tablature file into a measured music21 Score."""
    if path.stat().st_size > MAX_MEMBER_BYTES:
        raise ValueError(f"ASCII tab file {path.name} exceeds {MAX_MEMBER_BYTES} byte limit.")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    title = path.stem
    artist: str | None = None
    tempo_bpm: float | None = None
    ts_str = "4/4"
    capo_fret = 0
    explicit_tuning: tuple[int, ...] | None = None

    systems: list[tuple[list[str], list[str]]] = []
    cur_labels: list[str] = []
    cur_bodies: list[str] = []

    def _flush_system() -> None:
        nonlocal cur_labels, cur_bodies
        if 4 <= len(cur_bodies) <= 8:
            systems.append((cur_labels, cur_bodies))
        cur_labels = []
        cur_bodies = []

    for line in lines:
        m = _TAB_LINE_RE.match(line)
        if m and ("-" in m.group("body") or "|" in m.group("body")):
            cur_labels.append(m.group("label"))
            cur_bodies.append(m.group("body"))
            continue

        _flush_system()
        hm = _HEADER_RE.match(line)
        if hm:
            key = hm.group("key").lower()
            val = hm.group("val").strip()
            if key in {"title", "song"} and val:
                title = val
            elif key in {"artist", "composer", "by"} and val:
                artist = val
            elif key in {"tempo", "bpm"}:
                num_m = re.search(r"\d+(?:\.\d+)?", val)
                if num_m:
                    tempo_bpm = float(num_m.group(0))
            elif key in {"time", "meter"}:
                ts_m = re.match(r"^(\d+/\d+)", val)
                if ts_m:
                    ts_str = ts_m.group(1)
            elif key == "capo":
                num_m = re.search(r"\d+", val)
                if num_m:
                    capo_fret = int(num_m.group(0))
            elif key == "tuning":
                tokens = [tok for tok in re.split(r"[\s,-]+", val) if tok]
                if 4 <= len(tokens) <= 8:
                    try:
                        resolved_low_to_high = _resolve_tuning(list(reversed(tokens)))
                        explicit_tuning = resolved_low_to_high
                    except ValueError:
                        pass

    _flush_system()
    if not systems:
        raise ValueError(f"No valid ASCII guitar tab staves found in {path.name}.")

    base_tuning = explicit_tuning or _resolve_tuning(systems[0][0]) or _DEFAULT_GUITAR_TUNING
    ts_obj = music21.meter.TimeSignature(ts_str)
    bar_ql = Fraction(ts_obj.barDuration.quarterLength).limit_denominator(24)

    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title, composer=artist)
    part = music21.stream.Part(id="P1")
    part.partName = "Guitar"
    m21_inst = music21.instrument.AcousticGuitar()
    m21_inst.partName = "Guitar"
    part.insert(0, m21_inst)
    set_guitar_track_meta(part, GuitarTrackMeta(open_string_midi=base_tuning, capo_fret=capo_fret))

    measure_num = 1
    prev_el: music21.note.Note | music21.chord.Chord | None = None

    for labels, bodies in systems:
        sys_tuning = explicit_tuning or _resolve_tuning(labels)
        if len(sys_tuning) != len(base_tuning):
            sys_tuning = base_tuning[: len(bodies)]

        bar_rows = [_split_bar_segments(b) for b in bodies]
        n_bars = min((len(r) for r in bar_rows), default=0)
        for b_idx in range(n_bars):
            segments = [bar_rows[s_idx][b_idx] for s_idx in range(len(bodies))]
            repeat_start = any(seg.startswith("*") for seg in segments)
            repeat_end = any(seg.endswith("*") for seg in segments)
            cleaned_segs = [seg.strip("*") for seg in segments]
            width = max((len(seg) for seg in cleaned_segs), default=1)
            if width <= 0:
                continue

            events_by_col: dict[int, list[tuple[int, int, str | None, float]]] = {}
            for s_idx, seg in enumerate(cleaned_segs):
                padded = seg.ljust(width, "-")
                col = 0
                while col < width:
                    ch = padded[col]
                    if ch.isdigit():
                        end_col = col + 1
                        while end_col < width and padded[end_col].isdigit():
                            end_col += 1
                        fret = int(padded[col:end_col])
                        if 0 <= fret <= 24:
                            prev_ch = padded[col - 1] if col > 0 else "-"
                            next_ch = padded[end_col] if end_col < width else "-"
                            tech: str | None = None
                            if prev_ch in {"h", "H"}:
                                tech = "hammer"
                            elif prev_ch in {"p", "P"}:
                                tech = "pull"
                            elif prev_ch in {"/", "\\"}:
                                tech = "slide"
                            bend_semis = 2.0 if next_ch in {"b", "B"} else 0.0
                            events_by_col.setdefault(col, []).append(
                                (s_idx + 1, fret, tech, bend_semis)
                            )
                        col = end_col
                    else:
                        col += 1

            m21_measure = music21.stream.Measure(number=measure_num)
            if measure_num == 1:
                m21_measure.insert(0, music21.meter.TimeSignature(ts_str))
                if tempo_bpm is not None:
                    m21_measure.insert(0, music21.tempo.MetronomeMark(number=tempo_bpm))
            if repeat_start:
                m21_measure.leftBarline = music21.bar.Repeat(direction="start")
            if repeat_end:
                m21_measure.rightBarline = music21.bar.Repeat(direction="end")

            sorted_cols = sorted(events_by_col)
            if not sorted_cols:
                m21_measure.insert(0, music21.note.Rest(quarterLength=bar_ql))
                prev_el = None
            else:
                step = Fraction(1, 4) if len(sorted_cols) > int(bar_ql * 2) else Fraction(1, 2)
                offsets: list[Fraction] = []
                for col in sorted_cols:
                    off = _snap_offset((col / max(1, width)) * float(bar_ql), bar_ql, step=step)
                    if offsets and off <= offsets[-1]:
                        off = min(bar_ql - step, offsets[-1] + step)
                    offsets.append(off)

                # Deduplicate if bar is ultra-dense
                dedup_cols: list[tuple[Fraction, list[tuple[int, int, str | None, float]]]] = []
                for off, col in zip(offsets, sorted_cols, strict=True):
                    if dedup_cols and off == dedup_cols[-1][0]:
                        dedup_cols[-1][1].extend(events_by_col[col])
                    elif off < bar_ql:
                        dedup_cols.append((off, list(events_by_col[col])))

                for idx, (off, evs) in enumerate(dedup_cols):
                    next_off = dedup_cols[idx + 1][0] if idx + 1 < len(dedup_cols) else bar_ql
                    ql = max(Fraction(1, 8), min(next_off - off, bar_ql - off))
                    specs: list[tuple[int, int | None, int | None]] = []
                    hammer = False
                    pull = False
                    slide = False
                    bend_semis = 0.0
                    for s_num, fret, tech, b_semi in evs:
                        if s_num <= len(sys_tuning):
                            midi_val = sys_tuning[s_num - 1] + fret + capo_fret
                            if 0 <= midi_val <= 127:
                                specs.append((midi_val, s_num, fret))
                        if tech == "hammer":
                            hammer = True
                        elif tech == "pull":
                            pull = True
                        elif tech == "slide":
                            slide = True
                        bend_semis = max(bend_semis, b_semi)

                    if not specs:
                        continue
                    el = build_fretted_element(specs, music21.duration.Duration(ql))
                    attach_guitar_techniques(
                        part,
                        m21_measure,
                        prev_el,
                        el,
                        off,
                        hammer_from_prev=hammer,
                        pull_from_prev=pull,
                        slide_from_prev=slide,
                        bend_semitones=bend_semis,
                    )
                    m21_measure.insert(off, el)
                    prev_el = el

                fill_stream_rest_gaps(m21_measure, bar_ql)

            part.append(m21_measure)
            measure_num += 1

    if measure_num == 1:
        raise ValueError(f"No readable tab measures found in {path.name}.")

    score.insert(0, part)
    return score, []
