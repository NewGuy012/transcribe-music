"""Native ingestion for Guitar Pro 3-5 (.gp3/.gp4/.gp5 via pyguitarpro) and GP7/8 (.gp GPIF XML)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import music21

from transcribe_music.ingest._io import read_bounded_zip_member
from transcribe_music.models import INSTRUMENT_SPECS, Instrument
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    attach_guitar_techniques,
    build_fretted_element,
    element_midis,
    mark_tie,
    set_guitar_track_meta,
    try_int,
)

_NOTE_VALUE_QL: dict[str, Fraction] = {
    "Whole": Fraction(4, 1),
    "Half": Fraction(2, 1),
    "Quarter": Fraction(1, 1),
    "Eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
}
_DEFAULT_GUITAR_TUNING = INSTRUMENT_SPECS[Instrument.GUITAR].open_string_midi


@dataclass(frozen=True)
class _GpBeatEvent:
    ql: Fraction
    notes: tuple[tuple[int, int | None, int | None], ...]
    is_tie: bool = False
    hop_origin: bool = False
    hop_dest: bool = False
    slide_origin: bool = False
    bend_semis: float = 0.0
    harmonic: bool = False
    palm_mute: bool = False


@dataclass(frozen=True)
class _GpMeasureData:
    number: int
    ts: tuple[int, int] | None
    fifths: int | None
    repeat_start: bool
    repeat_close: int | None
    voices: tuple[tuple[int, tuple[_GpBeatEvent, ...]], ...]


@dataclass(frozen=True)
class _GpTrackData:
    part_id: str
    name: str
    gm_program: int
    open_strings: tuple[int, ...]
    capo_fret: int
    is_guitar: bool
    measures: tuple[_GpMeasureData, ...]


def _m21_instrument(program: int, track_name: str) -> music21.instrument.Instrument:
    clamped = max(0, min(127, program))
    try:
        inst = music21.instrument.instrumentFromMidiProgram(clamped)
    except music21.instrument.InstrumentException:
        inst = music21.instrument.Instrument()
    if track_name:
        inst.partName = track_name
    return inst


def _build_gp_score(
    *,
    title: str,
    composer: str | None,
    tempo_bpm: float | None,
    skipped_percussion: int,
    tracks: list[_GpTrackData],
    source_name: str,
) -> tuple[music21.stream.Score, list[str]]:
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title, composer=composer)
    warnings = (
        [f"Skipped {skipped_percussion} percussion track(s) from {source_name}."]
        if skipped_percussion > 0
        else []
    )

    for t_idx, track in enumerate(tracks):
        part = music21.stream.Part(id=track.part_id)
        part.partName = track.name
        part.insert(0, _m21_instrument(track.gm_program, track.name))
        if track.is_guitar and track.open_strings:
            set_guitar_track_meta(
                part,
                GuitarTrackMeta(open_string_midi=track.open_strings, capo_fret=track.capo_fret),
            )

        prev_ts: tuple[int, int] = (4, 4)
        emitted_ts: tuple[int, int] | None = None
        emitted_ks: int | None = None
        prev_el_by_voice: dict[int, music21.note.Note | music21.chord.Chord | None] = {}
        pending_by_voice: dict[int, tuple[bool, bool, bool]] = {}

        for m_idx, gm in enumerate(track.measures):
            m21_m = music21.stream.Measure(number=gm.number)
            ts_sig = gm.ts if (gm.ts and gm.ts[0] > 0 and gm.ts[1] > 0) else prev_ts
            prev_ts = ts_sig
            if emitted_ts != ts_sig:
                m21_m.insert(0, music21.meter.TimeSignature(f"{ts_sig[0]}/{ts_sig[1]}"))
                emitted_ts = ts_sig
            bar_ql = Fraction(ts_sig[0] * 4, ts_sig[1])

            if gm.fifths is not None:
                fifths = max(-7, min(7, gm.fifths))
                if emitted_ks != fifths:
                    m21_m.insert(0, music21.key.KeySignature(fifths))
                    emitted_ks = fifths

            if gm.repeat_start:
                m21_m.leftBarline = music21.bar.Repeat(direction="start")
            if gm.repeat_close is not None and gm.repeat_close >= 0:
                rep = music21.bar.Repeat(direction="end")
                if gm.repeat_close > 0:
                    rep.times = gm.repeat_close
                m21_m.rightBarline = rep

            if m_idx == 0 and t_idx == 0 and tempo_bpm and tempo_bpm > 0:
                m21_m.insert(0, music21.tempo.MetronomeMark(number=float(tempo_bpm)))

            if not gm.voices:
                m21_m.insert(0, music21.note.Rest(quarterLength=bar_ql))
                part.append(m21_m)
                continue

            multi_voice = len(gm.voices) > 1
            for v_idx, beats in gm.voices:
                target: music21.stream.Stream = (
                    music21.stream.Voice(id=str(v_idx + 1)) if multi_voice else m21_m
                )
                offset_ql = Fraction(0, 1)
                prev_el = prev_el_by_voice.get(v_idx)
                pend_hop, pend_slide, prev_pm = pending_by_voice.get(v_idx, (False, False, False))

                for beat in beats:
                    ql = beat.ql
                    if ql <= 0:
                        continue
                    if offset_ql >= bar_ql:
                        break
                    if offset_ql + ql > bar_ql:
                        ql = bar_ql - offset_ql

                    dur_obj = music21.duration.Duration(ql)
                    if not beat.notes:
                        target.insert(offset_ql, music21.note.Rest(duration=dur_obj))
                        prev_el = None
                        pend_hop = pend_slide = prev_pm = False
                        offset_ql += ql
                        continue

                    el = build_fretted_element(list(beat.notes), dur_obj)
                    if beat.is_tie and prev_el is not None:
                        mark_tie(prev_el, el)

                    hammer_now = pull_now = False
                    if (pend_hop or beat.hop_dest) and prev_el is not None:
                        if max(element_midis(el)) >= max(element_midis(prev_el)):
                            hammer_now = True
                        else:
                            pull_now = True

                    attach_guitar_techniques(
                        part,
                        target,
                        prev_el,
                        el,
                        offset_ql,
                        hammer_from_prev=hammer_now,
                        pull_from_prev=pull_now,
                        slide_from_prev=pend_slide,
                        bend_semitones=beat.bend_semis,
                        harmonic=beat.harmonic,
                        palm_mute=beat.palm_mute,
                        prev_palm_mute=prev_pm,
                    )
                    target.insert(offset_ql, el)
                    prev_el = el
                    pend_hop = beat.hop_origin
                    pend_slide = beat.slide_origin
                    prev_pm = beat.palm_mute
                    offset_ql += ql

                if offset_ql < bar_ql:
                    target.insert(offset_ql, music21.note.Rest(quarterLength=bar_ql - offset_ql))

                prev_el_by_voice[v_idx] = prev_el
                pending_by_voice[v_idx] = (pend_hop, pend_slide, prev_pm)
                if multi_voice:
                    m21_m.insert(0, target)

            part.append(m21_m)
        score.insert(0, part)

    return score, warnings


def _gp35_bend_semitones(effect: object) -> float:
    if not getattr(effect, "isBend", False):
        return 0.0
    bend = getattr(effect, "bend", None)
    if bend is None:
        return 0.0
    points = getattr(bend, "points", None) or []
    if points:
        max_q = max(float(getattr(pt, "value", 0) or 0) for pt in points)
        if max_q > 0:
            return max_q / 2.0
    raw_val = float(getattr(bend, "value", 0) or 0)
    return raw_val / 50.0 if raw_val > 0 else 0.0


def parse_gp35_file(path: Path) -> tuple[music21.stream.Score, list[str]]:
    """Parse a Guitar Pro 3-5 (.gp3/.gp4/.gp5) file via optional `pyguitarpro`."""
    try:
        import guitarpro
        import guitarpro.models as gm
    except ImportError as exc:
        raise RuntimeError(
            f"Loading {path.name} requires the optional 'gp' extra (`pyguitarpro`). "
            "Install with: uv sync --extra gp"
        ) from exc

    song = guitarpro.parse(str(path))
    non_percussion = [
        tr
        for tr in song.tracks
        if not tr.isPercussionTrack and getattr(tr.channel, "channel", 0) != 9
    ]
    if not non_percussion:
        raise ValueError(f"{path.name} contains only percussion tracks; no pitched tracks to convert.")

    quarter_ticks = gm.Duration.quarterTime or 960
    slide_out_types = {gm.SlideType.shiftSlideTo, gm.SlideType.legatoSlideTo}
    tracks: list[_GpTrackData] = []

    for track in non_percussion:
        gm_prog = int(getattr(track.channel, "instrument", 25))
        capo_offset = int(getattr(track, "offset", 0) or 0)
        is_guitar = 24 <= gm_prog <= 31
        open_strings = tuple(
            int(s.value)
            for s in sorted(
                getattr(track, "strings", None) or (),
                key=lambda st: int(getattr(st, "number", 0)),
            )
        )
        n_strings = len(open_strings) if open_strings else 6

        measures: list[_GpMeasureData] = []
        for m_idx, gp_m in enumerate(track.measures):
            header = gp_m.header
            ts_num = int(header.timeSignature.numerator)
            ts_den = int(header.timeSignature.denominator.value)
            fifths = int(header.keySignature.value[0]) if hasattr(header.keySignature, "value") else 0
            repeat_close = int(getattr(header, "repeatClose", -1) or -1)

            voices: list[tuple[int, tuple[_GpBeatEvent, ...]]] = []
            for v_idx, gp_voice in enumerate(gp_m.voices):
                if not gp_voice.beats or all(b.status == gm.BeatStatus.empty for b in gp_voice.beats):
                    continue
                beat_events: list[_GpBeatEvent] = []
                for beat in gp_voice.beats:
                    ql = Fraction(beat.duration.time) / Fraction(quarter_ticks)
                    if ql <= 0:
                        continue
                    playable = [n for n in beat.notes if n.type != gm.NoteType.rest]
                    if beat.status == gm.BeatStatus.rest or not playable:
                        beat_events.append(_GpBeatEvent(ql=ql, notes=()))
                        continue

                    note_specs: list[tuple[int, int | None, int | None]] = []
                    hop_origin = slide_origin = harmonic = palm_mute = False
                    bend_semis = 0.0
                    for gp_note in playable:
                        try:
                            midi_val = int(gp_note.realValue) + capo_offset
                        except (IndexError, AttributeError, TypeError, ValueError):
                            midi_val = int(gp_note.value) + capo_offset
                        if not (0 <= midi_val <= 127):
                            continue
                        s_num = (
                            int(gp_note.string)
                            if (is_guitar and 1 <= gp_note.string <= n_strings and 0 <= gp_note.value <= 24)
                            else None
                        )
                        fret = int(gp_note.value) if s_num is not None else None
                        note_specs.append((midi_val, s_num, fret))

                        eff = getattr(gp_note, "effect", None)
                        if eff is not None:
                            hop_origin = hop_origin or bool(getattr(eff, "hammer", False))
                            slide_origin = slide_origin or any(
                                sl in slide_out_types for sl in (getattr(eff, "slides", None) or ())
                            )
                            harmonic = harmonic or bool(getattr(eff, "isHarmonic", False))
                            palm_mute = palm_mute or bool(getattr(eff, "palmMute", False))
                            bend_semis = max(bend_semis, _gp35_bend_semitones(eff))

                    beat_events.append(
                        _GpBeatEvent(
                            ql=ql,
                            notes=tuple(note_specs),
                            is_tie=all(n.type == gm.NoteType.tie for n in playable),
                            hop_origin=hop_origin,
                            slide_origin=slide_origin,
                            bend_semis=bend_semis,
                            harmonic=harmonic,
                            palm_mute=palm_mute,
                        )
                    )
                if beat_events:
                    voices.append((v_idx, tuple(beat_events)))

            measures.append(
                _GpMeasureData(
                    number=header.number or (m_idx + 1),
                    ts=(ts_num, ts_den),
                    fifths=fifths,
                    repeat_start=bool(getattr(header, "isRepeatOpen", False)),
                    repeat_close=repeat_close if repeat_close >= 0 else None,
                    voices=tuple(voices),
                )
            )

        tracks.append(
            _GpTrackData(
                part_id=f"P{track.number}",
                name=(track.name or "").strip() or f"Track {track.number}",
                gm_program=gm_prog,
                open_strings=open_strings,
                capo_fret=capo_offset,
                is_guitar=is_guitar,
                measures=tuple(measures),
            )
        )

    return _build_gp_score(
        title=(song.title or "").strip() or path.stem,
        composer=(song.artist or song.music or "").strip() or None,
        tempo_bpm=float(song.tempo) if song.tempo else None,
        skipped_percussion=len(song.tracks) - len(non_percussion),
        tracks=tracks,
        source_name=path.name,
    )


def _gpif_parse_rhythm_ql(rhythm_el: ET.Element | None) -> Fraction:
    if rhythm_el is None:
        return Fraction(1, 1)
    ql = _NOTE_VALUE_QL.get((rhythm_el.findtext("NoteValue") or "Quarter").strip(), Fraction(1, 1))
    dots_el = rhythm_el.find("AugmentationDot")
    if dots_el is not None:
        dots = try_int(dots_el.attrib.get("count", "1")) or 1
        if dots == 1:
            ql *= Fraction(3, 2)
        elif dots >= 2:
            ql *= Fraction(7, 4)
    tuplet_el = rhythm_el.find("PrimaryTuplet")
    if tuplet_el is not None:
        num = try_int(tuplet_el.attrib.get("num")) or 1
        den = try_int(tuplet_el.attrib.get("den")) or 1
        if num > 0 and den > 0:
            ql *= Fraction(den, num)
    return ql


def _gpif_is_percussion(track_el: ET.Element) -> bool:
    inst_type = (track_el.findtext("./InstrumentSet/Type") or "").strip().lower()
    inst_name = (track_el.findtext("./InstrumentSet/Name") or "").strip().lower()
    if inst_type in {"drumkit", "percussion"} or "drum" in inst_name:
        return True
    return any(
        try_int(track_el.findtext(p)) == 9
        for p in ("./GeneralMidi/PrimaryChannel", "./Sounds/Sound/MIDI/PrimaryChannel")
    )


def _gpif_tuning_and_capo(track_el: ET.Element) -> tuple[tuple[int, ...] | None, int]:
    tuning: tuple[int, ...] | None = None
    capo = 0
    for prop in track_el.findall("./Staves/Staff/Properties/Property"):
        name = prop.attrib.get("name", "").strip()
        if name == "Tuning":
            vals = [v for tok in (prop.findtext("Pitches") or "").split() if (v := try_int(tok)) is not None]
            if vals:
                tuning = tuple(reversed(vals))  # GPIF stores low-to-high; reverse to string 1..N
        elif name == "CapoFret" and (
            c := try_int(prop.findtext("Fret") or prop.findtext("Number"))
        ) is not None:
            capo = max(0, c)
    return tuning, capo


def _gpif_initial_tempo(root: ET.Element) -> float | None:
    for auto in root.findall("./MasterTrack/Automations/Automation"):
        if (auto.findtext("Type") or "").strip() == "Tempo":
            first = (auto.findtext("Value") or "").strip().split()[:1]
            if first:
                try:
                    bpm = float(first[0])
                    if bpm > 0:
                        return bpm
                except ValueError:
                    continue
    return None


def parse_gpif_file(gp_path: Path) -> tuple[music21.stream.Score, list[str]] | None:
    """Parse a `.gp` (GP7/8) file via stdlib ZIP+XML if `Content/score.gpif` is present."""
    xml_bytes = read_bounded_zip_member(
        gp_path,
        lambda name: name.lower() in {"content/score.gpif", "score.gpif"},
    )
    if xml_bytes is None:
        return None
    head_upper = xml_bytes[:4096].upper()
    if b"<!DOCTYPE" in head_upper or b"<!ENTITY" in head_upper:
        raise ValueError(f"Unsafe XML DTD/entity declaration in {gp_path.name}")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    tracks_elems = root.findall("./Tracks/Track")
    pitched_tracks = [(i, tr) for i, tr in enumerate(tracks_elems) if not _gpif_is_percussion(tr)]
    if not pitched_tracks:
        raise ValueError(f"{gp_path.name} contains only percussion tracks; no pitched tracks to convert.")

    bars_by_id = {b.attrib["id"]: b for b in root.findall("./Bars/Bar") if "id" in b.attrib}
    voices_by_id = {v.attrib["id"]: v for v in root.findall("./Voices/Voice") if "id" in v.attrib}
    beats_by_id = {b.attrib["id"]: b for b in root.findall("./Beats/Beat") if "id" in b.attrib}
    notes_by_id = {n.attrib["id"]: n for n in root.findall("./Notes/Note") if "id" in n.attrib}
    rhythms_by_id = {r.attrib["id"]: r for r in root.findall("./Rhythms/Rhythm") if "id" in r.attrib}
    master_bars = root.findall("./MasterBars/MasterBar")

    tracks: list[_GpTrackData] = []
    for out_idx, (track_pos, track_el) in enumerate(pitched_tracks):
        gm_prog = 25
        for p in ("./GeneralMidi/Program", "./Sounds/Sound/MIDI/Program"):
            if (prog := try_int(track_el.findtext(p))) is not None:
                gm_prog = max(0, min(127, prog))
                break
        tuning, capo_fret = _gpif_tuning_and_capo(track_el)
        is_guitar = 24 <= gm_prog <= 31 or tuning is not None
        open_strings = tuning or (_DEFAULT_GUITAR_TUNING if is_guitar else ())

        measures: list[_GpMeasureData] = []
        for m_idx, mb_el in enumerate(master_bars):
            ts_sig: tuple[int, int] | None = None
            ts_txt = (mb_el.findtext("Time") or "").strip()
            if "/" in ts_txt:
                num_s, den_s = ts_txt.split("/", 1)
                if (n := try_int(num_s)) is not None and (d := try_int(den_s)) is not None:
                    ts_sig = (n, d)

            repeat_el = mb_el.find("Repeat")
            rep_start = repeat_el is not None and repeat_el.attrib.get("start") == "true"
            rep_close: int | None = None
            if repeat_el is not None and repeat_el.attrib.get("end") == "true":
                cnt = try_int(repeat_el.attrib.get("count"))
                rep_close = cnt if (cnt is not None and cnt > 0) else 0

            bar_ids = (mb_el.findtext("Bars") or "").split()
            bar_el = bars_by_id.get(bar_ids[track_pos]) if track_pos < len(bar_ids) else None
            voice_ids = [
                vid
                for vid in ((bar_el.findtext("Voices") or "").split() if bar_el is not None else [])
                if vid != "-1" and vid in voices_by_id
            ]

            voices: list[tuple[int, tuple[_GpBeatEvent, ...]]] = []
            for v_idx, vid in enumerate(voice_ids):
                beat_ids = [
                    bid for bid in (voices_by_id[vid].findtext("Beats") or "").split() if bid in beats_by_id
                ]
                beat_events: list[_GpBeatEvent] = []
                for bid in beat_ids:
                    beat_el = beats_by_id[bid]
                    if beat_el.find("GraceNotes") is not None:
                        continue
                    r_tag = beat_el.find("Rhythm")
                    ql = _gpif_parse_rhythm_ql(
                        rhythms_by_id.get(r_tag.attrib.get("ref", "") if r_tag is not None else "")
                    )
                    note_ids = [
                        nid for nid in (beat_el.findtext("Notes") or "").split() if nid in notes_by_id
                    ]
                    if not note_ids:
                        beat_events.append(_GpBeatEvent(ql=ql, notes=()))
                        continue

                    note_specs: list[tuple[int, int | None, int | None]] = []
                    is_tie = hop_origin = hop_dest = slide_origin = harmonic = palm_mute = False
                    bend_semis = 0.0
                    for nid in note_ids:
                        n_el = notes_by_id[nid]
                        props = {
                            k: prop
                            for prop in n_el.findall("./Properties/Property")
                            if (k := prop.attrib.get("name", "").strip())
                        }
                        tie_el = n_el.find("Tie")
                        if tie_el is not None and tie_el.attrib.get("destination") == "true":
                            is_tie = True

                        s_num: int | None = None
                        if (
                            "String" in props
                            and open_strings
                            and (gpif_s := try_int(props["String"].findtext("String"))) is not None
                        ):
                            cand = len(open_strings) - gpif_s
                            if 1 <= cand <= len(open_strings):
                                s_num = cand
                        fret_num = try_int(props["Fret"].findtext("Fret")) if "Fret" in props else None

                        midi_val: int | None = None
                        if (
                            "Midi" in props
                            and (raw_m := try_int(props["Midi"].findtext("Number"))) is not None
                        ):
                            midi_val = raw_m + capo_fret
                        if midi_val is None and s_num is not None and fret_num is not None and open_strings:
                            midi_val = open_strings[s_num - 1] + fret_num + capo_fret
                        if midi_val is None or not (0 <= midi_val <= 127):
                            continue

                        note_specs.append((midi_val, s_num, fret_num))
                        harmonic = harmonic or ("Harmonic" in props or "HarmonicType" in props)
                        palm_mute = palm_mute or ("PalmMuted" in props)
                        slide_origin = slide_origin or ("Slide" in props)
                        hop_origin = hop_origin or ("HopOOrigin" in props)
                        hop_dest = hop_dest or ("HopODestination" in props)
                        if "Bended" in props and "BendDestinationValue" in props:
                            try:
                                val = float((props["BendDestinationValue"].findtext("Float") or "0").strip())
                                bend_semis = max(bend_semis, val / 50.0)
                            except ValueError:
                                pass
                        legato_el = n_el.find("Legato")
                        if legato_el is not None:
                            hop_origin = hop_origin or legato_el.attrib.get("origin") == "true"
                            hop_dest = hop_dest or legato_el.attrib.get("destination") == "true"

                    beat_events.append(
                        _GpBeatEvent(
                            ql=ql,
                            notes=tuple(note_specs),
                            is_tie=is_tie,
                            hop_origin=hop_origin,
                            hop_dest=hop_dest,
                            slide_origin=slide_origin,
                            bend_semis=bend_semis,
                            harmonic=harmonic,
                            palm_mute=palm_mute,
                        )
                    )
                if beat_events:
                    voices.append((v_idx, tuple(beat_events)))

            measures.append(
                _GpMeasureData(
                    number=m_idx + 1,
                    ts=ts_sig,
                    fifths=try_int(mb_el.findtext("./Key/AccidentalCount")),
                    repeat_start=rep_start,
                    repeat_close=rep_close,
                    voices=tuple(voices),
                )
            )

        tracks.append(
            _GpTrackData(
                part_id=f"P{out_idx + 1}",
                name=(track_el.findtext("Name") or "").strip() or f"Track {out_idx + 1}",
                gm_program=gm_prog,
                open_strings=open_strings,
                capo_fret=capo_fret,
                is_guitar=is_guitar,
                measures=tuple(measures),
            )
        )

    title = (root.findtext("./Score/Title") or "").strip() or gp_path.stem
    artist = (
        (root.findtext("./Score/Artist") or "").strip()
        or (root.findtext("./Score/Music") or "").strip()
        or None
    )
    return _build_gp_score(
        title=title,
        composer=artist,
        tempo_bpm=_gpif_initial_tempo(root),
        skipped_percussion=len(tracks_elems) - len(pitched_tracks),
        tracks=tracks,
        source_name=gp_path.name,
    )
