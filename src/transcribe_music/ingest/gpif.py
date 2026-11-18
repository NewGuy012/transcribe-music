"""Native stdlib parser for Guitar Pro 7/8 (.gp) ZIP containers housing Content/score.gpif XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
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


def _find_gpif_xml_bytes(gp_path: Path) -> bytes | None:
    return read_bounded_zip_member(
        gp_path,
        lambda name: name.lower() in {"content/score.gpif", "score.gpif"},
    )


def _note_properties(note_el: ET.Element) -> dict[str, ET.Element]:
    props: dict[str, ET.Element] = {}
    for prop in note_el.findall("./Properties/Property"):
        name = prop.attrib.get("name", "").strip()
        if name:
            props[name] = prop
    return props


def _parse_rhythm_ql(rhythm_el: ET.Element | None) -> Fraction:
    if rhythm_el is None:
        return Fraction(1, 1)
    nv = (rhythm_el.findtext("NoteValue") or "Quarter").strip()
    ql = _NOTE_VALUE_QL.get(nv, Fraction(1, 1))
    dots_el = rhythm_el.find("AugmentationDot")
    if dots_el is not None:
        dot_count = int(dots_el.attrib.get("count", "1") or "1")
        if dot_count == 1:
            ql *= Fraction(3, 2)
        elif dot_count >= 2:
            ql *= Fraction(7, 4)
    tuplet_el = rhythm_el.find("PrimaryTuplet")
    if tuplet_el is not None:
        num = int(tuplet_el.attrib.get("num", "1") or "1")
        den = int(tuplet_el.attrib.get("den", "1") or "1")
        if num > 0 and den > 0:
            ql *= Fraction(den, num)
    return ql


def _is_percussion_track(track_el: ET.Element) -> bool:
    inst_type = (track_el.findtext("./InstrumentSet/Type") or "").strip().lower()
    inst_name = (track_el.findtext("./InstrumentSet/Name") or "").strip().lower()
    if inst_type in {"drumkit", "percussion"} or "drum" in inst_name:
        return True
    for ch_path in ("./GeneralMidi/PrimaryChannel", "./Sounds/Sound/MIDI/PrimaryChannel"):
        ch_txt = track_el.findtext(ch_path)
        if ch_txt and ch_txt.strip().isdigit() and int(ch_txt.strip()) == 9:
            return True
    return False


def _extract_track_midi_program(track_el: ET.Element) -> int:
    for path in ("./GeneralMidi/Program", "./Sounds/Sound/MIDI/Program"):
        txt = track_el.findtext(path)
        if txt and txt.strip().isdigit():
            return max(0, min(127, int(txt.strip())))
    return 25


def _extract_track_tuning_and_capo(track_el: ET.Element) -> tuple[tuple[int, ...] | None, int]:
    tuning: tuple[int, ...] | None = None
    capo = 0
    for prop in track_el.findall("./Staves/Staff/Properties/Property"):
        name = prop.attrib.get("name", "").strip()
        if name == "Tuning":
            pitches_txt = (prop.findtext("Pitches") or "").strip()
            vals = [int(tok) for tok in pitches_txt.split() if tok.lstrip("-").isdigit()]
            if vals:
                # GPIF stores Tuning low-to-high; reverse to string 1 (highest) -> string N (lowest)
                tuning = tuple(reversed(vals))
        elif name == "CapoFret":
            capo_txt = (prop.findtext("Fret") or prop.findtext("Number") or "").strip()
            if capo_txt.isdigit():
                capo = int(capo_txt)
    return tuning, capo


def _extract_initial_tempo(root: ET.Element) -> float | None:
    for auto in root.findall("./MasterTrack/Automations/Automation"):
        if (auto.findtext("Type") or "").strip() == "Tempo":
            val_txt = (auto.findtext("Value") or "").strip()
            first = val_txt.split()[0] if val_txt else ""
            try:
                bpm = float(first)
                if bpm > 0:
                    return bpm
            except ValueError:
                continue
    return None


def parse_gpif_file(gp_path: Path) -> tuple[music21.stream.Score, list[str]] | None:
    """Parse a `.gp` (GP7/8) file via stdlib ZIP+XML if `Content/score.gpif` is present."""
    xml_bytes = _find_gpif_xml_bytes(gp_path)
    if xml_bytes is None:
        return None
    head_upper = xml_bytes[:4096].upper()
    if b"<!DOCTYPE" in head_upper or b"<!ENTITY" in head_upper:
        raise ValueError(f"Unsafe XML DTD/entity declaration in {gp_path.name}")

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    title = (root.findtext("./Score/Title") or "").strip() or gp_path.stem
    artist = (
        (root.findtext("./Score/Artist") or "").strip()
        or (root.findtext("./Score/Music") or "").strip()
        or None
    )
    tempo_bpm = _extract_initial_tempo(root)

    tracks_elems = root.findall("./Tracks/Track")
    pitched_tracks: list[tuple[int, ET.Element]] = [
        (idx, tr) for idx, tr in enumerate(tracks_elems) if not _is_percussion_track(tr)
    ]
    if not pitched_tracks:
        raise ValueError(f"{gp_path.name} contains only percussion tracks; no pitched tracks to convert.")

    warnings: list[str] = []
    if len(pitched_tracks) < len(tracks_elems):
        skipped = len(tracks_elems) - len(pitched_tracks)
        warnings.append(f"Skipped {skipped} percussion track(s) from {gp_path.name}.")

    bars_by_id = {b.attrib["id"]: b for b in root.findall("./Bars/Bar") if "id" in b.attrib}
    voices_by_id = {v.attrib["id"]: v for v in root.findall("./Voices/Voice") if "id" in v.attrib}
    beats_by_id = {b.attrib["id"]: b for b in root.findall("./Beats/Beat") if "id" in b.attrib}
    notes_by_id = {n.attrib["id"]: n for n in root.findall("./Notes/Note") if "id" in n.attrib}
    rhythms_by_id = {r.attrib["id"]: r for r in root.findall("./Rhythms/Rhythm") if "id" in r.attrib}
    master_bars = root.findall("./MasterBars/MasterBar")

    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title, composer=artist)

    for out_idx, (track_pos, track_el) in enumerate(pitched_tracks):
        track_name = (track_el.findtext("Name") or "").strip() or f"Track {out_idx + 1}"
        gm_prog = _extract_track_midi_program(track_el)
        tuning, capo_fret = _extract_track_tuning_and_capo(track_el)
        is_guitar = 24 <= gm_prog <= 31 or tuning is not None
        open_strings = tuning or (_DEFAULT_GUITAR_TUNING if is_guitar else ())

        part = music21.stream.Part(id=f"P{out_idx + 1}")
        part.partName = track_name
        try:
            m21_inst = music21.instrument.instrumentFromMidiProgram(gm_prog)
        except Exception:  # noqa: BLE001 — unknown GM program fallback
            m21_inst = music21.instrument.Instrument()
        m21_inst.partName = track_name
        part.insert(0, m21_inst)

        if is_guitar and open_strings:
            set_guitar_track_meta(
                part, GuitarTrackMeta(open_string_midi=open_strings, capo_fret=capo_fret)
            )

        prev_ts: tuple[int, int] | None = None
        prev_ks: int | None = None
        prev_el_by_voice: dict[int, music21.note.Note | music21.chord.Chord | None] = {}
        prev_pending_tech_by_voice: dict[int, tuple[bool, bool, bool, bool]] = {}

        for m_idx, mb_el in enumerate(master_bars):
            m21_measure = music21.stream.Measure(number=m_idx + 1)

            ts_txt = (mb_el.findtext("Time") or "").strip()
            if "/" in ts_txt:
                num_s, den_s = ts_txt.split("/", 1)
                ts_num, ts_den = int(num_s), int(den_s)
            elif prev_ts is not None:
                ts_num, ts_den = prev_ts
            else:
                ts_num, ts_den = 4, 4
            if prev_ts != (ts_num, ts_den):
                m21_measure.insert(0, music21.meter.TimeSignature(f"{ts_num}/{ts_den}"))
                prev_ts = (ts_num, ts_den)
            bar_ql = Fraction(ts_num * 4, ts_den)

            acc_txt = (mb_el.findtext("./Key/AccidentalCount") or "").strip()
            if acc_txt.lstrip("-").isdigit():
                fifths = max(-7, min(7, int(acc_txt)))
                if prev_ks != fifths:
                    m21_measure.insert(0, music21.key.KeySignature(fifths))
                    prev_ks = fifths

            repeat_el = mb_el.find("Repeat")
            if repeat_el is not None:
                if repeat_el.attrib.get("start") == "true":
                    m21_measure.leftBarline = music21.bar.Repeat(direction="start")
                if repeat_el.attrib.get("end") == "true":
                    cnt = repeat_el.attrib.get("count")
                    rep = music21.bar.Repeat(direction="end")
                    if cnt and cnt.isdigit() and int(cnt) > 0:
                        rep.times = int(cnt)
                    m21_measure.rightBarline = rep

            if m_idx == 0 and out_idx == 0 and tempo_bpm is not None:
                m21_measure.insert(0, music21.tempo.MetronomeMark(number=tempo_bpm))

            bar_ids = (mb_el.findtext("Bars") or "").split()
            bar_el = bars_by_id.get(bar_ids[track_pos]) if track_pos < len(bar_ids) else None
            voice_ids = [
                vid
                for vid in ((bar_el.findtext("Voices") or "").split() if bar_el is not None else [])
                if vid != "-1" and vid in voices_by_id
            ]

            active_voices: list[tuple[int, list[str]]] = []
            for v_idx, vid in enumerate(voice_ids):
                beat_ids = [
                    bid for bid in (voices_by_id[vid].findtext("Beats") or "").split() if bid in beats_by_id
                ]
                if beat_ids:
                    active_voices.append((v_idx, beat_ids))

            if not active_voices:
                m21_measure.insert(0, music21.note.Rest(quarterLength=bar_ql))
                part.append(m21_measure)
                continue

            multi_voice = len(active_voices) > 1
            for v_idx, beat_ids in active_voices:
                target_stream: music21.stream.Stream = (
                    music21.stream.Voice(id=str(v_idx + 1)) if multi_voice else m21_measure
                )
                offset_ql = Fraction(0, 1)
                prev_el = prev_el_by_voice.get(v_idx)
                pend_h, pend_p, pend_s, prev_pm = prev_pending_tech_by_voice.get(
                    v_idx, (False, False, False, False)
                )

                for bid in beat_ids:
                    beat_el = beats_by_id[bid]
                    if beat_el.find("GraceNotes") is not None:
                        continue
                    rhythm_ref = ""
                    rhythm_tag = beat_el.find("Rhythm")
                    if rhythm_tag is not None:
                        rhythm_ref = rhythm_tag.attrib.get("ref", "")
                    ql = _parse_rhythm_ql(rhythms_by_id.get(rhythm_ref))
                    if ql <= 0:
                        continue
                    if offset_ql >= bar_ql:
                        break
                    if offset_ql + ql > bar_ql:
                        ql = bar_ql - offset_ql

                    dur_obj = music21.duration.Duration(ql)
                    note_ids = [
                        nid for nid in (beat_el.findtext("Notes") or "").split() if nid in notes_by_id
                    ]
                    if not note_ids:
                        target_stream.insert(offset_ql, music21.note.Rest(duration=dur_obj))
                        prev_el = None
                        pend_h = pend_p = pend_s = prev_pm = False
                        offset_ql += ql
                        continue

                    note_specs: list[tuple[int, int | None, int | None]] = []
                    is_tie_dest = False
                    next_hop_origin = False
                    cur_hop_dest = False
                    next_slide_origin = False
                    bend_semis = 0.0
                    harmonic = False
                    palm_mute = False

                    for nid in note_ids:
                        n_el = notes_by_id[nid]
                        props = _note_properties(n_el)
                        tie_el = n_el.find("Tie")
                        if tie_el is not None and tie_el.attrib.get("destination") == "true":
                            is_tie_dest = True

                        s_num: int | None = None
                        fret_num: int | None = None
                        if "String" in props:
                            s_txt = (props["String"].findtext("String") or "").strip()
                            if s_txt.lstrip("-").isdigit() and open_strings:
                                gpif_s = int(s_txt)
                                cand_s = len(open_strings) - gpif_s
                                if 1 <= cand_s <= len(open_strings):
                                    s_num = cand_s
                        if "Fret" in props:
                            f_txt = (props["Fret"].findtext("Fret") or "").strip()
                            if f_txt.lstrip("-").isdigit():
                                fret_num = int(f_txt)

                        sounding_midi: int | None = None
                        if "Midi" in props:
                            m_txt = (props["Midi"].findtext("Number") or "").strip()
                            if m_txt.lstrip("-").isdigit():
                                sounding_midi = int(m_txt) + capo_fret
                        if sounding_midi is None and s_num is not None and fret_num is not None and open_strings:
                            sounding_midi = open_strings[s_num - 1] + fret_num + capo_fret
                        if sounding_midi is None or not (0 <= sounding_midi <= 127):
                            continue

                        note_specs.append((sounding_midi, s_num, fret_num))
                        if "Harmonic" in props or "HarmonicType" in props:
                            harmonic = True
                        if "PalmMuted" in props:
                            palm_mute = True
                        if "Bended" in props and "BendDestinationValue" in props:
                            val_txt = (props["BendDestinationValue"].findtext("Float") or "0").strip()
                            try:
                                bend_semis = max(bend_semis, float(val_txt) / 50.0)
                            except ValueError:
                                pass
                        if "Slide" in props:
                            next_slide_origin = True
                        if "HopOOrigin" in props:
                            next_hop_origin = True
                        if "HopODestination" in props:
                            cur_hop_dest = True
                        legato_el = n_el.find("Legato")
                        if legato_el is not None:
                            if legato_el.attrib.get("origin") == "true":
                                next_hop_origin = True
                            if legato_el.attrib.get("destination") == "true":
                                cur_hop_dest = True

                    if not note_specs:
                        target_stream.insert(offset_ql, music21.note.Rest(duration=dur_obj))
                        prev_el = None
                        pend_h = pend_p = pend_s = prev_pm = False
                    else:
                        el = build_fretted_element(note_specs, dur_obj)
                        if is_tie_dest and prev_el is not None:
                            mark_tie(prev_el, el)

                        hammer_now = False
                        pull_now = False
                        if (pend_h or cur_hop_dest) and prev_el is not None:
                            prev_top = max(element_midis(prev_el))
                            cur_top = max(element_midis(el))
                            if cur_top >= prev_top:
                                hammer_now = True
                            else:
                                pull_now = True

                        attach_guitar_techniques(
                            part,
                            target_stream,
                            prev_el,
                            el,
                            offset_ql,
                            hammer_from_prev=hammer_now,
                            pull_from_prev=pull_now,
                            slide_from_prev=pend_s,
                            bend_semitones=bend_semis,
                            harmonic=harmonic,
                            palm_mute=palm_mute,
                            prev_palm_mute=prev_pm,
                        )
                        target_stream.insert(offset_ql, el)
                        prev_el = el
                        pend_h = next_hop_origin
                        pend_p = False
                        pend_s = next_slide_origin
                        prev_pm = palm_mute
                    offset_ql += ql

                if offset_ql < bar_ql:
                    target_stream.insert(offset_ql, music21.note.Rest(quarterLength=bar_ql - offset_ql))

                prev_el_by_voice[v_idx] = prev_el
                prev_pending_tech_by_voice[v_idx] = (pend_h, pend_p, pend_s, prev_pm)
                if multi_voice:
                    m21_measure.insert(0, target_stream)

            part.append(m21_measure)

        score.insert(0, part)

    return score, warnings
