"""Direct ingestion and inspection for digital symbolic music formats."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import music21

from transcribe_music.ingest.ascii_tab import parse_ascii_tab_file
from transcribe_music.ingest.gpif import parse_gpif_file
from transcribe_music.models import PartInfo, ScoreInspection
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    attach_guitar_techniques,
    build_fretted_element,
    coerce_to_score,
    element_midis,
    format_tuning_label,
    get_guitar_track_meta,
    iter_playable_notes,
    mark_tie,
    set_guitar_track_meta,
)

logger = logging.getLogger(__name__)

MUSICXML_EXTENSIONS = frozenset({".musicxml", ".xml", ".mxl"})
CONVERTER21_EXTENSIONS: dict[str, str] = {
    ".krn": "converter21_humdrum",
    ".mei": "converter21_mei",
    ".abc": "converter21_abc",
}
MIDI_EXTENSIONS = frozenset({".mid", ".midi"})
GUITARPRO_EXTENSIONS = frozenset({".gp3", ".gp4", ".gp5"})
MUSESCORE_EXTENSIONS = frozenset({".mscz", ".mscx"})
MUSESCORE_GP_EXTENSIONS = frozenset({".gp", ".gpx"})
CAPELLA_EXTENSIONS = frozenset({".cap", ".capx"})
LILYPOND_EXTENSIONS = frozenset({".ly"})
ASCII_TAB_EXTENSIONS = frozenset({".tab"})
_LOCAL_ASCII_TAB_EXTENSIONS = frozenset({".tab", ".txt"})

DIGITAL_EXTENSIONS = (
    MUSICXML_EXTENSIONS
    | frozenset(CONVERTER21_EXTENSIONS)
    | MIDI_EXTENSIONS
    | GUITARPRO_EXTENSIONS
    | MUSESCORE_EXTENSIONS
    | MUSESCORE_GP_EXTENSIONS
    | CAPELLA_EXTENSIONS
    | LILYPOND_EXTENSIONS
    | ASCII_TAB_EXTENSIONS
)

_CONVERTER21_REGISTERED = False


@dataclass(frozen=True)
class LoadedDigitalScore:
    """Parsed symbolic score along with provenance metadata."""

    score: music21.stream.Score
    backend: str
    format_name: str
    is_sounding_pitch_source: bool
    warnings: list[str] = field(default_factory=list)


def ensure_converter21_registered() -> None:
    """Idempotently register converter21 parsers for Humdrum (.krn), MEI, and ABC."""
    global _CONVERTER21_REGISTERED
    if _CONVERTER21_REGISTERED:
        return
    import converter21

    converter21.register()
    _CONVERTER21_REGISTERED = True


def _to_loaded_score(
    parsed: music21.stream.Stream,
    path: Path,
    backend: str,
    *,
    format_name: str | None = None,
    is_sounding: bool = False,
) -> LoadedDigitalScore:
    score, warnings = coerce_to_score(parsed, path.stem)
    return LoadedDigitalScore(
        score=score,
        backend=backend,
        format_name=format_name or path.suffix.lstrip(".").lower(),
        is_sounding_pitch_source=is_sounding,
        warnings=warnings,
    )


def _find_musescore_executable() -> str | None:
    """Locate a local MuseScore binary for .mscz/.mscx -> MusicXML conversion if installed."""
    for name in ("mscore", "mscore4", "musescore", "musescore4"):
        found = shutil.which(name)
        if found:
            return found

    mac_candidates = (
        Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
        Path("/Applications/MuseScore 3.app/Contents/MacOS/mscore"),
        Path.home() / "Applications" / "MuseScore 4.app" / "Contents" / "MacOS" / "mscore",
    )
    for candidate in mac_candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _load_musescore_file(path: Path) -> LoadedDigitalScore:
    mscore_bin = _find_musescore_executable()
    if mscore_bin is None:
        suffix = path.suffix.lower()
        if suffix in MUSESCORE_GP_EXTENSIONS:
            hint = "Export as .musicxml, .mxl, or .gp5 from Guitar Pro/MuseScore first, or install MuseScore 4."
            prefix = f"Cannot parse Guitar Pro 6/7/8 ({path.name}) directly because MuseScore Studio CLI was not found."
        elif suffix in CAPELLA_EXTENSIONS:
            hint = "Export as .musicxml or .mxl from Capella/MuseScore first, or install MuseScore 4."
            prefix = f"Cannot parse Capella ({path.name}) directly because MuseScore Studio CLI was not found."
        else:
            hint = "Export the score as .mxl or .musicxml from MuseScore first, or install MuseScore 4."
            prefix = f"Cannot parse {path.name} directly because MuseScore Studio CLI was not found."
        raise RuntimeError(f"{prefix} {hint}")

    with tempfile.TemporaryDirectory(prefix="tm_mscore_") as tmp_dir:
        tmp_xml = Path(tmp_dir) / f"{path.stem}.musicxml"
        proc = subprocess.run(
            [mscore_bin, str(path.resolve()), "-o", str(tmp_xml)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not tmp_xml.exists():
            detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            raise RuntimeError(f"MuseScore failed converting {path.name} to MusicXML: {detail}")
        parsed = music21.converter.parse(str(tmp_xml), format="musicxml")
        return _to_loaded_score(parsed, path, "musescore")


def _load_capella_file(path: Path) -> LoadedDigitalScore:
    """Load Capella .capx via music21 native parser (falling back to MuseScore) or .cap via MuseScore."""
    if path.suffix.lower() == ".capx":
        try:
            from music21.capella.fromCapellaXML import CapellaImporter

            parsed = CapellaImporter().scoreFromFile(str(path))
            return _to_loaded_score(parsed, path, "capella_xml", format_name="capx")
        except Exception as exc:
            if _find_musescore_executable() is not None:
                loaded = _load_musescore_file(path)
                loaded.warnings.append(f"Native .capx parser failed ({exc}); converted via MuseScore CLI.")
                return loaded
            raise RuntimeError(
                f"Failed parsing Capella XML file {path.name}: {exc}. "
                "Install MuseScore 4 or export as .musicxml/.mxl."
            ) from exc
    return _load_musescore_file(path)


def _load_lilypond_file(
    path: Path,
    quantize_divisors: tuple[int, ...] = (4, 3),
) -> LoadedDigitalScore:
    """Convert LilyPond (.ly) via python-ly ('ly musicxml') or lilypond CLI ('--midi')."""
    ly_bin = shutil.which("ly")
    if ly_bin is not None:
        proc = subprocess.run(
            [ly_bin, "musicxml", str(path.resolve())],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        xml_text = proc.stdout or ""
        if proc.returncode == 0 and ("<score-partwise" in xml_text or "<score-timewise" in xml_text):
            with tempfile.TemporaryDirectory(prefix="tm_ly_") as tmp_dir:
                tmp_xml = Path(tmp_dir) / f"{path.stem}.musicxml"
                tmp_xml.write_text(xml_text, encoding="utf-8")
                parsed = music21.converter.parse(str(tmp_xml), format="musicxml")
                return _to_loaded_score(parsed, path, "python_ly", format_name="ly")

    lilypond_bin = shutil.which("lilypond")
    if lilypond_bin is not None:
        with tempfile.TemporaryDirectory(prefix="tm_lily_") as tmp_dir:
            out_prefix = Path(tmp_dir) / path.stem
            proc = subprocess.run(
                [lilypond_bin, "-dno-print-pages", "-o", str(out_prefix), str(path.resolve())],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            midi_files = sorted(Path(tmp_dir).glob("*.mid")) + sorted(Path(tmp_dir).glob("*.midi"))
            if proc.returncode == 0 and midi_files:
                parsed = music21.converter.parse(
                    str(midi_files[0]),
                    format="midi",
                    quantizePost=True,
                    quarterLengthDivisors=quantize_divisors or (4, 3),
                )
                return _to_loaded_score(
                    parsed, path, "lilypond_midi", format_name="ly", is_sounding=True
                )
            detail = proc.stderr.strip() or proc.stdout.strip() or "no MIDI output produced (missing \\midi block?)"
            raise RuntimeError(f"LilyPond failed converting {path.name}: {detail}")

    raise RuntimeError(
        f"Cannot parse LilyPond ({path.name}) because neither 'ly' (python-ly) nor 'lilypond' CLI was found. "
        "Install python-ly or LilyPond, or download the score's .mid/.musicxml version."
    )


def _gp_instrument_from_program(program: int, track_name: str) -> music21.instrument.Instrument:
    try:
        inst = music21.instrument.instrumentFromMidiProgram(max(0, min(127, int(program))))
    except Exception:  # noqa: BLE001 — unknown GM program fallback
        inst = music21.instrument.Instrument()
    if track_name:
        inst.partName = track_name
    return inst


def _gp_bend_semitones(effect: object) -> float:
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


def _load_guitarpro_score(path: Path) -> LoadedDigitalScore:
    """Convert a Guitar Pro (.gp3/.gp4/.gp5) song into a measured music21 Score."""
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

    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(
        title=(song.title or "").strip() or path.stem,
        composer=(song.artist or song.music or "").strip() or None,
    )

    warnings: list[str] = []
    if len(non_percussion) < len(song.tracks):
        skipped = len(song.tracks) - len(non_percussion)
        warnings.append(f"Skipped {skipped} percussion track(s) from {path.name}.")

    quarter_ticks = gm.Duration.quarterTime or 960
    slide_out_types = {gm.SlideType.shiftSlideTo, gm.SlideType.legatoSlideTo}

    for t_idx, track in enumerate(non_percussion):
        part = music21.stream.Part(id=f"P{track.number}")
        track_name = (track.name or "").strip() or f"Track {track.number}"
        part.partName = track_name

        gm_prog = int(getattr(track.channel, "instrument", 25))
        m21_inst = _gp_instrument_from_program(gm_prog, track_name)
        part.insert(0, m21_inst)

        capo_offset = int(getattr(track, "offset", 0) or 0)
        is_guitar_prog = 24 <= gm_prog <= 31
        raw_strings = getattr(track, "strings", None) or ()
        open_strings = tuple(
            int(s.value)
            for s in sorted(raw_strings, key=lambda st: int(getattr(st, "number", 0)))
        )
        n_strings = len(open_strings) if open_strings else 6
        if is_guitar_prog and open_strings:
            set_guitar_track_meta(
                part, GuitarTrackMeta(open_string_midi=open_strings, capo_fret=capo_offset)
            )

        prev_ts: tuple[int, int] | None = None
        prev_ks: int | None = None
        prev_el_by_voice: dict[int, music21.note.Note | music21.chord.Chord | None] = {}
        prev_pending_by_voice: dict[int, tuple[bool, bool, bool]] = {}

        for m_idx, gp_measure in enumerate(track.measures):
            header = gp_measure.header
            m21_measure = music21.stream.Measure(number=header.number or (m_idx + 1))

            ts_num = int(header.timeSignature.numerator)
            ts_den = int(header.timeSignature.denominator.value)
            ts_sig = (ts_num, ts_den)
            if prev_ts != ts_sig:
                m21_measure.insert(0, music21.meter.TimeSignature(f"{ts_num}/{ts_den}"))
                prev_ts = ts_sig
            bar_ql = Fraction(ts_num * 4, ts_den)

            raw_fifths = int(header.keySignature.value[0]) if hasattr(header.keySignature, "value") else 0
            fifths = max(-7, min(7, raw_fifths))
            if prev_ks != fifths:
                m21_measure.insert(0, music21.key.KeySignature(fifths))
                prev_ks = fifths

            if getattr(header, "isRepeatOpen", False):
                m21_measure.leftBarline = music21.bar.Repeat(direction="start")
            repeat_close = int(getattr(header, "repeatClose", -1) or -1)
            if repeat_close >= 0:
                rep = music21.bar.Repeat(direction="end")
                if repeat_close > 0:
                    rep.times = repeat_close
                m21_measure.rightBarline = rep

            if m_idx == 0 and t_idx == 0 and song.tempo:
                m21_measure.insert(0, music21.tempo.MetronomeMark(number=float(song.tempo)))

            active_voices = [
                (v_idx, v)
                for v_idx, v in enumerate(gp_measure.voices)
                if v.beats and any(b.status != gm.BeatStatus.empty for b in v.beats)
            ]
            if not active_voices:
                m21_measure.insert(0, music21.note.Rest(quarterLength=bar_ql))
                part.append(m21_measure)
                continue

            multi_voice = len(active_voices) > 1
            for v_idx, gp_voice in active_voices:
                target_stream: music21.stream.Stream = (
                    music21.stream.Voice(id=str(v_idx + 1)) if multi_voice else m21_measure
                )
                offset_ql = Fraction(0, 1)
                prev_el = prev_el_by_voice.get(v_idx)
                pend_hop, pend_slide, prev_pm = prev_pending_by_voice.get(
                    v_idx, (False, False, False)
                )

                for beat in gp_voice.beats:
                    ql = Fraction(beat.duration.time) / Fraction(quarter_ticks)
                    if ql <= 0:
                        continue
                    if offset_ql >= bar_ql:
                        break
                    if offset_ql + ql > bar_ql:
                        ql = bar_ql - offset_ql

                    dur_obj = music21.duration.Duration(ql)
                    playable_notes = [n for n in beat.notes if n.type != gm.NoteType.rest]

                    if beat.status == gm.BeatStatus.rest or not playable_notes:
                        target_stream.insert(offset_ql, music21.note.Rest(duration=dur_obj))
                        prev_el = None
                        pend_hop = pend_slide = prev_pm = False
                        offset_ql += ql
                        continue

                    note_specs: list[tuple[int, int | None, int | None]] = []
                    is_tie = all(n.type == gm.NoteType.tie for n in playable_notes)
                    next_hop = False
                    next_slide = False
                    bend_semis = 0.0
                    harmonic = False
                    palm_mute = False

                    for gp_note in playable_notes:
                        try:
                            sounding_midi = int(gp_note.realValue) + capo_offset
                        except IndexError:
                            sounding_midi = int(gp_note.value) + capo_offset
                        if not (0 <= sounding_midi <= 127):
                            continue
                        s_num = (
                            int(gp_note.string)
                            if (is_guitar_prog and 1 <= gp_note.string <= n_strings and 0 <= gp_note.value <= 24)
                            else None
                        )
                        fret_num = int(gp_note.value) if s_num is not None else None
                        note_specs.append((sounding_midi, s_num, fret_num))

                        eff = getattr(gp_note, "effect", None)
                        if eff is not None:
                            if getattr(eff, "hammer", False):
                                next_hop = True
                            if any(sl in slide_out_types for sl in (getattr(eff, "slides", None) or ())):
                                next_slide = True
                            if getattr(eff, "isHarmonic", False):
                                harmonic = True
                            if getattr(eff, "palmMute", False):
                                palm_mute = True
                            bend_semis = max(bend_semis, _gp_bend_semitones(eff))

                    if not note_specs:
                        target_stream.insert(offset_ql, music21.note.Rest(duration=dur_obj))
                        prev_el = None
                        pend_hop = pend_slide = prev_pm = False
                    else:
                        el = build_fretted_element(note_specs, dur_obj)
                        if is_tie and prev_el is not None:
                            mark_tie(prev_el, el)

                        hammer_now = False
                        pull_now = False
                        if pend_hop and prev_el is not None:
                            if max(element_midis(el)) >= max(element_midis(prev_el)):
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
                            slide_from_prev=pend_slide,
                            bend_semitones=bend_semis,
                            harmonic=harmonic,
                            palm_mute=palm_mute,
                            prev_palm_mute=prev_pm,
                        )
                        target_stream.insert(offset_ql, el)
                        prev_el = el
                        pend_hop = next_hop
                        pend_slide = next_slide
                        prev_pm = palm_mute
                    offset_ql += ql

                if offset_ql < bar_ql:
                    target_stream.insert(offset_ql, music21.note.Rest(quarterLength=bar_ql - offset_ql))

                prev_el_by_voice[v_idx] = prev_el
                prev_pending_by_voice[v_idx] = (pend_hop, pend_slide, prev_pm)
                if multi_voice:
                    m21_measure.insert(0, target_stream)

            part.append(m21_measure)

        score.insert(0, part)

    return LoadedDigitalScore(
        score=score,
        backend="pyguitarpro",
        format_name=path.suffix.lstrip(".").lower(),
        is_sounding_pitch_source=True,
        warnings=warnings,
    )


def load_digital_score(
    path: Path | str,
    quantize_divisors: tuple[int, ...] = (4, 3),
) -> LoadedDigitalScore:
    """Load a digital symbolic music file directly into a music21 Score."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Digital score file not found: {src}")

    suffix = src.suffix.lower()
    if suffix == ".pdf":
        raise ValueError(
            f"Expected a digital score file, got PDF ({src.name}). "
            "Use the PDF fallback command (`transcribe-music pdf`) or unified `convert`."
        )
    if suffix not in (DIGITAL_EXTENSIONS | _LOCAL_ASCII_TAB_EXTENSIONS):
        supported = ", ".join(sorted(DIGITAL_EXTENSIONS))
        raise ValueError(f"Unsupported digital score format {suffix!r}. Supported: {supported}")

    if suffix in _LOCAL_ASCII_TAB_EXTENSIONS:
        score, warnings = parse_ascii_tab_file(src)
        return LoadedDigitalScore(
            score=score,
            backend="ascii_tab",
            format_name=suffix.lstrip("."),
            is_sounding_pitch_source=True,
            warnings=warnings,
        )

    if suffix in GUITARPRO_EXTENSIONS:
        return _load_guitarpro_score(src)

    if suffix == ".gp":
        gpif_res = parse_gpif_file(src)
        if gpif_res is not None:
            score, warnings = gpif_res
            return LoadedDigitalScore(
                score=score,
                backend="gpif",
                format_name="gp",
                is_sounding_pitch_source=True,
                warnings=warnings,
            )

    if suffix in (MUSESCORE_EXTENSIONS | MUSESCORE_GP_EXTENSIONS):
        return _load_musescore_file(src)

    if suffix in CAPELLA_EXTENSIONS:
        return _load_capella_file(src)

    if suffix in LILYPOND_EXTENSIONS:
        return _load_lilypond_file(src, quantize_divisors=quantize_divisors)

    if suffix in MIDI_EXTENSIONS:
        parsed = music21.converter.parse(
            str(src),
            format="midi",
            quantizePost=True,
            quarterLengthDivisors=quantize_divisors or (4, 3),
        )
        return _to_loaded_score(parsed, src, "midi", format_name="midi", is_sounding=True)

    if suffix in CONVERTER21_EXTENSIONS:
        ensure_converter21_registered()
        parsed = music21.converter.parse(str(src))
        return _to_loaded_score(parsed, src, CONVERTER21_EXTENSIONS[suffix])

    parsed = music21.converter.parse(str(src))
    return _to_loaded_score(parsed, src, "musicxml")


def summarize_score_parts(score: music21.stream.Score) -> list[PartInfo]:
    """Extract readable summary metadata for all parts inside a Score."""
    infos: list[PartInfo] = []
    for idx, part in enumerate(score.parts, start=1):
        inst = part.recurse().getElementsByClass(music21.instrument.Instrument).first()
        clef = part.recurse().getElementsByClass(music21.clef.Clef).first()
        measures = list(part.getElementsByClass(music21.stream.Measure))
        pitches = [m for el in iter_playable_notes(part) for m in element_midis(el)]

        inst_name = getattr(inst, "instrumentName", None) if inst is not None else None
        midi_prog = getattr(inst, "midiProgram", None) if inst is not None else None
        name = (
            (part.partName or "").strip()
            or (inst_name or "").strip()
            or (str(part.id).strip() if part.id else "")
            or f"Part {idx}"
        )
        clef_name = type(clef).__name__ if clef is not None else None
        meta = get_guitar_track_meta(part)
        tuning = format_tuning_label(meta.open_string_midi) if meta is not None else None
        capo = meta.capo_fret if meta is not None and meta.capo_fret > 0 else None

        infos.append(
            PartInfo(
                index=idx,
                part_id=str(part.id or f"P{idx}"),
                name=name,
                instrument_name=inst_name,
                midi_program=int(midi_prog) if midi_prog is not None else None,
                clef=clef_name,
                measure_count=len(measures),
                note_count=len(pitches),
                pitch_min=min(pitches) if pitches else None,
                pitch_max=max(pitches) if pitches else None,
                tuning=tuning,
                capo=capo,
            )
        )
    return infos


def inspect_digital_score(path: Path | str) -> ScoreInspection:
    """Inspect a digital score file and list all parts, instruments, and ranges."""
    src = Path(path)
    loaded = load_digital_score(src)
    title = loaded.score.metadata.title if loaded.score.metadata else src.stem
    return ScoreInspection(
        source_path=src,
        format_name=loaded.format_name,
        title=title or src.stem,
        parts=summarize_score_parts(loaded.score),
    )
