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

from transcribe_music.models import PartInfo, ScoreInspection

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
UNSUPPORTED_GUITARPRO_EXTENSIONS = frozenset({".gp", ".gpx"})

DIGITAL_EXTENSIONS = (
    MUSICXML_EXTENSIONS
    | frozenset(CONVERTER21_EXTENSIONS)
    | MIDI_EXTENSIONS
    | GUITARPRO_EXTENSIONS
    | MUSESCORE_EXTENSIONS
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


def _coerce_to_score(
    parsed: music21.stream.Stream,
    default_title: str,
) -> tuple[music21.stream.Score, list[str]]:
    """Normalize a music21 parse result (Score, Part, or Opus) into a Score."""
    warnings: list[str] = []
    if isinstance(parsed, music21.stream.Opus):
        scores = list(parsed.scores)
        if not scores:
            raise ValueError("Opus file contains zero scores.")
        if len(scores) > 1:
            warnings.append(
                f"Input Opus contains {len(scores)} movements/tunes; loaded the first score only."
            )
        score = scores[0]
    elif isinstance(parsed, music21.stream.Score):
        score = parsed
    elif isinstance(parsed, music21.stream.Part):
        score = music21.stream.Score()
        score.insert(0, parsed)
    else:
        score = music21.stream.Score()
        part = music21.stream.Part()
        for el in list(parsed):
            if isinstance(el, music21.metadata.Metadata):
                score.metadata = el
            elif not isinstance(el, music21.layout.LayoutBase):
                part.append(el)
        score.insert(0, part)

    if score.metadata is None:
        score.metadata = music21.metadata.Metadata()
    if not score.metadata.title:
        score.metadata.title = default_title
    return score, warnings


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
        raise RuntimeError(
            f"Cannot parse {path.name} directly because MuseScore Studio CLI was not found. "
            "Export the score as .mxl or .musicxml from MuseScore first, or install MuseScore 4."
        )

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
        score, warnings = _coerce_to_score(parsed, path.stem)
        return LoadedDigitalScore(
            score=score,
            backend="musescore",
            format_name=path.suffix.lstrip(".").lower(),
            is_sounding_pitch_source=False,
            warnings=warnings,
        )


def _gp_instrument_from_program(program: int, track_name: str) -> music21.instrument.Instrument:
    try:
        inst = music21.instrument.instrumentFromMidiProgram(max(0, min(127, int(program))))
    except Exception:  # noqa: BLE001 — unknown GM program fallback
        inst = music21.instrument.Instrument()
    if track_name:
        inst.partName = track_name
    return inst


def _mark_gp_tie(
    prev_el: music21.note.Note | music21.chord.Chord,
    cur_el: music21.note.Note | music21.chord.Chord,
) -> None:
    if prev_el.tie is None:
        prev_el.tie = music21.tie.Tie("start")
    elif prev_el.tie.type == "stop":
        prev_el.tie = music21.tie.Tie("continue")
    cur_el.tie = music21.tie.Tie("stop")


def _load_guitarpro_score(path: Path) -> LoadedDigitalScore:
    """Convert a Guitar Pro (.gp3/.gp4/.gp5) song into a measured music21 Score."""
    import guitarpro
    import guitarpro.models as gm

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

    for t_idx, track in enumerate(non_percussion):
        part = music21.stream.Part(id=f"P{track.number}")
        track_name = (track.name or "").strip() or f"Track {track.number}"
        part.partName = track_name

        gm_prog = int(getattr(track.channel, "instrument", 25))
        m21_inst = _gp_instrument_from_program(gm_prog, track_name)
        part.insert(0, m21_inst)

        capo_offset = int(getattr(track, "offset", 0) or 0)
        is_guitar_prog = 24 <= gm_prog <= 31

        prev_ts: tuple[int, int] | None = None
        prev_ks: int | None = None
        prev_el_by_voice: dict[int, music21.note.Note | music21.chord.Chord | None] = {}

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
                        offset_ql += ql
                        continue

                    pitches: list[int] = []
                    note_tab: list[tuple[int, int]] = []
                    is_tie = all(n.type == gm.NoteType.tie for n in playable_notes)
                    for gp_note in playable_notes:
                        try:
                            sounding_midi = int(gp_note.realValue) + capo_offset
                        except IndexError:
                            sounding_midi = int(gp_note.value) + capo_offset
                        if 0 <= sounding_midi <= 127:
                            pitches.append(sounding_midi)
                            if is_guitar_prog and 1 <= gp_note.string <= 6 and 0 <= gp_note.value <= 24:
                                note_tab.append((int(gp_note.string), int(gp_note.value)))

                    if not pitches:
                        target_stream.insert(offset_ql, music21.note.Rest(duration=dur_obj))
                        prev_el = None
                    else:
                        # Deduplicate unison pitches while preserving order
                        seen_p: set[int] = set()
                        dedup_pitches: list[int] = []
                        dedup_tab: list[tuple[int, int]] = []
                        for idx_p, p_val in enumerate(pitches):
                            if p_val not in seen_p:
                                seen_p.add(p_val)
                                dedup_pitches.append(p_val)
                                if idx_p < len(note_tab):
                                    dedup_tab.append(note_tab[idx_p])

                        if len(dedup_pitches) == 1:
                            el: music21.note.Note | music21.chord.Chord = music21.note.Note(
                                dedup_pitches[0], duration=dur_obj
                            )
                        else:
                            el = music21.chord.Chord(dedup_pitches, duration=dur_obj)

                        if len(dedup_tab) == len(dedup_pitches):
                            for s_num, fret in dedup_tab:
                                el.articulations.append(music21.articulations.StringIndication(s_num))
                                el.articulations.append(music21.articulations.FretIndication(fret))

                        if is_tie and prev_el is not None:
                            _mark_gp_tie(prev_el, el)

                        target_stream.insert(offset_ql, el)
                        prev_el = el
                    offset_ql += ql

                if offset_ql < bar_ql:
                    target_stream.insert(offset_ql, music21.note.Rest(quarterLength=bar_ql - offset_ql))

                prev_el_by_voice[v_idx] = prev_el
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
    if suffix in UNSUPPORTED_GUITARPRO_EXTENSIONS:
        raise ValueError(
            f"Guitar Pro 6/7/8 ({suffix}) is not directly supported. "
            "Export as .musicxml, .mxl, or .gp5 from Guitar Pro or MuseScore first."
        )
    if suffix not in DIGITAL_EXTENSIONS:
        supported = ", ".join(sorted(DIGITAL_EXTENSIONS))
        raise ValueError(f"Unsupported digital score format {suffix!r}. Supported: {supported}")

    if suffix in GUITARPRO_EXTENSIONS:
        return _load_guitarpro_score(src)

    if suffix in MUSESCORE_EXTENSIONS:
        return _load_musescore_file(src)

    if suffix in MIDI_EXTENSIONS:
        parsed = music21.converter.parse(
            str(src),
            format="midi",
            quantizePost=True,
            quarterLengthDivisors=quantize_divisors or (4, 3),
        )
        score, warnings = _coerce_to_score(parsed, src.stem)
        return LoadedDigitalScore(
            score=score,
            backend="midi",
            format_name="midi",
            is_sounding_pitch_source=True,
            warnings=warnings,
        )

    if suffix in CONVERTER21_EXTENSIONS:
        ensure_converter21_registered()
        backend = CONVERTER21_EXTENSIONS[suffix]
        parsed = music21.converter.parse(str(src))
        score, warnings = _coerce_to_score(parsed, src.stem)
        return LoadedDigitalScore(
            score=score,
            backend=backend,
            format_name=suffix.lstrip("."),
            is_sounding_pitch_source=False,
            warnings=warnings,
        )

    # MusicXML (.musicxml, .xml, .mxl)
    parsed = music21.converter.parse(str(src))
    score, warnings = _coerce_to_score(parsed, src.stem)
    return LoadedDigitalScore(
        score=score,
        backend="musicxml",
        format_name=suffix.lstrip("."),
        is_sounding_pitch_source=False,
        warnings=warnings,
    )


def summarize_score_parts(score: music21.stream.Score) -> list[PartInfo]:
    """Extract readable summary metadata for all parts inside a Score."""
    infos: list[PartInfo] = []
    for idx, part in enumerate(score.parts, start=1):
        inst = part.recurse().getElementsByClass(music21.instrument.Instrument).first()
        clef = part.recurse().getElementsByClass(music21.clef.Clef).first()
        measures = list(part.getElementsByClass(music21.stream.Measure))

        pitches: list[int] = []
        note_count = 0
        for el in part.recurse().notes:
            if isinstance(el, music21.harmony.ChordSymbol):
                continue
            if isinstance(el, music21.note.Note):
                note_count += 1
                pitches.append(int(el.pitch.midi))
            elif isinstance(el, music21.chord.Chord):
                note_count += len(el.pitches)
                pitches.extend(int(p.midi) for p in el.pitches)

        inst_name = getattr(inst, "instrumentName", None) if inst is not None else None
        midi_prog = getattr(inst, "midiProgram", None) if inst is not None else None
        name = (
            (part.partName or "").strip()
            or (inst_name or "").strip()
            or (str(part.id).strip() if part.id else "")
            or f"Part {idx}"
        )
        clef_name = type(clef).__name__ if clef is not None else None

        infos.append(
            PartInfo(
                index=idx,
                part_id=str(part.id or f"P{idx}"),
                name=name,
                instrument_name=inst_name,
                midi_program=int(midi_prog) if midi_prog is not None else None,
                clef=clef_name,
                measure_count=len(measures),
                note_count=note_count,
                pitch_min=min(pitches) if pitches else None,
                pitch_max=max(pitches) if pitches else None,
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
