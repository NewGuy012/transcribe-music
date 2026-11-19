"""Direct ingestion and inspection for digital symbolic music formats."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import music21

from transcribe_music.ingest.ascii_tab import parse_ascii_tab_file
from transcribe_music.ingest.guitarpro import parse_gp35_file, parse_gpif_file
from transcribe_music.models import PartInfo, ScoreInspection
from transcribe_music.score_utils import (
    coerce_to_score,
    element_midis,
    find_executable,
    format_tuning_label,
    get_guitar_track_meta,
    iter_playable_notes,
)

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


@dataclass(frozen=True)
class LoadedDigitalScore:
    """Parsed symbolic score along with provenance metadata."""

    score: music21.stream.Score
    backend: str
    format_name: str
    is_sounding_pitch_source: bool
    warnings: list[str] = field(default_factory=list)


@cache
def ensure_converter21_registered() -> None:
    """Register converter21 parsers once for Humdrum (.krn), MEI, and ABC."""
    try:
        import converter21
    except ImportError as exc:
        raise RuntimeError(
            "Loading .krn/.mei/.abc requires the optional 'formats' extra (`converter21`). "
            "Install with: uv sync --extra formats"
        ) from exc
    converter21.register()


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
    return find_executable(
        ("mscore", "mscore4", "musescore", "musescore4"),
        (
            Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
            Path("/Applications/MuseScore 3.app/Contents/MacOS/mscore"),
            Path.home() / "Applications" / "MuseScore 4.app" / "Contents" / "MacOS" / "mscore",
        ),
    )


def _load_musescore_file(path: Path) -> LoadedDigitalScore:
    mscore_bin = _find_musescore_executable()
    if mscore_bin is None:
        suffix = path.suffix.lower()
        if suffix in MUSESCORE_GP_EXTENSIONS:
            label = f"Guitar Pro 6/7/8 ({path.name})"
            hint = "Export as .musicxml, .mxl, or .gp5 from Guitar Pro/MuseScore first, or install MuseScore 4."
        elif suffix in CAPELLA_EXTENSIONS:
            label = f"Capella ({path.name})"
            hint = "Export as .musicxml or .mxl from Capella/MuseScore first, or install MuseScore 4."
        else:
            label = path.name
            hint = "Export the score as .mxl or .musicxml from MuseScore first, or install MuseScore 4."
        raise RuntimeError(f"Cannot parse {label} directly because MuseScore Studio CLI was not found. {hint}")

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


def _load_lilypond_file(path: Path) -> LoadedDigitalScore:
    """Convert LilyPond (.ly) via pure-parser `python-ly` ('ly musicxml') without executing Scheme."""
    ly_bin = shutil.which("ly")
    if ly_bin is None:
        raise RuntimeError(
            f"Cannot parse LilyPond ({path.name}) because 'ly' (python-ly) was not found. "
            "Install python-ly or export the score as .musicxml/.mid."
        )
    proc = subprocess.run(
        [ly_bin, "musicxml", str(path.resolve())],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    xml_text = proc.stdout or ""
    if proc.returncode != 0 or ("<score-partwise" not in xml_text and "<score-timewise" not in xml_text):
        detail = proc.stderr.strip() or proc.stdout.strip() or "invalid MusicXML output"
        raise RuntimeError(f"python-ly failed converting {path.name}: {detail}")
    with tempfile.TemporaryDirectory(prefix="tm_ly_") as tmp_dir:
        tmp_xml = Path(tmp_dir) / f"{path.stem}.musicxml"
        tmp_xml.write_text(xml_text, encoding="utf-8")
        parsed = music21.converter.parse(str(tmp_xml), format="musicxml")
        return _to_loaded_score(parsed, path, "python_ly", format_name="ly")


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
            "Use `transcribe-music convert` on PDF files."
        )
    if suffix not in (DIGITAL_EXTENSIONS | _LOCAL_ASCII_TAB_EXTENSIONS):
        supported = ", ".join(sorted(DIGITAL_EXTENSIONS))
        raise ValueError(f"Unsupported digital score format {suffix!r}. Supported: {supported}")

    if suffix in _LOCAL_ASCII_TAB_EXTENSIONS:
        score, warnings = parse_ascii_tab_file(src)
        return LoadedDigitalScore(score, "ascii_tab", suffix.lstrip("."), True, warnings)

    if suffix in GUITARPRO_EXTENSIONS:
        score, warnings = parse_gp35_file(src)
        return LoadedDigitalScore(score, "pyguitarpro", suffix.lstrip("."), True, warnings)

    if suffix == ".gp":
        gpif_res = parse_gpif_file(src)
        if gpif_res is not None:
            score, warnings = gpif_res
            return LoadedDigitalScore(score, "gpif", "gp", True, warnings)

    if suffix in (MUSESCORE_EXTENSIONS | MUSESCORE_GP_EXTENSIONS):
        return _load_musescore_file(src)
    if suffix in CAPELLA_EXTENSIONS:
        return _load_capella_file(src)
    if suffix in LILYPOND_EXTENSIONS:
        return _load_lilypond_file(src)

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

    return _to_loaded_score(music21.converter.parse(str(src)), src, "musicxml")


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
        meta = get_guitar_track_meta(part)
        infos.append(
            PartInfo(
                index=idx,
                part_id=str(part.id or f"P{idx}"),
                name=name,
                instrument_name=inst_name,
                midi_program=int(midi_prog) if midi_prog is not None else None,
                clef=type(clef).__name__ if clef is not None else None,
                measure_count=len(measures),
                note_count=len(pitches),
                pitch_min=min(pitches) if pitches else None,
                pitch_max=max(pitches) if pitches else None,
                tuning=format_tuning_label(meta.open_string_midi) if meta is not None else None,
                capo=meta.capo_fret if meta is not None and meta.capo_fret > 0 else None,
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
