"""Core domain models, instrument specifications, and inspection metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class Instrument(str, Enum):
    """Supported target instruments for score adaptation and fingering generation."""

    PIANO = "piano"
    GUITAR = "guitar"
    VIOLIN = "violin"
    CELLO = "cello"


class SourceType(str, Enum):
    """Supported score ingestion source types."""

    DIGITAL = "digital"
    PDF = "pdf"


class OmrBackend(str, Enum):
    """Supported Optical Music Recognition backends for PDF fallback."""

    AUTO = "auto"
    AUDIVERIS = "audiveris"
    HOMR = "homr"


class GuitarStaffMode(str, Enum):
    """Supported MusicXML staff layouts when exporting guitar scores."""

    NOTATION = "notation"
    TAB = "tab"
    BOTH = "both"


@dataclass(frozen=True)
class InstrumentSpec:
    """Acoustic, notation, and biomechanical metadata for an instrument."""

    instrument: Instrument
    display_name: str
    midi_min: int
    midi_max: int
    clef_mode: Literal["grand", "treble", "treble_8vb", "bass"]
    transpose_semitones: int
    gm_programs: tuple[int, ...]
    open_string_midi: tuple[int, ...]  # Ordered string 1 (highest) -> string N (lowest)


INSTRUMENT_SPECS: dict[Instrument, InstrumentSpec] = {
    Instrument.PIANO: InstrumentSpec(
        instrument=Instrument.PIANO,
        display_name="Piano",
        midi_min=21,  # A0
        midi_max=108,  # C8
        clef_mode="grand",
        transpose_semitones=0,
        gm_programs=tuple(range(8)),
        open_string_midi=(),
    ),
    Instrument.GUITAR: InstrumentSpec(
        instrument=Instrument.GUITAR,
        display_name="Guitar",
        midi_min=40,  # E2 sounding (E3 written)
        midi_max=88,  # E6 sounding (E7 written)
        clef_mode="treble_8vb",
        transpose_semitones=-12,
        gm_programs=tuple(range(24, 32)),
        open_string_midi=(64, 59, 55, 50, 45, 40),  # E4, B3, G3, D3, A2, E2
    ),
    Instrument.VIOLIN: InstrumentSpec(
        instrument=Instrument.VIOLIN,
        display_name="Violin",
        midi_min=55,  # G3
        midi_max=105,  # A7
        clef_mode="treble",
        transpose_semitones=0,
        gm_programs=(40,),
        open_string_midi=(76, 69, 62, 55),  # E5, A4, D4, G3
    ),
    Instrument.CELLO: InstrumentSpec(
        instrument=Instrument.CELLO,
        display_name="Cello",
        midi_min=36,  # C2
        midi_max=84,  # C6
        clef_mode="bass",
        transpose_semitones=0,
        gm_programs=(42,),
        open_string_midi=(57, 50, 43, 36),  # A3, D3, G2, C2
    ),
}


@dataclass(frozen=True)
class PartInfo:
    """Summary metadata for a single part/track inside a digital score."""

    index: int
    part_id: str
    name: str
    instrument_name: str | None = None
    midi_program: int | None = None
    clef: str | None = None
    measure_count: int = 0
    note_count: int = 0
    pitch_min: int | None = None
    pitch_max: int | None = None
    tuning: str | None = None
    capo: int | None = None


@dataclass(frozen=True)
class ScoreInspection:
    """Structured overview of all parts detected inside a digital music file."""

    source_path: Path
    format_name: str
    title: str | None = None
    parts: list[PartInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))


@dataclass(frozen=True)
class TranscriptionConfig:
    """Runtime configuration for score ingestion, part selection, enrichment, and caching."""

    instrument: Instrument
    part: str | None = None
    title: str | None = None
    composer: str | None = None
    use_cache: bool = True
    enrich_chords: bool = True
    chord_window_beats: float = 2.0
    include_slash_chords: bool = True
    roman_numerals: bool = False
    enrich_fingering: bool = True
    preserve_existing_chords: bool = False
    preserve_existing_fingering: bool = False
    preserve_existing_tab: bool = True
    guitar_staff: GuitarStaffMode = GuitarStaffMode.NOTATION
    quantize_divisors: tuple[int, ...] = (4, 3)
    cache_dir: Path = Path(".transcribe_cache")
    pdf_dpi: int = 300
    omr_backend: OmrBackend = OmrBackend.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument", Instrument(self.instrument))
        object.__setattr__(self, "chord_window_beats", float(self.chord_window_beats))
        object.__setattr__(self, "include_slash_chords", bool(self.include_slash_chords))
        object.__setattr__(self, "roman_numerals", bool(self.roman_numerals))
        object.__setattr__(self, "guitar_staff", GuitarStaffMode(self.guitar_staff))
        object.__setattr__(self, "omr_backend", OmrBackend(self.omr_backend))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "quantize_divisors", tuple(self.quantize_divisors))
        if not math.isfinite(self.chord_window_beats) or self.chord_window_beats <= 0.0:
            raise ValueError(
                f"chord_window_beats must be a positive finite number, got {self.chord_window_beats}"
            )
        if not (150 <= self.pdf_dpi <= 600):
            raise ValueError(f"pdf_dpi must be between 150 and 600, got {self.pdf_dpi}")


@dataclass(frozen=True)
class TranscriptionResult:
    """Structured result metadata returned by CLI, SDK, and pipeline."""

    job_hash: str
    source_type: SourceType
    source_ref: str
    instrument: Instrument
    musicxml_path: Path
    cached: bool = False
    elapsed_sec: float = 0.0
    measure_count: int = 0
    note_count: int = 0
    chord_count: int = 0
    fingering_count: int = 0
    backend: str | None = None
    selected_part: str | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", SourceType(self.source_type))
        object.__setattr__(self, "instrument", Instrument(self.instrument))
        object.__setattr__(self, "musicxml_path", Path(self.musicxml_path))


@dataclass(frozen=True)
class ScoreSearchResult:
    """Search hit for a downloadable digital symbolic music score."""

    title: str
    format_ext: str
    source: str
    filename: str
    composer: str | None = None
    download_url: str | None = None
    edition_note: str | None = None
    local_corpus_path: str | None = None
    pdmx_dir: str | None = None
    pdmx_member: str | None = None
    gp_dir: str | None = None
    gp_member: str | None = None
    score: float = 0.0
