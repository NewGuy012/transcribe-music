"""Core domain models, instrument specifications, and inspection metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


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


class PartInfo(BaseModel):
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


class ScoreInspection(BaseModel):
    """Structured overview of all parts detected inside a digital music file."""

    source_path: Path
    format_name: str
    title: str | None = None
    parts: list[PartInfo] = Field(default_factory=list)


class TranscriptionConfig(BaseModel):
    """Runtime configuration for score ingestion, part selection, enrichment, and caching."""

    instrument: Instrument
    part: str | None = None
    use_cache: bool = True
    enrich_chords: bool = True
    enrich_fingering: bool = True
    preserve_existing_chords: bool = False
    preserve_existing_fingering: bool = False
    preserve_existing_tab: bool = True
    guitar_staff: GuitarStaffMode = GuitarStaffMode.NOTATION
    quantize_divisors: tuple[int, ...] = (4, 3)
    cache_dir: Path = Path(".transcribe_cache")
    pdf_dpi: int = Field(default=300, ge=150, le=600)
    omr_backend: OmrBackend = OmrBackend.AUTO


class TranscriptionResult(BaseModel):
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
    warnings: list[str] = Field(default_factory=list)


class ScoreSearchResult(BaseModel):
    """Search hit for a downloadable digital symbolic music score."""

    title: str
    composer: str | None = None
    format_ext: str
    source: str
    filename: str
    download_url: str | None = None
    page_url: str | None = None
    edition_note: str | None = None
    local_corpus_path: str | None = None
    pdmx_dir: str | None = None
    pdmx_member: str | None = None
    gp_dir: str | None = None
    gp_member: str | None = None
    score: float = 0.0
