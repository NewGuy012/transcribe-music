"""Core domain models, instrument specifications, and timestamp parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Instrument(str, Enum):
    """Supported instruments for transcription and fingering generation."""

    PIANO = "piano"
    GUITAR = "guitar"
    VIOLIN = "violin"
    CELLO = "cello"


class SourceType(str, Enum):
    """Supported ingestion source types."""

    YOUTUBE = "youtube"
    AUDIO = "audio"
    PDF = "pdf"


@dataclass(frozen=True)
class InstrumentSpec:
    """Acoustic, notation, and biomechanical metadata for an instrument."""

    instrument: Instrument
    display_name: str
    midi_min: int
    midi_max: int
    clef_mode: Literal["grand", "treble", "treble_8vb", "bass"]
    transpose_semitones: int
    max_polyphony: int
    gm_programs: tuple[int, ...]
    ensemble_fallback_programs: tuple[int, ...]
    open_string_midi: tuple[int, ...]  # Ordered string 1 (highest) -> string N (lowest)
    supports_tab: bool
    supports_thumb_fingering: bool


INSTRUMENT_SPECS: dict[Instrument, InstrumentSpec] = {
    Instrument.PIANO: InstrumentSpec(
        instrument=Instrument.PIANO,
        display_name="Piano",
        midi_min=21,  # A0
        midi_max=108,  # C8
        clef_mode="grand",
        transpose_semitones=0,
        max_polyphony=10,
        gm_programs=tuple(range(0, 8)),
        ensemble_fallback_programs=(),
        open_string_midi=(),
        supports_tab=False,
        supports_thumb_fingering=False,
    ),
    Instrument.GUITAR: InstrumentSpec(
        instrument=Instrument.GUITAR,
        display_name="Guitar",
        midi_min=40,  # E2 sounding (E3 written)
        midi_max=88,  # E6 sounding (E7 written)
        clef_mode="treble_8vb",
        transpose_semitones=-12,
        max_polyphony=6,
        gm_programs=tuple(range(24, 32)),
        ensemble_fallback_programs=(),
        open_string_midi=(64, 59, 55, 50, 45, 40),  # E4, B3, G3, D3, A2, E2
        supports_tab=True,
        supports_thumb_fingering=False,
    ),
    Instrument.VIOLIN: InstrumentSpec(
        instrument=Instrument.VIOLIN,
        display_name="Violin",
        midi_min=55,  # G3
        midi_max=105,  # A7
        clef_mode="treble",
        transpose_semitones=0,
        max_polyphony=2,
        gm_programs=(40,),
        ensemble_fallback_programs=(48, 49),
        open_string_midi=(76, 69, 62, 55),  # E5, A4, D4, G3
        supports_tab=False,
        supports_thumb_fingering=False,
    ),
    Instrument.CELLO: InstrumentSpec(
        instrument=Instrument.CELLO,
        display_name="Cello",
        midi_min=36,  # C2
        midi_max=84,  # C6
        clef_mode="bass",
        transpose_semitones=0,
        max_polyphony=2,
        gm_programs=(42,),
        ensemble_fallback_programs=(48, 49, 43),
        open_string_midi=(57, 50, 43, 36),  # A3, D3, G2, C2
        supports_tab=False,
        supports_thumb_fingering=True,
    ),
}

_TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def parse_timestamp(value: str | int | float) -> float:
    """Parse seconds or HH:MM:SS / MM:SS string into non-negative float seconds."""
    if isinstance(value, (int, float)):
        sec = float(value)
        if sec < 0:
            raise ValueError(f"Timestamp must be non-negative, got {value}")
        return sec

    raw = value.strip()
    if not raw:
        raise ValueError("Empty timestamp")

    try:
        sec = float(raw)
        if sec < 0:
            raise ValueError(f"Timestamp must be non-negative, got {value}")
        return sec
    except ValueError:
        pass

    match = _TIME_RE.match(raw)
    if not match:
        raise ValueError(
            f"Invalid timestamp {value!r}. Expected seconds (e.g. '75.5'), 'MM:SS', or 'HH:MM:SS'."
        )
    hours_s, mins_s, secs_s = match.groups()
    hours = int(hours_s) if hours_s is not None else 0
    mins = int(mins_s)
    secs = float(secs_s)
    if mins >= 60 or secs >= 60:
        raise ValueError(f"Minutes/seconds must be < 60 in timestamp {value!r}")
    return hours * 3600.0 + mins * 60.0 + secs


class TimeRange(BaseModel):
    """Start and end timestamps (in seconds) for YouTube slicing."""

    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)

    @field_validator("end_sec")
    @classmethod
    def _validate_order(cls, v: float, info) -> float:
        start = info.data.get("start_sec")
        if start is not None and v <= start:
            raise ValueError(f"end_sec ({v}) must be greater than start_sec ({start})")
        return v

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @classmethod
    def from_inputs(cls, start: str | int | float, end: str | int | float) -> TimeRange:
        return cls(start_sec=parse_timestamp(start), end_sec=parse_timestamp(end))


class TranscriptionConfig(BaseModel):
    """Runtime configuration for transcription, enrichment, and caching."""

    instrument: Instrument
    concerto_mode: Literal["auto", "solo", "concerto"] = "auto"
    device: Literal["auto", "mps", "cpu"] = "auto"
    use_cache: bool = True
    enrich_chords: bool = True
    enrich_fingering: bool = True
    tempo_bpm: float | None = Field(default=None, gt=20.0, lt=320.0)
    quantize_divisors: tuple[int, ...] = (4, 3)  # 16th notes + triplets
    cache_dir: Path = Path(".transcribe_cache")
    pdf_dpi: int = Field(default=300, ge=150, le=600)


class TranscriptionResult(BaseModel):
    """Structured result metadata returned by CLI/SDK/pipeline."""

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
    warnings: list[str] = Field(default_factory=list)
