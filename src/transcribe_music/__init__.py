"""transcribe-music public Python SDK."""

from transcribe_music.models import (
    Instrument,
    PartInfo,
    ScoreInspection,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.pipeline import (
    convert_music,
    inspect_score,
    transcribe_pdf,
    transcribe_score,
)

__version__ = "0.2.0"

__all__ = [
    "Instrument",
    "PartInfo",
    "ScoreInspection",
    "SourceType",
    "TranscriptionConfig",
    "TranscriptionResult",
    "__version__",
    "convert_music",
    "inspect_score",
    "transcribe_pdf",
    "transcribe_score",
]
