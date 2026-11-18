"""transcribe-music public Python SDK."""

from importlib.metadata import PackageNotFoundError, version

from transcribe_music.ingest.digital import inspect_digital_score as inspect_score
from transcribe_music.ingest.pdmx import PdmxStatus, get_pdmx_status, sync_pdmx
from transcribe_music.ingest.search import download_score_result, search_digital_scores
from transcribe_music.models import (
    GuitarStaffMode,
    Instrument,
    OmrBackend,
    PartInfo,
    ScoreInspection,
    ScoreSearchResult,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.pipeline import convert_music

try:
    __version__ = version("transcribe-music")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "GuitarStaffMode",
    "Instrument",
    "OmrBackend",
    "PartInfo",
    "PdmxStatus",
    "ScoreInspection",
    "ScoreSearchResult",
    "SourceType",
    "TranscriptionConfig",
    "TranscriptionResult",
    "__version__",
    "convert_music",
    "download_score_result",
    "get_pdmx_status",
    "inspect_score",
    "search_digital_scores",
    "sync_pdmx",
]
