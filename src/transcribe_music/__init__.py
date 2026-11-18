"""transcribe-music public Python SDK."""

from transcribe_music.ingest.pdmx import PdmxStatus, get_pdmx_status, sync_pdmx
from transcribe_music.ingest.search import (
    ManualDownloadRequiredError,
    download_score_result,
    search_digital_scores,
)
from transcribe_music.models import (
    Instrument,
    OmrBackend,
    PartInfo,
    ScoreInspection,
    ScoreSearchResult,
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
    "ManualDownloadRequiredError",
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
    "transcribe_pdf",
    "transcribe_score",
]
