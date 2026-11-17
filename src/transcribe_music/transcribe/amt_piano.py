"""Direct PyTorch adapter for Transkun piano transcription (no pkg_resources dependency)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any

import moduleconf
import numpy as np
import pretty_midi
import soundfile as sf
import soxr
import torch
import transkun
from transkun.Data import writeMidi

_LAST_TORCH_DEVICE: ContextVar[str | None] = ContextVar("_LAST_TORCH_DEVICE", default=None)


def record_torch_device(device: str) -> None:
    """Record the torch device actually used for the current transcription call."""
    _LAST_TORCH_DEVICE.set(device)


def consume_last_torch_device() -> str | None:
    """Return and clear the most recently recorded torch device for this context."""
    dev = _LAST_TORCH_DEVICE.get()
    _LAST_TORCH_DEVICE.set(None)
    return dev


def resolve_torch_device(preferred: str = "auto") -> str:
    """Resolve PyTorch compute device cleanly."""
    if preferred in {"cpu", "mps"}:
        if preferred == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return preferred
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_with_cpu_fallback[T](preferred: str, fn: Callable[[str], T]) -> T:
    """Run torch callable on resolved device, falling back to CPU if MPS fails."""
    resolved = resolve_torch_device(preferred)
    _LAST_TORCH_DEVICE.set(None)
    try:
        out = fn(resolved)
        if _LAST_TORCH_DEVICE.get() is None:
            _LAST_TORCH_DEVICE.set(resolved)
        return out
    except Exception:
        if resolved != "cpu":
            _LAST_TORCH_DEVICE.set(None)
            out = fn("cpu")
            if _LAST_TORCH_DEVICE.get() is None:
                _LAST_TORCH_DEVICE.set("cpu")
            return out
        raise


@lru_cache(maxsize=2)
def _load_transkun_model(device: str) -> Any:
    pkg_dir = Path(transkun.__file__).resolve().parent
    weight_path = pkg_dir / "pretrained" / "2.0.pt"
    conf_path = pkg_dir / "pretrained" / "2.0.conf"
    if not weight_path.exists() or not conf_path.exists():
        raise FileNotFoundError(f"Bundled transkun weights missing in {pkg_dir / 'pretrained'}")

    conf_mgr = moduleconf.parseFromFile(str(conf_path))
    model_cls = conf_mgr["Model"].module.TransKun
    conf = conf_mgr["Model"].config
    checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)

    model = model_cls(conf=conf)
    state_dict = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _read_stereo_float32(wav_path: Path) -> tuple[int, np.ndarray]:
    audio, fs = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return int(fs), audio


@contextmanager
def _suppress_transkun_torch_warnings() -> Iterator[None]:
    """Suppress harmless PyTorch 2.x/3.14 deprecation warnings triggered by bundled Transkun."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*`torch\.jit\.script` is not supported in Python 3\.14\+.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.utils\.checkpoint: the use_reentrant parameter.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*None of the inputs have requires_grad=True.*",
            category=UserWarning,
        )
        yield


def transcribe_piano_transkun(
    wav_path: Path,
    output_midi_path: Path,
    device: str = "auto",
) -> pretty_midi.PrettyMIDI:
    """Transcribe solo piano WAV into PrettyMIDI using bundled Transkun v2 model."""
    fs, audio = _read_stereo_float32(wav_path)

    def _run(dev: str) -> Any:
        with _suppress_transkun_torch_warnings():
            model = _load_transkun_model(dev)
            resampled = soxr.resample(audio, fs, model.fs) if fs != model.fs else audio
            x = torch.from_numpy(resampled).to(dev)
            with torch.inference_mode():
                return model.transcribe(x, discardSecondHalf=False)

    notes_est = run_with_cpu_fallback(device, _run)
    output_midi_path = Path(output_midi_path)
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    writeMidi(notes_est).write(str(output_midi_path))
    return pretty_midi.PrettyMIDI(str(output_midi_path))
