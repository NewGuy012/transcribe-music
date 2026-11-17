"""Direct PyTorch adapter for Transkun piano transcription (no pkg_resources dependency)."""

from __future__ import annotations

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


def resolve_torch_device(preferred: str = "auto") -> str:
    """Resolve PyTorch compute device cleanly."""
    if preferred in {"cpu", "mps"}:
        if preferred == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def transcribe_piano_transkun(
    wav_path: Path,
    output_midi_path: Path,
    device: str = "auto",
) -> pretty_midi.PrettyMIDI:
    """Transcribe solo piano WAV into PrettyMIDI using bundled Transkun v2 model."""
    resolved_device = resolve_torch_device(device)
    fs, audio = _read_stereo_float32(wav_path)

    try:
        model = _load_transkun_model(resolved_device)
        if fs != model.fs:
            audio = soxr.resample(audio, fs, model.fs)
        x = torch.from_numpy(audio).to(resolved_device)
        with torch.inference_mode():
            notes_est = model.transcribe(x, discardSecondHalf=False)
    except Exception:
        if resolved_device != "cpu":
            # Automatic CPU fallback if any MPS op fails on specific macOS versions
            model = _load_transkun_model("cpu")
            if fs != model.fs:
                audio = soxr.resample(audio, fs, model.fs)
            x = torch.from_numpy(audio).to("cpu")
            with torch.inference_mode():
                notes_est = model.transcribe(x, discardSecondHalf=False)
        else:
            raise

    output_midi_path = Path(output_midi_path)
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    out_midi = writeMidi(notes_est)
    out_midi.write(str(output_midi_path))
    return pretty_midi.PrettyMIDI(str(output_midi_path))
