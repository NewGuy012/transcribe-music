"""Multi-instrument PyTorch AMT adapter powered by mt3-infer (MR-MT3 & YourMT3)."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from transcribe_music.ingest.audio import load_mono_audio
from transcribe_music.models import INSTRUMENT_SPECS, Instrument, InstrumentSpec
from transcribe_music.storage.catalog import hash_file
from transcribe_music.transcribe._mt3_compat import ensure_mt3_transformers_compat
from transcribe_music.transcribe.amt_piano import record_torch_device, run_with_cpu_fallback


@dataclass(frozen=True)
class _CheckpointSpec:
    rel_path: str
    url: str
    sha256: str
    label: str


_CHECKPOINT_SPECS: dict[str, _CheckpointSpec] = {
    "yourmt3": _CheckpointSpec(
        rel_path="yourmt3/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt",
        url=(
            "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/"
            "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"
        ),
        sha256="ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25",
        label="YourMT3",
    ),
    "mr_mt3": _CheckpointSpec(
        rel_path="mr_mt3/mt3.pth",
        url="https://huggingface.co/gudgud1014/MR-MT3/resolve/main/mt3.pth",
        sha256="b8a3807ed265059abd25ad7f68142c06c35e8f6144dcaa45bd55946a3745398f",
        label="MR-MT3",
    ),
}


def _ensure_verified_checkpoint(ckpt_root: Path, model_name: str) -> Path:
    """Download atomically via .part file and SHA-256 verify an MT3 model checkpoint."""
    spec = _CHECKPOINT_SPECS[model_name]
    dest = ckpt_root / spec.rel_path
    if dest.exists():
        if hash_file(dest) == spec.sha256:
            return dest
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f"{dest.suffix}.part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(spec.url, timeout=120) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != spec.sha256:
            raise RuntimeError(f"{spec.label} checkpoint SHA-256 verification failed")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _mido_to_pretty_midi(midi_obj, temp_path: Path) -> pretty_midi.PrettyMIDI:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    midi_obj.save(str(temp_path))
    return pretty_midi.PrettyMIDI(str(temp_path))


def _extract_in_range_track(
    tracks: list[pretty_midi.Instrument],
    spec: InstrumentSpec,
) -> pretty_midi.PrettyMIDI:
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    merged = pretty_midi.Instrument(program=spec.gm_programs[0], name=spec.display_name)
    for track in tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            if spec.midi_min <= note.pitch <= spec.midi_max:
                merged.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    )
                )
    merged.notes.sort(key=lambda n: (n.start, n.pitch))
    out.instruments.append(merged)
    return out


def _filter_pretty_midi_for_instrument(
    full_midi: pretty_midi.PrettyMIDI,
    instrument: Instrument,
    warnings: list[str],
) -> pretty_midi.PrettyMIDI:
    """Select target soloist tracks from multi-track MT3 MIDI with pitch/program checks."""
    spec = INSTRUMENT_SPECS[instrument]
    non_drum = [inst for inst in full_midi.instruments if not inst.is_drum and inst.notes]
    if not non_drum:
        return pretty_midi.PrettyMIDI(initial_tempo=120.0)

    selected = [inst for inst in non_drum if inst.program in spec.gm_programs]
    if not selected and spec.ensemble_fallback_programs:
        selected = [inst for inst in non_drum if inst.program in spec.ensemble_fallback_programs]
        if selected:
            warnings.append(
                f"No dedicated {spec.display_name} GM track detected; extracted ensemble strings "
                "constrained to soloist pitch register."
            )

    if not selected:
        best = max(
            non_drum,
            key=lambda inst: (
                sum(1 for n in inst.notes if spec.midi_min <= n.pitch <= spec.midi_max),
                len(inst.notes),
            ),
        )
        selected = [best]
        warnings.append(
            f"Target GM program for {spec.display_name} absent; used dominant in-range track "
            f"(program {best.program})."
        )

    return _extract_in_range_track(selected, spec)


def transcribe_with_mt3(
    wav_path: Path,
    instrument: Instrument,
    use_multi_track: bool,
    cache_dir: Path,
    output_midi_path: Path,
    device: str = "auto",
) -> tuple[pretty_midi.PrettyMIDI, list[str]]:
    """Run MR-MT3 (fast solo) or YourMT3 (concerto/multi-instrument) on WAV audio."""
    ensure_mt3_transformers_compat()
    import mt3_infer

    warnings: list[str] = []
    ckpt_root = Path(cache_dir) / "mt3_checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    mono, sr = load_mono_audio(wav_path, target_sr=16000)
    if not mono.size:
        return pretty_midi.PrettyMIDI(initial_tempo=120.0), ["Input audio slice was empty."]

    peak = float(abs(mono).max())
    if peak > 1.0:
        mono = mono / peak

    model_name = "yourmt3" if use_multi_track else "mr_mt3"
    ckpt_path = str(_ensure_verified_checkpoint(ckpt_root, model_name))

    def _run(dev: str):
        model = mt3_infer.load_model(
            model_name,
            checkpoint_path=ckpt_path,
            device=dev,
            auto_download=False,
        )
        out = model.transcribe(mono, sr=sr)
        record_torch_device(getattr(model, "device_str", dev))
        return out

    mido_midi = run_with_cpu_fallback(device, _run)
    raw_midi_path = Path(output_midi_path).with_name(f"{Path(output_midi_path).stem}_raw.mid")
    try:
        full_midi = _mido_to_pretty_midi(mido_midi, raw_midi_path)
    finally:
        raw_midi_path.unlink(missing_ok=True)

    if use_multi_track:
        solo_midi = _filter_pretty_midi_for_instrument(full_midi, instrument, warnings)
    else:
        solo_midi = _extract_in_range_track(full_midi.instruments, INSTRUMENT_SPECS[instrument])

    solo_midi.write(str(output_midi_path))
    return solo_midi, warnings
