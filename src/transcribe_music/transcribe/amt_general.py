"""Multi-instrument PyTorch AMT adapter powered by mt3-infer (MR-MT3 & YourMT3)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pretty_midi

from transcribe_music.ingest.audio import load_mono_audio
from transcribe_music.models import INSTRUMENT_SPECS, Instrument, InstrumentSpec
from transcribe_music.transcribe.amt_piano import run_with_cpu_fallback

_YOURMT3_CKPT_REL = (
    "yourmt3/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt"
)
_YOURMT3_HF_URL = (
    "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/"
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"
)


def _ensure_yourmt3_checkpoint(ckpt_root: Path) -> Path:
    """Download YourMT3 MoE checkpoint via direct HTTPS (avoids git-lfs prerequisite)."""
    dest = ckpt_root / _YOURMT3_CKPT_REL
    if dest.exists() and dest.stat().st_size > 100_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".ckpt.part")
    urllib.request.urlretrieve(_YOURMT3_HF_URL, str(tmp))
    tmp.replace(dest)
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
    import mt3_infer

    warnings: list[str] = []
    ckpt_root = Path(cache_dir) / "mt3_checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MT3_CHECKPOINT_DIR", str(ckpt_root))

    mono, sr = load_mono_audio(wav_path, target_sr=16000)
    if not mono.size:
        return pretty_midi.PrettyMIDI(initial_tempo=120.0), ["Input audio slice was empty."]

    peak = float(abs(mono).max())
    if peak > 1.0:
        mono = mono / peak

    model_name = "yourmt3" if use_multi_track else "mr_mt3"
    ckpt_path = str(_ensure_yourmt3_checkpoint(ckpt_root)) if model_name == "yourmt3" else None

    def _run(dev: str):
        model = mt3_infer.load_model(
            model_name,
            checkpoint_path=ckpt_path,
            device=dev,
            auto_download=True,
        )
        return model.transcribe(mono, sr=sr)

    mido_midi = run_with_cpu_fallback(device, _run)
    raw_midi_path = Path(output_midi_path).with_name(f"{Path(output_midi_path).stem}_raw.mid")
    full_midi = _mido_to_pretty_midi(mido_midi, raw_midi_path)

    if use_multi_track:
        solo_midi = _filter_pretty_midi_for_instrument(full_midi, instrument, warnings)
    else:
        solo_midi = _extract_in_range_track(full_midi.instruments, INSTRUMENT_SPECS[instrument])

    solo_midi.write(str(output_midi_path))
    return solo_midi, warnings
