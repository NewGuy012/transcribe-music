"""Tests for concerto pruning, quantization, multi-page stitching, and Transkun adapter."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import numpy as np
import pretty_midi
import soundfile as sf

from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.models import Instrument
from transcribe_music.transcribe._mt3_compat import ensure_mt3_transformers_compat
from transcribe_music.transcribe.amt_piano import transcribe_piano_transkun
from transcribe_music.transcribe.concerto import (
    detect_ensemble_accompaniment,
    prune_concerto_notes,
)
from transcribe_music.transcribe.omr_pdf import stitch_musicxml_pages
from transcribe_music.transcribe.quantize import midi_to_quantized_score


def test_concerto_pruning_enforces_range_and_polyphony() -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=42)  # Cello
    # Valid cello solo notes + 3rd weaker orchestral bleed note at t=0.0
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=48, start=0.0, end=0.5))  # C3
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=55, start=0.01, end=0.5))  # G3
    inst.notes.append(pretty_midi.Note(velocity=40, pitch=60, start=0.02, end=0.5))  # bleed
    # Out-of-range bass / piccolo bleed + micro-blip + ghost note
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=28, start=1.0, end=1.5))  # below C2
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=50, start=1.0, end=1.02))  # < 45ms blip
    inst.notes.append(pretty_midi.Note(velocity=10, pitch=52, start=1.5, end=2.0))  # ghost vel
    midi.instruments.append(inst)

    pruned = prune_concerto_notes(midi, Instrument.CELLO)
    notes = pruned.instruments[0].notes
    assert len(notes) == 2
    assert {n.pitch for n in notes} == {48, 55}


def test_ensemble_detector_mid_side(tmp_path: Path) -> None:
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    solo = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    mono_stereo = np.stack([solo, solo], axis=1).astype(np.float32)
    solo_wav = tmp_path / "solo.wav"
    sf.write(solo_wav, mono_stereo, sr)
    assert detect_ensemble_accompaniment(solo_wav) is False

    wide_l = solo + 0.35 * np.sin(2 * np.pi * 330.0 * t)
    wide_r = solo - 0.35 * np.sin(2 * np.pi * 330.0 * t)
    wide_wav = tmp_path / "concerto.wav"
    sf.write(wide_wav, np.stack([wide_l, wide_r], axis=1).astype(np.float32), sr)
    assert detect_ensemble_accompaniment(wide_wav) is True


def test_quantize_piano_guitar_and_cello_tutti(tmp_path: Path) -> None:
    # Piano Grand Staff
    midi_pno = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pno = pretty_midi.Instrument(program=0)
    pno.notes.append(pretty_midi.Note(85, 48, 0.0, 0.5))  # C3 -> LH
    pno.notes.append(pretty_midi.Note(85, 64, 0.0, 0.5))  # E4 -> RH
    pno.notes.append(pretty_midi.Note(85, 67, 0.5, 1.0))  # G4 -> RH
    midi_pno.instruments.append(pno)

    score_pno = midi_to_quantized_score(midi_pno, Instrument.PIANO, tempo_bpm=120.0)
    assert len(score_pno.parts) == 2
    xml_pno = tmp_path / "pno.musicxml"
    metrics, _ = export_musicxml(score_pno, xml_pno, Instrument.PIANO)
    assert metrics.note_count == 3

    # Guitar transposition (-12 semitones: sounding E2=40 -> written E3=52)
    midi_gtr = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    gtr = pretty_midi.Instrument(program=24)
    gtr.notes.append(pretty_midi.Note(90, 40, 0.0, 0.5))
    midi_gtr.instruments.append(gtr)
    score_gtr = midi_to_quantized_score(midi_gtr, Instrument.GUITAR, tempo_bpm=120.0)
    first_note = next(score_gtr.recurse().notes)
    assert first_note.pitch.midi == 52  # Written E3 sounds E2!

    # Cello concerto Tutti cue (>= 2 empty measures)
    midi_cel = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    cel = pretty_midi.Instrument(program=42)
    cel.notes.append(pretty_midi.Note(95, 48, 0.0, 0.5))  # Measure 1
    cel.notes.append(pretty_midi.Note(95, 55, 6.0, 6.5))  # Measure 4 (bars 2-3 empty = Tutti!)
    midi_cel.instruments.append(cel)
    score_cel = midi_to_quantized_score(
        midi_cel, Instrument.CELLO, tempo_bpm=120.0, concerto_cues=True
    )
    marks = list(score_cel.recurse().getElementsByClass(music21.expressions.RehearsalMark))
    assert any(m.content == "Tutti" for m in marks)


def test_stitch_multipage_musicxml(tmp_path: Path) -> None:
    p1 = tmp_path / "page1.musicxml"
    p2 = tmp_path / "page2.musicxml"
    for path, pitches in [(p1, ["C4", "D4", "E4", "F4"]), (p2, ["G4", "A4", "B4", "C5"])]:
        s = music21.stream.Score()
        part = music21.stream.Part()
        for p in pitches:
            part.append(music21.note.Note(p, quarterLength=1.0))
        part.makeMeasures(inPlace=True)
        s.insert(0, part)
        s.write("musicxml", fp=str(path))

    merged = stitch_musicxml_pages([p1, p2], Instrument.VIOLIN, title="Merged")
    out_xml = tmp_path / "merged.musicxml"
    metrics, _ = export_musicxml(merged, out_xml, Instrument.VIOLIN)
    assert metrics.measure_count == 2
    assert metrics.note_count == 8
    root = ET.parse(out_xml).getroot()
    assert [m.get("number") for m in root.findall(".//measure")] == ["1", "2"]


def test_transkun_bundled_weights_run_without_pkg_resources(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    # Plucky synthetic piano-like decay on C4 (261.63 Hz) + E4 (329.63 Hz) + G4 (392.00 Hz)
    env = np.exp(-3.5 * t)
    wave = env * (
        0.3 * np.sin(2 * np.pi * 261.63 * t)
        + 0.25 * np.sin(2 * np.pi * 329.63 * t)
        + 0.25 * np.sin(2 * np.pi * 392.00 * t)
    )
    wav_path = tmp_path / "chord.wav"
    sf.write(wav_path, wave.astype(np.float32), sr)

    midi_path = tmp_path / "out.mid"
    midi = transcribe_piano_transkun(wav_path, midi_path, device="cpu")
    assert midi_path.exists()
    assert isinstance(midi, pretty_midi.PrettyMIDI)


def test_mt3_compat_enables_yourmt3_and_mrmt3_inference() -> None:
    import copy

    import torch
    from transformers import T5Config

    ensure_mt3_transformers_compat()
    ensure_mt3_transformers_compat()  # idempotent

    from mt3_infer.models.mr_mt3.t5 import T5ForConditionalGeneration as MRMT3Model
    from mt3_infer.models.yourmt3.config.config import audio_cfg as base_audio_cfg
    from mt3_infer.models.yourmt3.config.config import model_cfg as base_model_cfg
    from mt3_infer.models.yourmt3.model.ymt3 import YourMT3
    from mt3_infer.models.yourmt3.utils.task_manager import TaskManager

    # 1. YourMT3 Perceiver-TF + Multi-T5 (MoE + multi-channel autoregressive KV cache)
    acfg = copy.deepcopy(base_audio_cfg)
    mcfg = copy.deepcopy(base_model_cfg)
    mcfg["encoder_type"] = "perceiver-tf"
    mcfg["decoder_type"] = "multi-t5"
    mcfg["pre_decoder_type"] = "mc_shared_linear"
    mcfg["encoder"]["perceiver-tf"].update(
        {
            "num_latents": 8,
            "d_latent": 32,
            "d_latents": 32,
            "d_model": 32,
            "kv_dim": 64,
            "num_blocks": 1,
            "num_local_transformers_per_block": 1,
            "num_temporal_transformers_per_block": 1,
            "num_self_attention_heads": 4,
            "num_cross_attention_heads": 4,
            "qk_channels": 32,
            "v_channels": 32,
            "ff_layer_type": "moe",
            "ff_widening_factor": 2,
            "moe_num_experts": 2,
            "moe_topk": 1,
            "hidden_act": "silu",
            "position_encoding_type": "rope",
            "rope_partial_pe": False,
        }
    )
    mcfg["decoder"]["multi-t5"].update(
        {"d_model": 32, "num_heads": 2, "num_layers": 2, "num_channels": 4}
    )
    mcfg["conv_out_channels"] = 32
    ymt3 = YourMT3(audio_cfg=acfg, model_cfg=mcfg, task_manager=TaskManager()).eval()
    with torch.inference_mode():
        pred_ids = ymt3.inference(torch.zeros(1, 1, acfg["input_frames"]), max_token_length=4)
    assert pred_ids.shape[0] == 1
    assert pred_ids.shape[1] == 4

    # 2. YourMT3 T5 encoder + T5 decoder with trainable relative attention bias
    acfg_t5 = copy.deepcopy(base_audio_cfg)
    acfg_t5["n_mels"] = 32
    mcfg_t5 = copy.deepcopy(base_model_cfg)
    mcfg_t5["encoder"]["t5"].update(
        {"d_model": 32, "num_heads": 2, "num_layers": 1, "position_encoding_type": "trainable"}
    )
    mcfg_t5["decoder"]["t5"].update(
        {"d_model": 32, "num_heads": 2, "num_layers": 1, "position_encoding_type": "trainable"}
    )
    ymt3_t5 = YourMT3(audio_cfg=acfg_t5, model_cfg=mcfg_t5, task_manager=TaskManager()).eval()
    with torch.inference_mode():
        pred_t5 = ymt3_t5.inference(torch.zeros(1, 1, acfg_t5["input_frames"]), max_token_length=4)
    assert pred_t5.shape[0] == 1

    # 3. MR-MT3 T5ForConditionalGeneration still generates cleanly
    mr_cfg = T5Config(
        d_model=32,
        d_ff=64,
        d_kv=16,
        num_heads=2,
        num_layers=2,
        num_decoder_layers=2,
        vocab_size=32,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    mr_cfg.encoder_vocab_size = 32
    mr_model = MRMT3Model(mr_cfg).eval()
    with torch.inference_mode():
        mr_out = mr_model.generate(inputs=torch.randn(1, 4, 32), max_length=5)
    assert mr_out.shape[0] == 1
