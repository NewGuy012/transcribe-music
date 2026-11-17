"""Tests for concerto pruning, quantization, multi-page stitching, and Transkun adapter."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import numpy as np
import pretty_midi
import pytest
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
    import warnings

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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        midi = transcribe_piano_transkun(wav_path, midi_path, device="cpu")
    assert midi_path.exists()
    assert isinstance(midi, pretty_midi.PrettyMIDI)
    messages = [str(w.message) for w in caught]
    assert not any("torch.jit.script" in msg for msg in messages)
    assert not any("torch.utils.checkpoint" in msg for msg in messages)
    assert not any("requires_grad=True" in msg for msg in messages)


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


def test_mt3_compat_enables_mps_device_for_mrmt3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch
    from transformers import T5Config

    ensure_mt3_transformers_compat()
    from mt3_infer.adapters.mr_mt3 import MRMT3Adapter
    from mt3_infer.models.mr_mt3.t5 import T5ForConditionalGeneration as MRMT3Model
    from mt3_infer.utils import framework

    expected_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    assert framework.get_device("mps") == expected_dev

    adapter = MRMT3Adapter()
    cfg = T5Config.from_dict(adapter.model_config)
    ckpt = tmp_path / "mt3.pth"
    torch.save(MRMT3Model(cfg).state_dict(), ckpt)
    adapter.load_model(str(ckpt), device="mps")
    assert adapter.device_str == expected_dev

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="MPS requested"):
        assert framework.get_device("mps") == "cpu"


def test_ensure_verified_checkpoint_replaces_corrupt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib
    import io

    from transcribe_music.transcribe import amt_general

    payload = b"valid-mr-mt3-checkpoint-bytes"
    expected_sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        amt_general._CHECKPOINT_SPECS,
        "mr_mt3",
        amt_general._CheckpointSpec(
            rel_path="mr_mt3/mt3.pth",
            url="https://example.invalid/mt3.pth",
            sha256=expected_sha,
            label="MR-MT3",
        ),
    )

    corrupt = tmp_path / "mr_mt3" / "mt3.pth"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"corrupted")

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        amt_general.urllib.request,
        "urlopen",
        lambda url, timeout=120: _FakeResp(payload),
    )
    resolved = amt_general._ensure_verified_checkpoint(tmp_path, "mr_mt3")
    assert resolved == corrupt
    assert resolved.read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))


def test_quantize_triplets_and_mixed_grids_export_cleanly(tmp_path: Path) -> None:
    from transcribe_music.enrich.fingering import apply_fingering

    # 120 BPM => 1 beat = 0.5s
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    cello = pretty_midi.Instrument(program=42)
    # Measure 1, beat 1: exact triplets (0, 1/3, 2/3 beats)
    for k in range(3):
        cello.notes.append(pretty_midi.Note(90, 50 + k, k / 6.0, (k + 1) / 6.0))
    # Beat 2: 16th + dotted 8th (1.0, 1.25 beats)
    cello.notes.append(pretty_midi.Note(90, 55, 0.5, 0.625))
    cello.notes.append(pretty_midi.Note(90, 57, 0.625, 1.0))
    # Beats 3-4: double stops with raw triplet-adjacent onsets crossing into binary beat
    # (mirrors the Haydn concerto m.7 timing pattern that previously produced 0.3333 / 0.4167)
    cello.notes.append(pretty_midi.Note(100, 62, 1.0, 1.1944))
    cello.notes.append(pretty_midi.Note(100, 66, 1.0, 1.1944))
    cello.notes.append(pretty_midi.Note(100, 64, 1.1944, 1.4302))
    cello.notes.append(pretty_midi.Note(100, 67, 1.1944, 1.4302))
    cello.notes.append(pretty_midi.Note(100, 67, 1.4302, 1.615))
    cello.notes.append(pretty_midi.Note(100, 69, 1.615, 1.928))
    # Cross-beat tie from triplet beat (beat 4 + 2/3 = 2.333s) into binary beat (beat 5 + 1/4 = 2.625s)
    cello.notes.append(pretty_midi.Note(95, 60, 2.333, 2.625))
    cello.notes.append(pretty_midi.Note(95, 62, 2.625, 3.0))
    midi.instruments.append(cello)

    score = midi_to_quantized_score(midi, Instrument.CELLO, tempo_bpm=120.0)
    for el in score.recurse().notesAndRests:
        assert el.duration.type not in {"inexpressible", "zero", "2048th", "duplex-maxima"}

    apply_fingering(score, Instrument.CELLO)
    out_xml = tmp_path / "triplets_mixed.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.CELLO)
    assert not warnings
    assert metrics.measure_count == 2
    assert metrics.note_count >= 12
    assert metrics.fingering_count >= 12

    root = ET.parse(out_xml).getroot()
    time_mods = {
        (tm.findtext("actual-notes"), tm.findtext("normal-notes"))
        for tm in root.findall(".//time-modification")
    }
    assert time_mods <= {("3", "2")}
    assert ("3", "2") in time_mods


def test_mt3_compat_suppresses_third_party_noise_and_keeps_unrelated_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import subprocess
    import sys

    from transcribe_music.transcribe._mt3_compat import (
        _THIRD_PARTY_LOG_FILTERS,
        _DropLogSubstringsFilter,
    )

    ensure_mt3_transformers_compat()
    ensure_mt3_transformers_compat()

    for logger_name, _ in _THIRD_PARTY_LOG_FILTERS:
        compat_filters = [
            f
            for f in logging.getLogger(logger_name).filters
            if isinstance(f, _DropLogSubstringsFilter)
        ]
        assert len(compat_filters) == 1

    with caplog.at_level(logging.WARNING):
        logging.getLogger("transformers.models.t5.modeling_t5").warning(
            "Instantiating a decoder T5Attention without passing `layer_idx` is not recommended"
        )
        logging.getLogger("transformers.models.t5.modeling_t5").warning(
            "Unexpected model error that should remain visible"
        )

    assert "without passing `layer_idx`" not in caplog.text
    assert "Unexpected model error that should remain visible" in caplog.text

    script = (
        "from transcribe_music.transcribe._mt3_compat import ensure_mt3_transformers_compat; "
        "ensure_mt3_transformers_compat(); "
        "import mt3_infer; "
        "from mt3_infer.models.yourmt3.model import ymt3; "
        "from transformers import T5Config; "
        "from transformers.models.t5.modeling_t5 import T5Attention; "
        "cfg = T5Config(is_decoder=True, d_model=32, d_kv=16, num_heads=2, num_layers=1); "
        "_ = cfg.use_return_dict; "
        "_ = T5Attention(cfg, layer_idx=None)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    combined = proc.stdout + proc.stderr
    assert "Redirects are currently not supported in MacOs" not in combined
    assert "torch.cuda.amp.autocast" not in combined
    assert "without passing `layer_idx`" not in combined
    assert "use_return_dict" not in combined


def test_transcribe_with_mt3_suppresses_stdout_chatter_keeps_domain_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import mido
    import mt3_infer

    from transcribe_music.transcribe import amt_general

    wav_path = tmp_path / "excerpt.wav"
    sf.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)
    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"ckpt")

    monkeypatch.setattr(amt_general, "_ensure_verified_checkpoint", lambda *_args: ckpt_path)

    class _FakeModel:
        device_str = "cpu"

        def transcribe(self, audio: np.ndarray, sr: int = 16000) -> mido.MidiFile:
            print("Running inference on 1 segments...")
            print("Got 1 prediction batches")
            mid = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("program_change", program=48, time=0))  # String Ensemble 1
            track.append(mido.Message("note_on", note=48, velocity=90, time=0))
            track.append(mido.Message("note_off", note=48, velocity=0, time=480))
            mid.tracks.append(track)
            return mid

    def _fake_load_model(*_args, **_kwargs):
        print("Loading YourMT3 model: YPTF.MoE+Multi (noPS)")
        print("Model loaded successfully! (type: YourMT3)")
        return _FakeModel()

    monkeypatch.setattr(mt3_infer, "load_model", _fake_load_model)

    out_mid = tmp_path / "out.mid"
    midi, warnings = amt_general.transcribe_with_mt3(
        wav_path=wav_path,
        instrument=Instrument.CELLO,
        use_multi_track=True,
        cache_dir=tmp_path / "cache",
        output_midi_path=out_mid,
        device="cpu",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert out_mid.exists()
    assert len(midi.instruments[0].notes) == 1
    assert any("No dedicated Cello GM track detected" in w for w in warnings)
