"""Unit and Scenario P4 benchmark tests for dsp_engine."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp_engine import (
    DenoiseParams,
    load_wav_bytes,
    run_full_pipeline,
    save_wav_bytes,
    synthesize_dtmf_signal,
)


def test_all_16_dtmf_symbols_decoded_under_noise() -> None:
    sequence = "123A456B789C*0#D"
    fs = 8000
    clean, noisy = synthesize_dtmf_signal(
        sequence=sequence,
        fs=fs,
        tone_ms=80.0,
        pause_ms=60.0,
        lead_silence_ms=200.0,
        tail_silence_ms=150.0,
        snr_db=10.0,
        seed=101,
    )
    wav_bytes = save_wav_bytes(fs, noisy)
    params = DenoiseParams(window_type="hann", nperseg=512, overlap_pct=75.0, alpha=2.0, beta=0.02)
    result = run_full_pipeline(wav_bytes, params, target_sequence=sequence, clean_reference=clean)

    assert result["decoding"]["sequence"] == sequence
    assert len(result["decoding"]["tones"]) == len(sequence)
    assert result["metrics"]["target_eval"]["exact_match"] is True
    assert result["metrics"]["target_eval"]["accuracy_pct"] == 100.0


def test_repeated_identical_digits_and_short_bursts() -> None:
    sequence = "1122**##"
    fs = 8000
    _, noisy = synthesize_dtmf_signal(
        sequence=sequence,
        fs=fs,
        tone_ms=45.0,
        pause_ms=45.0,
        lead_silence_ms=180.0,
        tail_silence_ms=150.0,
        snr_db=12.0,
        seed=202,
    )
    wav_bytes = save_wav_bytes(fs, noisy)
    params = DenoiseParams(window_type="hann", nperseg=256, overlap_pct=75.0, alpha=1.8, beta=0.02)
    result = run_full_pipeline(wav_bytes, params)

    assert result["decoding"]["sequence"] == sequence


def test_scenario_p4_quantitative_targets() -> None:
    sequence = "0812345#"
    fs = 8000
    clean, noisy = synthesize_dtmf_signal(
        sequence=sequence,
        fs=fs,
        tone_ms=85.0,
        pause_ms=65.0,
        lead_silence_ms=220.0,
        tail_silence_ms=180.0,
        snr_db=22.0,
        seed=42,
    )
    wav_bytes = save_wav_bytes(fs, noisy)
    params = DenoiseParams(window_type="hann", nperseg=256, overlap_pct=75.0, alpha=2.2, beta=0.02)
    result = run_full_pipeline(wav_bytes, params, target_sequence=sequence, clean_reference=clean)
    metrics = result["metrics"]

    assert result["decoding"]["sequence"] == sequence
    assert metrics["snr_out_db"] is not None and metrics["snr_out_db"] >= 25.0
    assert metrics["snr_ref_db"] is not None and metrics["snr_ref_db"] >= 25.0
    assert metrics["pre_echo_db"] is not None and metrics["pre_echo_db"] <= -30.0
    assert metrics["residual_noise_db"] is not None and metrics["residual_noise_db"] <= 0.0
    assert metrics["musical_noise_db"] is not None and metrics["musical_noise_db"] < 6.0
    assert metrics["frame_latency_ms"] <= 40.0


def test_tradeoff_window_length_affects_pre_echo() -> None:
    sequence = "59#"
    fs = 8000
    _, noisy = synthesize_dtmf_signal(
        sequence=sequence,
        fs=fs,
        tone_ms=70.0,
        pause_ms=90.0,
        lead_silence_ms=250.0,
        tail_silence_ms=200.0,
        snr_db=18.0,
        seed=7,
    )
    wav_bytes = save_wav_bytes(fs, noisy)

    narrow = run_full_pipeline(
        wav_bytes,
        DenoiseParams(window_type="hann", nperseg=256, overlap_pct=75.0, alpha=2.0, beta=0.02),
    )
    wide = run_full_pipeline(
        wav_bytes,
        DenoiseParams(window_type="hann", nperseg=2048, overlap_pct=75.0, alpha=2.0, beta=0.02),
    )

    assert narrow["metrics"]["pre_echo_db"] is not None
    assert wide["metrics"]["pre_echo_db"] is not None
    assert wide["metrics"]["pre_echo_db"] > narrow["metrics"]["pre_echo_db"]


def test_tradeoff_aggressive_alpha_zero_floor_increases_musical_noise() -> None:
    sequence = "0812#"
    fs = 8000
    _, noisy = synthesize_dtmf_signal(
        sequence=sequence,
        fs=fs,
        tone_ms=80.0,
        pause_ms=65.0,
        lead_silence_ms=300.0,
        tail_silence_ms=250.0,
        snr_db=12.0,
        seed=99,
    )
    wav_bytes = save_wav_bytes(fs, noisy)

    balanced = run_full_pipeline(
        wav_bytes,
        DenoiseParams(window_type="hann", nperseg=512, overlap_pct=75.0, alpha=1.8, beta=0.05),
    )
    aggressive = run_full_pipeline(
        wav_bytes,
        DenoiseParams(window_type="hann", nperseg=512, overlap_pct=75.0, alpha=5.5, beta=0.0),
    )

    assert balanced["metrics"]["musical_noise_db"] is not None
    assert aggressive["metrics"]["musical_noise_db"] is not None
    assert aggressive["metrics"]["musical_noise_db"] > balanced["metrics"]["musical_noise_db"]


def test_wav_formats_stereo_uint8_and_validation() -> None:
    fs = 8000
    _, mono = synthesize_dtmf_signal("12", fs=fs, snr_db=20.0)
    stereo = np.stack([mono, mono * 0.5], axis=1)

    buf_stereo = io.BytesIO()
    wavfile.write(buf_stereo, fs, (np.clip(stereo, -1, 1) * 32767).astype(np.int16))
    fs_out, loaded = load_wav_bytes(buf_stereo.getvalue())
    assert fs_out == fs
    assert loaded.ndim == 1

    buf_u8 = io.BytesIO()
    u8 = np.clip((mono + 1.0) * 127.5, 0, 255).astype(np.uint8)
    wavfile.write(buf_u8, fs, u8)
    _, loaded_u8 = load_wav_bytes(buf_u8.getvalue())
    assert loaded_u8.min() >= -1.05 and loaded_u8.max() <= 1.05

    buf_low_fs = io.BytesIO()
    wavfile.write(buf_low_fs, 2000, np.zeros(200, dtype=np.int16))
    with pytest.raises(ValueError, match="Sample rate"):
        load_wav_bytes(buf_low_fs.getvalue())
