"""Flask HTTP API integration tests for Phase 1 & 2 endpoints."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(upload_dir=tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def test_index_and_health(client) -> None:
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert b"Aplikasi Web Denoising STFT" in resp_index.data
    assert b"Simulator Numpad Telepon" in resp_index.data

    resp_js = client.get("/static/app.js")
    assert resp_js.status_code == 200
    assert b"/api/process" in resp_js.data
    assert b"/api/generate-wav" in resp_js.data

    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200
    assert resp_health.get_json() == {"status": "ok"}


def test_sample_wav_and_full_process_and_reprocess(client) -> None:
    sample_resp = client.get("/api/sample-wav?sequence=0812345%23&snr_db=12.0")
    assert sample_resp.status_code == 200
    assert sample_resp.data.startswith(b"RIFF")

    # Initial upload & processing
    proc_resp = client.post(
        "/api/process",
        data={
            "audio": (io.BytesIO(sample_resp.data), "demo.wav"),
            "window_type": "hann",
            "nperseg": "512",
            "overlap_pct": "75",
            "alpha": "2.0",
            "beta": "0.02",
            "silent_mode": "auto",
            "target_sequence": "0812345#",
        },
        content_type="multipart/form-data",
    )
    assert proc_resp.status_code == 200
    payload = proc_resp.get_json()
    assert payload["decoding"]["sequence"] == "0812345#"
    assert payload["metrics"]["target_eval"]["exact_match"] is True
    assert "waveform" in payload["plots"]
    assert "removed" in payload["plots"]["waveform"]
    assert "fft" in payload["plots"]
    assert "spectrogram" in payload["plots"]
    assert "diff" in payload["plots"]["spectrogram"]

    # Fetch generated denoised WAV audio URL
    wav_resp = client.get(payload["denoised_audio_url"])
    assert wav_resp.status_code == 200
    assert wav_resp.data.startswith(b"RIFF")

    # Re-process existing file_id with tweaked parameters & manual silent range
    reproc_resp = client.post(
        "/api/process",
        data={
            "file_id": payload["file_id"],
            "window_type": "hamming",
            "nperseg": "256",
            "overlap_pct": "50",
            "alpha": "1.5",
            "beta": "0.03",
            "silent_mode": "manual",
            "silent_start_s": "0.02",
            "silent_end_s": "0.18",
        },
        content_type="multipart/form-data",
    )
    assert reproc_resp.status_code == 200
    assert reproc_resp.get_json()["decoding"]["sequence"] == "0812345#"


def test_validation_rejects_bad_files_and_paths(client) -> None:
    bad_ext = client.post(
        "/api/process",
        data={"audio": (io.BytesIO(b"not wav"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert bad_ext.status_code == 400

    bad_header = client.post(
        "/api/process",
        data={"audio": (io.BytesIO(b"FAKEHEADER"), "fake.wav")},
        content_type="multipart/form-data",
    )
    assert bad_header.status_code == 400

    bad_upload = client.get("/api/uploads/../../app.py")
    assert bad_upload.status_code == 404


def test_generate_wav_numpad_roundtrip_and_validation(client) -> None:
    # Noisy custom numpad synthesis at 16000 Hz
    gen_resp = client.post(
        "/api/generate-wav",
        json={
            "sequence": "0857*12#A",
            "fs": 16000,
            "tone_ms": 80,
            "pause_ms": 60,
            "add_noise": True,
            "snr_db": 10.0,
        },
    )
    assert gen_resp.status_code == 200
    assert gen_resp.data.startswith(b"RIFF")
    assert "dtmf_0857star12hashA.wav" in gen_resp.headers["Content-Disposition"]

    # Round-trip the generated WAV back into /api/process
    proc_resp = client.post(
        "/api/process",
        data={
            "audio": (io.BytesIO(gen_resp.data), "dtmf_0857star12hashA.wav"),
            "nperseg": "512",
            "alpha": "2.0",
            "beta": "0.02",
            "target_sequence": "0857*12#A",
        },
        content_type="multipart/form-data",
    )
    assert proc_resp.status_code == 200
    payload = proc_resp.get_json()
    assert payload["decoding"]["sequence"] == "0857*12#A"
    assert payload["metrics"]["target_eval"]["exact_match"] is True

    # Clean synthesis (add_noise=False)
    clean_resp = client.post(
        "/api/generate-wav",
        json={"sequence": "123", "fs": 8000, "add_noise": False},
    )
    assert clean_resp.status_code == 200
    assert clean_resp.data.startswith(b"RIFF")

    # Reject empty/invalid sequence
    bad_seq = client.post("/api/generate-wav", json={"sequence": "XYZ"})
    assert bad_seq.status_code == 400
