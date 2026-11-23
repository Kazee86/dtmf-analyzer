"""Flask web server for STFT Denoising & DTMF Extraction (Phase 1 & 2)."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for

from dsp_engine import (
    DenoiseParams,
    HIGH_FREQS,
    LOW_FREQS,
    run_full_pipeline,
    save_wav_bytes,
    synthesize_dtmf_signal,
)

BASE_DIR = Path(__file__).resolve().parent
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    UPLOAD_DIR = Path("/tmp/uploads")
else:
    UPLOAD_DIR = BASE_DIR / "static" / "uploads"

try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    UPLOAD_DIR = Path("/tmp/uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def create_app(upload_dir: Path | None = None) -> Flask:
    """Application factory for Flask app (enables isolated upload_dir in tests)."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["UPLOAD_DIR"] = upload_dir or UPLOAD_DIR
    try:
        Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        app.config["UPLOAD_DIR"] = Path("/tmp/uploads")
        Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    def _cleanup_uploads(max_age_seconds: int = 3600, max_files: int = 80) -> None:
        target_dir = Path(app.config["UPLOAD_DIR"])
        try:
            wav_files = [p for p in target_dir.glob("*.wav") if p.is_file()]
            now = time.time()
            for p in wav_files:
                if now - p.stat().st_mtime > max_age_seconds:
                    p.unlink(missing_ok=True)
            wav_files = sorted(
                [p for p in target_dir.glob("*.wav") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
            )
            if len(wav_files) > max_files:
                for p in wav_files[: len(wav_files) - max_files]:
                    p.unlink(missing_ok=True)
        except OSError:
            pass

    def _parse_params(form: Any) -> DenoiseParams:
        def _float(name: str, default: float) -> float:
            val = form.get(name, "").strip()
            return float(val) if val else default

        def _int(name: str, default: int) -> int:
            val = form.get(name, "").strip()
            return int(float(val)) if val else default

        nfft_raw = form.get("nfft", "").strip()
        return DenoiseParams(
            window_type=form.get("window_type", "hann"),
            nperseg=_int("nperseg", 512),
            overlap_pct=_float("overlap_pct", 75.0),
            nfft=int(float(nfft_raw)) if nfft_raw else None,
            alpha=_float("alpha", 2.0),
            beta=_float("beta", 0.02),
            silent_mode=form.get("silent_mode", "auto"),
            silent_ms=_float("silent_ms", 150.0),
            silent_start_s=_float("silent_start_s", 0.0),
            silent_end_s=_float("silent_end_s", 0.2),
        )

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            low_freqs=LOW_FREQS,
            high_freqs=HIGH_FREQS,
        )

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/uploads/<path:filename>")
    def serve_upload(filename: str) -> Response:
        if not re.match(r"^[a-f0-9]{32}_(noisy|denoised)\.wav$", filename):
            return jsonify({"error": "Nama berkas tidak valid."}), 404
        resp = send_from_directory(app.config["UPLOAD_DIR"], filename, mimetype="audio/wav")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _build_dtmf_wav_response(
        sequence: str,
        fs: int,
        tone_ms: float,
        pause_ms: float,
        snr_db: float | None,
        seed: int | None,
        filename: str,
    ) -> tuple[Response, int] | Response:
        cleaned = "".join(ch for ch in sequence.upper() if ch in "0123456789*#ABCD")
        if not cleaned:
            return jsonify({"error": "Masukkan minimal 1 tombol DTMF valid (0–9, *, #, A–D)."}), 400
        if len(cleaned) > 40:
            return jsonify({"error": "Panjang sekuens maksimal 40 tombol."}), 400
        if fs < 4000 or fs > 48000:
            return jsonify({"error": "Sample rate harus di antara 4000 Hz hingga 48000 Hz."}), 400
        if tone_ms < 20.0 or tone_ms > 1000.0:
            return jsonify({"error": "Durasi nada harus antara 20 ms hingga 1000 ms."}), 400
        if pause_ms < 10.0 or pause_ms > 1000.0:
            return jsonify({"error": "Durasi jeda harus antara 10 ms hingga 1000 ms."}), 400
        if snr_db is not None:
            snr_db = max(-5.0, min(40.0, float(snr_db)))

        try:
            _, audio_out = synthesize_dtmf_signal(
                sequence=cleaned,
                fs=fs,
                tone_ms=tone_ms,
                pause_ms=pause_ms,
                lead_silence_ms=220.0,
                tail_silence_ms=180.0,
                snr_db=snr_db,
                seed=seed,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        wav_bytes = save_wav_bytes(fs, audio_out)
        return Response(
            wav_bytes,
            mimetype="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/sample-wav")
    def sample_wav() -> tuple[Response, int] | Response:
        """Generate a synthetic noisy DTMF WAV for instant testing in Phase 1-2."""
        seq = request.args.get("sequence", "0812345#")
        try:
            snr_db = float(request.args.get("snr_db", "12.0"))
        except ValueError:
            return jsonify({"error": "Parameter snr_db harus berupa angka."}), 400
        return _build_dtmf_wav_response(
            sequence=seq,
            fs=8000,
            tone_ms=90.0,
            pause_ms=65.0,
            snr_db=snr_db,
            seed=42,
            filename="sample_dtmf_noisy.wav",
        )

    @app.post("/api/generate-wav")
    def generate_wav() -> tuple[Response, int] | Response:
        """Synthesize a custom DTMF WAV file from interactive numpad input."""
        payload = request.get_json(silent=True) or request.form
        seq = str(payload.get("sequence", "")).strip()
        try:
            fs = int(float(payload.get("fs", 8000)))
            tone_ms = float(payload.get("tone_ms", 90.0))
            pause_ms = float(payload.get("pause_ms", 65.0))
            add_noise_raw = payload.get("add_noise", True)
            if isinstance(add_noise_raw, str):
                add_noise = add_noise_raw.lower() in ("true", "1", "yes", "on")
            else:
                add_noise = bool(add_noise_raw)
            snr_db = float(payload.get("snr_db", 10.0)) if add_noise else None
        except (TypeError, ValueError):
            return jsonify({"error": "Parameter generator WAV harus berupa angka yang valid."}), 400

        safe_slug = re.sub(r"[^0-9A-Za-z]+", "", seq.replace("*", "star").replace("#", "hash"))[:20]
        out_name = f"dtmf_{safe_slug or 'tone'}.wav"
        return _build_dtmf_wav_response(
            sequence=seq,
            fs=fs,
            tone_ms=tone_ms,
            pause_ms=pause_ms,
            snr_db=snr_db,
            seed=None,
            filename=out_name,
        )

    @app.post("/api/process")
    def process_audio() -> tuple[Response, int] | Response:
        _cleanup_uploads()
        upload_dir_path = Path(app.config["UPLOAD_DIR"])

        try:
            params = _parse_params(request.form)
        except ValueError as exc:
            return jsonify({"error": f"Parameter tidak valid: {exc}"}), 400

        target_sequence = request.form.get("target_sequence") or None
        file_id = request.form.get("file_id", "").strip()
        uploaded = request.files.get("audio")

        noisy_bytes: bytes | None = None
        original_filename = "audio.wav"

        if uploaded and uploaded.filename:
            if not uploaded.filename.lower().endswith(".wav"):
                return jsonify({"error": "Format berkas harus .wav."}), 400
            noisy_bytes = uploaded.read()
            if not noisy_bytes.startswith(b"RIFF") or b"WAVE" not in noisy_bytes[:16]:
                return jsonify({"error": "Header berkas WAV tidak valid (bukan RIFF/WAVE)."}), 400
            file_id = uuid.uuid4().hex
            original_filename = Path(uploaded.filename).name
            (upload_dir_path / f"{file_id}_noisy.wav").write_bytes(noisy_bytes)
        elif file_id and FILE_ID_PATTERN.match(file_id):
            noisy_path = upload_dir_path / f"{file_id}_noisy.wav"
            if not noisy_path.is_file():
                return jsonify({"error": "Berkas audio sebelumnya tidak ditemukan, silakan unggah ulang."}), 404
            noisy_bytes = noisy_path.read_bytes()
            original_filename = request.form.get("filename", "audio.wav")
        else:
            return jsonify({"error": "Harap unggah berkas audio (.wav) terlebih dahulu."}), 400

        try:
            result = run_full_pipeline(
                wav_bytes=noisy_bytes,
                params=params,
                target_sequence=target_sequence,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Terjadi kesalahan pemrosesan DSP: {exc}"}), 500

        denoised_name = f"{file_id}_denoised.wav"
        noisy_name = f"{file_id}_noisy.wav"
        (upload_dir_path / denoised_name).write_bytes(result.pop("denoised_wav_bytes"))

        result["file_id"] = file_id
        result["filename"] = original_filename
        result["noisy_audio_url"] = url_for("serve_upload", filename=noisy_name)
        result["denoised_audio_url"] = url_for("serve_upload", filename=denoised_name)
        return jsonify(result)

    @app.errorhandler(413)
    def file_too_large(_err: Any) -> tuple[Response, int]:
        return jsonify({"error": "Ukuran berkas melebihi batas maksimal 15 MB."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
