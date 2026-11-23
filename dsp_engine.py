"""Core DSP engine for STFT spectral subtraction denoising and DTMF extraction."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

LOW_FREQS: tuple[float, ...] = (697.0, 770.0, 852.0, 941.0)
HIGH_FREQS: tuple[float, ...] = (1209.0, 1336.0, 1477.0, 1633.0)
ALL_DTMF_FREQS: tuple[float, ...] = LOW_FREQS + HIGH_FREQS

DTMF_TABLE: dict[tuple[float, float], str] = {
    (697.0, 1209.0): "1",
    (697.0, 1336.0): "2",
    (697.0, 1477.0): "3",
    (697.0, 1633.0): "A",
    (770.0, 1209.0): "4",
    (770.0, 1336.0): "5",
    (770.0, 1477.0): "6",
    (770.0, 1633.0): "B",
    (852.0, 1209.0): "7",
    (852.0, 1336.0): "8",
    (852.0, 1477.0): "9",
    (852.0, 1633.0): "C",
    (941.0, 1209.0): "*",
    (941.0, 1336.0): "0",
    (941.0, 1477.0): "#",
    (941.0, 1633.0): "D",
}

VALID_WINDOWS: tuple[str, ...] = ("hann", "hamming", "blackman", "boxcar")
VALID_DTMF_CHARS: set[str] = set(DTMF_TABLE.values())
SYMBOL_TO_FREQS: dict[str, tuple[float, float]] = {sym: freqs for freqs, sym in DTMF_TABLE.items()}
EPS: float = 1e-18


@dataclass(frozen=True)
class DenoiseParams:
    """Validated parameters for STFT spectral subtraction."""

    window_type: str = "hann"
    nperseg: int = 512
    overlap_pct: float = 75.0
    nfft: int | None = None
    alpha: float = 2.0
    beta: float = 0.02
    silent_mode: str = "auto"
    silent_ms: float = 150.0
    silent_start_s: float = 0.0
    silent_end_s: float = 0.2


def validate_params(params: DenoiseParams) -> DenoiseParams:
    """Validate and normalize STFT denoising parameters."""
    window_type = params.window_type.lower().strip()
    if window_type not in VALID_WINDOWS:
        raise ValueError(
            f"Tipe window '{params.window_type}' tidak valid. Pilih: {', '.join(VALID_WINDOWS)}."
        )

    nperseg = int(params.nperseg)
    if nperseg < 64 or nperseg > 8192:
        raise ValueError("Panjang window (nperseg) harus berada di antara 64 hingga 8192 sampel.")

    overlap_pct = float(params.overlap_pct)
    if overlap_pct < 0.0 or overlap_pct >= 95.0:
        raise ValueError("Overlap harus berada di antara 0% hingga < 95%.")

    nfft = int(params.nfft) if params.nfft is not None else nperseg
    if nfft < nperseg:
        nfft = nperseg

    alpha = float(params.alpha)
    if alpha < 0.0 or alpha > 10.0:
        raise ValueError("Faktor oversubtraction (alpha) harus berada di antara 0.0 hingga 10.0.")

    beta = float(params.beta)
    if beta < 0.0 or beta > 0.5:
        raise ValueError("Spectral floor (beta) harus berada di antara 0.0 hingga 0.5.")

    silent_mode = params.silent_mode.lower().strip()
    if silent_mode not in ("auto", "initial", "manual"):
        raise ValueError("Mode estimasi segmen senyap harus 'auto', 'initial', atau 'manual'.")

    silent_ms = float(params.silent_ms)
    if silent_ms < 20.0 or silent_ms > 5000.0:
        raise ValueError("Durasi segmen senyap awal harus antara 20 ms hingga 5000 ms.")

    silent_start_s = float(params.silent_start_s)
    silent_end_s = float(params.silent_end_s)
    if silent_mode == "manual" and (silent_start_s < 0.0 or silent_end_s <= silent_start_s):
        raise ValueError("Rentang waktu segmen derau manual tidak valid (akhir harus > awal >= 0).")

    return DenoiseParams(
        window_type=window_type,
        nperseg=nperseg,
        overlap_pct=overlap_pct,
        nfft=nfft,
        alpha=alpha,
        beta=beta,
        silent_mode=silent_mode,
        silent_ms=silent_ms,
        silent_start_s=silent_start_s,
        silent_end_s=silent_end_s,
    )


def _normalize_wav_samples(fs: int, raw: np.ndarray) -> tuple[int, np.ndarray]:
    """Normalize raw WAV PCM/float samples to mono float64 in [-1.0, 1.0]."""
    if fs < 4000:
        raise ValueError(
            f"Sample rate {fs} Hz terlalu rendah untuk DTMF (minimal 4000 Hz)."
        )
    if raw.size == 0:
        raise ValueError("Berkas audio WAV kosong.")

    if raw.ndim > 1:
        raw = raw.astype(np.float64).mean(axis=1)

    if np.issubdtype(raw.dtype, np.unsignedinteger):
        # 8-bit PCM WAV is unsigned [0, 255] centered at 128
        samples = (raw.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.signedinteger):
        scale = float(1 << (raw.dtype.itemsize * 8 - 1))
        samples = raw.astype(np.float64) / scale
    elif np.issubdtype(raw.dtype, np.floating):
        samples = raw.astype(np.float64)
        if not np.all(np.isfinite(samples)):
            raise ValueError("Berkas audio mengandung nilai NaN atau tak hingga.")
    else:
        raise ValueError(f"Format sampel WAV tidak didukung: {raw.dtype}")

    if samples.size < 64:
        raise ValueError("Durasi audio terlalu pendek (minimal 64 sampel).")

    return int(fs), samples


def load_wav_bytes(data: bytes) -> tuple[int, np.ndarray]:
    """Read WAV bytes from memory and return (sample_rate, mono_float64_samples)."""
    try:
        fs, raw = wavfile.read(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Gagal membaca berkas WAV: {exc}") from exc
    return _normalize_wav_samples(int(fs), raw)


def load_wav_file(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV file from disk and return (sample_rate, mono_float64_samples)."""
    try:
        fs, raw = wavfile.read(str(path))
    except Exception as exc:
        raise ValueError(f"Gagal membaca berkas WAV '{path}': {exc}") from exc
    return _normalize_wav_samples(int(fs), raw)


def save_wav_bytes(fs: int, samples: np.ndarray) -> bytes:
    """Serialize float64 audio samples to 16-bit PCM WAV bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, int(fs), pcm)
    return buf.getvalue()


def _estimate_noise_psd(
    mag_sq: np.ndarray,
    times: np.ndarray,
    silent_mode: str,
    silent_ms: float,
    silent_start_s: float = 0.0,
    silent_end_s: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate noise power spectral density from silent STFT frames."""
    n_frames = mag_sq.shape[1]
    frame_energy = mag_sq.sum(axis=0)

    if silent_mode == "manual":
        silent_mask = (times >= silent_start_s) & (times <= silent_end_s)
        if not np.any(silent_mask):
            mid = 0.5 * (silent_start_s + silent_end_s)
            nearest = int(np.argmin(np.abs(times - mid)))
            silent_mask = np.zeros(n_frames, dtype=bool)
            silent_mask[nearest] = True
    elif silent_mode == "initial":
        silent_mask = times <= (silent_ms / 1000.0)
        if not np.any(silent_mask):
            silent_mask = np.zeros(n_frames, dtype=bool)
            silent_mask[0] = True
    else:
        max_e = float(np.max(frame_energy)) if frame_energy.size > 0 else 0.0
        non_zero = frame_energy > (1e-10 * max_e)
        pool = frame_energy[non_zero] if np.sum(non_zero) >= 4 else frame_energy
        k = max(2, min(pool.size, int(np.ceil(0.15 * pool.size))))
        threshold = float(np.partition(pool, k - 1)[k - 1])
        silent_mask = (frame_energy <= threshold) & (non_zero if np.sum(non_zero) >= 4 else True)

    noise_psd = np.maximum(mag_sq[:, silent_mask].mean(axis=1, keepdims=True), EPS)
    return noise_psd, silent_mask


def stft_denoise(
    samples: np.ndarray,
    fs: int,
    params: DenoiseParams,
) -> dict[str, Any]:
    """Apply STFT power spectral subtraction with spectral floor and ISTFT reconstruction."""
    cfg = validate_params(params)
    warnings: list[str] = []

    eff_nperseg = min(cfg.nperseg, samples.size)
    if eff_nperseg < cfg.nperseg:
        warnings.append(
            f"Panjang audio ({samples.size} sampel) lebih pendek dari window ({cfg.nperseg}); "
            f"disesuaikan menjadi {eff_nperseg} sampel."
        )

    eff_nfft = max(cfg.nfft or eff_nperseg, eff_nperseg)
    hop = max(1, int(round(eff_nperseg * (1.0 - cfg.overlap_pct / 100.0))))
    noverlap = eff_nperseg - hop

    win_arr = signal.get_window(cfg.window_type, eff_nperseg)
    is_cola = bool(signal.check_COLA(win_arr, eff_nperseg, noverlap))
    if not is_cola:
        warnings.append(
            f"Kombinasi window '{cfg.window_type}' dan overlap {cfg.overlap_pct:.0f}% "
            "tidak memenuhi kondisi COLA sempurna; SciPy menormalisasi ulang amplitudo saat ISTFT."
        )

    freqs, times, zxx_noisy = signal.stft(
        samples,
        fs=fs,
        window=cfg.window_type,
        nperseg=eff_nperseg,
        noverlap=noverlap,
        nfft=eff_nfft,
    )

    mag_sq = np.abs(zxx_noisy) ** 2
    noise_psd, silent_frame_mask = _estimate_noise_psd(
        mag_sq,
        times,
        cfg.silent_mode,
        cfg.silent_ms,
        cfg.silent_start_s,
        cfg.silent_end_s,
    )

    frame_energies = mag_sq.sum(axis=0)
    peak_frame_e = float(np.max(frame_energies)) + EPS
    noise_frame_e = float(noise_psd.sum())
    noise_rel_db = 10.0 * np.log10((noise_frame_e + EPS) / peak_frame_e)
    if noise_rel_db < -60.0:
        warnings.append(
            f"Lantai derau terestimasi sangat rendah ({noise_rel_db:.1f} dB di bawah puncak); "
            "berkas audio kemungkinan sudah bersih atau memiliki jeda senyap dalam sehingga perubahan visual STFT sangat halus. "
            "Coba pilih mode segmen 'Manual' pada bagian yang berderau atau uji dengan contoh WAV berderau."
        )

    clean_mag_sq = np.maximum(
        mag_sq - cfg.alpha * noise_psd,
        cfg.beta * noise_psd,
    )
    zxx_clean = np.sqrt(clean_mag_sq) * np.exp(1j * np.angle(zxx_noisy))

    _, denoised = signal.istft(
        zxx_clean,
        fs=fs,
        window=cfg.window_type,
        nperseg=eff_nperseg,
        noverlap=noverlap,
        nfft=eff_nfft,
    )

    if denoised.size >= samples.size:
        denoised = denoised[: samples.size]
    else:
        denoised = np.pad(denoised, (0, samples.size - denoised.size))

    return {
        "freqs": freqs,
        "times": times,
        "zxx_noisy": zxx_noisy,
        "zxx_clean": zxx_clean,
        "noise_psd": noise_psd,
        "silent_frame_mask": silent_frame_mask,
        "denoised": denoised,
        "eff_nperseg": eff_nperseg,
        "noverlap": noverlap,
        "hop": hop,
        "eff_nfft": eff_nfft,
        "is_cola": is_cola,
        "warnings": warnings,
    }


def _vectorized_goertzel(
    frames: np.ndarray,
    fs: int,
    target_freqs: tuple[float, ...],
) -> np.ndarray:
    """Compute Goertzel magnitude-squared across all frames simultaneously.

    frames shape: (num_frames, frame_len)
    returns powers shape: (len(target_freqs), num_frames)
    """
    n_freqs = len(target_freqs)
    n_frames, frame_len = frames.shape

    win = np.hanning(frame_len)
    w_frames = frames * win[None, :]
    omegas = 2.0 * np.pi * np.asarray(target_freqs, dtype=np.float64) / float(fs)
    coeffs = (2.0 * np.cos(omegas))[:, None]

    s1 = np.zeros((n_freqs, n_frames), dtype=np.float64)
    s2 = np.zeros((n_freqs, n_frames), dtype=np.float64)

    for idx in range(frame_len):
        s0 = w_frames[:, idx][None, :] + coeffs * s1 - s2
        s2 = s1
        s1 = s0

    powers = s1 * s1 + s2 * s2 - coeffs * s1 * s2
    sum_w_sq = float(np.sum(win) ** 2) + EPS
    # Normalized single-sided tone power matching time-domain mean-square power
    return (2.0 * powers) / sum_w_sq


def _single_segment_goertzel(
    segment: np.ndarray,
    fs: int,
) -> tuple[float, float, float, float, str, float]:
    """Measure exact peak DTMF frequencies and confidence on an isolated tone burst."""
    n = segment.size
    win = np.hanning(n)
    w_seg = segment * win

    # Refine low/high peak frequencies using zero-padded FFT inside +-4% tolerance
    nfft_refine = max(4096, 1 << int(np.ceil(np.log2(max(n, 1) * 4))))
    spec = np.abs(np.fft.rfft(w_seg, n=nfft_refine))
    fft_freqs = np.fft.rfftfreq(nfft_refine, d=1.0 / fs)

    powers = _vectorized_goertzel(segment[None, :], fs, ALL_DTMF_FREQS)[:, 0]
    low_powers = powers[:4]
    high_powers = powers[4:]

    low_idx = int(np.argmax(low_powers))
    high_idx = int(np.argmax(high_powers))
    nom_low = LOW_FREQS[low_idx]
    nom_high = HIGH_FREQS[high_idx]

    def _peak_near(target: float) -> float:
        band = (fft_freqs >= target * 0.96) & (fft_freqs <= target * 1.04)
        if not np.any(band):
            return target
        band_freqs = fft_freqs[band]
        band_spec = spec[band]
        return float(band_freqs[int(np.argmax(band_spec))])

    meas_low = _peak_near(nom_low)
    meas_high = _peak_near(nom_high)

    # Confidence based on separation against second-best peaks in low and high groups
    low_sorted = np.sort(low_powers)
    high_sorted = np.sort(high_powers)
    low_margin = low_sorted[-1] / (low_sorted[-2] + EPS)
    high_margin = high_sorted[-1] / (high_sorted[-2] + EPS)
    confidence = float(min(1.0, (np.log10(min(low_margin, high_margin) + 1.0)) / 1.5))

    symbol = DTMF_TABLE[(nom_low, nom_high)]
    return meas_low, meas_high, nom_low, nom_high, symbol, confidence


def decode_dtmf(samples: np.ndarray, fs: int) -> dict[str, Any]:
    """Segment active bursts and decode DTMF symbols using vectorized Goertzel."""
    # ponytail: 20ms window with 5ms hop for fine temporal separation on fast dialing / short pauses
    frame_len = max(64, int(round(0.020 * fs)))
    hop_len = max(16, int(round(0.005 * fs)))

    if samples.size < frame_len:
        frame_len = samples.size
        hop_len = max(1, frame_len // 2)

    n_frames = 1 + max(0, (samples.size - frame_len) // hop_len)
    starts = np.arange(n_frames) * hop_len
    idx_matrix = starts[:, None] + np.arange(frame_len)[None, :]
    frames = samples[idx_matrix]

    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
    global_max_rms = float(np.max(frame_rms))
    noise_floor_rms = float(np.percentile(frame_rms, 10)) if n_frames >= 4 else float(np.min(frame_rms))
    # Cap threshold ceiling to prevent suppressing valid quiet tones in short-pause recordings
    rms_thresh = max(0.02 * global_max_rms, min(0.35 * global_max_rms, 1.8 * noise_floor_rms), 1e-4)

    powers = _vectorized_goertzel(frames, fs, ALL_DTMF_FREQS)
    low_powers = powers[:4, :]
    high_powers = powers[4:, :]

    frame_symbols: list[str | None] = [None] * n_frames
    frame_mean_power = np.mean(frames * frames, axis=1) + EPS

    for t in range(n_frames):
        if frame_rms[t] < rms_thresh:
            continue

        lp = low_powers[:, t]
        hp = high_powers[:, t]
        li = int(np.argmax(lp))
        hi = int(np.argmax(hp))

        l_sorted = np.sort(lp)
        h_sorted = np.sort(hp)
        l_dom = l_sorted[-1] / (l_sorted[-2] + EPS)
        h_dom = h_sorted[-1] / (h_sorted[-2] + EPS)
        if l_dom < 2.5 or h_dom < 2.5:
            continue

        # Twist check: high vs low tone power within 10 dB
        twist_ratio = hp[hi] / (lp[li] + EPS)
        if twist_ratio < 0.1 or twist_ratio > 10.0:
            continue

        # Energy concentration in the winning pair vs frame total power
        tone_ratio = (lp[li] + hp[hi]) / frame_mean_power[t]
        if tone_ratio < 0.25:
            continue

        frame_symbols[t] = DTMF_TABLE[(LOW_FREQS[li], HIGH_FREQS[hi])]

    # Group contiguous identical symbols without bridging to preserve tight inter-digit pauses
    tones: list[dict[str, Any]] = []
    min_frames = 3  # ~25-30 ms minimum burst span (covers >= 30 ms DTMF bursts reliably)
    cur_sym: str | None = None
    seg_start = 0

    def _flush_segment(sym: str, f_start: int, f_end: int) -> None:
        if (f_end - f_start + 1) < min_frames:
            return
        s_idx = int(starts[f_start])
        e_idx = int(min(samples.size, starts[f_end] + frame_len))
        segment = samples[s_idx:e_idx]
        if segment.size < 32:
            return
        meas_low, meas_high, nom_low, nom_high, refined_sym, conf = _single_segment_goertzel(
            segment, fs
        )
        # Validate measured peaks stay within +-3.5% of nominal DTMF frequencies
        if (
            abs(meas_low - nom_low) / nom_low > 0.035
            or abs(meas_high - nom_high) / nom_high > 0.035
        ):
            return
        tones.append(
            {
                "index": len(tones) + 1,
                "symbol": refined_sym,
                "start_time": round(s_idx / fs, 4),
                "end_time": round(e_idx / fs, 4),
                "duration_ms": round(((e_idx - s_idx) / fs) * 1000.0, 1),
                "f_low": round(meas_low, 2),
                "f_high": round(meas_high, 2),
                "nominal_low": int(nom_low),
                "nominal_high": int(nom_high),
                "confidence": round(conf * 100.0, 1),
            }
        )

    for idx, sym in enumerate(frame_symbols):
        if sym == cur_sym:
            continue
        if cur_sym is not None:
            _flush_segment(cur_sym, seg_start, idx - 1)
        cur_sym = sym
        seg_start = idx
    if cur_sym is not None:
        _flush_segment(cur_sym, seg_start, n_frames - 1)

    sequence = "".join(t["symbol"] for t in tones)
    return {"sequence": sequence, "tones": tones}


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two symbol strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def evaluate_target_sequence(detected: str, target: str | None) -> dict[str, Any] | None:
    """Compare detected DTMF sequence against a user-provided reference sequence."""
    if target is None:
        return None
    cleaned_target = "".join(ch for ch in target.upper().strip() if ch in VALID_DTMF_CHARS)
    if not cleaned_target:
        return None

    dist = _levenshtein(detected, cleaned_target)
    denom = max(len(detected), len(cleaned_target), 1)
    accuracy = max(0.0, (1.0 - dist / denom) * 100.0)
    return {
        "target": cleaned_target,
        "detected": detected,
        "exact_match": detected == cleaned_target,
        "edit_distance": dist,
        "accuracy_pct": round(accuracy, 1),
    }


def _build_activity_masks(
    n_samples: int,
    fs: int,
    tones: list[dict[str, Any]],
    noisy_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct boolean sample masks for active tone regions and pure silent regions."""
    active_mask = np.zeros(n_samples, dtype=bool)
    guard_mask = np.zeros(n_samples, dtype=bool)
    guard_samples = int(0.020 * fs)

    if tones:
        for tone in tones:
            s_i = max(0, int(round(tone["start_time"] * fs)))
            e_i = min(n_samples, int(round(tone["end_time"] * fs)))
            active_mask[s_i:e_i] = True
            g_s = max(0, s_i - guard_samples)
            g_e = min(n_samples, e_i + guard_samples)
            guard_mask[g_s:g_e] = True
    else:
        win = max(64, int(0.020 * fs))
        sq = noisy_samples * noisy_samples
        kernel = np.ones(win, dtype=np.float64) / win
        env = np.convolve(sq, kernel, mode="same")
        thresh = float(np.percentile(env, 60))
        active_mask = env > thresh
        guard_mask = active_mask

    silent_mask = ~guard_mask
    return active_mask, silent_mask


def compute_quality_metrics(
    noisy: np.ndarray,
    denoised: np.ndarray,
    fs: int,
    stft_out: dict[str, Any],
    tones: list[dict[str, Any]],
    total_latency_ms: float,
    target_sequence: str | None = None,
    clean_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute blind and optional reference-based Scenario P4 quality metrics."""
    active_mask, silent_mask = _build_activity_masks(noisy.size, fs, tones, noisy)

    def _blind_snr(sig: np.ndarray) -> float | None:
        if not np.any(active_mask) or np.sum(silent_mask) < max(32, int(0.01 * fs)):
            return None
        p_active = float(np.mean(sig[active_mask] ** 2))
        p_noise = float(np.mean(sig[silent_mask] ** 2))
        if p_noise <= EPS:
            return 60.0
        p_signal = max(p_active - p_noise, EPS)
        return float(np.clip(10.0 * np.log10(p_signal / p_noise), -20.0, 60.0))

    snr_in = _blind_snr(noisy)
    snr_out = _blind_snr(denoised)
    snr_gain = (snr_out - snr_in) if (snr_out is not None and snr_in is not None) else None

    snr_ref = None
    if clean_reference is not None and clean_reference.size == denoised.size:
        err = denoised - clean_reference
        p_ref = float(np.sum(clean_reference ** 2))
        p_err = float(np.sum(err ** 2))
        snr_ref = round(float(10.0 * np.log10((p_ref + EPS) / (p_err + EPS))), 2)

    # Pre-Echo (dB): excess energy in 20 ms pre-onset windows relative to active tone power
    pre_echo_db: float | None = None
    if tones and np.any(active_mask):
        p_active_out = float(np.mean(denoised[active_mask] ** 2)) + EPS
        p_silent_out = float(np.mean(denoised[silent_mask] ** 2)) if np.any(silent_mask) else 0.0
        guard_len = max(16, int(0.020 * fs))
        pre_powers: list[float] = []
        for tone in tones:
            onset = int(round(tone["start_time"] * fs))
            if onset >= guard_len:
                win_pre = denoised[onset - guard_len : onset]
                pre_powers.append(float(np.mean(win_pre ** 2)))
        if pre_powers:
            excess_pre = max(float(np.mean(pre_powers)) - p_silent_out, EPS)
            pre_echo_db = round(float(10.0 * np.log10(excess_pre / p_active_out)), 2)

    # Musical Noise (dB): isolated spectral peak spikiness ratio (CV = std/mean)
    # measured on pure silent STFT frames away from tone boundaries
    stft_times = stft_out["times"]
    half_win_sec = (stft_out["eff_nperseg"] / float(fs)) * 0.5
    pure_silent_frames = np.zeros(stft_times.size, dtype=bool)
    for fi, tc in enumerate(stft_times):
        s_i = max(0, int(round((tc - half_win_sec) * fs)))
        e_i = min(noisy.size, int(round((tc + half_win_sec) * fs)))
        if e_i > s_i and np.all(silent_mask[s_i:e_i]):
            pure_silent_frames[fi] = True

    silent_frames = pure_silent_frames if np.any(pure_silent_frames) else stft_out["silent_frame_mask"]
    mag_sq_in = np.abs(stft_out["zxx_noisy"]) ** 2
    mag_sq_out = np.abs(stft_out["zxx_clean"]) ** 2

    def _spectral_cv(mag_sq_silent: np.ndarray) -> float:
        mean_val = float(np.mean(mag_sq_silent))
        if mean_val <= EPS:
            return 0.0
        return float(np.std(mag_sq_silent)) / mean_val

    musical_noise_db: float | None = None
    residual_noise_db: float | None = None
    if np.any(silent_frames):
        m_in_silent = mag_sq_in[:, silent_frames]
        m_out_silent = mag_sq_out[:, silent_frames]
        cv_in = _spectral_cv(m_in_silent)
        cv_out = _spectral_cv(m_out_silent)
        if cv_in > EPS and cv_out > 0.0:
            musical_noise_db = round(float(10.0 * np.log10(cv_out / cv_in)), 2)
        elif cv_out == 0.0:
            musical_noise_db = 0.0
        p_in_silent = float(np.mean(m_in_silent))
        p_out_silent = float(np.mean(m_out_silent))
        if p_in_silent > EPS:
            residual_noise_db = round(float(10.0 * np.log10((p_out_silent + EPS) / p_in_silent)), 2)

    n_frames = max(1, stft_out["zxx_noisy"].shape[1])
    frame_latency_ms = total_latency_ms / n_frames

    # Qualitative diagnostic status for UI cards
    snr_val = snr_ref if snr_ref is not None else (snr_out if snr_out is not None else 0.0)
    if snr_val >= 25.0 and (musical_noise_db is None or musical_noise_db <= 5.5):
        quality_badge = "optimal"
    elif snr_val >= 15.0:
        quality_badge = "moderate"
    else:
        quality_badge = "poor"

    return {
        "snr_in_db": round(snr_in, 2) if snr_in is not None else None,
        "snr_out_db": round(snr_out, 2) if snr_out is not None else None,
        "snr_gain_db": round(snr_gain, 2) if snr_gain is not None else None,
        "snr_ref_db": snr_ref,
        "pre_echo_db": pre_echo_db,
        "musical_noise_db": musical_noise_db,
        "residual_noise_db": residual_noise_db,
        "frame_latency_ms": round(frame_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 2),
        "n_stft_frames": n_frames,
        "quality_badge": quality_badge,
        "target_eval": evaluate_target_sequence(
            "".join(t["symbol"] for t in tones), target_sequence
        ),
    }


def _downsample_waveform(samples: np.ndarray, fs: int, max_points: int = 2000) -> tuple[list[float], list[float]]:
    """Downsample waveform using min-max buckets so bursts and envelopes remain crisp."""
    n = samples.size
    if n <= max_points:
        times = np.arange(n, dtype=np.float64) / fs
        return np.round(times, 5).tolist(), np.round(samples, 5).tolist()

    n_buckets = max_points // 2
    edges = np.linspace(0, n, n_buckets + 1, dtype=int)
    t_out: list[float] = []
    y_out: list[float] = []

    for i in range(n_buckets):
        start, end = int(edges[i]), int(edges[i + 1])
        if end <= start:
            continue
        chunk = samples[start:end]
        idx_min = int(np.argmin(chunk))
        idx_max = int(np.argmax(chunk))
        if idx_min <= idx_max:
            pair = (start + idx_min, start + idx_max)
        else:
            pair = (start + idx_max, start + idx_min)
        for idx in pair:
            t_out.append(round(idx / fs, 5))
            y_out.append(round(float(samples[idx]), 5))
        if len(t_out) >= max_points:
            break

    return t_out, y_out


def _compute_fft_spectrum(
    samples: np.ndarray,
    fs: int,
    max_freq: float = 4000.0,
) -> tuple[list[float], list[float]]:
    """Compute Welch-averaged STFT magnitude spectrum (dB) over diagnostic frequency band."""
    nperseg_spec = min(samples.size, 2048)
    noverlap_spec = nperseg_spec // 2
    freqs, _, zxx_spec = signal.stft(
        samples,
        fs=fs,
        window="hann",
        nperseg=nperseg_spec,
        noverlap=noverlap_spec,
    )
    mag_mean = np.sqrt(np.mean(np.abs(zxx_spec) ** 2, axis=1))
    band = freqs <= min(max_freq, fs / 2.0)
    freqs = freqs[band]
    mag_db = 20.0 * np.log10(mag_mean[band] + 1e-9)
    return np.round(freqs, 2).tolist(), np.round(mag_db, 2).tolist()


def _block_pool_2d(
    arr: np.ndarray,
    axis: int,
    target_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an array along axis using block averaging so long recordings keep all bursts."""
    size = arr.shape[axis]
    edges = np.linspace(0, size, target_bins + 1, dtype=int)
    slices = []
    centers = []
    for i in range(target_bins):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            continue
        if axis == 0:
            slices.append(np.mean(arr[s:e, :], axis=0))
        else:
            slices.append(np.mean(arr[:, s:e], axis=1))
        centers.append((s, e))
    pooled = np.stack(slices, axis=axis)
    return pooled, np.asarray(centers)


def _downsample_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    zxx: np.ndarray,
    max_freq: float = 4000.0,
    max_time_bins: int = 260,
    max_freq_bins: int = 200,
) -> dict[str, Any]:
    """Prepare 2D STFT magnitude spectrogram (dB) with block-mean pooling for long files."""
    f_mask = freqs <= min(max_freq, float(freqs[-1]))
    f_crop = freqs[f_mask]
    mag_pow = np.abs(zxx[f_mask, :]) ** 2
    t_crop = times

    if f_crop.size > max_freq_bins:
        mag_pow, f_edges = _block_pool_2d(mag_pow, axis=0, target_bins=max_freq_bins)
        f_crop = np.array([0.5 * (f_crop[s] + f_crop[e - 1]) for s, e in f_edges])

    if t_crop.size > max_time_bins:
        mag_pow, t_edges = _block_pool_2d(mag_pow, axis=1, target_bins=max_time_bins)
        t_crop = np.array([0.5 * (t_crop[s] + t_crop[e - 1]) for s, e in t_edges])

    z_db = 10.0 * np.log10(mag_pow + 1e-18)
    return {
        "freqs": np.round(f_crop, 1).tolist(),
        "times": np.round(t_crop, 4).tolist(),
        "z_db": np.round(z_db, 2).tolist(),
    }


def build_plot_payload(
    noisy: np.ndarray,
    denoised: np.ndarray,
    fs: int,
    stft_out: dict[str, Any],
) -> dict[str, Any]:
    """Build lightweight JSON-serializable Plotly data for waveform, FFT, and spectrogram."""
    t_noisy, y_noisy = _downsample_waveform(noisy, fs)
    t_clean, y_clean = _downsample_waveform(denoised, fs)
    t_removed, y_removed = _downsample_waveform(noisy - denoised, fs)

    f_noisy, fft_noisy_db = _compute_fft_spectrum(noisy, fs)
    f_clean, fft_clean_db = _compute_fft_spectrum(denoised, fs)

    spec_noisy = _downsample_spectrogram(stft_out["freqs"], stft_out["times"], stft_out["zxx_noisy"])
    spec_clean = _downsample_spectrogram(stft_out["freqs"], stft_out["times"], stft_out["zxx_clean"])
    z_diff = np.maximum(
        np.asarray(spec_noisy["z_db"]) - np.asarray(spec_clean["z_db"]),
        0.0,
    )

    return {
        "waveform": {
            "times_noisy": t_noisy,
            "noisy": y_noisy,
            "times_denoised": t_clean,
            "denoised": y_clean,
            "times_removed": t_removed,
            "removed": y_removed,
        },
        "fft": {
            "freqs_noisy": f_noisy,
            "mag_noisy_db": fft_noisy_db,
            "freqs_denoised": f_clean,
            "mag_denoised_db": fft_clean_db,
            "dtmf_markers": {
                "low": list(LOW_FREQS),
                "high": list(HIGH_FREQS),
            },
        },
        "spectrogram": {
            "noisy": spec_noisy,
            "denoised": spec_clean,
            "diff": {
                "freqs": spec_noisy["freqs"],
                "times": spec_noisy["times"],
                "z_db": np.round(z_diff, 2).tolist(),
            },
        },
    }


def run_full_pipeline(
    wav_bytes: bytes,
    params: DenoiseParams,
    target_sequence: str | None = None,
    clean_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute end-to-end sampling, STFT denoising, DTMF decoding, and metrics computation."""
    t0 = time.perf_counter()
    fs, noisy = load_wav_bytes(wav_bytes)
    stft_out = stft_denoise(noisy, fs, params)
    denoised = stft_out["denoised"]
    decoding = decode_dtmf(denoised, fs)
    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    metrics = compute_quality_metrics(
        noisy=noisy,
        denoised=denoised,
        fs=fs,
        stft_out=stft_out,
        tones=decoding["tones"],
        total_latency_ms=total_latency_ms,
        target_sequence=target_sequence,
        clean_reference=clean_reference,
    )
    plots = build_plot_payload(noisy, denoised, fs, stft_out)
    denoised_wav_bytes = save_wav_bytes(fs, denoised)

    return {
        "sample_rate": fs,
        "duration_sec": round(noisy.size / fs, 3),
        "n_samples": int(noisy.size),
        "effective_params": {
            "window_type": params.window_type,
            "nperseg": stft_out["eff_nperseg"],
            "noverlap": stft_out["noverlap"],
            "hop": stft_out["hop"],
            "overlap_pct": round((stft_out["noverlap"] / stft_out["eff_nperseg"]) * 100.0, 1),
            "nfft": stft_out["eff_nfft"],
            "alpha": params.alpha,
            "beta": params.beta,
            "silent_mode": params.silent_mode,
            "silent_ms": params.silent_ms,
            "is_cola": stft_out["is_cola"],
        },
        "warnings": stft_out["warnings"],
        "decoding": decoding,
        "metrics": metrics,
        "plots": plots,
        "denoised_wav_bytes": denoised_wav_bytes,
    }


def synthesize_dtmf_signal(
    sequence: str = "0812345#",
    fs: int = 8000,
    tone_ms: float = 85.0,
    pause_ms: float = 65.0,
    lead_silence_ms: float = 200.0,
    tail_silence_ms: float = 150.0,
    snr_db: float | None = 12.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean DTMF sequence and its AWGN-corrupted counterpart."""
    symbols = [ch for ch in sequence.upper().strip() if ch in SYMBOL_TO_FREQS]
    if not symbols:
        raise ValueError("Sekuens DTMF tidak mengandung simbol valid (0-9, *, #, A-D).")

    lead_len = max(0, int(round((lead_silence_ms / 1000.0) * fs)))
    tail_len = max(0, int(round((tail_silence_ms / 1000.0) * fs)))
    tone_len = max(32, int(round((tone_ms / 1000.0) * fs)))
    pause_len = max(0, int(round((pause_ms / 1000.0) * fs)))

    ramp_len = min(tone_len // 6, max(4, int(round(0.003 * fs))))
    envelope = np.ones(tone_len, dtype=np.float64)
    if ramp_len > 0:
        half_cos = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        envelope[:ramp_len] = half_cos
        envelope[-ramp_len:] = half_cos[::-1]

    t_tone = np.arange(tone_len, dtype=np.float64) / fs
    chunks: list[np.ndarray] = [np.zeros(lead_len, dtype=np.float64)]

    for idx, sym in enumerate(symbols):
        f_low, f_high = SYMBOL_TO_FREQS[sym]
        tone = 0.4 * (np.sin(2.0 * np.pi * f_low * t_tone) + np.sin(2.0 * np.pi * f_high * t_tone))
        chunks.append(tone * envelope)
        if idx < len(symbols) - 1 and pause_len > 0:
            chunks.append(np.zeros(pause_len, dtype=np.float64))

    chunks.append(np.zeros(tail_len, dtype=np.float64))
    clean = np.concatenate(chunks)

    if snr_db is None:
        return clean, clean.copy()

    rng = np.random.default_rng(seed)
    active_samples = clean[np.abs(clean) > 1e-6]
    sig_power = float(np.mean(active_samples ** 2)) if active_samples.size > 0 else float(np.mean(clean ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=clean.size)
    noisy = clean + noise
    return clean, noisy

