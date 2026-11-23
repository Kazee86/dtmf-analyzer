# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Mahasiswa, dosen pengampu, serta peneliti dan engineer sistem komunikasi / DSP yang menguji serta menganalisis performa reduksi derau pada sinyal pensinyalan telekomunikasi.

## Product Purpose
Aplikasi diagnostik web interaktif untuk mereduksi derau pada berkas audio `.wav` menggunakan pipeline STFT Denoising (estimasi Noise PSD pada segmen senyap, subtraksi spektral, rekonstruksi ISTFT), mengekstrak dan mendekode nada DTMF (Dual-Tone Multi-Frequency) menjadi simbol keypad telepon, serta menyajikan visualisasi spektrum dan spektrogram secara instan di browser.

## Positioning
Laboratorium diagnostik DSP dua arah: menggabungkan simulator sintesis nada keypad telepon (generator DTMF + injeksi derau AWGN) dan mesin uji evaluasi trade-off parameter STFT (resolusi waktu vs frekuensi, SNR out vs musical noise) dengan tolok ukur kuantitatif Skenario P4.

## Operating Context
Pengujian dilakukan melalui browser desktop. Alur kerja mencakup pengunggahan berkas rekaman audio nyata atau pembuatan sinyal uji via keypad simulator, penyesuaian parameter komputasi, dan evaluasi hasil decoding serta grafik spektrogram secara berdampingan.

## Capabilities and Constraints
- Komputasi backend berbasis Python (NumPy, SciPy) dengan batas berkas audio maksimal 15 MB.
- Standar frekuensi nada DTMF mengacu pada ITU-T Q.23 (Frekuensi Rendah: 697, 770, 852, 941 Hz; Frekuensi Tinggi: 1209, 1336, 1477, 1633 Hz).
- Pipeline STFT mendukung variasi panjang window (256, 512, 1024, 2048 sampel), jenis window (Hann, Hamming, Blackman, Rectangular), faktor oversubstitusi (α), dan rasio overlap.
- Target metrik evaluasi kuantitatif (Skenario P4): SNR_out ≥ 25.0 dB, Pre-Echo ≤ -30.0 dB, Musical Noise ≤ 0.0 dB, Latensi komputasi DSP ≤ 40.0 ms.
- Antarmuka visual menyajikan plot waveform, spektrum frekuensi, spektrogram sebelum/sesudah filter, dan log riwayat decoding via Plotly.js.

## Brand Commitments
Identitas akademik resmi: "Aplikasi Web Denoising STFT untuk Ekstraksi Nada DTMF pada Kondisi Berderau" oleh Kelompok G (Mata Kuliah Pemrosesan Sinyal Digital, Semester 5).

## Evidence on Hand
- Kode sumber fungsional: `app.py`, `dsp_engine.py`, `templates/index.html`.
- Spesifikasi teknis: `prompt_prd_dtmf_stft.md`.
- Laporan & makalah acuan: `Aplikasi Web Denoising STFT untuk Ekstraksi Nada DTMF pada Kondisi Berderau_2.pdf`.
- Pengujian otomatis: test suite di direktori `tests/`.

## Product Principles
- **Presisi Matematis**: Metrik evaluasi dan deteksi nada harus mencerminkan hasil komputasi DSP riil tanpa manipulasi visual.
- **Keterbacaan Diagnostik**: Visualisasi grafik dan perbandingan spektrum dirancang untuk memudahkan pemahaman kompromi trade-off parameter sinyal.
- **Efisiensi Alur Uji**: Mengeliminasi kebutuhan instalasi perangkat lunak desktop (seperti MATLAB) dengan menyediakan generator dan decoder dalam satu antarmuka terpadu.

## Accessibility & Inclusion
Tata letak kontrol terstruktur berbasis grid, kontras warna memadai untuk visualisasi plot teknis, dan indikator status proses yang jelas bagi pengguna.
