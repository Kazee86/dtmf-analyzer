(() => {
  // DOM Element References
  const form = document.getElementById("processForm");
  const audioInput = document.getElementById("audioInput");
  const loadSampleBtn = document.getElementById("loadSampleBtn");
  const fileLabel = document.getElementById("fileLabel");
  const noisyPlayer = document.getElementById("noisyPlayer");
  const denoisedPlayer = document.getElementById("denoisedPlayer");

  const overlapRange = document.getElementById("overlapRange");
  const overlapVal = document.getElementById("overlapVal");
  const alphaRange = document.getElementById("alphaRange");
  const alphaVal = document.getElementById("alphaVal");
  const betaRange = document.getElementById("betaRange");
  const betaVal = document.getElementById("betaVal");
  const silentModeSelect = document.getElementById("silentModeSelect");
  const silentMsGroup = document.getElementById("silentMsGroup");
  const silentManualGroup = document.getElementById("silentManualGroup");
  const targetInput = document.getElementById("targetInput");

  const submitBtn = document.getElementById("submitBtn");
  const submitSpinner = document.getElementById("submitSpinner");
  const submitText = document.getElementById("submitText");

  const errorBanner = document.getElementById("errorBanner");
  const errorText = document.getElementById("errorText");
  const warningBanner = document.getElementById("warningBanner");
  const warningList = document.getElementById("warningList");

  const decodedSequenceEl = document.getElementById("decodedSequence");
  const copyBtn = document.getElementById("copyBtn");
  const audioMetaBadge = document.getElementById("audioMetaBadge");
  const tonesTableBody = document.getElementById("tonesTableBody");
  const targetComparisonBox = document.getElementById("targetComparisonBox");
  const targetAccuracyText = document.getElementById("targetAccuracyText");
  const targetDiffText = document.getElementById("targetDiffText");

  const colaBadge = document.getElementById("colaBadge");
  const metricSnrIn = document.getElementById("metricSnrIn");
  const metricSnrOut = document.getElementById("metricSnrOut");
  const metricSnrGain = document.getElementById("metricSnrGain");
  const cardSnrOut = document.getElementById("cardSnrOut");
  const metricMusicalNoise = document.getElementById("metricMusicalNoise");
  const metricResidualNoise = document.getElementById("metricResidualNoise");
  const cardMusicalNoise = document.getElementById("cardMusicalNoise");
  const metricPreEcho = document.getElementById("metricPreEcho");
  const cardPreEcho = document.getElementById("cardPreEcho");
  const metricFrameLatency = document.getElementById("metricFrameLatency");
  const metricTotalLatency = document.getElementById("metricTotalLatency");
  const cardLatency = document.getElementById("cardLatency");
  const metricResolution = document.getElementById("metricResolution");
  const metricFramesCount = document.getElementById("metricFramesCount");

  // Numpad Generator DOM
  const toggleNumpadBtn = document.getElementById("toggleNumpadBtn");
  const numpadPanelBody = document.getElementById("numpadPanelBody");
  const numpadDisplay = document.getElementById("numpadDisplay");
  const numpadFreqReadout = document.getElementById("numpadFreqReadout");
  const numpadSoundToggle = document.getElementById("numpadSoundToggle");
  const numpadBackspaceBtn = document.getElementById("numpadBackspaceBtn");
  const numpadClearBtn = document.getElementById("numpadClearBtn");
  const genAddNoise = document.getElementById("genAddNoise");
  const genSnrRange = document.getElementById("genSnrRange");
  const genSnrVal = document.getElementById("genSnrVal");
  const genSnrGroup = document.getElementById("genSnrGroup");
  const genFsSelect = document.getElementById("genFsSelect");
  const genToneMs = document.getElementById("genToneMs");
  const genPauseMs = document.getElementById("genPauseMs");
  const genPlayer = document.getElementById("genPlayer");
  const genStatusText = document.getElementById("genStatusText");
  const genCreateBtn = document.getElementById("genCreateBtn");
  const genDownloadLink = document.getElementById("genDownloadLink");
  const genSendToTestBtn = document.getElementById("genSendToTestBtn");

  let selectedFile = null;
  let currentFileId = null;
  let currentFilename = "audio.wav";
  let lastSequence = "";
  let objectUrlPreview = null;
  let generatedWavUrl = null;
  let audioCtx = null;

  // Initialize Lenis Smooth Scroll
  let lenisInstance = null;
  if (typeof window.Lenis !== "undefined") {
    try {
      lenisInstance = new window.Lenis({
        duration: 1.1,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        orientation: "vertical",
        smoothWheel: true,
      });

      function raf(time) {
        if (lenisInstance) {
          lenisInstance.raf(time);
          requestAnimationFrame(raf);
        }
      }
      requestAnimationFrame(raf);
    } catch {
      // Graceful fallback if Lenis fails
    }
  }

  // GSAP Initial Entrance Stagger
  if (typeof window.gsap !== "undefined") {
    window.gsap.from(".workstation-panel", {
      opacity: 0,
      y: 12,
      stagger: 0.08,
      duration: 0.5,
      ease: "power2.out",
    });
  }

  // DTMF ITU-T Q.23 Dual Tone Frequency Table
  const DTMF_FREQS_MAP = {
    "1": [697, 1209], "2": [697, 1336], "3": [697, 1477], "A": [697, 1633],
    "4": [770, 1209], "5": [770, 1336], "6": [770, 1477], "B": [770, 1633],
    "7": [852, 1209], "8": [852, 1336], "9": [852, 1477], "C": [852, 1633],
    "*": [941, 1209], "0": [941, 1336], "#": [941, 1477], "D": [941, 1633],
  };

  function playDtmfBeep(lowFreq, highFreq) {
    if (!numpadSoundToggle.checked) return;
    const AudioContextCls = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCls) return;
    try {
      if (!audioCtx) audioCtx = new AudioContextCls();
      if (audioCtx.state === "suspended") audioCtx.resume();

      const now = audioCtx.currentTime;
      const dur = 0.13;
      const gain = audioCtx.createGain();
      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.14, now + 0.01);
      gain.gain.setValueAtTime(0.14, now + dur - 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + dur);

      const osc1 = audioCtx.createOscillator();
      const osc2 = audioCtx.createOscillator();
      osc1.type = "sine";
      osc2.type = "sine";
      osc1.frequency.setValueAtTime(lowFreq, now);
      osc2.frequency.setValueAtTime(highFreq, now);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(audioCtx.destination);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + dur);
      osc2.stop(now + dur);
    } catch {
      // AudioContext autoplay restrictions handled silently
    }
  }

  function triggerTactileFeedback(element) {
    if (typeof window.Motion !== "undefined" && typeof window.Motion.animate === "function") {
      window.Motion.animate(element, { scale: [1, 0.94, 1] }, { duration: 0.14, easing: "ease-out" });
    } else if (typeof window.gsap !== "undefined") {
      window.gsap.fromTo(element, { scale: 0.95 }, { scale: 1, duration: 0.15, ease: "power1.out" });
    }
  }

  function showError(message) {
    errorText.textContent = message;
    errorBanner.classList.remove("hidden");
    if (typeof window.gsap !== "undefined") {
      window.gsap.fromTo(errorBanner, { opacity: 0, y: -8 }, { opacity: 1, y: 0, duration: 0.3 });
    }
  }

  function hideError() {
    errorBanner.classList.add("hidden");
  }

  function renderWarnings(warnings) {
    warningList.innerHTML = "";
    if (!warnings || warnings.length === 0) {
      warningBanner.classList.add("hidden");
      return;
    }
    warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      warningList.appendChild(li);
    });
    warningBanner.classList.remove("hidden");
  }

  function setCardTone(cardEl, status) {
    cardEl.classList.remove(
      "border-telecom-border", "bg-telecom-panelElevated/60",
      "border-emerald-500/50", "bg-emerald-950/30",
      "border-amber-500/50", "bg-amber-950/30",
      "border-rose-500/50", "bg-rose-950/30"
    );
    if (status === "good") {
      cardEl.classList.add("border-emerald-500/50", "bg-emerald-950/30");
    } else if (status === "warn") {
      cardEl.classList.add("border-amber-500/50", "bg-amber-950/30");
    } else if (status === "bad") {
      cardEl.classList.add("border-rose-500/50", "bg-rose-950/30");
    } else {
      cardEl.classList.add("border-telecom-border", "bg-telecom-panelElevated/60");
    }
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitSpinner.classList.toggle("hidden", !isLoading);
    submitText.textContent = isLoading ? "Memproses STFT & Dekoding..." : "Jalankan Analisis STFT & Dekoding";
  }

  function updateLocalAudioPreview(file) {
    if (objectUrlPreview) {
      URL.revokeObjectURL(objectUrlPreview);
    }
    objectUrlPreview = URL.createObjectURL(file);
    noisyPlayer.src = objectUrlPreview;
  }

  overlapRange.addEventListener("input", () => {
    overlapVal.textContent = `${overlapRange.value}%`;
  });

  alphaRange.addEventListener("input", () => {
    alphaVal.textContent = Number(alphaRange.value).toFixed(1);
  });

  betaRange.addEventListener("input", () => {
    betaVal.textContent = Number(betaRange.value).toFixed(3);
  });

  silentModeSelect.addEventListener("change", () => {
    silentMsGroup.classList.toggle("hidden", silentModeSelect.value !== "initial");
    silentManualGroup.classList.toggle("hidden", silentModeSelect.value !== "manual");
  });

  audioInput.addEventListener("change", () => {
    const file = audioInput.files[0];
    if (!file) return;
    selectedFile = file;
    currentFileId = null;
    currentFilename = file.name;
    fileLabel.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    updateLocalAudioPreview(file);
    hideError();
  });

  loadSampleBtn.addEventListener("click", async () => {
    hideError();
    loadSampleBtn.disabled = true;
    const origText = loadSampleBtn.textContent;
    loadSampleBtn.textContent = "Memuat...";
    try {
      const resp = await fetch("/api/sample-wav?sequence=0812345%23&snr_db=10.0");
      if (!resp.ok) {
        throw new Error("Gagal membuat contoh audio dari server.");
      }
      const blob = await resp.blob();
      const sampleFile = new File([blob], "sample_dtmf_0812345_snr10dB.wav", { type: "audio/wav" });
      selectedFile = sampleFile;
      currentFileId = null;
      currentFilename = sampleFile.name;
      fileLabel.textContent = `${sampleFile.name} (${(sampleFile.size / 1024).toFixed(1)} KB)`;
      if (!targetInput.value.trim()) {
        targetInput.value = "0812345#";
      }
      updateLocalAudioPreview(sampleFile);
      await runProcess();
    } catch (err) {
      showError(err.message);
    } finally {
      loadSampleBtn.disabled = false;
      loadSampleBtn.textContent = origText;
    }
  });

  // Numpad Generator Interactions
  toggleNumpadBtn.addEventListener("click", () => {
    const willHide = !numpadPanelBody.classList.contains("hidden");
    if (typeof window.gsap !== "undefined") {
      if (willHide) {
        window.gsap.to(numpadPanelBody, {
          opacity: 0,
          duration: 0.2,
          onComplete: () => {
            numpadPanelBody.classList.add("hidden");
            toggleNumpadBtn.textContent = "Tampilkan Numpad";
            toggleNumpadBtn.setAttribute("aria-expanded", "false");
          },
        });
      } else {
        numpadPanelBody.classList.remove("hidden");
        toggleNumpadBtn.textContent = "Sembunyikan Numpad";
        toggleNumpadBtn.setAttribute("aria-expanded", "true");
        window.gsap.fromTo(numpadPanelBody, { opacity: 0 }, { opacity: 1, duration: 0.25 });
      }
    } else {
      numpadPanelBody.classList.toggle("hidden", willHide);
      toggleNumpadBtn.textContent = willHide ? "Tampilkan Numpad" : "Sembunyikan Numpad";
      toggleNumpadBtn.setAttribute("aria-expanded", String(!willHide));
    }
  });

  document.querySelectorAll(".dtmf-key-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      triggerTactileFeedback(btn);
      const key = btn.dataset.dtmfKey;
      const fl = Number(btn.dataset.low);
      const fh = Number(btn.dataset.high);
      if (numpadDisplay.value.length < 40) {
        numpadDisplay.value += key;
      }
      numpadFreqReadout.textContent = `Tombol ${key} : ${fl} Hz + ${fh} Hz`;
      playDtmfBeep(fl, fh);
    });
  });

  numpadDisplay.addEventListener("input", () => {
    const raw = numpadDisplay.value.toUpperCase();
    const cleaned = raw.replace(/[^0-9*#ABCD]/g, "");
    if (cleaned !== numpadDisplay.value) {
      numpadDisplay.value = cleaned;
    }
    const lastChar = cleaned.slice(-1);
    if (lastChar && DTMF_FREQS_MAP[lastChar]) {
      const [fl, fh] = DTMF_FREQS_MAP[lastChar];
      numpadFreqReadout.textContent = `Tombol ${lastChar} : ${fl} Hz + ${fh} Hz`;
      playDtmfBeep(fl, fh);
    }
  });

  numpadBackspaceBtn.addEventListener("click", () => {
    triggerTactileFeedback(numpadBackspaceBtn);
    numpadDisplay.value = numpadDisplay.value.slice(0, -1);
    if (!numpadDisplay.value) {
      numpadFreqReadout.textContent = "Siap memanggil : Tekan tombol";
    }
  });

  numpadClearBtn.addEventListener("click", () => {
    triggerTactileFeedback(numpadClearBtn);
    numpadDisplay.value = "";
    numpadFreqReadout.textContent = "Siap memanggil : Tekan tombol";
  });

  function updateSnrUi() {
    const enabled = genAddNoise.checked;
    genSnrGroup.classList.toggle("opacity-40", !enabled);
    genSnrRange.disabled = !enabled;
    genSnrVal.textContent = enabled ? `${Number(genSnrRange.value).toFixed(1)} dB` : "Bersih (Tanpa Derau)";
  }

  genAddNoise.addEventListener("change", updateSnrUi);
  genSnrRange.addEventListener("input", updateSnrUi);
  updateSnrUi();

  async function synthesizeWavFromNumpad() {
    hideError();
    const seq = numpadDisplay.value.trim();
    if (!seq) {
      throw new Error("Tekan minimal 1 tombol pada Numpad terlebih dahulu.");
    }

    const body = {
      sequence: seq,
      fs: Number(genFsSelect.value),
      tone_ms: Number(genToneMs.value),
      pause_ms: Number(genPauseMs.value),
      add_noise: genAddNoise.checked,
      snr_db: Number(genSnrRange.value),
    };

    const resp = await fetch("/api/generate-wav", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      let errMsg = "Gagal menyintesis file WAV.";
      try {
        const errJson = await resp.json();
        errMsg = errJson.error || errMsg;
      } catch {
        // Fallback default error
      }
      throw new Error(errMsg);
    }

    const blob = await resp.blob();
    if (generatedWavUrl) URL.revokeObjectURL(generatedWavUrl);
    generatedWavUrl = URL.createObjectURL(blob);

    const safeSlug = seq.replace(/\*/g, "star").replace(/#/g, "hash").replace(/[^0-9A-Za-z]/g, "").slice(0, 20);
    const fname = `dtmf_${safeSlug || "tone"}.wav`;

    genPlayer.src = generatedWavUrl;
    genPlayer.load();
    genDownloadLink.href = generatedWavUrl;
    genDownloadLink.download = fname;
    genDownloadLink.setAttribute("aria-disabled", "false");
    genDownloadLink.classList.remove("pointer-events-none", "opacity-40");
    genStatusText.textContent = `Siap : ${(blob.size / 1024).toFixed(1)} KB (${fname})`;

    return new File([blob], fname, { type: "audio/wav" });
  }

  genCreateBtn.addEventListener("click", async () => {
    triggerTactileFeedback(genCreateBtn);
    genCreateBtn.disabled = true;
    const prev = genCreateBtn.textContent;
    genCreateBtn.textContent = "Menyintesis...";
    try {
      await synthesizeWavFromNumpad();
    } catch (err) {
      showError(err.message);
    } finally {
      genCreateBtn.disabled = false;
      genCreateBtn.textContent = prev;
    }
  });

  genSendToTestBtn.addEventListener("click", async () => {
    triggerTactileFeedback(genSendToTestBtn);
    genSendToTestBtn.disabled = true;
    const prev = genSendToTestBtn.textContent;
    genSendToTestBtn.textContent = "Mengirim...";
    try {
      const file = await synthesizeWavFromNumpad();
      selectedFile = file;
      currentFileId = null;
      currentFilename = file.name;
      fileLabel.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      targetInput.value = numpadDisplay.value.trim().toUpperCase();
      updateLocalAudioPreview(file);
      await runProcess();
    } catch (err) {
      showError(err.message);
    } finally {
      genSendToTestBtn.disabled = false;
      genSendToTestBtn.textContent = prev;
    }
  });

  copyBtn.addEventListener("click", async () => {
    if (!lastSequence) return;
    try {
      await navigator.clipboard.writeText(lastSequence);
      const prev = copyBtn.textContent;
      copyBtn.textContent = "Tersalin!";
      setTimeout(() => {
        copyBtn.textContent = prev;
      }, 1500);
    } catch {
      showError("Browser tidak mengizinkan akses clipboard.");
    }
  });

  function renderDecoding(data) {
    const decoding = data.decoding;
    lastSequence = decoding.sequence || "";
    decodedSequenceEl.textContent = lastSequence || "(Tidak ada nada DTMF terdeteksi)";
    copyBtn.disabled = !lastSequence;

    audioMetaBadge.textContent = `${data.sample_rate} Hz : ${data.duration_sec}s`;
    audioMetaBadge.classList.remove("hidden");

    if (typeof window.gsap !== "undefined") {
      window.gsap.fromTo(decodedSequenceEl, { opacity: 0.5, scale: 0.98 }, { opacity: 1, scale: 1, duration: 0.25 });
    }

    const targetEval = data.metrics.target_eval;
    if (targetEval) {
      targetComparisonBox.classList.remove("hidden");
      targetAccuracyText.textContent = `${targetEval.accuracy_pct}%`;
      targetAccuracyText.className = `text-2xl font-bold font-mono mt-1 ${
        targetEval.exact_match ? "text-emerald-400" : targetEval.accuracy_pct >= 75 ? "text-amber-400" : "text-rose-400"
      }`;
      targetDiffText.textContent = `Target: ${targetEval.target} : Error: ${targetEval.edit_distance}`;
    } else {
      targetComparisonBox.classList.add("hidden");
    }

    tonesTableBody.innerHTML = "";
    if (!decoding.tones || decoding.tones.length === 0) {
      tonesTableBody.innerHTML = `
        <tr>
          <td colspan="7" class="px-3 py-5 text-center text-slate-400">
            Tidak ada burst DTMF valid yang terdeteksi. Silakan sesuaikan α atau perkecil window.
          </td>
        </tr>`;
      return;
    }

    decoding.tones.forEach((t) => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-telecom-panelElevated/50 transition";
      tr.innerHTML = `
        <td class="px-3 py-2 text-slate-400 font-mono">${t.index}</td>
        <td class="px-3 py-2 font-mono font-bold text-cyan-400 text-sm">${t.symbol}</td>
        <td class="px-3 py-2 font-mono text-slate-200">${t.start_time.toFixed(3)} sampai ${t.end_time.toFixed(3)}</td>
        <td class="px-3 py-2 font-mono text-slate-300">${t.duration_ms} ms</td>
        <td class="px-3 py-2 font-mono text-amber-300">${t.f_low} <span class="text-slate-500">(${t.nominal_low})</span></td>
        <td class="px-3 py-2 font-mono text-cyan-300">${t.f_high} <span class="text-slate-500">(${t.nominal_high})</span></td>
        <td class="px-3 py-2 font-mono text-emerald-400 font-semibold">${t.confidence}%</td>
      `;
      tonesTableBody.appendChild(tr);
    });
  }

  function renderMetrics(data) {
    const m = data.metrics;
    const p = data.effective_params;

    colaBadge.classList.remove("hidden", "bg-emerald-950/60", "text-emerald-300", "border-emerald-800", "bg-amber-950/60", "text-amber-300", "border-amber-800");
    if (p.is_cola) {
      colaBadge.classList.add("bg-emerald-950/60", "text-emerald-300", "border-emerald-800");
      colaBadge.textContent = "COLA Terpenuhi";
    } else {
      colaBadge.classList.add("bg-amber-950/60", "text-amber-300", "border-amber-800");
      colaBadge.textContent = "Non-COLA (Dinormalisasi)";
    }

    metricSnrIn.textContent = m.snr_in_db !== null ? `${m.snr_in_db.toFixed(1)} dB` : "0.0 dB";

    if (m.snr_out_db !== null) {
      metricSnrOut.textContent = `${m.snr_out_db.toFixed(1)} dB`;
      const gainStr = m.snr_gain_db !== null ? `Δ ${m.snr_gain_db >= 0 ? "+" : ""}${m.snr_gain_db.toFixed(1)} dB` : "";
      metricSnrGain.textContent = `${gainStr} : Target ≥ 25.0 dB`;
      setCardTone(cardSnrOut, m.snr_out_db >= 25.0 ? "good" : m.snr_out_db >= 15.0 ? "warn" : "bad");
    } else {
      metricSnrOut.textContent = "0.0 dB";
      metricSnrGain.textContent = "Target P4 : ≥ 25.0 dB";
      setCardTone(cardSnrOut, "neutral");
    }

    if (m.musical_noise_db !== null) {
      metricMusicalNoise.textContent = `${m.musical_noise_db >= 0 ? "+" : ""}${m.musical_noise_db.toFixed(2)} dB`;
      const resText = m.residual_noise_db !== null ? `Lantai: ${m.residual_noise_db.toFixed(1)} dB` : "Target P4 : ≤ 0.0 dB";
      metricResidualNoise.textContent = `${resText} : Target ≤ 0.0 dB`;
      setCardTone(
        cardMusicalNoise,
        m.musical_noise_db <= 0.0 ? "good" : m.musical_noise_db <= 5.5 ? "warn" : "bad"
      );
    } else {
      metricMusicalNoise.textContent = "0.0 dB";
      metricResidualNoise.textContent = "Target P4 : ≤ 0.0 dB";
      setCardTone(cardMusicalNoise, "neutral");
    }

    if (m.pre_echo_db !== null) {
      metricPreEcho.textContent = `${m.pre_echo_db.toFixed(2)} dB`;
      setCardTone(
        cardPreEcho,
        m.pre_echo_db <= -30.0 ? "good" : m.pre_echo_db <= -20.0 ? "warn" : "bad"
      );
    } else {
      metricPreEcho.textContent = "0.0 dB";
      setCardTone(cardPreEcho, "neutral");
    }

    metricFrameLatency.textContent = `${m.frame_latency_ms.toFixed(3)} ms`;
    metricTotalLatency.textContent = `Total: ${m.total_latency_ms.toFixed(1)} ms : Target ≤ 40.0 ms`;
    setCardTone(cardLatency, m.frame_latency_ms <= 40.0 ? "good" : "bad");

    const df = (data.sample_rate / p.nfft).toFixed(1);
    const hopMs = ((p.hop / data.sample_rate) * 1000).toFixed(1);
    metricResolution.textContent = `${df} Hz / ${hopMs} ms`;
    metricFramesCount.textContent = `${m.n_stft_frames} frame STFT (N=${p.nperseg})`;

    if (typeof window.gsap !== "undefined") {
      window.gsap.fromTo(
        [cardSnrOut, cardMusicalNoise, cardPreEcho, cardLatency],
        { y: -3 },
        { y: 0, duration: 0.3, ease: "power1.out" }
      );
    }
  }

  function renderPlots(plots) {
    document.querySelectorAll(".plot-placeholder").forEach((el) => {
      el.classList.add("hidden");
    });

    const baseLayout = {
      autosize: true,
      margin: { t: 28, r: 20, b: 48, l: 56 },
      font: { family: "JetBrains Mono, monospace", size: 10, color: "#94a3b8" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#0a1017",
      xaxis: {
        gridcolor: "#1a2533",
        zerolinecolor: "#27384c",
        tickfont: { color: "#94a3b8", size: 9 },
        titlefont: { color: "#cbd5e1", size: 10 },
      },
      yaxis: {
        gridcolor: "#1a2533",
        zerolinecolor: "#27384c",
        tickfont: { color: "#94a3b8", size: 9 },
        titlefont: { color: "#cbd5e1", size: 10 },
      },
    };
    const plotConfig = { responsive: true, displaylogo: false };

    // 1. Waveform Comparison
    Plotly.newPlot(
      "waveformPlot",
      [
        {
          x: plots.waveform.times_noisy,
          y: plots.waveform.noisy,
          name: "Sebelum (Berderau)",
          type: "scatter",
          mode: "lines",
          line: { color: "rgba(148, 163, 184, 0.55)", width: 1.1 },
        },
        {
          x: plots.waveform.times_denoised,
          y: plots.waveform.denoised,
          name: "Sesudah (STFT)",
          type: "scatter",
          mode: "lines",
          line: { color: "#06b6d4", width: 1.2 },
        },
        {
          x: plots.waveform.times_removed,
          y: plots.waveform.removed,
          name: "Derau Dibuang",
          type: "scatter",
          mode: "lines",
          line: { color: "rgba(244, 63, 94, 0.7)", width: 0.9 },
        },
      ],
      {
        ...baseLayout,
        xaxis: { ...baseLayout.xaxis, title: "Waktu (detik)" },
        yaxis: { ...baseLayout.yaxis, title: "Amplitudo", range: [-1.05, 1.05] },
        legend: { orientation: "h", y: 1.15, font: { color: "#cbd5e1", size: 10 } },
      },
      plotConfig
    );

    // 2. FFT Spectrum with DTMF Frequency Markers
    const allMarkers = [
      ...plots.fft.dtmf_markers.low.map((f) => ({ f, color: "rgba(245, 158, 11, 0.7)" })),
      ...plots.fft.dtmf_markers.high.map((f) => ({ f, color: "rgba(6, 182, 212, 0.7)" })),
    ];
    const shapes = allMarkers.map((m) => ({
      type: "line",
      x0: m.f,
      x1: m.f,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: m.color, width: 1, dash: "dot" },
    }));

    Plotly.newPlot(
      "fftPlot",
      [
        {
          x: plots.fft.freqs_noisy,
          y: plots.fft.mag_noisy_db,
          name: "Sebelum (Berderau)",
          type: "scatter",
          mode: "lines",
          line: { color: "rgba(148, 163, 184, 0.55)", width: 1.1 },
        },
        {
          x: plots.fft.freqs_denoised,
          y: plots.fft.mag_denoised_db,
          name: "Sesudah (STFT)",
          type: "scatter",
          mode: "lines",
          line: { color: "#10b981", width: 1.4 },
        },
      ],
      {
        ...baseLayout,
        xaxis: { ...baseLayout.xaxis, title: "Frekuensi (Hz)" },
        yaxis: { ...baseLayout.yaxis, title: "Magnitudo (dB)" },
        shapes,
        legend: { orientation: "h", y: 1.12, font: { color: "#cbd5e1", size: 10 } },
      },
      plotConfig
    );

    // 3. Synchronized Spectrogram Heatmaps + Difference Heatmap
    const specNoisy = plots.spectrogram.noisy;
    const specClean = plots.spectrogram.denoised;
    const specDiff = plots.spectrogram.diff;

    const noisyFlat = [];
    for (const row of specNoisy.z_db) {
      for (const v of row) noisyFlat.push(v);
    }
    noisyFlat.sort((a, b) => a - b);
    const zMax = noisyFlat.length ? noisyFlat[noisyFlat.length - 1] : 0;
    const p15 = noisyFlat.length ? noisyFlat[Math.floor(noisyFlat.length * 0.15)] : -80;
    const zMin = Math.max(p15 - 5, zMax - 55);

    let diffMax = 1;
    for (const row of specDiff.z_db) {
      for (const v of row) {
        if (v > diffMax) diffMax = v;
      }
    }

    const specLayout = {
      ...baseLayout,
      margin: { t: 16, r: 24, b: 48, l: 56 },
      xaxis: { ...baseLayout.xaxis, title: "Waktu (detik)" },
      yaxis: { ...baseLayout.yaxis, title: "Frekuensi (Hz)" },
    };

    Plotly.newPlot(
      "specNoisyPlot",
      [
        {
          x: specNoisy.times,
          y: specNoisy.freqs,
          z: specNoisy.z_db,
          type: "heatmap",
          colorscale: "Viridis",
          zmin: zMin,
          zmax: zMax,
          colorbar: { title: "dB", thickness: 10, tickfont: { color: "#94a3b8", size: 9 }, titlefont: { color: "#cbd5e1", size: 10 } },
        },
      ],
      { ...specLayout },
      plotConfig
    );

    Plotly.newPlot(
      "specCleanPlot",
      [
        {
          x: specClean.times,
          y: specClean.freqs,
          z: specClean.z_db,
          type: "heatmap",
          colorscale: "Viridis",
          zmin: zMin,
          zmax: zMax,
          colorbar: { title: "dB", thickness: 10, tickfont: { color: "#94a3b8", size: 9 }, titlefont: { color: "#cbd5e1", size: 10 } },
        },
      ],
      { ...specLayout },
      plotConfig
    );

    Plotly.newPlot(
      "specDiffPlot",
      [
        {
          x: specDiff.times,
          y: specDiff.freqs,
          z: specDiff.z_db,
          type: "heatmap",
          colorscale: "Inferno",
          zmin: 0,
          zmax: Math.max(3, diffMax),
          colorbar: { title: "Δ dB", thickness: 10, tickfont: { color: "#94a3b8", size: 9 }, titlefont: { color: "#cbd5e1", size: 10 } },
        },
      ],
      { ...specLayout },
      plotConfig
    );

    requestAnimationFrame(() => {
      ["waveformPlot", "fftPlot", "specNoisyPlot", "specCleanPlot", "specDiffPlot"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) Plotly.Plots.resize(el);
      });
    });
  }

  async function runProcess() {
    hideError();
    if (!selectedFile && !currentFileId) {
      showError("Silakan pilih berkas audio .wav atau muat contoh WAV terlebih dahulu.");
      return;
    }

    const fd = new FormData();
    if (selectedFile && !currentFileId) {
      fd.append("audio", selectedFile);
    } else if (currentFileId) {
      fd.append("file_id", currentFileId);
      fd.append("filename", currentFilename);
    }

    fd.append("nperseg", document.getElementById("npersegSelect").value);
    fd.append("window_type", document.getElementById("windowTypeSelect").value);
    fd.append("overlap_pct", overlapRange.value);
    fd.append("alpha", alphaRange.value);
    fd.append("beta", betaRange.value);
    fd.append("silent_mode", silentModeSelect.value);
    fd.append("silent_ms", document.getElementById("silentMsInput").value);
    fd.append("silent_start_s", document.getElementById("silentStartInput").value);
    fd.append("silent_end_s", document.getElementById("silentEndInput").value);
    fd.append("target_sequence", targetInput.value);

    setLoading(true);
    try {
      const resp = await fetch("/api/process", {
        method: "POST",
        body: fd,
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "Terjadi kesalahan saat memproses audio.");
      }

      const prevFileId = currentFileId;
      currentFileId = data.file_id;
      currentFilename = data.filename || currentFilename;
      selectedFile = null;

      if (prevFileId !== currentFileId) {
        noisyPlayer.src = data.noisy_audio_url;
        noisyPlayer.load();
      }
      denoisedPlayer.src = `${data.denoised_audio_url}?t=${Date.now()}`;
      denoisedPlayer.load();

      renderWarnings(data.warnings);
      renderDecoding(data);
      renderMetrics(data);
      renderPlots(data.plots);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await runProcess();
  });
})();
