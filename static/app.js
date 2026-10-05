(() => {
  "use strict";

  const HOLD_MS = 500;          // press longer than this = hold-to-talk
  const MIN_RECORD_MS = 400;    // shorter recordings are discarded
  const MAX_RECORD_MS = 25000;  // Sarvam's REST limit is 30 s
  const NEXT_DELAY_MS = 600;
  // Tiny silent WAV, played inside the Start click to unlock audio on iOS/Safari.
  const SILENT_WAV =
    "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

  const $ = (id) => document.getElementById(id);
  const el = {
    setup: $("setup"), topic: $("topic"), language: $("language"), start: $("start"),
    status: $("status"), score: $("score"), mic: $("mic"),
    questionCard: $("questionCard"), question: $("question"),
    transcriptCard: $("transcriptCard"), transcript: $("transcript"),
    resultCard: $("resultCard"), resultLabel: $("resultLabel"), resultFeedback: $("resultFeedback"),
    errorBox: $("errorBox"), errorText: $("errorText"), retry: $("retry"),
  };

  const player = new Audio();
  const state = {
    run: 0,              // bumped on every Start/Stop so stale async work can bail out
    phase: "idle",
    topic: "", language: "en-IN",
    asked: [],           // every question asked this session, sent with each /question call
    current: null,       // question awaiting an answer
    correct: 0, answered: 0,
    stream: null, recorder: null, chunks: [], recStart: 0,
    pressActive: false, autoStop: null, retryFn: null,
  };

  class ApiError extends Error {}

  // ---------- UI helpers ----------
  function setPhase(phase, text) {
    state.phase = phase;
    document.body.dataset.phase = phase;
    if (text !== undefined) el.status.textContent = text;
    el.mic.disabled = !(phase === "ready" || phase === "recording");
    el.mic.classList.toggle("recording", phase === "recording");
    el.mic.setAttribute("aria-label", phase === "recording" ? "Stop recording" : "Record your answer");
  }

  function showError(message, retryFn = null) {
    el.errorText.textContent = message;
    el.errorBox.hidden = false;
    state.retryFn = retryFn;
    el.retry.hidden = !retryFn;
  }

  function clearError() {
    el.errorBox.hidden = true;
    el.retry.hidden = true;
    state.retryFn = null;
  }

  function updateScore() {
    el.score.textContent = `Score: ${state.correct} / ${state.answered}`;
  }

  function resetCards() {
    el.questionCard.hidden = el.transcriptCard.hidden = el.resultCard.hidden = true;
  }

  function lockSetup(running) {
    el.topic.disabled = el.language.disabled = running;
    el.setup.classList.toggle("running", running); // collapses the form to just the Stop button
    el.start.textContent = running ? "Stop" : "Start";
    el.start.classList.toggle("primary", !running);
    el.start.classList.toggle("stop", running);
  }

  // ---------- network ----------
  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch {
      throw new ApiError("Network error: could not reach the server. Check your connection and try again.");
    }
    if (!res.ok) {
      let message = `Server error (${res.status})`;
      try {
        const data = await res.json();
        if (data && data.error) message = data.error;
      } catch { /* non-JSON error body */ }
      throw new ApiError(message);
    }
    return res;
  }

  const postJson = (path, body) =>
    api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  // ---------- audio playback ----------
  function stopPlayback() {
    player.pause();
    player.dispatchEvent(new Event("ended")); // settles any pending play() promise
  }

  function playBlob(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const finish = (err) => {
        player.onended = player.onerror = null;
        URL.revokeObjectURL(url);
        err ? reject(err) : resolve();
      };
      player.onended = () => finish();
      player.onerror = () => finish(new Error("The audio could not be played."));
      player.src = url;
      player.play().catch((err) => finish(err));
    });
  }

  async function speak(text) {
    const res = await postJson("/speak", { text, language: state.language });
    await playBlob(await res.blob());
  }

  // ---------- microphone ----------
  function micErrorMessage(err) {
    if (!navigator.mediaDevices || !window.MediaRecorder || !window.isSecureContext) {
      return "This browser cannot record audio here. Use a current browser over HTTPS (or localhost).";
    }
    switch (err && err.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Microphone access was blocked. Allow the microphone in your browser's site settings, then press Start again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No microphone was found. Connect one and press Start again.";
      case "NotReadableError":
        return "The microphone is in use by another app. Close it and press Start again.";
      default:
        return `Could not access the microphone (${(err && err.name) || "unknown error"}).`;
    }
  }

  async function openMic() {
    if (!navigator.mediaDevices || !window.MediaRecorder || !window.isSecureContext) {
      throw new Error("unsupported");
    }
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }

  function closeMic() {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }

  function pickMimeType() {
    const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
    return options.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  }

  function beginRecording() {
    if (state.phase !== "ready" || !state.stream) return false;
    clearError();
    el.transcriptCard.hidden = el.resultCard.hidden = true;
    const run = state.run;
    const mimeType = pickMimeType();
    let recorder;
    try {
      recorder = new MediaRecorder(state.stream, mimeType ? { mimeType } : undefined);
    } catch (err) {
      showError(micErrorMessage(err));
      return false;
    }
    state.recorder = recorder;
    state.chunks = [];
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) state.chunks.push(e.data); };
    recorder.onstop = () => onRecordingStopped(run, recorder.mimeType || mimeType || "audio/webm");
    recorder.start();
    state.recStart = performance.now();
    state.autoStop = setTimeout(stopRecording, MAX_RECORD_MS);
    setPhase("recording", "Listening… tap the mic again to stop.");
    return true;
  }

  function stopRecording() {
    clearTimeout(state.autoStop);
    if (state.recorder && state.recorder.state === "recording") state.recorder.stop();
  }

  async function onRecordingStopped(run, mimeType) {
    if (run !== state.run) return;
    const duration = performance.now() - state.recStart;
    const blob = new Blob(state.chunks, { type: mimeType });
    state.recorder = null;
    if (duration < MIN_RECORD_MS || blob.size === 0) {
      setPhase("ready", "That was too short. Tap the mic and answer again.");
      return;
    }
    await transcribeAndCheck(run, blob);
  }

  // ---------- quiz flow ----------
  async function loadQuestion(run) {
    clearError();
    setPhase("loading", "Thinking of a question…");
    let question;
    try {
      const res = await postJson("/question", {
        topic: state.topic,
        language: state.language,
        previous_questions: state.asked,
      });
      ({ question } = await res.json());
    } catch (err) {
      if (run !== state.run) return;
      setPhase("idle", "Could not get a question.");
      showError(err.message, () => loadQuestion(state.run));
      return;
    }
    if (run !== state.run) return;

    state.asked.push(question);
    state.current = question;
    el.question.textContent = question;
    el.questionCard.hidden = false;
    el.transcriptCard.hidden = el.resultCard.hidden = true;

    setPhase("speaking", "Speaking the question…");
    try {
      await speak(question);
    } catch (err) {
      if (run !== state.run) return;
      showError(`Could not play the question aloud: ${err.message}`);
    }
    if (run !== state.run) return;
    setPhase("ready", "Your turn — tap the mic (or hold it) and answer.");
  }

  async function transcribeAndCheck(run, blob) {
    setPhase("transcribing", "Transcribing your answer…");
    let text;
    try {
      const form = new FormData();
      const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
      form.append("audio", blob, `answer.${ext}`);
      form.append("language", state.language);
      const res = await api("/transcribe", { method: "POST", body: form });
      ({ text } = await res.json());
    } catch (err) {
      if (run !== state.run) return;
      setPhase("ready", "Tap the mic to try answering again.");
      showError(err.message);
      return;
    }
    if (run !== state.run) return;

    el.transcript.textContent = text;
    el.transcriptCard.hidden = false;
    await checkAnswer(run, text);
  }

  async function checkAnswer(run, answer) {
    clearError();
    setPhase("checking", "Checking your answer…");
    let result;
    try {
      const res = await postJson("/check", {
        question: state.current,
        answer,
        language: state.language,
      });
      result = await res.json();
    } catch (err) {
      if (run !== state.run) return;
      setPhase("idle", "Could not check the answer.");
      showError(err.message, () => checkAnswer(state.run, answer));
      return;
    }
    if (run !== state.run) return;

    state.answered += 1;
    if (result.correct) state.correct += 1;
    updateScore();
    el.resultCard.className = `card result ${result.correct ? "correct" : "wrong"}`;
    el.resultLabel.textContent = result.correct ? "Correct" : "Wrong";
    el.resultFeedback.textContent = result.feedback;
    el.resultCard.hidden = false;

    setPhase("speaking", "Speaking the feedback…");
    try {
      await speak(result.feedback);
    } catch (err) {
      if (run !== state.run) return;
      showError(`Could not play the feedback aloud: ${err.message}`);
    }
    if (run !== state.run) return;
    await new Promise((r) => setTimeout(r, NEXT_DELAY_MS));
    if (run !== state.run) return;
    loadQuestion(run);
  }

  function endQuiz(message) {
    state.run += 1;
    clearTimeout(state.autoStop);
    if (state.recorder && state.recorder.state === "recording") {
      state.recorder.onstop = null;
      state.recorder.stop();
    }
    state.recorder = null;
    stopPlayback();
    closeMic();
    lockSetup(false);
    setPhase("idle", message);
  }

  async function startQuiz() {
    const topic = el.topic.value.trim();
    if (!topic) {
      el.topic.focus();
      return;
    }
    clearError();
    resetCards();
    state.run += 1;
    const run = state.run;
    Object.assign(state, {
      topic, language: el.language.value, asked: [], current: null, correct: 0, answered: 0,
    });
    updateScore();

    // Inside the click gesture: unlock audio playback, then ask for the microphone.
    player.src = SILENT_WAV;
    player.play().catch(() => {});
    lockSetup(true);
    setPhase("loading", "Allow microphone access if your browser asks…");
    try {
      await openMic();
    } catch (err) {
      if (run !== state.run) return;
      endQuiz("Microphone not available.");
      showError(micErrorMessage(err));
      return;
    }
    if (run !== state.run) {
      closeMic();
      return;
    }
    loadQuestion(run);
  }

  // ---------- events ----------
  el.setup.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!el.start.classList.contains("stop")) {
      startQuiz();
    } else {
      endQuiz("Quiz stopped. Press Start for a new one.");
    }
  });

  el.retry.addEventListener("click", () => {
    const fn = state.retryFn;
    if (fn) fn();
  });

  // Tap to start / tap to stop, or hold while talking and release to stop.
  el.mic.addEventListener("pointerdown", (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    e.preventDefault();
    if (state.phase === "ready") {
      el.mic.setPointerCapture(e.pointerId);
      state.pressActive = beginRecording();
    } else if (state.phase === "recording") {
      stopRecording();
    }
  });

  const release = () => {
    if (!state.pressActive) return;
    state.pressActive = false;
    if (state.phase === "recording" && performance.now() - state.recStart >= HOLD_MS) {
      stopRecording(); // it was a hold, not a tap
    }
  };
  el.mic.addEventListener("pointerup", release);
  el.mic.addEventListener("pointercancel", release);

  // Keyboard: Space/Enter toggles recording.
  el.mic.addEventListener("keydown", (e) => {
    if (e.key !== " " && e.key !== "Enter") return;
    e.preventDefault();
    if (e.repeat) return;
    if (state.phase === "ready") beginRecording();
    else if (state.phase === "recording") stopRecording();
  });

  el.mic.addEventListener("contextmenu", (e) => e.preventDefault());
  window.addEventListener("pagehide", closeMic);

  updateScore();
  lockSetup(false);
  setPhase("idle");
})();
