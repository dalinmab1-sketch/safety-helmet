const byId = (id) => document.getElementById(id);
const navLinks = [...document.querySelectorAll("[data-nav-target]")];
const navSections = navLinks
  .map((link) => byId(link.dataset.navTarget))
  .filter(Boolean);

const touchModeButton = byId("touch-mode-button");
const touchModeStorageKey = "hardhat-touch-mode";
const gestureButton = byId("gesture-button");
const gestureStatus = byId("gesture-status");
const gestureCanvas = document.createElement("canvas");
const gestureContext = gestureCanvas.getContext("2d", {willReadFrequently: true});
const gestureWasmUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const gestureModuleUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const recordingGestureHoldMs = 850;
const gestureReleaseMs = 650;
const swipeDistanceThreshold = 0.12;
const swipeWindowMs = 850;

let gestureEnabled = false;
let gestureRecognizer = null;
let gestureInitialization = null;
let gestureRuntimeStatus = "Loading";
let gestureRuntimeError = "";
let gestureLoopTimer = null;
let lastGestureTimestamp = 0;
let currentGestureLabel = "Loading hand tracking…";
let currentGestureConfidence = 0;
let recordingGestureCandidate = null;
let recordingGestureStartedAt = 0;
let recordingGestureLatchedFor = null;
let recordingGestureReleaseAt = 0;
let swipeSamples = [];
let swipeLatched = false;
let nonPointingSince = 0;
let lastStatusMessageAt = 0;

function setTouchMode(enabled) {
  document.body.classList.toggle("touch-mode", enabled);
  touchModeButton.setAttribute("aria-pressed", String(enabled));
  touchModeButton.textContent = `Touch mode: ${enabled ? "On" : "Off"}`;
  window.localStorage.setItem(touchModeStorageKey, String(enabled));
}

setTouchMode(window.localStorage.getItem(touchModeStorageKey) === "true");
touchModeButton.addEventListener("click", () => {
  setTouchMode(!document.body.classList.contains("touch-mode"));
});

function setActiveNavigation(sectionId) {
  navLinks.forEach((link) => {
    const active = link.dataset.navTarget === sectionId;
    link.classList.toggle("active", active);
    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}

function syncNavigationToHash() {
  const sectionId = window.location.hash.slice(1);
  setActiveNavigation(navSections.some((section) => section.id === sectionId) ? sectionId : "overview");
}

function syncNavigationToViewport() {
  const anchor = window.innerHeight * 0.28;
  const closestSection = navSections.reduce((closest, section) => {
    const rect = section.getBoundingClientRect();
    const distance = Math.abs(rect.top + rect.height / 2 - anchor);
    return distance < closest.distance ? {id: section.id, distance} : closest;
  }, {id: "overview", distance: Number.POSITIVE_INFINITY});
  setActiveNavigation(closestSection.id);
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => setActiveNavigation(link.dataset.navTarget));
});

window.addEventListener("hashchange", syncNavigationToHash);
syncNavigationToHash();
window.addEventListener("scroll", syncNavigationToViewport, {passive: true});
window.addEventListener("resize", syncNavigationToViewport);

function setHealth(id, text, state) {
  byId(id).textContent = text;
  const mark = byId(`${id}-mark`);
  mark.classList.toggle("good", state === "good");
  mark.classList.toggle("bad", state === "bad");
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function recordingCard(item) {
  const card = document.createElement("div");
  card.className = "recording-item";
  const file = document.createElement("div");
  file.className = "recording-file";
  const name = document.createElement("strong");
  name.textContent = item.name;
  const detail = document.createElement("small");
  detail.textContent = `${formatSize(item.size_bytes)} · ${new Date(item.modified * 1000).toLocaleString()}`;
  file.append(name, detail);
  const link = document.createElement("a");
  link.className = "download-link";
  link.href = `/recordings/${encodeURIComponent(item.name)}`;
  link.textContent = "Download";
  card.append(file, link);
  return card;
}

async function updateRecordings() {
  const response = await fetch("/api/recordings");
  if (!response.ok) throw new Error("Could not load saved videos.");
  const files = await response.json();
  for (const id of ["recordings-list", "latest-recording"]) {
    const container = byId(id);
    container.replaceChildren();
    const entries = id === "latest-recording" ? files.slice(0, 1) : files;
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No saved recordings yet.";
      container.append(empty);
    } else {
      entries.forEach((item) => container.append(recordingCard(item)));
    }
  }
}

async function updateStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) throw new Error("Could not read camera status.");
  const state = await response.json();
  byId("camera-label").textContent = state.demo_mode ? "Demo simulation" : state.camera_connected ? "Camera online" : (state.error ? "Camera offline" : "Connecting");
  byId("camera-dot").className = `system-dot ${state.demo_mode || state.camera_connected ? "online" : state.error ? "offline" : ""}`;
  byId("people-count").textContent = state.people;
  byId("helmet-count").textContent = state.helmet;
  byId("no-helmet-count").textContent = state.no_helmet;
  byId("unknown-count").textContent = state.unknown;
  byId("current-status").textContent = state.status;
  byId("feed-meta").textContent = `${state.resolution} · ${state.fps} FPS`;
  byId("recording-badge").classList.toggle("paused", !state.recording);
  byId("recording-badge").lastChild.textContent = state.recording ? " REC" : " PAUSED";
  byId("record-button").classList.toggle("paused", !state.recording_requested);
  byId("record-button").lastElementChild.textContent = state.recording_requested ? "Stop recording" : "Start recording";

  const cameraState = state.demo_mode || state.camera_connected ? "good" : state.error ? "bad" : "";
  setHealth("health-camera", state.demo_mode ? "Demo source" : state.camera_connected ? "Connected" : state.error ? "Unavailable" : "Starting", cameraState);
  setHealth("health-model", state.demo_mode ? "Simulated" : state.model_ready ? "Ready" : "Required", state.demo_mode || state.model_ready ? "good" : "bad");
  setHealth("health-record", state.recording ? "Recording" : state.recording_requested ? "Starting" : "Paused", state.recording ? "good" : "");
  setGestureEnabled(state.gesture_control_enabled);
  gestureButton.textContent = state.gesture_control_enabled ? "Turn gestures off" : "Turn gestures on";
  gestureButton.setAttribute("aria-pressed", String(state.gesture_control_enabled));
  updateGestureIndicator();
  byId("model-title").textContent = state.demo_mode ? state.status : state.model_ready ? state.status : state.error || "Camera preview is ready; helmet detection is offline";
  byId("model-message").textContent = state.demo_mode ? "Synthetic sample detections are shown; this is not a live camera or real helmet check." : state.model_message;
  byId("model-chip").textContent = state.demo_mode ? "DEMO MODE" : state.model_ready ? "MODEL READY" : "MODEL REQUIRED";
  byId("system-error").textContent = state.error || gestureRuntimeError || state.gesture_error || state.model_message;
  byId("feed-error").hidden = !state.error;
  byId("feed-error").textContent = state.error || "";
}

byId("record-button").addEventListener("click", async () => {
  const currentlyRequested = byId("record-button").classList.contains("paused");
  const response = await fetch("/api/recording", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled: currentlyRequested}),
  });
  if (!response.ok) {
    byId("system-error").textContent = "Could not change recording state.";
    return;
  }
  await updateStatus();
  await updateRecordings();
});

gestureButton.addEventListener("click", async () => {
  const currentlyEnabled = gestureButton.getAttribute("aria-pressed") === "true";
  const response = await fetch("/api/gesture-control", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled: !currentlyEnabled}),
  });
  if (!response.ok) {
    byId("system-error").textContent = "Could not change hand gesture controls.";
    return;
  }
  const state = await response.json();
  setGestureEnabled(state.gesture_control_enabled);
  gestureButton.textContent = state.gesture_control_enabled ? "Turn gestures off" : "Turn gestures on";
  gestureButton.setAttribute("aria-pressed", String(state.gesture_control_enabled));
  updateGestureIndicator();
});

function updateGestureIndicator() {
  if (!gestureEnabled) {
    gestureStatus.textContent = "Gesture control off";
    setHealth("health-gesture", "Off", "");
    return;
  }
  const runtimeLabel = gestureRuntimeStatus === "Unavailable" ? "Unavailable" : gestureRuntimeStatus;
  setHealth("health-gesture", runtimeLabel, gestureRuntimeStatus === "Enabled" ? "good" : gestureRuntimeStatus === "Unavailable" ? "bad" : "");
  if (currentGestureLabel && performance.now() - lastStatusMessageAt < 1300) {
    gestureStatus.textContent = currentGestureConfidence > 0
      ? `${currentGestureLabel} · ${Math.round(currentGestureConfidence * 100)}%`
      : currentGestureLabel;
  } else {
    gestureStatus.textContent = gestureRuntimeStatus === "Unavailable"
      ? "Hand tracking unavailable"
      : gestureRuntimeStatus === "Loading"
        ? "Loading hand tracking…"
        : "Waiting for hand gesture…";
  }
}

function setGestureEnabled(enabled) {
  if (gestureEnabled === enabled) return;
  gestureEnabled = enabled;
  if (!enabled) {
    if (gestureLoopTimer !== null) window.clearTimeout(gestureLoopTimer);
    gestureLoopTimer = null;
    resetGestureState();
    updateGestureIndicator();
    return;
  }
  if (gestureRecognizer) {
    gestureRuntimeStatus = "Enabled";
    gestureRuntimeError = "";
    runGestureDetection();
  } else if (!gestureInitialization) {
    gestureRuntimeStatus = "Loading";
    gestureRuntimeError = "";
    gestureInitialization = initializeGestureRecognizer()
      .catch((error) => {
        gestureRuntimeStatus = "Unavailable";
        gestureRuntimeError = error instanceof Error ? error.message : String(error);
        byId("system-error").textContent = `Hand tracking could not start: ${gestureRuntimeError}`;
      })
      .finally(() => {
        gestureInitialization = null;
        updateGestureIndicator();
      });
  }
  updateGestureIndicator();
}

async function initializeGestureRecognizer() {
  const {FilesetResolver, GestureRecognizer} = await import(gestureModuleUrl);
  const vision = await FilesetResolver.forVisionTasks(gestureWasmUrl);
  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {modelAssetPath: "/gesture-model", delegate: "CPU"},
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
  if (gestureEnabled) {
    gestureRuntimeStatus = "Enabled";
    currentGestureLabel = "Waiting for hand gesture…";
    runGestureDetection();
  }
}

function resetGestureState() {
  recordingGestureCandidate = null;
  recordingGestureStartedAt = 0;
  recordingGestureLatchedFor = null;
  recordingGestureReleaseAt = 0;
  swipeSamples = [];
  swipeLatched = false;
  nonPointingSince = 0;
  currentGestureLabel = "";
  currentGestureConfidence = 0;
}

function distanceBetween(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function isIndexOnlyPoint(landmarks, categoryName) {
  if (categoryName === "Pointing_Up") return true;
  const wrist = landmarks[0];
  const fingerPairs = [[8, 6], [12, 10], [16, 14], [20, 18]];
  const extended = fingerPairs.map(([tip, pip]) =>
    distanceBetween(landmarks[tip], wrist) > distanceBetween(landmarks[pip], wrist) * 1.08
  );
  return extended[0] && !extended[1] && !extended[2] && !extended[3];
}

function showGesture(label, confidence = 0) {
  currentGestureLabel = label;
  currentGestureConfidence = confidence;
  lastStatusMessageAt = performance.now();
  updateGestureIndicator();
}

function findSwipe(landmarks, now, categoryName) {
  if (!landmarks || !isIndexOnlyPoint(landmarks, categoryName)) {
    if (!nonPointingSince) nonPointingSince = now;
    if (now - nonPointingSince >= gestureReleaseMs) {
      swipeLatched = false;
      swipeSamples = [];
    }
    return null;
  }
  nonPointingSince = 0;
  if (swipeLatched) return null;

  const y = landmarks[8].y;
  swipeSamples.push({y, time: now});
  swipeSamples = swipeSamples.filter((sample) => now - sample.time <= swipeWindowMs);
  if (swipeSamples.length < 2) return null;

  const first = swipeSamples[0];
  const delta = y - first.y;
  if (Math.abs(delta) < swipeDistanceThreshold) return null;
  swipeLatched = true;
  swipeSamples = [];
  return delta < 0 ? "Swipe Up" : "Swipe Down";
}

async function setRecordingFromGesture(enabled) {
  const response = await fetch("/api/recording", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled}),
  });
  if (!response.ok) throw new Error("Could not change recording state from the gesture.");
  await response.json();
  await updateStatus();
  await updateRecordings();
}

function handleRecordingGesture(label, confidence, now) {
  if (label !== "Thumbs Up" && label !== "Open Palm") {
    if (recordingGestureCandidate && !recordingGestureReleaseAt) recordingGestureReleaseAt = now;
    if (recordingGestureReleaseAt && now - recordingGestureReleaseAt >= gestureReleaseMs) {
      recordingGestureCandidate = null;
      recordingGestureStartedAt = 0;
      recordingGestureLatchedFor = null;
      recordingGestureReleaseAt = 0;
    }
    return;
  }
  if (recordingGestureCandidate !== label) {
    recordingGestureCandidate = label;
    recordingGestureStartedAt = now;
    recordingGestureLatchedFor = null;
  }
  recordingGestureReleaseAt = 0;
  if (recordingGestureLatchedFor === label || now - recordingGestureStartedAt < recordingGestureHoldMs) return;

  recordingGestureLatchedFor = label;
  const shouldRecord = label === "Thumbs Up";
  showGesture(label, confidence);
  setRecordingFromGesture(shouldRecord).catch((error) => {
    byId("system-error").textContent = error.message;
  });
}

function processGestureFrame() {
  if (!gestureEnabled || !gestureRecognizer) return;
  try {
    const feed = byId("camera-feed");
    if (!gestureContext) throw new Error("This browser cannot create a hand-tracking canvas.");
    if (feed.complete && feed.naturalWidth > 0 && feed.naturalHeight > 0) {
      if (gestureCanvas.width !== feed.naturalWidth || gestureCanvas.height !== feed.naturalHeight) {
        gestureCanvas.width = feed.naturalWidth;
        gestureCanvas.height = feed.naturalHeight;
      }
      gestureContext.drawImage(feed, 0, 0, gestureCanvas.width, gestureCanvas.height);
      const now = performance.now();
      const timestamp = Math.max(now, lastGestureTimestamp + 1);
      lastGestureTimestamp = timestamp;
      const result = gestureRecognizer.recognizeForVideo(gestureCanvas, timestamp);
      const categories = result.gestures?.[0] || [];
      const category = categories.reduce((best, item) => item.score > (best?.score || 0) ? item : best, null);
      const landmarks = result.landmarks?.[0];
      const confidence = category?.score || 0;
      let label = null;

      if (category && confidence >= 0.65) {
        if (category.categoryName === "Thumb_Up") label = "Thumbs Up";
        if (category.categoryName === "Open_Palm") label = "Open Palm";
      }
      const swipe = findSwipe(label ? null : landmarks, now, category?.categoryName);
      if (label) {
        showGesture(label, confidence);
        handleRecordingGesture(label, confidence, now);
      } else {
        handleRecordingGesture(null, 0, now);
        if (swipe) {
          showGesture(swipe);
          const distance = Math.round(window.innerHeight * 0.65) * (swipe === "Swipe Up" ? -1 : 1);
          window.scrollBy({top: distance, behavior: "smooth"});
        }
      }
    }
  } catch (error) {
    gestureRuntimeStatus = "Unavailable";
    gestureRuntimeError = error instanceof Error ? error.message : String(error);
    byId("system-error").textContent = `Hand tracking stopped: ${gestureRuntimeError}`;
    gestureLoopTimer = null;
    updateGestureIndicator();
    return;
  }
  gestureLoopTimer = window.setTimeout(processGestureFrame, 100);
}

function runGestureDetection() {
  if (!gestureEnabled || !gestureRecognizer || gestureLoopTimer !== null) return;
  processGestureFrame();
}

async function refreshDashboard() {
  try {
    await Promise.all([updateStatus(), updateRecordings()]);
  } catch (error) {
    byId("system-error").textContent = error.message;
  }
}

refreshDashboard();
window.setInterval(refreshDashboard, 2000);
