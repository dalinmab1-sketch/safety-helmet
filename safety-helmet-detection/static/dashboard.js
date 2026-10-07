const byId = (id) => document.getElementById(id);
const navLinks = [...document.querySelectorAll("[data-nav-target]")];
const navSections = navLinks
  .map((link) => byId(link.dataset.navTarget))
  .filter(Boolean);

const touchModeButton = byId("touch-mode-button");
const touchModeStorageKey = "hardhat-touch-mode";

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
  setHealth("health-gesture", state.gesture_control_enabled ? state.gesture_error ? "Unavailable" : "Enabled" : "Off", state.gesture_control_enabled && !state.gesture_error ? "good" : "");
  byId("gesture-button").textContent = state.gesture_control_enabled ? "Turn gestures off" : "Turn gestures on";
  const gestureLabel = state.gesture_action || state.gesture || "Waiting for hand gesture";
  byId("gesture-status").textContent = state.gesture_confidence ? `${gestureLabel.replaceAll("_", " ")} · ${Math.round(state.gesture_confidence * 100)}%` : gestureLabel.replaceAll("_", " ");
  byId("model-title").textContent = state.demo_mode ? state.status : state.model_ready ? state.status : state.error || "Camera preview is ready; helmet detection is offline";
  byId("model-message").textContent = state.demo_mode ? "Synthetic sample detections are shown; this is not a live camera or real helmet check." : state.model_message;
  byId("model-chip").textContent = state.demo_mode ? "DEMO MODE" : state.model_ready ? "MODEL READY" : "MODEL REQUIRED";
  byId("system-error").textContent = state.error || state.gesture_error || state.model_message;
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

byId("gesture-button").addEventListener("click", async () => {
  const currentlyEnabled = byId("gesture-button").textContent === "Turn gestures off";
  const response = await fetch("/api/gesture-control", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled: !currentlyEnabled}),
  });
  if (!response.ok) {
    byId("system-error").textContent = "Could not change hand gesture controls.";
    return;
  }
  await updateStatus();
});

async function refreshDashboard() {
  try {
    await Promise.all([updateStatus(), updateRecordings()]);
  } catch (error) {
    byId("system-error").textContent = error.message;
  }
}

refreshDashboard();
window.setInterval(refreshDashboard, 2000);
