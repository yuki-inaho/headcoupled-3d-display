import { PointCloudRenderer } from "./renderer.js";

const byId = (id) => document.getElementById(id);
const setText = (id, value) => { const element = byId(id); if (element) element.textContent = value; };
const format = (value, digits = 2) => Number(value).toFixed(digits);
const wsUrl = (path) => `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}`;

// The physical display's width/height ratio (config/hardware_profile.*.json's
// display.width_m / height_m). Used only to warn when the browser viewport's aspect
// ratio has drifted from the real screen it is meant to represent.
const PHYSICAL_ASPECT_RATIO = 0.596 / 0.335;
const ASPECT_TOLERANCE = 0.02;

let renderer = null;
let lastCameraUrl = null;
let poseReconnectTimer = null;
let cameraReconnectTimer = null;

function statusChip(kind, text) {
  const element = byId("tracking-status");
  element.dataset.kind = kind;
  element.textContent = text;
}

async function loadProfile() {
  const response = await fetch("/api/profile", { cache: "no-store" });
  if (!response.ok) throw new Error(`Profile request failed: ${response.status}`);
  const payload = await response.json();
  const profile = payload.hardware_profile;
  const mount = payload.mount_summary;
  setText("profile-id", profile.profile_id);
  setText("profile-provenance", profile.provenance);
  setText("display-size", `${format(profile.display.width_m * 100, 1)} × ${format(profile.display.height_m * 100, 1)} cm`);
  setText("mount-height", `${format(mount.height_above_center_cm, 1)} cm`);
  setText("mount-pitch", `${format(mount.pitch_down_deg, 1)}°`);
  setText("mount-total-tilt", `${format(mount.total_axis_tilt_from_display_normal_deg, 1)}°`);
  setText("mount-forward", `${format(mount.forward_offset_cm, 1)} cm`);
  setText("mount-horizontal", `${format(mount.horizontal_offset_cm, 1)} cm`);
  setText("mount-centered", mount.horizontally_centered ? "中央" : "非中央");
  const warning = byId("profile-warning");
  if (payload.warning) {
    warning.hidden = false;
    warning.textContent = `注意: ${payload.warning}`;
    document.body.dataset.demoProfile = "true";
  }
  const scene = payload.scene_profile;
  if (!scene) throw new Error("Profile response carries no scene_profile");
  renderer = new PointCloudRenderer(byId("gl-canvas"), profile.display, scene);
  renderer.setViewMode(document.body.dataset.viewMode || "verification");
  // Exposed only so the Playwright E2E suite can drive setEye()/read scheduler debug
  // counters directly, without depending on live websocket timing to exercise bursts.
  // Not used by any production code path.
  window.__headcoupledRenderer = renderer;
  const info = await renderer.load(scene.point_cloud_asset);
  // Diagnostics hook. The renderer is module-scoped, and end-to-end tests need the
  // browser-side draw latency it measures; exposing a read-only summary function is
  // cheaper and less fragile than mirroring every percentile into a data attribute.
  window.headcoupledTimingSummary = () => renderer?.timingSummary() ?? null;
  setText("renderer-status", `${info.mode} / ${info.pointCount.toLocaleString()} points`);
  updateSceneVerificationHud(scene);
}

function updateSceneVerificationHud(scene) {
  setText("hud-anchor-z", `${format(scene.anchor_display_m[2], 3)} m`);
  setText("hud-grid-spacing", `${format(scene.grid_spacing_m * 100, 1)} cm`);
  setText("hud-back-wall-depth", `${format(scene.back_wall_z_m, 3)} m`);
  const canvas = byId("gl-canvas");
  const centerRaw = canvas.dataset.modelCenterDisplayM;
  if (centerRaw) {
    const center = JSON.parse(centerRaw);
    setText("hud-aabb-center", `${center.map((value) => format(value, 3)).join(", ")} m`);
  }
}

function updatePreviewResolutionReadout() {
  const image = byId("camera-preview");
  if (image.naturalWidth > 0 && image.naturalHeight > 0) {
    setText("hud-preview-resolution", `${image.naturalWidth} × ${image.naturalHeight}`);
  }
}

function connectPose() {
  clearTimeout(poseReconnectTimer);
  const socket = new WebSocket(wsUrl("/ws/pose"));
  socket.addEventListener("open", () => statusChip("ok", "追跡接続中"));
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type !== "tracking") return;
    const pose = message.payload;
    renderer?.setEye(pose.cyclopean_eye_display_m, pose.sequence);
    byId("pose-sequence").dataset.sequence = String(pose.sequence);
    setText("pose-sequence", `#${pose.sequence}`);
    setText("confidence", format(pose.confidence, 2));
    setText("tracking-fps", `${format(pose.tracking_fps, 1)} fps`);
    setText("inference-ms", `${format(pose.inference_ms, 1)} ms`);
    setText("stability", pose.stable ? "安定" : "移動中");
    const [x, y, z] = pose.cyclopean_eye_display_m;
    setText("eye-position", `${format(x, 3)}, ${format(y, 3)}, ${format(z, 3)} m`);
    const sourceLabel = {
      synthetic: "合成追跡",
      facemesh: "顔追跡（ライブ）",
      replay: "顔追跡（録画再生）",
      ipc: "顔追跡（IPCライブ）",
    }[pose.source] || "追跡入力";
    statusChip(pose.confidence >= 0.75 ? "ok" : "warn", sourceLabel);
  });
  socket.addEventListener("close", () => {
    statusChip("warn", "再接続中");
    poseReconnectTimer = setTimeout(connectPose, 800);
  });
  socket.addEventListener("error", () => socket.close());
}

function connectCamera() {
  clearTimeout(cameraReconnectTimer);
  const socket = new WebSocket(wsUrl("/ws/camera"));
  socket.binaryType = "arraybuffer";
  socket.addEventListener("message", (event) => {
    const blob = new Blob([event.data], { type: "image/jpeg" });
    const nextUrl = URL.createObjectURL(blob);
    byId("camera-preview").src = nextUrl;
    if (lastCameraUrl) URL.revokeObjectURL(lastCameraUrl);
    lastCameraUrl = nextUrl;
  });
  socket.addEventListener("close", () => {
    cameraReconnectTimer = setTimeout(connectCamera, 1000);
  });
  socket.addEventListener("error", () => socket.close());
}

async function runSyntheticCalibration() {
  const button = byId("calibrate-synthetic");
  const result = byId("calibration-result");
  button.disabled = true;
  button.textContent = "較正を計算中…";
  result.dataset.status = "running";
  result.textContent = "36本の合成頭部レイから外部姿勢を最適化しています。";
  try {
    const response = await fetch("/api/calibration/synthetic", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    const metrics = payload.result.metrics;
    const comparison = payload.result.comparison_to_ground_truth;
    result.dataset.status = payload.status;
    result.innerHTML = `
      <strong>較正成功</strong>
      <span>平均レイ残差 ${format(metrics.mean_point_to_ray_error_mm, 2)} mm</span>
      <span>高さ誤差 ${format(comparison.height_error_mm, 2)} mm</span>
      <span>ピッチ誤差 ${format(comparison.pitch_error_deg, 2)}°</span>
      <span>${metrics.sample_count}標本 / ${metrics.unique_target_count}画面点</span>`;
  } catch (error) {
    result.dataset.status = "failed";
    result.textContent = `較正失敗: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "合成較正を実行";
  }
}

function setViewMode(mode) {
  document.body.dataset.viewMode = mode;
  renderer?.setViewMode(mode);
  const button = byId("mode-toggle-button");
  button.setAttribute("aria-pressed", String(mode === "immersive"));
  button.textContent = mode === "immersive" ? "検証モードへ" : "没入モードへ";
}

/**
 * Compare the canvas's on-screen aspect ratio against the physical display's. Computed
 * on every resize/fullscreen change regardless of mode, because a wrong aspect ratio
 * silently breaks the off-axis projection's geometric correctness.
 */
function updateAspectState() {
  const canvas = byId("gl-canvas");
  const rect = canvas.getBoundingClientRect();
  const warning = byId("aspect-warning");
  if (rect.height <= 0) return;
  const viewportAspect = rect.width / rect.height;
  const relativeError = Math.abs(viewportAspect - PHYSICAL_ASPECT_RATIO) / PHYSICAL_ASPECT_RATIO;
  const aspectOk = relativeError <= ASPECT_TOLERANCE;
  document.body.dataset.aspectOk = String(aspectOk);
  const isFullscreen = Boolean(document.fullscreenElement);
  // "Physically verified" requires both fullscreen (so the canvas actually occupies the
  // real display) and a matching aspect ratio; a windowed browser tab proves nothing
  // about the physical projection no matter how close its aspect ratio happens to be.
  const physicallyVerified = isFullscreen && aspectOk;
  document.body.dataset.physicalProjectionVerified = String(physicallyVerified);
  if (physicallyVerified) {
    warning.hidden = true;
    return;
  }
  warning.hidden = false;
  warning.textContent = isFullscreen
    ? `警告: 全画面表示のアスペクト比(${viewportAspect.toFixed(3)})が物理ディスプレイ(${PHYSICAL_ASPECT_RATIO.toFixed(3)})と${(relativeError * 100).toFixed(1)}%ずれています。表示配置を確認してください。`
    : "物理投影は未検証です。全画面表示にすると実ディスプレイとの一致を確認できます。";
}

function setupControls() {
  byId("calibrate-synthetic").addEventListener("click", runSyntheticCalibration);
  byId("fullscreen-button").addEventListener("click", async () => {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  });
  byId("camera-toggle").addEventListener("click", () => {
    const panel = byId("camera-panel");
    panel.classList.toggle("collapsed");
    byId("camera-toggle").textContent = panel.classList.contains("collapsed") ? "映像を表示" : "映像を隠す";
  });
  byId("mode-toggle-button").addEventListener("click", () => {
    setViewMode(document.body.dataset.viewMode === "immersive" ? "verification" : "immersive");
  });
  byId("camera-preview").addEventListener("load", updatePreviewResolutionReadout);
  document.addEventListener("fullscreenchange", updateAspectState);
  window.addEventListener("resize", updateAspectState);
}

async function main() {
  setupControls();
  try {
    await loadProfile();
    connectPose();
    connectCamera();
    updateAspectState();
    document.body.dataset.ready = "true";
  } catch (error) {
    statusChip("error", "初期化失敗");
    setText("renderer-status", error.message);
    console.error(error);
  }
}

main();
