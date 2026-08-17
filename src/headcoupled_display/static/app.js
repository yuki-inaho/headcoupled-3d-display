import { PointCloudRenderer } from "./renderer.js";

const byId = (id) => document.getElementById(id);
const setText = (id, value) => { const element = byId(id); if (element) element.textContent = value; };
const format = (value, digits = 2) => Number(value).toFixed(digits);
const wsUrl = (path) => `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}`;

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
  renderer = new PointCloudRenderer(byId("gl-canvas"), profile.display);
  const info = await renderer.load("/static/assets/bunny.pcd");
  setText("renderer-status", `${info.mode} / ${info.pointCount.toLocaleString()} points`);
}

function connectPose() {
  clearTimeout(poseReconnectTimer);
  const socket = new WebSocket(wsUrl("/ws/pose"));
  socket.addEventListener("open", () => statusChip("ok", "追跡接続中"));
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type !== "tracking") return;
    const pose = message.payload;
    renderer?.setEye(pose.cyclopean_eye_display_m);
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
}

async function main() {
  setupControls();
  try {
    await loadProfile();
    connectPose();
    connectCamera();
    document.body.dataset.ready = "true";
  } catch (error) {
    statusChip("error", "初期化失敗");
    setText("renderer-status", error.message);
    console.error(error);
  }
}

main();
