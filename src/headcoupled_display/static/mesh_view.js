/**
 * Draw the recogniser's dense face mesh instead of the camera image.
 *
 * The point of this view is to show what the tracker actually sees, without putting a
 * picture of the operator's face on screen. Everything drawn here comes from the mesh
 * lane, so nothing on this canvas depends on the preview lane being connected.
 */

const MESH_MAGIC = "HC3M";
const MESH_VERSION = 1;
const MESH_HEADER_BYTES = 16;
const ACCEPTED_POINT_COUNTS = new Set([468, 478]);

// Recognition happens at this resolution and the packet carries those pixel coordinates
// verbatim, so the mapping into the canvas is a plain scale. The preview lane's 640x360
// is a different, display-only contract and must not be used here.
const RECOGNITION_WIDTH_PX = 1280;
const RECOGNITION_HEIGHT_PX = 720;

// MediaPipe's 478-point mesh appends 10 iris points to the 468-point one: 468-472 are
// one iris, 473-477 the other. Their centres are the two the metric pose uses for eye
// position, which is why they are drawn distinctly rather than as ordinary vertices.
const LEFT_IRIS_CENTRE = 468;
const RIGHT_IRIS_CENTRE = 473;

// The 12 landmarks the control lane carries and the PnP solve consumes. Drawn on top so
// an operator can see the points the pose is actually solved from.
const PNP_INDICES = [1, 6, 33, 133, 362, 263, 61, 291, 199, 168, 94, 4];

const MESH_COLOR = "rgba(120, 190, 235, 0.55)";
const PNP_COLOR = "#57d6b5";
const IRIS_COLOR = "#ffd166";
const IDLE_TEXT_COLOR = "rgba(180, 205, 235, 0.65)";

/**
 * Decode one mesh packet. Mirrors `decode_mesh_packet` in protocol.py; a payload that
 * does not match the declared point count is rejected rather than drawn partially.
 *
 * @param {ArrayBuffer} buffer
 * @returns {{points: Float32Array, sequence: number, pointCount: number}}
 */
export function decodeMeshPacket(buffer) {
  if (buffer.byteLength < MESH_HEADER_BYTES) throw new Error("mesh packet is truncated");
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== MESH_MAGIC) throw new Error(`bad mesh magic ${magic}`);
  const version = view.getUint16(4, false);
  if (version !== MESH_VERSION) throw new Error(`unsupported mesh version ${version}`);
  const pointCount = view.getUint16(6, false);
  if (!ACCEPTED_POINT_COUNTS.has(pointCount)) throw new Error(`bad point count ${pointCount}`);
  // sequence is a uint64; the low 32 bits are plenty for a frame counter and avoid
  // pulling in BigInt arithmetic just to display a number.
  const sequence = view.getUint32(12, false);
  const expected = MESH_HEADER_BYTES + pointCount * 8;
  if (buffer.byteLength !== expected) {
    throw new Error(`mesh declares ${pointCount} points but payload is ${buffer.byteLength}`);
  }
  const points = new Float32Array(pointCount * 2);
  for (let i = 0; i < pointCount * 2; i += 1) {
    points[i] = view.getFloat32(MESH_HEADER_BYTES + i * 4, false);
  }
  return { points, sequence, pointCount };
}

export class MeshView {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    if (!this.context) throw new Error("mesh view needs a 2D canvas context");
    this.latest = null;
    this.frameCount = 0;
    this.canvas.dataset.meshFrameCount = "0";
    this.canvas.dataset.meshPointCount = "0";
    this.drawIdle("入力待ち");
  }

  drawIdle(message) {
    const { context, canvas } = this;
    context.fillStyle = "#03080f";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = IDLE_TEXT_COLOR;
    context.font = "16px system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(message, canvas.width / 2, canvas.height / 2);
  }

  /** @param {ArrayBuffer} buffer one raw mesh packet */
  update(buffer) {
    this.latest = decodeMeshPacket(buffer);
    this.frameCount += 1;
    this.canvas.dataset.meshFrameCount = String(this.frameCount);
    this.canvas.dataset.meshPointCount = String(this.latest.pointCount);
    this.canvas.dataset.meshSequence = String(this.latest.sequence);
    this.draw();
  }

  draw() {
    if (this.latest === null) return;
    const { context, canvas } = this;
    const { points, pointCount } = this.latest;
    const scaleX = canvas.width / RECOGNITION_WIDTH_PX;
    const scaleY = canvas.height / RECOGNITION_HEIGHT_PX;

    context.fillStyle = "#03080f";
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Mirror horizontally so the operator sees themselves the way a mirror shows them.
    // The underlying coordinates are untouched; this is presentation only.
    const px = (index) => canvas.width - points[index * 2] * scaleX;
    const py = (index) => points[index * 2 + 1] * scaleY;

    context.fillStyle = MESH_COLOR;
    for (let i = 0; i < pointCount; i += 1) {
      context.fillRect(px(i) - 0.5, py(i) - 0.5, 1.6, 1.6);
    }

    context.fillStyle = PNP_COLOR;
    for (const index of PNP_INDICES) {
      if (index >= pointCount) continue;
      context.beginPath();
      context.arc(px(index), py(index), 2.4, 0, Math.PI * 2);
      context.fill();
    }

    if (pointCount > RIGHT_IRIS_CENTRE) {
      context.fillStyle = IRIS_COLOR;
      for (const index of [LEFT_IRIS_CENTRE, RIGHT_IRIS_CENTRE]) {
        context.beginPath();
        context.arc(px(index), py(index), 3.2, 0, Math.PI * 2);
        context.fill();
      }
    }
  }
}

/**
 * Keep a MeshView fed from /ws/mesh, reconnecting when the socket drops.
 *
 * The socket is opened regardless of which view is selected: switching to the mesh must
 * show the current face, not wait for the next packet after the click.
 */
export function connectMesh(view, { onStatus } = {}) {
  let reconnectTimer = null;
  const open = () => {
    clearTimeout(reconnectTimer);
    const url = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/mesh`;
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socket.addEventListener("message", (event) => {
      try {
        view.update(event.data);
        onStatus?.("ok");
      } catch (error) {
        // A malformed packet is a producer bug, not a reason to tear down the view.
        console.warn("mesh packet rejected:", error.message);
        onStatus?.("bad-packet");
      }
    });
    socket.addEventListener("close", () => {
      onStatus?.("closed");
      reconnectTimer = setTimeout(open, 1000);
    });
    socket.addEventListener("error", () => socket.close());
  };
  open();
}
