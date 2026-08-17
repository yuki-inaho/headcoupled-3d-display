/**
 * Draw the tracked head as a rigid template, instead of the camera image.
 *
 * The point of this view is to show what the tracker believes about the head without
 * putting a picture of the operator's face on screen.
 *
 * There used to be a third IPC lane that shipped all 478 recognised landmarks every
 * frame so this panel could draw them. That was removed. `/ws/pose` already carries
 * both eye centres and the forward axis, and those pin down a rigid frame completely,
 * so the same picture can be drawn from the pose the display is already receiving --
 * with nothing sent per frame. The dense lane also sat on the producer's critical path
 * (a synchronous, uncapped POST inside the capture loop), which cost far more than the
 * view was worth: 10.23 FPS measured against a 27-30 FPS recognition rate.
 *
 * What this view can and cannot show, stated plainly: it shows head *position and
 * orientation*, which is exactly what drives the display. It cannot show expression, and
 * it cannot show landmark-level tracking failure -- a template placed by a bad pose still
 * looks like a face. Judge tracking quality by the confidence and score readouts, not by
 * whether this drawing looks plausible.
 */

// Fixed display-frame window, in metres, mapped to the canvas. Deliberately not
// auto-fitted per frame: a view that rescales to whatever it sees would hide the very
// translation this panel exists to show.
const VIEW_X_HALF_SPAN_M = 0.30;
const VIEW_Y_MIN_M = -0.22;
const VIEW_Y_MAX_M = 0.26;

// Depth range used for brightness. The observer normally sits around 0.5-0.7 m out.
const NEAR_M = 0.30;
const FAR_M = 1.00;

const POINT_COLOR = [120, 190, 235];
const PNP_COLOR = "#57d6b5";
const IRIS_COLOR = "#ffd166";
const IDLE_TEXT_COLOR = "rgba(180, 205, 235, 0.65)";
const BACKGROUND = "#03080f";

/** @param {number[]} v */
function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!(n > 1e-9)) return null;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Gram-Schmidt: an orthonormal basis whose third axis is `forward`. */
function basisFrom(rightish, forward) {
  const z = normalize(forward);
  if (z === null) return null;
  const projected = [
    rightish[0] - dot(rightish, z) * z[0],
    rightish[1] - dot(rightish, z) * z[1],
    rightish[2] - dot(rightish, z) * z[2],
  ];
  const x = normalize(projected);
  if (x === null) return null; // eye axis parallel to forward: degenerate, refuse to draw
  return { x, y: cross(z, x), z };
}

/**
 * Rigid transform taking template points in the head frame to display metres.
 *
 * Built from the pose alone: the inter-ocular axis and the forward axis give the
 * rotation, and the cyclopean eye gives the translation. Returns null when the pose is
 * degenerate rather than drawing a head at a made-up orientation.
 *
 * @param {{leftEye:number[], rightEye:number[], forward:number[], cyclopean:number[]}} pose
 * @param {{leftIris:number[], rightIris:number[], neutralForward:number[]}} template
 */
export function rigidFromPose(pose, template) {
  const measured = basisFrom(
    [
      pose.rightEye[0] - pose.leftEye[0],
      pose.rightEye[1] - pose.leftEye[1],
      pose.rightEye[2] - pose.leftEye[2],
    ],
    pose.forward,
  );
  const model = basisFrom(
    [
      template.rightIris[0] - template.leftIris[0],
      template.rightIris[1] - template.leftIris[1],
      template.rightIris[2] - template.leftIris[2],
    ],
    template.neutralForward,
  );
  if (measured === null || model === null) return null;

  // R = B_display * B_head^T, both orthonormal, so this is a rotation with det +1.
  const rotation = [0, 1, 2].map((row) =>
    [0, 1, 2].map(
      (col) =>
        measured.x[row] * model.x[col] +
        measured.y[row] * model.y[col] +
        measured.z[row] * model.z[col],
    ),
  );
  const anchor = [0, 1, 2].map(
    (i) => 0.5 * (template.leftIris[i] + template.rightIris[i]),
  );
  const rotatedAnchor = rotation.map((row) => dot(row, anchor));
  const translation = [0, 1, 2].map((i) => pose.cyclopean[i] - rotatedAnchor[i]);
  return { rotation, translation };
}

export class FaceView {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    if (!this.context) throw new Error("face view needs a 2D canvas context");
    this.template = null;
    this.pose = null;
    this.frameCount = 0;
    this.canvas.dataset.faceFrameCount = "0";
    this.canvas.dataset.facePointCount = "0";
    this.drawIdle("入力待ち");
  }

  drawIdle(message) {
    const { context, canvas } = this;
    context.fillStyle = BACKGROUND;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = IDLE_TEXT_COLOR;
    context.font = "16px system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(message, canvas.width / 2, canvas.height / 2);
  }

  /** Install the template fetched from /api/face-model. Called once. */
  setTemplate(model) {
    const points = model.points_head_m;
    if (!Array.isArray(points) || points.length === 0) {
      throw new Error("face model has no points");
    }
    this.template = {
      points,
      leftIris: points[model.left_iris_index],
      rightIris: points[model.right_iris_index],
      neutralForward: model.neutral_forward_axis_head,
      pnpIndices: model.pnp_indices ?? [],
      leftIrisIndex: model.left_iris_index,
      rightIrisIndex: model.right_iris_index,
      isPersonal: Boolean(model.is_personal),
    };
    this.canvas.dataset.facePointCount = String(points.length);
    this.canvas.dataset.faceModelPersonal = String(this.template.isPersonal);
    if (this.pose !== null) this.draw();
  }

  /**
   * Feed one pose sample. Ignored until the template has arrived.
   * @param {object} state a TrackingState from /ws/pose
   */
  update(state) {
    this.pose = {
      leftEye: state.left_eye_display_m,
      rightEye: state.right_eye_display_m,
      forward: state.head_forward_display,
      cyclopean: state.cyclopean_eye_display_m,
      confidence: state.confidence,
    };
    this.frameCount += 1;
    this.canvas.dataset.faceFrameCount = String(this.frameCount);
    this.canvas.dataset.faceSequence = String(state.sequence);
    this.draw();
  }

  draw() {
    if (this.template === null) return this.drawIdle("顔モデル取得中");
    if (this.pose === null) return this.drawIdle("入力待ち");
    const placement = rigidFromPose(this.pose, this.template);
    if (placement === null) return this.drawIdle("姿勢が不定");

    const { context, canvas, template } = this;
    context.fillStyle = BACKGROUND;
    context.fillRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / (2 * VIEW_X_HALF_SPAN_M);
    const scaleY = canvas.height / (VIEW_Y_MAX_M - VIEW_Y_MIN_M);
    // Mirrored horizontally so the operator sees themselves the way a mirror shows
    // them. Presentation only: the display coordinates themselves are untouched.
    const toCanvas = (p) => [
      canvas.width / 2 - p[0] * scaleX,
      canvas.height - (p[1] - VIEW_Y_MIN_M) * scaleY,
    ];
    const place = (point) => {
      const r = placement.rotation;
      return [0, 1, 2].map(
        (i) =>
          r[i][0] * point[0] +
          r[i][1] * point[1] +
          r[i][2] * point[2] +
          placement.translation[i],
      );
    };

    let drawn = 0;
    for (const point of template.points) {
      const world = place(point);
      const [cx, cy] = toCanvas(world);
      if (cx < -4 || cx > canvas.width + 4 || cy < -4 || cy > canvas.height + 4) continue;
      // Nearer points brighter, so the head reads as a solid object rather than a
      // flat scatter. Without a depth cue the front and back of the head are the same
      // colour and the shape becomes ambiguous.
      const t = Math.min(Math.max((FAR_M - world[2]) / (FAR_M - NEAR_M), 0), 1);
      const level = 0.35 + 0.65 * t;
      context.fillStyle = `rgba(${Math.round(POINT_COLOR[0] * level)}, ${Math.round(
        POINT_COLOR[1] * level,
      )}, ${Math.round(POINT_COLOR[2] * level)}, 0.75)`;
      context.fillRect(cx - 0.5, cy - 0.5, 1.6, 1.6);
      drawn += 1;
    }

    context.fillStyle = PNP_COLOR;
    for (const index of template.pnpIndices) {
      if (index >= template.points.length) continue;
      const [cx, cy] = toCanvas(place(template.points[index]));
      context.beginPath();
      context.arc(cx, cy, 2.4, 0, Math.PI * 2);
      context.fill();
    }

    context.fillStyle = IRIS_COLOR;
    for (const index of [template.leftIrisIndex, template.rightIrisIndex]) {
      if (index >= template.points.length) continue;
      const [cx, cy] = toCanvas(place(template.points[index]));
      context.beginPath();
      context.arc(cx, cy, 3.2, 0, Math.PI * 2);
      context.fill();
    }

    this.canvas.dataset.faceDrawnPointCount = String(drawn);
  }
}

/**
 * Fetch the template once and hand it to the view.
 *
 * Deliberately not retried forever: if this fails the panel says so instead of sitting
 * on "入力待ち" while the real problem is a missing face model.
 */
export async function loadFaceTemplate(view, { onStatus } = {}) {
  try {
    const response = await fetch("/api/face-model");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    view.setTemplate(await response.json());
    onStatus?.("ok");
  } catch (error) {
    view.drawIdle("顔モデルを取得できません");
    onStatus?.(`error: ${error.message}`);
  }
}
