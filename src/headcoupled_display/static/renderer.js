import { loadAsciiPcd } from "./pcd.js";

// Rolling window size for the receive-to-draw / CPU-draw timing ring buffer. This is a
// debug window, not an unbounded log, so a fixed small size is intentional.
const TIMING_RING_SIZE = 240;

// Interactive zoom bounds and easing (mouse wheel). The cloud is scaled about the
// screen-plane origin, so the clamp is generous; 1.0 is the scene-profile placement.
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 20.0;
const ZOOM_EASE = 0.2;
const ZOOM_EPSILON = 1e-4;

// A few millimetres of explicit separation between the back-wall grid and the backdrop
// quad behind it. Both live at the scene profile's back_wall_z_m; without this offset
// they would occupy the identical depth and flicker (z-fighting). This is the
// "separate the depths on purpose" fix the workdoc asks for, not a disabled depth test.
const BACK_WALL_GRID_Z_OFFSET_M = 0.002;

// The backdrop quad is sized larger than the physical display so head movement within
// the expected range never reveals bare canvas past its edges. This is a generous
// heuristic, not a tight fit computed from the frustum; widen it if that ever happens.
const BACKDROP_MARGIN_FACTOR = 2.5;

const BACKDROP_COLOR = new Float32Array([0.04, 0.06, 0.09, 1.0]);
const GRID_COLOR = new Float32Array([0.2, 0.32, 0.42, 1.0]);
// Matches the --accent CSS custom property (#57d6b5) so the verification overlay reads
// as "UI", not as scene content.
const SCREEN_FRAME_COLOR = new Float32Array([0.341, 0.839, 0.71, 1.0]);

const POINT_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec3 a_position;
in vec3 a_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
  gl_PointSize = u_point_size;
  v_color = a_color;
}`;

const POINT_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 v_color;
out vec4 out_color;
void main() {
  vec2 centered = gl_PointCoord - vec2(0.5);
  if (dot(centered, centered) > 0.25) discard;
  float edge = smoothstep(0.25, 0.14, dot(centered, centered));
  out_color = vec4(v_color * (0.78 + 0.22 * edge), 1.0);
}`;

// Shared by every reference-geometry pass (backdrop, back-wall grid, floor grid, screen
// frame): flat-shaded lines/triangles at a uniform color, nothing fancier is needed for
// depth cues.
const GRID_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec3 a_position;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
}`;

const GRID_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 out_color;
void main() {
  out_color = u_color;
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compilation failed: ${message}`);
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${message}`);
  }
  return program;
}

function multiplyMat4(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let k = 0; k < 4; k += 1) value += a[k * 4 + row] * b[column * 4 + k];
      out[column * 4 + row] = value;
    }
  }
  return out;
}

function translationMatrix(x, y, z) {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

function scaleMatrix(scale) {
  return new Float32Array([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1]);
}

function frustumMatrix(left, right, bottom, top, near, far) {
  const out = new Float32Array(16);
  out[0] = (2 * near) / (right - left);
  out[5] = (2 * near) / (top - bottom);
  out[8] = (right + left) / (right - left);
  out[9] = (top + bottom) / (top - bottom);
  out[10] = -(far + near) / (far - near);
  out[11] = -1;
  out[14] = -(2 * far * near) / (far - near);
  return out;
}

function projectionForDisplay(display, eye, near = 0.05, far = 8.0) {
  const distance = Math.max(eye[2], 0.2);
  const left = (near * (-display.width_m / 2 - eye[0])) / distance;
  const right = (near * (display.width_m / 2 - eye[0])) / distance;
  const bottom = (near * (-display.height_m / 2 - eye[1])) / distance;
  const top = (near * (display.height_m / 2 - eye[1])) / distance;
  return frustumMatrix(left, right, bottom, top, near, far);
}

/**
 * Build `T(anchor) * S * T(-aabb_center)` for a cloud with the given bounds.
 *
 * Nothing about the asset's own coordinates is assumed: the scale comes from its
 * longest bounding-box edge and the offset from its bounding-box midpoint, so
 * swapping the asset cannot silently move the scene relative to the screen.
 * The midpoint is used rather than the centroid because the centroid follows
 * point density, which is a property of the scan, not of what should be framed.
 *
 * @param {{anchor_display_m: number[], longest_edge_m: number}} scene
 * @param {{min: number[], max: number[], center: number[]}} bounds
 * @returns {Float32Array} Column-major 4x4 model matrix.
 */
function modelMatrixForBounds(scene, bounds) {
  const span = [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ];
  const longestEdge = Math.max(span[0], span[1], span[2]);
  if (!Number.isFinite(longestEdge) || longestEdge <= 0) {
    throw new Error("Point cloud bounding box is degenerate; cannot derive a uniform scale");
  }
  const scale = scene.longest_edge_m / longestEdge;
  const anchor = scene.anchor_display_m;
  return multiplyMat4(
    translationMatrix(
      anchor[0] - scale * bounds.center[0],
      anchor[1] - scale * bounds.center[1],
      anchor[2] - scale * bounds.center[2],
    ),
    scaleMatrix(scale),
  );
}

function transformPoint(matrix, point) {
  return [0, 1, 2].map(
    (row) =>
      matrix[row] * point[0] +
      matrix[4 + row] * point[1] +
      matrix[8 + row] * point[2] +
      matrix[12 + row],
  );
}

/**
 * A line-list grid of lines spaced `spacing` apart on a plane where one axis is held
 * fixed. `axes` lists [uAxisIndex, vAxisIndex, fixedAxisIndex] into (x, y, z); `uRange`
 * and `vRange` are the [min, max] extents along the two free axes.
 */
function buildGridLines(axes, uRange, vRange, fixedValue, spacing) {
  if (!(spacing > 0)) throw new Error("Grid spacing must be positive");
  const [uAxis, vAxis, fixedAxis] = axes;
  const [uMin, uMax] = uRange;
  const [vMin, vMax] = vRange;
  const vertices = [];
  const pushVertex = (u, v) => {
    const point = [0, 0, 0];
    point[uAxis] = u;
    point[vAxis] = v;
    point[fixedAxis] = fixedValue;
    vertices.push(point[0], point[1], point[2]);
  };
  for (let u = uMin; u <= uMax + 1e-9; u += spacing) {
    pushVertex(u, vMin);
    pushVertex(u, vMax);
  }
  for (let v = vMin; v <= vMax + 1e-9; v += spacing) {
    pushVertex(uMin, v);
    pushVertex(uMax, v);
  }
  return new Float32Array(vertices);
}

/** A single quad (triangle strip, 4 vertices) at a fixed z, spanning [xMin,xMax]x[yMin,yMax]. */
function buildQuadStrip(xMin, xMax, yMin, yMax, z) {
  return new Float32Array([xMin, yMin, z, xMax, yMin, z, xMin, yMax, z, xMax, yMax, z]);
}

/**
 * Perimeter frame of the active display area at z=0 plus inward tick marks every
 * `spacing`, as one gl.LINES vertex list. This is the "screen plane basis" HUD overlay,
 * not physical scene content -- see PointCloudRenderer.drawScreenFrame.
 */
function buildScreenFrameGeometry(display, spacing) {
  if (!(spacing > 0)) throw new Error("Grid spacing must be positive");
  const halfW = display.width_m / 2;
  const halfH = display.height_m / 2;
  const tick = Math.min(spacing, halfW, halfH) * 0.18;
  const z = 0;
  const vertices = [];
  const pushSegment = (x1, y1, x2, y2) => vertices.push(x1, y1, z, x2, y2, z);
  pushSegment(-halfW, -halfH, halfW, -halfH);
  pushSegment(halfW, -halfH, halfW, halfH);
  pushSegment(halfW, halfH, -halfW, halfH);
  pushSegment(-halfW, halfH, -halfW, -halfH);
  for (let x = -halfW; x <= halfW + 1e-9; x += spacing) {
    pushSegment(x, halfH, x, halfH - tick);
    pushSegment(x, -halfH, x, -halfH + tick);
  }
  for (let y = -halfH; y <= halfH + 1e-9; y += spacing) {
    pushSegment(-halfW, y, -halfW + tick, y);
    pushSegment(halfW, y, halfW - tick, y);
  }
  return new Float32Array(vertices);
}

export class PointCloudRenderer {
  constructor(canvas, display, scene) {
    if (!scene || !Array.isArray(scene.anchor_display_m) || !(scene.longest_edge_m > 0)) {
      // No implicit default: a missing scene profile is a configuration error, and
      // silently guessing a placement would make a broken deployment look correct.
      throw new Error("PointCloudRenderer requires a scene profile from /api/profile");
    }
    this.canvas = canvas;
    this.display = display;
    this.scene = scene;
    this.model = null;
    this.baseModel = null;
    // Interactive zoom (mouse wheel): a multiplicative scale applied to the model
    // around the screen-plane origin, keeping the head-coupled parallax intact.
    // 1.0 is the scene-profile placement; wheel overrides it for inspection only.
    this.zoom = 1.0;
    this.zoomTarget = 1.0;
    this.eye = [0, 0, 0.67];
    this.pointCount = 0;
    this.positions = null;
    this.colors = null;
    this.running = true;
    // "verification" shows the screen-plane HUD overlay; "immersive" hides it. Kept in
    // sync with document.body.dataset.viewMode by app.js calling setViewMode().
    this.viewMode = "verification";
    this.staticGeometry = null;
    this.staticUploadCount = 0;

    // Dirty-draw scheduling state (steps 19-20): at most one requestAnimationFrame may
    // be pending at any time -- see scheduleDraw(). Pose updates, resize and mode
    // changes are the only triggers; nothing here runs an unconditional draw loop.
    this.rafHandle = null;
    this.pendingRafCount = 0;
    this.drawCount = 0;
    this.latestSequence = null;
    this.lastRenderedSequence = null;
    this.sequenceReversalCount = 0;
    this.latestReceivedAtMs = null;
    this.latestInferenceUnixMs = null;

    // Rolling receive-to-draw / CPU-draw timing samples, timestamped with
    // performance.timeOrigin + performance.now() so they are directly comparable to
    // server-side Unix-ns timestamps.
    this.timingRing = new Array(TIMING_RING_SIZE).fill(null);
    this.timingRingIndex = 0;

    this.gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: true,
      depth: true,
      preserveDrawingBuffer: true,
    });
    if (this.gl) {
      this.mode = "WebGL2";
      this.initializeWebGl();
    } else {
      this.context2d = canvas.getContext("2d", { alpha: false });
      if (!this.context2d) throw new Error("Neither WebGL2 nor Canvas2D is available");
      this.mode = "Canvas2D fallback";
      this.gpuTimingAvailable = false;
    }

    this.canvas.dataset.staticUploadCount = "0";
    this.canvas.dataset.drawCount = "0";
    this.canvas.dataset.pendingRafCount = "0";
    this.canvas.dataset.sequenceReversalCount = "0";
    this.canvas.dataset.lastRenderedSequence = "-1";
    this.canvas.dataset.gpuTimingAvailable = String(this.gpuTimingAvailable);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize(); // Also schedules the first draw; see resize() below.
  }

  initializeWebGl() {
    const gl = this.gl;
    this.program = createProgram(gl, POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER);
    this.positionLocation = gl.getAttribLocation(this.program, "a_position");
    this.colorLocation = gl.getAttribLocation(this.program, "a_color");
    this.mvpLocation = gl.getUniformLocation(this.program, "u_mvp");
    this.pointSizeLocation = gl.getUniformLocation(this.program, "u_point_size");
    this.vao = gl.createVertexArray();
    this.positionBuffer = gl.createBuffer();
    this.colorBuffer = gl.createBuffer();

    this.gridProgram = createProgram(gl, GRID_VERTEX_SHADER, GRID_FRAGMENT_SHADER);
    this.gridPositionLocation = gl.getAttribLocation(this.gridProgram, "a_position");
    this.gridMvpLocation = gl.getUniformLocation(this.gridProgram, "u_mvp");
    this.gridColorLocation = gl.getUniformLocation(this.gridProgram, "u_color");

    // Detection only. Using this extension correctly needs an async, multi-frame
    // query/readback protocol, which is out of scope for this step; what is promised
    // is that CPU wall time is never relabeled as GPU time when the extension is
    // unavailable (see draw()/recordTiming()).
    this.timerExtension = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.gpuTimingAvailable = this.timerExtension !== null;
  }

  /**
   * Build (or rebuild, reusing existing GPU buffers) the static reference geometry:
   * back-wall backdrop + grid, floor grid, and the verification screen frame. Called
   * once from load() -- i.e. at load time and whenever a new scene is loaded -- never
   * per frame. See draw()/drawStaticWorldGeometry() for how these are drawn.
   */
  buildStaticGeometry() {
    const gl = this.gl;
    if (!gl) return; // Canvas2D fallback has no reference geometry; out of scope here.
    const halfW = this.display.width_m / 2;
    const halfH = this.display.height_m / 2;
    const spacing = this.scene.grid_spacing_m;

    const backdropVertices = buildQuadStrip(
      -halfW * BACKDROP_MARGIN_FACTOR,
      halfW * BACKDROP_MARGIN_FACTOR,
      this.scene.floor_y_m,
      halfH * BACKDROP_MARGIN_FACTOR,
      this.scene.back_wall_z_m,
    );
    const backWallGridVertices = buildGridLines(
      [0, 1, 2],
      [-halfW, halfW],
      [-halfH, halfH],
      this.scene.back_wall_z_m + BACK_WALL_GRID_Z_OFFSET_M,
      spacing,
    );
    const floorGridVertices = buildGridLines(
      [0, 2, 1],
      [-halfW, halfW],
      [this.scene.floor_far_z_m, this.scene.floor_near_z_m],
      this.scene.floor_y_m,
      spacing,
    );
    const screenFrameVertices = buildScreenFrameGeometry(this.display, spacing);

    const previous = this.staticGeometry;
    this.staticGeometry = {
      backdrop: this.uploadStaticBuffer(previous?.backdrop?.buffer, backdropVertices, 4),
      backWallGrid: this.uploadStaticBuffer(
        previous?.backWallGrid?.buffer,
        backWallGridVertices,
        backWallGridVertices.length / 3,
      ),
      floorGrid: this.uploadStaticBuffer(
        previous?.floorGrid?.buffer,
        floorGridVertices,
        floorGridVertices.length / 3,
      ),
      screenFrame: this.uploadStaticBuffer(
        previous?.screenFrame?.buffer,
        screenFrameVertices,
        screenFrameVertices.length / 3,
      ),
    };
    this.staticUploadCount += 1;
    this.canvas.dataset.staticUploadCount = String(this.staticUploadCount);
  }

  uploadStaticBuffer(existingBuffer, vertices, vertexCount) {
    const gl = this.gl;
    const buffer = existingBuffer ?? gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    return { buffer, vertexCount };
  }

  async load(url) {
    const cloud = await loadAsciiPcd(url);
    this.positions = cloud.positions;
    this.colors = cloud.colors;
    this.pointCount = cloud.pointCount;
    this.bounds = cloud.bounds;
    // Computed once per asset, not per frame: the placement only depends on the
    // asset and the scene profile, neither of which changes while drawing.
    this.model = modelMatrixForBounds(this.scene, cloud.bounds);
    this.baseModel = this.model;
    // Zoom is a viewer override; reset it whenever a new scene is loaded so the
    // zoomed state can never leak across scenes.
    this.zoom = 1.0;
    this.zoomTarget = 1.0;
    if (this.gl) {
      const gl = this.gl;
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, cloud.positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(this.positionLocation);
      gl.vertexAttribPointer(this.positionLocation, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, cloud.colors, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(this.colorLocation);
      gl.vertexAttribPointer(this.colorLocation, 3, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      this.buildStaticGeometry();
    }
    // Exposed so end-to-end tests can assert the placement numerically instead of
    // inferring it from a screenshot.
    const placedCenter = transformPoint(this.model, cloud.bounds.center);
    const placedMin = transformPoint(this.model, cloud.bounds.min);
    const placedMax = transformPoint(this.model, cloud.bounds.max);
    this.canvas.dataset.rendererReady = "true";
    this.canvas.dataset.rendererMode = this.mode;
    this.canvas.dataset.sceneId = this.scene.scene_id ?? "";
    this.canvas.dataset.modelScale = String(this.model[0]);
    this.canvas.dataset.zoom = String(this.zoom);
    this.canvas.dataset.zoomTarget = String(this.zoomTarget);
    this.canvas.dataset.modelCenterDisplayM = JSON.stringify(placedCenter);
    this.canvas.dataset.modelMinDisplayM = JSON.stringify(placedMin);
    this.canvas.dataset.modelMaxDisplayM = JSON.stringify(placedMax);
    this.scheduleDraw();
    return { pointCount: cloud.pointCount, mode: this.mode };
  }

  /**
   * Update the eye position, optionally tagged with the pose's server-assigned
   * sequence number. A message whose sequence is older than the latest one already
   * applied is dropped rather than applied: this is what guarantees the renderer never
   * draws an older pose after a newer one has already been drawn, independent of
   * whatever order messages happen to arrive in. sequenceReversalCount makes that
   * (otherwise silent) drop observable.
   */
  /**
   * @param {number[]} eye - cyclopean eye position in display metres
   * @param {number} sequence - producer frame number, used to reject reordered poses
   * @param {number|null} [inferenceUnixNs] - when the producer finished inference, on the
   *   Unix clock. Comparable to `performance.timeOrigin + performance.now()` (modulo
   *   ordinary clock-sync error) and therefore the only pair from which
   *   recognition-to-WebGL latency can be derived. The producer's monotonic readings
   *   belong to another process and must never be subtracted from a browser timestamp.
   */
  setEye(eye, sequence, inferenceUnixNs = null) {
    if (!Array.isArray(eye) || eye.length !== 3 || eye.some((value) => !Number.isFinite(value))) return;
    if (Number.isFinite(sequence) && this.latestSequence !== null && sequence < this.latestSequence) {
      this.sequenceReversalCount += 1;
      this.canvas.dataset.sequenceReversalCount = String(this.sequenceReversalCount);
      return;
    }
    this.eye = [...eye];
    if (Number.isFinite(sequence)) this.latestSequence = sequence;
    this.latestReceivedAtMs = performance.timeOrigin + performance.now();
    this.latestInferenceUnixMs =
      typeof inferenceUnixNs === "number" && Number.isFinite(inferenceUnixNs)
        ? inferenceUnixNs / 1e6
        : null;
    this.scheduleDraw();
  }

  setViewMode(mode) {
    if (mode !== "immersive" && mode !== "verification") return;
    if (this.viewMode === mode) return;
    this.viewMode = mode;
    this.scheduleDraw();
  }

  /**
   * The model matrix actually drawn: the scene-profile placement scaled by the
   * interactive zoom around the screen-plane origin (z=0, the anchor). At
   * zoom === 1.0 this is exactly `baseModel`, so the default path is unchanged.
   * Column-major 4x4.
   */
  zoomedModel() {
    if (this.zoom === 1.0) return this.baseModel;
    return multiplyMat4(scaleMatrix(this.zoom), this.baseModel);
  }

  /**
   * Request a zoom target (mouse-wheel). Clamped to [MIN_ZOOM, MAX_ZOOM]; the
   * current value eases toward it in updateZoom(), which keeps re-drawing until
   * the two converge.
   */
  setZoom(target) {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number(target) || 1.0));
    if (!Number.isFinite(clamped)) return;
    this.zoomTarget = clamped;
    this.canvas.dataset.zoomTarget = String(clamped);
    this.scheduleDraw();
  }

  /**
   * Step the eased zoom toward the target. Returns true while still animating so
   * draw() can keep the dirty scheduler alive without an unconditional loop.
   */
  updateZoom() {
    if (Math.abs(this.zoom - this.zoomTarget) < ZOOM_EPSILON) {
      if (this.zoom !== this.zoomTarget) this.zoom = this.zoomTarget;
      return false;
    }
    this.zoom += (this.zoomTarget - this.zoom) * ZOOM_EASE;
    if (Math.abs(this.zoom - this.zoomTarget) < ZOOM_EPSILON) this.zoom = this.zoomTarget;
    this.canvas.dataset.zoom = String(this.zoom);
    return true;
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (this.gl) this.gl.viewport(0, 0, width, height);
    this.scheduleDraw();
  }

  /**
   * Ensure exactly one requestAnimationFrame is pending. Called only from the three
   * explicit triggers (pose update, resize, mode change) plus load() -- never from
   * inside draw() itself, which is what breaks the old unconditional redraw loop. If a
   * frame is already pending, it will pick up whatever is the latest state (eye,
   * sequence, viewMode) by the time it actually runs, so calling this again before
   * that happens is a no-op.
   */
  scheduleDraw() {
    if (this.rafHandle !== null) return;
    this.pendingRafCount = 1;
    this.canvas.dataset.pendingRafCount = "1";
    this.rafHandle = requestAnimationFrame(() => this.onAnimationFrame());
  }

  onAnimationFrame() {
    this.rafHandle = null;
    this.pendingRafCount = 0;
    this.canvas.dataset.pendingRafCount = "0";
    this.draw();
  }

  drawGridBuffer(entry, mvp, color, mode) {
    const gl = this.gl;
    gl.useProgram(this.gridProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, entry.buffer);
    gl.enableVertexAttribArray(this.gridPositionLocation);
    gl.vertexAttribPointer(this.gridPositionLocation, 3, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(this.gridMvpLocation, false, mvp);
    gl.uniform4fv(this.gridColorLocation, color);
    gl.drawArrays(mode, 0, entry.vertexCount);
  }

  /**
   * World-space depth cues: backdrop, back-wall grid, floor grid. All three share the
   * point cloud's view/projection and are drawn with depth testing on (see drawWebGl),
   * so they occlude and are occluded by the point cloud correctly as the eye moves.
   */
  drawStaticWorldGeometry(viewProjection) {
    const gl = this.gl;
    const geometry = this.staticGeometry;
    // Backdrop: an actual plane pushed through the same view/projection as everything
    // else, not just the canvas clear color -- so it participates in parallax instead
    // of looking pasted onto the screen. It is the single furthest surface in the
    // scene, so there is nothing behind it to conflict with.
    this.drawGridBuffer(geometry.backdrop, viewProjection, BACKDROP_COLOR, gl.TRIANGLE_STRIP);
    // Back-wall grid: offset BACK_WALL_GRID_Z_OFFSET_M in front of the coincident
    // backdrop plane (see the constant's doc comment) -- an explicit depth separation,
    // not a disabled depth test.
    this.drawGridBuffer(geometry.backWallGrid, viewProjection, GRID_COLOR, gl.LINES);
    // Floor grid: a different plane (y=floor_y_m) entirely, so it cannot z-fight with
    // either of the two above.
    this.drawGridBuffer(geometry.floorGrid, viewProjection, GRID_COLOR, gl.LINES);
  }

  drawScreenFrame(viewProjection) {
    this.drawGridBuffer(this.staticGeometry.screenFrame, viewProjection, SCREEN_FRAME_COLOR, this.gl.LINES);
  }

  drawWebGl() {
    const gl = this.gl;
    gl.clearColor(0.025, 0.035, 0.052, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.pointCount === 0 || !this.model) return;
    const projection = projectionForDisplay(this.display, this.eye);
    const view = translationMatrix(-this.eye[0], -this.eye[1], -this.eye[2]);
    const viewProjection = multiplyMat4(projection, view);
    const mvp = multiplyMat4(viewProjection, this.zoomedModel());

    // World pass: backdrop/wall/floor and the point cloud all share this
    // view/projection with depth testing on. The depth buffer -- not draw order --
    // is what keeps overlaps correct; drawing back-to-front here is only for
    // readability and a small amount of overdraw saved.
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    if (this.staticGeometry) this.drawStaticWorldGeometry(viewProjection);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.mvpLocation, false, mvp);
    gl.uniform1f(this.pointSizeLocation, Math.min(window.devicePixelRatio || 1, 2) * 2.15);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.pointCount);
    gl.bindVertexArray(null);

    // Screen-plane overlay: z=0 is a verification-mode HUD reference, not physical
    // scene content, so it must win against both the wall behind it (z<0) and the
    // point cloud (which straddles z=0). Depth testing is switched off only for this
    // last pass, and depth writes are switched off too so a stray write here cannot
    // corrupt next frame's world pass (each frame clears the depth buffer anyway, but
    // this keeps the intent explicit rather than relying on that).
    if (this.viewMode === "verification" && this.staticGeometry) {
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      this.drawScreenFrame(viewProjection);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
    }
  }

  drawCanvas2d() {
    const ctx = this.context2d;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.fillStyle = "#07101c";
    ctx.fillRect(0, 0, width, height);
    if (!this.positions || !this.colors || !this.model) return;
    const eye = this.eye;
    // Same model matrix as the WebGL path (zoom included), so the two paths can
    // never drift apart: only the rasterizer differs.
    const model = this.zoomedModel();
    const scale = model[0];
    const stride = this.pointCount > 6000 ? 3 : 1;
    for (let index = 0; index < this.pointCount; index += stride) {
      const offset = index * 3;
      const px = this.positions[offset] * scale + model[12];
      const py = this.positions[offset + 1] * scale + model[13];
      const pz = this.positions[offset + 2] * scale + model[14];
      const denominator = eye[2] - pz;
      if (denominator <= 0.01) continue;
      const ratio = eye[2] / denominator;
      const screenX = eye[0] + ratio * (px - eye[0]);
      const screenY = eye[1] + ratio * (py - eye[1]);
      const x = (screenX / this.display.width_m + 0.5) * width;
      const y = (0.5 - screenY / this.display.height_m) * height;
      if (x < -2 || x > width + 2 || y < -2 || y > height + 2) continue;
      const r = Math.round(this.colors[offset] * 255);
      const g = Math.round(this.colors[offset + 1] * 255);
      const b = Math.round(this.colors[offset + 2] * 255);
      ctx.fillStyle = `rgb(${r} ${g} ${b})`;
      const size = Math.max(1, Math.min(3, 2.2 * ratio));
      ctx.fillRect(x, y, size, size);
    }
  }

  recordTiming(drawStartMs, drawEndMs) {
    const cpuDrawMs = drawEndMs - drawStartMs;
    const receiveToDrawMs =
      this.latestReceivedAtMs === null ? null : drawStartMs - this.latestReceivedAtMs;
    const inferenceToDrawMs =
      this.latestInferenceUnixMs === null ? null : drawStartMs - this.latestInferenceUnixMs;
    this.timingRing[this.timingRingIndex] = {
      drawStartMs,
      cpuDrawMs,
      receiveToDrawMs,
      inferenceToDrawMs,
    };
    this.timingRingIndex = (this.timingRingIndex + 1) % this.timingRing.length;
  }

  /**
   * Summarise the timing ring. Computed on demand rather than per frame: a percentile
   * over 240 samples every frame would itself become part of what is being measured.
   *
   * `gpuTimingAvailable` is reported so a caller cannot present `cpuDrawP95Ms` as GPU
   * time on hardware where `EXT_disjoint_timer_query_webgl2` is missing.
   */
  timingSummary() {
    const samples = this.timingRing.filter((entry) => entry !== null);
    const percentile = (values, fraction) => {
      if (values.length === 0) return null;
      const sorted = [...values].sort((a, b) => a - b);
      const index = Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)));
      return sorted[index];
    };
    const cpu = samples.map((entry) => entry.cpuDrawMs);
    const receive = samples
      .map((entry) => entry.receiveToDrawMs)
      .filter((value) => value !== null);
    const inference = samples
      .map((entry) => entry.inferenceToDrawMs)
      .filter((value) => value !== null && value !== undefined);
    // Interval between consecutive draws. With a dirty scheduler every draw waits for
    // the next requestAnimationFrame, so this is the display/compositor cadence and it
    // is the floor under receiveToDraw: a pose can never be shown sooner than the next
    // frame. Reported so a receiveToDraw figure can be read against the cadence that
    // produced it instead of being blamed on the renderer.
    const starts = samples.map((entry) => entry.drawStartMs).sort((a, b) => a - b);
    const intervals = starts.slice(1).map((value, index) => value - starts[index]);
    return {
      sampleCount: samples.length,
      receiveSampleCount: receive.length,
      frameIntervalP50Ms: percentile(intervals, 0.5),
      frameIntervalP95Ms: percentile(intervals, 0.95),
      cpuDrawP50Ms: percentile(cpu, 0.5),
      cpuDrawP95Ms: percentile(cpu, 0.95),
      receiveToDrawP50Ms: percentile(receive, 0.5),
      receiveToDrawP95Ms: percentile(receive, 0.95),
      inferenceSampleCount: inference.length,
      inferenceToDrawP50Ms: percentile(inference, 0.5),
      inferenceToDrawP95Ms: percentile(inference, 0.95),
      drawCount: this.drawCount,
      sequenceReversalCount: this.sequenceReversalCount,
      lastRenderedSequence: this.lastRenderedSequence,
      gpuTimingAvailable: this.gpuTimingAvailable,
      mode: this.mode,
    };
  }

  draw() {
    if (!this.running) return;
    // Ease the interactive zoom toward its target before drawing; while it is still
    // moving, request the next frame so the animation progresses without an
    // unconditional redraw loop.
    const zoomAnimating = this.updateZoom();
    const drawStartMs = performance.timeOrigin + performance.now();
    if (this.gl) this.drawWebGl();
    else this.drawCanvas2d();
    const drawEndMs = performance.timeOrigin + performance.now();
    this.drawCount += 1;
    if (this.latestSequence !== null) this.lastRenderedSequence = this.latestSequence;
    this.recordTiming(drawStartMs, drawEndMs);
    this.canvas.dataset.drawCount = String(this.drawCount);
    this.canvas.dataset.lastRenderedSequence = String(this.lastRenderedSequence ?? -1);
    if (zoomAnimating) this.scheduleDraw();
  }

  dispose() {
    this.running = false;
    this.resizeObserver.disconnect();
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
  }
}
