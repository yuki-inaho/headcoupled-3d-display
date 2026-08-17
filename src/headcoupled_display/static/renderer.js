import { loadAsciiPcd } from "./pcd.js";

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
  const distance = Math.max(eye[2], 0.20);
  const left = near * (-display.width_m / 2 - eye[0]) / distance;
  const right = near * (display.width_m / 2 - eye[0]) / distance;
  const bottom = near * (-display.height_m / 2 - eye[1]) / distance;
  const top = near * (display.height_m / 2 - eye[1]) / distance;
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
    this.eye = [0, 0, 0.67];
    this.pointCount = 0;
    this.positions = null;
    this.colors = null;
    this.running = true;
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
    }
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    requestAnimationFrame(() => this.draw());
  }

  initializeWebGl() {
    const gl = this.gl;
    this.program = createProgram(
      gl,
      `#version 300 es
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
      }`,
      `#version 300 es
      precision highp float;
      in vec3 v_color;
      out vec4 out_color;
      void main() {
        vec2 centered = gl_PointCoord - vec2(0.5);
        if (dot(centered, centered) > 0.25) discard;
        float edge = smoothstep(0.25, 0.14, dot(centered, centered));
        out_color = vec4(v_color * (0.78 + 0.22 * edge), 1.0);
      }`,
    );
    this.positionLocation = gl.getAttribLocation(this.program, "a_position");
    this.colorLocation = gl.getAttribLocation(this.program, "a_color");
    this.mvpLocation = gl.getUniformLocation(this.program, "u_mvp");
    this.pointSizeLocation = gl.getUniformLocation(this.program, "u_point_size");
    this.vao = gl.createVertexArray();
    this.positionBuffer = gl.createBuffer();
    this.colorBuffer = gl.createBuffer();
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
    this.canvas.dataset.modelCenterDisplayM = JSON.stringify(placedCenter);
    this.canvas.dataset.modelMinDisplayM = JSON.stringify(placedMin);
    this.canvas.dataset.modelMaxDisplayM = JSON.stringify(placedMax);
    return { pointCount: cloud.pointCount, mode: this.mode };
  }

  setEye(eye) {
    if (!Array.isArray(eye) || eye.length !== 3 || eye.some((value) => !Number.isFinite(value))) return;
    this.eye = [...eye];
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
  }

  drawWebGl() {
    const gl = this.gl;
    gl.clearColor(0.025, 0.035, 0.052, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    if (this.pointCount === 0 || !this.model) return;
    const projection = projectionForDisplay(this.display, this.eye);
    const view = translationMatrix(-this.eye[0], -this.eye[1], -this.eye[2]);
    const mvp = multiplyMat4(projection, multiplyMat4(view, this.model));
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.mvpLocation, false, mvp);
    gl.uniform1f(this.pointSizeLocation, Math.min(window.devicePixelRatio || 1, 2) * 2.15);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.pointCount);
    gl.bindVertexArray(null);
  }

  drawCanvas2d() {
    const ctx = this.context2d;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.fillStyle = "#07101c";
    ctx.fillRect(0, 0, width, height);
    if (!this.positions || !this.colors || !this.model) return;
    const eye = this.eye;
    // Same model matrix as the WebGL path. Duplicating the placement constants here
    // is what let the two paths drift apart before; only the rasterizer differs now.
    const model = this.model;
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

  draw() {
    if (!this.running) return;
    if (this.gl) this.drawWebGl();
    else this.drawCanvas2d();
    requestAnimationFrame(() => this.draw());
  }

  dispose() {
    this.running = false;
    this.resizeObserver.disconnect();
  }
}
