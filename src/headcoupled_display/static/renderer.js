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

export class PointCloudRenderer {
  constructor(canvas, display) {
    this.canvas = canvas;
    this.display = display;
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
    this.canvas.dataset.rendererReady = "true";
    this.canvas.dataset.rendererMode = this.mode;
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
    if (this.pointCount === 0) return;
    const projection = projectionForDisplay(this.display, this.eye);
    const view = translationMatrix(-this.eye[0], -this.eye[1], -this.eye[2]);
    const model = multiplyMat4(translationMatrix(0, -0.055, -0.42), scaleMatrix(0.21));
    const mvp = multiplyMat4(projection, multiplyMat4(view, model));
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
    if (!this.positions || !this.colors) return;
    const eye = this.eye;
    const modelScale = 0.21;
    const modelY = -0.055;
    const modelZ = -0.42;
    const stride = this.pointCount > 6000 ? 3 : 1;
    for (let index = 0; index < this.pointCount; index += stride) {
      const offset = index * 3;
      const px = this.positions[offset] * modelScale;
      const py = this.positions[offset + 1] * modelScale + modelY;
      const pz = this.positions[offset + 2] * modelScale + modelZ;
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
