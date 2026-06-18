// @ts-check
// Client-side renderer for the posed NeuroMechFly mesh overlay (WebGL2).
//
// The server fits + poses the model (re-fit live from the operator's 3D
// corrections) and ships only geometry -- the topology + per-vertex colors once,
// the posed vertices per frame -- so the GPU here does the rasterization that the
// server used to. One offscreen WebGL2 context renders the mesh per camera into its
// own canvas; `app.js` copies that into each view and composites it under the
// editable skeleton (so the keypoints stay legible on top).
//
// Each camera is projected exactly the way `CameraGroup.project` does (a pinhole
// model, validated to sub-pixel agreement): the view matrix flips deeperfly's
// y-down/z-forward camera frame into OpenGL's y-up/z-back one (D = diag(1,-1,-1)),
// and the projection matrix is built straight from the intrinsics [fx, fy, cx, cy]
// and the footage size. Faces are shaded by a headlight at the camera, with a depth
// buffer for exact occlusion; its near/far planes are fit to the posed mesh each frame
// (see `depthRange`) so coincident surfaces (the eye on the head) keep their depth
// precision and don't z-fight when seen from the far rig cameras. The whole silhouette
// is composited translucent.

const VERT_SRC = `#version 300 es
uniform mat4 u_mvp;
in vec3 a_pos;
in vec3 a_col;
in vec3 a_nrm;
out vec3 v_world;
out vec3 v_col;
out vec3 v_nrm;
void main() {
  v_world = a_pos;
  v_col = a_col;
  v_nrm = a_nrm;
  gl_Position = u_mvp * vec4(a_pos, 1.0);
}`;

const FRAG_SRC = `#version 300 es
precision highp float;
uniform vec3 u_light;   // camera centre, world frame (a headlight)
in vec3 v_world;
in vec3 v_col;
in vec3 v_nrm;
out vec4 frag;
void main() {
  // Smooth (interpolated) per-vertex normal -- no faceting, so the overlay reads
  // like the docs model viewer. Two-sided so a flipped winding still lights up.
  // A zero-length normal (degenerate / doubled geometry, e.g. at the abdomen tip,
  // where vertex_normals cancels to 0) would make normalize() NaN and paint the
  // fragment white/black -- fall back to the unlit ambient term there.
  float nlen = length(v_nrm);
  vec3 n = nlen > 1e-6 ? v_nrm / nlen : vec3(0.0);
  vec3 l = normalize(u_light - v_world);
  float shade = 0.45 + 0.55 * abs(dot(n, l));   // matches the CPU/server shading
  frag = vec4(v_col * shade, 1.0);
}`;

export class MeshGL {
  constructor() {
    this.canvas = document.createElement("canvas");
    const gl = this.canvas.getContext("webgl2", {
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
    });
    this.gl = gl;
    this.ok = !!gl;
    this.nVerts = 0;
    this.indexCount = 0;
    this.faces = null; // Uint32Array (n_faces * 3)
    // Bounding sphere of the current frame's drawable mesh (world frame); the depth
    // planes are fit to it per camera so far views keep depth precision (no z-fighting).
    this.center = /** @type {[number, number, number]} */ ([0, 0, 0]);
    this.radius = 1;
    if (!gl) return;

    this.prog = link(gl, VERT_SRC, FRAG_SRC);
    this.locPos = gl.getAttribLocation(this.prog, "a_pos");
    this.locCol = gl.getAttribLocation(this.prog, "a_col");
    this.locNrm = gl.getAttribLocation(this.prog, "a_nrm");
    this.uMvp = gl.getUniformLocation(this.prog, "u_mvp");
    this.uLight = gl.getUniformLocation(this.prog, "u_light");
    this.posBuf = gl.createBuffer();
    this.colBuf = gl.createBuffer();
    this.nrmBuf = gl.createBuffer();
    this.idxBuf = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.enable(gl.DEPTH_TEST);
  }

  /**
   * Upload the static topology + per-vertex colors (once per session).
   * @param {ArrayBuffer} buf  header [u32 nVerts, u32 nFaces] + u32 faces + u8 rgb
   */
  loadAsset(buf) {
    if (!this.ok) return;
    const gl = this.gl;
    const head = new Uint32Array(buf, 0, 2);
    const nVerts = head[0];
    const nFaces = head[1];
    this.nVerts = nVerts;
    this.faces = new Uint32Array(buf, 8, nFaces * 3);
    const rgb = new Uint8Array(buf, 8 + nFaces * 3 * 4, nVerts * 3);
    const col = new Float32Array(nVerts * 3);
    for (let i = 0; i < col.length; i++) col[i] = rgb[i] / 255;

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.locCol);
    gl.vertexAttribPointer(this.locCol, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.enableVertexAttribArray(this.locPos);
    gl.vertexAttribPointer(this.locPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmBuf);
    gl.enableVertexAttribArray(this.locNrm);
    gl.vertexAttribPointer(this.locNrm, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  /**
   * Set the current frame's posed vertices, smooth normals and drawable faces.
   * @param {Float32Array} verts    (nVerts * 3) world positions
   * @param {Float32Array} normals  (nVerts * 3) smooth world-space normals
   * @param {Uint8Array} valid      (nFaces) 1 where all three vertices were posed
   */
  setVerts(verts, normals, valid) {
    if (!this.ok || !this.faces) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmBuf);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
    // Build an index buffer over only the drawable faces (occluded leg segments and
    // un-fit parts drop out frame to frame).
    const faces = this.faces;
    let n = 0;
    for (let f = 0; f < valid.length; f++) if (valid[f]) n++;
    const idx = new Uint32Array(n * 3);
    let j = 0;
    // Accumulate the drawable verts' AABB in the same pass (only indexed verts, so
    // un-posed NaN vertices never enter the bounds) -> a bounding sphere for the depth fit.
    let lox = Infinity, loy = Infinity, loz = Infinity;
    let hix = -Infinity, hiy = -Infinity, hiz = -Infinity;
    for (let f = 0; f < valid.length; f++) {
      if (!valid[f]) continue;
      for (let k = 0; k < 3; k++) {
        const vi = faces[3 * f + k];
        idx[j++] = vi;
        const x = verts[3 * vi], y = verts[3 * vi + 1], z = verts[3 * vi + 2];
        if (x < lox) lox = x;
        if (x > hix) hix = x;
        if (y < loy) loy = y;
        if (y > hiy) hiy = y;
        if (z < loz) loz = z;
        if (z > hiz) hiz = z;
      }
    }
    this.indexCount = idx.length;
    if (n > 0) {
      this.center = [(lox + hix) / 2, (loy + hiy) / 2, (loz + hiz) / 2];
      this.radius = 0.5 * Math.hypot(hix - lox, hiy - loy, hiz - loz) || 1e-3;
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);
  }

  /**
   * Render the current mesh through one camera into this renderer's canvas.
   * @param {import("./types.js").CameraProj} cam
   * @param {number} [supersample]  GL pixels per footage pixel (sharper on hi-DPI /
   *   zoomed views; the projection itself stays footage-based, only the raster grid
   *   is finer). Defaults to 1 (footage resolution).
   * @returns {HTMLCanvasElement | null}  the canvas, or null
   */
  render(cam, supersample = 1) {
    if (!this.ok || !this.indexCount) return null;
    const gl = this.gl;
    const [w, h] = cam.size;
    const rw = Math.max(1, Math.round(w * supersample));
    const rh = Math.max(1, Math.round(h * supersample));
    if (this.canvas.width !== rw || this.canvas.height !== rh) {
      this.canvas.width = rw;
      this.canvas.height = rh;
    }
    gl.viewport(0, 0, rw, rh);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.prog);
    const { near, far } = this.depthRange(cam);
    gl.uniformMatrix4fv(this.uMvp, false, mvpColumnMajor(cam, w, h, near, far));
    const c = cameraCentre(cam);
    gl.uniform3f(this.uLight, c[0], c[1], c[2]);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
    return this.canvas;
  }

  // Near/far planes bracketing the drawable mesh along this camera's optical axis.
  // Fitting them to the mesh's bounding sphere (rather than a fixed 0.01..1000) keeps
  // the depth buffer's precision on the fly, so coincident surfaces (the eye on the
  // head) don't z-fight when the camera is far away.
  /** @param {import("./types.js").CameraProj} cam @returns {{near: number, far: number}} */
  depthRange(cam) {
    const r = cam.rmat;
    const t = cam.tvec;
    const c = this.center;
    // Optical-axis distance to the sphere centre: third row of [R | t] applied to it.
    const d = r[6] * c[0] + r[7] * c[1] + r[8] * c[2] + t[2];
    const near = Math.max(d - this.radius, (d + this.radius) * 1e-3, 1e-4);
    const far = Math.max(d + this.radius, near * 1.001);
    return { near, far };
  }
}

/** Camera centre in world coordinates: -Rᵀ t. */
function cameraCentre(cam) {
  const r = cam.rmat;
  const t = cam.tvec;
  return [
    -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
    -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
    -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
  ];
}

// MVP = Proj · View, returned column-major for `uniformMatrix4fv`.
// View flips deeperfly's camera frame (x-right, y-down, z-forward) into OpenGL's
// (x-right, y-up, z-back) with D = diag(1, -1, -1); Proj is the pinhole intrinsics
// (validated to exactly reproduce CameraGroup.project). `near`/`far` set only the depth
// mapping (rows 3-4) -- the projected x/y are unaffected -- and are fit per frame.
function mvpColumnMajor(cam, w, h, near, far) {
  const [fx, fy, cx, cy] = cam.intr;
  const r = cam.rmat;
  const t = cam.tvec;
  // View (row-major): rows 1,2 negated (the y/z flip).
  const V = [
    r[0], r[1], r[2], t[0],
    -r[3], -r[4], -r[5], -t[1],
    -r[6], -r[7], -r[8], -t[2],
    0, 0, 0, 1,
  ];
  const A = -(far + near) / (far - near);
  const B = (-2 * far * near) / (far - near);
  const P = [
    (2 * fx) / w, 0, (w - 2 * cx) / w, 0,
    0, (2 * fy) / h, (2 * cy - h) / h, 0,
    0, 0, A, B,
    0, 0, -1, 0,
  ];
  return toColumnMajor(mul4(P, V));
}

/** Row-major 4x4 multiply. */
function mul4(a, b) {
  const o = new Array(16).fill(0);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) o[4 * i + j] += a[4 * i + k] * b[4 * k + j];
  return o;
}

/** Row-major -> column-major Float32Array (WebGL wants column-major). */
function toColumnMajor(m) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) o[4 * j + i] = m[4 * i + j];
  return o;
}

function link(gl, vsrc, fsrc) {
  const vs = compile(gl, gl.VERTEX_SHADER, vsrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`mesh shader link failed: ${gl.getProgramInfoLog(p)}`);
  }
  return p;
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(`mesh shader compile failed: ${gl.getShaderInfoLog(s)}`);
  }
  return s;
}
