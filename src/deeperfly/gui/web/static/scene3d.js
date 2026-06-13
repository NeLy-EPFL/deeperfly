// @ts-check
// A small, dependency-free 3D plot of the camera rig (and the current frame's
// triangulated pose) on a <canvas>. It exists so the operator can see where each
// camera sits and which way it looks -- shown on demand in a modal so it never
// crowds the editor.
//
// There is no 3D engine: world points are projected by a hand-rolled orbit
// camera (yaw/pitch/distance around a target) with a simple perspective divide,
// which is plenty for a schematic. Drag to orbit, wheel to zoom, double-click to
// reframe. Each camera is drawn as an RGB axis triad at its centre (x/right=red,
// y/down=green, z/optical=blue) -- the same schematic the bundle-adjustment
// notebook uses -- labelled with its name; the pose is the palette-coloured skeleton.
//
// This .js is the source -- no build step; VS Code type-checks it via `// @ts-check`.

/** @typedef {import("./types.js").Camera3D} Camera3D */
/** @typedef {import("./types.js").Point3} Point3 */

/** @typedef {[number, number, number]} Vec3 */

const WORLD_UP = /** @type {Vec3} */ ([0, 0, 1]);
const ORBIT_RATE = 0.01; // radians of orbit per pixel dragged
const WHEEL_ZOOM_RATE = 0.0015; // wheel delta -> distance factor
const PITCH_LIMIT = (Math.PI / 2) * 0.98; // clamp to avoid the gimbal pole
const NEAR = 1e-3; // points at/behind the eye are clipped

const sub = (/** @type {Vec3} */ a, /** @type {Vec3} */ b) =>
  /** @type {Vec3} */ ([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
const add = (/** @type {Vec3} */ a, /** @type {Vec3} */ b) =>
  /** @type {Vec3} */ ([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
const scale = (/** @type {Vec3} */ a, /** @type {number} */ s) =>
  /** @type {Vec3} */ ([a[0] * s, a[1] * s, a[2] * s]);
const dot = (/** @type {Vec3} */ a, /** @type {Vec3} */ b) =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (/** @type {Vec3} */ a, /** @type {Vec3} */ b) =>
  /** @type {Vec3} */ ([
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]);
const norm = (/** @type {Vec3} */ a) => {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return /** @type {Vec3} */ ([a[0] / n, a[1] / n, a[2] / n]);
};

export class Scene3D {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    /** @type {Camera3D[]} */
    this.cameras = [];
    /** @type {[number, number][]} */
    this.bones = [];
    /** @type {string[]} */
    this.colors = [];
    /** @type {Point3[] | null} */
    this.pts3d = null;

    // orbit state
    this.yaw = 0.7;
    this.pitch = 0.5;
    this.dist = 5;
    /** @type {Vec3} */
    this.target = [0, 0, 0];
    this.extent = 1; // scene radius; sizes the axis triads + the zoom range
    this.focal = 1; // pixels; set per resize

    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;

    this.canvas = canvas;
    this.ctx = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
    canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
    canvas.addEventListener("pointercancel", (e) => this.onPointerUp(e));
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => this.resetView());
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  /** @param {Camera3D[] | undefined} cameras */
  setCameras(cameras) {
    this.cameras = cameras ?? []; // an older server may omit cameras_3d
  }

  /**
   * @param {[number, number][]} bones
   * @param {[number, number, number][]} colors
   */
  setSkeleton(bones, colors) {
    this.bones = bones;
    this.colors = colors.map(([r, g, b]) => `rgb(${r},${g},${b})`);
  }

  /** @param {Point3[] | null} pts */
  setPoints3d(pts) {
    this.pts3d = pts;
    this.draw();
  }

  // Frame the whole rig: centre on everything and back the eye off proportionally.
  resetView() {
    /** @type {Vec3[]} */
    const pts = this.cameras.map((c) => c.position);
    for (const p of this.pts3d ?? []) if (p) pts.push(p);
    if (pts.length === 0) {
      this.target = [0, 0, 0];
      this.extent = 1;
    } else {
      /** @type {Vec3} */
      let c = [0, 0, 0];
      for (const p of pts) c = add(c, p);
      this.target = scale(c, 1 / pts.length);
      let r = 0;
      for (const p of pts) r = Math.max(r, Math.hypot(...sub(p, this.target)));
      this.extent = Math.max(r, 1e-3);
    }
    this.yaw = 0.7;
    this.pitch = 0.5;
    this.dist = this.extent * 2.4;
    this.draw();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = this.canvas.clientWidth || 1;
    const cssH = this.canvas.clientHeight || 1;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    this.focal = 0.5 * Math.min(cssW, cssH);
    this.draw();
  }

  // -- projection -------------------------------------------------------------

  // The current eye position and orthonormal view basis from the orbit angles.
  viewBasis() {
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    /** @type {Vec3} */
    const dir = [cp * Math.sin(this.yaw), cp * Math.cos(this.yaw), sp]; // target -> eye
    const eye = add(this.target, scale(dir, this.dist));
    const forward = norm(scale(dir, -1)); // eye -> target
    const right = norm(cross(forward, WORLD_UP));
    const up = cross(right, forward);
    return { eye, forward, right, up };
  }

  /**
   * @param {Vec3} p  a world point
   * @param {{eye: Vec3, forward: Vec3, right: Vec3, up: Vec3}} basis
   * @returns {[number, number] | null}  canvas px, or null if behind the eye
   */
  project(p, basis) {
    const rel = sub(p, basis.eye);
    const z = dot(rel, basis.forward);
    if (z <= NEAR) return null;
    const x = dot(rel, basis.right);
    const y = dot(rel, basis.up);
    const cssW = this.canvas.clientWidth || 1;
    const cssH = this.canvas.clientHeight || 1;
    return [cssW / 2 + (x / z) * this.focal, cssH / 2 - (y / z) * this.focal];
  }

  // -- drawing ----------------------------------------------------------------

  draw() {
    const ctx = this.ctx;
    const cssW = this.canvas.clientWidth || 1;
    const cssH = this.canvas.clientHeight || 1;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, cssW, cssH);
    if (this.cameras.length === 0) return;
    const basis = this.viewBasis();
    this.drawAxes(basis);
    this.drawPose(basis);
    this.cameras.forEach((cam) => this.drawCamera(cam, basis));
  }

  /** @param {{eye: Vec3, forward: Vec3, right: Vec3, up: Vec3}} basis */
  drawAxes(basis) {
    const ctx = this.ctx;
    const len = this.extent * 0.25;
    /** @type {[Vec3, string][]} */
    const axes = [
      [[len, 0, 0], "#ff5555"],
      [[0, len, 0], "#55ff55"],
      [[0, 0, len], "#5599ff"],
    ];
    const o = this.project(this.target, basis);
    if (!o) return;
    ctx.lineWidth = 1.5;
    for (const [axis, color] of axes) {
      const tip = this.project(add(this.target, axis), basis);
      if (!tip) continue;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(o[0], o[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
    }
  }

  /** @param {{eye: Vec3, forward: Vec3, right: Vec3, up: Vec3}} basis */
  drawPose(basis) {
    const pts = this.pts3d;
    if (!pts) return;
    const ctx = this.ctx;
    const screen = pts.map((p) => (p ? this.project(p, basis) : null));
    ctx.lineWidth = 2;
    for (const [a, b] of this.bones) {
      const sa = screen[a];
      const sb = screen[b];
      if (!sa || !sb) continue;
      ctx.strokeStyle = this.colors[a] || "#fff";
      ctx.beginPath();
      ctx.moveTo(sa[0], sa[1]);
      ctx.lineTo(sb[0], sb[1]);
      ctx.stroke();
    }
    for (let i = 0; i < screen.length; i++) {
      const s = screen[i];
      if (!s) continue;
      ctx.fillStyle = this.colors[i] || "#fff";
      ctx.beginPath();
      ctx.arc(s[0], s[1], 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /**
   * Draw one camera as an RGB axis triad at its centre -- x/right=red,
   * y/down=green, z/optical=blue (the rows of its rotation matrix, the same
   * schematic the bundle-adjustment notebook uses) -- labelled with its name.
   * @param {Camera3D} cam
   * @param {{eye: Vec3, forward: Vec3, right: Vec3, up: Vec3}} basis
   */
  drawCamera(cam, basis) {
    const ctx = this.ctx;
    const centre = this.project(cam.position, basis);
    if (!centre) return;
    // Triad length tracks each camera's distance from the target (like the
    // notebook's `norm(tvec) * 0.2`), so the axes read at any rig scale.
    const L = (Math.hypot(...sub(cam.position, this.target)) || this.extent) * 0.2;
    // `up` is the negated image-y row, so image-down (the green axis) is -up.
    /** @type {[Vec3, string][]} */
    const axes = [
      [cam.right, "#ff5b5b"], // image x (red)
      [scale(cam.up, -1), "#5bff5b"], // image y, pointing down (green)
      [cam.forward, "#5b9bff"], // optical axis -- the way it looks (blue)
    ];
    ctx.lineWidth = 2;
    for (const [axis, color] of axes) {
      const tip = this.project(add(cam.position, scale(axis, L)), basis);
      if (!tip) continue;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(centre[0], centre[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
    }
    ctx.fillStyle = "#ddd";
    ctx.beginPath();
    ctx.arc(centre[0], centre[1], 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.strokeText(cam.name, centre[0] + 6, centre[1]);
    ctx.fillText(cam.name, centre[0] + 6, centre[1]);
  }

  // -- interaction ------------------------------------------------------------

  /** @param {PointerEvent} e */
  onPointerDown(e) {
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.canvas.setPointerCapture(e.pointerId);
  }

  /** @param {PointerEvent} e */
  onPointerMove(e) {
    if (!this.dragging) return;
    this.yaw -= (e.clientX - this.lastX) * ORBIT_RATE;
    this.pitch += (e.clientY - this.lastY) * ORBIT_RATE;
    this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.draw();
  }

  /** @param {PointerEvent} e */
  onPointerUp(e) {
    this.dragging = false;
    if (this.canvas.hasPointerCapture(e.pointerId)) {
      this.canvas.releasePointerCapture(e.pointerId);
    }
  }

  /** @param {WheelEvent} e */
  onWheel(e) {
    e.preventDefault();
    this.dist *= Math.exp(e.deltaY * WHEEL_ZOOM_RATE);
    this.dist = Math.max(this.extent * 0.2, Math.min(this.extent * 20, this.dist));
    this.draw();
  }
}
