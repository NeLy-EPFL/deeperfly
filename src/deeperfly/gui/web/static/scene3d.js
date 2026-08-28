// @ts-check
// A small, dependency-free 3D view of the scene -- the camera rig, the current
// frame's triangulated pose, the fitted NeuroMechFly skeleton, and the posed NMF
// mesh -- on a <canvas>. It is shown on demand in a floating panel (see app.js) that
// overlays the editor without blocking it, and lets the operator inspect the 3D
// reconstruction the overlays project.
//
// There is no 3D engine for the schematic parts: world points are projected by a
// hand-rolled orbit camera (yaw/pitch/distance around a target) with a simple
// perspective divide. The virtual lens is fairly long (`FOCAL_FACTOR`, ~35deg field
// of view) so the perspective is gentle -- a wide lens made near parts of the fly
// balloon. Drag to orbit, Shift/right-drag to pan, wheel to zoom,
// double-click to reframe. Each camera is an RGB axis triad at its centre
// (x/right=red, y/down=green, z/optical=blue) -- the same schematic the
// bundle-adjustment notebook uses -- labelled with its name; the triangulated pose is
// the palette-coloured skeleton and the NMF skeleton is mint (matching the 2D overlay).
//
// The NMF mesh is rendered by the shared WebGL `MeshGL` (a callback set from app.js):
// this view hands it a synthetic pinhole camera built from the orbit basis -- chosen
// so its projection matches `project()` below pixel-for-pixel -- and composites the
// returned canvas translucently behind the schematic, so the skeletons read on top.
//
// This .js is the source -- no build step; VS Code type-checks it via `// @ts-check`.

/** @typedef {import("./types.js").Camera3D} Camera3D */
/** @typedef {import("./types.js").CameraProj} CameraProj */
/** @typedef {import("./types.js").Point3} Point3 */

/** @typedef {[number, number, number]} Vec3 */

// The fitted NMF skeleton's colour (mint), the same the 2D reprojection overlay uses.
const NMF_COLOR = "rgba(80,230,180,0.95)";

const WORLD_UP = /** @type {Vec3} */ ([0, 0, 1]);
const ORIGIN = /** @type {Vec3} */ ([0, 0, 0]); // the world origin, drawn as the axis triad
const ORBIT_RATE = 0.01; // radians of orbit per pixel dragged
const WHEEL_ZOOM_RATE = 0.0015; // wheel delta -> distance factor
const PITCH_LIMIT = (Math.PI / 2) * 0.98; // clamp to avoid the gimbal pole
const NEAR = 1e-3; // points at/behind the eye are clipped

// A long-ish virtual lens: focal = FOCAL_FACTOR * min(W, H) px, i.e. a ~35deg field of
// view, so perspective is gentle (was 0.5 ~ 90deg, which looked fish-eyed up close).
const FOCAL_FACTOR = 1.6;
// Reset/zoom framing is set as the fraction of the half-frame a world radius should
// fill, converted to a camera distance by `fillDist` -- so the framing is independent
// of both the lens and the canvas resolution.
const RESET_FILL = 0.42; // the whole scene radius fills ~42% of the half-frame at reset
const ZOOM_IN_FILL = 50; // closest: the fly radius may overfill to ~50x the half-frame
const ZOOM_OUT_FILL = 0.05; // farthest: the scene radius shrinks to ~5% of the half-frame

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
    this.edges = [];
    /** @type {string[]} */
    this.colors = [];
    /** @type {string[]} */
    this.edgeColors = [];
    /** @type {Point3[] | null} */
    this.pts3d = null;
    /** @type {Point3[] | null} */
    this.nmf3d = null;
    // The shared WebGL mesh renderer: (cam, supersample) -> a canvas, or null when
    // the mesh is unavailable. Set from app.js; this view only supplies the camera.
    /** @type {((cam: CameraProj, ss: number) => (HTMLCanvasElement | null)) | null} */
    this.meshRenderer = null;

    // What is drawn (toggled from the panel); the NMF layers are also gated by data.
    this.showCameras = true;
    this.showPose = true;
    this.showNmf = true;
    this.showMesh = true;
    this.showAxes = true; // the world-origin X/Y/Z triad

    // orbit state
    this.yaw = 0.7;
    this.pitch = 0.5;
    this.dist = 5;
    /** @type {Vec3} */
    this.target = [0, 0, 0];
    this.extent = 1; // scene radius (rig + pose); sizes the axis triads + zoom-out
    this.flyScale = 1; // the fly's own radius (rig-independent); sets the zoom-in limit
    this.focal = 1; // pixels; set per resize

    this.dragging = false;
    this.panning = false; // this drag pans (Shift / middle / right) rather than orbits
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
    canvas.addEventListener("contextmenu", (e) => e.preventDefault()); // right-drag pans
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  /** @param {Camera3D[] | undefined} cameras */
  setCameras(cameras) {
    this.cameras = cameras ?? []; // an older server may omit cameras_3d
  }

  /**
   * @param {[number, number][]} edges
   * @param {[number, number, number][]} colors  one per POINT
   * @param {[number, number, number][]} [edgeColors]  one per EDGE; the skeleton's own,
   *   which is not any endpoint's colour once an edge is coloured explicitly.
   */
  setSkeleton(edges, colors, edgeColors) {
    const css = ([r, g, b]) => `rgb(${r},${g},${b})`;
    this.edges = edges;
    this.colors = colors.map(css);
    this.edgeColors = (edgeColors || []).map(css);
  }

  /** @param {Point3[] | null} pts  the triangulated pose for this frame */
  setPoints3d(pts) {
    this.pts3d = pts;
    this.draw();
  }

  /** @param {Point3[] | null} pts  the fitted NMF skeleton joints for this frame */
  setNmf3d(pts) {
    this.nmf3d = pts;
    this.draw();
  }

  /**
   * @param {((cam: CameraProj, ss: number) => (HTMLCanvasElement | null)) | null} fn
   *   renders the posed NMF mesh through a pinhole camera (the shared WebGL renderer)
   */
  setMeshRenderer(fn) {
    this.meshRenderer = fn;
  }

  /**
   * Toggle a layer's visibility and repaint. Keys: `cameras`, `pose`, `nmf`, `mesh`, `axes`.
   * @param {Partial<{cameras: boolean, pose: boolean, nmf: boolean, mesh: boolean, axes: boolean}>} vis
   */
  setVisibility(vis) {
    if (vis.cameras !== undefined) this.showCameras = vis.cameras;
    if (vis.pose !== undefined) this.showPose = vis.pose;
    if (vis.nmf !== undefined) this.showNmf = vis.nmf;
    if (vis.mesh !== undefined) this.showMesh = vis.mesh;
    if (vis.axes !== undefined) this.showAxes = vis.axes;
    this.draw();
  }

  // Frame the whole scene, but centre on the fly (so zoom + pan home in on the model,
  // not on empty space between the rig and the fly). `extent` spans the rig for
  // zoom-out; `flyScale` is the fly's own radius, so the zoom-in limit lets you get
  // right up to the model however large the rig is.
  resetView() {
    /** @type {Vec3[]} */
    const camPts = this.cameras.map((c) => c.position);
    /** @type {Vec3[]} */
    const flyPts = [];
    for (const p of this.pts3d ?? []) if (p) flyPts.push(p);
    for (const p of this.nmf3d ?? []) if (p) flyPts.push(p);
    const focus = flyPts.length ? flyPts : camPts; // centre on the fly when present
    const all = [...camPts, ...flyPts];
    if (all.length === 0) {
      this.target = [0, 0, 0];
      this.extent = 1;
      this.flyScale = 1;
    } else {
      /** @type {Vec3} */
      let c = [0, 0, 0];
      for (const p of focus) c = add(c, p);
      this.target = scale(c, 1 / focus.length);
      let r = 0;
      for (const p of all) r = Math.max(r, Math.hypot(...sub(p, this.target)));
      this.extent = Math.max(r, 1e-3);
      let fr = 0;
      for (const p of flyPts) fr = Math.max(fr, Math.hypot(...sub(p, this.target)));
      this.flyScale = flyPts.length ? Math.max(fr, 1e-3) : this.extent;
    }
    this.yaw = 0.7;
    this.pitch = 0.5;
    this.dist = this.fillDist(this.extent, RESET_FILL);
    this.draw();
  }

  // The camera distance at which a sphere of the given world `radius` fills `fill` of
  // the half-frame. Since focal = FOCAL_FACTOR * minDim, the minDim cancels, so the
  // result is purely geometric (independent of the canvas size).
  /** @param {number} radius @param {number} fill */
  fillDist(radius, fill) {
    return (radius * 2 * FOCAL_FACTOR) / fill;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = this.canvas.clientWidth || 1;
    const cssH = this.canvas.clientHeight || 1;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    this.focal = FOCAL_FACTOR * Math.min(cssW, cssH);
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

  // A synthetic pinhole camera (deeperfly's world->camera convention) for the orbit
  // view, so `MeshGL` renders the mesh exactly aligned with `project()` above. The
  // intrinsics put a focal of `this.focal` px at a centred principal point; the
  // extrinsics map the world into a camera frame whose rows are right / image-down
  // (= -up) / forward, translated so the eye is the origin.
  /** @returns {CameraProj} */
  meshCam() {
    const { eye, forward, right, up } = this.viewBasis();
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    return {
      name: "orbit",
      intr: [this.focal, this.focal, w / 2, h / 2],
      rmat: [
        right[0], right[1], right[2],
        -up[0], -up[1], -up[2],
        forward[0], forward[1], forward[2],
      ],
      tvec: [-dot(eye, right), dot(eye, up), -dot(eye, forward)],
      size: [w, h],
    };
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
    // The mesh is a translucent backdrop (rendered by WebGL); the schematic draws on
    // top, so the skeletons stay legible over (and through) the body.
    this.drawMesh(cssW, cssH);
    if (this.showAxes) this.drawAxes(basis);
    if (this.showPose)
      this.drawSkeleton(
        this.pts3d,
        basis,
        (i) => this.colors[i] || "#fff",
        (k, a) => this.edgeColors[k] || this.colors[a] || "#fff",
      );
    if (this.showNmf) this.drawSkeleton(this.nmf3d, basis, () => NMF_COLOR);
    if (this.showCameras) this.cameras.forEach((cam) => this.drawCamera(cam, basis));
  }

  // Composite the posed NMF mesh (WebGL) translucently behind the schematic. The
  // orbit camera matches `project()`, so it lands pixel-aligned with the skeletons.
  /** @param {number} cssW @param {number} cssH */
  drawMesh(cssW, cssH) {
    if (!this.showMesh || !this.meshRenderer) return;
    const ss = Math.min(3, window.devicePixelRatio || 1);
    const rendered = this.meshRenderer(this.meshCam(), ss);
    if (!rendered) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.drawImage(rendered, 0, 0, cssW, cssH);
    ctx.restore();
  }

  // The world origin as a labelled X/Y/Z triad (x=red, y=green, z=blue), so the
  // operator can see where (0,0,0) is and how the world axes lie. Toggled via `showAxes`.
  /** @param {{eye: Vec3, forward: Vec3, right: Vec3, up: Vec3}} basis */
  drawAxes(basis) {
    const ctx = this.ctx;
    const len = this.extent * 0.25;
    /** @type {[Vec3, string, string][]} */
    const axes = [
      [[len, 0, 0], "#ff5555", "x"],
      [[0, len, 0], "#55ff55", "y"],
      [[0, 0, len], "#5599ff", "z"],
    ];
    const o = this.project(ORIGIN, basis);
    if (!o) return;
    ctx.lineWidth = 1.5;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const [axis, color, label] of axes) {
      const tip = this.project(axis, basis); // axis is the world point (origin + offset)
      if (!tip) continue;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(o[0], o[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(label, tip[0] + 3, tip[1]);
    }
  }

  /**
   * Draw a skeleton (the shared bones) from world points, each joint/bone coloured by
   * `colorAt(pointIndex)`. Used for both the triangulated pose (palette) and the
   * fitted NMF skeleton (mint).
   * @param {Point3[] | null} pts
   * @param {{eye: Vec3, forward: Vec3, right: Vec3, up: Vec3}} basis
   * @param {(i: number) => string} colorAt
   * @param {((k: number, a: number) => string)=} edgeColorAt
   */
  drawSkeleton(pts, basis, colorAt, edgeColorAt) {
    if (!pts) return;
    const ctx = this.ctx;
    const screen = pts.map((p) => (p ? this.project(p, basis) : null));
    ctx.lineWidth = 2;
    for (let k = 0; k < this.edges.length; k++) {
      const [a, b] = this.edges[k];
      const sa = screen[a];
      const sb = screen[b];
      if (!sa || !sb) continue;
      ctx.strokeStyle = edgeColorAt ? edgeColorAt(k, a) : colorAt(a);
      ctx.beginPath();
      ctx.moveTo(sa[0], sa[1]);
      ctx.lineTo(sb[0], sb[1]);
      ctx.stroke();
    }
    for (let i = 0; i < screen.length; i++) {
      const s = screen[i];
      if (!s) continue;
      ctx.fillStyle = colorAt(i);
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
    // Shift, the middle button, or the right button pans; a plain left drag orbits.
    this.panning = e.shiftKey || e.button === 1 || e.button === 2;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.canvas.setPointerCapture(e.pointerId);
  }

  /** @param {PointerEvent} e */
  onPointerMove(e) {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    if (this.panning) {
      this.pan(dx, dy);
    } else {
      this.yaw -= dx * ORBIT_RATE;
      this.pitch += dy * ORBIT_RATE;
      this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
    }
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.draw();
  }

  // Slide the target (and so the eye) in the view plane, so the grabbed point tracks
  // the cursor: one screen pixel is ~dist/focal world units at the target's depth.
  /** @param {number} dx @param {number} dy  pointer deltas in CSS pixels */
  pan(dx, dy) {
    const { right, up } = this.viewBasis();
    const perPx = this.dist / this.focal;
    this.target = add(this.target, add(scale(right, -dx * perPx), scale(up, dy * perPx)));
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
    // Zoom in until the fly (its own radius, not the rig's) overfills the view, so the
    // model can be inspected up close; zoom out until the whole rig is small.
    const minDist = this.fillDist(this.flyScale, ZOOM_IN_FILL);
    const maxDist = this.fillDist(this.extent, ZOOM_OUT_FILL);
    this.dist = Math.max(minDist, Math.min(maxDist, this.dist));
    this.draw();
  }
}
