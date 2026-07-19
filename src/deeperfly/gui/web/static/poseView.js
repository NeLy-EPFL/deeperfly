// @ts-check
// One camera's frame plus its draggable 2D skeleton overlay, on a <canvas>.
//
// A port of the old Qt PoseView, grown a few editor conveniences: the frame is
// drawn fit-to-canvas (letterboxed) and can be zoomed (mouse wheel or trackpad
// pinch, toward the cursor) and panned (drag on empty space, or a trackpad
// two-finger scroll). Frames are loaded via `loadFrame`; while a
// new one decodes the previous frame stays up *blurred* (a cheap upscaled
// thumbnail, not a per-frame filter) so scrubbing never flashes black -- the blur
// reads as "not the live frame yet". The skeleton is drawn in image-pixel
// coordinates mapped through that fit+zoom. Pointer events pick the nearest joint
// within a screen-pixel tolerance. A press on a joint selects it; an actual drag
// (past a small threshold) moves it, emitting a throttled `onDragging` and a final
// `onDragged` -- a click without movement just selects, so it never creates a
// spurious edit. A click on empty space clears the selection, while a drag on empty
// space pans (so panning never deselects). Holding a modifier turns a click or a
// rubber-band drag into a multi-select: Shift replaces the selection with what the
// gesture picks, Ctrl/Cmd adds to it (Ctrl/Cmd+click toggles a single joint).
// Right-click toggles a point's fixed flag. An occluded joint has no
// observed pixel, so it draws as a derived (reprojected) point; dragging it is still
// allowed and reports `wasInvisible` on `onDragged` so the app can un-occlude it.
// Hovering a joint reports it via `onHover` so the app can emphasize the same point
// across every view. The app stays in control of what a drag does to the 3D point.
//
// Beyond the editable overlay the view can also draw the individual point *sources* as
// their own layers -- ground truth, the raw detector prediction, and the 3D reprojection
// ("projected") -- each toggled independently, so the operator can inspect the full set
// of one source at a time. The editable skeleton is the "Combined" layer: one integrated
// skeleton that picks each joint by precedence (ground truth > detected > projected). It
// is the only interactive layer; the source layers are read-only.
//
// The view also draws two read-only reference skeletons the app toggles -- the "latent"
// skeleton (the current 3D estimate reprojected, i.e. the projected source) and the
// fitted NMF model -- plus per-joint name labels. Shown *with* the combined skeleton, a
// reference is a faint under-glow *beneath* it (so the limb palette always reads on top)
// with its *disagreement* against the placed point drawn back on top as a per-joint
// "leash" that is silent at coincidence and grows with the residual -- so a live 3D drag
// shows exactly where triangulation is pulling each point. With the combined skeleton
// toggled off a source/reference instead draws in its own bright, dashed/dotted style so
// it stays fully visible on its own.
//
// This .js is the source -- no build step; VS Code type-checks it via
// `// @ts-check` and the JSDoc types.

/** @typedef {import("./types.js").Point} Point */

/**
 * @typedef {object} PoseViewCallbacks
 * @property {(view: number, point: number, x: number, y: number) => void} onDragging
 * @property {(view: number, point: number, x: number, y: number, wasInvisible: boolean) => void} onDragged
 * @property {(view: number, point: number) => void} onToggleFixed
 * @property {(view: number, point: number, additive: boolean) => void} onSelect  a joint was clicked/grabbed; additive (Ctrl/Cmd) toggles it in the selection, else it replaces
 * @property {(view: number, points: number[], additive: boolean) => void} onSelectRegion  a marquee enclosed these joints in this view; additive (Ctrl/Cmd-drag) adds them, else (Shift-drag) replaces the selection
 * @property {(point: number, additive: boolean) => void} onSelectKeypointAllViews  a joint was double-clicked: select it across every view
 * @property {() => void} onBackground  the background was clicked (empty space, no drag): clear the selection
 * @property {(view: number) => void} onActiveView  the pointer entered/moved over this view (drives the "select all in this view" gesture)
 * @property {(point: number | null) => void} onHover  the hovered joint changed (cross-view)
 */

const POINT_RADIUS_PX = 4; // drawn joint radius in screen px (constant under zoom)
const HOVER_SCALE = 1.6; // how much a hovered joint grows (its bones thicken too, so it needn't balloon)
const HOVER_BONE_WIDTH = 3; // a hovered joint's connected bones thicken to this (screen px)
const HIT_TOLERANCE_PX = 14; // how close a click must be to grab a joint, screen px
const DRAG_THRESHOLD_PX = 3; // movement (screen px) before a press becomes a drag
const MAX_ZOOM = 10; // cap on the user wheel-zoom factor over fit
const WHEEL_ZOOM_RATE = 0.0015; // mouse-wheel delta -> zoom factor sensitivity
const PINCH_ZOOM_RATE = 0.01; // trackpad pinch: a higher gain than the wheel (its per-event delta is tiny) so the pinch tracks the fingers
const WHEEL_NOTCH_MIN = 50; // |deltaY| (px) at/above which a wheel step reads as a chunky mouse-wheel notch; a trackpad's steps are smaller / fractional / horizontal
const WHEEL_BURST_GAP_MS = 150; // a quiet gap this long ends a wheel "burst" -- the next event re-classifies the device (mouse vs trackpad)
const BONE_WIDTH = 1.5; // the editable skeleton's bone width (screen px)

// Each joint marker encodes its *source* -- the whole point of the unified editor is
// that one glance tells you where a point came from:
//   ground truth (authored)  -> solid lime ring over a filled disc
//   detector prediction      -> thin dark ring over a filled disc that fades with confidence
//   derived (reprojected 3D) -> a hollow circle in the point's own limb palette
//     colour, no fill: "computed, not observed". A view the operator occluded shows the
//     same way -- occluding just deletes the observation, leaving the point derived, so
//     there is nothing to distinguish it from a view the detector never fired in.
const FIXED_COLOR = "#7CFC00"; // ring on a ground-truth point (lime green)
const SELECT_COLOR = "#3fd0ff"; // ring on a selected point (cyan; lime = ground truth)
const MARQUEE_STROKE = "rgba(63,208,255,0.9)"; // Shift+drag rubber-band border (cyan, matches selection)
const MARQUEE_FILL = "rgba(63,208,255,0.12)"; // its translucent fill
const MARQUEE_ADD_STROKE = "rgba(124,252,0,0.9)"; // Ctrl/⌘-drag = add: lime, matching the add affordance
const MARQUEE_ADD_FILL = "rgba(124,252,0,0.14)"; // its translucent fill

// The two read-only reference overlays -- the latent 3D reprojection (amber) and the
// fitted NMF model (mint) -- are drawn UNDERNEATH the editable skeleton, so the
// limb palette always owns the top layer instead of being painted over. Each is
// drawn as a soft under-glow skeleton; its overall shape (and where it bends away from
// the palette) is the ambient signal. Its *disagreement* with the placed point at one
// joint -- a "leash" from the point to where the reference lands -- is drawn on top
// only for the joint under the cursor (or being dragged), so the default view stays
// uncluttered and the difference line appears on demand. Each overlay is identifiable
// without relying on colour: the editable skeleton is a solid line with a filled disc,
// the latent a dashed line with a hollow ring, the NMF a dotted line with a hollow square.
const LATENT_RGB = "255,176,64"; // amber
const NMF_RGB = "80,230,180"; // mint
const GHOST_ALPHA = 0.4; // a reference overlay's soft under-glow (a faint halo beneath the palette)
const LEASH_MIN_PX = 2.5; // below this editable<->reference screen gap the leash needs no connector line
const LEASH_FULL_PX = 16; // at/above this gap the reference marker + leash reach full emphasis

/**
 * Guess whether a wheel event came from a trackpad (vs a mouse wheel), from the shape of
 * the first event in a burst. A mouse wheel and a trackpad scroll are indistinguishable
 * *mid-gesture* (a momentum flick's peak deltas rival a mouse notch), so the caller only
 * trusts this at a burst's start, when a trackpad's deltas are still small. Classified by
 * delta shape ONLY -- ctrlKey (a pinch, or Ctrl+wheel) is deliberately NOT a tell here, so a
 * transient ctrlKey can't latch the persistent device flag (releasing Ctrl mid-burst on a
 * Ctrl+mouse-wheel would otherwise flip to panning); onWheel handles ctrlKey as a per-event
 * zoom short-circuit instead, and a real trackpad pinch is still caught by its small delta.
 * Tells:
 *   - line/page delta mode (Firefox mouse wheel) is a mouse
 *   - a horizontal or fractional delta is a trackpad
 *   - a small vertical pixel step is a trackpad; a large one is a mouse-wheel notch
 * @param {WheelEvent} e
 * @returns {boolean}
 */
function looksLikeTrackpad(e) {
  if (e.deltaMode !== 0) return false;
  if (e.deltaX !== 0) return true;
  if (!Number.isInteger(e.deltaY)) return true;
  return Math.abs(e.deltaY) < WHEEL_NOTCH_MIN;
}

export class PoseView {
  /**
   * @param {number} viewIndex
   * @param {HTMLCanvasElement} canvas
   * @param {PoseViewCallbacks} cb
   */
  constructor(viewIndex, canvas, cb) {
    /** @type {HTMLImageElement | null} */
    this.img = null;
    /** @type {HTMLCanvasElement | null} */
    this.meshImg = null; // posed NMF mesh overlay (this view's copy), drawn when meshVisible
    /** @type {HTMLCanvasElement | null} */
    this.meshCanvas = null; // backing 2D canvas the GPU render is copied into
    /** @type {[number, number][]} */
    this.bones = [];
    /** @type {string[]} */
    this.colors = [];
    /** @type {Point[]} */
    this.pts = [];
    /** @type {Point[] | null} */
    this.latent = null; // latent 3D reprojection (display only) = the "projected" source
    /** @type {Point[] | null} */
    this.detected = null; // raw detector prediction per point (display only) = the "detected" source
    /** @type {Point[] | null} */
    this.nmf = null; // fitted NMF model reprojection (display only), drawn when nmfVisible
    /** @type {boolean[] | null} */
    this.fixed = null;
    /** @type {boolean[] | null} */
    this.invisible = null;
    /** @type {(number | null)[] | null} */
    this.conf = null; // per-point detector confidence, for the low-confidence fade
    /** @type {string[]} */
    this.pointNames = [];
    /** @type {number | null} */
    this.highlight = null; // hovered joint (set by the app across all views)
    /** @type {Set<number>} */
    this.selectionSet = new Set(); // this view's selected joint indices (cyan ring)
    // Modifier+drag rubber-band, in CSS px, while a marquee is in progress (else null).
    /** @type {{ x0: number, y0: number, x1: number, y1: number } | null} */
    this.marquee = null;
    this.marqueeing = false; // a Shift/Ctrl press is arming/dragging a marquee
    this.marqueeAdditive = false; // Ctrl/Cmd-drag adds to the selection; Shift-drag replaces it
    this.editable = false;
    this.zoomable = false;
    // Point-source layer toggles. "Combined" is the integrated editable skeleton
    // (precedence GT > detected > projected) and the only interactive layer; the three
    // source layers are read-only. Combined is on by default (the plain editing view).
    this.combinedVisible = true;
    this.gtVisible = false;
    this.detectedVisible = false;
    this.projectedVisible = false;
    this.nmfVisible = false;
    this.meshVisible = false;
    this.labelsVisible = false;
    /** @type {number | null} */
    this.dragging = null;
    this.dragInvisible = false; // was the grabbed joint obscured? (reported on release)
    this.panning = false;
    this.moved = false; // has the current press moved past the drag threshold?

    // Frame loading: while a newer frame decodes we keep drawing the last-loaded
    // one *blurred* (so the view never blanks to black mid-scrub, but the blur
    // reads as "this isn't the live frame yet"). `loadToken` drops out-of-order
    // loads when scrubbing fast.
    this.stale = false;
    this.loadToken = 0;
    /** @type {HTMLCanvasElement | null} */
    this.staleCanvas = null; // a tiny downscaled copy of img -- upscaled = cheap blur

    // image -> CSS-pixel fit (recomputed on resize / new image)
    this.imgW = 1;
    this.imgH = 1;
    // user zoom/pan applied on top of the fit
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    // effective transform (fit * zoom/pan), recomputed by applyTransform()
    this.fitScale = 1;
    this.fitOffX = 0;
    this.fitOffY = 0;
    this.scale = 1;
    this.offX = 0;
    this.offY = 0;

    // press bookkeeping (CSS px) for the click-vs-drag threshold and panning
    this.downX = 0;
    this.downY = 0;
    this.panOrigX = 0;
    this.panOrigY = 0;
    // Wheel-gesture classification: mouse wheel and trackpad look alike mid-gesture, so the
    // device is decided once at the START of each wheel burst (see looksLikeTrackpad) and
    // held until a quiet gap starts a new burst -- keeping a momentum flick from flipping
    // to "mouse" (and zooming) when its peak deltas briefly rival a wheel notch.
    this._wheelIsTrackpad = false;
    this._lastWheelTime = 0;
    /** @type {number | null} */
    this._hover = null; // last hover reported, to debounce onHover

    /** @type {{ x: number, y: number } | null} */
    this.pendingDrag = null;
    this.rafId = 0;

    this.viewIndex = viewIndex;
    this.canvas = canvas;
    this.ctx = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
    this.cb = cb;

    canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
    canvas.addEventListener("pointercancel", (e) => this.onPointerUp(e));
    canvas.addEventListener("pointerleave", () => this.onPointerLeave());
    // Whichever view the pointer is over is the "active" one -- the target of the
    // "select every point in this view" (v) gesture.
    canvas.addEventListener("pointerenter", () => this.cb.onActiveView(this.viewIndex));
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("dblclick", (e) => this.onDblClick(e));
    new ResizeObserver(() => this.layoutAndDraw()).observe(canvas);
  }

  // -- setup ------------------------------------------------------------------

  /**
   * @param {[number, number][]} bones
   * @param {[number, number, number][]} colors
   */
  setSkeleton(bones, colors) {
    this.bones = bones;
    this.colors = colors.map(([r, g, b]) => `rgb(${r},${g},${b})`);
  }

  /** @param {string[]} names  per-point labels, drawn when labels are visible */
  setPointNames(names) {
    this.pointNames = names;
  }

  /** @param {HTMLImageElement} img */
  setImage(img) {
    this.img = img;
    this.imgW = img.naturalWidth || this.imgW;
    this.imgH = img.naturalHeight || this.imgH;
    this.stale = false;
    this.layoutAndDraw();
  }

  /**
   * Load a new frame image by URL. Until it decodes, the last-loaded frame stays
   * on screen blurred (see `stale`), so scrubbing never flashes black. Loads that
   * a faster scrub supersedes are dropped via `loadToken`.
   * @param {string} url
   */
  loadFrame(url) {
    const token = ++this.loadToken;
    // Snapshot the current frame *once* on the sharp -> stale transition; while
    // stale `img` doesn't change, so the snapshot (and its blur) stays valid.
    if (this.img && !this.stale) {
      this.renderStaleThumb();
      this.stale = true;
    }
    this.draw();
    const img = new Image();
    img.onload = () => {
      if (token !== this.loadToken) return; // a newer frame already supersedes this
      this.setImage(img);
    };
    img.src = url;
  }

  /** @param {boolean} visible  whether the posed NMF mesh is drawn over the frame */
  setMeshVisible(visible) {
    if (this.meshVisible === visible) return;
    this.meshVisible = visible;
    this.draw();
  }

  /**
   * Capture this view's posed mesh from the shared WebGL canvas into its own
   * backing canvas (the GL canvas is reused across views, so each view keeps a
   * copy to composite under its skeleton). Pass `null` to clear the overlay.
   * @param {HTMLCanvasElement | null} source  the rendered GL canvas (footage-sized)
   */
  captureMesh(source) {
    if (!source) {
      this.meshImg = null;
      if (this.meshVisible) this.draw();
      return;
    }
    if (
      !this.meshCanvas ||
      this.meshCanvas.width !== source.width ||
      this.meshCanvas.height !== source.height
    ) {
      this.meshCanvas = document.createElement("canvas");
      this.meshCanvas.width = source.width;
      this.meshCanvas.height = source.height;
    }
    const mctx = this.meshCanvas.getContext("2d");
    mctx.clearRect(0, 0, this.meshCanvas.width, this.meshCanvas.height);
    mctx.drawImage(source, 0, 0);
    this.meshImg = this.meshCanvas;
    if (this.meshVisible) this.draw();
  }

  // Downscale the current frame into a tiny offscreen canvas. Drawing that small
  // canvas back up to full size (with smoothing) is the blur -- far cheaper than a
  // per-draw `ctx.filter`, which matters because draw() also runs on every hover.
  renderStaleThumb() {
    const img = this.img;
    if (!img) return;
    const MAX = 64; // longest side of the downscaled copy
    const w = img.naturalWidth || this.imgW;
    const h = img.naturalHeight || this.imgH;
    const s = Math.min(1, MAX / Math.max(w, h));
    const tw = Math.max(1, Math.round(w * s));
    const th = Math.max(1, Math.round(h * s));
    const c = this.staleCanvas ?? (this.staleCanvas = document.createElement("canvas"));
    c.width = tw;
    c.height = th;
    const cx = /** @type {CanvasRenderingContext2D} */ (c.getContext("2d"));
    cx.clearRect(0, 0, tw, th);
    cx.drawImage(img, 0, 0, tw, th);
  }

  // Image size hint so the canvas keeps the right aspect before the first frame.
  /**
   * @param {number} height
   * @param {number} width
   */
  setImageSize(height, width) {
    this.imgH = height;
    this.imgW = width;
    this.layoutAndDraw();
  }

  /**
   * Apply a fresh points-payload slice for this view in a *single* repaint. The
   * per-reply update used to call one setter per field, each doing its own draw()
   * (3-5 full-canvas redraws per view per reply); a live 3D drag streams many
   * replies, so folding them into one draw keeps the canvas from thrashing.
   *
   * A field left `undefined` is kept as-is: the server omits `nmf` on mid-drag
   * replies (it skips the costly per-frame re-fit), so the model overlay just holds
   * until the drag settles instead of flickering off.
   *
   * @param {object} data
   * @param {Point[]} [data.points]
   * @param {boolean[] | null} [data.fixed]
   * @param {boolean[] | null} [data.invisible]  per-point "obscured" mask, or null when not in 3D
   * @param {(number | null)[] | null} [data.conf]  per-point detector confidence (low fades the fill)
   * @param {Point[] | null} [data.latent]  the latent 3D reprojection to ghost, or null
   * @param {Point[] | null} [data.detected]  the raw detector prediction (the "detected" source), or null
   * @param {Point[] | null} [data.nmf]  the fitted NMF model reprojection to ghost, or null
   */
  setFrameData(data) {
    if (data.points) {
      // Keep the actively dragged joint pinned to the cursor: the server's live
      // re-solve reprojects it a hair off, and letting that fight the mouse feels
      // like resistance (mirrors the old Qt PoseView.set_points).
      const held = this.dragging !== null ? this.pts[this.dragging] : null;
      this.pts = data.points.slice();
      if (this.dragging !== null && held) this.pts[this.dragging] = held;
    }
    if (data.fixed !== undefined) this.fixed = data.fixed;
    if (data.invisible !== undefined) this.invisible = data.invisible;
    if (data.conf !== undefined) this.conf = data.conf;
    if (data.latent !== undefined) this.latent = data.latent;
    // The raw detections are static within a frame, so the mid-drag stream omits them
    // (undefined) and this view keeps the set from the last plain/navigation fetch.
    if (data.detected !== undefined) this.detected = data.detected;
    if (data.nmf !== undefined) this.nmf = data.nmf;
    this.draw();
  }

  /** @param {number | null} point */
  setHighlight(point) {
    if (this.highlight === point) return;
    this.highlight = point;
    this.draw();
  }

  /** @param {Set<number>} set  the joint indices selected in this view (may be empty) */
  setSelection(set) {
    this.selectionSet = set;
    this.draw();
  }

  /** @param {boolean} editable */
  setEditable(editable) {
    this.editable = editable;
  }

  /** @param {boolean} zoomable  whether wheel-zoom + pan are allowed (large views only) */
  setZoomable(zoomable) {
    if (this.zoomable === zoomable) return;
    this.zoomable = zoomable;
    if (!zoomable) this.resetZoom(); // thumbnails always show the whole frame
  }

  /** @param {boolean} visible  whether the combined editable skeleton + joints are drawn */
  setCombinedVisible(visible) {
    if (this.combinedVisible === visible) return;
    this.combinedVisible = visible;
    this.draw();
  }

  /** @param {boolean} visible  whether the ground-truth source layer is drawn */
  setGtVisible(visible) {
    if (this.gtVisible === visible) return;
    this.gtVisible = visible;
    this.draw();
  }

  /** @param {boolean} visible  whether the detected (raw prediction) source layer is drawn */
  setDetectedVisible(visible) {
    if (this.detectedVisible === visible) return;
    this.detectedVisible = visible;
    this.draw();
  }

  /** @param {boolean} visible  whether the projected (3D reprojection) source layer is drawn */
  setProjectedVisible(visible) {
    if (this.projectedVisible === visible) return;
    this.projectedVisible = visible;
    this.draw();
  }

  /** @param {boolean} visible  whether the fitted NMF model is ghosted on top */
  setNmfVisible(visible) {
    if (this.nmfVisible === visible) return;
    this.nmfVisible = visible;
    this.draw();
  }

  /** @param {boolean} visible  whether per-joint name labels are drawn */
  setLabelsVisible(visible) {
    if (this.labelsVisible === visible) return;
    this.labelsVisible = visible;
    this.draw();
  }

  resetZoom() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.applyTransform();
    this.draw();
  }

  // -- layout + drawing -------------------------------------------------------

  layoutAndDraw() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = this.canvas.clientWidth || 1;
    const cssH = this.canvas.clientHeight || 1;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    this.fitScale = Math.min(cssW / this.imgW, cssH / this.imgH);
    this.fitOffX = (cssW - this.imgW * this.fitScale) / 2;
    this.fitOffY = (cssH - this.imgH * this.fitScale) / 2;
    this.applyTransform();
    this.draw();
  }

  // Fold the user zoom/pan into the effective image->canvas transform.
  applyTransform() {
    this.scale = this.fitScale * this.zoom;
    this.offX = this.fitOffX + this.panX;
    this.offY = this.fitOffY + this.panY;
  }

  /**
   * @param {number} x
   * @param {number} y
   * @returns {[number, number]}
   */
  toCanvas(x, y) {
    return [this.offX + x * this.scale, this.offY + y * this.scale];
  }

  draw() {
    const ctx = this.ctx;
    const cssW = this.canvas.clientWidth || 1;
    const cssH = this.canvas.clientHeight || 1;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cssW, cssH);
    const dw = this.imgW * this.scale;
    const dh = this.imgH * this.scale;
    if (this.stale && this.staleCanvas) {
      // Blurred placeholder: upscaling the tiny snapshot (with smoothing on) is the
      // blur, which already reads as "loading, not the live frame". Don't dim it --
      // a brightness change flickers as the sharp frame swaps back in.
      const smooth = ctx.imageSmoothingEnabled;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.staleCanvas, this.offX, this.offY, dw, dh);
      ctx.imageSmoothingEnabled = smooth;
    } else if (this.img) {
      ctx.drawImage(this.img, this.offX, this.offY, dw, dh);
    }
    // The posed NMF mesh sits between the frame and the editable skeleton, so the
    // keypoints stay legible on top of it. The GPU renders the silhouette opaque;
    // compositing it at reduced alpha makes it a translucent overlay.
    if (this.meshVisible && this.meshImg) {
      const a = ctx.globalAlpha;
      ctx.globalAlpha = 0.6;
      ctx.drawImage(this.meshImg, this.offX, this.offY, dw, dh);
      ctx.globalAlpha = a;
    }
    // Point-source layers + the read-only reference overlays. With the combined skeleton
    // shown, the projected source and the NMF model sit UNDERNEATH it as faint under-glows
    // (so the limb palette owns the top layer), and their disagreement with the placed
    // point is drawn back on top as an on-demand leash. The ground-truth / detected source
    // layers, when shown alongside the combined skeleton, draw as markers on top (no bones
    // -- the combined skeleton already carries the bones). With the combined skeleton
    // hidden, every checked source draws as its own bright standalone layer instead.
    if (this.combinedVisible) {
      if (this.nmfVisible && this.nmf) this.drawReference(this.nmf, NMF_RGB, true);
      // The projected source under the combined skeleton is the disagreement overlay: skip
      // the ghost bones that merely retrace the skeleton's own projected points (both ends
      // unobserved here), then leash only the observed joints on demand.
      if (this.projectedVisible && this.latent) this.drawReference(this.latent, LATENT_RGB, true, this.observedMask());
      this.drawSkeleton();
      // Overlaid on the combined skeleton: markers only (it already draws the bones), and no
      // labels (it already draws them at the effective position -- drawing again would double).
      if (this.gtVisible) this.drawSourceLayer("gt", false, false);
      if (this.detectedVisible) this.drawSourceLayer("detected", false, false);
      // NMF marks joints undetected in this view (it is the only cue for them); the
      // projected source does not -- the skeleton already draws those as inline
      // projection markers, so a bare leash marker there would just double up.
      if (this.nmfVisible && this.nmf) this.drawLeashes(this.nmf, NMF_RGB, "square", true);
      if (this.projectedVisible && this.latent) this.drawLeashes(this.latent, LATENT_RGB, "ring", false);
    } else {
      if (this.nmfVisible && this.nmf) this.drawReference(this.nmf, NMF_RGB, false);
      // No combined skeleton to carry them, so each standalone source draws its own bones; only
      // the FIRST visible source draws the name labels, so stacking layers never doubles them.
      let labeled = false;
      if (this.gtVisible) { this.drawSourceLayer("gt", true, !labeled); labeled = true; }
      if (this.detectedVisible) { this.drawSourceLayer("detected", true, !labeled); labeled = true; }
      // The standalone projected layer draws in the limb palette (dashed), not amber -- the
      // amber `latent` styling is reserved for the disagreement ghost under the combined skeleton.
      if (this.projectedVisible && this.latent) this.drawSourceLayer("projected", true, !labeled);
    }
    // The Shift+drag selection rubber-band sits on top of everything (CSS px, like
    // the rest of draw()).
    if (this.marquee) this.drawMarquee();
  }

  // The modifier+drag rubber-band: a translucent rectangle in CSS px. Cyan for a fresh
  // selection (Shift+drag), lime with a "+" badge when it adds to the selection
  // (Ctrl/⌘-drag) -- the same colour the "adding" cursor/pill affordance uses.
  drawMarquee() {
    const m = /** @type {{x0:number,y0:number,x1:number,y1:number}} */ (this.marquee);
    const x = Math.min(m.x0, m.x1);
    const y = Math.min(m.y0, m.y1);
    const w = Math.abs(m.x1 - m.x0);
    const h = Math.abs(m.y1 - m.y0);
    const ctx = this.ctx;
    const add = this.marqueeAdditive;
    ctx.fillStyle = add ? MARQUEE_ADD_FILL : MARQUEE_FILL;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = add ? MARQUEE_ADD_STROKE : MARQUEE_STROKE;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    if (add) {
      ctx.fillStyle = MARQUEE_ADD_STROKE;
      ctx.font = "bold 14px ui-monospace, monospace";
      ctx.textBaseline = "top";
      ctx.fillText("+", x + 3, y + 2);
    }
  }

  // The effective drawn position of joint `i`: the observed/authored pixel (ground
  // truth or the detector prediction, held in `pts`) when there is one, else the 3D
  // reprojection (`latent`/proj) as a fallback -- so an occluded or undetected joint
  // still shows where triangulation places it instead of vanishing.
  /** @param {number} i @returns {Point | null} */
  effectivePos(i) {
    const p = this.pts[i];
    if (p) return p;
    const proj = this.latent;
    return proj && i < proj.length ? proj[i] : null;
  }

  // The classified source of joint `i`: an authored GT pixel, the detector's prediction
  // (an observed pixel held in `pts`), or a point derived by reprojecting the 3D (no
  // observed pixel -- the detector fired nothing here, or the operator occluded the view,
  // which NaNs the observation out just the same). This is what the marker style encodes.
  /** @param {number} i @returns {"gt" | "prediction" | "projection"} */
  pointSource(i) {
    if (this.fixed != null && i < this.fixed.length && this.fixed[i]) return "gt";
    return this.pts[i] ? "prediction" : "projection";
  }

  // Which joints carry an observed pixel (ground truth or a detection) in this view. Used
  // to trim the projected source's under-glow to the bones that actually disagree with the
  // combined skeleton -- a bone whose two ends are both unobserved is drawn by the skeleton
  // itself at the very same reprojected positions, so ghosting it too is pure redundancy.
  /** @returns {boolean[]} */
  observedMask() {
    const mask = new Array(this.pts.length);
    for (let i = 0; i < this.pts.length; i++) mask[i] = this.pts[i] != null;
    return mask;
  }

  // The positions of a single point source: ground truth is the authored pixel (held in
  // `pts`) wherever the GT flag is set; detected is the raw detector pixel; projected is
  // the 3D reprojection (`latent`). Null where the source has no point in this view.
  /** @param {"gt" | "detected" | "projected"} kind @returns {(Point | null)[]} */
  sourcePositions(kind) {
    if (kind === "detected") return this.detected || [];
    if (kind === "projected") return this.latent || [];
    const out = new Array(this.pts.length).fill(null);
    for (let i = 0; i < this.pts.length; i++) {
      if (this.fixed && this.fixed[i] && this.pts[i]) out[i] = this.pts[i];
    }
    return out;
  }

  // The unified editable skeleton: the colored bones (a bone touching the hovered joint
  // thickens, so hover reads on the whole limb, not just the dot), then each joint drawn
  // with a marker whose fill + ring encode its source (ground truth / prediction /
  // derived reprojection). Bones and joints both use the effective position, so a joint
  // seen only via triangulation still connects into the skeleton.
  drawSkeleton() {
    const ctx = this.ctx;
    const hi = this.highlight;
    const n = Math.max(this.pts.length, this.latent ? this.latent.length : 0);
    for (const [a, b] of this.bones) {
      const pa = this.effectivePos(a);
      const pb = this.effectivePos(b);
      if (!pa || !pb) continue;
      const [ax, ay] = this.toCanvas(pa[0], pa[1]);
      const [bx, by] = this.toCanvas(pb[0], pb[1]);
      ctx.strokeStyle = this.colors[a] || "#fff";
      ctx.lineWidth = a === hi || b === hi ? HOVER_BONE_WIDTH : BONE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    for (let i = 0; i < n; i++) {
      const p = this.effectivePos(i);
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      const isHover = i === this.highlight;
      const source = this.pointSource(i);
      const r = POINT_RADIUS_PX * (isHover ? HOVER_SCALE : 1);
      // FILL: an observed point (ground truth or prediction) gets a filled disc in its
      // limb palette colour -- a prediction's fill fades with the detector's
      // confidence, so faint points that want a second look read as faint. A derived
      // point (reprojected 3D, incl. an occluded view) is left hollow, so an *observed*
      // point is unmistakably distinct from a *computed* one.
      if (source !== "projection") {
        let fillAlpha = 1;
        if (source === "prediction" && this.conf && i < this.conf.length && this.conf[i] != null) {
          fillAlpha = Math.max(0.4, Math.min(1, /** @type {number} */ (this.conf[i])));
        }
        ctx.globalAlpha = fillAlpha;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = this.colors[i] || "#fff";
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      // RING: encodes the source. Ground truth gets a bold lime ring; a derived point a
      // hollow ring in its own limb palette colour (its only mark, since it has no
      // fill); a prediction a thin dark ring, or a white one under the cursor. Hover
      // still grows the disc and thickens the bones, so the point stays legible either way.
      if (source === "gt") {
        ctx.strokeStyle = FIXED_COLOR;
        ctx.lineWidth = 2.5;
      } else if (source === "projection") {
        ctx.strokeStyle = this.colors[i] || "#fff";
        ctx.lineWidth = 2;
      } else if (isHover) {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = "rgba(0,0,0,0.6)";
        ctx.lineWidth = 1;
      }
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      // A selected joint gets an extra outer ring (this view holds its own subset of
      // the selection), distinct from every source ring so the two can coexist.
      if (this.selectionSet.has(i)) {
        ctx.beginPath();
        ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
        ctx.strokeStyle = SELECT_COLOR;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (this.labelsVisible) this.drawLabel(i, cx, cy, r);
    }
  }

  // One point source drawn as its own layer, at that source's own positions, in the limb
  // palette -- the same marker vocabulary the combined skeleton uses, so a source reads the
  // same whether it is merged in or inspected on its own:
  //   ground truth -> a filled palette disc under a bold lime ring
  //   detected     -> a filled palette disc (faded by the detector's confidence) under a thin dark ring
  //   projected    -> a hollow palette ring (no fill), with DASHED palette bones (the "derived, not
  //                   observed" cue -- the same dash the reprojection uses, but in the palette rather
  //                   than a flat amber, so it reads limb-by-limb like every other layer)
  // `withBones` draws the bones between two present points (on for a standalone layer; off when
  // overlaid on the combined skeleton, which already carries the bones). `withLabels` draws the
  // per-joint name labels -- only one layer should, so the combined skeleton owns them when it
  // is shown, and only the first visible standalone source owns them otherwise (no doubling).
  // These layers are read-only, so their markers do NOT hover-scale (unlike the combined
  // skeleton) -- a fixed size avoids a stale cross-view hover leaving a lone marker enlarged.
  /** @param {"gt" | "detected" | "projected"} kind @param {boolean} withBones @param {boolean} withLabels */
  drawSourceLayer(kind, withBones, withLabels) {
    const ctx = this.ctx;
    const pos = this.sourcePositions(kind);
    const projected = kind === "projected";
    if (withBones) {
      ctx.save();
      ctx.lineWidth = BONE_WIDTH;
      if (projected) ctx.setLineDash([5, 3]);
      for (const [a, b] of this.bones) {
        const pa = pos[a];
        const pb = pos[b];
        if (!pa || !pb) continue;
        const [ax, ay] = this.toCanvas(pa[0], pa[1]);
        const [bx, by] = this.toCanvas(pb[0], pb[1]);
        ctx.strokeStyle = this.colors[a] || "#fff";
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      ctx.restore();
    }
    for (let i = 0; i < pos.length; i++) {
      const p = pos[i];
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      const r = POINT_RADIUS_PX; // read-only layer: fixed size, no hover-scale
      // FILL: gt/detected get a filled palette disc (a detection fades with confidence);
      // projected is left hollow, matching the combined skeleton's derived-point marker.
      if (!projected) {
        let fillAlpha = 1;
        if (kind === "detected" && this.conf && i < this.conf.length && this.conf[i] != null) {
          fillAlpha = Math.max(0.4, Math.min(1, /** @type {number} */ (this.conf[i])));
        }
        ctx.globalAlpha = fillAlpha;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = this.colors[i] || "#fff";
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      // RING: ground truth = bold lime; projected = its own limb colour (its only mark);
      // detected = a thin dark ring.
      if (kind === "gt") {
        ctx.strokeStyle = FIXED_COLOR;
        ctx.lineWidth = 2.5;
      } else if (projected) {
        ctx.strokeStyle = this.colors[i] || "#fff";
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = "rgba(0,0,0,0.6)";
        ctx.lineWidth = 1;
      }
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      if (this.labelsVisible && withLabels) this.drawLabel(i, cx, cy, r);
    }
  }

  // A reference overlay (latent or NMF) as a receding under-glow beneath the editable
  // skeleton: a soft, translucent, wider skeleton in the overlay's colour whose only
  // job is to give the reference's overall shape without competing with the palette on
  // top. When `ghost` is false (the editable skeleton is hidden) it instead draws in a
  // bright, distinct line style -- dashed for the latent, dotted for the NMF -- with
  // hollow markers, so it stays fully legible as the only thing on screen.
  /**
   * @param {Point[]} pts  the reference points to draw
   * @param {string} rgb  the overlay's "r,g,b" colour
   * @param {boolean} ghost  true = soft under-glow; false = bright, on its own
   * @param {boolean[]} [observed]  ghost only: draw a bone only when at least one endpoint
   *   is observed here, so the under-glow doesn't retrace the skeleton's own projected points
   */
  drawReference(pts, rgb, ghost, observed) {
    const ctx = this.ctx;
    ctx.save();
    if (ghost) {
      // A faint, wide, round-capped glow: at coincidence it reads as a soft halo behind
      // the palette; where the reference bends away it shows as colour peeking out.
      ctx.strokeStyle = `rgba(${rgb},${GHOST_ALPHA})`;
      ctx.lineWidth = BONE_WIDTH + 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      for (const [a, b] of this.bones) {
        const pa = pts[a];
        const pb = pts[b];
        if (!pa || !pb) continue;
        if (observed && !observed[a] && !observed[b]) continue;
        const [ax, ay] = this.toCanvas(pa[0], pa[1]);
        const [bx, by] = this.toCanvas(pb[0], pb[1]);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
      ctx.restore();
      return;
    }
    // Bright, on-its-own style: the line style alone (dashed vs dotted) tells the two
    // overlays apart with the palette gone.
    const square = rgb === NMF_RGB;
    ctx.strokeStyle = `rgba(${rgb},0.95)`;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(square ? [1.5, 3.5] : [5, 3]);
    ctx.lineCap = square ? "round" : "butt";
    for (const [a, b] of this.bones) {
      const pa = pts[a];
      const pb = pts[b];
      if (!pa || !pb) continue;
      const [ax, ay] = this.toCanvas(pa[0], pa[1]);
      const [bx, by] = this.toCanvas(pb[0], pb[1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const p of pts) {
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      if (square) this.strokeSquare(cx, cy, 2.5);
      else {
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // The reference's *disagreement* with the placed point, drawn on top of the palette:
  // for each joint a short leash from the edited point to where the reference lands,
  // plus a marker there. It stays silent while they coincide (below LEASH_MIN_PX) so
  // the good case costs nothing, and its emphasis grows with the residual up to
  // LEASH_FULL_PX -- so a live 3D drag's feedback is the thing that pops. The active /
  // selected / hovered joint always shows its leash, however small, as an anchor.
  /**
   * @param {Point[]} ref  the reference points
   * @param {string} rgb  the overlay's "r,g,b" colour
   * @param {"ring" | "square"} shape  the marker drawn at the reference position
   * @param {boolean} markUndetected  mark joints with no editable point (undetected in
   *   this view) with a bare marker; off for the projection, which the skeleton already
   *   draws inline as a projection-sourced marker
   */
  drawLeashes(ref, rgb, shape, markUndetected) {
    const ctx = this.ctx;
    ctx.save();
    ctx.lineCap = "round";
    for (let i = 0; i < ref.length; i++) {
      const pr = ref[i];
      if (!pr) continue;
      const [rx, ry] = this.toCanvas(pr[0], pr[1]);
      const pe = this.pts[i];
      // Undetected in this view (no editable point) but triangulated elsewhere: there is
      // no disagreement to leash, but still mark where the estimate places the point --
      // surfacing that occluded keypoint is the overlay's whole job for such points.
      if (!pe) {
        if (markUndetected) this.strokeLeash(shape, null, rx, ry, 3, `rgba(${rgb},0.95)`);
        continue;
      }
      // The disagreement line is drawn only for the joint under the cursor (hover syncs
      // across every view) or the one being dragged -- so the default view stays clean
      // and the "how far is the estimate pulling this point" line appears on demand.
      if (i !== this.dragging && i !== this.highlight) continue;
      const [ex, ey] = this.toCanvas(pe[0], pe[1]);
      const d = Math.hypot(rx - ex, ry - ey);
      const t = Math.max(0, Math.min(1, (d - LEASH_MIN_PX) / (LEASH_FULL_PX - LEASH_MIN_PX)));
      // Coincident (below the threshold): just the marker, no zero-length connector;
      // past it the connector too, both growing with the residual.
      const from = d >= LEASH_MIN_PX ? [ex, ey] : null;
      this.strokeLeash(shape, /** @type {[number, number] | null} */ (from), rx, ry, 2.5 + 2.5 * t, `rgba(${rgb},${0.55 + 0.45 * t})`);
    }
    ctx.restore();
  }

  /**
   * Draw one leash -- a marker at the reference position, optionally with a connector
   * from the placed point -- as a dark casing beneath a coloured top, so both read on
   * any frame. `from` null draws the marker alone (a bare reprojection, no disagreement).
   * @param {"ring" | "square"} shape
   * @param {[number, number] | null} from  the placed point in canvas px, or null for no connector
   * @param {number} rx @param {number} ry  the reference position in canvas px
   * @param {number} r  the marker radius
   * @param {string} color  the coloured (top) stroke as an rgba() string
   */
  strokeLeash(shape, from, rx, ry, r, color) {
    const ctx = this.ctx;
    /** @type {[string, number][]} */
    const passes = [["rgba(0,0,0,0.5)", 2.5], [color, 1.25]];
    for (const [style, width] of passes) {
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      if (from) {
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        ctx.lineTo(rx, ry);
        ctx.stroke();
      }
      this.markerPath(shape, rx, ry, r);
      ctx.stroke();
    }
  }

  /**
   * Trace a marker outline at (x, y) into the current path (caller strokes it).
   * @param {"ring" | "square"} shape @param {number} x @param {number} y @param {number} r
   */
  markerPath(shape, x, y, r) {
    const ctx = this.ctx;
    ctx.beginPath();
    if (shape === "square") ctx.rect(x - r, y - r, r * 2, r * 2);
    else ctx.arc(x, y, r, 0, Math.PI * 2);
  }

  /**
   * Stroke a square outline centred at (x, y) with half-size r.
   * @param {number} x @param {number} y @param {number} r
   */
  strokeSquare(x, y, r) {
    this.ctx.strokeRect(x - r, y - r, r * 2, r * 2);
  }

  /**
   * @param {number} i  point index
   * @param {number} cx  joint centre, canvas x
   * @param {number} cy  joint centre, canvas y
   * @param {number} r  the joint's drawn radius (the label clears it)
   */
  drawLabel(i, cx, cy, r) {
    const name = this.pointNames[i];
    if (!name) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,0.85)"; // outline for legibility over any frame
    ctx.fillStyle = "#fff";
    const tx = cx + r + 3;
    ctx.strokeText(name, tx, cy);
    ctx.fillText(name, tx, cy);
    ctx.restore();
  }

  // -- interaction ------------------------------------------------------------

  /**
   * @param {PointerEvent} e
   * @returns {[number, number]}  pointer position in CSS pixels relative to the canvas
   */
  cssXY(e) {
    const rect = this.canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  /**
   * @param {number} mx
   * @param {number} my
   * @returns {[number, number]}
   */
  toImage(mx, my) {
    return [(mx - this.offX) / this.scale, (my - this.offY) / this.scale];
  }

  // The joint nearest the pointer within the grab tolerance, or null. Uses the *effective*
  // drawn position, so a derived (reprojected) point -- a hollow circle with no observed
  // pixel, drawn at `latent` -- is grabbable just like an observed one; dragging it authors
  // ground truth at the drop (and un-occludes the view if it was occluded).
  /**
   * @param {number} ix
   * @param {number} iy
   * @returns {number | null}
   */
  nearestPoint(ix, iy) {
    const tol = HIT_TOLERANCE_PX / Math.max(this.scale, 1e-6);
    /** @type {number | null} */
    let best = null;
    let bestD = tol;
    const n = Math.max(this.pts.length, this.latent ? this.latent.length : 0);
    for (let i = 0; i < n; i++) {
      const p = this.effectivePos(i);
      if (!p) continue;
      const d = Math.hypot(p[0] - ix, p[1] - iy);
      if (d <= bestD) {
        best = i;
        bestD = d;
      }
    }
    return best;
  }

  // Every joint whose effective (drawn) position falls inside the image-space rect --
  // the Shift+drag marquee's hit-test. Corners come in any order; they are normalized
  // here. Uses the same effective position as nearestPoint, so derived/reprojected
  // joints are selectable too.
  /**
   * @param {number} ax
   * @param {number} ay
   * @param {number} bx
   * @param {number} by
   * @returns {number[]}
   */
  pointsInRect(ax, ay, bx, by) {
    const x0 = Math.min(ax, bx);
    const x1 = Math.max(ax, bx);
    const y0 = Math.min(ay, by);
    const y1 = Math.max(ay, by);
    /** @type {number[]} */
    const hits = [];
    const n = Math.max(this.pts.length, this.latent ? this.latent.length : 0);
    for (let i = 0; i < n; i++) {
      const p = this.effectivePos(i);
      if (!p) continue;
      if (p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1) hits.push(i);
    }
    return hits;
  }

  /** @param {PointerEvent} e */
  onPointerDown(e) {
    const [mx, my] = this.cssXY(e);
    this.downX = mx;
    this.downY = my;
    this.moved = false;
    const [ix, iy] = this.toImage(mx, my);
    const canGrab = this.editable && this.combinedVisible;
    const addMod = e.ctrlKey || e.metaKey; // Ctrl/Cmd = add to the selection

    // A modifier + primary button is multi-select, resolved on release (onPointerUp):
    // Shift replaces the selection with what the gesture picks, Ctrl/Cmd adds to it. A
    // drag rubber-bands a region; a click (no drag) picks the single hit point. Here we
    // just arm the marquee and swallow the event -- it must neither move a point nor pan.
    if (e.button === 0 && (e.shiftKey || addMod) && canGrab) {
      e.preventDefault();
      this.marqueeing = true;
      this.marqueeAdditive = addMod;
      this.marquee = { x0: mx, y0: my, x1: mx, y1: my };
      this.canvas.setPointerCapture(e.pointerId);
      return;
    }

    const point = canGrab ? this.nearestPoint(ix, iy) : null;

    if (point !== null) {
      // Right-click confirms/clears the point's ground truth, no drag. Don't collapse an
      // existing multi-selection: only re-select the point when it isn't already selected
      // here, so right-clicking one of several selected joints toggles it in place.
      if (e.button === 2) {
        e.preventDefault();
        if (!this.selectionSet.has(point)) this.cb.onSelect(this.viewIndex, point, false);
        this.cb.onToggleFixed(this.viewIndex, point);
        return;
      }
      if (e.button !== 0) return; // only the primary button drags
      e.preventDefault();
      this.cb.onSelect(this.viewIndex, point, false); // selecting happens on press, not release
      // An obscured joint can still be dragged -- doing so un-obscures it (the app
      // un-flags it on release via `wasInvisible`).
      this.dragInvisible = this.invisible != null && !!this.invisible[point];
      this.dragging = point;
      this.canvas.setPointerCapture(e.pointerId);
      return;
    }

    // Empty space (or a non-editable / overlay-hidden view): pan, if allowed.
    if (this.zoomable && (e.button === 0 || e.button === 1)) {
      e.preventDefault();
      this.panning = true;
      this.panOrigX = this.panX;
      this.panOrigY = this.panY;
      this.canvas.setPointerCapture(e.pointerId);
      this.canvas.style.cursor = "grabbing";
    }
  }

  /** @param {PointerEvent} e */
  onPointerMove(e) {
    const [mx, my] = this.cssXY(e);
    if (!this.moved && Math.hypot(mx - this.downX, my - this.downY) > DRAG_THRESHOLD_PX) {
      this.moved = true;
    }

    if (this.marqueeing && this.marquee) {
      e.preventDefault();
      this.marquee.x1 = mx;
      this.marquee.y1 = my;
      this.draw();
      return;
    }

    if (this.dragging !== null) {
      if (!this.moved) return; // a press that has not yet become a drag
      e.preventDefault();
      const [ix, iy] = this.toImage(mx, my);
      this.pts[this.dragging] = [ix, iy];
      this.draw();
      // Throttle the network round-trip to one per animation frame.
      this.pendingDrag = { x: ix, y: iy };
      if (!this.rafId) {
        this.rafId = requestAnimationFrame(() => {
          this.rafId = 0;
          if (this.dragging !== null && this.pendingDrag) {
            this.cb.onDragging(this.viewIndex, this.dragging, this.pendingDrag.x, this.pendingDrag.y);
          }
        });
      }
      return;
    }

    if (this.panning) {
      e.preventDefault();
      this.panX = this.panOrigX + (mx - this.downX);
      this.panY = this.panOrigY + (my - this.downY);
      this.applyTransform();
      this.draw();
      return;
    }

    // Idle: report hover so the app can emphasize this joint in every view.
    if (this.editable && this.combinedVisible) {
      const [ix, iy] = this.toImage(mx, my);
      const point = this.nearestPoint(ix, iy);
      if (point !== this._hover) {
        this._hover = point;
        this.cb.onHover(point);
      }
      this.canvas.style.cursor = point !== null ? "pointer" : this.zoomable ? "grab" : "default";
    }
  }

  /** @param {PointerEvent} e */
  onPointerUp(e) {
    if (this.marqueeing) {
      e.preventDefault();
      this.marqueeing = false;
      const rect = this.marquee;
      this.marquee = null;
      const additive = this.marqueeAdditive; // Ctrl/Cmd = add; Shift = replace
      const canGrab = this.editable && this.combinedVisible;
      if (this.moved && rect) {
        // A genuine rubber-band: Shift replaces the selection with the enclosed joints,
        // Ctrl/Cmd adds them (rect corners -> image space).
        const [ax, ay] = this.toImage(rect.x0, rect.y0);
        const [bx, by] = this.toImage(rect.x1, rect.y1);
        this.cb.onSelectRegion(this.viewIndex, this.pointsInRect(ax, ay, bx, by), additive);
      } else if (canGrab && e.type !== "pointercancel") {
        // A modifier+click without a drag: Ctrl/Cmd toggles the single hit joint in the
        // selection; Shift selects just it (or, on empty space, clears the selection). A
        // pointercancel (the browser aborted the captured press -- focus loss, a competing
        // gesture) is not a deliberate click, so it must neither toggle nor clear (mirrors
        // the pan branch below).
        const [ix, iy] = this.toImage(...this.cssXY(e));
        const point = this.nearestPoint(ix, iy);
        if (point !== null) this.cb.onSelect(this.viewIndex, point, additive);
        else if (!additive) this.cb.onBackground();
      }
      this.draw();
      return;
    }
    if (this.dragging !== null) {
      e.preventDefault();
      const point = this.dragging;
      this.dragging = null;
      if (this.rafId) {
        cancelAnimationFrame(this.rafId);
        this.rafId = 0;
      }
      // A genuine drag commits the move; a click without movement only selected.
      if (this.moved) {
        const [ix, iy] = this.toImage(...this.cssXY(e));
        this.pts[point] = [ix, iy];
        this.cb.onDragged(this.viewIndex, point, ix, iy, this.dragInvisible);
      }
      return;
    }
    if (this.panning) {
      e.preventDefault();
      this.panning = false;
      this.canvas.style.cursor = this.zoomable ? "grab" : "default";
      // A press on empty space that never moved is a click on the background: clear the
      // selection. A real pan (moved past the threshold) leaves the selection alone, and
      // a pointercancel (the browser took over) must not be read as a deliberate click.
      if (!this.moved && e.type !== "pointercancel") this.cb.onBackground();
    }
  }

  onPointerLeave() {
    // Drop the hover when the cursor leaves so no view stays falsely emphasized.
    if (this.dragging === null && this.panning === false && this._hover !== null) {
      this._hover = null;
      this.cb.onHover(null);
    }
  }

  /**
   * Wheel input drives both zoom and pan, split by device:
   *   - mouse wheel (or Ctrl+wheel)  -> zoom toward the cursor at WHEEL_ZOOM_RATE
   *   - trackpad pinch (a wheel with ctrlKey the browser synthesizes) -> zoom, but at the
   *     higher PINCH_ZOOM_RATE, since a pinch's per-event delta is tiny
   *   - trackpad two-finger scroll, once zoomed in -> pan both axes (content-grab)
   * The device is classified once per wheel burst (see the fields' note) so a fast scroll
   * flick never briefly reads as a mouse wheel and zooms. Pan is gated on zoom > 1: at the
   * fit there is nothing to pan to, so a scroll there zooms instead -- which also keeps a
   * mouse wheel wrongly classified as a trackpad (small-delta / hi-res mice) from dead-ending
   * with no way to zoom in.
   * @param {WheelEvent} e
   */
  onWheel(e) {
    if (!this.zoomable) return;
    e.preventDefault();
    if (e.timeStamp - this._lastWheelTime > WHEEL_BURST_GAP_MS) {
      this._wheelIsTrackpad = looksLikeTrackpad(e);
    }
    this._lastWheelTime = e.timeStamp;
    // A trackpad two-finger scroll (classified trackpad, no ctrlKey) pans -- but only when
    // zoomed in; otherwise it falls through to the zoom path below (see the doc note).
    if (this._wheelIsTrackpad && !e.ctrlKey && this.zoom > 1) {
      this.panByScroll(e.deltaX, e.deltaY);
      return;
    }
    // Zoom toward the cursor. A trackpad pinch's tiny pixel delta gets the higher gain; a
    // mouse wheel (or Ctrl+wheel, a big or line-mode delta) keeps the notch rate.
    const [mx, my] = this.cssXY(e);
    const pinch = e.deltaMode === 0 && Math.abs(e.deltaY) < WHEEL_NOTCH_MIN;
    this.zoomAtCursor(mx, my, e.deltaY, pinch ? PINCH_ZOOM_RATE : WHEEL_ZOOM_RATE);
  }

  /**
   * Zoom toward a cursor point, keeping the image pixel under it fixed. Snaps back to the
   * letterboxed fit when the zoom would drop to <= 1x.
   * @param {number} mx  cursor CSS x
   * @param {number} my  cursor CSS y
   * @param {number} deltaY  wheel delta (negative = zoom in)
   * @param {number} rate  delta -> zoom-factor gain
   */
  zoomAtCursor(mx, my, deltaY, rate) {
    const z0 = this.zoom;
    const z1 = Math.min(MAX_ZOOM, Math.max(1, z0 * Math.exp(-deltaY * rate)));
    if (z1 === z0) return;
    if (z1 <= 1.0001) {
      this.resetZoom(); // snap cleanly back to the letterboxed fit
      return;
    }
    const [ix, iy] = this.toImage(mx, my);
    const newScale = this.fitScale * z1;
    this.zoom = z1;
    this.panX = mx - ix * newScale - this.fitOffX;
    this.panY = my - iy * newScale - this.fitOffY;
    this.applyTransform();
    this.draw();
  }

  /**
   * Pan the view by a trackpad two-finger scroll, content-grab style: the image tracks the
   * fingers (matching the grab-drag), 1:1 in CSS px -- pan is applied in CSS px on top of
   * the fit, so the mapping holds at any zoom. onWheel only routes here when zoomed in (at
   * the fit a scroll is sent to zoom instead); the guard below is a defensive backstop.
   * @param {number} dx  scroll delta x, CSS px
   * @param {number} dy  scroll delta y, CSS px
   */
  panByScroll(dx, dy) {
    if (this.zoom <= 1) return;
    this.panX -= dx;
    this.panY -= dy;
    this.applyTransform();
    this.draw();
  }

  /** @param {MouseEvent} e */
  onDblClick(e) {
    // Double-clicking a joint selects that keypoint across every view (a fast way to
    // act on one point everywhere); double-clicking empty space resets the zoom.
    if (this.editable && this.combinedVisible) {
      const [ix, iy] = this.toImage(...this.cssXY(e));
      const point = this.nearestPoint(ix, iy);
      if (point !== null) {
        e.preventDefault();
        this.cb.onSelectKeypointAllViews(point, e.ctrlKey || e.metaKey);
        return;
      }
    }
    if (!this.zoomable) return;
    e.preventDefault();
    this.resetZoom();
  }
}
