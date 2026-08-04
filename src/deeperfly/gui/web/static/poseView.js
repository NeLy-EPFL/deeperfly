// @ts-check
// One camera's frame plus its draggable 2D skeleton overlay, on a <canvas>.
//
// A port of the old Qt PoseView, grown a few editor conveniences: the frame is
// drawn fit-to-canvas (letterboxed) and can be zoomed (any wheel/scroll or trackpad
// pinch, toward the cursor) and panned (drag on empty space). Frames are loaded via
// `loadFrame`; while a
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
// The view draws the point *sources* as independently toggled layers. GROUND TRUTH is the
// EDITABLE layer (a filled palette disc under a bold lime ring): whenever it is shown a drag
// authors GT. DETECTED is the detector's output (a filled disc, faded by confidence, under a
// thin dark ring) wherever it is still usable -- a view the operator flagged occluded
// ("Projected") has rejected its detection, so none is drawn there and the joint's position falls
// through to its reprojection. "COMBINED" is not a visibility switch -- it is a MERGE toggle over
// GT + Detected: on, the two draw as ONE skeleton (each joint = GT if authored, else the detector's
// usable point, else its reprojected point, so a bone never drops out just because one endpoint is
// only derived); off, GT and Detected draw as two separate overlaid skeletons (Detected beneath,
// the editable GT skeleton on top). The "projected" source is the REPROJECTED SKELETON -- an
// independent overlay of the full 3D reprojection (hollow rings joined by thick, semi-transparent,
// DASHED palette edges), its own layer in every mode (it never merges in), so the 3D's opinion of
// every joint reads at a glance and the gap to a placed pixel is the live disagreement.
//
// GT is editable whenever it is shown. A drag MOVES an existing GT point, or SPAWNS one from a
// seed -- a detected node, a reprojected hollow point, or (when the joint has none of those in
// this view) a faint "Unplaced" placeholder ghost -- and the point reads as GT under the cursor the
// instant the drag starts (before the server sets its GT flag). See nodeAt / anchorPos /
// grabCandidates + drawSkeleton / drawPlaceholders. Selection rings, name labels, and hover
// emphasis are drawn once on top by drawJointOverlay, anchored at the joint's best visible
// position, so they still read on a joint that has no GT yet (only a detected / projected / seed).
//
// The fitted NMF model is a read-only reference: a faint mint under-glow *beneath* the skeleton
// (so the limb palette always reads on top) with its *disagreement* against the placed point
// drawn back on top as a per-joint "leash" -- silent at coincidence, growing with the residual.
// With the combined skeleton toggled off a source/reference instead draws in its own bright,
// dashed/dotted style so it stays fully visible on its own.
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
 * @property {((view: number, landmark: number, x: number, y: number) => void)} [onPlaceLandmark]
 *   a calibration landmark is armed and was clicked into place at image coords (x, y)
 * @property {(view: number) => void} onActiveView  the pointer entered/moved over this view (drives the "select all in this view" gesture)
 * @property {(point: number | null) => void} onHover  the hovered joint changed (cross-view)
 */

const POINT_RADIUS_PX = 4; // drawn joint radius in screen px (constant under zoom)
const HOVER_SCALE = 1.6; // how much a hovered joint grows (its bones thicken too, so it needn't balloon)
const HOVER_BONE_WIDTH = 3; // a hovered joint's connected bones thicken to this (screen px)
const HIT_TOLERANCE_PX = 14; // how close a click must be to grab a joint, screen px
const DRAG_THRESHOLD_PX = 3; // movement (screen px) before a press becomes a drag
const MAX_ZOOM = 10; // cap on the user wheel-zoom factor over fit
const WHEEL_ZOOM_RATE = 0.007; // mouse-wheel delta -> zoom factor sensitivity (~2x per notch)
const PINCH_ZOOM_RATE = 0.01; // trackpad pinch: a higher gain than the wheel (its per-event delta is tiny) so the pinch tracks the fingers
const WHEEL_NOTCH_MIN = 50; // |deltaY| (px) below which a step is a tiny one (trackpad pinch or accelerated mouse notch) and gets the higher zoom gain; at/above it's a chunky wheel notch
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

// The fitted NMF model is a read-only reference overlay (mint) drawn UNDERNEATH the editable
// skeleton, so the limb palette always owns the top layer instead of being painted over. It is
// drawn as a soft under-glow skeleton whose overall shape is the ambient signal, with its
// *disagreement* with the placed point at one joint -- a "leash" from the point to where the
// model lands -- drawn on top only for the joint under the cursor (or being dragged), so the
// default view stays uncluttered and the difference line appears on demand. (The 3D reprojection
// is instead its own independent overlay, the reprojected skeleton -- see drawReprojection.) On
// its own, each overlay is identifiable without relying on colour: the editable skeleton is a
// solid line with a filled disc, the reprojected skeleton a dashed line with hollow rings, the
// NMF a dotted line with a hollow square.
const NMF_RGB = "80,230,180"; // mint
const GHOST_ALPHA = 0.4; // the NMF reference's soft under-glow (a faint halo beneath the palette)
const LEASH_MIN_PX = 2.5; // below this editable<->reference screen gap the leash needs no connector line
const LEASH_FULL_PX = 16; // at/above this gap the reference marker + leash reach full emphasis

// The reprojected skeleton is its own independent overlay (the "projected" source, toggled on
// its own -- on by default): the full 3D reprojection in the limb palette -- hollow rings at
// every reprojected joint joined by THICK, semi-transparent, DASHED edges -- so where
// triangulation places each joint reads at a glance as a distinct "derived, not observed" layer
// that never competes with the solid editable skeleton on top. It draws the same way whether or
// not the combined skeleton is shown. See drawReprojection.
const PROJ_WIDTH = 5; // reprojected-skeleton bone width (screen px): thick, well above the editable 1.5
const PROJ_ALPHA = 0.4; // ... and semi-transparent, so the solid editable palette always wins on top
const PROJ_DASH = [7, 5]; // ... and dashed (screen px on/off), the "derived, not observed" cue

// The "Unplaced" layer (see drawPlaceholders): a faint, draggable seed at a joint this view has
// NOTHING else to grab (no GT / detected / reprojected point) -- a joint triangulation rejected,
// or one the detector never fired. A small dashed hollow ring with a faint centre dot in the
// joint's limb palette, at reduced opacity, so it reads as "not observed -- drag me to place",
// clearly apart from the observed (filled disc), reprojected (solid hollow ring) and NMF markers.
const PLACEHOLDER_ALPHA = 0.55; // the Unplaced seed's opacity: faint, but grabbable at a glance
const PLACEHOLDER_DASH = [2, 3]; // its dashed hollow ring (screen px on/off)

// The "Absent" tombstone (see drawAbsent): a joint the operator declared NOT on this animal.
// Deliberately achromatic -- every other marker is drawn in the joint's limb palette, so grey
// says "outside the anatomy" at a glance and cannot be mistaken for a faint observation. A cross
// rather than a ring for the same reason: rings mean "a position, just not observed", and an
// absent joint has no position at all.
const ABSENT_ALPHA = 0.45;
const ABSENT_COLOR = "#8a8f98";
const ABSENT_MARK_PX = 4; // half-arm of the cross, screen px (constant under zoom)

// The reprojection-distance warning (see drawReprojWarnings): when a joint's authored/detected
// anchor sits farther than the (image-px) threshold from where the 3D reprojects it, flag it with
// an amber->red ring on the anchor plus a connector to the reprojected point, so a disagreement
// between the hand 2D label and the multi-view 3D pops for review. Amber at the threshold, ramping
// to red at WARN_RED_MULT x it -- the eye lands on the worst joints first.
const WARN_AMBER = "255,176,0"; // "r,g,b" at the threshold (a joint just over the line)
const WARN_RED = "255,60,60"; // ... blended to this at/above WARN_RED_MULT x the threshold
const WARN_RED_MULT = 2; // distance / threshold at which the cue reaches full red
const WARN_RING_PAD = 5; // the warning ring's radius beyond the joint marker (screen px) -- clears the r+3 selection ring

/**
 * Blend two "r,g,b" strings, returning "r,g,b" at fraction t (0 = a, 1 = b).
 * @param {string} a @param {string} b @param {number} t @returns {string}
 */
function mixRgb(a, b, t) {
  const pa = a.split(",").map(Number);
  const pb = b.split(",").map(Number);
  return pa.map((c, i) => Math.round(c + (pb[i] - c) * t)).join(",");
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
    /** @type {Point[] | null} */
    this.placeholder = null; // seed position per joint ABSENT from this view (no GT / detected / projected), a faint draggable ghost so a GT can still be placed; null elsewhere
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
    // Layer toggles. Ground truth is the EDITABLE layer: whenever it is shown a drag
    // authors GT (moving a GT point, or spawning one from the position drawn there). The
    // detected layer is a read-only reference that auto-hides once an annotation skeleton
    // exists -- it seeded that skeleton and would otherwise double every joint. The
    // projected layer is the reprojection of the derived 3D, its own dashed overlay, on by
    // default as a guide. There used to be a fourth, "Combined", merging GT and detected
    // into one skeleton; the instance owns every position now, so there is nothing to merge.
    this.detectedVisible = true;
    this.projectedVisible = true;
    // The "Unplaced" layer: faint, draggable ghost seeds for joints a view has nothing
    // to grab for (no GT / detected / reprojected point) -- e.g. a joint triangulation
    // rejected. On by default; dragging a ghost authors GT like any other seed.
    // Calibration landmarks observed in THIS view at the current frame, or null. Their own
    // layer because they are their own namespace: a landmark is not a skeleton joint, has no
    // bones, no confidence and no 3D of its own here, and it drives only the rig solve.
    this.landmarks = null;
    this.landmarkNames = [];
    this.landmarksVisible = true;
    // Index of the landmark a click will place, or -1. Armed from the Landmarks panel, so a
    // click means "place THIS landmark" rather than needing a separate drag target.
    this.armedLandmark = -1;
    this.nmfVisible = false;
    this.meshVisible = false;
    this.labelsVisible = false;
    // A master "hide everything" switch (the `h` peek): when set, draw() renders only the frame
    // and skips every overlay below it, and canGrab goes false so a point can't be dragged while
    // it is invisible. It OVERRIDES -- but never mutates -- the per-layer toggles above, so
    // clearing it restores exactly what was shown. Off by default (all overlays drawn).
    this.overlaysHidden = false;
    // The reprojection-distance warning: a data-quality check independent of the layer toggles
    // above (it reads the authored/detected anchor and the reprojection directly, whatever is
    // shown). On by default, kept in sync with the `checked` checkbox + threshold input in
    // index.html; the app restores an operator's persisted preference over these defaults.
    this.warnVisible = true;
    this.warnThreshold = 8; // image px: an anchor<->reprojection gap above this flags the joint
    /** @type {number | null} */
    // Whether this frame carries an annotation skeleton. When it does, EVERY joint's
    // position comes from `pts` -- the instance owns them -- and `fixed` only decides
    // whether the node reads as the operator's pixel or as a derived one. Before that,
    // `pts` is the old GT-over-detection resolution and the layer precedence applies.
    this.instanceMode = false;
    this.dragging = null;
    this.dragInvisible = false; // was the grabbed joint obscured? (reported on release)
    /** @type {boolean[] | null} per-point "not on this animal" -- see drawAbsent */
    this.absent = null;
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
   * @param {Point[] | null} [data.placeholder]  seed positions for joints with nothing else to grab in this view (the "Unplaced" ghosts), or null
   * @param {boolean[] | null} [data.absent]  per-point "not on this animal" (amputated); drawn as a tombstone, never draggable
   */
  setFrameData(data) {
    if (data.instanceMode !== undefined) this.instanceMode = !!data.instanceMode;
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
    // Absence gates whether a joint is drawn at all, so -- unlike `detected` / `placeholder`
    // -- it rides EVERY reply including the lean mid-drag stream. Keeping the last value on
    // `undefined` would still be correct; assigning it unconditionally is the guarantee.
    if (data.absent !== undefined) this.absent = data.absent;
    if (data.conf !== undefined) this.conf = data.conf;
    if (data.latent !== undefined) this.latent = data.latent;
    // The raw detections are static within a frame, so the mid-drag stream omits them
    // (undefined) and this view keeps the set from the last plain/navigation fetch.
    if (data.detected !== undefined) this.detected = data.detected;
    if (data.nmf !== undefined) this.nmf = data.nmf;
    // Placeholder seeds depend on the frame's GT / occlusion / 3D state, so they ride
    // the verbose (settle / navigation) reply -- omitted (undefined) mid-drag, keep as-is.
    if (data.placeholder !== undefined) this.placeholder = data.placeholder;
    if (data.landmarks !== undefined) this.landmarks = data.landmarks;
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

  // Whether this view accepts editing gestures right now: it must be an editable (large,
  // writer-owned) view AND the Ground truth layer must be shown -- GT is the editable layer, so
  // a drag authors GT and hover/selection are live only while GT is visible. Turning GT off makes
  // the view inspect-only (pan/zoom still work). The `h` peek (overlaysHidden) also drops it: a
  // point you cannot see must not be draggable, so a peek can't be misread as an edit surface.
  get canGrab() {
    return this.editable && !this.overlaysHidden;
  }

  /** @param {boolean} zoomable  whether wheel-zoom + pan are allowed (large views only) */
  setZoomable(zoomable) {
    if (this.zoomable === zoomable) return;
    this.zoomable = zoomable;
    if (!zoomable) this.resetZoom(); // thumbnails always show the whole frame
  }

  /** @param {boolean} visible  merge mode: on = GT + Detected draw as one skeleton, off = two separate skeletons */


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


  /** @param {boolean} visible  whether the reprojection-distance warning cue is drawn */
  setWarnVisible(visible) {
    if (this.warnVisible === visible) return;
    this.warnVisible = visible;
    this.draw();
  }

  /** @param {number} px  image-px threshold above which a joint's anchor<->reprojection gap warns */
  setWarnThreshold(px) {
    if (this.warnThreshold === px) return;
    this.warnThreshold = px;
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

  /** @param {boolean} hidden  master peek: hide every overlay (frame only), leaving each layer's own toggle as-is */
  setOverlaysHidden(hidden) {
    if (this.overlaysHidden === hidden) return;
    this.overlaysHidden = hidden;
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
    // "Hide all overlays" (the `h` peek): a momentary, non-destructive look at the raw frame.
    // Everything below draws an overlay, so stopping here leaves just the image -- the mesh,
    // reprojection, NMF reference, skeleton(s), per-joint marks, reprojection warnings, and the
    // marquee all vanish. The per-layer toggles are untouched, so clearing the peek restores
    // exactly what was shown; editing is gated too (see canGrab), so nothing can move unseen.
    if (this.overlaysHidden) return;
    // The posed NMF mesh sits between the frame and the editable skeleton, so the
    // keypoints stay legible on top of it. The GPU renders the silhouette opaque;
    // compositing it at reduced alpha makes it a translucent overlay.
    if (this.meshVisible && this.meshImg) {
      const a = ctx.globalAlpha;
      ctx.globalAlpha = 0.6;
      ctx.drawImage(this.meshImg, this.offX, this.offY, dw, dh);
      ctx.globalAlpha = a;
    }
    // The reprojected skeleton is an independent overlay drawn beneath everything else: the full
    // 3D reprojection as hollow rings joined by thick, dashed, semi-transparent palette edges
    // (see drawReprojection). It is its own layer in every mode and never merges in. It owns the
    // name labels only when it is the sole visible layer -- i.e. neither GT nor Detected is shown
    // (the joint overlay below claims them otherwise).
    const anySkeleton = true; // the annotation skeleton is always drawn
    const reprojLabels = this.labelsVisible && !anySkeleton;
    if (this.projectedVisible && this.latent) this.drawReprojection(this.latent, reprojLabels);
    // The "Unplaced" seeds sit above the reprojection but below the editable skeleton. They only
    // exist where nothing else is drawn (see placeholderPos), so ordering never hides a real point.
    // Ghosts only before there is a skeleton. After that every cell of the instance has a
    // drawn position -- filled from this same chain where it has no evidence-backed seed
    // (state.py _seed_instance) -- so a ghost could only be a second, contradictory dot for one
    // cell. That is also why the layer lost its toggle: the state it serves lasts from arriving
    // on a frame until the first keystroke.
    if (!this.instanceMode && this.placeholder) this.drawPlaceholders();
    if (this.landmarksVisible && this.landmarks) this.drawLandmarks();
    // Beneath the skeleton(s), the NMF model's faint under-glow (ghosted so the limb palette owns
    // the top layer when a skeleton sits on it; drawn bright + standalone when nothing does).
    if (this.nmfVisible && this.nmf) this.drawReference(this.nmf, NMF_RGB, anySkeleton);
    // The annotation skeleton: one skeleton, each joint at its ground-truth pixel or at the
    // position derived for it. There used to be a "Combined" toggle here, choosing between
    // that and drawing GT and Detected as two separate layers -- a merge that no longer
    // happens. The instance owns every position now, so there is nothing to merge it with;
    // the detections are a reference overlay with its own visibility toggle.
    this.drawSkeleton();
    // One pass on top of every skeleton for the cross-cutting per-joint marks: selection rings,
    // name labels, and hover emphasis -- each anchored at the joint's best visible position, so a
    // selected/hovered joint with no GT yet (only detected / projected) still reads.
    this.drawJointOverlay();
    // The NMF's disagreement with the placed point, drawn back on top as an on-demand leash.
    if (this.nmfVisible && this.nmf && anySkeleton) this.drawLeashes(this.nmf, NMF_RGB, "square", true);
    // The reprojection-distance warning, drawn topmost among the annotations so a joint whose 2D
    // label disagrees with the multi-view 3D is impossible to miss. Needs a 3D solve (this.latent)
    // but is independent of the Projected overlay toggle -- the reprojection data is always here.
    // Tombstones last among the marker passes, so they are never hidden by a stale layer.
    this.drawAbsent();
    if (this.warnVisible && this.latent) this.drawReprojWarnings();
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

  // The authored GT pixel of joint `i` (held in `pts` wherever the GT flag is set), or null.
  /** @param {number} i @returns {Point | null} */
  gtPos(i) {
    return this.fixed && i < this.fixed.length && this.fixed[i] && this.pts[i]
      ? this.pts[i]
      : null;
  }

  // The detector's USABLE prediction for joint `i` (the "detected" source), or null. A view the
  // operator flagged occluded ("Projected") is an assertion that its pixel is not readable here:
  // it is dropped from the 3D solve, so it is not a position source either. The joint then falls
  // through to its reprojection -- the best estimate left for this view -- everywhere a position
  // is resolved (nodeAt, anchorPos, grabCandidates, sourcePositions). Without this, occluding a
  // view left its rejected pixel drawn as the joint's position, contradicting the state it reports.
  /** @param {number} i @returns {Point | null} */
  detPos(i) {
    if (this.invisible && this.invisible[i]) return null;
    return this.detected && i < this.detected.length ? this.detected[i] : null;
  }

  // The reprojection of the derived 3D for joint `i` in this view, or null when there is none (a
  // 2D-only result, or a joint with no solvable 3D). The joint's *derived* position -- what it
  // falls back to when the view carries no usable observation.
  /** @param {number} i @returns {Point | null} */
  latentPos(i) {
    return this.latent && i < this.latent.length ? this.latent[i] : null;
  }

  // The reprojected position of joint `i` when it is actually DRAWN as the joint's position -- so
  // the anchor and the hit-test only ever land on something the operator can see. That is either
  // because the reprojection overlay is shown (its hollow ring is right there), or because the view
  // has no usable observation (occluded) and the merged skeleton is therefore drawing the joint at
  // its reprojection -- which it does whether or not that overlay is on (see nodeAt/drawSkeleton).
  /** @param {number} i @returns {Point | null} */
  shownLatentPos(i) {
    const p = this.latentPos(i);
    if (!p) return null;
    if (this.projectedVisible) return p;
    return this.invisible && this.invisible[i] ? p : null;
  }

  // Calibration landmarks: a diamond plus its name, in one warm colour distinct from every
  // limb palette entry. Deliberately a different SHAPE, not just a different hue -- a
  // landmark is a different kind of thing from a keypoint, and shape survives colour
  // blindness and a busy frame in a way hue does not.
  drawLandmarks() {
    const ctx = this.ctx;
    // Sizes are CSS pixels, like every other marker pass -- the canvas transform is baked
    // into toCanvas(), so a marker must NOT also divide by the zoom or it shrinks as you
    // zoom in, which is exactly backwards.
    const size = 5;
    ctx.save();
    ctx.lineWidth = 1.6;
    for (let i = 0; i < this.landmarks.length; i++) {
      const pt = this.landmarks[i];
      if (!pt) continue;
      const [cx, cy] = this.toCanvas(pt[0], pt[1]);
      const armed = i === this.armedLandmark;
      ctx.beginPath(); // a diamond
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx + size, cy);
      ctx.lineTo(cx, cy + size);
      ctx.lineTo(cx - size, cy);
      ctx.closePath();
      ctx.fillStyle = armed ? "#ffd479" : "#e8a33d";
      ctx.fill();
      ctx.strokeStyle = "#1b1f24";
      ctx.stroke();
      const name = this.landmarkNames[i];
      if (name && this.labelsVisible) {
        ctx.fillStyle = "#e8a33d";
        ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(name, cx + size + 3, cy + 3);
      }
    }
    ctx.restore();
  }

  /** @param {number} index  landmark a click places, or -1 to disarm */
  setArmedLandmark(index) {
    if (this.armedLandmark === index) return;
    this.armedLandmark = index;
    this.draw();
  }

  /** @param {string[]} names */
  setLandmarkNames(names) {
    this.landmarkNames = names || [];
  }

  /** @param {boolean} visible */
  setLandmarksVisible(visible) {
    if (this.landmarksVisible === visible) return;
    this.landmarksVisible = visible;
    this.draw();
  }

  // The "Unplaced" placeholder seed for joint `i` -- a faint draggable ghost for a joint the
  // view has NOTHING to grab for (no GT, no detected, no reprojection), so a GT can still be
  // authored where triangulation dropped the point. Null unless the Unplaced layer is on and the
  // server sent a seed here. Suppressed the instant a real point exists for the joint (GT /
  // detected / projected) so a momentarily stale seed array never shows a ghost under a real
  // marker -- the seed only shows where the joint is genuinely absent.
  // True when joint `i` is not on this animal (amputated / ablated). Such a joint has no
  // position of its own in any view; it is drawn as a tombstone and cannot be dragged.
  /** @param {number} i @returns {boolean} */
  isAbsent(i) {
    return !!(this.absent && this.absent[i]);
  }

  // Where to draw an absent joint's tombstone: the server's seed position for the cell. Unlike
  // placeholderPos this ignores the "Unplaced" toggle -- the tombstone is the only handle the
  // operator has to select the joint and un-declare it, so it must never be hideable.
  /** @param {number} i @returns {Point | null} */
  absentPos(i) {
    if (!this.isAbsent(i)) return null;
    if (!this.placeholder || i >= this.placeholder.length || !this.placeholder[i]) return null;
    return this.placeholder[i];
  }

  /** @param {number} i @returns {Point | null} */
  placeholderPos(i) {
    if (this.instanceMode) return null; // the instance draws every cell itself
    if (!this.placeholder || i >= this.placeholder.length || !this.placeholder[i]) return null;
    if (this.gtPos(i)) return null;
    if (this.detectedVisible && this.detPos(i)) return null;
    if (this.projectedVisible && this.latentPos(i)) return null;
    return this.placeholder[i];
  }

  // One joint's node in the *editable* skeleton -- its drawn position + source, or null when it
  // has nothing to draw here. GT wins over detected. While a joint is being dragged it is authored
  // as ground truth, so render it as GT under the cursor immediately -- even before the server sets
  // its GT flag and even if it had no pixel before (a spawn from a detected / projected seed).
  // Below the GT pixel it folds in, as fallbacks, first the detector's usable
  // point (an occluded view has none -- see detPos) and then the reprojected point, so the merged
  // skeleton stays fully connected (no bone drops out just because one endpoint is only derived).
  // The reprojection stands in either when its overlay is shown, or -- whatever that toggle says --
  // when the view has no usable observation at all: an occluded ("Projected") cell HAS no position
  // but the derived one, so that is where the joint is drawn. A "projected" node normally draws no
  // filled disc (the overlay's hollow ring beneath it is its "derived, not observed" marker); with
  // the overlay hidden it draws that ring itself -- see drawSkeleton.
  /** @param {number} i @returns {{ pos: Point, src: "gt" | "detected" | "projected" | "placeholder" | "absent" } | null} */
  nodeAt(i) {
    // An absent joint is not on the animal, so it precedes every other source: it must never
    // resolve to GT / detected / projected, and drawSkeleton drops the bones that touch it.
    if (this.isAbsent(i)) {
      const ap = this.absentPos(i);
      return ap ? { pos: ap, src: "absent" } : null;
    }
    if (i === this.dragging && this.moved && this.pts[i]) {
      return { pos: this.pts[i], src: "gt" };
    }
    const g = this.gtPos(i);
    if (g) return { pos: g, src: "gt" };
    // The instance owns every position, so a non-GT node is still drawn from `pts` -- at
    // the reprojection of its point's current 3D, or its frozen seed, whichever the
    // operator chose. It reads as derived (thin ring) rather than authored (lime ring).
    if (this.instanceMode && this.pts[i]) {
      return { pos: this.pts[i], src: "projected" };
    }
    if (this.detectedVisible) {
      const d = this.detPos(i);
      if (d) return { pos: d, src: "detected" };
    }
    {
      const occluded = !!(this.invisible && this.invisible[i]);
      const p = this.latentPos(i);
      if (p && (this.projectedVisible || occluded)) return { pos: p, src: "projected" };
      // Last resort: the Unplaced seed. Without this a joint whose only position is a placeholder
      // returned null here, drawSkeleton's `if (!na || !nb) continue` dropped every bone touching
      // it, and the operator was left hunting a lone unconnected dot among 38 -- reported from the
      // GUI as "the point is actually there but it just isn't connected to anything and so really
      // difficult to spot". The seed IS the joint's drawn position, so the skeleton should reach it.
      const ph = this.placeholderPos(i);
      if (ph) return { pos: ph, src: "placeholder" };
    }
    return null;
  }

  // The best drawn position of joint `i` across the *visible* layers, for anchoring the
  // selection ring / label / hover emphasis -- so a selected or hovered joint stays marked
  // even when it has no GT yet (only a detected or projected point). Precedence follows what
  // the operator sees: GT, then the detector's usable point (none in an occluded view), then the
  // reprojection wherever that is what the joint is drawn at (see shownLatentPos). Honours the drag
  // override (a spawned GT tracks the cursor) so its ring/label follow immediately.
  /** @param {number} i @returns {Point | null} */
  anchorPos(i) {
    if (i === this.dragging && this.moved && this.pts[i]) return this.pts[i];
    const g = this.gtPos(i);
    if (g) return g;
    // The instance's own drawn position, for the same reason grabCandidates needs it: the
    // ring and the label must anchor where the joint IS, not where the reprojection is.
    if (this.instanceMode && this.pts[i]) return this.pts[i];
    if (this.detectedVisible) {
      const d = this.detPos(i);
      if (d) return d;
    }
    const p = this.shownLatentPos(i);
    if (p) return p;
    return this.placeholderPos(i); // last resort: the Unplaced seed, so its ring/label anchor
  }

  // The positions of a single point source: ground truth is the authored pixel (held in `pts`)
  // wherever the GT flag is set; detected is the detector's usable pixel -- a view flagged occluded
  // ("Projected") shows none, since the operator has rejected that pixel and the joint's position
  // there is its reprojection (see detPos). Null where the source has no point in this view. (The
  // 3D reprojection is not a source layer here -- it draws as its own overlay, the reprojected
  // skeleton; see drawReprojection.)
  /** @param {"gt" | "detected"} kind @returns {(Point | null)[]} */
  sourcePositions(kind) {
    if (kind === "detected") {
      const n = this.detected ? this.detected.length : 0;
      const det = new Array(n).fill(null);
      for (let i = 0; i < n; i++) det[i] = this.detPos(i);
      return det;
    }
    const out = new Array(this.pts.length).fill(null);
    for (let i = 0; i < this.pts.length; i++) {
      if (this.fixed && this.fixed[i] && this.pts[i]) out[i] = this.pts[i];
    }
    return out;
  }

  // The editable skeleton: the colored bones (a bone touching the hovered joint thickens, so
  // hover reads on the whole limb, not just the dot), then each joint drawn with a marker
  // whose fill + ring encode its source. Each joint
  // is GT if authored, else the detector's point, else its reprojected point when neither exists
  // and the projected overlay is shown (so the merged skeleton stays connected) -- a projected
  // node carries its bone but draws no disc, deferring to the reprojection overlay's hollow ring;
  // off, only GT is drawn (the Detected layer draws separately underneath). A joint with no node
  // at all (undetected with the projected overlay hidden, or GT off) is skipped -- it shows only
  // in the projected overlay. Selection rings and name labels are drawn by drawJointOverlay on
  // top, so they anchor consistently.
  drawSkeleton() {
    const ctx = this.ctx;
    const hi = this.highlight;
    const n = Math.max(
      this.pts.length,
      this.detected ? this.detected.length : 0,
      this.latent ? this.latent.length : 0,
    );
    /** @type {({ pos: Point, src: "gt" | "detected" | "projected" | "placeholder" | "absent" } | null)[]} */
    const nodes = new Array(n);
    for (let i = 0; i < n; i++) nodes[i] = this.nodeAt(i);
    for (const [a, b] of this.bones) {
      const na = nodes[a];
      const nb = nodes[b];
      if (!na || !nb) continue;
      // A limb that is not on the animal has no bones. This is the one node source that
      // breaks the chain on purpose -- a placeholder deliberately keeps it connected.
      if (na.src === "absent" || nb.src === "absent") continue;
      const [ax, ay] = this.toCanvas(na.pos[0], na.pos[1]);
      const [bx, by] = this.toCanvas(nb.pos[0], nb.pos[1]);
      ctx.strokeStyle = this.colors[a] || "#fff";
      ctx.lineWidth = a === hi || b === hi ? HOVER_BONE_WIDTH : BONE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      if (!node) continue;
      // A projected node is NOT an observed point, so it draws no filled disc: usually the
      // reprojection overlay's hollow ring sits right beneath it as its "derived, not observed"
      // marker, and drawJointOverlay carries its selection / hover / label. With that overlay
      // hidden the node is still here (an occluded view's position IS its reprojection), so it
      // draws the hollow ring itself -- same vocabulary, so the joint never becomes a bone that
      // ends in empty space.
      if (node.src === "placeholder") continue; // bones only; drawPlaceholders draws its marker
      if (node.src === "absent") continue; // drawAbsent draws its tombstone
      if (node.src === "projected") {
        if (!this.projectedVisible) {
          const [px, py] = this.toCanvas(node.pos[0], node.pos[1]);
          ctx.save();
          ctx.globalAlpha = PROJ_ALPHA;
          ctx.lineWidth = 2;
          ctx.strokeStyle = this.colors[i] || "#fff";
          ctx.beginPath();
          ctx.arc(px, py, POINT_RADIUS_PX * (i === this.highlight ? HOVER_SCALE : 1), 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
        continue;
      }
      const [cx, cy] = this.toCanvas(node.pos[0], node.pos[1]);
      const isHover = i === this.highlight;
      const r = POINT_RADIUS_PX * (isHover ? HOVER_SCALE : 1);
      // FILL: a filled disc in the limb palette colour -- a detected point's fill fades with
      // the detector's confidence, so faint points that want a second look read as faint.
      let fillAlpha = 1;
      if (node.src === "detected" && this.conf && i < this.conf.length && this.conf[i] != null) {
        fillAlpha = Math.max(0.4, Math.min(1, /** @type {number} */ (this.conf[i])));
      }
      ctx.globalAlpha = fillAlpha;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = this.colors[i] || "#fff";
      ctx.fill();
      ctx.globalAlpha = 1;
      // RING: ground truth gets a bold lime ring; a detected point a thin dark ring, or a
      // white one under the cursor. Hover still grows the disc and thickens the bones.
      if (node.src === "gt") {
        ctx.strokeStyle = FIXED_COLOR;
        ctx.lineWidth = 2.5;
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
    }
  }

  // One pass on top of every skeleton for the per-joint marks that must anchor consistently
  // however a joint is drawn: the selection ring, the name label, and hover emphasis. Each is
  // placed at anchorPos(i) -- GT else detected else projected -- so a selected or hovered joint
  // still reads even before it has a GT pixel (a fresh detected-only or projected-only joint).
  // Selection + hover belong to the editable layer; labels ride whenever a
  // GT/Detected skeleton is shown (else the projected overlay owns them, see draw()).
  drawJointOverlay() {
    const anySkeleton = true;
    const wantLabels = this.labelsVisible && anySkeleton;
    const wantSel = this.selectionSet.size > 0;
    const hi = this.highlight;
    const wantHover = hi != null;
    if (!wantLabels && !wantSel && !wantHover) return;
    const ctx = this.ctx;
    const n = Math.max(
      this.pts.length,
      this.detected ? this.detected.length : 0,
      this.latent ? this.latent.length : 0,
    );
    for (let i = 0; i < n; i++) {
      const needLabel = wantLabels;
      const needSel = wantSel && this.selectionSet.has(i);
      const needHover = wantHover && i === hi;
      if (!needLabel && !needSel && !needHover) continue;
      const p = this.anchorPos(i);
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      const r = POINT_RADIUS_PX * (i === hi ? HOVER_SCALE : 1);
      // Hover emphasis for a joint the editable skeleton drew no disc for -- one with no node, or
      // a projected-fallback node (which carries a bone but no marker) -- so cross-view hover still
      // reads on it. Joints drawn with a gt/detected disc already got their hover-scaled marker +
      // white ring in drawSkeleton.
      const node = this.nodeAt(i);
      if (needHover && (!node || node.src === "projected")) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      // A selected joint gets an extra outer ring, distinct from every source ring so the two
      // can coexist.
      if (needSel) {
        ctx.beginPath();
        ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
        ctx.strokeStyle = SELECT_COLOR;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (needLabel) this.drawLabel(i, cx, cy, r);
    }
  }

  // The reprojection-distance warning pass: for each joint whose anchor -- the authored GT pixel,
  // else the detector's raw prediction (NOT the projected fallback) -- sits farther than
  // warnThreshold IMAGE px from its 3D reprojection, draw a connector from the anchor to the
  // reprojected point and a ring on the anchor, coloured amber at the threshold and ramping to red
  // at WARN_RED_MULT x it. The gap is measured on the raw image-space coords (BEFORE toCanvas), so
  // the threshold is resolution-meaningful and stable under zoom/pan. Occluded joints (the operator
  // deliberately dropped the observation, so a mismatch is expected) and the actively-dragged joint
  // (its anchor is pinned to the cursor, not the solve) are skipped. Purely visual: hit-testing is
  // data-driven and never consults what is drawn, so this cannot disturb hover / selection / drag.
  // Absent joints are skipped: they have no 3D by construction, so a distance warning there
  // would be a permanent red flag against geometry that should not exist.
  drawReprojWarnings() {
    const thr = this.warnThreshold;
    const latent = this.latent;
    if (!(thr > 0) || !latent) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.lineCap = "round";
    const n = Math.min(this.pts.length, latent.length);
    for (let i = 0; i < n; i++) {
      if (i === this.dragging) continue; // don't fight the cursor mid-drag
      if (this.invisible && this.invisible[i]) continue; // occluded: observation intentionally dropped
      const proj = latent[i];
      if (!proj) continue;
      const anchor = this.gtPos(i) || this.detPos(i); // GT, else predicted; else nothing to check
      if (!anchor) continue;
      const d = Math.hypot(anchor[0] - proj[0], anchor[1] - proj[1]); // IMAGE px
      if (d <= thr) continue;
      const t = Math.max(0, Math.min(1, (d / thr - 1) / (WARN_RED_MULT - 1))); // 0 at thr, 1 at WARN_RED_MULT x thr
      const rgb = mixRgb(WARN_AMBER, WARN_RED, t);
      const [ax, ay] = this.toCanvas(anchor[0], anchor[1]);
      const [px, py] = this.toCanvas(proj[0], proj[1]);
      const r = POINT_RADIUS_PX * (i === this.highlight ? HOVER_SCALE : 1) + WARN_RING_PAD;
      // A dark casing under a coloured top, so the cue reads on any frame (mirrors strokeLeash).
      /** @type {[string, number][]} */
      const passes = [["rgba(0,0,0,0.55)", 4], [`rgba(${rgb},${0.85 + 0.15 * t})`, 2]];
      for (const [style, width] of passes) {
        ctx.strokeStyle = style;
        ctx.lineWidth = width;
        ctx.beginPath(); // the connector: anchor -> where the 3D reprojects it
        ctx.moveTo(ax, ay);
        ctx.lineTo(px, py);
        ctx.stroke();
        ctx.beginPath(); // the warning ring on the anchor
        ctx.arc(ax, ay, r, 0, Math.PI * 2);
        ctx.stroke();
        // Mark the reprojection end too when the Projected overlay is off (which would otherwise
        // draw its own hollow ring there) -- so the connector never points at empty space.
        if (!this.projectedVisible) {
          ctx.beginPath();
          ctx.arc(px, py, POINT_RADIUS_PX, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
    ctx.restore();
  }

  // One point source (ground truth or the raw detections) drawn as its own read-only layer, at
  // that source's own positions, in the limb palette -- the same marker vocabulary the combined
  // skeleton uses, so a source reads the same whether it is merged in or inspected on its own:
  //   ground truth -> a filled palette disc under a bold lime ring
  //   detected     -> a filled palette disc (faded by the detector's confidence) under a thin dark ring
  // (The 3D reprojection is NOT a source layer -- it is the reprojected skeleton, its own overlay;
  // see drawReprojection.) `withBones` draws the bones between two present points (on for a
  // standalone layer; off when overlaid on the combined skeleton, which already carries the
  // bones). `withLabels` draws the per-joint name labels -- only one layer should, so the combined
  // skeleton owns them when it is shown, and only the first visible standalone source owns them
  // otherwise (no doubling). These layers are read-only, so their markers do NOT hover-scale
  // (unlike the combined skeleton) -- a fixed size avoids a stale cross-view hover leaving a lone
  // marker enlarged.
  /** @param {"gt" | "detected"} kind @param {boolean} withBones @param {boolean} withLabels */
  drawSourceLayer(kind, withBones, withLabels) {
    const ctx = this.ctx;
    const pos = this.sourcePositions(kind);
    if (withBones) {
      ctx.save();
      ctx.lineWidth = BONE_WIDTH;
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
      // FILL: a filled palette disc; a detection fades with the detector's confidence.
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
      // RING: ground truth = bold lime; detected = a thin dark ring.
      if (kind === "gt") {
        ctx.strokeStyle = FIXED_COLOR;
        ctx.lineWidth = 2.5;
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

  // The reprojected 3D drawn as its own independent overlay: the full reprojected skeleton in the
  // limb palette -- hollow rings at every reprojected joint joined by thick, semi-transparent,
  // DASHED edges. It reads as a distinct "derived, not observed" layer beneath the solid editable
  // skeleton (and stands alone when that is hidden); where a reprojected point sits off its placed
  // pixel, that gap is the live disagreement. Every joint is shown at once (no hover gating), so a
  // live 3D drag makes the whole thing track the estimate. Its hollow points double as spawn seeds
  // -- grabbing one authors a ground-truth point there (see nearestPoint). `withLabels` draws the
  // per-joint names, on only when this is the sole visible layer (nothing above claims them).
  /** @param {Point[]} pts  the reprojected points for this view @param {boolean} withLabels */
  drawReprojection(pts, withLabels) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = PROJ_ALPHA;
    ctx.lineJoin = "round";
    ctx.lineWidth = PROJ_WIDTH;
    ctx.setLineDash(PROJ_DASH);
    for (const [a, b] of this.bones) {
      const pa = pts[a];
      const pb = pts[b];
      if (!pa || !pb) continue;
      const [ax, ay] = this.toCanvas(pa[0], pa[1]);
      const [bx, by] = this.toCanvas(pb[0], pb[1]);
      ctx.strokeStyle = this.colors[a] || "#fff";
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    // Hollow rings, drawn solid (the dash is the edges' cue, not the nodes').
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      ctx.strokeStyle = this.colors[i] || "#fff";
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS_PX, 0, Math.PI * 2);
      ctx.stroke();
    }
    // Labels at full opacity (the overlay itself is translucent, but faint text is unreadable).
    if (withLabels) {
      ctx.globalAlpha = 1;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        if (!p) continue;
        const [cx, cy] = this.toCanvas(p[0], p[1]);
        this.drawLabel(i, cx, cy, POINT_RADIUS_PX);
      }
    }
    ctx.restore();
  }

  // The "Unplaced" layer: a faint, draggable seed at every joint this view has nothing else to grab
  // (no GT / detected / reprojected point) -- e.g. a joint triangulation rejected, or one the
  // detector never fired. Each is drawn as a small dashed hollow ring with a faint centre dot in
  // the joint's limb palette at reduced opacity, so it reads as "not observed -- drag me to place"
  // rather than an observed point. Its position is the server's sensible seed (the raw detection,
  // else a nearby frame's pixel, else a neighbour / view centroid). Grabbing one authors a
  // ground-truth point there (see grabCandidates); placeholderPos suppresses it the instant a real
  // point exists. The actively-dragged joint is skipped -- it draws as GT under the cursor. The
  // selection ring, hover emphasis and name label ride the shared drawJointOverlay pass (its
  // anchor falls through to placeholderPos), so a selected / hovered missing joint still reads.
  // The tombstone layer: a joint the operator has declared NOT on this animal (an amputated leg,
  // an ablated antenna). Drawn as a small dim grey cross -- no fill, no ring, and drawSkeleton
  // drops every bone touching it, so the limb visibly ends at the stump. Deliberately always
  // drawn (no toggle): it is the only handle for selecting the joint to un-declare it, and a
  // declared amputation must not be confusable with "nobody has looked here yet".
  drawAbsent() {
    if (!this.absent) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = ABSENT_ALPHA;
    ctx.strokeStyle = ABSENT_COLOR;
    ctx.lineWidth = 2;
    for (let i = 0; i < this.absent.length; i++) {
      const p = this.absentPos(i);
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      const r = ABSENT_MARK_PX;
      ctx.beginPath();
      ctx.moveTo(cx - r, cy - r);
      ctx.lineTo(cx + r, cy + r);
      ctx.moveTo(cx + r, cy - r);
      ctx.lineTo(cx - r, cy + r);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawPlaceholders() {
    const ctx = this.ctx;
    ctx.save();
    for (let i = 0; i < this.placeholder.length; i++) {
      if (i === this.dragging && this.moved) continue; // the active drag owns this joint (GT under the cursor)
      const p = this.placeholderPos(i);
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      const color = this.colors[i] || "#fff";
      ctx.globalAlpha = PLACEHOLDER_ALPHA;
      ctx.fillStyle = color;
      ctx.beginPath(); // a faint centre dot marks the grab point
      ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = color; // a dashed hollow ring: "not observed"
      ctx.lineWidth = 1.5;
      ctx.setLineDash(PLACEHOLDER_DASH);
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS_PX, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  // A reference overlay (the NMF model) as a receding under-glow beneath the editable
  // skeleton: a soft, translucent, wider skeleton in the overlay's colour whose only
  // job is to give the reference's overall shape without competing with the palette on
  // top. When `ghost` is false (the editable skeleton is hidden) it instead draws in a
  // bright, distinct line style -- dotted, with hollow square markers -- so it stays
  // fully legible as the only thing on screen.
  /**
   * @param {Point[]} pts  the reference points to draw
   * @param {string} rgb  the overlay's "r,g,b" colour
   * @param {boolean} ghost  true = soft under-glow; false = bright, on its own
   */
  drawReference(pts, rgb, ghost) {
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

  // The grabbable positions of joint `i`, for hit-testing. A drag on any of them authors ground
  // truth for that joint: its GT pixel (grabbing it MOVES the GT), its detected point (when the
  // Detected layer is shown and the view has not been flagged occluded -- grabbing it SPAWNS a GT
  // there), and its reprojected point wherever that is drawn as the joint's position (likewise a
  // spawn seed; see shownLatentPos). So whatever the operator sees is grabbable, and every grab
  // resolves to the same authoring gesture on the joint index.
  /** @param {number} i @returns {(Point | null)[]} */
  grabCandidates(i) {
    // An absent joint offers only its tombstone, so click- and rubber-band selection still find
    // it (you must be able to select it to un-declare it). The press handler refuses the drag.
    if (this.isAbsent(i)) return [this.absentPos(i)];
    return [
      this.gtPos(i),
      // The instance's own drawn position. Without this a non-GT joint is grabbable only by
      // coincidence -- `pts[i]` equals the reprojection under the default display, so
      // `shownLatentPos` happens to answer at the same pixel. Hide the reprojection (`p`) and
      // it answers null; switch to seed positions (`s`) and it answers at a DIFFERENT pixel
      // than the one drawn. Either way the joint stops being clickable while still visible --
      // and with the detections auto-hidden, the contralateral joints this project exists for
      // would be the ones to go.
      this.instanceMode ? this.pts[i] : null,
      this.detectedVisible ? this.detPos(i) : null,
      this.shownLatentPos(i),
      // The Unplaced placeholder is the lowest-priority seed: it only exists where the three
      // above are absent (see placeholderPos), so it never competes with a real point.
      this.placeholderPos(i),
    ];
  }

  // The joint nearest the pointer within the grab tolerance, or null -- the nearest of any joint's
  // grab candidates (see grabCandidates).
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
    const n = Math.max(
      this.pts.length,
      this.detected ? this.detected.length : 0,
      this.latent ? this.latent.length : 0,
    );
    for (let i = 0; i < n; i++) {
      for (const p of this.grabCandidates(i)) {
        if (!p) continue;
        const d = Math.hypot(p[0] - ix, p[1] - iy);
        if (d <= bestD) {
          best = i;
          bestD = d;
        }
      }
    }
    return best;
  }

  // Every joint the image-space rect encloses -- the Shift+drag marquee's hit-test. Corners come
  // in any order; they are normalized here. A joint counts if ANY of its grab candidates lands
  // inside, matching nearestPoint so whatever is grabbable is also marquee-selectable.
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
    const inside = (/** @type {Point} */ p) => !!p && p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1;
    const n = Math.max(
      this.pts.length,
      this.detected ? this.detected.length : 0,
      this.latent ? this.latent.length : 0,
    );
    for (let i = 0; i < n; i++) {
      if (this.grabCandidates(i).some(inside)) hits.push(i);
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
    const canGrab = this.canGrab;
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
      // A joint that is not on this animal cannot be placed anywhere: selecting it is how you
      // reach the un-declare gesture, but dragging it would author ground truth for a limb that
      // does not exist (and the server refuses it anyway). Select and stop.
      if (this.isAbsent(point)) return;
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
    if (this.canGrab) {
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
    // A landmark is armed: this click places it, and nothing else runs. Handled first and
    // exclusively, because a landmark has no joint to grab and no bones to select -- letting
    // the selection machinery also fire would make one click do two unrelated things.
    if (
      this.armedLandmark >= 0 &&
      this.canGrab &&
      !this.moved &&
      !this.marqueeing &&
      this.dragging === null &&
      e.type !== "pointercancel"
    ) {
      e.preventDefault();
      const [ix, iy] = this.toImage(...this.cssXY(e));
      this.cb.onPlaceLandmark?.(this.viewIndex, this.armedLandmark, ix, iy);
      return;
    }
    if (this.marqueeing) {
      e.preventDefault();
      this.marqueeing = false;
      const rect = this.marquee;
      this.marquee = null;
      const additive = this.marqueeAdditive; // Ctrl/Cmd = add; Shift = replace
      const canGrab = this.canGrab;
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
   * Wheel input ALWAYS zooms toward the cursor -- deliberately device-agnostic. A mouse wheel,
   * a trackpad two-finger scroll, and a trackpad pinch (a ctrlKey wheel the browser synthesizes)
   * all zoom. A wheel event's shape can't reliably tell a mouse from a trackpad (macOS scroll
   * acceleration makes a mouse notch look "precise", like a trackpad), and every attempt to
   * split them mis-routed the mouse to panning; zoom-on-scroll is what the annotator wants.
   * Panning stays on drag (empty-space grab, see onPointerDown). A tiny per-event delta (a
   * pinch, or an accelerated mouse notch) gets the higher gain so it still tracks; a chunky
   * wheel/line-mode notch keeps the slower notch rate.
   * @param {WheelEvent} e
   */
  onWheel(e) {
    if (!this.zoomable) return;
    e.preventDefault();
    const [mx, my] = this.cssXY(e);
    const tiny = e.deltaMode === 0 && Math.abs(e.deltaY) < WHEEL_NOTCH_MIN;
    this.zoomAtCursor(mx, my, e.deltaY, tiny ? PINCH_ZOOM_RATE : WHEEL_ZOOM_RATE);
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

  /** @param {MouseEvent} e */
  onDblClick(e) {
    // Double-clicking a joint selects that keypoint across every view (a fast way to
    // act on one point everywhere); double-clicking empty space resets the zoom.
    if (this.canGrab) {
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
