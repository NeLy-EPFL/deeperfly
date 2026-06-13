// @ts-check
// One camera's frame plus its draggable 2D skeleton overlay, on a <canvas>.
//
// A port of the old Qt PoseView, grown a few editor conveniences: the frame is
// drawn fit-to-canvas (letterboxed) and can be zoomed (wheel, toward the cursor)
// and panned (drag on empty space); the skeleton is drawn in image-pixel
// coordinates mapped through that fit+zoom. Pointer events pick the nearest joint
// within a screen-pixel tolerance. A press on a joint selects it; an actual drag
// (past a small threshold) moves it, emitting a throttled `onDragging` and a final
// `onDragged` -- a click without movement just selects, so it never creates a
// spurious edit. Right-click (or a tap while "pin mode" is on) toggles a point's
// fixed flag. An "invisible" (obscured) joint is drawn ghosted; dragging it is
// allowed and reports `wasInvisible` on `onDragged` so the app can un-obscure it.
// Hovering a joint reports it via `onHover` so the app can emphasize the same point
// across every view. The app stays in control of what a drag does to the 3D point.
//
// Beyond the editable overlay the view can also draw two read-only extras that the
// app toggles: the "latent" skeleton (the current 3D estimate reprojected and drawn
// as a bright dashed overlay on top, so you can see where triangulation puts each
// point even when it lands on the editable skeleton) and per-joint name labels.
//
// This .js is the source -- no build step; VS Code type-checks it via
// `// @ts-check` and the JSDoc types.

/** @typedef {import("./types.js").Point} Point */

/**
 * @typedef {object} PoseViewCallbacks
 * @property {(view: number, point: number, x: number, y: number) => void} onDragging
 * @property {(view: number, point: number, x: number, y: number, wasInvisible: boolean) => void} onDragged
 * @property {(view: number, point: number) => void} onToggleFixed
 * @property {(view: number, point: number) => void} onSelect  a joint was clicked/grabbed
 * @property {(point: number | null) => void} onHover  the hovered joint changed (cross-view)
 */

const POINT_RADIUS_PX = 4; // drawn joint radius in screen px (constant under zoom)
const HOVER_SCALE = 1.9; // how much a hovered joint grows
const HIT_TOLERANCE_PX = 14; // how close a click must be to grab a joint, screen px
const DRAG_THRESHOLD_PX = 3; // movement (screen px) before a press becomes a drag
const MAX_ZOOM = 10; // cap on the user wheel-zoom factor over fit
const WHEEL_ZOOM_RATE = 0.0015; // wheel delta -> zoom factor sensitivity

const FIXED_COLOR = "#7CFC00"; // ring on a fixed (finalized) point (lime green)
const INVISIBLE_COLOR = "#ff5dd0"; // dashed ring on an invisible (obscured) point (magenta)
const INVISIBLE_FILL_ALPHA = 0.35; // an obscured joint's fill is dimmed to read as "ghosted"
const SELECT_COLOR = "#3fd0ff"; // ring on the last-selected point (cyan; lime = fixed)
const LATENT_COLOR = "rgba(255,176,64,0.95)"; // the latent-skeleton overlay (amber, drawn on top)

export class PoseView {
  /**
   * @param {number} viewIndex
   * @param {HTMLCanvasElement} canvas
   * @param {PoseViewCallbacks} cb
   * @param {() => boolean} pinMode  whether a tap toggles fixed instead of dragging
   */
  constructor(viewIndex, canvas, cb, pinMode) {
    /** @type {HTMLImageElement | null} */
    this.img = null;
    /** @type {[number, number][]} */
    this.bones = [];
    /** @type {string[]} */
    this.colors = [];
    /** @type {Point[]} */
    this.pts = [];
    /** @type {Point[] | null} */
    this.latent = null; // latent 3D reprojection (display only), drawn when latentVisible
    /** @type {boolean[] | null} */
    this.fixed = null;
    /** @type {boolean[] | null} */
    this.invisible = null;
    /** @type {string[]} */
    this.pointNames = [];
    /** @type {number | null} */
    this.highlight = null; // hovered joint (set by the app across all views)
    /** @type {number | null} */
    this.selected = null; // last-selected joint, shown only in its own view
    this.editable = false;
    this.zoomable = false;
    this.overlayVisible = true;
    this.latentVisible = false;
    this.labelsVisible = false;
    /** @type {number | null} */
    this.dragging = null;
    this.dragInvisible = false; // was the grabbed joint obscured? (reported on release)
    this.panning = false;
    this.moved = false; // has the current press moved past the drag threshold?

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
    this.pinMode = pinMode;

    canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
    canvas.addEventListener("pointercancel", (e) => this.onPointerUp(e));
    canvas.addEventListener("pointerleave", () => this.onPointerLeave());
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
    this.layoutAndDraw();
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

  /** @param {Point[]} pts */
  setPoints(pts) {
    // Keep the actively dragged joint pinned to the cursor: the server's live
    // re-solve reprojects it a hair off, and letting that fight the mouse feels
    // like resistance (mirrors the old Qt PoseView.set_points).
    const held = this.dragging !== null ? this.pts[this.dragging] : null;
    this.pts = pts.slice();
    if (this.dragging !== null && held) this.pts[this.dragging] = held;
    this.draw();
  }

  /** @param {boolean[] | null} fixed */
  setFixed(fixed) {
    this.fixed = fixed;
    this.draw();
  }

  /** @param {boolean[] | null} invisible  per-point "obscured" mask, or null when not in 3D */
  setInvisible(invisible) {
    this.invisible = invisible;
    this.draw();
  }

  /** @param {Point[] | null} pts  the latent 3D reprojection to ghost, or null */
  setLatent(pts) {
    this.latent = pts;
    if (this.latentVisible) this.draw();
  }

  /** @param {number | null} point */
  setHighlight(point) {
    if (this.highlight === point) return;
    this.highlight = point;
    this.draw();
  }

  /** @param {number | null} point  the selected joint, or null if not selected in this view */
  setSelected(point) {
    if (this.selected === point) return;
    this.selected = point;
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

  /** @param {boolean} visible  whether the editable skeleton + joints are drawn */
  setOverlayVisible(visible) {
    if (this.overlayVisible === visible) return;
    this.overlayVisible = visible;
    this.draw();
  }

  /** @param {boolean} visible  whether the latent 3D reprojection is ghosted on top */
  setLatentVisible(visible) {
    if (this.latentVisible === visible) return;
    this.latentVisible = visible;
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
    if (this.img) {
      ctx.drawImage(this.img, this.offX, this.offY, this.imgW * this.scale, this.imgH * this.scale);
    }
    if (this.overlayVisible) {
      // bones first, joints on top
      ctx.lineWidth = 1.5;
      for (const [a, b] of this.bones) {
        const pa = this.pts[a];
        const pb = this.pts[b];
        if (!pa || !pb) continue;
        const [ax, ay] = this.toCanvas(pa[0], pa[1]);
        const [bx, by] = this.toCanvas(pb[0], pb[1]);
        ctx.strokeStyle = this.colors[a] || "#fff";
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      for (let i = 0; i < this.pts.length; i++) {
        const p = this.pts[i];
        if (!p) continue;
        const [cx, cy] = this.toCanvas(p[0], p[1]);
        const isHover = i === this.highlight;
        const isFixed = this.fixed != null && i < this.fixed.length && this.fixed[i];
        const isInvisible = this.invisible != null && i < this.invisible.length && this.invisible[i];
        const r = POINT_RADIUS_PX * (isHover ? HOVER_SCALE : 1);
        // An obscured joint reads as "ghosted": a dimmed fill under a dashed ring.
        ctx.globalAlpha = isInvisible ? INVISIBLE_FILL_ALPHA : 1;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = this.colors[i] || "#fff";
        ctx.fill();
        ctx.globalAlpha = 1;
        if (isInvisible) {
          ctx.strokeStyle = INVISIBLE_COLOR;
          ctx.lineWidth = 2;
          ctx.setLineDash([3, 2]);
        } else if (isFixed) {
          ctx.strokeStyle = FIXED_COLOR;
          ctx.lineWidth = 2.5;
        } else if (isHover) {
          ctx.strokeStyle = "white";
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = "rgba(0,0,0,0.6)";
          ctx.lineWidth = 1;
        }
        ctx.stroke();
        ctx.setLineDash([]);
        // The last-selected joint gets an extra outer ring (only its own view sets
        // `selected`), distinct from the lime "fixed" ring so the two can coexist.
        if (i === this.selected) {
          ctx.beginPath();
          ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
          ctx.strokeStyle = SELECT_COLOR;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        if (this.labelsVisible) this.drawLabel(i, cx, cy, r);
      }
    }
    // The latent reprojection is read-only, so draw it *on top* of the editable
    // overlay: when calibration is good it lands right on the skeleton, and drawing
    // it underneath an opaque overlay would hide it. It can also show on its own
    // (e.g. with the skeleton toggled off).
    if (this.latentVisible && this.latent) this.drawLatent();
  }

  // The latent 3D reprojection: a dashed skeleton with small hollow joints in one
  // bright colour, so it reads as a distinct reference over the editable overlay.
  drawLatent() {
    const ctx = this.ctx;
    const latent = this.latent;
    if (!latent) return;
    ctx.save();
    ctx.strokeStyle = LATENT_COLOR;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    for (const [a, b] of this.bones) {
      const pa = latent[a];
      const pb = latent[b];
      if (!pa || !pb) continue;
      const [ax, ay] = this.toCanvas(pa[0], pa[1]);
      const [bx, by] = this.toCanvas(pb[0], pb[1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const p of latent) {
      if (!p) continue;
      const [cx, cy] = this.toCanvas(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
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
    for (let i = 0; i < this.pts.length; i++) {
      const p = this.pts[i];
      if (!p) continue;
      const d = Math.hypot(p[0] - ix, p[1] - iy);
      if (d <= bestD) {
        best = i;
        bestD = d;
      }
    }
    return best;
  }

  /** @param {PointerEvent} e */
  onPointerDown(e) {
    const [mx, my] = this.cssXY(e);
    this.downX = mx;
    this.downY = my;
    this.moved = false;
    const [ix, iy] = this.toImage(mx, my);
    const canGrab = this.editable && this.overlayVisible;
    const point = canGrab ? this.nearestPoint(ix, iy) : null;

    if (point !== null) {
      // Tap-to-pin (touch-friendly) or right-click both toggle fixed, no drag.
      if (this.pinMode() || e.button === 2) {
        e.preventDefault();
        this.cb.onSelect(this.viewIndex, point);
        this.cb.onToggleFixed(this.viewIndex, point);
        return;
      }
      if (e.button !== 0) return; // only the primary button drags
      e.preventDefault();
      this.cb.onSelect(this.viewIndex, point); // selecting happens on press, not release
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
    if (this.editable && this.overlayVisible) {
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
    }
  }

  onPointerLeave() {
    // Drop the hover when the cursor leaves so no view stays falsely emphasized.
    if (this.dragging === null && this.panning === false && this._hover !== null) {
      this._hover = null;
      this.cb.onHover(null);
    }
  }

  /** @param {WheelEvent} e */
  onWheel(e) {
    if (!this.zoomable) return;
    e.preventDefault();
    const [mx, my] = this.cssXY(e);
    const z0 = this.zoom;
    let z1 = Math.min(MAX_ZOOM, Math.max(1, z0 * Math.exp(-e.deltaY * WHEEL_ZOOM_RATE)));
    if (z1 === z0) return;
    if (z1 <= 1.0001) {
      this.resetZoom(); // snap cleanly back to the letterboxed fit
      return;
    }
    // Keep the image point under the cursor fixed while the zoom changes.
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
    if (!this.zoomable) return;
    e.preventDefault();
    this.resetZoom();
  }
}
