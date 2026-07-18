// @ts-check
// The editor controller: lays out one PoseView per camera and routes edits to
// the server. There is one unified editing model -- a drag authors ground-truth 2D at
// the drop and, when the result carries 3D, re-solves the 3D point live and refreshes
// every view (pinning the dragged view on release). Each joint is drawn with a marker
// whose style tells its source apart: ground truth (lime ring over a filled disc),
// detector prediction (dark ring over a disc that fades with confidence), or a point
// derived by reprojecting the 3D (a hollow palette circle -- no observation in this
// view). A selected joint's per-view state (predicted / ground truth / occluded) is
// shown in a status widget and set by clicking it or the `l` / `o` keys; occluding a
// view deletes its observation so it drops from the triangulation and then shows as a
// derived point, and dragging it back in un-occludes it.
//
// Two layouts share the same PoseView instances. "grid" shows every camera in an
// equal grid; "focus" shows one large editable view plus a strip of live,
// clickable thumbnails (the other cameras) -- which keeps each camera big enough
// to correct precisely when there are many cameras. The large view(s) can be
// zoomed (wheel) and panned (drag on empty space); thumbnails always show the
// whole frame. Every view stays live in both layouts, so a 3D re-solve still
// animates the thumbnails. Grid is the default; the layout switch (or f / g) toggles
// it and the [ / ] keys cycle which camera is focused.
//
// Hovering a joint emphasizes the same joint in every view; clicking a joint
// selects it (a cyan ring marks the last selection) so the two Reset buttons can
// revert it -- in just its view, or across all views.
//
// Display extras the operator toggles: the editable skeleton itself, per-joint
// name labels, and the read-only "3D estimate" skeleton (the triangulated estimate
// reprojected, ghosted over every view). A non-modal floating panel shows the 3D
// view -- the camera rig, the 3D pose, and the fitted NMF skeleton + mesh (see
// scene3d.js); it overlays the editor without blocking it (drag the title bar to move
// it, the corner to resize), so the main frame scrubber still steps the 3D pose
// through time. Almost everything has a keyboard shortcut; `?` opens a help list of them.
//
// This .js is the source -- there is no build step. VS Code type-checks it via
// `// @ts-check` and the JSDoc payload types in types.js.

import { EditSocket, fetchCorrected, fetchMeta, fetchNmfAsset, fetchNmfVerts, fetchPoints, fetchScene, frameUrl, saveCorrections, shutdownServer } from "./api.js";
import { MeshGL } from "./meshGL.js";
import { PoseView } from "./poseView.js";
import { Scene3D } from "./scene3d.js";

/** @typedef {import("./types.js").Meta} Meta */
/** @typedef {import("./types.js").PointsPayload} PointsPayload */
/** @typedef {import("./types.js").CorrectedFrame} CorrectedFrame */
/** @typedef {import("./types.js").EditMode} EditMode */
/** @typedef {"grid" | "focus"} Layout */
/** @typedef {{ key: string, mod?: boolean, shift?: boolean, global?: boolean, hidden?: boolean, label: string, desc: string, run: (e: KeyboardEvent) => void }} Binding */
/** @typedef {{ root: HTMLDivElement, set: (value: string) => void, setDisabled: (disabled: boolean) => void }} Segmented */

// The published "NeuroMechFly keypoint locations" reference (the docs site). It is
// opened in a new tab on demand, so its heavy model + WASM assets are fetched only
// when the operator asks for it -- never on editor load.
const KEYPOINTS_DOC_URL = "https://nely-epfl.github.io/deeperfly/keypoints/viewer.html";

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @returns {T}
 */
function el(id) {
  return /** @type {T} */ (document.getElementById(id));
}

/**
 * Build a two-choice "switch" -- a row of buttons with exactly one active -- as
 * a compact stand-in for a 2-option dropdown. `set(value)` highlights the active
 * button; clicking a button calls `onChange` with its value.
 * @param {[string, string][]} options  [label, value] pairs
 * @param {(value: string) => void} onChange
 * @returns {Segmented}
 */
function segmented(options, onChange) {
  const root = document.createElement("div");
  root.className = "segmented";
  /** @type {Map<string, HTMLButtonElement>} */
  const buttons = new Map();
  for (const [label, value] of options) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seg-btn";
    btn.textContent = label;
    btn.addEventListener("click", () => onChange(value));
    buttons.set(value, btn);
    root.append(btn);
  }
  return {
    root,
    set: (value) => buttons.forEach((btn, v) => btn.classList.toggle("is-active", v === value)),
    setDisabled: (disabled) => buttons.forEach((btn) => (btn.disabled = disabled)),
  };
}

/**
 * Does a keydown event match a binding (modifiers respected)?
 * @param {KeyboardEvent} e
 * @param {Binding} b
 */
function matches(e, b) {
  if (b.mod) {
    return (
      (e.ctrlKey || e.metaKey) &&
      e.key.toLowerCase() === b.key &&
      (b.shift ?? false) === e.shiftKey
    );
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  return e.key === b.key; // shift is implied by the key itself (e.g. "?", "R")
}

class App {
  /** @type {Meta} */
  meta;
  /** @type {PoseView[]} */
  views = [];
  /** @type {HTMLDivElement[]} */
  cells = [];
  /** @type {EditSocket} */
  socket;
  // Monotonic id stamped on every edit sent over the socket. A reply carrying an
  // older seq is a superseded mid-drag re-solve and is dropped in applyPoints, so a
  // reply landing after release can't repaint the joint to a stale spot (snap-back).
  editSeq = 0;
  frame = 0;
  // The editor has a single, unified editing model: a drag authors ground-truth 2D and
  // (when the result has 3D) re-solves the 3D live -- there is no user-facing 2D/3D mode
  // to switch. `mode` is just the wire value that requests the plain per-view overlay
  // (ground truth over prediction, with a projection fallback the canvas draws as a
  // distinct source); whether a drag re-solves 3D is decided by `meta.has_3d`, not this.
  /** @type {EditMode} */
  mode = "edit_2d";
  dirty = false;
  /** @type {Layout} */
  layout = "grid";
  focused = 0;
  // The last joint the operator clicked, and the view they clicked it in -- what
  // the two Reset buttons act on.
  /** @type {number | null} */
  selectedPoint = null;
  selectedView = 0;
  // The latest per-view ground-truth / occluded masks (from the points payload), so the
  // status widget can report the selected joint's source. Null until the first payload.
  /** @type {boolean[][] | null} */
  fixedMask = null;
  /** @type {boolean[][] | null} */
  invisibleMask = null;
  // On-demand 3D view (rig + 3D pose + NMF skeleton/mesh), built lazily on first open.
  /** @type {Scene3D | null} */
  scene = null;
  sceneOpen = false;
  helpOpen = false;
  helpBuilt = false;
  // True once a deliberate Close is under way: stops the unsaved-changes guard
  // (beforeunload) from nagging after the operator has already decided.
  closing = false;
  closeConfirmOpen = false;
  // The corrected-frames side panel: the list (sorted, with per-frame counts), whether
  // the panel is open, the row elements keyed by frame (for the current-frame
  // highlight), and a debounce timer coalescing post-edit refreshes.
  /** @type {CorrectedFrame[]} */
  correctedFrames = [];
  framesOpen = false;
  /** @type {Map<number, HTMLTableRowElement>} */
  frameRows = new Map();
  correctedTimer = 0;
  /** @type {Binding[]} */
  bindings = [];

  /** @type {HTMLDivElement} */
  viewsEl = el("views");
  /** @type {HTMLDivElement} */
  stageEl = el("stage");
  /** @type {HTMLDivElement} */
  stripEl = el("strip");
  /** @type {HTMLInputElement} */
  slider = el("frame-slider");
  /** @type {HTMLInputElement} */
  number = el("frame-number");
  /** @type {HTMLSpanElement} */
  totalEl = el("frame-total");
  /** @type {Segmented} */
  layoutSwitch;
  /** @type {HTMLDivElement} */
  layoutWrap = el("layout-wrap");
  /** @type {HTMLInputElement} */
  skeletonCheck = el("show-skeleton");
  /** @type {HTMLInputElement} */
  labelsCheck = el("show-labels");
  /** @type {HTMLLabelElement} */
  latentWrap = el("latent-wrap");
  /** @type {HTMLInputElement} */
  latentCheck = el("show-latent");
  /** @type {HTMLLabelElement} */
  nmfWrap = el("nmf-wrap");
  /** @type {HTMLInputElement} */
  nmfCheck = el("show-nmf");
  /** @type {HTMLLabelElement} */
  meshWrap = el("mesh-wrap");
  /** @type {HTMLInputElement} */
  meshCheck = el("show-mesh");
  /** @type {MeshGL | null} */
  meshGL = null;
  meshAssetLoaded = false;
  meshReq = 0;
  meshTimer = 0;
  /** @type {HTMLDivElement} */
  pointStatus = el("point-status");
  /** @type {HTMLSpanElement} */
  pointStatusName = el("point-status-name");
  /** @type {Segmented} */
  stateSwitch;
  /** @type {HTMLButtonElement} */
  resetViewBtn = el("reset-view");
  /** @type {HTMLButtonElement} */
  resetAllBtn = el("reset-all");
  /** @type {HTMLButtonElement} */
  resetFrameBtn = el("reset-frame");
  /** @type {HTMLButtonElement} */
  confirmPointBtn = el("confirm-point");
  /** @type {HTMLButtonElement} */
  confirmFrameBtn = el("confirm-frame");
  /** @type {HTMLButtonElement} */
  undoBtn = el("undo");
  /** @type {HTMLButtonElement} */
  redoBtn = el("redo");
  /** @type {HTMLButtonElement} */
  keypointsBtn = el("keypoints");
  /** @type {HTMLButtonElement} */
  camerasBtn = el("cameras");
  /** @type {HTMLButtonElement} */
  helpBtn = el("help");
  /** @type {HTMLButtonElement} */
  saveBtn = el("save");
  /** @type {HTMLButtonElement} */
  closeBtn = el("close-editor");
  /** @type {HTMLSpanElement} */
  statusEl = el("status");
  /** @type {HTMLButtonElement} */
  framesToggleBtn = el("frames-toggle");
  /** @type {HTMLSpanElement} */
  framesCountEl = el("frames-count");
  /** @type {HTMLElement} */
  sidebarEl = el("sidebar");
  /** @type {HTMLButtonElement} */
  framesCollapseBtn = el("frames-collapse");
  /** @type {HTMLButtonElement} */
  framesPrevBtn = el("frames-prev");
  /** @type {HTMLButtonElement} */
  framesNextBtn = el("frames-next");
  /** @type {HTMLTableSectionElement} */
  framesTbody = /** @type {HTMLTableElement} */ (el("frames-table")).tBodies[0];
  /** @type {HTMLDivElement} */
  framesEmptyEl = el("frames-empty");
  /** @type {HTMLDivElement} */
  closeOverlay = el("close-overlay");
  /** @type {HTMLButtonElement} */
  closeCancelBtn = el("close-cancel");
  /** @type {HTMLButtonElement} */
  closeDiscardBtn = el("close-discard");
  /** @type {HTMLButtonElement} */
  closeSaveBtn = el("close-save");
  /** @type {HTMLDivElement} */
  stoppedOverlay = el("stopped-overlay");
  /** @type {HTMLDivElement} */
  helpOverlay = el("help-overlay");
  /** @type {HTMLButtonElement} */
  helpClose = el("help-close");
  /** @type {HTMLDivElement} */
  helpBody = el("help-body");
  /** @type {HTMLDivElement} */
  sceneOverlay = el("scene-overlay");
  /** @type {HTMLDivElement} */
  sceneHead = el("scene-head");
  /** @type {HTMLButtonElement} */
  sceneClose = el("scene-close");
  /** @type {HTMLCanvasElement} */
  sceneCanvas = el("scene-canvas");
  /** @type {HTMLInputElement} */
  sceneAxesCheck = el("scene-axes");
  /** @type {HTMLInputElement} */
  sceneCamerasCheck = el("scene-cameras");
  /** @type {HTMLInputElement} */
  scenePoseCheck = el("scene-pose");
  /** @type {HTMLLabelElement} */
  sceneNmfWrap = el("scene-nmf-wrap");
  /** @type {HTMLInputElement} */
  sceneNmfCheck = el("scene-nmf");
  /** @type {HTMLLabelElement} */
  sceneMeshWrap = el("scene-mesh-wrap");
  /** @type {HTMLInputElement} */
  sceneMeshCheck = el("scene-mesh");
  // True once the shared GL renderer holds the current frame's posed mesh verts (so
  // the 3D view can render the mesh even when the 2D mesh overlay is off).
  sceneMeshReady = false;
  sceneMeshTimer = 0;

  async init() {
    this.meta = await fetchMeta();
    this.dirty = this.meta.dirty;
    // Grid is the default (`layout` is initialised to it); focus stays a click / f away.
    this.bindings = this.buildBindings();
    this.buildControls();
    this.buildViews();
    this.relayout();
    this.socket = new EditSocket((p) => this.applyPoints(p, true));
    await this.goToFrame(0);
    this.updateSelected();
    this.updateDirty();
    await this.refreshCorrected(); // populate the list (any corrections loaded from disk)
    // Closing instantly when there is nothing to lose, prompting otherwise: the
    // browser shows its generic "leave site?" dialog only while edits are unsaved.
    window.addEventListener("beforeunload", (e) => {
      if (this.dirty && !this.closing) {
        e.preventDefault();
        e.returnValue = "";
      }
    });
    window.addEventListener("keydown", (e) => this.onKey(e));
  }

  // -- construction -----------------------------------------------------------

  buildControls() {
    const last = Math.max(0, this.meta.n_frames - 1);
    for (const input of [this.slider, this.number]) {
      input.min = "0";
      input.max = String(last);
      input.value = "0";
    }
    this.totalEl.textContent = `/ ${last}`;
    this.slider.addEventListener("input", () => this.goToFrame(Number(this.slider.value)));
    this.number.addEventListener("change", () => this.goToFrame(Number(this.number.value)));

    // A single camera has nothing to focus, so the layout choice is hidden.
    this.layoutWrap.style.display = this.meta.n_views > 1 ? "" : "none";
    this.layoutSwitch = segmented(
      [["Grid", "grid"], ["Focus", "focus"]],
      (v) => this.setLayout(/** @type {Layout} */ (v))
    );
    this.layoutSwitch.set(this.layout);
    el("layout-switch").append(this.layoutSwitch.root);

    // The selected joint's per-view source/state: click a chip to set it. "Ground
    // truth" is an authored GT pixel; "Occluded" drops the view from the 3D solve;
    // "Predicted" is neither (following the detector / reprojection).
    this.stateSwitch = segmented(
      [["Predicted", "normal"], ["Ground truth", "fixed"], ["Occluded", "invisible"]],
      (v) => this.setSelectedState(v)
    );
    el("point-status-states").append(this.stateSwitch.root);

    this.skeletonCheck.addEventListener("change", () => this.applySkeleton());
    this.labelsCheck.addEventListener("change", () => this.applyLabels());
    // The latent overlay is the reprojected 3D estimate -- meaningless without 3D.
    this.latentWrap.style.display = this.meta.has_3d ? "" : "none";
    this.latentCheck.addEventListener("change", () => this.applyLatent());
    // The NMF overlay is the fitted inverse-kinematics model -- only when present.
    this.nmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.nmfCheck.addEventListener("change", () => this.applyNmf());
    // The NMF mesh overlay (rendered on the client GPU) -- only when a fitted model
    // is present. The head/abdomen size is estimated from the data by the IK stage
    // (no operator knob), so the overlay just follows the model.
    this.meshWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.meshCheck.addEventListener("change", () => this.applyMesh());

    this.resetViewBtn.addEventListener("click", () => this.resetSelectedView());
    this.resetAllBtn.addEventListener("click", () => this.resetSelectedAll());
    this.resetFrameBtn.addEventListener("click", () => this.resetFrame());
    this.confirmPointBtn.addEventListener("click", () => this.confirmSelectedJoint());
    this.confirmFrameBtn.addEventListener("click", () => this.confirmFrame());
    this.undoBtn.addEventListener("click", () => this.undo());
    this.redoBtn.addEventListener("click", () => this.redo());
    this.keypointsBtn.addEventListener("click", () => this.openKeypoints());
    this.framesToggleBtn.addEventListener("click", () => this.toggleFrames());
    this.framesCollapseBtn.addEventListener("click", () => this.closeFrames());
    this.framesPrevBtn.addEventListener("click", () => this.jumpCorrected(-1));
    this.framesNextBtn.addEventListener("click", () => this.jumpCorrected(1));
    this.camerasBtn.addEventListener("click", () => this.toggleScene());
    this.helpBtn.addEventListener("click", () => this.toggleHelp());
    this.helpClose.addEventListener("click", () => this.closeHelp());
    this.sceneClose.addEventListener("click", () => this.closeScene());
    this.initSceneDrag();
    // The 3D-view layer toggles; the NMF layers only exist when a model was fit.
    this.sceneNmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.sceneMeshWrap.style.display = this.meta.has_nmf ? "" : "none";
    for (const c of [this.sceneAxesCheck, this.sceneCamerasCheck, this.scenePoseCheck, this.sceneNmfCheck, this.sceneMeshCheck]) {
      c.addEventListener("change", () => this.applySceneToggles());
    }
    // Click outside the dialog body (on the dim backdrop) closes the modal dialogs. The
    // 3D view is a non-modal floating panel (no backdrop), so it closes only via its ✕,
    // the `c` toggle, or Esc.
    this.helpOverlay.addEventListener("click", (e) => {
      if (e.target === this.helpOverlay) this.closeHelp();
    });
    this.saveBtn.addEventListener("click", () => this.save());
    this.closeBtn.addEventListener("click", () => this.requestClose());
    this.closeCancelBtn.addEventListener("click", () => this.closeCloseConfirm());
    this.closeDiscardBtn.addEventListener("click", () => this.shutdown());
    this.closeSaveBtn.addEventListener("click", () => this.saveAndShutdown());
    this.closeOverlay.addEventListener("click", (e) => {
      if (e.target === this.closeOverlay) this.closeCloseConfirm();
    });
  }

  buildViews() {
    const cols = Math.max(1, Math.ceil(Math.sqrt(this.meta.n_views)));
    this.viewsEl.style.setProperty("--cols", String(cols));
    /** @type {import("./poseView.js").PoseViewCallbacks} */
    const cb = {
      onDragging: (v, p, x, y) => this.onDragging(v, p, x, y),
      onDragged: (v, p, x, y, wasInvisible) => this.onDragged(v, p, x, y, wasInvisible),
      onToggleFixed: (v, p) => this.onToggleFixed(v, p),
      onSelect: (v, p) => this.onSelect(v, p),
      onHover: (p) => this.onHover(p),
    };
    this.meta.camera_names.forEach((name, v) => {
      const cell = document.createElement("div");
      cell.className = "cell";
      const label = document.createElement("div");
      label.className = "cell-label";
      label.textContent = name;
      const canvas = document.createElement("canvas");
      cell.append(label, canvas);
      // In the focus layout a thumbnail is non-editable; clicking it promotes it
      // to the large editable view.
      cell.addEventListener("click", () => {
        if (this.layout === "focus" && v !== this.focused) this.setFocused(v);
      });
      this.cells.push(cell);

      const view = new PoseView(v, canvas, cb);
      view.setSkeleton(this.meta.bones, this.meta.point_colors);
      view.setPointNames(this.meta.point_names);
      const size = this.meta.image_sizes[name];
      if (size) view.setImageSize(size[0], size[1]);
      this.views.push(view);
    });
  }

  // -- layout -----------------------------------------------------------------

  /** @param {Layout} layout */
  setLayout(layout) {
    this.layout = layout;
    this.layoutSwitch.set(layout);
    this.relayout();
  }

  /** @param {number} view */
  setFocused(view) {
    this.focused = view;
    this.relayout();
  }

  // Reparent the persistent cells into the stage/strip for the current layout.
  // Moving a cell resizes its canvas, so each PoseView re-fits via its
  // ResizeObserver -- no points need re-fetching.
  relayout() {
    this.viewsEl.classList.toggle("layout-focus", this.layout === "focus");
    this.viewsEl.classList.toggle("layout-grid", this.layout === "grid");
    if (this.layout === "grid") {
      this.stageEl.replaceChildren(...this.cells);
      this.stripEl.replaceChildren();
    } else {
      this.stageEl.replaceChildren(this.cells[this.focused]);
      this.stripEl.replaceChildren(...this.cells.filter((_, v) => v !== this.focused));
    }
    this.cells.forEach((cell, v) => {
      cell.classList.toggle("is-focused", this.layout === "focus" && v === this.focused);
    });
    this.updateViewRoles();
  }

  // Only the large view(s) are editable and zoomable: every view in the grid, or
  // just the focused view in the focus layout. Thumbnails stay live but read-only
  // and unzoomed (so they always show the whole frame).
  updateViewRoles() {
    this.views.forEach((view, v) => {
      const large = this.layout === "grid" || v === this.focused;
      view.setEditable(large);
      view.setZoomable(large);
    });
  }

  // -- frame navigation -------------------------------------------------------

  /** @param {number} t */
  async goToFrame(t) {
    const last = Math.max(0, this.meta.n_frames - 1);
    t = Math.max(0, Math.min(Math.round(t), last));
    this.frame = t;
    this.slider.value = String(t);
    this.number.value = String(t);
    this.updateActiveFrameRow();
    this.meta.camera_names.forEach((name, v) => {
      this.views[v].loadFrame(frameUrl(name, t));
    });
    this.scheduleMeshRefresh();
    await this.refreshPoints();
    if (this.sceneOpen) {
      this.refreshScenePoints(); // snappy skeleton scrub
      this.scheduleSceneMesh(); // mesh catches up once the scrub settles
    }
  }

  async refreshPoints() {
    this.applyPoints(await fetchPoints(this.frame, this.mode));
  }

  /** Whether a drag re-solves the 3D point live (only meaningful when the result has 3D). */
  get resolves3d() {
    return this.meta.has_3d;
  }

  /**
   * @param {PointsPayload} p
   * @param {boolean} [fromEdit]  true for a WebSocket edit reply (which may change
   *   the corrected-frames list); false for a plain frame fetch on navigation.
   */
  applyPoints(p, fromEdit = false) {
    // undo/redo can revert an edit on a *different* frame than the one being viewed;
    // navigate there (which re-fetches the now-updated frame) before applying.
    if (fromEdit && p.goto != null && p.goto !== this.frame) {
      this.goToFrame(p.goto);
      return;
    }
    if (p.frame !== this.frame) return; // a stale reply after a fast scrub
    // Drop a superseded edit's reply: a live 3D drag streams many edits, and an
    // in-flight mid-drag re-solve that lands after release (or after a newer edit)
    // must not repaint the joint to a stale position (the snap-back). Replies echo
    // the sending edit's seq; only the latest edit's reply (seq === editSeq) wins.
    // Plain frame fetches carry no seq and always apply.
    if (fromEdit && p.seq !== this.editSeq) return;
    // The per-view masks drive the source-styled markers (ground truth / occluded) and
    // the status widget; they are meaningful whether or not the result carries 3D.
    this.fixedMask = p.fixed;
    this.invisibleMask = p.invisible;
    // `nmf` is omitted on mid-drag replies (the server skips the per-frame re-fit);
    // when absent, leave each view's model overlay as-is instead of clearing it.
    const hasNmf = "nmf" in p;
    this.views.forEach((view, v) => {
      view.setFrameData({
        points: p.points[v],
        fixed: p.fixed[v],
        invisible: p.invisible[v],
        conf: "conf" in p && p.conf ? p.conf[v] : undefined,
        latent: p.proj ? p.proj[v] : null,
        nmf: hasNmf ? (p.nmf ? p.nmf[v] : null) : undefined,
      });
    });
    // The NMF mesh follows the (re-fit) latent skeleton: refresh it after an edit
    // settles, coalescing a live drag's many replies into one GPU render.
    this.scheduleMeshRefresh();
    this.dirty = p.dirty;
    if (p.can_undo != null) this.undoBtn.disabled = !p.can_undo;
    if (p.can_redo != null) this.redoBtn.disabled = !p.can_redo;
    this.updateDirty();
    this.updateStatusWidget();
    // An edit may have added or cleared this frame's labels; refresh the list
    // (debounced, so a live drag's stream of replies coalesces into one fetch).
    if (fromEdit) this.scheduleCorrectedRefresh();
  }

  // -- display toggles --------------------------------------------------------

  applySkeleton() {
    const visible = this.skeletonCheck.checked;
    this.views.forEach((view) => view.setOverlayVisible(visible));
  }

  applyLabels() {
    const visible = this.labelsCheck.checked;
    this.views.forEach((view) => view.setLabelsVisible(visible));
  }

  applyLatent() {
    const visible = this.latentCheck.checked;
    this.views.forEach((view) => view.setLatentVisible(visible));
  }

  applyNmf() {
    const visible = this.nmfCheck.checked;
    this.views.forEach((view) => view.setNmfVisible(visible));
  }

  applyMesh() {
    const visible = this.meshCheck.checked;
    this.views.forEach((view) => view.setMeshVisible(visible));
    if (visible) this.refreshMesh();
  }

  /** Ensure the static mesh topology + colors are loaded into the GPU (once). */
  async ensureMeshAsset() {
    if (!this.meta.has_nmf) return false;
    if (!this.meshGL) this.meshGL = new MeshGL();
    if (!this.meshGL.ok) return false;
    if (!this.meshAssetLoaded) {
      try {
        this.meshGL.loadAsset(await fetchNmfAsset());
        this.meshAssetLoaded = true;
      } catch (e) {
        console.error("could not load the NMF mesh asset", e);
        return false;
      }
    }
    return true;
  }

  /** Re-fetch the posed vertices for the current frame and render every view. */
  async refreshMesh() {
    if (!this.meshCheck.checked) return;
    if (!(await this.ensureMeshAsset())) return;
    const req = ++this.meshReq;
    const frame = this.frame;
    let buf;
    try {
      buf = await fetchNmfVerts(frame);
    } catch (e) {
      return; // overlay simply stays where it was
    }
    if (req !== this.meshReq || frame !== this.frame) return; // superseded
    const gl = this.meshGL;
    const nV = gl.nVerts;
    const nF = gl.faces.length / 3;
    // Payload: verts (nV*3 f32), smooth normals (nV*3 f32), valid faces (nF u8).
    gl.setVerts(
      new Float32Array(buf, 0, nV * 3),
      new Float32Array(buf, nV * 12, nV * 3),
      new Uint8Array(buf, nV * 24, nF),
    );
    // Render at the on-screen device-pixel size (capped) rather than the footage
    // size, so the overlay is as crisp as the docs model viewer instead of an
    // upscaled footage-resolution image.
    const ss = this.meshSupersample();
    const cams = this.meta.cameras_proj || [];
    this.views.forEach((view, v) => view.captureMesh(gl.render(cams[v], ss)));
  }

  /**
   * How many GL pixels to render per footage pixel, so the overlay matches the
   * sharpest view it is shown in (the big editing view drives this). Capped to keep
   * the offscreen canvas bounded.
   */
  meshSupersample() {
    const dpr = window.devicePixelRatio || 1;
    let best = 1;
    for (const view of this.views) {
      if (view.scale) best = Math.max(best, view.scale * dpr);
    }
    return Math.min(4, Math.max(1, best));
  }

  /** Coalesce rapid mesh refreshes (a scrub or a live drag) into one render. */
  scheduleMeshRefresh() {
    if (!this.meshCheck.checked) return;
    clearTimeout(this.meshTimer);
    this.meshTimer = setTimeout(() => this.refreshMesh(), 90);
  }

  /** @param {HTMLInputElement} check  flip a checkbox from a shortcut, then apply */
  toggleCheck(check, apply) {
    check.checked = !check.checked;
    apply();
  }

  // -- hover / selection ------------------------------------------------------

  /** @param {number | null} point  the hovered joint, emphasized in every view */
  onHover(point) {
    this.views.forEach((view) => view.setHighlight(point));
  }

  /**
   * @param {number} view
   * @param {number} point
   */
  onSelect(view, point) {
    this.selectedView = view;
    this.selectedPoint = point;
    this.updateSelected();
  }

  // Mark the selected joint (a cyan ring in its own view only) and enable the
  // per-point Reset buttons once there is a point to reset. (The whole-frame
  // reset needs no selection, so it stays enabled.)
  updateSelected() {
    this.views.forEach((view, v) => {
      view.setSelected(v === this.selectedView ? this.selectedPoint : null);
    });
    const has = this.selectedPoint !== null;
    this.resetViewBtn.disabled = !has;
    this.resetAllBtn.disabled = !has;
    this.updateStatusWidget();
  }

  // -- point status widget ----------------------------------------------------

  /** @returns {"normal" | "fixed" | "invisible"} the selected joint's state in its view */
  selectedState() {
    const v = this.selectedView;
    const p = this.selectedPoint;
    if (p === null) return "normal";
    if (this.invisibleMask && this.invisibleMask[v][p]) return "invisible";
    if (this.fixedMask && this.fixedMask[v][p]) return "fixed";
    return "normal";
  }

  // Show the selected joint's name, its view, and its per-view source -- only when the
  // result has 3D (occluding a view needs a 3D solve to drop it from). The widget stays
  // visible with its state chips disabled (and a "—" placeholder) until a joint is
  // selected, so the control reads as present-but-unavailable, like the Reset buttons.
  updateStatusWidget() {
    const p = this.selectedPoint;
    const show = this.meta.has_3d;
    this.pointStatus.hidden = !show;
    if (!show) return;
    const has = p !== null;
    this.stateSwitch.setDisabled(!has);
    if (!has) {
      this.pointStatusName.textContent = "—";
      this.stateSwitch.set(""); // no joint -> no active chip
      return;
    }
    const name = this.meta.point_names[p] ?? `#${p}`;
    const cam = this.meta.camera_names[this.selectedView] ?? `view ${this.selectedView}`;
    this.pointStatusName.textContent = `${name} · ${cam}`;
    this.stateSwitch.set(this.selectedState());
  }

  // Click a state chip to set the selected joint to that state. The states are
  // mutually exclusive, so one toggle takes it anywhere: toggling fixed/obscured
  // sets it (clearing the other), and "normal" clears whichever flag is set.
  /** @param {string} target  "normal" | "fixed" | "invisible" */
  setSelectedState(target) {
    if (!this.meta.has_3d || this.selectedPoint === null) return;
    const current = this.selectedState();
    if (target === current) return;
    const v = this.selectedView;
    const p = this.selectedPoint;
    if (target === "fixed") this.onToggleFixed(v, p);
    else if (target === "invisible") this.onToggleInvisible(v, p);
    else if (current === "fixed") this.onToggleFixed(v, p); // -> normal
    else if (current === "invisible") this.onToggleInvisible(v, p); // -> normal
  }

  // -- edit routing -----------------------------------------------------------

  // Send an edit over the socket, stamped with a monotonic seq the server echoes
  // back so applyPoints can drop a superseded reply (a mid-drag re-solve that lands
  // after release, or after a newer edit) instead of repainting a stale position.
  /** @param {import("./types.js").EditMessage} msg */
  sendEdit(msg) {
    this.socket.send({ ...msg, seq: ++this.editSeq });
  }

  /**
   * @param {number} view
   * @param {number} point
   * @param {number} x
   * @param {number} y
   */
  onDragging(view, point, x, y) {
    // A drag authors ground truth at the drop; when the result has 3D it also re-solves
    // the 3D live (streaming one edit per animation frame) so the reprojection follows.
    // With no 3D there is nothing to re-solve mid-drag -- the commit lands on release.
    if (this.resolves3d) {
      this.sendEdit({ type: "edit_3d", view, point, x, y, frame: this.frame, fix: false, mode: this.mode });
    }
  }

  /**
   * @param {number} view
   * @param {number} point
   * @param {number} x
   * @param {number} y
   * @param {boolean} [wasInvisible]  whether the grabbed joint was obscured (now un-obscured by the drag)
   */
  onDragged(view, point, x, y, wasInvisible = false) {
    if (!this.resolves3d) {
      // No 3D to re-solve: the drop is simply this view's ground-truth pixel.
      this.sendEdit({ type: "edit_2d", view, point, x, y, frame: this.frame, mode: this.mode });
      return;
    }
    // Releasing a drag pins the dragged view at the drop pixel (a finalized
    // constraint) so the placed point stays put -- including a previously occluded
    // view: dragging it in is the operator asserting where the point is, so it is
    // both un-occluded (server-side) and finalized here rather than left to drift
    // back to the reprojection.
    this.sendEdit({ type: "edit_3d", view, point, x, y, frame: this.frame, fix: true, mode: this.mode });
  }

  /**
   * @param {number} view
   * @param {number} point
   */
  onToggleFixed(view, point) {
    if (!this.meta.has_3d) return;
    this.sendEdit({ type: "toggle_fixed", view, point, frame: this.frame, mode: this.mode });
  }

  /**
   * @param {number} view
   * @param {number} point
   */
  onToggleInvisible(view, point) {
    if (!this.meta.has_3d) return;
    this.sendEdit({ type: "toggle_invisible", view, point, frame: this.frame, mode: this.mode });
  }

  // Keyboard shortcuts (l / o): act on the last-selected joint in its view.
  toggleSelectedFixed() {
    if (this.selectedPoint !== null) this.onToggleFixed(this.selectedView, this.selectedPoint);
  }

  toggleSelectedInvisible() {
    if (this.selectedPoint !== null) this.onToggleInvisible(this.selectedView, this.selectedPoint);
  }

  // Revert the last-selected joint in just the view it was selected in.
  resetSelectedView() {
    if (this.selectedPoint === null) return;
    this.sendEdit({
      type: "reset_point_view",
      view: this.selectedView,
      point: this.selectedPoint,
      frame: this.frame,
      mode: this.mode,
    });
  }

  // Revert the last-selected joint across every view (and its 3D point).
  resetSelectedAll() {
    if (this.selectedPoint === null) return;
    this.sendEdit({ type: "reset_point", point: this.selectedPoint, frame: this.frame, mode: this.mode });
  }

  // Revert every joint in the current frame across all views (and their 3D
  // points) -- a clean slate for the frame, independent of any selection.
  resetFrame() {
    this.sendEdit({ type: "reset_frame", frame: this.frame, mode: this.mode });
  }

  // -- confirm (promote suggestions to ground truth) --------------------------

  // Confirm the selected joint's suggested position as GT in every view.
  confirmSelectedJoint() {
    if (this.selectedPoint === null) return;
    const point = this.selectedPoint;
    const targets = this.views.map((_, v) => [v, point]);
    this.sendEdit({ type: "confirm", targets, sources: "all", frame: this.frame, mode: this.mode });
  }

  // Confirm every suggested point in the current frame (all views) as GT.
  confirmFrame() {
    /** @type {[number, number][]} */
    const targets = [];
    for (let v = 0; v < this.meta.n_views; v++) {
      for (let p = 0; p < this.meta.n_points; p++) targets.push([v, p]);
    }
    this.sendEdit({ type: "confirm", targets, sources: "all", frame: this.frame, mode: this.mode });
  }

  // -- undo / redo ------------------------------------------------------------

  undo() {
    this.sendEdit({ type: "undo", frame: this.frame, mode: this.mode });
  }

  redo() {
    this.sendEdit({ type: "redo", frame: this.frame, mode: this.mode });
  }

  async save() {
    const r = await saveCorrections();
    this.dirty = r.dirty;
    this.updateDirty();
    this.statusEl.textContent = "saved";
    setTimeout(() => (this.statusEl.textContent = ""), 3000);
  }

  updateDirty() {
    document.title = `deeperfly gui — ${this.meta.results_path}${this.dirty ? " *" : ""}`;
    this.saveBtn.disabled = !this.dirty;
  }

  // -- corrected-frames list --------------------------------------------------

  // Pull the frames carrying corrections and repaint the side panel. Called on load
  // (any sidecar loaded from disk) and, debounced, after each edit settles -- so the
  // list tracks every drag, obscure, and reset live.
  async refreshCorrected() {
    let frames;
    try {
      frames = (await fetchCorrected()).frames;
    } catch (_) {
      return; // a transient failure just leaves the list as it was
    }
    this.correctedFrames = frames;
    this.renderFrameList();
  }

  // Coalesce rapid refreshes (a live 3D drag fires a stream of edits) into one fetch.
  scheduleCorrectedRefresh() {
    clearTimeout(this.correctedTimer);
    this.correctedTimer = setTimeout(() => this.refreshCorrected(), 150);
  }

  // Rebuild the table from the current list, update the count chip + empty state,
  // and keep the current frame highlighted. Each row jumps to its frame on click.
  renderFrameList() {
    const n = this.correctedFrames.length;
    this.framesCountEl.textContent = String(n);
    this.framesCountEl.classList.toggle("is-zero", n === 0);
    this.framesEmptyEl.hidden = n > 0;
    this.frameRows.clear();
    const rows = this.correctedFrames.map(({ frame, count }) => {
      const tr = document.createElement("tr");
      const fcell = document.createElement("td");
      fcell.textContent = String(frame);
      const ccell = document.createElement("td");
      ccell.textContent = String(count);
      tr.append(fcell, ccell);
      tr.addEventListener("click", () => this.goToFrame(frame));
      this.frameRows.set(frame, tr);
      return tr;
    });
    this.framesTbody.replaceChildren(...rows);
    this.updateActiveFrameRow();
  }

  // Highlight the row for the current frame (when it is a corrected one) and, while
  // the panel is open, scroll it into view -- so scrubbing keeps the list in sync.
  updateActiveFrameRow() {
    this.frameRows.forEach((tr, frame) => {
      const active = frame === this.frame;
      tr.classList.toggle("is-current", active);
      if (active && this.framesOpen) tr.scrollIntoView({ block: "nearest" });
    });
  }

  openFrames() {
    this.sidebarEl.hidden = false;
    this.framesOpen = true;
    this.updateActiveFrameRow(); // scroll the current frame into view now it is shown
  }

  closeFrames() {
    this.sidebarEl.hidden = true;
    this.framesOpen = false;
  }

  toggleFrames() {
    if (this.framesOpen) this.closeFrames();
    else this.openFrames();
  }

  // Step to the previous / next corrected frame (wrapping at the ends), so the
  // operator can walk their corrections without hunting on the scrubber.
  /** @param {number} dir  -1 for the previous corrected frame, +1 for the next */
  jumpCorrected(dir) {
    const frames = this.correctedFrames.map((f) => f.frame);
    if (frames.length === 0) return;
    let target;
    if (dir > 0) {
      target = frames.find((f) => f > this.frame) ?? frames[0]; // wrap to the first
    } else {
      const earlier = frames.filter((f) => f < this.frame);
      target = earlier.length ? earlier[earlier.length - 1] : frames[frames.length - 1];
    }
    this.goToFrame(target);
  }

  // -- close / shutdown -------------------------------------------------------

  // The Close button: stop the server outright when nothing is at stake, else
  // ask whether to save the pending corrections first.
  requestClose() {
    if (this.dirty) this.openCloseConfirm();
    else this.shutdown();
  }

  openCloseConfirm() {
    this.closeOverlay.hidden = false;
    this.closeConfirmOpen = true;
  }

  closeCloseConfirm() {
    this.closeOverlay.hidden = true;
    this.closeConfirmOpen = false;
  }

  // "Save & close": only stop the server once the save actually lands, so a
  // failed write leaves the editor open with the corrections intact.
  async saveAndShutdown() {
    try {
      await this.save();
    } catch (_) {
      this.closeCloseConfirm();
      this.statusEl.textContent = "save failed";
      return;
    }
    await this.shutdown();
  }

  // Stop the server and replace the editor with a "stopped" notice. The socket is
  // closed first so uvicorn's graceful shutdown isn't held up by the live WS, and
  // the request error (the server may drop the connection mid-reply) is ignored.
  async shutdown() {
    this.closing = true; // the close is deliberate -- don't nag on unload
    this.closeCloseConfirm();
    this.socket?.ws.close();
    await shutdownServer();
    this.stoppedOverlay.hidden = false;
  }

  // -- keypoint reference -----------------------------------------------------

  // Open the NeuroMechFly keypoint-locations reference (the docs viewer) in a new
  // tab. Linking out keeps the editor lean: that page's model/WASM assets load only
  // when the operator opens it, never with the editor.
  openKeypoints() {
    window.open(KEYPOINTS_DOC_URL, "_blank", "noopener");
  }

  // -- 3D scene view ----------------------------------------------------------

  // Let the operator drag the floating 3D panel by its title bar. The header's own
  // controls (close button, layer checkboxes) keep working: a press on one of them
  // starts no drag. The panel is clamped to stay on screen.
  initSceneDrag() {
    /** @type {{ x: number, y: number } | null} */
    let grab = null;
    this.sceneHead.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      if (/** @type {HTMLElement} */ (e.target).closest("button, input, label")) return;
      const r = this.sceneOverlay.getBoundingClientRect();
      grab = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.sceneHead.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    this.sceneHead.addEventListener("pointermove", (e) => {
      if (grab) this.placeScene(e.clientX - grab.x, e.clientY - grab.y);
    });
    const end = (/** @type {PointerEvent} */ e) => {
      grab = null;
      if (this.sceneHead.hasPointerCapture(e.pointerId)) this.sceneHead.releasePointerCapture(e.pointerId);
    };
    this.sceneHead.addEventListener("pointerup", end);
    this.sceneHead.addEventListener("pointercancel", end);
  }

  // Move the floating panel's top-left to (left, top), kept within the viewport so it
  // can't be dragged out of reach. Switches off any CSS edge anchoring first.
  /** @param {number} left @param {number} top */
  placeScene(left, top) {
    const maxL = Math.max(0, window.innerWidth - this.sceneOverlay.offsetWidth);
    const maxT = Math.max(0, window.innerHeight - this.sceneOverlay.offsetHeight);
    this.sceneOverlay.style.left = `${Math.max(0, Math.min(maxL, left))}px`;
    this.sceneOverlay.style.top = `${Math.max(0, Math.min(maxT, top))}px`;
    this.sceneOverlay.style.right = "auto";
    this.sceneOverlay.style.bottom = "auto";
  }

  ensureScene() {
    if (this.scene) return this.scene;
    this.scene = new Scene3D(this.sceneCanvas);
    this.scene.setCameras(this.meta.cameras_3d);
    this.scene.setSkeleton(this.meta.bones, this.meta.point_colors);
    // The mesh is drawn by the shared WebGL renderer; the scene only hands it the
    // orbit camera. Null until the current frame's posed verts are uploaded.
    this.scene.setMeshRenderer((cam, ss) =>
      this.sceneMeshReady && this.meshGL ? this.meshGL.render(cam, ss) : null
    );
    this.applySceneToggles();
    return this.scene;
  }

  applySceneToggles() {
    this.scene?.setVisibility({
      axes: this.sceneAxesCheck.checked,
      cameras: this.sceneCamerasCheck.checked,
      pose: this.scenePoseCheck.checked,
      nmf: this.sceneNmfCheck.checked,
      mesh: this.sceneMeshCheck.checked,
    });
  }

  // Upload the current frame's posed mesh verts to the shared GL renderer, so the
  // 3D view can render the mesh independently of the 2D mesh-overlay toggle.
  async ensureSceneMesh() {
    if (!this.meta.has_nmf || !(await this.ensureMeshAsset())) return;
    const frame = this.frame;
    let buf;
    try {
      buf = await fetchNmfVerts(frame);
    } catch (e) {
      return; // the mesh layer just stays where it was (or empty)
    }
    if (frame !== this.frame) return; // superseded by a newer frame
    const gl = this.meshGL;
    const nV = gl.nVerts;
    const nF = gl.faces.length / 3;
    gl.setVerts(
      new Float32Array(buf, 0, nV * 3),
      new Float32Array(buf, nV * 12, nV * 3),
      new Uint8Array(buf, nV * 24, nF),
    );
    this.sceneMeshReady = true;
    if (this.sceneOpen) this.scene?.draw();
  }

  // Pull the frame's 3D pose (the cheap part) and repaint the skeletons at once.
  async refreshScenePoints() {
    if (!this.scene) return;
    const s = await fetchScene(this.frame);
    if (s.frame !== this.frame) return; // a stale reply after a fast scrub
    this.scene.setPoints3d(s.points3d);
    this.scene.setNmf3d(s.nmf3d ?? null);
  }

  // Coalesce the 3D view's posed-mesh refreshes (the heavy part) during a scrub.
  scheduleSceneMesh() {
    if (!this.sceneOpen) return;
    clearTimeout(this.sceneMeshTimer);
    this.sceneMeshTimer = setTimeout(() => this.ensureSceneMesh(), 90);
  }

  async refreshScene() {
    await this.refreshScenePoints();
    await this.ensureSceneMesh();
  }

  async openScene() {
    const scene = this.ensureScene();
    this.sceneOverlay.hidden = false;
    this.sceneOpen = true;
    scene.resize(); // the canvas only has a size now that the modal is visible
    await this.refreshScene();
    scene.resetView(); // frame the scene once the pose + mesh are loaded
  }

  closeScene() {
    this.sceneOverlay.hidden = true;
    this.sceneOpen = false;
  }

  toggleScene() {
    if (this.sceneOpen) this.closeScene();
    else this.openScene();
  }

  // -- keyboard help ----------------------------------------------------------

  /** @returns {Binding[]} the active shortcut bindings (some depend on the result) */
  buildBindings() {
    const has3d = this.meta.has_3d;
    const multi = this.meta.n_views > 1;
    /** @type {Binding[]} */
    const b = [
      { key: "ArrowLeft", label: "← / →", desc: "Previous / next frame (Shift: ±10)", run: (e) => this.step(e.shiftKey ? -10 : -1) },
      { key: "ArrowRight", hidden: true, label: "→", desc: "", run: (e) => this.step(e.shiftKey ? 10 : 1) },
    ];
    if (multi) {
      b.push({ key: "g", label: "g", desc: "Grid layout", run: () => this.setLayout("grid") });
      b.push({ key: "f", label: "f", desc: "Focus layout", run: () => this.setLayout("focus") });
      b.push({ key: "[", label: "[ / ]", desc: "Focus the previous / next camera", run: () => this.cycleFocus(-1) });
      b.push({ key: "]", hidden: true, label: "]", desc: "", run: () => this.cycleFocus(1) });
    }
    b.push({ key: "s", label: "s", desc: "Toggle skeleton", run: () => this.toggleCheck(this.skeletonCheck, () => this.applySkeleton()) });
    b.push({ key: "n", label: "n", desc: "Toggle keypoint labels", run: () => this.toggleCheck(this.labelsCheck, () => this.applyLabels()) });
    if (has3d) {
      b.push({ key: "p", label: "p", desc: "Toggle 3D estimate overlay", run: () => this.toggleCheck(this.latentCheck, () => this.applyLatent()) });
    }
    if (this.meta.has_nmf) {
      b.push({ key: "m", label: "m", desc: "Toggle NMF skeleton overlay", run: () => this.toggleCheck(this.nmfCheck, () => this.applyNmf()) });
      b.push({ key: "M", label: "Shift+M", desc: "Toggle NMF mesh overlay", run: () => this.toggleCheck(this.meshCheck, () => this.applyMesh()) });
    }
    if (has3d) {
      b.push({ key: "l", label: "l", desc: "Confirm / clear ground truth for the selected point (this view)", run: () => this.toggleSelectedFixed() });
      b.push({ key: "o", label: "o", desc: "Occlude / reveal the selected point (this view)", run: () => this.toggleSelectedInvisible() });
    }
    b.push({ key: "Enter", label: "Enter", desc: "Confirm the selected point as ground truth (all views)", run: () => this.confirmSelectedJoint() });
    b.push({ key: "a", label: "a", desc: "Confirm every suggested point in this frame", run: () => this.confirmFrame() });
    b.push({ key: "r", label: "r", desc: "Discard the selected point's label in its view", run: () => this.resetSelectedView() });
    b.push({ key: "R", label: "Shift+R", desc: "Discard the selected point's label in all views", run: () => this.resetSelectedAll() });
    b.push({ key: "z", mod: true, label: "Ctrl/⌘+Z", desc: "Undo", run: () => this.undo() });
    b.push({ key: "y", mod: true, label: "Ctrl/⌘+Y", desc: "Redo", run: () => this.redo() });
    b.push({ key: "z", mod: true, shift: true, hidden: true, label: "Ctrl/⌘+Shift+Z", desc: "Redo", run: () => this.redo() });
    b.push({ key: "c", label: "c", desc: "Show / hide the 3D view", run: () => this.toggleScene() });
    b.push({ key: "j", label: "j", desc: "Show / hide the labelled-frames list", run: () => this.toggleFrames() });
    b.push({ key: "k", label: "k", desc: "Open the keypoint reference (docs, new tab)", run: () => this.openKeypoints() });
    b.push({ key: "s", mod: true, global: true, label: "Ctrl/⌘+S", desc: "Save labels", run: () => this.save() });
    b.push({ key: "?", label: "?", desc: "Toggle this help", run: () => this.toggleHelp() });
    return b;
  }

  buildHelp() {
    const rows = this.bindings
      .filter((b) => !b.hidden)
      .map((b) => `<tr><td class="key"><kbd>${b.label}</kbd></td><td>${b.desc}</td></tr>`);
    rows.push(`<tr><td class="key"><kbd>Esc</kbd></td><td>Close this dialog or the 3D view</td></tr>`);
    this.helpBody.innerHTML = `<table class="shortcuts"><tbody>${rows.join("")}</tbody></table>`;
  }

  openHelp() {
    if (!this.helpBuilt) {
      this.buildHelp();
      this.helpBuilt = true;
    }
    this.helpOverlay.hidden = false;
    this.helpOpen = true;
  }

  closeHelp() {
    this.helpOverlay.hidden = true;
    this.helpOpen = false;
  }

  toggleHelp() {
    if (this.helpOpen) this.closeHelp();
    else this.openHelp();
  }

  // -- keyboard dispatch ------------------------------------------------------

  /** @param {number} d */
  step(d) {
    this.goToFrame(this.frame + d);
  }

  /** @param {number} d  switch to the focus layout and move the focus by d cameras */
  cycleFocus(d) {
    if (this.meta.n_views < 2) return;
    if (this.layout !== "focus") this.setLayout("focus");
    const n = this.meta.n_views;
    this.setFocused((this.focused + d + n) % n);
  }

  /** @param {KeyboardEvent} e */
  onKey(e) {
    // Escape always backs out of an open dialog first.
    if (e.key === "Escape") {
      if (this.closeConfirmOpen) {
        this.closeCloseConfirm();
        e.preventDefault();
      } else if (this.helpOpen) {
        this.closeHelp();
        e.preventDefault();
      } else if (this.sceneOpen) {
        this.closeScene();
        e.preventDefault();
      }
      return;
    }
    const tag = (document.activeElement?.tagName ?? "").toLowerCase();
    const typing = tag === "input" || tag === "select" || tag === "textarea";
    for (const b of this.bindings) {
      if (!matches(e, b)) continue;
      if (typing && !b.global) return; // let the focused control keep the key
      e.preventDefault();
      b.run(e);
      return;
    }
  }
}

new App().init();
