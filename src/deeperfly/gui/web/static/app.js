// @ts-check
// The editor controller: lays out one PoseView per camera and routes edits to
// the server. There is one unified editing model -- a drag authors ground-truth 2D at
// the drop and, when the result carries 3D, re-solves the 3D point live and refreshes
// every view (pinning the dragged view on release). Each joint is drawn with a marker
// whose style tells its source apart: ground truth (lime ring over a filled disc),
// detector prediction (dark ring over a disc that fades with confidence), or a point
// derived by reprojecting the 3D (a hollow palette circle -- no observation in this
// view). Annotation is two steps: build a selection of (point, view) cells, then set its
// state from one combined control -- a chip picks Ground truth (Enter) or Projected (o)
// for the whole selection at once, and Reset (r) clears the labels back to the
// detector. That chip row doubles as the status readout: the active chip is the
// selection's shared state (detected / ground truth / projected; none lit when the cells
// disagree), and its name field shows the single cell's "point . camera" or, for several,
// the count. Marking a view Projected deletes its observation so it drops from the
// triangulation and then follows the reprojection; dragging it back in restores it.
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
// Hovering a joint emphasizes the same joint in every view and peeks at its name +
// state in the status widget; clicking one selects it (a cyan ring), and clicking the
// background clears the selection (panning does not). Ctrl/Cmd+click adds/removes a
// joint, double-click selects that keypoint in every view, a Shift+drag box rubber-bands
// a fresh selection while Ctrl/Cmd+drag adds to it, and `a` / `v` select all / the
// hovered view -- then a state chip (or Reset) applies to the whole selection at once.
//
// Display layers the operator toggles: the ground-truth layer (the editable one -- a drag moves
// or spawns GT), the raw-detection layer, "Combined" (a merge toggle: on, GT + detected draw as
// one skeleton; off, as two separate skeletons), per-joint name labels, and the "projected" 3D
// reprojection as its own overlay -- the reprojected skeleton (hollow rings joined by thick,
// dashed, semi-transparent edges; on by default), whose points double as spawn seeds for a
// ground-truth drag. A non-modal floating panel shows the 3D
// view -- the camera rig, the 3D pose, and the fitted NMF skeleton + mesh (see
// scene3d.js); it overlays the editor without blocking it (drag the title bar to move
// it, the corner to resize), so the main frame scrubber still steps the 3D pose
// through time. Almost everything has a keyboard shortcut; `?` opens a help list of them.
//
// This .js is the source -- there is no build step. VS Code type-checks it via
// `// @ts-check` and the JSDoc payload types in types.js.

import { EditSocket, fetchCorrected, fetchMeta, fetchNmfAsset, fetchNmfVerts, fetchPoints, fetchScene, fetchSuggestions, frameUrl, saveCorrections, shutdownServer } from "./api.js";
import { MeshGL } from "./meshGL.js";
import { PoseView } from "./poseView.js";
import { Scene3D } from "./scene3d.js";

/** @typedef {import("./types.js").Meta} Meta */
/** @typedef {import("./types.js").PointsPayload} PointsPayload */
/** @typedef {import("./types.js").CorrectedFrame} CorrectedFrame */
/** @typedef {import("./types.js").Suggestion} Suggestion */
/** @typedef {import("./types.js").SuggestionsPayload} SuggestionsPayload */
/** @typedef {"labeled" | "suggest"} SidebarTab */
/** @typedef {import("./types.js").EditMode} EditMode */
/** @typedef {"grid" | "focus"} Layout */
/** @typedef {{ key: string, mod?: boolean, shift?: boolean, global?: boolean, hidden?: boolean, group?: string, label: string, desc: string, run: (e: KeyboardEvent) => void }} Binding */
/** @typedef {{ root: HTMLDivElement, set: (value: string) => void, setDisabled: (disabled: boolean) => void, setDisabledValue: (value: string, disabled: boolean) => void }} Segmented */

// The published "NeuroMechFly keypoint locations" reference (the docs site). It is
// opened in a new tab on demand, so its heavy model + WASM assets are fetched only
// when the operator asks for it -- never on editor load.
const KEYPOINTS_DOC_URL = "https://nely-epfl.github.io/deeperfly/keypoints/viewer.html";

// localStorage keys for the reprojection-distance warning -- the one display preference that
// persists across sessions (an operator's tolerance for label-vs-reprojection disagreement).
const WARN_ON_KEY = "deeperfly.warn.enabled";
const WARN_PX_KEY = "deeperfly.warn.threshold";
const WARN_PX_MIN = 1;
const WARN_PX_MAX = 200;

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @returns {T}
 */
function el(id) {
  return /** @type {T} */ (document.getElementById(id));
}

/**
 * A table cell with a class and text -- the sidebar lists build a lot of these.
 * @param {string} className
 * @param {string} text
 * @returns {HTMLTableCellElement}
 */
function cell(className, text) {
  const td = document.createElement("td");
  td.className = className;
  td.textContent = text;
  return td;
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
    setDisabledValue: (value, disabled) => {
      const btn = buttons.get(value);
      if (btn) btn.disabled = disabled;
    },
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

// -- OS-aware modifier rendering --------------------------------------------
// Detect macOS so shortcut labels can show the platform's own glyphs (⌘⌥⇧⌃) rather
// than Ctrl/Alt/Shift. This is COSMETIC ONLY: `matches` collapses Ctrl and Cmd into a
// single test (e.ctrlKey || e.metaKey), so a misdetection can mislabel a chip but can
// never break a shortcut -- no binding branches on the platform. iPadOS reports itself
// as "Mac" in its desktop UA, so it is disambiguated by touch points (a real Mac
// reports none). `userAgentData.platform` (Chromium-only) returns "macOS"; the older
// `platform`/`userAgent` return "MacIntel"/"Macintosh" -- all match /mac/i.
const _nav = /** @type {any} */ (navigator);
const IS_MAC =
  /mac/i.test(_nav.userAgentData?.platform || _nav.platform || _nav.userAgent || "") &&
  (_nav.maxTouchPoints ?? 0) === 0;

// Glyphs per platform, and the left-to-right order a chord renders in. Apple's HIG order
// is Control, Option, Shift, Command (⌃⌥⇧⌘) glued with no separators; Windows/Linux read
// Ctrl+Alt+Shift+Key joined with "+". `mod` is the primary command key (Ctrl or ⌘).
const MOD_GLYPH = IS_MAC
  ? { mod: "⌘", alt: "⌥", shift: "⇧", ctrl: "⌃" }
  : { mod: "Ctrl", alt: "Alt", shift: "Shift", ctrl: "Ctrl" };
const MOD_ORDER = IS_MAC ? ["ctrl", "alt", "shift", "mod"] : ["mod", "ctrl", "alt", "shift"];

/**
 * Render a keyboard chord as a platform-correct label, e.g. `hint("Z", ["mod"])` gives
 * "⌘Z" on macOS and "Ctrl+Z" elsewhere. `key` is the already-display-ready base key.
 * @param {string} key
 * @param {("mod"|"alt"|"shift"|"ctrl")[]} [mods]
 * @returns {string}
 */
function hint(key, mods = []) {
  const parts = MOD_ORDER.filter((m) => mods.includes(/** @type {any} */ (m))).map(
    (m) => MOD_GLYPH[/** @type {"mod"|"alt"|"shift"|"ctrl"} */ (m)],
  );
  parts.push(key);
  return IS_MAC ? parts.join("") : parts.join("+");
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
  // The current selection: a set of (view, point) cells the state control
  // (the Detected / Ground truth / Projected chips + Reset) acts on, keyed
  // "view:point". `selAnchor` is the most-recently-added cell -- the single cell
  // whose name the widget shows -- and `activeView` is the camera the pointer is
  // over (the target of the `v` "select every point in this view" gesture).
  /** @type {Set<string>} */
  selection = new Set();
  /** @type {{ view: number, point: number } | null} */
  selAnchor = null;
  activeView = 0;
  // The joint the pointer is currently over (its (view, point)), or null. While set it
  // temporarily takes over the status widget's readout -- the name field and the lit
  // state chip -- so hovering any joint peeks at its identity and state without
  // disturbing the selection (the chips still act on the selection, not the hover).
  /** @type {{ view: number, point: number } | null} */
  hoverCell = null;
  // The latest per-view ground-truth mask and "projected" mask (from the points payload),
  // so the status widget can report each selected joint's source. `projectedMask` marks a
  // cell with no observed pixel here -- the operator occluded the view, or the detector
  // never fired -- whose drawn position follows the 3D reprojection. Null until the first
  // payload.
  /** @type {boolean[][] | null} */
  fixedMask = null;
  /** @type {boolean[][] | null} */
  projectedMask = null;
  /** @type {boolean[][] | null} [view][point] "not on this animal" IN THIS FRAME -- broadcast over views */
  absentMask = null;
  /** @type {number[]} point indices absent in EVERY frame, so the badge can say which scope */
  absentRecording = [];
  // Per-view mask of cells the detector actually fired for (a finite raw prediction).
  // A cell with no detection can never be reset *to* the detector, so the "Detected"
  // state chip is disabled for it. Built from the verbose payload's `pred`; static
  // within a frame, so it survives the mid-drag edit stream (which omits `pred`).
  /** @type {boolean[][] | null} */
  detectedMask = null;
  // On-demand 3D view (rig + 3D pose + NMF skeleton/mesh), built lazily on first open.
  /** @type {Scene3D | null} */
  scene = null;
  sceneOpen = false;
  helpOpen = false;
  helpBuilt = false;
  // Live "adding to selection" affordance: `addMod` tracks whether the add-modifier
  // (Ctrl/⌘) is currently held and `overViews` whether the pointer is over the camera
  // views. When both hold, <body> gets the `adding` class so the cursor turns "copy" and
  // a hint pill appears -- teaching the "Ctrl/⌘ = add" convention at the moment of use.
  addMod = false;
  overViews = false;
  // True once a deliberate Close is under way: stops the unsaved-changes guard
  // (beforeunload) from nagging after the operator has already decided.
  closing = false;
  closeConfirmOpen = false;
  // The corrected-frames side panel: the list (sorted, each with a reviewed flag),
  // whether the panel is open, the row elements keyed by frame (for the current-frame
  // highlight), and a debounce timer coalescing post-edit refreshes.
  /** @type {CorrectedFrame[]} */
  correctedFrames = [];
  framesOpen = false;
  /** @type {Map<number, HTMLTableRowElement>} */
  frameRows = new Map();
  correctedTimer = 0;
  // The side panel's two tabs. "labeled" is the frames-with-GT list above; "suggest" is
  // the ranked queue from `deeperfly labels-suggest` (a static sidecar, so it is fetched
  // on load / on save / on tab activation, never per edit). `suggestions` is null until
  // the first fetch resolves and stays null when no queue has been computed -- the tab
  // then explains how to make one instead of looking broken. Which frames are DONE is
  // not read from the sidecar but joined live from `correctedFrames` at render time, so a
  // row flips the moment its frame is labeled, with no refetch.
  /** @type {SidebarTab} */
  sidebarTab = "labeled";
  /** @type {SuggestionsPayload | null} */
  suggestions = null;
  /** @type {Map<number, HTMLTableRowElement[]>} */
  suggestRows = new Map();
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
  hideAllCheck = el("show-hide-all");
  /** @type {HTMLInputElement} */
  combinedCheck = el("show-combined");
  /** @type {HTMLInputElement} */
  labelsCheck = el("show-labels");
  /** @type {HTMLLabelElement} */
  gtWrap = el("gt-wrap");
  /** @type {HTMLInputElement} */
  gtCheck = el("show-gt");
  /** @type {HTMLLabelElement} */
  detectedWrap = el("detected-wrap");
  /** @type {HTMLInputElement} */
  detectedCheck = el("show-detected");
  /** @type {HTMLLabelElement} */
  projectedWrap = el("projected-wrap");
  /** @type {HTMLInputElement} */
  projectedCheck = el("show-projected");
  /** @type {HTMLLabelElement} */
  placeholderWrap = el("placeholder-wrap");
  /** @type {HTMLInputElement} */
  placeholderCheck = el("show-placeholder");
  /** @type {HTMLDivElement} */
  warnSection = el("warn-section");
  /** @type {HTMLLabelElement} */
  warnWrap = el("warn-wrap");
  /** @type {HTMLInputElement} */
  warnCheck = el("show-warn");
  /** @type {HTMLLabelElement} */
  warnThresholdWrap = el("warn-threshold-wrap");
  /** @type {HTMLInputElement} */
  warnThresholdInput = el("warn-threshold");
  /** @type {HTMLDivElement} */
  referenceSection = el("reference-section");
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
  /** @type {HTMLSpanElement} */
  pointStatusName = el("point-status-name");
  absentBtn = el("act-absent");
  absentBadge = el("absent-badge");
  /** @type {Segmented} */
  stateSwitch;
  /** @type {Segmented} */
  sidebarTabs;
  /** @type {HTMLButtonElement} */
  actResetBtn = el("act-reset");
  /** @type {HTMLButtonElement} */
  undoBtn = el("undo");
  /** @type {HTMLButtonElement} */
  redoBtn = el("redo");
  /** @type {HTMLDivElement} */
  showWrap = el("show-wrap");
  /** @type {HTMLButtonElement} */
  showToggle = el("show-toggle");
  /** @type {HTMLDivElement} */
  showMenu = el("show-menu");
  showMenuOpen = false;
  /** @type {HTMLButtonElement} */
  layoutToggle = el("layout-toggle");
  /** @type {HTMLDivElement} */
  layoutMenu = el("layout-menu");
  layoutMenuOpen = false;
  /** @type {HTMLDivElement} */
  layoutArrangeSection = el("layout-arrange-section");
  /** @type {HTMLDivElement} */
  layoutArrangeRow = el("layout-arrange-row");
  /** @type {HTMLButtonElement} */
  resetViewBtn = el("reset-view");
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
  sidebarTabsEl = el("sidebar-tabs");
  /** @type {HTMLSpanElement} */
  suggestCountEl = el("suggest-count");
  /** @type {HTMLDivElement} */
  labeledPane = el("labeled-pane");
  /** @type {HTMLDivElement} */
  suggestPane = el("suggest-pane");
  /** @type {HTMLDivElement} */
  suggestStatusEl = el("suggest-status");
  /** @type {HTMLTableSectionElement} */
  suggestTbody = /** @type {HTMLTableElement} */ (el("suggest-table")).tBodies[0];
  /** @type {HTMLDivElement} */
  suggestEmptyEl = el("suggest-empty");
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
  readonlyBanner = el("readonly-banner");
  /** @type {HTMLButtonElement} */
  readonlyTakeover = el("readonly-takeover");
  // True while another browser holds the writer slot: no edits leave this tab, the
  // edit affordances are disabled, and the read-only banner is shown. Panning and
  // zooming to inspect stay available. Flipped by the server's role handshake.
  readOnly = false;
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
    this.applyOsHints();
    this.buildViews();
    // Sync the (persisted) reprojection-warning state into the freshly-built views. With no saved
    // override this matches their constructor defaults, so the setters early-return -- no extra draw.
    this.applyWarn();
    this.applyWarnThreshold();
    this.relayout();
    this.socket = new EditSocket(
      (p) => this.applyPoints(p, true),
      (r) => this.applyRole(r),
    );
    await this.goToFrame(0);
    this.updateSelected();
    this.updateDirty();
    await this.refreshCorrected(); // populate the list (any corrections loaded from disk)
    this.refreshSuggestions(); // the ranked queue, if one has been computed (not awaited)
    // Closing instantly when there is nothing to lose, prompting otherwise: the
    // browser shows its generic "leave site?" dialog only while edits are unsaved.
    window.addEventListener("beforeunload", (e) => {
      if (this.dirty && !this.closing) {
        e.preventDefault();
        e.returnValue = "";
      }
    });
    window.addEventListener("keydown", (e) => this.onKey(e));

    // Live "adding to selection" affordance. Track the add-modifier (Ctrl/⌘) from every
    // key event's modifier state -- keyup included, so releasing the key clears it -- and
    // reset on blur (a modifier released while the window is unfocused fires no keyup).
    // `overViews` is driven by the views container's enter/leave (pointerenter/leave do
    // not bubble from the child canvases, so they fire once per real boundary crossing).
    const setAddMod = (/** @type {boolean} */ down) => {
      if (down === this.addMod) return;
      this.addMod = down;
      this.updateAddingHint();
    };
    window.addEventListener("keydown", (e) => setAddMod(e.ctrlKey || e.metaKey));
    window.addEventListener("keyup", (e) => setAddMod(e.ctrlKey || e.metaKey));
    window.addEventListener("blur", () => setAddMod(false));
    this.viewsEl.addEventListener("pointerenter", () => {
      this.overViews = true;
      this.updateAddingHint();
    });
    this.viewsEl.addEventListener("pointerleave", () => {
      this.overViews = false;
      this.updateAddingHint();
    });
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

    // A single camera has no arrangement to choose, so only the Grid/Focus segment is
    // hidden -- the Layout menu itself stays, since it also holds "Reset view", which one
    // (still zoomable) camera can use too.
    const multiCam = this.meta.n_views > 1;
    this.layoutArrangeSection.style.display = multiCam ? "" : "none";
    this.layoutArrangeRow.style.display = multiCam ? "" : "none";
    // The default button title advertises the Grid/Focus + step-focus keys; drop that clause
    // for a single camera, where only "Reset view" remains.
    if (!multiCam) this.layoutToggle.title = "View — reset zoom & pan on the camera (0)";
    this.layoutSwitch = segmented(
      [["Grid", "grid"], ["Focus", "focus"]],
      (v) => this.setLayout(/** @type {Layout} */ (v))
    );
    this.layoutSwitch.set(this.layout);
    el("layout-switch").append(this.layoutSwitch.root);

    // The selection's per-view state, as both a readout and a setter: clicking a chip
    // applies that state to every selected cell at once. "Ground truth" is an authored
    // GT pixel (Confirm); "Projected" drops the view from the 3D solve so the point
    // follows the reprojection (Occlude); "Detected" is the detector's own 2D peak,
    // reached via the Reset button (clear back to the detector / reprojection). The wire
    // value for the third chip stays "invisible"/occlude (see the server's edit types).
    this.stateSwitch = segmented(
      [["Ground truth", "fixed"], ["Detected", "normal"], ["Projected", "projected"]],
      (v) => this.setSelectedState(v)
    );
    el("point-status-states").append(this.stateSwitch.root);
    this.absentBtn.addEventListener("click", (e) =>
      this.toggleAbsentSelection(e.shiftKey ? "recording" : "frame"),
    );

    // The side panel's tab strip, in place of a plain title: two lists that differ in
    // both columns and ordering (labeled frames in time order; suggested frames in rank
    // order), so they are separate tabs rather than one filtered list. Reuses the
    // established `.segmented` component, so the strip needs no new visual language.
    this.sidebarTabs = segmented(
      [["Labeled", "labeled"], ["Suggested", "suggest"]],
      (v) => this.setSidebarTab(/** @type {SidebarTab} */ (v)),
    );
    this.sidebarTabsEl.append(this.sidebarTabs.root);
    this.setSidebarTab(this.sidebarTab);
    // Pin the name readout to its widest possible value so hovering / selecting different
    // joints never reflows the widget (and thus never nudges the controls after it).
    this.reserveStatusNameWidth();

    this.hideAllCheck.addEventListener("change", () => this.applyHideAll());
    this.combinedCheck.addEventListener("change", () => this.applyCombined());
    this.labelsCheck.addEventListener("change", () => this.applyLabels());
    // The ground-truth and detected source layers exist without 3D (they are the authored
    // pixels and the raw detector output); only the projected source needs a 3D solve.
    this.gtCheck.addEventListener("change", () => this.applyGt());
    this.detectedCheck.addEventListener("change", () => this.applyDetected());
    this.projectedWrap.style.display = this.meta.has_3d ? "" : "none";
    this.projectedCheck.addEventListener("change", () => this.applyProjected());
    // The "Unplaced" placeholder seeds are the guarantee that no joint is ever unreachable: a cell
    // with nothing else drawn still gets a faint ghost to drag into a GT label (the authored 2D
    // needs no prior 3D), so the layer is available with or without a 3D solve.
    this.placeholderCheck.addEventListener("change", () => this.applyPlaceholder());
    // The reprojection-distance warning flags joints whose GT/detected pixel is far from the 3D
    // reprojection -- only meaningful with a 3D solve, so it shares the projected row's has_3d gate.
    const warnAvailable = this.meta.has_3d;
    this.warnSection.style.display = warnAvailable ? "" : "none";
    this.warnWrap.style.display = warnAvailable ? "" : "none";
    this.warnThresholdWrap.style.display = warnAvailable ? "" : "none";
    // Restore the persisted preference (the one persisted display setting) over the HTML defaults
    // (on, 8 px), so an operator's tolerance survives a reload. The initial fan-out into the views
    // happens after buildViews() in init().
    const savedOn = localStorage.getItem(WARN_ON_KEY);
    if (savedOn !== null) this.warnCheck.checked = savedOn === "1";
    const savedPx = Number(localStorage.getItem(WARN_PX_KEY));
    if (Number.isFinite(savedPx) && savedPx > 0) this.warnThresholdInput.value = String(savedPx);
    this.warnCheck.addEventListener("change", () => this.applyWarn());
    this.warnThresholdInput.addEventListener("input", () => this.applyWarnThreshold());
    // On commit (blur / Enter) snap the shown value back to the clamped one we actually applied.
    this.warnThresholdInput.addEventListener("change", () => {
      this.warnThresholdInput.value = String(this.clampWarnPx());
      this.applyWarnThreshold();
    });
    // The NMF overlay is the fitted inverse-kinematics model -- only when present. The
    // "Reference" section heading is hidden with it, so it never dangles over no rows.
    this.referenceSection.style.display = this.meta.has_nmf ? "" : "none";
    this.nmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.nmfCheck.addEventListener("change", () => this.applyNmf());
    // The NMF mesh overlay (rendered on the client GPU) -- only when a fitted model
    // is present. The head/abdomen size is estimated from the data by the IK stage
    // (no operator knob), so the overlay just follows the model.
    this.meshWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.meshCheck.addEventListener("change", () => this.applyMesh());

    this.actResetBtn.addEventListener("click", () => this.resetSelection());
    this.undoBtn.addEventListener("click", () => this.undo());
    this.redoBtn.addEventListener("click", () => this.redo());
    // The "Show" overlay-toggle popover: the button opens/closes it; a click anywhere
    // outside closes it (a click on a checkbox inside stays open, so several can be
    // toggled). stopPropagation keeps the opening click from reaching that outside handler.
    this.showToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleShowMenu();
    });
    document.addEventListener("click", (e) => {
      if (this.showMenuOpen && !this.showWrap.contains(/** @type {Node} */ (e.target))) {
        this.closeShowMenu();
      }
    });
    // The "Layout" popover mirrors "Show": the button toggles it, a click outside closes
    // it, and it holds the Grid/Focus arrangement plus "Reset view". Opening one popover
    // closes the other (see openLayoutMenu / openShowMenu), so they never overlap.
    this.layoutToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleLayoutMenu();
    });
    document.addEventListener("click", (e) => {
      if (this.layoutMenuOpen && !this.layoutWrap.contains(/** @type {Node} */ (e.target))) {
        this.closeLayoutMenu();
      }
    });
    this.resetViewBtn.addEventListener("click", () => this.resetView());
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
    // "Take over editing" from the read-only banner: claim the writer slot. The
    // server replies with a role handshake that flips this tab out of read-only.
    this.readonlyTakeover.addEventListener("click", () => this.socket.claim());
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
      onSelect: (v, p, additive) => this.onSelect(v, p, additive),
      onSelectRegion: (v, points, additive) => this.onSelectRegion(v, points, additive),
      onSelectKeypointAllViews: (p, additive) => this.onSelectKeypointAllViews(p, additive),
      onBackground: () => this.clearSelection(),
      onActiveView: (v) => this.onActiveView(v),
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
      // A read-only browser can still pan/zoom the large views to inspect, but no
      // view is editable while another operator holds the writer slot.
      view.setEditable(large && !this.readOnly);
      view.setZoomable(large);
    });
  }

  // Apply the server's role handshake for this browser. The first browser to
  // connect is the writer (editable); every later one is read-only until it takes
  // over (the banner's "Take over editing") or the writer disconnects and it is
  // promoted. Read-only means no edits leave this tab and the edit affordances are
  // disabled -- a banner explains why -- while panning/zooming to inspect stays.
  /** @param {import("./types.js").RoleMessage} r */
  applyRole(r) {
    const readOnly = r.role !== "writer";
    if (readOnly === this.readOnly) return;
    this.readOnly = readOnly;
    document.body.classList.toggle("read-only", readOnly);
    this.readonlyBanner.hidden = !readOnly;
    this.updateViewRoles(); // re-apply per-view editability
    this.updateDirty(); // the Save button is disabled while read-only
    this.renderFrameList(); // re-render so the reviewed checkboxes track editability
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
    // Fetch verbose so the reply carries `pred` (the raw detections) for the Detected
    // source layer. Detections are static within a frame, so this rides the navigation
    // fetch only -- the mid-drag edit stream stays lean (no `pred`), and each view keeps
    // the detections it already has.
    this.applyPoints(await fetchPoints(this.frame, this.mode, true));
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
    // A one-line, self-dismissing notice: the server refused an edit, or confirmed a
    // recording-wide declaration. Non-blocking on purpose -- this gesture happens once per
    // recording, so a modal would be worse than the thing it guards against.
    if (p.notice) {
      this.statusEl.textContent = p.notice;
      clearTimeout(this._noticeTimer);
      this._noticeTimer = setTimeout(() => (this.statusEl.textContent = ""), 4000);
    }
    // The per-view masks drive the source-styled markers (ground truth / projected) and
    // the status widget; they are meaningful whether or not the result carries 3D.
    this.fixedMask = p.fixed;
    // A cell with no observed pixel (null in `points`) follows the 3D reprojection -- the
    // "projected" state. That covers both an operator-occluded view (`p.invisible`) and
    // one the detector missed, since display_pts2d NaNs out both; the status chips read
    // it off this so an undetected view reads as "Projected", not "Detected" (the
    // occluded-only `p.invisible` mask still rides through to each view for the drag
    // un-occlude, but is a strict subset here).
    this.projectedMask = p.points.map((row) => row.map((pt) => pt == null));
    // Absence rides every reply (it gates drawing), so assign unconditionally rather than
    // keeping a previous value the way `pred` / `placeholder` do.
    if (p.absent) {
      this.absentMask = p.absent;
      this.absentRecording = p.absent_recording ?? [];
      this.updateAbsentBadge();
    }
    // Which cells the detector fired for -- a finite raw prediction. Only the verbose
    // navigation fetch carries `pred`; on the mid-drag edit stream (no `pred`) the
    // detections are unchanged within the frame, so keep the mask we already have.
    if (p.pred) this.detectedMask = p.pred.map((row) => row.map((pt) => pt != null));
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
        // `pred` (the raw detections) rides the verbose navigation fetch only; when absent
        // (the mid-drag stream) leave each view's detected set unchanged.
        detected: p.pred ? p.pred[v] : undefined,
        // The Unplaced seeds ride the same verbose reply (`placeholder`); omitted mid-drag -> keep.
        placeholder: p.placeholder ? p.placeholder[v] : undefined,
        absent: p.absent ? p.absent[v] : undefined,
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

  // The master "hide all overlays" peek (the `h` shortcut): hide every overlay in every view for
  // a clean look at the raw frames, or restore them. Non-destructive -- it overrides the per-layer
  // toggles without changing them, so clearing it brings back exactly what was shown. It also
  // gates editing (canGrab) so a point can't be dragged while hidden.
  applyHideAll() {
    const hidden = this.hideAllCheck.checked;
    this.views.forEach((view) => view.setOverlaysHidden(hidden));
  }

  // "Combined" is the merge toggle over the Ground truth + Detected layers: on, they draw as
  // one merged skeleton (GT where authored, else the detector's point); off, as two separate
  // overlaid skeletons. It no longer shows / hides the whole skeleton.
  applyCombined() {
    const merged = this.combinedCheck.checked;
    this.views.forEach((view) => view.setCombinedVisible(merged));
  }

  applyLabels() {
    const visible = this.labelsCheck.checked;
    this.views.forEach((view) => view.setLabelsVisible(visible));
  }

  applyGt() {
    const visible = this.gtCheck.checked;
    this.views.forEach((view) => view.setGtVisible(visible));
  }

  applyDetected() {
    const visible = this.detectedCheck.checked;
    this.views.forEach((view) => view.setDetectedVisible(visible));
  }

  applyProjected() {
    const visible = this.projectedCheck.checked;
    this.views.forEach((view) => view.setProjectedVisible(visible));
  }

  applyPlaceholder() {
    const visible = this.placeholderCheck.checked;
    this.views.forEach((view) => view.setPlaceholderVisible(visible));
  }

  applyWarn() {
    const on = this.warnCheck.checked;
    localStorage.setItem(WARN_ON_KEY, on ? "1" : "0");
    this.views.forEach((view) => view.setWarnVisible(on));
  }

  // The threshold input clamped to a sane pixel range; falls back to the last-applied value when
  // the field is momentarily empty / non-numeric (mid-edit), so a partial keystroke never resets it.
  clampWarnPx() {
    const px = Number(this.warnThresholdInput.value);
    if (!Number.isFinite(px) || px <= 0) return this.views[0]?.warnThreshold ?? 8;
    return Math.min(WARN_PX_MAX, Math.max(WARN_PX_MIN, Math.round(px)));
  }

  applyWarnThreshold() {
    const px = this.clampWarnPx();
    localStorage.setItem(WARN_PX_KEY, String(px));
    this.views.forEach((view) => view.setWarnThreshold(px));
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

  // -- the "Show" overlay-toggle popover --------------------------------------

  openShowMenu() {
    this.closeLayoutMenu(); // only one popover open at a time
    this.showMenu.hidden = false;
    this.showMenuOpen = true;
    this.showToggle.setAttribute("aria-expanded", "true");
    this.showToggle.classList.add("is-open");
  }

  closeShowMenu() {
    this.showMenu.hidden = true;
    this.showMenuOpen = false;
    this.showToggle.setAttribute("aria-expanded", "false");
    this.showToggle.classList.remove("is-open");
  }

  toggleShowMenu() {
    if (this.showMenuOpen) this.closeShowMenu();
    else this.openShowMenu();
  }

  // -- the "Layout" popover (arrangement + view reset) ------------------------

  openLayoutMenu() {
    this.closeShowMenu(); // only one popover open at a time
    this.layoutMenu.hidden = false;
    this.layoutMenuOpen = true;
    this.layoutToggle.setAttribute("aria-expanded", "true");
    this.layoutToggle.classList.add("is-open");
  }

  closeLayoutMenu() {
    this.layoutMenu.hidden = true;
    this.layoutMenuOpen = false;
    this.layoutToggle.setAttribute("aria-expanded", "false");
    this.layoutToggle.classList.remove("is-open");
  }

  toggleLayoutMenu() {
    if (this.layoutMenuOpen) this.closeLayoutMenu();
    else this.openLayoutMenu();
  }

  /** Reset zoom + pan on every camera back to the letterboxed fit (the "tight fit"). */
  resetView() {
    this.views.forEach((view) => view.resetZoom());
    this.closeLayoutMenu();
  }

  // -- hover / selection ------------------------------------------------------

  /** @param {number | null} point  the hovered joint, emphasized in every view */
  onHover(point) {
    this.views.forEach((view) => view.setHighlight(point));
    // Peek: while the pointer is over a joint, the status widget shows *that* joint's
    // name + state (in the view under the cursor), reverting to the selection readout
    // the moment the cursor leaves it.
    this.hoverCell = point == null ? null : { view: this.activeView, point };
    this.updateStatusWidget();
  }

  /** @param {number} view @param {number} point @returns {string} the selection-set key */
  selKey(view, point) {
    return `${view}:${point}`;
  }

  /** @returns {[number, number][]} the selection as (view, point) pairs */
  selCells() {
    return [...this.selection].map((k) => {
      const [v, p] = k.split(":");
      return [Number(v), Number(p)];
    });
  }

  // A single joint was clicked. `additive` (Ctrl/Cmd-click) toggles just that cell in
  // the selection; a plain click (or Shift-click) replaces the selection with it. Either
  // way the clicked cell becomes the anchor (what the status widget inspects).
  /**
   * @param {number} view
   * @param {number} point
   * @param {boolean} [additive]
   */
  onSelect(view, point, additive = false) {
    const key = this.selKey(view, point);
    if (additive) {
      if (this.selection.has(key)) this.selection.delete(key);
      else this.selection.add(key);
    } else {
      this.selection.clear();
      this.selection.add(key);
    }
    this.selAnchor = this.selection.has(key) ? { view, point } : null;
    this.activeView = view;
    this.updateSelected();
  }

  // A modifier+drag marquee enclosed `points` in `view`. `additive` (Ctrl/Cmd-drag) adds
  // them to the selection; otherwise (Shift-drag) it replaces the whole selection with
  // them -- so an empty Shift-drag clears the selection.
  /**
   * @param {number} view
   * @param {number[]} points
   * @param {boolean} [additive]
   */
  onSelectRegion(view, points, additive = true) {
    if (!additive) this.selection.clear();
    for (const p of points) this.selection.add(this.selKey(view, p));
    if (points.length) this.selAnchor = { view, point: points[points.length - 1] };
    else if (!additive) this.selAnchor = null;
    this.activeView = view;
    this.updateSelected();
  }

  // A joint was double-clicked: select that keypoint in every view.
  /**
   * @param {number} point
   * @param {boolean} [additive]
   */
  onSelectKeypointAllViews(point, additive = false) {
    if (!additive) this.selection.clear();
    for (let v = 0; v < this.meta.n_views; v++) this.selection.add(this.selKey(v, point));
    this.selAnchor = { view: this.activeView, point };
    this.updateSelected();
  }

  /** @param {number} view  the camera the pointer is over (drives the `v` gesture) */
  onActiveView(view) {
    this.activeView = view;
  }

  /** Select every point in every view. */
  selectAll() {
    this.selection.clear();
    for (let v = 0; v < this.meta.n_views; v++) {
      for (let p = 0; p < this.meta.n_points; p++) this.selection.add(this.selKey(v, p));
    }
    this.selAnchor = null;
    this.updateSelected();
  }

  /** Select every point in the active view (the one the pointer is over). */
  selectActiveView() {
    const v = this.activeView;
    this.selection.clear();
    for (let p = 0; p < this.meta.n_points; p++) this.selection.add(this.selKey(v, p));
    this.selAnchor = null;
    this.updateSelected();
  }

  /** Clear the selection (Esc, or the next plain click replaces it anyway). */
  clearSelection() {
    if (this.selection.size === 0) return;
    this.selection.clear();
    this.selAnchor = null;
    this.updateSelected();
  }

  // Push each view its own subset of the selection (the cyan rings), enable the Reset
  // button while anything is selected, and refresh the combined state control.
  updateSelected() {
    // Invariant: a single-cell selection always has that cell as its anchor, so the
    // status widget stays usable however the selection got down to one -- a plain
    // click, a Shift+click that *removed* the other cell (which nulls the anchor), or
    // a select-all/select-view on a 1-point rig.
    if (this.selection.size === 1) {
      const [v, p] = this.selCells()[0];
      this.selAnchor = { view: v, point: p };
    }
    this.views.forEach((view, v) => {
      /** @type {Set<number>} */
      const set = new Set();
      for (let p = 0; p < this.meta.n_points; p++) {
        if (this.selection.has(this.selKey(v, p))) set.add(p);
      }
      view.setSelection(set);
    });
    this.actResetBtn.disabled = this.selection.size === 0;
    this.updateStatusWidget();
  }

  // -- point state control ----------------------------------------------------

  /**
   * @param {number} view
   * @param {number} point
   * @returns {"normal" | "fixed" | "projected" | "absent"} the cell's per-view state
   */
  cellState(view, point) {
    // Absence is checked FIRST and short-circuits. An absent cell is null in `points`, so it
    // would otherwise fall through to "projected" and the readout would report the machine's
    // reprojected guess back to the operator as if it were their own label.
    if (this.absentMask && this.absentMask[view][point]) return "absent";
    if (this.fixedMask && this.fixedMask[view][point]) return "fixed";
    // No observed pixel here -- the operator occluded the view, or the detector never
    // fired -- so the drawn position blindly follows the 3D reprojection: the "projected"
    // state. `projectedMask` is the null-in-`points` set, which is exactly that (occluded
    // is a strict subset, so it needs no separate check); GT is never null, so order is
    // moot but fixed is checked first for clarity.
    if (this.projectedMask && this.projectedMask[view][point]) return "projected";
    return "normal";
  }

  /**
   * Whether the detector produced a raw prediction for this cell. A cell with no
   * detection can never fall back *to* the detector, so its "Detected" chip is disabled.
   * @param {number} view
   * @param {number} point
   */
  cellDetected(view, point) {
    return !!(this.detectedMask && this.detectedMask[view][point]);
  }

  /**
   * The state shared by every selected cell, or null when they disagree (or none are
   * selected). This is what lights a chip up as the readout.
   * @returns {"normal" | "fixed" | "projected" | "absent" | null}
   */
  uniformState() {
    const cells = this.selCells();
    if (!cells.length) return null;
    const first = this.cellState(cells[0][0], cells[0][1]);
    return cells.every(([v, p]) => this.cellState(v, p) === first) ? first : null;
  }

  // The combined state control -- one chip row that both reports and sets the selection's
  // state. The name field shows the single cell's "point · camera", the count for
  // several, or "—" for none. The active chip is the selection's shared state (nothing
  // lit when the cells disagree). Chips are live whenever something is selected; the
  // "Projected" chip additionally needs 3D (occluding a view only means something when
  // there is a solve to drop it from), and the "Detected" chip needs at least one selected
  // cell the detector actually fired for (else there is no detection to fall back to).
  // The status readout is always one of a finite, enumerable set of strings --
  // "<point> · <camera>", "<n> points", or "—" -- and the name lists are fixed for the
  // session, so its worst-case pixel width is knowable up front. Measure it once (in the
  // element's own font) and pin the field to it, so the readout is a fixed-size box:
  // nothing after it -- the spacer, then the whole right cluster -- can ever shift as the
  // operator hovers or selects different joints. Runs in buildControls, where the element
  // is already in the DOM (getComputedStyle needs that) and the font never changes after.
  reserveStatusNameWidth() {
    const ctx = document.createElement("canvas").getContext("2d");
    if (!ctx) return; // no 2D canvas: fall back to the CSS width + max-width
    const cs = getComputedStyle(this.pointStatusName);
    ctx.font = cs.font || `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    const width = (s) => ctx.measureText(String(s)).width;
    const widest = (arr) => arr.reduce((m, s) => Math.max(m, width(s)), 0);
    const pair = widest(this.meta.point_names) + width(" · ") + widest(this.meta.camera_names);
    const count = width(`${this.meta.n_points * this.meta.n_views} points`);
    this.pointStatusName.style.width = `${Math.ceil(Math.max(pair, count) + 3)}px`;
  }

  updateStatusWidget() {
    const n = this.selection.size;
    // Hovering a joint takes over the readout (name + lit chip) with that joint's own
    // identity and state -- a transient peek. The chips' enabled/disabled state stays
    // governed by the selection, since clicking a chip still acts on the selection.
    const hov = this.hoverCell;
    const cellName = (view, point) => {
      const name = this.meta.point_names[point] ?? `#${point}`;
      const cam = this.meta.camera_names[view] ?? `view ${view}`;
      return `${name} · ${cam}`;
    };
    let txt;
    if (hov) txt = cellName(hov.view, hov.point);
    else if (n === 0) txt = "—";
    else if (n === 1 && this.selAnchor) txt = cellName(this.selAnchor.view, this.selAnchor.point);
    else txt = `${n} points`;
    this.pointStatusName.textContent = txt;
    // The field is pinned to the worst-case width (reserveStatusNameWidth), so for a real
    // config it never clips: set a title only on the off chance a pathological name
    // overflows, so it stays readable on hover -- and leave it empty otherwise so the
    // widget's own explanatory tooltip still shows.
    this.pointStatusName.title =
      this.pointStatusName.scrollWidth > this.pointStatusName.clientWidth ? txt : "";
    // Absence readout + availability. `aria-pressed` (mirrored by a class) is the state: a
    // selection of joints that are ALL absent shows the toggle pressed, so the button doubles as
    // the answer to "is this joint declared absent?" without a fourth chip.
    const selPts = [...new Set(this.selCells().map(([, p]) => p))];
    const allAbsent =
      selPts.length > 0 && selPts.every((p) => this.absentMask && this.absentMask[0][p]);
    this.absentBtn.disabled = n === 0 || this.readOnly;
    this.absentBtn.setAttribute("aria-pressed", String(allAbsent));
    this.absentBtn.classList.toggle("is-absent", allAbsent);
    // The three source chips answer "where did this marker come from", which has no meaning for
    // a joint that is not on the animal -- so they are all disabled while it is declared absent.
    this.stateSwitch.setDisabled(n === 0 || allAbsent);
    this.stateSwitch.setDisabledValue("projected", n === 0 || allAbsent);
    // "Detected" resets a cell back to the detector's raw prediction -- meaningless for a
    // cell the detector never fired for (it would just fall through to the reprojection).
    // Disable the chip unless at least one selected cell has a detection, so an
    // undetected joint can't be "converted to Detected".
    const anyDetected = this.selCells().some(([v, p]) => this.cellDetected(v, p));
    this.stateSwitch.setDisabledValue("normal", n === 0 || allAbsent || !anyDetected);
    // The lit chip: the hovered joint's own state while hovering (the readout peek),
    // else the selection's shared state (nothing lit when the cells disagree).
    this.stateSwitch.set(hov ? this.cellState(hov.view, hov.point) : (this.uniformState() ?? ""));
  }

  // Click a state chip to apply that state to the whole selection at once. The chips are
  // the bulk verbs in disguise: "Ground truth" confirms, "Projected" occludes (drops the
  // view so the point follows the reprojection), and "Detected" resets back to the
  // detector (the same as the Reset button) -- so a multi-select changes state in one
  // undoable step.
  /** @param {string} target  "normal" | "fixed" | "projected" */
  setSelectedState(target) {
    if (this.selection.size === 0) return;
    if (target === "fixed") this.confirmSelection();
    else if (target === "projected") this.occludeSelection();
    else this.resetSelection();
  }

  // -- edit routing -----------------------------------------------------------

  // Send an edit over the socket, stamped with a monotonic seq the server echoes
  // back so applyPoints can drop a superseded reply (a mid-drag re-solve that lands
  // after release, or after a newer edit) instead of repainting a stale position.
  /** @param {import("./types.js").EditMessage} msg */
  sendEdit(msg) {
    if (this.readOnly) return; // a read-only browser cannot mutate the shared state
    // Ask for the verbose reply (which carries the refreshed `pred` + `placeholder` seeds, both
    // affected by the edit) on every settle / discrete edit, so the Unplaced ghosts appear/vanish
    // as points are placed and cleared. The one exception is the mid-drag live stream (edit_3d
    // with fix:false, ~60x/s) -- it stays lean, and the seeds refresh on the settle reply.
    const liveDrag = msg.type === "edit_3d" && msg.fix === false;
    const verbose = msg.verbose ?? !liveDrag;
    this.socket.send({ ...msg, verbose, seq: ++this.editSeq });
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

  // -- actions on the selection (confirm / reset / occlude) -------------------

  // Confirm the selection: snapshot each selected cell's shown position as ground
  // truth (prediction where the detector fired, else the projection). One undo step.
  confirmSelection() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "confirm", targets, sources: "all", frame: this.frame, mode: this.mode });
  }

  // Reset the selection: clear each selected cell's authored label (ground truth or
  // occlusion) back to unset, so it falls back to the detector prediction. One undo step.
  resetSelection() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "reset", targets, frame: this.frame, mode: this.mode });
  }

  // Occlude the selection: flag each selected cell unreadable in its view (dropping it
  // from the 3D solve). Reverse via Reset / undo. One undo step. Deliberately NOT gated on
  // 3D: occlusion is an authored label, and on a 2D-only pass gating it would leave the
  // operator reaching for the far stronger "not on this animal" instead.
  occludeSelection() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "occlude", targets, frame: this.frame, mode: this.mode });
  }

  // Toggle "not on this animal" for the selected joint(s) -- an amputated leg, an ablated
  // antenna. View-independent, so the (view, point) selection collapses to a point SET:
  // "absent in rf but present in lf" is not expressible. Frames are different -- a leg can
  // be lost part-way through a recording -- so `scope` says which the gesture meant:
  // "frame" (the default, `x`) or "recording" (`Shift+X`, the convenience for the common
  // case of an animal that arrives with a leg already missing).
  //
  // Sends an explicit `absent` value rather than a per-point toggle so a mixed selection
  // resolves one way (set unless every selected point is already absent) instead of
  // splitting. For the recording scope that test reads the whole-recording set, so
  // Shift+X on a partly-declared point extends it rather than clearing it.
  /** @param {"frame" | "recording"} scope */
  toggleAbsentSelection(scope = "frame") {
    const targets = this.selCells();
    if (!targets.length) return;
    const pts = [...new Set(targets.map(([, p]) => p))];
    const whole = new Set(this.absentRecording ?? []);
    const allAbsent =
      scope === "recording"
        ? pts.every((p) => whole.has(p))
        : pts.every((p) => this.absentMask && this.absentMask[0][p]);
    this.sendEdit({
      type: "set_absent",
      targets,
      absent: !allAbsent,
      scope,
      frame: this.frame,
      mode: this.mode,
    });
  }

  // A persistent readout of what this animal is missing. Absence removes the joint from every
  // view, so without a standing badge a declared amputation is indistinguishable from a rig that
  // never mapped those points -- and from a bug.
  updateAbsentBadge() {
    const mask = this.absentMask;
    if (!mask || !mask.length) {
      this.absentBadge.hidden = true;
      return;
    }
    const points = [];
    for (let p = 0; p < mask[0].length; p++) if (mask[0][p]) points.push(p);
    if (!points.length) {
      this.absentBadge.hidden = true;
      return;
    }
    const names = points.map((p) => this.meta.point_names[p] ?? `#${p}`);
    // The mask is this FRAME's, so the badge says "here" unless every absent point is
    // absent in every frame -- a per-frame declaration must not read like an amputation,
    // and vice versa. `absent_recording` (the whole-recording subset) rides the payload so
    // the two can be told apart without fetching the full mask.
    const whole = new Set(this.absentRecording ?? []);
    const allWhole = points.every((p) => whole.has(p));
    const scope = allWhole ? "" : " here";
    this.absentBadge.hidden = false;
    this.absentBadge.textContent =
      `✕ absent${scope}: ${names.length} point${names.length > 1 ? "s" : ""}`;
    this.absentBadge.title =
      `Not on this animal, every view${allWhole ? ", every frame" : ` (frame ${this.frame})`}: ` +
      `${names.join(", ")}. Select a joint and press x (this frame) or Shift+X ` +
      `(whole recording) to change this.`;
  }

  // -- undo / redo ------------------------------------------------------------

  undo() {
    this.sendEdit({ type: "undo", frame: this.frame, mode: this.mode });
  }

  redo() {
    this.sendEdit({ type: "redo", frame: this.frame, mode: this.mode });
  }

  async save() {
    if (this.readOnly) return; // read-only: the writer owns saving the shared state
    const r = await saveCorrections();
    this.dirty = r.dirty;
    this.updateDirty();
    this.statusEl.textContent = "saved";
    setTimeout(() => (this.statusEl.textContent = ""), 3000);
    // A save is when the sidecar's own view of "already labeled" could have moved, so
    // it is the one edit-side moment worth re-reading the queue's staleness for.
    this.refreshSuggestions();
  }

  updateDirty() {
    document.title = `deeperfly gui — ${this.meta.results_path}${this.dirty ? " *" : ""}`;
    this.saveBtn.disabled = !this.dirty || this.readOnly;
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
    // The queue's "done" marks are joined from this very list, so re-render it here
    // rather than refetching: a suggested frame flips to done the instant its first
    // point is dragged, on the refresh the edit already triggers.
    this.renderSuggestList();
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
    const rows = this.correctedFrames.map(({ frame, reviewed }) => {
      const tr = document.createElement("tr");
      const fcell = document.createElement("td");
      fcell.textContent = String(frame);
      // A "reviewed" tick box per frame -- the operator's "I've finished checking this"
      // flag. Its own clicks must not bubble to the row (which jumps to the frame).
      const rcell = document.createElement("td");
      rcell.className = "reviewed-cell";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = reviewed;
      box.disabled = this.readOnly;
      box.title = reviewed ? "Reviewed — click to un-mark" : "Mark this frame reviewed";
      box.addEventListener("click", (e) => e.stopPropagation());
      box.addEventListener("change", () => this.toggleReviewed(frame, box.checked, box));
      rcell.append(box);
      tr.append(fcell, rcell);
      tr.addEventListener("click", () => this.goToFrame(frame));
      this.frameRows.set(frame, tr);
      return tr;
    });
    this.framesTbody.replaceChildren(...rows);
    this.updateActiveFrameRow();
  }

  // Mark a frame reviewed (or clear it) from its checkbox in the Labels list -- a
  // per-frame "I've finished checking this" flag, independent of the point labels. It
  // persists with the labels and keeps the frame listed even after its labels are
  // reset. A read-only browser cannot change it, so the checkbox reverts. The frame may
  // not be the one on screen, so we update the list from here (an edit reply for a
  // non-current frame does not repaint) plus an optimistic local flip so the tick sticks.
  /**
   * @param {number} frame
   * @param {boolean} value
   * @param {HTMLInputElement} box
   */
  toggleReviewed(frame, value, box) {
    if (this.readOnly) {
      box.checked = !value; // read-only: undo the visual toggle, change nothing
      return;
    }
    const row = this.correctedFrames.find((f) => f.frame === frame);
    if (row) row.reviewed = value;
    this.sendEdit({ type: "set_reviewed", frame, reviewed: value, mode: this.mode });
    this.scheduleCorrectedRefresh();
  }

  // Highlight the row for the current frame (when it is a corrected one) and, while
  // the panel is open, scroll it into view -- so scrubbing keeps the list in sync.
  updateActiveFrameRow() {
    this.frameRows.forEach((tr, frame) => tr.classList.toggle("is-current", frame === this.frame));
    this.suggestRows.forEach((trs, frame) => {
      for (const tr of trs) tr.classList.toggle("is-current", frame === this.frame);
    });
    if (!this.framesOpen) return;
    // Only the VISIBLE tab is scrolled: scrolling a hidden pane is at best wasted and at
    // worst a surprise jump the moment that tab is shown.
    const active =
      this.sidebarTab === "suggest"
        ? this.suggestRows.get(this.frame)?.[0]
        : this.frameRows.get(this.frame);
    active?.scrollIntoView({ block: "nearest" });
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

  // Step to the previous / next frame of the ACTIVE tab's list (wrapping at the ends),
  // so the operator can walk it without hunting on the scrubber. On the Labeled tab
  // that is the corrected frames in time order; on Suggested it is the queue in rank
  // order (see jumpSuggested) -- the ↑/↓ buttons and keys mean "step my list" either way.
  /** @param {number} dir  -1 for the previous entry, +1 for the next */
  jumpCorrected(dir) {
    if (this.sidebarTab === "suggest") {
      this.jumpSuggested(dir);
      return;
    }
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

  // -- suggested-frames list --------------------------------------------------
  //
  // The ranked queue written by `deeperfly labels-suggest`: which frames are most worth a
  // human's next pass, ordered by how much the cameras disagree about the detector's own
  // 2D (NOT by its confidence -- the model is confidently wrong exactly where it is
  // wrong). The queue is a file on disk, not a live computation, so this list is read
  // once per load / save / tab activation. Everything the operator needs to trust or
  // distrust it -- how stale it is, whether the count fell short of what was asked, and
  // whether the result's contralateral 2D was reseeded (which would make the *stored*
  // reprojection error rank nothing) -- is composed by the server and shown in the status
  // strip, so the panel can never present a superseded queue as current.

  // Fetch the queue and repaint the tab. A missing sidecar is the normal starting state
  // and resolves to `present: false`; a transient failure leaves whatever was there.
  async refreshSuggestions() {
    let payload;
    try {
      payload = await fetchSuggestions();
    } catch (_) {
      return;
    }
    this.suggestions = payload;
    this.renderSuggestList();
  }

  // The queue rows to render: none until the fetch resolves, none for a queue computed
  // against a different recording (`stale.level === "hard"` -- nothing in it can be
  // trusted, so a plausible-looking list would be worse than an empty one).
  /** @returns {Suggestion[]} */
  suggestEntries() {
    const s = this.suggestions;
    if (!s || !s.present || s.stale?.level === "hard") return [];
    return s.frames ?? [];
  }

  // Rebuild the queue table, its badge, its status strip and its empty state. Which rows
  // are DONE is joined live from `correctedFrames` (the same source the Labeled tab
  // uses, so the two can never disagree) rather than trusted from the sidecar, which was
  // written before this session's edits.
  renderSuggestList() {
    const entries = this.suggestEntries();
    /** @type {Map<number, boolean>} frame -> reviewed, for every frame with a decision */
    const live = new Map(this.correctedFrames.map((f) => [f.frame, f.reviewed]));
    const done = entries.filter((s) => live.has(s.frame) || s.labeled).length;
    this.renderSuggestStatus(entries.length, done);
    this.suggestRows.clear();
    /** @type {HTMLTableRowElement[]} */
    const rows = [];
    for (const s of entries) {
      const labeled = live.has(s.frame) || s.labeled;
      const reviewed = live.get(s.frame) ?? s.reviewed;
      rows.push(...this.suggestRowsFor(s, labeled, reviewed));
    }
    this.suggestTbody.replaceChildren(...rows);
    this.updateActiveFrameRow();
  }

  // One queue entry as two table rows: the numbers (rank, frame, time, score) and, under
  // them, the WHY -- the kind chip plus the server's one-line reason. Both rows carry the
  // same classes and the same click target, so the pair reads and behaves as one row.
  /**
   * @param {Suggestion} s
   * @param {boolean} labeled  the frame now carries ground truth (or is marked reviewed)
   * @param {boolean} reviewed
   * @returns {HTMLTableRowElement[]}
   */
  suggestRowsFor(s, labeled, reviewed) {
    const num = document.createElement("tr");
    num.append(
      cell("rank-cell", s.rank == null ? "" : String(s.rank)),
      cell("frame-cell", String(s.frame)),
      cell("t-cell", s.t_s == null ? "—" : s.t_s.toFixed(2)),
      cell("score-cell", s.score == null ? "—" : s.score.toFixed(3)),
    );
    const why = document.createElement("tr");
    const wcell = cell("why-cell", "");
    wcell.colSpan = 4;
    const chip = document.createElement("span");
    chip.className = `kind-chip is-${s.kind === "diversity" ? "diversity" : "most-wrong"}`;
    chip.textContent = s.kind === "diversity" ? "diversity" : "most wrong";
    chip.title =
      s.kind === "diversity"
        ? "Picked on a uniform time grid rather than by score, so the round still sees typical poses and not only the hard tail — a low score here is deliberate."
        : "Picked by score: the views disagree most about the detector's 2D here.";
    wcell.append(chip);
    if (labeled) {
      const state = document.createElement("span");
      state.className = "kind-chip is-done";
      state.textContent = reviewed ? "reviewed ✓" : "labeled";
      state.title = reviewed
        ? "You have labeled this frame and ticked it reviewed."
        : "This frame now carries ground truth — done for this round.";
      wcell.append(state);
    }
    const text = document.createElement("span");
    text.className = "suggest-why";
    text.textContent = s.reason?.summary ?? "";
    wcell.append(text);
    why.append(wcell);

    const pct = s.percentile == null ? "" : ` — ${s.percentile.toFixed(1)}th percentile of this recording`;
    const title = `Frame ${s.frame}${s.t_s == null ? "" : ` at ${s.t_s.toFixed(2)} s`}${pct}. Scores rank within this recording only.`;
    for (const tr of [num, why]) {
      tr.classList.add("suggest-row");
      tr.classList.toggle("is-labeled", labeled);
      tr.classList.toggle("is-reviewed", reviewed);
      tr.title = title;
      tr.addEventListener("click", () => this.goToFrame(s.frame));
    }
    this.suggestRows.set(s.frame, [num, why]);
    return [num, why];
  }

  // The badge, the empty state and the status strip above the queue. The strip is where
  // the queue admits its own limits: staleness first (styled by how much it matters),
  // then the caveats the server composed. Without them a short or superseded queue reads
  // exactly like a fresh top-N.
  /** @param {number} total @param {number} done */
  renderSuggestStatus(total, done) {
    const s = this.suggestions;
    const stale = s?.stale;
    const hard = stale?.level === "hard";
    this.suggestCountEl.hidden = !s?.present;
    this.suggestCountEl.textContent = `${done} / ${total}`;
    this.suggestCountEl.classList.toggle("is-zero", total === 0);
    this.suggestCountEl.title = s?.present
      ? `${done} of ${total} suggested frames labeled`
      : "";

    const lines = [];
    for (const r of stale?.reasons ?? []) lines.push({ text: r, cls: stale?.level ?? "none" });
    // Progress is composed here, not by the server: `done` is counted live from the
    // frames just edited, so this line stays true without refetching the sidecar (the
    // server's own `progress` tier deliberately carries no reason string for that reason).
    if (!hard && done > 0) {
      lines.push({
        text: `${done} of ${total} suggested frames labeled — recompute for a fresh queue`,
        cls: "progress",
      });
    }
    if (!hard) for (const n of s?.notes ?? []) lines.push({ text: n, cls: "note" });
    this.suggestStatusEl.hidden = lines.length === 0;
    this.suggestStatusEl.replaceChildren(
      ...lines.map(({ text, cls }) => {
        const div = document.createElement("div");
        div.className = `status-line is-${cls}`;
        div.textContent = text;
        return div;
      }),
    );

    // The empty state doubles as the instruction: the exact command, with the resolved
    // directory, so producing a queue needs no doc lookup.
    const cmd = s?.command ?? "deeperfly labels-suggest <results dir>";
    this.suggestEmptyEl.hidden = total > 0;
    this.suggestEmptyEl.textContent = !s
      ? "Loading…"
      : hard
        ? `This queue was computed for a different recording. Recompute it:\n${cmd}`
        : s.present
          ? // Present but with nothing to show. The reason is not always "no frame
            // scored high enough" -- entries can also have been dropped as outside the
            // recording -- so the wording states the fact and leaves the why to the
            // notes above, which carry it.
            "The queue has no frames to show."
          : `No suggestions yet. Rank the frames most worth labeling next:\n${cmd}`;
  }

  // Switch the side panel's tab: swap the panes, re-point the ↑/↓ nav, and (on the
  // Suggested tab) re-read the sidecar, which may have been recomputed while the editor
  // was open. Rendering is idempotent, so activating a tab is always safe.
  /** @param {SidebarTab} tab */
  setSidebarTab(tab) {
    this.sidebarTab = tab;
    this.sidebarTabs.set(tab);
    this.labeledPane.hidden = tab !== "labeled";
    this.suggestPane.hidden = tab !== "suggest";
    this.sidebarEl.classList.toggle("tab-suggest", tab === "suggest");
    this.updateSidebarNavTitles();
    if (tab === "suggest") this.refreshSuggestions();
    this.updateActiveFrameRow();
  }

  // The ↑/↓ buttons' tooltips name whichever list they currently step.
  updateSidebarNavTitles() {
    const what = this.sidebarTab === "suggest" ? "suggested frame to label" : "labeled frame";
    this.framesPrevBtn.title = `Previous ${what} (↑)`;
    this.framesNextBtn.title = `Next ${what} (↓)`;
  }

  // Walk the queue in RANK order (not time order), skipping frames already done, so one
  // keystroke moves to the next frame actually worth a pass. Wraps at both ends; falls
  // back to the full list once every entry is done, so the keys never go dead.
  /** @param {number} dir  -1 for the previous entry in the queue, +1 for the next */
  jumpSuggested(dir) {
    const entries = this.suggestEntries();
    if (entries.length === 0) return;
    const live = new Set(this.correctedFrames.map((f) => f.frame));
    const pending = entries.filter((s) => !live.has(s.frame) && !s.labeled);
    const walk = pending.length ? pending : entries;
    const at = walk.findIndex((s) => s.frame === this.frame);
    // Not on a queue frame: enter the queue at its top (or bottom, stepping backwards).
    const next = at < 0 ? (dir > 0 ? 0 : walk.length - 1) : (at + dir + walk.length) % walk.length;
    this.goToFrame(walk[next].frame);
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
    const lastFrame = () => Math.max(0, this.meta.n_frames - 1);
    /** @type {Binding[]} */
    const b = [
      // Frame scrub. Arrows already accept Shift (the matcher only rejects Ctrl/Cmd/Alt),
      // so the ±10 tier needs no extra plumbing -- run() reads e.shiftKey. The coarse ±100
      // jump lives on PageUp/PageDn, NOT Alt+Arrow: Alt+Arrow is the browser's Back/Forward
      // and the matcher can't even represent it, so it would navigate away and lose unsaved
      // labels. Up/Down hop through the side panel's active list -- labeled frames, or the
      // suggestion queue in rank order (mirroring the sidebar's ↑/↓ buttons);
      // Home/End jump to the first/last frame. All are registered as real (non-global)
      // bindings so `matches`->preventDefault suppresses the browser's own scroll/history
      // default, while non-global lets the frame-number input keep native caret + stepping.
      { key: "ArrowLeft", label: "← / →", desc: "", run: (e) => this.step(e.shiftKey ? -10 : -1) },
      { key: "ArrowRight", hidden: true, label: "→", desc: "", run: (e) => this.step(e.shiftKey ? 10 : 1) },
      { key: "PageUp", hidden: true, label: "PgUp", desc: "", run: () => this.step(-100) },
      { key: "PageDown", hidden: true, label: "PgDn", desc: "", run: () => this.step(100) },
      { key: "ArrowUp", hidden: true, label: "↑", desc: "", run: () => this.jumpCorrected(-1) },
      { key: "ArrowDown", hidden: true, label: "↓", desc: "", run: () => this.jumpCorrected(1) },
      { key: "Home", hidden: true, label: "Home", desc: "", run: () => this.goToFrame(0) },
      { key: "End", hidden: true, label: "End", desc: "", run: () => this.goToFrame(lastFrame()) },
    ];
    if (multi) {
      b.push({ key: "g", group: "cam", label: "g", desc: "Grid layout", run: () => this.setLayout("grid") });
      b.push({ key: "f", group: "cam", label: "f", desc: "Focus layout", run: () => this.setLayout("focus") });
      b.push({ key: "[", group: "cam", label: "[ / ]", desc: "Focus the previous / next camera", run: () => this.cycleFocus(-1) });
      b.push({ key: "]", hidden: true, label: "]", desc: "", run: () => this.cycleFocus(1) });
    }
    // Reset zoom/pan on every camera (also on the Layout menu). Outside the multi-camera
    // block: a single, still-zoomable camera benefits too.
    b.push({ key: "0", group: "cam", label: "0", desc: "Reset the view — fit every camera", run: () => this.resetView() });
    // Master peek: hide every overlay at once for an unobstructed look at the raw frames, then
    // press again to restore them exactly as they were. Leads the "show" group -- it governs
    // all the per-layer toggles below it.
    b.push({ key: "h", group: "show", label: "h", desc: "Hide all overlays — an unobstructed look at the raw frames", run: () => this.toggleCheck(this.hideAllCheck, () => this.applyHideAll()) });
    b.push({ key: "s", group: "show", label: "s", desc: "Combined — merge ground truth + detected into one skeleton", run: () => this.toggleCheck(this.combinedCheck, () => this.applyCombined()) });
    b.push({ key: "n", group: "show", label: "n", desc: "Keypoint names", run: () => this.toggleCheck(this.labelsCheck, () => this.applyLabels()) });
    b.push({ key: "i", group: "show", label: "i", desc: "Unplaced-point seeds — draggable ghosts wherever a joint has nothing else to grab", run: () => this.toggleCheck(this.placeholderCheck, () => this.applyPlaceholder()) });
    if (has3d) {
      b.push({ key: "p", group: "show", label: "p", desc: "Reprojected skeleton (3D reprojection)", run: () => this.toggleCheck(this.projectedCheck, () => this.applyProjected()) });
      b.push({ key: "w", group: "show", label: "w", desc: "Reprojection-distance warning", run: () => this.toggleCheck(this.warnCheck, () => this.applyWarn()) });
    }
    if (this.meta.has_nmf) {
      b.push({ key: "m", group: "show", label: "m", desc: "NMF skeleton overlay", run: () => this.toggleCheck(this.nmfCheck, () => this.applyNmf()) });
      b.push({ key: "M", group: "show", label: hint("M", ["shift"]), desc: "NMF mesh overlay", run: () => this.toggleCheck(this.meshCheck, () => this.applyMesh()) });
    }
    // Selecting points. `a`/`v` are keyboard entries; the mouse gestures and the
    // "Ctrl/⌘ = add" rule are rendered in the curated help (buildHelp), so these carry
    // no group. Ctrl/⌘+A is an idempotent alias matching the universal Select-All.
    b.push({ key: "a", label: "a", desc: "Select all points (every view)", run: () => this.selectAll() });
    b.push({ key: "a", mod: true, hidden: true, label: hint("A", ["mod"]), desc: "", run: () => this.selectAll() });
    b.push({ key: "v", label: "v", desc: "Select every point in the view under the cursor", run: () => this.selectActiveView() });
    // Acting on the selection: 1 / 2 / 3 set the whole selection's state, left-to-right in the
    // same order as the status-card chips (Ground truth · Detected · Projected). Enter / r / o
    // stay as hidden aliases so the older muscle memory -- and the Reset button's r -- keep working.
    b.push({ key: "1", group: "edit", label: "1", desc: "Ground truth — confirm the selection", run: () => this.confirmSelection() });
    b.push({ key: "Enter", group: "edit", hidden: true, label: "Enter", desc: "", run: () => this.confirmSelection() });
    b.push({ key: "2", group: "edit", label: "2", desc: "Detected — reset the selection to the detector", run: () => this.resetSelection() });
    b.push({ key: "r", group: "edit", hidden: true, label: "r", desc: "", run: () => this.resetSelection() });
    b.push({ key: "Backspace", hidden: true, label: "Backspace", desc: "", run: () => this.resetSelection() });
    b.push({ key: "Delete", hidden: true, label: "Delete", desc: "", run: () => this.resetSelection() });
    b.push({ key: "3", group: "edit", label: "3", desc: "Projected — no usable observation here; drop the view from the 3D solve and follow the reprojection", run: () => this.occludeSelection() });
    b.push({ key: "o", group: "edit", hidden: true, label: "o", desc: "", run: () => this.occludeSelection() });
    b.push({ key: "x", group: "edit", label: "x", desc: "Absent — this keypoint is not on this animal (amputated / ablated). This frame, every view; press again to un-mark", run: () => this.toggleAbsentSelection("frame") });
    // Uppercase key rather than `shift: true`: for a non-mod binding this keymap takes
    // shift as implied by the key itself (see `matches`), the same way Shift+M works.
    b.push({ key: "X", group: "edit", label: hint("X", ["shift"]), desc: "Absent for the whole recording — the usual case, an animal that arrives with a leg already missing", run: () => this.toggleAbsentSelection("recording") });
    b.push({ key: "z", mod: true, group: "hist", label: hint("Z", ["mod"]), desc: "Undo", run: () => this.undo() });
    // Redo answers to both ⌘Y and ⇧⌘Z; the help shows whichever the platform expects
    // (⇧⌘Z is the macOS idiom, Ctrl+Y the Windows/Linux one) while the other stays a
    // hidden alias that still works.
    b.push({ key: "y", mod: true, group: "hist", hidden: IS_MAC, label: hint("Y", ["mod"]), desc: "Redo", run: () => this.redo() });
    b.push({ key: "z", mod: true, shift: true, group: "hist", hidden: !IS_MAC, label: hint("Z", ["mod", "shift"]), desc: "Redo", run: () => this.redo() });
    b.push({ key: "s", mod: true, global: true, group: "hist", label: hint("S", ["mod"]), desc: "Save labels", run: () => this.save() });
    b.push({ key: "c", group: "panel", label: "c", desc: "Show / hide the 3D scene", run: () => this.toggleScene() });
    b.push({ key: "j", group: "panel", label: "j", desc: "Show / hide the side panel — labeled frames + the suggested queue", run: () => this.toggleFrames() });
    b.push({ key: "k", group: "panel", label: "k", desc: "Open the labeling guide (keypoint map, new tab)", run: () => this.openKeypoints() });
    b.push({ key: "?", group: "panel", label: "?", desc: "Toggle this help", run: () => this.toggleHelp() });
    return b;
  }

  /**
   * The help rows for the visible bindings tagged with a group, rendered from each
   * binding's own OS-correct label + description.
   * @param {string} group
   * @returns {string[]}
   */
  bindingRows(group) {
    return this.bindings
      .filter((b) => !b.hidden && b.group === group)
      .map((b) => `<tr><td class="key"><kbd>${b.label}</kbd></td><td>${b.desc}</td></tr>`);
  }

  buildHelp() {
    // A curated, GROUPED view of the shortcuts. The behaviour lives in `bindings`
    // (buildBindings); this is the human-facing map, so the frame-nav ladder and the
    // selection model -- neither of which maps one-key-to-one-row -- read clearly. Every
    // key is rendered through `hint()`, so the chips match the operator's own OS, and the
    // groups are ordered by the core annotation loop: navigate → select → edit, then the
    // set-once configuration groups, then history/file and the panels.
    const kbd = (/** @type {string} */ label) => `<kbd>${label}</kbd>`;
    const row = (/** @type {string[]} */ keys, /** @type {string} */ desc) =>
      `<tr><td class="key">${keys.map(kbd).join(" ")}</td><td>${desc}</td></tr>`;
    const section = (
      /** @type {string} */ title,
      /** @type {string[]} */ rows,
      /** @type {string} */ note = "",
    ) =>
      `<h3 class="legend-title">${title}</h3>`
      + (note ? `<p class="legend-note">${note}</p>` : "")
      + `<table class="shortcuts"><tbody>${rows.join("")}</tbody></table>`;

    const out = [];

    // Navigate frames -- the full escalation ladder, spelled out (the multipliers are
    // not guessable, so they must be visible).
    out.push(section("Navigate frames", [
      row(["←", "→"], "Previous / next frame"),
      row([hint("←", ["shift"]), hint("→", ["shift"])], "Jump 10 frames"),
      row(["PgUp", "PgDn"], "Jump 100 frames"),
      row(["↑", "↓"], "Previous / next labelled frame"),
      row(["Home", "End"], "First / last frame"),
    ]));

    // Select points -- one rule leads, then the base gestures once each. The add-modifier
    // is the app's convention (Ctrl/⌘), spelled for this OS.
    const add = MOD_GLYPH.mod;
    out.push(section("Select points", [
      row(["Click"], "Select a point"),
      row([hint("Click", ["mod"])], "Add / remove a point"),
      row(["Double-click"], "Select a point in every view"),
      row([hint("Drag", ["shift"])], "Rubber-band a new selection"),
      row([hint("Drag", ["mod"])], "Rubber-band, adding to the selection"),
      row(["a"], "Select every point (all views)"),
      row(["v"], "Select every point in the view under the cursor"),
      row(["Esc"], "Clear the selection (or click the background)"),
    ], `<b>Hold ${add} to add</b> to the current selection — with click, double-click, or drag. Without it, the gesture starts a fresh selection.`));

    // Edit the selection -- the point-move gesture, then the confirm/reset/mark bindings,
    // then the right-click GT toggle (only meaningful with a 3D solve to feed).
    const editRows = [row(["Drag a point"], "Move it — authors ground truth (grab a detected node or a reprojected point to spawn one)")]
      .concat(this.bindingRows("edit"));
    if (this.meta.has_3d) editRows.push(row(["Right-click"], "Confirm / clear a point as ground truth"));
    out.push(section("Edit the selection", editRows));

    // Rendered whenever the "cam" group has rows: multi-camera lists Grid/Focus/step + Reset
    // view; a single camera still lists Reset view (its only "cam" row), under a "View" title.
    const camRows = this.bindingRows("cam");
    if (camRows.length) out.push(section(this.meta.n_views > 1 ? "Cameras & layout" : "View", camRows));
    out.push(section("Show overlays", this.bindingRows("show")));
    out.push(section("History & file", this.bindingRows("hist")));
    out.push(section("Panels & guide", this.bindingRows("panel")));

    // The labeling guide (the interactive keypoint map) leads the panel as the one
    // actionable link; the shortcut groups and then the colour/marker legend follow. The
    // heavy external page loads only when the operator clicks through (a new tab).
    const guide = `<h3 class="legend-title">Labeling guide</h3>`
      + `<p class="legend-note">Where each keypoint sits on the fly — an interactive 3D map (opens in a new tab).</p>`
      + `<p><a class="help-link" href="${KEYPOINTS_DOC_URL}" target="_blank" rel="noopener">Open the keypoint map ↗</a></p>`;
    this.helpBody.innerHTML = guide + out.join("") + this.buildLegend();
  }

  // The legend, built from the server meta so it reflects whatever config is loaded:
  // the keypoint colours come straight from the skeleton's per-limb palette (no L/R
  // assumption), and the marker/overlay rows mirror how poseView.js draws them.
  buildLegend() {
    const esc = (s) =>
      String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
    const limbs = (this.meta.limbs || [])
      .map((lb) => {
        const [r, g, b] = lb.color;
        const name = esc(lb.name).replace(/_/g, " ");
        return `<span class="legend-limb"><i class="limb-dot" style="background:rgb(${r},${g},${b})"></i>${name}</span>`;
      })
      .join("");
    const colours = `<h3 class="legend-title">Keypoint colours</h3>`
      + `<p class="legend-note">Each keypoint takes its limb's colour from the skeleton palette (from your config).</p>`
      + `<div class="legend-limbs">${limbs}</div>`;

    // Marker vocabulary -- what a keypoint's marker tells you about where it came from.
    const markers = [
      [`<i class="mk m-gt"></i>`, `<b>Ground truth</b> — you authored it (dragged or confirmed); trusted.`],
      [`<i class="mk m-pred"></i>`, `<b>Detected</b> — the detector's raw 2D; the fill fades as confidence drops.`],
    ];
    if (this.meta.has_3d) {
      markers.push([
        `<i class="mk m-proj"></i>`,
        `<b>Projected</b> — the 3D reprojected here; no usable observation in this view (you occluded it, or the detector missed). This <i>is</i> the joint's position here: setting a point Projected drops that view from the solve, so it stops showing the rejected detection and follows the reprojection instead. The reprojected skeleton is also its own overlay: hollow rings joined by thick, dashed, semi-transparent limb-palette edges (on by default). Drag a reprojected point to spawn ground truth there. Reject a joint in so many views that fewer than two are left and there is no 3D to follow, so nothing is drawn — the joint then falls back to its faint <b>Unplaced</b> seed (<kbd>i</kbd>) in every view, still draggable, so you can always get back to it.`,
      ]);
    }
    // The "Unplaced" seed is the guarantee that no joint is ever unreachable, so it is always
    // in the vocabulary -- with or without a 3D solve (only its causes differ).
    const missingWhy = this.meta.has_3d
      ? `You see it where the point has no observation of its own and no reprojection to fall back on — triangulation dropped it, or you set it Projected in too many views — and wherever a joint's only position was a reprojection you have hidden.`
      : `You see it wherever the detector fired nothing for that joint in that view.`;
    const missingWhere = this.meta.has_3d
      ? `the reprojection if there is one, else the raw detector pixel, a nearby frame, a neighboring joint, or the image center`
      : `the raw detector pixel, a nearby frame, a neighboring joint, or the image center`;
    markers.push([
      `<i class="mk m-placeholder"></i>`,
      `<b>Unplaced</b> — a faint dashed ghost drawn wherever a view has nothing else to grab, so no joint is ever unreachable (<kbd>i</kbd>). ${missingWhy} Its spot is only a guess (${missingWhere}), so drag it to where the joint really is: that authors ground truth there like any other drag.`,
    ]);
    // Absence is a claim about the ANIMAL, not about a view, so it is described last and framed
    // against the three source states above: those answer "where did this marker come from",
    // this one answers "does this joint exist at all".
    markers.push([
      `<i class="mk m-absent"></i>`,
      `<b>Absent</b> — this keypoint is not on this animal: an amputated leg, an ablated antenna. Not the same as <b>Projected</b> ("it exists but I cannot place it from <i>this</i> view") and not the same as <b>Unplaced</b> ("nobody has placed it yet"). Select the joint and press <kbd>x</kbd> to mark it — one gesture covers <i>every frame and every view</i>, because it is one fact about the animal. It then draws as a dim grey ✕ with no bones, contributes nothing to the 3D solve, and is excluded from the training export (neither ground truth nor "occluded"). Press <kbd>x</kbd> again, or <kbd>Ctrl+Z</kbd>, to un-mark it: nothing you labeled underneath is lost.`,
    ]);
    const markerRows = markers
      .map(([m, d]) => `<div class="legend-row">${m}<span>${d}</span></div>`)
      .join("");
    const markerBlock = `<h3 class="legend-title">Point sources — where a point came from</h3>`
      + `<p class="legend-note"><b>Ground truth</b> is the editable layer — drag a point to move it, or drag a detected / projected node to create one. <b>Combined</b> merges Ground truth + Detected into one skeleton (GT where authored, else the detector's point, else the reprojection — a view marked Projected shows no detection, since you rejected it); turn it off to see them as two separate skeletons. <b>Projected</b> stays its own overlay.</p>`
      + `<div class="legend-rows">${markerRows}</div>`;

    // Read-only reference overlays (shown only when the result carries them). The projected
    // source is documented in the marker block above; here we cover the NMF model overlay.
    const refs = [];
    if (this.meta.has_nmf) {
      refs.push([
        `<span class="swatch swatch-nmf"></span>`,
        `<b>NMF skeleton</b> — the fitted NeuroMechFly model joints (dotted mint).`,
      ]);
    }
    const refBlock = refs.length
      ? `<h3 class="legend-title">Reference overlays</h3>`
        + `<div class="legend-rows">`
        + refs.map(([m, d]) => `<div class="legend-row">${m}<span>${d}</span></div>`).join("")
        + `</div>`
      : "";

    return `<div class="legend">${colours}${markerBlock}${refBlock}</div>`;
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

  // Rewrite the modifier spellings baked into the static HTML (toolbar/sidebar tooltips
  // and the ⇧M chip) so every on-screen surface uses the SAME OS-correct keys as the help
  // panel -- no "Ctrl/Cmd" on a Mac in one place and "⌘" in another. Runs once at init,
  // after meta is known (the tooltip text is meta-conditional).
  applyOsHints() {
    const add = MOD_GLYPH.mod;
    const setTitle = (/** @type {string} */ id, /** @type {string} */ title) => {
      const e = document.getElementById(id);
      if (e) e.title = title;
    };
    setTitle("undo", `Undo (${hint("Z", ["mod"])})`);
    setTitle("redo", `Redo (${IS_MAC ? hint("Z", ["mod", "shift"]) : hint("Y", ["mod"])})`);
    setTitle("save", `Save the ground-truth labels (${hint("S", ["mod"])})`);
    // The frame row advertises the whole navigation ladder.
    const frameLabel = document.querySelector("#controls .frame-row label");
    if (frameLabel instanceof HTMLElement) {
      frameLabel.title =
        `Jump to a frame — ← / → step 1 · ${hint("←", ["shift"])} steps 10 · PgUp / PgDn jump 100 · ↑ / ↓ step the side panel's list · Home / End first / last`;
    }
    // The sidebar's frame nav echoes its keys, naming whichever tab's list it steps.
    this.updateSidebarNavTitles();
    // Overlay toggles: the "Show" button's summary and the NMF-mesh chip.
    const mesh = hint("M", ["shift"]);
    setTitle(
      "show-toggle",
      `Show / hide the view layers (keyboard: h Hide all · s Combined · n Names${this.meta.has_3d ? " · p Reprojected · w Reproj. warning" : ""}${this.meta.has_nmf ? ` · m NMF skeleton · ${mesh} NMF mesh` : ""})`,
    );
    const meshChip = document.querySelector("#mesh-wrap kbd");
    if (meshChip) meshChip.textContent = mesh;
    // The live "adding" pill's key glyph (⌘ on macOS, Ctrl elsewhere).
    const addKey = document.getElementById("add-hint-key");
    if (addKey) addKey.textContent = add;
    setTitle("mesh-wrap", `Overlay the fitted NeuroMechFly mesh, rendered on the GPU onto each view (${mesh})`);
    // The selection status card's how-to, re-spelled for this OS and this model.
    setTitle(
      "point-status",
      `The selected point(s): click a chip to set the whole selection's state — Ground truth (1), Detected (2), or Projected (3) — or the Reset button (r) to clear the labels back to the detector. Select with click, ${add}-click to add/remove, double-click (all views), Shift-drag (new region), ${add}-drag to add; a = all, v = this view. Click the background or Esc to clear. Hover a point to peek at its name + state.`,
    );
  }

  // Reflect the live "adding to selection" state on <body> (see the `addMod`/`overViews`
  // fields): only when the add-modifier is held AND the pointer is over the views, so the
  // affordance appears exactly when a click/drag would add rather than replace.
  updateAddingHint() {
    document.body.classList.toggle("adding", this.addMod && this.overViews);
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
      if (this.showMenuOpen || this.layoutMenuOpen) {
        this.closeShowMenu();
        this.closeLayoutMenu();
        e.preventDefault();
      } else if (this.closeConfirmOpen) {
        this.closeCloseConfirm();
        e.preventDefault();
      } else if (this.helpOpen) {
        this.closeHelp();
        e.preventDefault();
      } else if (this.sceneOpen) {
        this.closeScene();
        e.preventDefault();
      } else if (this.selection.size) {
        this.clearSelection();
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
