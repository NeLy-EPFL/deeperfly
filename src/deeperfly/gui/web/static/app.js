// @ts-check
// The editor controller: lays out one PoseView per camera and routes edits to
// the server. There is one unified editing model -- a drag authors ground-truth 2D at
// the drop and, when the result carries 3D, re-solves the 3D point live and refreshes
// every view (pinning the dragged view on release).
//
// There is one annotation skeleton per frame -- an "instance" -- created by dragging a joint,
// double-clicking one, or pressing `g`, and seeded from the detections. Each of its joints is
// either the operator's own pixel (a lime ring over a filled disc) or derived from the joints
// they have placed in other views (a hollow palette circle); a joint nothing in the frame can
// place is drawn faint and dashed, because its position is a guess the editor made rather than
// anything the geometry produced. The detections themselves stay as a read-only reference layer
// that hides itself once a skeleton exists.
//
// Annotation is two steps: build a selection of (point, view) cells, then set a fact on it.
// The facts are orthogonal and each has its own toggle, pressed when set: GT (Enter / Backspace
// place and clear) says WHERE the joint is; Hidden (`e`) says whether the cell is INCLUDED IN THE
// TRAINING LOSS and nothing else -- it moves no joint, hides no marker, changes no 3D and blocks
// no verb; and Absent (`x`, "not on this animal") covers every frame and view at once. Reset (`r`)
// retracts them. The toggles ARE the readout -- two of them can be pressed at once, which is a
// state no single status line could report.
//
//
// Two layouts share the same PoseView instances. "grid" shows every camera in an
// equal grid; "focus" shows one large editable view plus a strip of live,
// clickable thumbnails (the other cameras) -- which keeps each camera big enough
// to correct precisely when there are many cameras. The large view(s) can be
// zoomed (wheel) and panned (drag on empty space); thumbnails always show the
// whole frame. Every view stays live in both layouts, so a 3D re-solve still
// animates the thumbnails. Grid is the default; the layout switch (or l) toggles
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
// or spawns GT), the raw-detection reference layer (auto-hidden once an annotation skeleton
// exists), "Seed positions" (draw a non-GT joint where the skeleton started rather than at the
// reprojection of its 3D), per-joint name labels, and the "projected" 3D
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

import { BundleAdjustPanel } from "./baPanel.js";
import { EditSocket, cancelJob, configSchema, configValues, fetchCorrected, fetchMeta, fetchNmfAsset, fetchNmfVerts, fetchPoints, fetchRecordings, fetchScene, fetchSuggestions, frameUrl, jobs as fetchJobs, openRecording, saveAllCorrections, setConfig, shutdownServer, submitJob } from "./api.js";
import { MeshGL } from "./meshGL.js";
import { PoseView } from "./poseView.js";
import { Scene3D } from "./scene3d.js";

/** @typedef {import("./types.js").Meta} Meta */
/** @typedef {import("./types.js").PointsPayload} PointsPayload */
/** @typedef {import("./types.js").CorrectedFrame} CorrectedFrame */
/** @typedef {import("./types.js").Suggestion} Suggestion */
/** @typedef {import("./types.js").SuggestionsPayload} SuggestionsPayload */
/** @typedef {"recordings"|"labeled"|"suggest"|"instances"|"marks"|"bundle"|"jobs"|"settings"} TabId */

// The side panel's tabs, in strip order -- one pane on screen at a time. Activating a tab
// IS the lazy-load trigger, which is what keeps the three expensive panes free until asked
// for: Recording opens every recording's labels.h5, Settings re-composes the project config
// and parses two TOML documents per read, Jobs starts a two-second poll. The panes that
// render from `meta` alone have nothing to run, so they are absent from `tabActivated`.
const SIDEBAR_TABS = [
  { id: "recordings", pane: "recording-pane" },
  { id: "labeled", pane: "labeled-pane" },
  { id: "suggest", pane: "suggest-pane" },
  { id: "instances", pane: "instances-pane" },
  { id: "marks", pane: "marks-pane" },
  { id: "jobs", pane: "jobs-pane" },
  { id: "bundle", pane: "ba-pane" },
  { id: "settings", pane: "settings-pane" },
];
/** @type {TabId} */
const SIDEBAR_DEFAULT_TAB = "labeled";
const SIDEBAR_TAB_KEY = "deeperfly.sidebar.tab";
const SIDEBAR_OPEN_KEY = "deeperfly.sidebar.open";
const SIDEBAR_DEFAULT_OPEN = true;
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

// ... and for its sibling check, the label-coverage gauge (poseView.js drawGtCoverage): whether
// it is shown, and how many GT views a keypoint must have before it stops being flagged. Both
// persist for the same reason the warning's do -- they are the operator's standing audit
// settings, not a per-frame choice. Off by default; `u` and the Show menu turn it on.
const COVER_ON_KEY = "deeperfly.cover.enabled";
const COVER_MIN_KEY = "deeperfly.cover.min";
// Two is the geometric floor (one pixel fixes a viewing ray, not a point) and the annotation
// solve's own bar -- `min_gt_for_exclusive`, config default 2, is where GT alone determines a
// point's 3D. The ceiling is the camera count, applied from /api/meta at build time: asking for
// more views than the rig has would flag every joint of every frame forever.
const COVER_MIN_DEFAULT = 2;
const COVER_MIN_FLOOR = 1;

// How long after a navigation the editor warms the frame it expects to be asked for next
// (see `schedulePrefetch`). Long enough that a run of held-down arrow keys does not fire a
// prefetch per keystroke -- each one supersedes the last -- and short enough that a
// deliberate step has its successor waiting before the operator asks for it.
const PREFETCH_DELAY_MS = 150;

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
  // (the GT / Hidden / Absent toggles + Reset) acts on, keyed
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
  // disturbing the selection (the toggles still act on the selection, not the hover).
  /** @type {{ view: number, point: number } | null} */
  hoverCell = null;
  // The latest per-view ground-truth mask and "projected" mask (from the points payload),
  // so the status widget can report each selected joint's source. `projectedMask` marks a
  // cell with no observed pixel here -- the detector never fired -- whose drawn position
  // follows the 3D reprojection. Null until the first payload.
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
  //: Whether the current frame carries an annotation skeleton. Drives the auto-hide of the
  //: detected layer and what a double-click on a joint means.
  hasInstance = false;
  detectedMask = null;
  //: The operator's "exclude this detection from triangulation" mask, per (view, point).
  //: Narrower than `projectedMask`, which is every cell with no position of its own --
  //: the facts readout has to say "excluded" only where the operator actually said so.
  excludedMask = null;
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
  //: The project's OTHER recordings holding unsaved labels (never the open one -- that is
  //: `dirty`, which every edit reply refreshes). The server keeps every recording it has
  //: opened, so switching loses nothing and unsaved work is a property of the *project*:
  //: this is what the close prompt and the beforeunload guard read, and what the unsaved
  //: cue counts. Kept as "the others" rather than the whole set because the whole set
  //: would go one stale on every edit -- `dirty` moves per keystroke, this only on a
  //: switch or a save.
  /** @type {Set<string>} */
  dirtyOthers = new Set();
  //: Bumped by every recording rebuild, and checked after every `await` in the refresh*
  //: methods -- there is no AbortController anywhere in this codebase. Without it, an
  //: /api/points fetch issued against the OLD session resolves after the swap and
  //: applyPoints' only defense (its frame guard) passes, because the rebuild navigates to
  //: frame 0 and the stale reply is for frame 0.
  epoch = 0;
  //: The in-flight rebuild, or null. The server broadcasts the reload to every socket
  //: INCLUDING the one that asked, so the switching tab enters twice.
  /** @type {Promise<void> | null} */
  rebuilding = null;
  // The corrected-frames side panel: the list (sorted, each with a reviewed flag),
  // whether the panel is open, the row elements keyed by frame (for the current-frame
  // highlight), and a debounce timer coalescing post-edit refreshes.
  /** @type {CorrectedFrame[]} */
  correctedFrames = [];
  framesOpen = false;
  /** @type {Map<number, HTMLTableRowElement>} */
  frameRows = new Map();
  correctedTimer = 0;
  // The frame warm-up (see `schedulePrefetch`): the pending timer, and the images being
  // warmed. The images are held only so a garbage collector cannot cancel a load that
  // nothing else references yet; each pass replaces the previous array, so at most the
  // last warm-up's worth is pinned.
  prefetchTimer = 0;
  /** @type {HTMLImageElement[]} */
  prefetchImgs = [];
  // The side panel's tabs. "labeled" is the frames-with-GT list; "suggest" is the ranked
  // queue from `deeperfly labels-suggest` (a static sidecar, so it is fetched on load / on
  // save / on tab activation, never per edit). `suggestions` is null until the first fetch
  // resolves and stays null when no queue has been computed -- the pane then explains how
  // to make one instead of looking broken. Which frames are DONE is not read from the
  // sidecar but joined live from `correctedFrames` at render time, so a row flips the moment
  // its frame is labeled, with no refetch.
  /** @type {Map<TabId, {id: TabId, pane: string, tab: HTMLButtonElement, body: HTMLElement}>} */
  tabs = new Map();
  /** @type {TabId} which pane is showing. Persisted, and preserved across a recording
   *  switch -- it is the operator's arrangement, not the recording's data. */
  sidebarTab = SIDEBAR_DEFAULT_TAB;
  //: Which list the up/down KEYS step. Normally the list tab on screen -- but from the
  //: Settings or Jobs tab there is none, so it REMEMBERS the last one the operator used
  //: rather than going dead, and the nav buttons' tooltips name whichever it is.
  /** @type {"labeled" | "suggest"} */
  navList = "labeled";
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
  /** @type {HTMLInputElement} */
  hideAllCheck = el("show-hide-all");
  /** @type {HTMLInputElement} */
  autoHideCheck = el("detected-autohide");
  reviewedBtn = el("reviewed-toggle");
  //: Whether the current frame is marked reviewed, mirrored from the payload.
  reviewed = false;
  // Both live in the Skeleton popover, and exactly one of them is ever enabled: the server
  // refuses Create on a frame that already has a skeleton and Reseed on one that does not.
  /** @type {HTMLButtonElement} */
  skeletonCreateBtn = el("skeleton-create");
  /** @type {HTMLButtonElement} */
  reseedBtn = el("reseed");
  skeletonToggle = el("skeleton-toggle");
  skeletonMenu = el("skeleton-menu");
  skeletonMenuOpen = false;
  //: How a new skeleton is seeded. Server-side state; mirrored here for the switch.
  seedMode = "triangulate";
  // How a point's 3D is derived once its GT views are exclusive. A camera cannot see
  // distance along its own optical axis, so two GT views facing each other leave that one
  // direction nearly free -- "on" lets the unlabeled views fix it (and only it), "off" is
  // the older solve-from-my-pixels-alone. Server-side state, mirrored here for the switch.
  //: "on" (the default) or "off": whether unlabeled views help fix the depth GT cannot.
  solveStabilizers = "on";
  /** @type {HTMLInputElement} */
  labelsCheck = el("show-labels");
  /** @type {HTMLLabelElement} */
  /** @type {HTMLInputElement} */
  /** @type {HTMLLabelElement} */
  detectedWrap = el("detected-wrap");
  //: The row's own title, kept so the auto-hide note can be swapped in and back out.
  _detectedTitle = "";
  /** @type {HTMLInputElement} */
  detectedCheck = el("show-detected");
  /** @type {HTMLLabelElement} */
  projectedWrap = el("projected-wrap");
  /** @type {HTMLInputElement} */
  projectedCheck = el("show-projected");
  /** @type {HTMLLabelElement} */
  /** @type {HTMLInputElement} */
  /** @type {HTMLDivElement} */
  checksSection = el("checks-section");
  /** @type {HTMLLabelElement} */
  warnWrap = el("warn-wrap");
  /** @type {HTMLInputElement} */
  warnCheck = el("show-warn");
  /** @type {HTMLLabelElement} */
  warnThresholdWrap = el("warn-threshold-wrap");
  /** @type {HTMLInputElement} */
  warnThresholdInput = el("warn-threshold");
  /** @type {HTMLLabelElement} */
  coverWrap = el("cover-wrap");
  /** @type {HTMLInputElement} */
  coverCheck = el("show-cover");
  /** @type {HTMLLabelElement} */
  coverMinWrap = el("cover-min-wrap");
  /** @type {HTMLInputElement} */
  coverMinInput = el("cover-min");
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
  gtBtn = el("act-gt");
  excludeBtn = el("act-exclude");
  absentBtn = el("act-absent");
  absentBadge = el("absent-badge");
  /** @type {Segmented} */
  /** @type {Segmented} */
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
  /** @type {HTMLDivElement} */
  /** @type {HTMLButtonElement} */
  /** @type {HTMLDivElement} */
  /** @type {HTMLSpanElement} */
  recordingNameEl = el("recording-name");
  /** @type {HTMLDivElement} */
  recordingListEl = el("recording-list");
  /** @type {HTMLDivElement} */
  recordingEmptyEl = el("recording-empty");
  //: The last /api/recordings payload, kept so a role change can re-render the rows
  //: (a read-only tab may not switch) without another round trip.
  /** @type {any} */
  recordings = null;
  /** @type {HTMLButtonElement} */
  /** @type {HTMLDivElement} */
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
  //: The closed panel's re-open handle on the right edge. Shown only while the panel is
  //: hidden, so it and the panel's own ✕ are never on screen at the same time.
  /** @type {HTMLButtonElement} */
  sidebarRailBtn = el("sidebar-rail");
  /** @type {HTMLSpanElement} */
  framesCountEl = el("frames-count");
  /** @type {HTMLElement} */
  sidebarEl = el("sidebar");
  /** @type {HTMLButtonElement} */
  framesCollapseBtn = el("frames-collapse");
  //: The Recording tab, hidden for a bare results.h5 session with no project to list.
  /** @type {HTMLButtonElement} */
  recordingsTab = el("tab-recordings");
  /** @type {HTMLSpanElement} */
  labeledCountEl = el("labeled-count");
  /** @type {HTMLSpanElement} */
  suggestTallyEl = el("suggest-tally");
  /** @type {HTMLSpanElement} */
  marksCountEl = el("marks-count");
  /** @type {HTMLSpanElement} */
  jobsCountEl = el("jobs-count");
  /** @type {HTMLSpanElement} */
  instanceStateEl = el("instance-state");
  /** @type {HTMLDivElement} */
  instancesList = el("instances-list");
  /** @type {HTMLTableSectionElement} */
  framesTbody = /** @type {HTMLTableElement} */ (el("frames-table")).tBodies[0];
  /** @type {HTMLDivElement} */
  framesEmptyEl = el("frames-empty");
  /** @type {HTMLDivElement} */
  labeledPane = el("labeled-pane");
  /** @type {HTMLDivElement} */
  suggestPane = el("suggest-pane");
  /** @type {HTMLDivElement} */
  jobsPane = el("jobs-pane");
  /** @type {HTMLDivElement} */
  marksPane = el("marks-pane");
  /** @type {HTMLDivElement} */
  marksList = el("marks-list");
  /** @type {HTMLDivElement} */
  marksEmpty = el("marks-empty");
  /** @type {HTMLDivElement} */
  settingsPane = el("settings-pane");
  /** @type {HTMLDivElement} */
  settingsList = el("settings-list");
  /** @type {HTMLDivElement} */
  settingsEmpty = el("settings-empty");
  // Cached so switching tabs does not refetch the schema (it is static for the session).
  configSchemaCache = null;
  /** @type {HTMLDivElement} */
  jobsActions = el("jobs-actions");
  /** @type {HTMLDivElement} */
  jobsList = el("jobs-list");
  /** @type {HTMLDivElement} */
  jobsEmpty = el("jobs-empty");
  // Poll handle for the jobs panel. Polled rather than pushed over /ws: that socket
  // carries the single-writer EDITING stream, and a read-only tab must still see the
  // queue. A 2 s poll of a few JSON rows is cheaper than the alternative.
  jobsTimer = null;
  // Index of the armed calibration landmark (-1 = none). Armed from the Landmarks tab;
  // while armed, a click on any view PLACES it rather than selecting a joint.
  armedLandmark = -1;
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
  /** @type {HTMLSpanElement} the close prompt's list of recordings with unsaved labels */
  closeListEl = el("close-list");
  /** @type {HTMLButtonElement} the toolbar's unsaved-changes cue (hidden while clean) */
  unsavedChip = el("unsaved");
  /** @type {HTMLDivElement} */
  stoppedOverlay = el("stopped-overlay");
  /** @type {HTMLDivElement} */
  readonlyBanner = el("readonly-banner");
  /** @type {HTMLButtonElement} */
  readonlyTakeover = el("readonly-takeover");
  /** @type {HTMLDivElement} */
  uncalBanner = el("uncal-banner");
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
    // A reload lands on a server that may already be holding OTHER recordings' unsaved
    // labels (this page is not where they were made). Without this the fresh page would
    // report itself clean and let the operator close the editor on them.
    this.applyDirtyRecordings(this.meta.dirty_recordings);
    // Grid is the default (`layout` is initialised to it); focus stays a click / l away.
    this.bindings = this.buildBindings();
    this.buildControls();
    this.buildSidebar();
    this.applyOsHints();
    this.buildViews();
    // Sync the (persisted) reprojection-warning state into the freshly-built views. With no saved
    // override this matches their constructor defaults, so the setters early-return -- no extra draw.
    this.applyWarn();
    this.applyWarnThreshold();
    this.applyCover();
    this.applyCoverMin();
    this.relayout();
    this.socket = new EditSocket(
      (p) => this.applyPoints(p, true),
      (r) => this.applyRole(r),
      () => this.rebuildForNewRecording(),
    );
    await this.goToFrame(0);
    this.renderInstance();
    // Reseed starts disabled on a frame with no skeleton. applyPoints only re-syncs when
    // `has_instance` CHANGES, and frame 0 usually has none -- so nothing there would have
    // disabled it, and it would offer an act the server is about to refuse.
    this.syncSkeletonMenu();
    this.updateSelected();
    this.updateDirty();
    await this.refreshCorrected(); // populate the list (any corrections loaded from disk)
    this.refreshSuggestions(); // the ranked queue, if one has been computed (not awaited)
    // Closing instantly when there is nothing to lose, prompting otherwise: the
    // browser shows its generic "leave site?" dialog only while edits are unsaved.
    window.addEventListener("beforeunload", (e) => {
      if (this.projectDirty && !this.closing) {
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

  // Everything in the editor's chrome that is DERIVED from /api/meta: ranges, names, and
  // which controls exist at all. Split out of `buildControls` so a recording switch can
  // re-apply it WITHOUT re-running the one-time wiring -- which would double every
  // listener (two edits per click on undo/GT/Absent) and append a second copy of every
  // segmented switch, still wired to `this`. Every statement here must be idempotent.
  applyMeta() {
    const last = Math.max(0, this.meta.n_frames - 1);
    // `.max` before `.value`: a range input clamps its value to its max, so the other
    // order silently pins a longer recording to the previous one's last frame.
    for (const input of [this.slider, this.number]) {
      input.min = "0";
      input.max = String(last);
      input.value = "0";
    }
    this.totalEl.textContent = `/ ${last}`;

    // A single camera has no arrangement to choose, so only the Grid/Focus segment is
    // hidden -- the Layout menu itself stays, since it also holds "Reset view", which one
    // (still zoomable) camera can use too. It also registers no `l` binding, so a session
    // left in focus would be stuck there with no control that could get it out.
    const multiCam = this.meta.n_views > 1;
    this.layoutArrangeSection.style.display = multiCam ? "" : "none";
    this.layoutArrangeRow.style.display = multiCam ? "" : "none";
    if (!multiCam) {
      this.layout = "grid";
      this.focused = 0;
    }
    this.layoutSwitch.set(this.layout);

    // Which layers can exist at all. `has_cameras` is absent on older servers, where its
    // absence means "calibrated" -- the prior behavior.
    this.uncalBanner.hidden = this.meta.has_cameras !== false;
    this.projectedWrap.style.display = this.meta.has_3d ? "" : "none";
    const warnAvailable = this.meta.has_3d;
    this.warnWrap.style.display = warnAvailable ? "" : "none";
    this.warnThresholdWrap.style.display = warnAvailable ? "" : "none";
    // The coverage check counts labeled views, so it needs cameras to count over and nothing else
    // -- no rig, no detections, no 3D. It is therefore available on a fresh uncalibrated project,
    // where it is the only check there is, and unavailable on a single camera, where "two views"
    // is a requirement no amount of labeling could ever meet. Its ceiling follows the rig.
    const coverAvailable = this.meta.n_views > 1;
    this.coverWrap.style.display = coverAvailable ? "" : "none";
    this.coverMinWrap.style.display = coverAvailable ? "" : "none";
    this.coverMinInput.max = String(this.coverCeiling);
    this.coverMinInput.value = String(this.clampCoverMin());
    // The section heading spans both checks, so it survives as long as either one does.
    this.checksSection.style.display = warnAvailable || coverAvailable ? "" : "none";
    this.referenceSection.style.display = this.meta.has_nmf ? "" : "none";
    this.nmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.meshWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.sceneNmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.sceneMeshWrap.style.display = this.meta.has_nmf ? "" : "none";

    // Server-mirrored editor settings: a fresh EditorState reverts both server-side, so
    // the switches have to follow the reset fields rather than keep the old choice.
    this.nongtSwitch.set(this.nongtDisplay);
    this.seedSwitch.set(this.seedMode);
    this.stabilizeSwitch.set(this.solveStabilizers);
    // No rig means no triangulation to choose between: every view is an independent 2D
    // canvas. Hidden rather than disabled, following the other geometry-dependent rows.
    const noRig = this.meta.has_cameras === false;
    el("stabilize-section").style.display = noRig ? "none" : "";
    el("stabilize-row").style.display = noRig ? "none" : "";
    // Pin the name readout to its widest possible value so hovering / selecting different
    // joints never reflows the widget -- the widest value depends on the point names.
    this.reserveStatusNameWidth();

    // Which recording is open. This is the loudest stale-data bug a forgotten re-apply
    // could produce, so it lives here rather than at any of its call sites.
    this.recordingsTab.hidden = !this.meta.project_root;
    this.recordingNameEl.textContent = this.meta.recording ?? "recording";
    // A session with no project has no Recording tab to show. `applyMeta` runs before
    // `buildSidebar` at boot (so `tabs` is still empty and the restore does its own check),
    // but it also runs on every switch, where leaving the active tab pointing at a hidden
    // chip would strand the panel on a blank pane.
    if (this.recordingsTab.hidden && this.tabs.size && this.tabActive("recordings")) {
      this.setSidebarTab(SIDEBAR_DEFAULT_TAB);
    }
  }

  buildControls() {
    this.slider.addEventListener("input", () => this.goToFrame(Number(this.slider.value)));
    this.number.addEventListener("change", () => this.goToFrame(Number(this.number.value)));

    this.layoutSwitch = segmented(
      [["Grid", "grid"], ["Focus", "focus"]],
      (v) => this.setLayout(/** @type {Layout} */ (v))
    );
    el("layout-switch").append(this.layoutSwitch.root);

    // No state control here, deliberately. A cell does not HAVE a state you assign; it has
    // an authored pixel or it does not, and its detection is excluded from triangulation or
    // it is not. `#point-status-facts` reports those facts and the buttons beside it are the
    // verbs -- create GT from what is shown, delete GT, toggle the exclusion.
    this.gtBtn.addEventListener("click", () => this.toggleSelectionGt());
    this.excludeBtn.addEventListener("click", () => this.toggleSelectionExclude());
    this.absentBtn.addEventListener("click", (e) =>
      this.toggleAbsentSelection(e.shiftKey ? "recording" : "frame"),
    );

    // The side panel's tab strip is static markup wired in `buildSidebar`, not built here:
    // its chips carry per-pane counts and one is hidden for a project-less session, neither
    // of which the `.segmented` component does.
    this.hideAllCheck.addEventListener("change", () => this.applyHideAll());
    this._detectedTitle = this.detectedWrap.title;
    this.autoHideCheck.addEventListener("change", () => this.applyDetected());
    // Where an unplaced joint is drawn -- a MODE of the one skeleton, not a layer, which is
    // why it is a two-value switch nested under it rather than a checkbox among the layers.
    this.nongtSwitch = segmented(
      [["3D reprojection", "reprojection"], ["seed", "seed"]],
      (v) => this.setNongtDisplay(v),
    );
    el("nongt-switch").append(this.nongtSwitch.root);
    this.seedSwitch = segmented(
      [["Triangulated", "triangulate"], ["Each view's own", "copy"]],
      (v) => this.setSeedMode(v),
    );
    el("seed-switch").append(this.seedSwitch.root);
    // How the 3D is derived once your labels are exclusive. A solve setting, not a display
    // one, so it belongs beside the seeding mode rather than in Show: both answer "where do
    // the numbers come from", and neither changes what is drawn on top of them.
    this.stabilizeSwitch = segmented(
      [["My pixels + other views", "on"], ["My pixels only", "off"]],
      (v) => this.setSolveStabilizers(v),
    );
    el("stabilize-switch").append(this.stabilizeSwitch.root);
    this.skeletonCreateBtn.addEventListener("click", () => {
      this.createInstance();
      this.closeSkeletonMenu();
    });
    this.reseedBtn.addEventListener("click", () => {
      this.reseedInstance();
      this.closeSkeletonMenu();
    });
    this.skeletonToggle.addEventListener("click", () => this.toggleSkeletonMenu());
    this.reviewedBtn.addEventListener("click", () => this.toggleReviewedCurrent());
    this.labelsCheck.addEventListener("change", () => this.applyLabels());
    // The ground-truth and detected source layers exist without 3D (they are the authored
    // pixels and the raw detector output); only the projected source needs a 3D solve.
    this.detectedCheck.addEventListener("change", () => this.applyDetected());
    // No solved rig -> say so, once, at the top. `has_cameras` is distinct from
    // `has_3d`: a calibrated recording whose triangulation stage has not run also has no
    // 3D, and that is a "run the pipeline" state, not an "uncalibrated project" one.
    // Older servers omit the field, so absence means "calibrated" (the prior behavior).
    this.projectedCheck.addEventListener("change", () => this.applyProjected());
    // The "Unplaced" placeholder seeds are the guarantee that no joint is ever unreachable: a cell
    // with nothing else drawn still gets a faint ghost to drag into a GT label (the authored 2D
    // needs no prior 3D), so the layer is available with or without a 3D solve.
    // The reprojection-distance warning flags joints whose asserted pixel -- the operator's GT,
    // else what the skeleton draws there (poseView.js ``warnAnchor``) -- is far from the 3D
    // reprojection; only meaningful with a 3D solve, so it shares the projected row's has_3d gate.
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
    // The label-coverage check: which keypoints you have not yet labeled in enough views of this
    // frame. Restored from the same kind of persisted preference, but OFF unless the operator
    // turned it on -- it answers "what is left here?", and an editor that opens with a ring on
    // every joint has buried the skeleton under its own to-do list.
    if (localStorage.getItem(COVER_ON_KEY) === "1") this.coverCheck.checked = true;
    const savedMin = Number(localStorage.getItem(COVER_MIN_KEY));
    if (Number.isFinite(savedMin) && savedMin > 0) {
      this.coverMinInput.value = String(Math.round(savedMin));
    }
    this.coverCheck.addEventListener("change", () => this.applyCover());
    this.coverMinInput.addEventListener("input", () => this.applyCoverMin());
    this.coverMinInput.addEventListener("change", () => {
      this.coverMinInput.value = String(this.clampCoverMin());
      this.applyCoverMin();
    });
    // The NMF overlay is the fitted inverse-kinematics model -- only when present. The
    // "Reference" section heading is hidden with it, so it never dangles over no rows.
    this.nmfCheck.addEventListener("change", () => this.applyNmf());
    // The NMF mesh overlay (rendered on the client GPU) -- only when a fitted model
    // is present. The head/abdomen size is estimated from the data by the IK stage
    // (no operator knob), so the overlay just follows the model.
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
    // The unsaved cue is a button, not a label: the operator's answer to "something is
    // unsaved" is almost always "then save it", and the alternative -- reaching for the
    // Save button two controls along -- makes the cue a thing you read past.
    this.unsavedChip.addEventListener("click", () => this.save());
    this.resetViewBtn.addEventListener("click", () => this.resetView());
    this.camerasBtn.addEventListener("click", () => this.toggleScene());
    this.helpBtn.addEventListener("click", () => this.toggleHelp());
    this.helpClose.addEventListener("click", () => this.closeHelp());
    this.sceneClose.addEventListener("click", () => this.closeScene());
    this.initSceneDrag();
    // The 3D-view layer toggles; the NMF layers only exist when a model was fit.
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
    // Last: the segmented switches above must exist before applyMeta sets them.
    this.applyMeta();
  }

  buildViews() {
    const cols = Math.max(1, Math.ceil(Math.sqrt(this.meta.n_views)));
    this.viewsEl.style.setProperty("--cols", String(cols));
    /** @type {import("./poseView.js").PoseViewCallbacks} */
    const cb = {
      onDragging: (v, p, x, y) => this.onDragging(v, p, x, y),
      onDragged: (v, p, x, y) => this.onDragged(v, p, x, y),
      onToggleFixed: (v, p) => this.onToggleFixed(v, p),
      onSelect: (v, p, additive) => this.onSelect(v, p, additive),
      onSelectRegion: (v, points, additive) => this.onSelectRegion(v, points, additive),
      onSelectKeypointAllViews: (p, additive) => this.onSelectKeypointAllViews(p, additive),
      onBackground: () => this.clearSelection(),
      onActiveView: (v) => this.onActiveView(v),
      onHover: (p) => this.onHover(p),
      onPlaceLandmark: (v, i, x, y) => this.onPlaceLandmark(v, i, x, y),
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
      view.setLandmarkNames((this.meta.landmarks || []).map((l) => l.name));
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

  // One key flips the arrangement, rather than one key per layout: with only two of them
  // the operator never has to recall which one they are already in. The Layout menu's
  // segment stays the explicit picker (and `[` / `]` still jump straight into focus).
  toggleLayout() {
    this.setLayout(this.layout === "grid" ? "focus" : "grid");
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
    this.syncSkeletonMenu(); // neither verb may author from a read-only tab
    this.renderFrameList(); // re-render so the reviewed checkboxes track editability
    // A demoted tab must not be able to switch the recording out from under the writer,
    // and a promoted one should stop saying it cannot.
    if (this.recordings) this.renderRecordings(this.recordings);
  }

  // -- frame navigation -------------------------------------------------------

  /**
   * @param {number} t
   * @param {NavHint} [hint]  what the caller expects to be asked for next, so the frames
   *   can be warmed while the operator looks at this one. Omitted by navigations with no
   *   next frame to guess (a slider scrub, a click on a list row): those are either
   *   already saturating the connection or are a one-off jump.
   */
  async goToFrame(t, hint) {
    const last = Math.max(0, this.meta.n_frames - 1);
    t = Math.max(0, Math.min(Math.round(t), last));
    this.frame = t;
    this.slider.value = String(t);
    this.number.value = String(t);
    this.updateActiveFrameRow();
    this.meta.camera_names.forEach((name, v) => {
      this.views[v].loadFrame(frameUrl(name, t));
    });
    this.schedulePrefetch(hint);
    this.scheduleMeshRefresh();
    await this.refreshPoints();
    if (this.sceneOpen) {
      this.refreshScenePoints(); // snappy skeleton scrub
      this.scheduleSceneMesh(); // mesh catches up once the scrub settles
    }
  }

  // Warm the frames the operator is about to ask for, in every view, once this navigation
  // has settled. A frame URL stamped with the session token is served `immutable`, so a
  // warmed frame is a memory-cache hit when `loadFrame` asks for it -- no second request,
  // no decode, nothing to invalidate. Stepping is where this pays: the server answers a
  // step-forward for a whole rig in ~5 ms (each camera's decoder is already sitting on that
  // frame), so the picture is decoded and cached well before the keystroke that wants it.
  //
  // Deliberately narrow. Only a caller that knows what comes next passes a hint, so a
  // slider scrub -- already one request per camera per pointer move, in flight -- adds
  // nothing to the queue. The timer coalesces a run of keystrokes into one warm-up, and
  // the epoch check keeps a recording switch from warming the previous animal's frames.
  /** @param {NavHint} [hint] */
  schedulePrefetch(hint) {
    clearTimeout(this.prefetchTimer);
    if (!hint) return;
    const last = Math.max(0, this.meta.n_frames - 1);
    /** @type {number[]} */
    const targets = [];
    for (const t of [hint.step ? this.frame + hint.step : null, hint.then ?? null]) {
      if (t != null && t >= 0 && t <= last && t !== this.frame && !targets.includes(t))
        targets.push(t);
    }
    if (targets.length === 0) return;
    const epoch = this.epoch;
    this.prefetchTimer = setTimeout(() => {
      if (epoch !== this.epoch) return;
      this.prefetchImgs = targets.flatMap((t) =>
        this.meta.camera_names.map((name) => {
          const img = new Image();
          img.src = frameUrl(name, t);
          return img;
        }),
      );
    }, PREFETCH_DELAY_MS);
  }

  async refreshPoints() {
    // Fetch verbose so the reply carries `pred` (the raw detections) for the Detected
    // source layer. Detections are static within a frame, so this rides the navigation
    // fetch only -- the mid-drag edit stream stays lean (no `pred`), and each view keeps
    // the detections it already has.
    const epoch = this.epoch;
    const payload = await fetchPoints(this.frame, this.mode, true);
    // A recording switch landed while this was in flight: the reply describes the
    // recording just closed, and the rebuild has already navigated to frame 0 -- which is
    // exactly the frame this reply is for, so applyPoints' frame guard would let it past.
    if (epoch !== this.epoch) return;
    this.applyPoints(payload);
  }

  /** A one-line, self-dismissing status notice. Non-blocking on purpose: these are
   * refusals and confirmations, not decisions, and a modal would interrupt a labeling pass
   * for something the operator can simply read.
   * @param {string} msg */
  flash(msg) {
    this.statusEl.textContent = msg;
    clearTimeout(this._noticeTimer);
    this._noticeTimer = setTimeout(() => (this.statusEl.textContent = ""), 4000);
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
    if (p.notice) this.flash(p.notice);
    // The per-view masks drive the source-styled markers (ground truth / projected) and
    // the status widget; they are meaningful whether or not the result carries 3D.
    this.fixedMask = p.fixed;
    // A cell with no observed pixel (null in `points`) follows the 3D reprojection -- the
    // "projected" state. That now means exactly one thing: the detector fired nothing there.
    // The Hidden flag used to land in here too, because it NaN'd the cell's detection out
    // server-side; it no longer touches a position, so the two masks are independent and each
    // says only what it is named for.
    this.projectedMask = p.points.map((row) => row.map((pt) => pt == null));
    // The Hidden flag: which cells are held out of the training loss. Read by the card's toggle
    // and passed to every view, which strikes a bar through the joint (poseView drawHidden).
    if (p.invisible) this.excludedMask = p.invisible;
    if (p.has_instance != null && p.has_instance !== this.hasInstance) {
      this.hasInstance = p.has_instance;
      // The Instance pane reports this frame's skeleton, so it follows the payload that
      // decides whether there is one.
      this.renderInstance();
      this.applyDetected(); // the auto-hide rule is derived from it, so re-resolve
      this.syncSkeletonMenu();
    }
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
    // How many views carry a GT pixel for each keypoint of this frame -- the label-coverage
    // check's whole input, and a column sum of a mask the payload already carries. Computed here
    // rather than asked of the server for exactly that reason: it is a view of data the client
    // holds, so it costs no round trip, stays correct on the lean mid-drag stream, and adds
    // nothing to the wire. Every view is handed the same array (see PoseView.gtViews).
    const gtViews = (p.fixed[0] ?? []).map((_, i) =>
      p.fixed.reduce((n, row) => n + (row[i] ? 1 : 0), 0),
    );
    this.views.forEach((view, v) => {
      view.setFrameData({
        points: p.points[v],
        fixed: p.fixed[v],
        gtViews,
        instanceMode: !!p.has_instance,
        invented: p.invented ? p.invented[v] : undefined,
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
        // Landmarks ride every reply: a handful of points per view, so there is no
        // reason to make them a verbose-only field that could go stale mid-drag.
        landmarks: p.landmarks ? p.landmarks[v] : undefined,
      });
    });
    // The NMF mesh follows the (re-fit) latent skeleton: refresh it after an edit
    // settles, coalescing a live drag's many replies into one GPU render.
    this.scheduleMeshRefresh();
    this.dirty = p.dirty;
    if (p.reviewed != null) {
      this.reviewed = !!p.reviewed;
      this.reviewedBtn.setAttribute("aria-pressed", String(this.reviewed));
      this.reviewedBtn.classList.toggle("is-on", this.reviewed);
      this.reviewedBtn.disabled = this.readOnly;
    }
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

  // Where a non-GT joint of the instance is drawn. The reprojection of its point's current
  // 3D is the default and the multiview payoff -- label two views and the other five move to
  // where the geometry says the joint is. The seed is where the skeleton started: the honest
  // single-view answer, and what to look at when the geometry is suspect. Server-side, since
  // it is the server that resolves the position (EditorState.nongt_display).
  //: Where a joint you have NOT placed is drawn: "reprojection" (the default) or "seed".
  nongtDisplay = "reprojection";

  /** @param {string} value */
  setSeedMode(value) {
    this.seedMode = value;
    this.seedSwitch.set(value);
    this.sendEdit({ type: "set_seed_mode", value, frame: this.frame, mode: this.mode });
  }

  /**
   * Choose how a point's 3D is derived once its GT views are exclusive.
   *
   * Unlike the seeding mode this changes every point that is already solved, so the reply
   * is a full re-derived frame -- which is the point: flip it and the joints whose GT pair
   * has a poor baseline visibly move, which is the only direct read on what the unlabeled
   * views were contributing.
   *
   * @param {string} value "on" (default) or "off"
   */
  setSolveStabilizers(value) {
    this.solveStabilizers = value;
    this.stabilizeSwitch.set(value);
    this.sendEdit({
      type: "set_solve_stabilizers",
      value,
      frame: this.frame,
      mode: this.mode,
    });
  }

  // Which of the Skeleton menu's two verbs is live. They are mutually exclusive by
  // construction -- create_instance refuses a frame that already has a skeleton
  // (state.py:693) and reseed_instance one that does not (state.py:735) -- so disabling each
  // in the other's state is the honest rendering of that, and doubles as the readout for
  // whether this frame has been started. `g` / Shift+G stay bound either way: the server
  // flashes a notice, so the key never silently does nothing.
  syncSkeletonMenu() {
    this.skeletonCreateBtn.disabled = this.hasInstance || this.readOnly;
    this.reseedBtn.disabled = !this.hasInstance || this.readOnly;
  }

  // Start this frame without authoring anything. Idempotent: with a skeleton already there the
  // server flashes a notice rather than the key doing nothing.
  createInstance() {
    this.sendEdit({ type: "create_instance", frame: this.frame, mode: this.mode });
  }

  reseedInstance() {
    this.sendEdit({ type: "reseed_instance", frame: this.frame, mode: this.mode });
  }

  /** @param {string} value */
  setNongtDisplay(value) {
    this.nongtDisplay = value;
    this.nongtSwitch.set(value);
    this.sendEdit({ type: "set_nongt_display", value, frame: this.frame, mode: this.mode });
  }

  applyLabels() {
    const visible = this.labelsCheck.checked;
    this.views.forEach((view) => view.setLabelsVisible(visible));
  }


  //: Whether the detected reference layer is actually drawn: the operator's standing intent,
  //: minus the auto-hide rule. Derived rather than stored, so "auto-hidden" and "I unchecked
  //: it" never collapse into the same state -- the checkbox keeps meaning what they asked for.
  detectedShown() {
    return (
      this.detectedCheck.checked &&
      !(this.autoHideCheck.checked && this.hasInstance)
    );
  }

  // `t` has to do something every time it is pressed. When the layer is suppressed by the
  // auto-hide rule rather than by the operator, the honest response is to lift the rule -- not
  // to toggle a checkbox whose state is already what they wanted.
  toggleDetected() {
    if (this.detectedCheck.checked && !this.detectedShown()) {
      this.autoHideCheck.checked = false;
    } else {
      this.detectedCheck.checked = !this.detectedCheck.checked;
    }
    this.applyDetected();
  }

  applyDetected() {
    const shown = this.detectedShown();
    this.views.forEach((view) => view.setDetectedVisible(shown));
    // The row says what is actually drawn while the checkbox keeps saying what was asked for.
    const suppressed = this.detectedCheck.checked && !shown;
    this.detectedWrap.classList.toggle("is-suppressed", suppressed);
    this.detectedWrap.title = suppressed
      ? "auto-hidden: this frame has an annotation skeleton (t shows them again)"
      : this._detectedTitle;
  }

  applyProjected() {
    const visible = this.projectedCheck.checked;
    this.views.forEach((view) => view.setProjectedVisible(visible));
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

  applyCover() {
    // The checkbox is the operator's standing intent and persists as such; what is drawn also
    // needs the check to be AVAILABLE. Unlike the reprojection warning -- inert without a 3D solve
    // by construction (poseView.js drawReprojWarnings returns on a null `latent`) -- this one would
    // happily flag all 38 joints of a single-camera session against a two-view requirement that
    // session can never meet. So the availability gate is applied here, not left to the drawing.
    const wanted = this.coverCheck.checked;
    localStorage.setItem(COVER_ON_KEY, wanted ? "1" : "0");
    const on = wanted && this.meta.n_views > 1;
    this.views.forEach((view) => view.setCoverVisible(on));
  }

  // How many GT views a keypoint needs, clamped to what this rig could ever supply. Falls back to
  // the last-applied value while the field is momentarily empty mid-edit (as clampWarnPx does), so
  // a partial keystroke never silently re-flags the whole frame.
  clampCoverMin() {
    const n = Number(this.coverMinInput.value);
    if (!Number.isFinite(n) || n <= 0) {
      return this.views[0]?.coverMin ?? COVER_MIN_DEFAULT;
    }
    return Math.min(this.coverCeiling, Math.max(COVER_MIN_FLOOR, Math.round(n)));
  }

  //: The most GT views a keypoint of THIS rig could ever have -- the input's ceiling.
  get coverCeiling() {
    return Math.max(COVER_MIN_FLOOR, this.meta.n_views);
  }

  applyCoverMin() {
    const n = this.clampCoverMin();
    localStorage.setItem(COVER_MIN_KEY, String(n));
    this.views.forEach((view) => view.setCoverMin(n));
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
    this.closeSkeletonMenu(); // only one popover open at a time
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

  // -- the recording picker ---------------------------------------------------
  //
  // Which recording is open, and the project's others with the counts that decide which
  // is worth opening next. Fetched on open and never polled: the listing reads every
  // recording's labels.h5, which is cheap once and wasteful every two seconds.

  /** Fetch the project's recordings and render them into the menu. */
  async refreshRecordings() {
    const epoch = this.epoch;
    try {
      const listing = await fetchRecordings();
      if (epoch !== this.epoch) return; // it marks the previously-open recording active
      this.recordings = listing;
      // The authoritative answer to "where is the unsaved work" -- the server knows which
      // sessions it is holding, and this listing is the only reply that carries the counts
      // to go with them.
      this.applyDirtyRecordings(listing.dirty_recordings);
    } catch (err) {
      // Leave whatever is already listed: a transient failure should not blank a menu
      // the operator is looking at.
      this.recordingEmptyEl.hidden = false;
      this.recordingEmptyEl.textContent = `could not list recordings: ${err.message || err}`;
      return;
    }
    this.renderRecordings(this.recordings);
    // The listing may have named a recording this page had not heard of as unsaved (the
    // edits were made before a reload, or in another tab), so the toolbar cue follows it.
    this.updateDirty();
  }

  // Patched in PLACE, keyed by slug -- never rebuilt from scratch. This runs on every visit
  // to the tab, on a read-only role change, and on the far side of a recording switch, and a
  // replaceChildren() at any of those makes the whole list blink out and come back under the
  // operator who was just reading it (taking the scroll position with it). Selecting a
  // recording is the worst case: the list is exactly what the click was aimed at, and the one
  // thing that actually changed is which row is marked open.
  /** @param {any} payload  the /api/recordings body */
  renderRecordings(payload) {
    if (!payload || payload.enabled === false) {
      this.recordingListEl.replaceChildren();
      this.recordingEmptyEl.hidden = false;
      this.recordingEmptyEl.textContent =
        payload?.reason ?? "no project, so no other recordings to switch to";
      return;
    }
    const rows = payload.recordings ?? [];
    this.recordingEmptyEl.hidden = rows.length > 0;
    if (!rows.length) this.recordingEmptyEl.textContent = "this project has no recordings";
    const existing = new Map();
    for (const node of this.recordingListEl.children) {
      if (node instanceof HTMLElement && node.dataset.slug) existing.set(node.dataset.slug, node);
    }
    // `children` is live, so this is the standard keyed reconcile: put the row for entry `i`
    // at index `i`, and everything below `i` is already settled.
    const kids = this.recordingListEl.children;
    rows.forEach((rec, i) => {
      const row = this.recordingRow(existing.get(rec.slug), rec);
      if (kids[i] !== row) this.recordingListEl.insertBefore(row, kids[i] ?? null);
    });
    // Whatever is left past the end of the listing is a recording that is no longer there.
    while (kids.length > rows.length) this.recordingListEl.lastElementChild.remove();
  }

  /**
   * One row of the recording list -- built on first sight of its slug, then updated in place.
   * It carries NO click listener of its own: the list delegates (see buildSidebar), so a
   * reused row cannot quietly accumulate a second one.
   * @param {HTMLElement|undefined} row  the row already showing for this slug, if any
   * @param {any} rec  one entry of the /api/recordings listing
   * @returns {HTMLButtonElement}
   */
  recordingRow(row, rec) {
    let btn = /** @type {HTMLButtonElement|undefined} */ (row);
    if (!btn) {
      btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.slug = rec.slug; // the delegated handler's only input
      const name = document.createElement("span");
      name.className = "rec-name";
      name.textContent = rec.slug;
      // Every trailing span exists from the start and is written every time, so a row can be
      // reused whatever its flags say -- an absent span would mean rebuilding it instead.
      // `.rec-dirty` is the "unsaved labels live here" dot: the count in the toolbar says how
      // many recordings, and this says which.
      const dot = document.createElement("span");
      dot.className = "rec-dirty";
      dot.textContent = "●";
      const flag = document.createElement("span");
      flag.className = "rec-flag";
      const stats = document.createElement("span");
      stats.className = "rec-stats";
      btn.append(name, dot, flag, stats);
    }
    btn.className = "rec-row" + (rec.active ? " is-active" : "");
    const dot = /** @type {HTMLElement} */ (btn.querySelector(".rec-dirty"));
    dot.hidden = !rec.dirty;
    // A recording with no results.h5 has never been run: it opens as an uncalibrated 2D
    // session, which is a different (and much emptier) editor. Say so before the click, not
    // after. `.rec-flag` has no author `display`, so `hidden` genuinely removes it.
    const flag = /** @type {HTMLElement} */ (btn.querySelector(".rec-flag"));
    flag.textContent = rec.has_results ? "" : "2D only";
    flag.hidden = !!rec.has_results;
    /** @type {HTMLElement} */ (btn.querySelector(".rec-stats")).textContent = rec.labeled_frames
      ? `${rec.labeled_frames} labeled · ${rec.gt_points.toLocaleString()} pts`
      : "unlabeled";
    const frames = rec.n_frames == null ? "?" : rec.n_frames.toLocaleString();
    let title =
      `${rec.slug} — ${frames} frames, ${rec.gt_points.toLocaleString()} ground-truth ` +
      `points in ${rec.labeled_frames} frame(s), ${rec.reviewed_frames} reviewed, ` +
      `${rec.occluded.toLocaleString()} hidden marks`;
    // Counts from a recording the editor is holding are its LIVE ones (the server serves the
    // session, not the sidecar), so an unsaved row's numbers already include the work that
    // has not reached disk -- which is exactly why the row has to say so.
    if (rec.dirty) title += "\n(unsaved labels, kept in memory — Ctrl/Cmd+S saves them)";
    // The open recording has nowhere to go, and a read-only tab must not swap the recording
    // out from under the writer.
    if (rec.active) title += "\n(open)";
    else if (this.readOnly) title += "\n(this tab is read-only — another browser is editing)";
    btn.disabled = Boolean(rec.active) || this.readOnly;
    btn.title = title;
    return btn;
  }

  // No prompt, ever. The server keeps the recording being left -- its unsaved labels AND
  // its undo history -- so switching costs nothing and loses nothing, and the operator can
  // work across a project's recordings the way they work across its frames. The one moment
  // unsaved work is actually at stake is closing the editor, and that is where it is asked
  // about (requestClose).
  /** @param {string} slug */
  requestSwitch(slug) {
    this.doSwitch(slug, false);
  }

  /**
   * @param {string} slug
   * @param {boolean} discard  throw away the CURRENT recording's unsaved labels on the way
   *   out, instead of keeping them in memory. Nothing in the UI passes true; it is the
   *   server's contract, kept reachable rather than pretended away.
   */
  async doSwitch(slug, discard) {
    this.statusEl.textContent = `opening ${slug}…`;
    let reply;
    try {
      reply = await openRecording(slug, discard);
    } catch (err) {
      // Nothing was swapped: the server builds the new session completely before it
      // rebinds anything, so a failure leaves this recording open and intact.
      this.statusEl.textContent = `could not open ${slug}: ${err.message || err}`;
      return;
    }
    // Which recordings are unsaved changes as of this switch: the one just left joins the
    // list if it was dirty. Taken from the reply rather than inferred, so the cue is right
    // before the rebuild's own /api/meta lands.
    this.applyDirtyRecordings(reply?.dirty_recordings, slug);
    // The server pushes the reload to every browser, this one included; doing it here
    // too covers a tab whose socket has dropped, and a second reload is a no-op.
    await this.rebuildForNewRecording();
    // Said after the rebuild, or the rebuild's own status writes would bury it. A recording
    // that came back from memory is the whole point of the registry, and it is also the one
    // case where the editor is showing labels that are NOT what its labels.h5 says.
    if (reply?.restored) this.flash(`${slug} — restored, with your unsaved edits`);
  }

  // Everything the App holds that belongs to ONE recording. Anything missing here is the
  // previous animal's data presented as this one's -- and the [view][point] masks are
  // worse than misleading: a stale (view, point) whose view >= the new n_views makes
  // updateStatusWidget throw, which takes the editor down.
  //
  // Deliberately NOT reset:
  //   readOnly            the writer slot is per SOCKET and the socket survives the swap;
  //                       resetting it would silently promote a reader to writer
  //   closing             setting it would permanently disable the unsaved-changes guard
  //                       for the NEW recording's edits
  //   framesOpen          the operator's arrangement, not the recording's data
  //   sidebarTab          likewise -- a switch keeps you on the pane you were working in
  //   recordings          the PROJECT's listing, which a switch does not change; only which
  //                       row is marked open does, and that is corrected in place below
  //   editSeq / meshReq   bumped, never zeroed: zeroing risks an old reply matching again
  resetRecordingState() {
    this.frame = 0;
    this.focused = 0; // else relayout hands replaceChildren an undefined cell
    this.selection.clear();
    this.selAnchor = null;
    this.activeView = 0;
    this.hoverCell = null;
    this.fixedMask = null;
    this.projectedMask = null;
    this.absentMask = null;
    this.absentRecording = []; // else the previous animal's amputations are announced
    this.detectedMask = null;
    this.excludedMask = null;
    this.hasInstance = false;
    this.reviewed = false;
    this.reviewedBtn.setAttribute("aria-pressed", "false");
    this.reviewedBtn.classList.remove("is-on");
    // The new session's undo stack is empty. The next payload will say so, but an enabled
    // Undo in the meantime offers history belonging to the recording just closed.
    this.undoBtn.disabled = true;
    this.redoBtn.disabled = true;
    this.seedMode = "triangulate"; // the new EditorState's server-side defaults
    this.nongtDisplay = "reprojection";
    this.solveStabilizers = "on";
    this.correctedFrames = [];
    this.frameRows.clear();
    this.suggestions = null; // labels_suggest.json is per recording
    this.syncSkeletonMenu(); // the new frame 0 has no skeleton until the payload says so
    // The listing marks the OPEN recording active and disables its row, so it does have to be
    // corrected here -- but by MOVING the mark, not by blanking the list. This runs mid-switch
    // with the Recording pane very likely still on screen (it is where the click came from),
    // and emptying it there is the one refresh the operator actually sees. `this.meta` is
    // already the new recording at this point (rebuildForNewRecording fetches it first), and
    // the counts are re-read when the tab next asks for them.
    if (this.recordings?.recordings) {
      for (const rec of this.recordings.recordings) rec.active = rec.slug === this.meta.recording;
      this.renderRecordings(this.recordings);
    }
    // The Bundle-adjust pane is built once and outlives the switch, so its own per-recording
    // state -- the fix/free matrix above all, which is keyed by camera name -- has to be
    // dropped here with everything else. See BundleAdjustPanel.forget.
    this.baPanel?.forget();
    // A warm-up aimed at the recording just closed. Its callback would bail on the epoch
    // anyway; this also lets go of the previous animal's decoded pictures right away.
    clearTimeout(this.prefetchTimer);
    this.prefetchImgs = [];
    this.addMod = false;
    this.overViews = false;
    document.body.classList.remove("adding");
  }

  // The open recording changed underneath this page (switched here, or in another tab).
  // Everything the editor built at boot came from /api/meta, so all of it is rebuilt --
  // in place, with no page load.
  //
  // The SOCKET is deliberately kept. `/ws` reads `session` out of its enclosing scope on
  // every message and the switch handler rebinds that variable, so the open socket
  // already talks to the new session; reconnecting would re-run the writer-slot handshake
  // and could turn this tab from writer to reader without telling anyone.
  //
  // `closing` is never set here either -- it permanently disables the beforeunload guard,
  // and the NEW recording's edits still need protecting.
  async rebuildForNewRecording() {
    // The server broadcasts to every socket including the one that asked, so the
    // switching tab arrives here twice: once from the push and once from doSwitch. A page
    // load made the second call a no-op; an async rebuild does not.
    if (this.rebuilding) return this.rebuilding;
    this.rebuilding = (async () => {
      try {
        // 1 -- quiesce. Each of these would otherwise fire against the new recording
        // carrying the old one's intent.
        clearTimeout(this.correctedTimer);
        clearTimeout(this.meshTimer);
        clearTimeout(this.sceneMeshTimer);
        this.stopJobsPolling();
        this.epoch++; // invalidates every in-flight fetch
        this.editSeq++;
        this.meshReq++;
        this.closeShowMenu();
        this.closeSkeletonMenu();
        this.closeHelp();
        this.closeCloseConfirm();
        this.armLandmark(-1); // fans out to the OLD views, and clears body.arming-landmark

        // 2 -- tear the canvases down. buildViews APPENDS, so without this both rigs
        // would be live: relayout re-attaches the old canvases and applyPoints indexes
        // past the end of p.points and throws.
        for (const view of this.views) view.destroy();
        this.stageEl.replaceChildren();
        this.stripEl.replaceChildren();
        this.views = [];
        this.cells = [];

        // 3 -- new meta. This also re-stamps api.js's module-level cache token, which
        // frameUrl reads at CALL time, so it must precede goToFrame or the new frames go
        // out under the old recording's token and miss the cache in both directions.
        this.meta = await fetchMeta();
        this.dirty = this.meta.dirty;
        // The recordings left behind with unsaved labels -- the state the whole switch is
        // built on, and the reason the cue must survive the rebuild that blanks everything
        // else belonging to one recording.
        this.applyDirtyRecordings(this.meta.dirty_recordings);

        // 4 + 5 -- per-recording state, then the meta-derived chrome. Never the wiring.
        this.resetRecordingState();
        this.applyMeta();
        // `l`, `[`, `]`, `p`, `w`, `u`, `m`, Shift+M and `b` are meta-gated. onKey reads
        // this.bindings at dispatch time, so reassigning is enough -- the keydown
        // listener must NOT be re-added.
        this.bindings = this.buildBindings();
        this.applyOsHints();
        this.helpBuilt = false; // the legend is built from this rig's limbs and points

        // 6 -- rebuild the canvases, then fan out every display toggle: new PoseViews
        // take their constructor defaults, which match the HTML at boot but not the
        // operator's current checkboxes.
        this.buildViews();
        this.applyHideAll();
        this.applyLabels();
        this.applyDetected();
        this.applyProjected();
        this.applyWarn();
        this.applyWarnThreshold();
        this.applyCover();
        this.applyCoverMin();
        this.applyNmf();
        this.applyMesh();
        this.relayout();

        // 7 -- re-seed the 3D scene in place; never `new Scene3D` on the same canvas.
        if (this.scene) {
          this.scene.setCameras(this.meta.cameras_3d);
          this.scene.setSkeleton(this.meta.bones, this.meta.point_colors);
          this.scene.setPoints3d(null);
          this.scene.setNmf3d(null);
        }
        this.sceneMeshReady = false;

        // 8 -- repaint. The lists are emptied before the fetch, so the old recording's
        // frame numbers are never on screen under the new recording's name.
        this.renderFrameList();
        this.renderLandmarks();
        this.renderInstance();
        await this.goToFrame(0);
        this.updateSelected();
        this.updateDirty();
        await this.refreshCorrected();
        this.refreshSuggestions();
        if (this.tabActive("recordings")) this.refreshRecordings();
        // A switch keeps you on the pane you were working in, and this one was just
        // emptied by resetRecordingState -- so if it is the pane on screen it has to be
        // re-read now rather than when it is next activated. (A switch can arrive from
        // another browser, which is how this pane gets to be the visible one.)
        if (this.baPanel && this.tabActive("bundle")) this.refreshBundleAdjust();
        this.statusEl.textContent = `opened ${this.meta.recording}`;
      } finally {
        this.rebuilding = null;
      }
    })();
    return this.rebuilding;
  }

  // -- the "Layout" popover (arrangement + view reset) ------------------------




  /** Reset zoom + pan on every camera back to the letterboxed fit (the "tight fit"). */
  resetView() {
    this.views.forEach((view) => view.resetZoom());
  }

  // -- the "Skeleton" popover (how a new one is seeded, and reseeding this frame) ----

  openSkeletonMenu() {
    this.closeShowMenu();
    this.skeletonMenu.hidden = false;
    this.skeletonMenuOpen = true;
    this.skeletonToggle.setAttribute("aria-expanded", "true");
    this.skeletonToggle.classList.add("is-open");
  }

  closeSkeletonMenu() {
    this.skeletonMenu.hidden = true;
    this.skeletonMenuOpen = false;
    this.skeletonToggle.setAttribute("aria-expanded", "false");
    this.skeletonToggle.classList.remove("is-open");
  }

  toggleSkeletonMenu() {
    if (this.skeletonMenuOpen) this.closeSkeletonMenu();
    else this.openSkeletonMenu();
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
    // Double-clicking the detected skeleton is how a frame is started deliberately: it
    // creates the annotation skeleton, seeded from the detections, without authoring
    // anything. Once one exists the gesture goes back to selecting the joint everywhere.
    if (!this.hasInstance && !this.readOnly) {
      this.sendEdit({ type: "create_instance", frame: this.frame, mode: this.mode });
      return;
    }
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
    this.updateStatusWidget(); // gates every card button, Reset included
  }

  // -- point state control ----------------------------------------------------


  /**
   * Whether the detector produced a raw prediction for this cell. A cell with no
   * detection can never fall back *to* the detector, so its "Detected" chip is disabled.
   * @param {number} view
   * @param {number} point
   */
  cellDetected(view, point) {
    return !!(this.detectedMask && this.detectedMask[view][point]);
  }


  // The combined state control -- one chip row that both reports and sets the selection's
  // state. The name field shows the single cell's "point · camera", the count for
  // several, or "—" for none. The active chip is the selection's shared state (nothing
  // lit when the cells disagree). Chips are live whenever something is selected; the
  // "Projected" chip additionally needs 3D (there is no reprojection to follow without one),
  // and the "Detected" chip needs at least one selected cell the detector actually fired for
  // (else there is no detection to fall back to).
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
    // identity -- a transient peek. The toggles' enabled/disabled state stays
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
    // The facts line: what IS true of this cell, not a state to assign. Reports the hovered
    // joint while hovering (the peek), else the selection's shared description, else how
    // many cells disagree.
    // Verb availability. Hidden is available on any cell of a joint that is on the animal --
    // including one that already carries GT, which is the pairing worth recording.
    const anyWithoutGt = this.selCells().some(([v, p]) => !(this.fixedMask && this.fixedMask[v][p]));
    const anyWithGt = this.selCells().some(([v, p]) => this.fixedMask && this.fixedMask[v][p]);
    // Each button reports its own fact by being pressed, which is what retired the text
    // readout beside the name: it was saying what three toggles could show themselves.
    const allGt = n > 0 && !anyWithoutGt;
    this.gtBtn.setAttribute("aria-pressed", String(allGt));
    this.gtBtn.classList.toggle("is-on", allGt);
    this.gtBtn.disabled = n === 0 || this.readOnly || allAbsent;
    const anyHidden = this.selCells().some(
      ([v, p]) => this.excludedMask && this.excludedMask[v][p],
    );
    const allHidden =
      n > 0 && this.selCells().every(([v, p]) => this.excludedMask && this.excludedMask[v][p]);
    this.excludeBtn.setAttribute("aria-pressed", String(allHidden));
    this.excludeBtn.classList.toggle("is-on", allHidden);
    // Reset retracts both facts, so it is only live when there is one to retract -- the
    // rightmost button in the card must not be the one most often a no-op.
    this.actResetBtn.disabled =
      n === 0 || this.readOnly || !(anyWithGt || anyHidden);
    // Labeled cells included: carrying a pixel is no reason to be unable to withhold it.
    this.excludeBtn.disabled = n === 0 || this.readOnly || allAbsent;
  }


  // Create GT for the selection at the position already drawn there. The bulk half of a
  // drag: it authors the dot the operator is looking at so the joint becomes theirs, ready
  // to nudge. Cells with nothing visible are skipped server-side rather than invented.
  // One control for one mutually exclusive pair. Pressed means the selection carries the
  // operator's pixels, so the click clears them; otherwise it places them. A mixed selection
  // places, so the first click completes it -- the same rule the Hidden toggle uses, and the one
  // that makes a second click always the inverse of the first.
  toggleSelectionGt() {
    const cells = this.selCells();
    if (!cells.length) return;
    const allGt = cells.every(([v, p]) => this.fixedMask && this.fixedMask[v][p]);
    if (allGt) this.deleteSelectionGt();
    else this.createSelectionGt();
  }

  createSelectionGt() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "confirm", targets, frame: this.frame, mode: this.mode });
  }

  // Delete the selection's GT pixels, leaving any exclusion alone. Distinct from Reset,
  // which retracts both -- see EditorState.clear_gt_targets.
  deleteSelectionGt() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "clear_gt_targets", targets, frame: this.frame, mode: this.mode });
  }

  // Toggle "hold this cell out of the training loss" over the selection. Its own axis: it never
  // reads or writes the GT pixel (a cell can carry both), and it is inert in the solve and in the
  // display beyond its own strike-through mark -- which cells the loss uses is the one fact about a
  // cell that geometry cannot supply, so it needs its own switch rather than a third GT state.
  toggleSelectionExclude() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "toggle_exclude", targets, frame: this.frame, mode: this.mode });
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
   */
  onDragged(view, point, x, y) {
    if (!this.resolves3d) {
      // No 3D to re-solve: the drop is simply this view's ground-truth pixel.
      this.sendEdit({ type: "edit_2d", view, point, x, y, frame: this.frame, mode: this.mode });
      return;
    }
    // Releasing a drag pins the dragged view at the drop pixel (a finalized constraint) so the
    // placed point stays put -- including one marked Hidden, which the drag leaves marked: the
    // operator is asserting where the joint is, not that the pixels show it.
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

  // (There is no per-cell `onToggleInvisible` handler: nothing in the canvas is bound to it, and
  // the one it used to carry gated the flag on `has_3d` -- which EditorState.toggle_invisible
  // explicitly refuses to do, because which cells to train on is decided in labeling rounds that
  // run before any triangulation exists. `e` on the selection is the gesture.)

  // -- the verbs on the selection ---------------------------------------------
  //
  // Not states. A cell carries a pixel the operator created or it does not, and its
  // detection is excluded from triangulation or it is not; these are the retractions and
  // assertions that move between those facts. `createSelectionGt` / `deleteSelectionGt` /
  // `toggleSelectionExclude` live above, beside the readout that reports the facts.


  // Reset the selection: retract both authored facts -- the ground-truth pixel and the Hidden
  // flag -- so the cell falls back to nothing authored. The one verb that spans both axes, which
  // is why it is named for starting over rather than for either of them. One undo step.
  resetSelection() {
    const targets = this.selCells();
    if (!targets.length) return;
    this.sendEdit({ type: "reset", targets, frame: this.frame, mode: this.mode });
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

  // Save EVERY recording holding unsaved labels, not just the open one. Edits survive a
  // recording switch, so "unsaved" is a project-wide state: a Save that wrote only this
  // recording would leave the title starred and the cue lit, and send the operator hunting
  // for which other recording still needed a click.
  async save() {
    if (this.readOnly) return; // read-only: the writer owns saving the shared state
    const targets = this.dirtySlugs().length;
    const r = await saveAllCorrections();
    this.dirty = !!r.dirty;
    this.applyDirtyRecordings(r.dirty_recordings);
    this.updateDirty();
    // A write that failed leaves that recording's work in memory only, so it is named and
    // it THROWS: `saveAndShutdown` awaits this, and closing the editor on a sidecar that
    // could not be written is the one outcome nothing else can undo.
    if (r.failed?.length) {
      const names = r.failed.map((f) => f.recording || "this recording").join(", ");
      this.flash(`could not save ${names} — still unsaved`);
      throw new Error(`could not save ${names}`);
    }
    this.flash(targets > 1 ? `saved ${targets} recordings` : "saved");
    // A save is when the sidecar's own view of "already labeled" could have moved, so
    // it is the one edit-side moment worth re-reading the queue's staleness for.
    this.refreshSuggestions();
    // The listing's counts come from the saved sidecars for every recording but this one,
    // so a save is when the other rows stop being behind.
    if (this.tabActive("recordings")) this.refreshRecordings();
  }

  /** Every recording with unsaved labels, the open one first when it is one of them.
   * @returns {string[]} */
  dirtySlugs() {
    const here = this.meta?.recording;
    const others = [...this.dirtyOthers].filter((s) => s !== here);
    return this.dirty && here ? [here, ...others] : others;
  }

  //: Whether ANYTHING is unsaved, here or in another recording -- what the close prompt and
  //: the beforeunload guard ask. `dirty` alone would let an operator close the editor on
  //: another recording's hand work, which is the one thing switching freely must not cost.
  get projectDirty() {
    return this.dirty || this.dirtySlugs().length > 0;
  }

  /** Adopt a server-sent `dirty_recordings` list (meta, a save reply, a switch reply).
   * @param {string[] | undefined} list
   * @param {string} [active]  the open recording, when it is about to change (a switch
   *   reply arrives before the rebuild has fetched the new meta)
   */
  applyDirtyRecordings(list, active) {
    if (!Array.isArray(list)) return;
    const here = active ?? this.meta?.recording;
    // The open recording is tracked by `dirty` (refreshed by every edit reply); keeping it
    // out of this set is what stops the two disagreeing for the moment between an edit and
    // the next listing.
    this.dirtyOthers = new Set(list.filter((slug) => slug && slug !== here));
  }

  // Every readout of "you have unsaved work": the browser tab's title, the Save button, the
  // toolbar cue, and the dot on each affected row of the recording list. All of it is driven
  // from the project-wide state, so nothing here can say "saved" while another recording's
  // labels are still only in memory.
  updateDirty() {
    const slugs = this.dirtySlugs();
    const dirty = this.projectDirty;
    document.title = `deeperfly gui — ${this.meta.results_path}${dirty ? " *" : ""}`;
    this.saveBtn.disabled = !dirty || this.readOnly;
    this.saveBtn.classList.toggle("is-dirty", dirty && !this.readOnly);
    this.unsavedChip.hidden = !dirty;
    // Shown but inert in a read-only tab: the state is worth knowing there (the writer's
    // work is what would be lost), and clicking it must not look like it did something.
    this.unsavedChip.disabled = this.readOnly;
    // The count is recordings, not edits: what the operator needs to know is whether the
    // unsaved work is all in front of them or partly in a recording they have left.
    this.unsavedChip.textContent =
      slugs.length > 1 ? `● unsaved · ${slugs.length}` : "● unsaved";
    this.unsavedChip.title = dirty
      ? `Unsaved labels in ${slugs.length ? slugs.join(", ") : "this recording"}` +
        " — click to save them (Ctrl/Cmd+S). They are kept in memory while you switch " +
        "recordings, and only closing the editor can lose them."
      : "";
    // The list marks WHICH recordings, which is the half the count cannot carry. Patched in
    // place (renderRecordings reconciles by slug), so this repaints no row it need not.
    if (this.recordings?.recordings) {
      const set = new Set(slugs);
      let moved = false;
      for (const rec of this.recordings.recordings) {
        const next = set.has(rec.slug);
        moved ||= Boolean(rec.dirty) !== next;
        rec.dirty = next;
      }
      if (moved) this.renderRecordings(this.recordings);
    }
  }

  // -- corrected-frames list --------------------------------------------------

  // Pull the frames carrying corrections and repaint the side panel. Called on load
  // (any sidecar loaded from disk) and, debounced, after each edit settles -- so the
  // list tracks every drag, obscure, and reset live.
  async refreshCorrected() {
    const epoch = this.epoch;
    let frames;
    try {
      frames = (await fetchCorrected()).frames;
    } catch (_) {
      return; // a transient failure just leaves the list as it was
    }
    if (epoch !== this.epoch) return; // these are the previous recording's frames
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
    for (const badge of [this.framesCountEl, this.labeledCountEl]) {
      badge.textContent = String(n);
      badge.classList.toggle("is-zero", n === 0);
    }
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
      rcell.append(this.reviewedTick(frame, reviewed, false));
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
      if (box) box.checked = !value; // read-only: undo the visual toggle, change nothing
      return;
    }
    const row = this.correctedFrames.find((f) => f.frame === frame);
    if (row) {
      row.reviewed = value;
    } else if (value) {
      // Ticking a frame that carries no label yet -- legitimate ("I looked, the
      // predictions were right"), and reachable now that the tick lives in the queue.
      // Without this the optimistic flip lands nowhere and the box appears to spring
      // back until the refresh returns. Insert in frame order, which is the order the
      // server returns and the Labeled list renders in.
      const at = this.correctedFrames.findIndex((f) => f.frame > frame);
      this.correctedFrames.splice(at < 0 ? this.correctedFrames.length : at, 0, {
        frame,
        reviewed: true,
      });
    }
    this.sendEdit({ type: "set_reviewed", frame, reviewed: value, mode: this.mode });
    // Repaint both lists from the optimistic state so the queue's own tick, its row
    // styling and the unreviewed warning all move together on the click rather than
    // 150 ms later when the debounced refetch lands.
    this.renderFrameList();
    this.renderSuggestList();
    this.scheduleCorrectedRefresh();
  }

  // Flip the CURRENT frame's reviewed flag from the keyboard, without opening anything.
  //
  // Until this existed the flag had exactly one control: a checkbox in a side panel that is
  // hidden by default (`j`), on a tab the operator may not be on. The result was not the
  // occasional missed tick but wholesale loss -- an audit of 16 recordings found three of
  // them at ZERO ticks while carrying 22 finished frames of hand labelling, and the other
  // thirteen at 100%. A flag that decides which frames are trusted downstream cannot live
  // only behind two disclosures.
  toggleReviewedCurrent() {
    if (this.readOnly) return;
    const row = this.correctedFrames.find((f) => f.frame === this.frame);
    this.toggleReviewed(this.frame, !(row?.reviewed ?? false), null);
  }

  // One "reviewed" checkbox, shared by the Labeled list and the suggestion queue so the
  // two can never drift in behaviour. Clicks are kept off the row, which jumps to the frame.
  // `withText` is off in the Labeled table, which already has a Reviewed column heading,
  // and on in the queue, where the row has no column to say what the box means.
  /**
   * @param {number} frame
   * @param {boolean} reviewed
   * @param {boolean} [withText]
   * @returns {HTMLLabelElement}
   */
  reviewedTick(frame, reviewed, withText = true) {
    const wrap = document.createElement("label");
    wrap.className = "reviewed-tick";
    wrap.classList.toggle("is-on", reviewed);
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = reviewed;
    box.disabled = this.readOnly;
    wrap.title = reviewed
      ? "You have checked this frame — click to un-mark"
      : "Tick once you have checked this frame's points";
    wrap.addEventListener("click", (e) => e.stopPropagation());
    box.addEventListener("change", () => this.toggleReviewed(frame, box.checked, box));
    wrap.append(box);
    if (withText) {
      const text = document.createElement("span");
      text.textContent = reviewed ? "reviewed ✓" : "mark reviewed";
      wrap.append(text);
    }
    return wrap;
  }

  // Highlight the row for the current frame (when it is a corrected one) and, while
  // the panel is open, scroll it into view -- so scrubbing keeps the list in sync.
  updateActiveFrameRow() {
    this.frameRows.forEach((tr, frame) => tr.classList.toggle("is-current", frame === this.frame));
    this.suggestRows.forEach((trs, frame) => {
      for (const tr of trs) tr.classList.toggle("is-current", frame === this.frame);
    });
    if (!this.framesOpen) return;
    // Both lists are highlighted -- rows in the hidden pane keep their mark for when its tab
    // comes back -- but only the pane on screen is SCROLLED. Scrolling a hidden pane is at
    // best wasted and at worst a surprise jump the moment its tab is shown.
    if (!this.tabActive(this.navList)) return;
    const active =
      this.navList === "suggest"
        ? this.suggestRows.get(this.frame)?.[0]
        : this.frameRows.get(this.frame);
    active?.scrollIntoView({ block: "nearest" });
  }

  openFrames() {
    this.sidebarEl.hidden = false;
    // The rail is the panel's absence made clickable, so the two are never both on screen.
    this.sidebarRailBtn.hidden = true;
    this.sidebarRailBtn.setAttribute("aria-expanded", "true");
    this.framesOpen = true;
    localStorage.setItem(SIDEBAR_OPEN_KEY, "1");
    this.syncTabEffects();
    this.updateActiveFrameRow(); // scroll the current frame into view now it is shown
  }

  closeFrames() {
    this.sidebarEl.hidden = true;
    this.sidebarRailBtn.hidden = false;
    this.sidebarRailBtn.setAttribute("aria-expanded", "false");
    this.framesOpen = false;
    localStorage.setItem(SIDEBAR_OPEN_KEY, "0");
    // Hiding the panel disarms too: an armed landmark whose pane is no longer on screen is
    // the same silent-placement surprise as one whose tab was left behind.
    this.armLandmark(-1);
    this.syncTabEffects();
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
    this.stepList(this.navList, dir);
  }

  /** The labeled frame `dir` away from `from` (wrapping at the ends), or null if none.
   *
   * Split out of `jumpLabeled` so the prefetch hint can ask the very same question a
   * second time -- "and where would the next press land?" -- rather than reimplementing
   * the wrap. A hint that guessed differently from the jump would warm the wrong picture
   * and pay for it twice, so the two answers have to come from one place.
   * @param {number} from
   * @param {number} dir
   * @returns {number | null}
   */
  nextLabeled(from, dir) {
    const frames = this.correctedFrames.map((f) => f.frame); // ascending, in time order
    if (frames.length === 0) return null;
    if (dir > 0) return frames.find((f) => f > from) ?? frames[0]; // wrap to the first
    const earlier = frames.filter((f) => f < from);
    return earlier.length ? earlier[earlier.length - 1] : frames[frames.length - 1];
  }

  /** @param {number} dir  -1 for the previous labeled frame, +1 for the next */
  jumpLabeled(dir) {
    const target = this.nextLabeled(this.frame, dir);
    if (target == null) return;
    // Walking this list one keypress at a time is the primary annotation loop, and unlike a
    // scrubber drag it is entirely predictable -- so warm where the next press goes. Worth
    // hinting precisely here because a labeled frame is nowhere near the one on screen: it
    // costs a full seek and GOP walk (~250 ms for a rig on 1984x512 footage), against ~3 ms
    // once the browser holds it. Skipped when the list has a single entry, where the wrap
    // would name the frame already on screen.
    const after = this.nextLabeled(target, dir);
    this.goToFrame(target, after != null && after !== target ? { then: after } : undefined);
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
  // -- project settings ------------------------------------------------------
  //
  // Generated from GET /api/schema, which is itself DERIVED from the *Params dataclasses.
  // So this panel cannot drift from the code -- a new option appears with nothing to keep
  // in sync -- and each field's help text is the prose already written for it, which is
  // better than anything a form label would say. The four open-ended sections (cameras,
  // skeleton, sources, the detection plan) are NAMED as needing the file rather than
  // rendered as empty forms.

  /** @type {any} The Bundle-adjust pane owns itself; built on first activation. */
  baPanel = null;

  /** Build the Bundle-adjust pane on first use, then refresh it.
   *
   * Lazy for the same reason Settings and Jobs are: the plan route gathers every label in the
   * session to compute readiness, which is wasted work for an operator who never opens it.
   */
  refreshBundleAdjust() {
    if (this.baPanel === null) {
      this.baPanel = new BundleAdjustPanel(
        /** @type {HTMLElement} */ (document.getElementById("ba-pane")),
        {
          isReadOnly: () => Boolean(this.readOnly),
          isDirty: () => Boolean(this.dirty),
          // A new rig changes every derived position, so the views must be redrawn from the
          // server rather than from anything this page still holds.
          onRigChanged: () => this.refreshPoints(),
        },
      );
    }
    this.baPanel.refresh();
  }

  async refreshSettings() {
    let schema = this.configSchemaCache;
    let values;
    try {
      if (!schema) {
        schema = await configSchema();
        this.configSchemaCache = schema;
      }
      values = await configValues();
    } catch (err) {
      this.settingsList.replaceChildren();
      this.settingsEmpty.hidden = false;
      this.settingsEmpty.textContent = `Could not read the settings: ${err}`;
      return;
    }
    if (!values.enabled) {
      this.settingsList.replaceChildren();
      this.settingsEmpty.hidden = false;
      this.settingsEmpty.textContent =
        "Open a project to change its settings from the editor — a bare results.h5 has no " +
        "profile to write them to.";
      return;
    }
    this.settingsEmpty.hidden = true;
    this.renderSettings(schema, values);
  }

  /** @param {any} schema @param {any} values */
  renderSettings(schema, values) {
    this.settingsList.replaceChildren();
    for (const section of schema.sections || []) {
      const current = values.sections[section.name];
      if (!current || current.error) continue;
      const box = document.createElement("div");
      box.className = "settings-section";
      const title = document.createElement("h4");
      title.textContent = `[${section.name}]`;
      box.append(title);
      if (section.doc) {
        const doc = document.createElement("p");
        doc.className = "sec-doc";
        doc.textContent = section.doc;
        box.append(doc);
      }
      for (const field of section.fields) {
        const state = current[field.name];
        if (!state) continue;
        box.append(this.settingRow(section.name, field, state));
      }
      this.settingsList.append(box);
    }
    if ((schema.undescribable || []).length) {
      const note = document.createElement("p");
      note.className = "setting-doc";
      note.style.marginTop = "12px";
      note.textContent =
        "Not editable here (open-ended, and they live in the config file): " +
        schema.undescribable.join(", ") + ".";
      this.settingsList.append(note);
    }
  }

  /** @param {string} section @param {any} field @param {any} state */
  settingRow(section, field, state) {
    const row = document.createElement("div");
    row.className = "setting-row";
    const label = document.createElement("span");
    label.className = "setting-key" + (state.overridden ? " overridden" : "");
    label.textContent = field.name;
    label.title = `${field.type} · default ${JSON.stringify(field.default)}`;
    row.append(label);

    const controls = document.createElement("span");
    controls.append(this.settingInput(section, field, state));
    if (state.overridden) {
      const reset = document.createElement("button");
      reset.className = "setting-reset";
      reset.textContent = "↺";
      reset.title = "Clear this override — the profile stops mentioning the key entirely";
      reset.addEventListener("click", () => this.applySetting(section, field.name, null));
      controls.append(reset);
    }
    row.append(controls);

    if (field.doc) {
      const doc = document.createElement("p");
      doc.className = "setting-doc";
      doc.textContent = field.doc;
      row.append(doc);
    }
    return row;
  }

  /** @param {string} section @param {any} field @param {any} state */
  settingInput(section, field, state) {
    const type = String(field.type || "");
    const commit = (value) => this.applySetting(section, field.name, value);

    // A list or dict field cannot be edited from a form without inventing a schema for its
    // shape (bounds, marker placements, shared-parameter groups). Showing it read-only is
    // honest, and the profile file is right there.
    if (type.startsWith("list") || type.startsWith("dict")) {
      const shown = document.createElement("input");
      shown.type = "text";
      shown.value = JSON.stringify(state.value);
      shown.disabled = true;
      shown.title = "Edit this one in the profile file — a form cannot express its shape";
      return shown;
    }
    if (type === "bool") {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = Boolean(state.value);
      box.addEventListener("change", () => commit(box.checked));
      return box;
    }
    if (field.choices && field.choices.length) {
      const select = document.createElement("select");
      for (const choice of field.choices) {
        const option = document.createElement("option");
        option.value = String(choice);
        option.textContent = String(choice);
        option.selected = String(choice) === String(state.value);
        select.append(option);
      }
      select.addEventListener("change", () => commit(select.value));
      return select;
    }
    const numeric = type === "int" || type === "float";
    const input = document.createElement("input");
    input.type = numeric ? "number" : "text";
    if (type === "float") input.step = "any";
    input.value = state.value === null ? "" : String(state.value);
    // On commit (blur / Enter), not per keystroke: every write validates server-side and
    // rewrites the profile, so firing on input would mean a file write per character.
    input.addEventListener("change", () => {
      const raw = input.value.trim();
      if (raw === "") return commit(null); // emptied = reset to default
      commit(numeric ? Number(raw) : raw);
    });
    return input;
  }

  /** @param {string} section @param {string} key @param {any} value */
  async applySetting(section, key, value) {
    try {
      await setConfig(section, key, value);
    } catch (err) {
      window.alert(`Could not set ${section}.${key}: ${err}`);
    }
    this.refreshSettings();
  }

  // -- calibration landmarks -------------------------------------------------
  //
  // Non-skeleton points that make a from-scratch rig solvable. The gesture is deliberately
  // NOT a drag: a landmark has no detection to grab and no reprojection to nudge, so there
  // is nothing on the canvas to start a drag from until one exists. Arming a landmark makes
  // the next click place it, which is the only interaction that works from an empty frame.

  renderLandmarks() {
    const marks = this.meta.landmarks || [];
    // The chip's badge, so whether this project declares any landmarks at all is readable
    // from whichever pane is showing.
    this.marksCountEl.textContent = String(marks.length);
    this.marksCountEl.classList.toggle("is-zero", marks.length === 0);
    this.marksList.replaceChildren();
    if (!marks.length) {
      this.marksEmpty.hidden = false;
      this.marksEmpty.textContent =
        "This project declares no calibration landmarks. Add them to landmarks.toml — " +
        "a static point (a coverslip scratch, the tether tip) is worth far more to the " +
        "rig solve than more keypoint frames.";
      return;
    }
    this.marksEmpty.hidden = true;
    marks.forEach((mark, i) => {
      const row = document.createElement("div");
      row.className = "mark-row" + (i === this.armedLandmark ? " armed" : "");
      row.title = i === this.armedLandmark
        ? "Armed — click in any view to place it here. Click this row again to disarm."
        : "Click to arm, then click in each view where you can see this landmark.";
      const dot = document.createElement("span");
      dot.className = "mark-diamond";
      const name = document.createElement("span");
      name.className = "mark-name";
      name.textContent = mark.name;
      const kind = document.createElement("span");
      kind.className = "mark-kind";
      kind.textContent = mark.static ? "static" : "per-frame";
      const count = document.createElement("span");
      count.className = "mark-count";
      count.textContent = `${mark.observations}`;
      count.title = "Observations placed so far, across every view and frame";
      row.append(dot, name, kind, count);
      row.addEventListener("click", () =>
        this.armLandmark(i === this.armedLandmark ? -1 : i),
      );
      this.marksList.append(row);
    });
    const hint = document.createElement("div");
    hint.className = "marks-hint";
    hint.textContent =
      "Arm a landmark, then click the SAME feature in as many views as can see it. " +
      "A static landmark keeps one 3D position for the whole recording, so re-placing it " +
      "in more frames sharpens it — but always on the same feature.";
    this.marksList.append(hint);
  }

  /** @param {number} index  landmark to arm, or -1 to disarm */
  armLandmark(index) {
    if (this.armedLandmark === index) return;
    this.armedLandmark = index;
    this.views.forEach((view) => view.setArmedLandmark(index));
    document.body.classList.toggle("arming-landmark", index >= 0);
    if (this.tabActive("marks")) this.renderLandmarks();
  }

  /** @param {number} view @param {number} landmark @param {number} x @param {number} y */
  onPlaceLandmark(view, landmark, x, y) {
    if (this.readOnly) return;
    this.sendEdit({
      type: "set_landmark",
      view,
      landmark,
      x,
      y,
      frame: this.frame,
      mode: this.mode,
    });
    // The count in the panel comes from meta, which is fetched once -- so bump it locally
    // rather than refetching the whole payload for one number.
    const mark = (this.meta.landmarks || [])[landmark];
    if (mark) mark.observations += 1;
    if (this.tabActive("marks")) this.renderLandmarks();
  }

  // -- pipeline jobs ---------------------------------------------------------
  //
  // Each row IS a CLI command, printed verbatim. That is deliberate: the GUI teaches the
  // CLI rather than hiding it, and a job that fails is reproducible by copy-paste instead
  // of requiring someone to reverse-engineer what the GUI did.
  //
  // There is no progress percentage. The commands emit human log lines and a
  // terminal-sized progress bar; a number synthesized from those would be a fiction with a
  // spinner attached, so the last log line is shown instead -- which is the honest signal.

  startJobsPolling() {
    this.refreshJobs();
    if (this.jobsTimer === null) {
      this.jobsTimer = window.setInterval(() => this.refreshJobs(), 2000);
    }
  }

  stopJobsPolling() {
    if (this.jobsTimer !== null) {
      window.clearInterval(this.jobsTimer);
      this.jobsTimer = null;
    }
  }

  async refreshJobs() {
    const epoch = this.epoch;
    let payload;
    try {
      payload = await fetchJobs();
    } catch {
      // A transient fetch failure must not blank a list the operator is reading.
      return;
    }
    if (epoch !== this.epoch) return;
    if (!payload.enabled) {
      this.jobsActions.replaceChildren();
      this.jobsList.replaceChildren();
      // Say WHY there are no buttons; an unexplained empty panel reads as broken.
      this.jobsEmpty.textContent = payload.reason || "No job queue for this session.";
      this.jobsEmpty.hidden = false;
      this.stopJobsPolling();
      return;
    }
    this.renderJobActions();
    this.renderJobs(payload.jobs || []);
  }

  renderJobActions() {
    if (this.jobsActions.childElementCount) return; // built once
    const recording = this.meta.recording;
    const actions = [
      ["Suggest frames", "labels-suggest", ["."], "Rank which frames are worth labelling next"],
      ["Export labels", "labels-export", ["."], "Write the training/eval dataset from the saved ground truth"],
      ["Check calibration", "calibrate", ["--dry-run"], "Report how close this project is to a solvable camera rig"],
      ["Solve calibration", "calibrate", [], "Solve the rig from the labels (writes a calibration; does not accept it)"],
    ];
    for (const [text, kind, argv, why] of actions) {
      const btn = document.createElement("button");
      btn.textContent = text;
      btn.title = why;
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          await submitJob({ kind, argv, label: text, recording });
        } finally {
          btn.disabled = false;
          this.refreshJobs();
        }
      });
      this.jobsActions.append(btn);
    }
  }

  /** @param {any[]} jobs */
  renderJobs(jobs) {
    // Queued + running, on the Jobs chip, so work in flight is visible from whichever pane
    // the operator is actually using -- which is the whole point of starting a job.
    const busy = jobs.filter((j) => j.state === "queued" || j.state === "running").length;
    this.jobsCountEl.hidden = busy === 0;
    this.jobsCountEl.textContent = String(busy);
    this.jobsEmpty.hidden = jobs.length > 0;
    if (!jobs.length) {
      this.jobsEmpty.textContent = "No jobs yet — start one above.";
    }
    this.jobsList.replaceChildren();
    for (const job of jobs) {
      const row = document.createElement("div");
      row.className = "job-row";

      const head = document.createElement("div");
      head.className = "job-head";
      const state = document.createElement("span");
      state.className = `job-state ${job.state}`;
      state.textContent = job.state;
      const label = document.createElement("span");
      label.className = "job-label";
      label.textContent = job.label || job.kind;
      head.append(state, label);
      if (job.state === "queued" || job.state === "running") {
        const cancel = document.createElement("button");
        cancel.className = "job-cancel";
        cancel.textContent = "Cancel";
        cancel.addEventListener("click", async () => {
          cancel.disabled = true;
          await cancelJob(job.id);
          this.refreshJobs();
        });
        head.append(cancel);
      }
      const elapsed = document.createElement("span");
      elapsed.className = "job-elapsed";
      elapsed.textContent = job.elapsed === null ? "" : `${job.elapsed.toFixed(1)}s`;
      head.append(elapsed);
      row.append(head);

      const cmd = document.createElement("div");
      cmd.className = "job-cmd";
      cmd.textContent = job.command;
      cmd.title = "Click to copy — this is exactly what runs, so it is reproducible in a terminal";
      cmd.addEventListener("click", () => navigator.clipboard?.writeText(job.command));
      row.append(cmd);

      if (job.tail && job.tail.length) {
        const tail = document.createElement("div");
        tail.className = "job-tail";
        tail.textContent = job.tail.slice(-3).join("\n");
        row.append(tail);
      }
      if (job.error) {
        const err = document.createElement("div");
        err.className = "job-tail";
        err.style.color = "#e0625f";
        err.textContent = job.error;
        row.append(err);
      }
      this.jobsList.append(row);
    }
  }

  async refreshSuggestions() {
    const epoch = this.epoch;
    let payload;
    try {
      payload = await fetchSuggestions();
    } catch (_) {
      return;
    }
    if (epoch !== this.epoch) return; // the previous recording's queue
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
      state.textContent = "labeled";
      state.title = "This frame now carries ground truth — done for this round.";
      wcell.append(state);
    }
    // The reviewed tick, IN the queue row. It used to be a read-only chip here and a real
    // checkbox only in the Labeled tab, so ticking a frame meant leaving the tab you work
    // from -- and the flag got forgotten wholesale: an audit found three recordings with
    // 22 finished frames (42-148 GT cells each, all DRAGGED) and not one tick between
    // them, while the other thirteen were ticked 100%. That is the signature of a flag
    // that lives somewhere other than where the work happens.
    //
    // Offered on EVERY suggested frame, not only labeled ones: under the auto-correction
    // workflow "I looked and the predictions were right" is a real review with nothing to
    // store, and the flag is the only place that fact can go.
    wcell.append(this.reviewedTick(s.frame, reviewed));
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
    // One tally, on the Suggested tab. It used to be mirrored onto the toolbar's panel
    // button; the rail that replaced that button is too narrow for "<done> / <total>", and a
    // queue is worked from its own tab anyway.
    const badge = this.suggestTallyEl;
    badge.hidden = !s?.present;
    badge.textContent = `${done} / ${total}`;
    badge.classList.toggle("is-zero", total === 0);
    badge.title = s?.present ? `${done} of ${total} suggested frames labeled` : "";

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
    // Labeled but never ticked, counted over the WHOLE recording rather than just the
    // queue. This is the standing check the corpus needed and did not have: an audit of
    // 16 recordings found 22 finished frames carrying 42-148 hand-dragged GT cells each
    // with the flag unset, and three whole recordings at zero ticks. Nothing in the editor
    // said so, because the only place the state was visible was a column in the other tab.
    // Now `reviewed` decides which frames are trusted for pretraining, so a silent
    // disagreement between work done and work ticked is a data-integrity bug, not cosmetics.
    const unticked = this.correctedFrames.filter((f) => !f.reviewed).length;
    if (unticked > 0) {
      lines.push({
        text:
          `${unticked} labeled frame${unticked === 1 ? "" : "s"} in this recording ` +
          `${unticked === 1 ? "is" : "are"} not ticked reviewed`,
        cls: "warn",
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

  // This frame's one annotation skeleton, as a readout. There is exactly one per frame
  // and there cannot yet be more -- the labels format reserves an instance axis but
  // refuses any row with instance != 0 -- so this does not pretend to be a list, and it
  // carries no verbs: Create (g) and Reseed (Shift+G) stay on the frame row, where the
  // muscle memory is and where they deliberately do not share a hotspot. What it adds
  // that nothing else answers is how much of this frame is actually yours.
  renderInstance() {
    const V = this.meta.n_views;
    const P = this.meta.n_points;
    let placed = 0;
    if (this.fixedMask) {
      for (let v = 0; v < V; v++) {
        for (let pt = 0; pt < P; pt++) if (this.fixedMask[v]?.[pt]) placed++;
      }
    }
    this.instanceStateEl.textContent = this.hasInstance ? `${placed} / ${V * P}` : "none";
    const seed =
      this.seedMode === "copy" ? "each view's own detection" : "the triangulated detections";
    const rows = this.hasInstance
      ? [
          ["Skeleton", "created for this frame"],
          ["Your pixels", `${placed} of ${V * P} cells`],
          ["Seeded from", seed],
        ]
      : [
          ["Skeleton", "none yet — drag a joint, double-click one, or press g"],
          ["A new one seeds from", seed],
        ];
    this.instancesList.replaceChildren(
      ...rows.map(([k, v]) => {
        const row = document.createElement("div");
        row.className = "inst-row";
        const key = document.createElement("span");
        key.className = "inst-key";
        key.textContent = k;
        const val = document.createElement("span");
        val.className = "inst-val";
        val.textContent = v;
        row.append(key, val);
        return row;
      }),
    );
  }

  // -- the sidebar's tabs -----------------------------------------------------
  //
  // One pane on screen at a time behind a wrapping strip of chips. Two consequences drive
  // everything below: activating a tab is the lazy-load trigger (a pane nobody opens costs
  // nothing), and "leaving a tab" is a real event, which is where the landmark disarm and
  // the jobs poll's stop belong.

  // One-time: bind each tab, restore the persisted one, and wire the panel-level controls.
  buildSidebar() {
    for (const spec of SIDEBAR_TABS) {
      const tab = /** @type {HTMLButtonElement} */ (el(`tab-${spec.id}`));
      const body = el(spec.pane);
      this.tabs.set(spec.id, { ...spec, tab, body });
      tab.addEventListener("click", () => this.setSidebarTab(spec.id));
    }
    // A tab that no longer exists in the table, or one hidden for this session (Recording,
    // with no project), would otherwise restore as a blank panel with nothing lit.
    const stored = /** @type {TabId} */ (localStorage.getItem(SIDEBAR_TAB_KEY));
    if (this.tabs.has(stored) && !this.tabs.get(stored).tab.hidden) this.sidebarTab = stored;
    if (this.sidebarTab === "labeled" || this.sidebarTab === "suggest") {
      this.navList = this.sidebarTab;
    }
    // Painted WITHOUT the side effects: the fetch and the poll are fired once, below.
    this.paintTabs();
    // One control per direction, each where the operator is already looking: the ✕ inside the
    // panel closes it, the right-edge rail (which only exists while it is closed) opens it.
    this.sidebarRailBtn.addEventListener("click", () => this.openFrames());
    this.framesCollapseBtn.addEventListener("click", () => this.closeFrames());
    // The recording rows are patched in place, so their click handler is delegated to the
    // list once here rather than re-attached per row on every render (a re-render would
    // otherwise stack a second listener on every row it reused).
    this.recordingListEl.addEventListener("click", (ev) => {
      const row = /** @type {HTMLElement} */ (ev.target)?.closest?.(".rec-row");
      if (row instanceof HTMLButtonElement && !row.disabled && row.dataset.slug) {
        this.requestSwitch(row.dataset.slug);
      }
    });
    const open = localStorage.getItem(SIDEBAR_OPEN_KEY);
    if (open === null ? SIDEBAR_DEFAULT_OPEN : open === "1") this.openFrames();
    // Not closeFrames(): the panel is already hidden in the HTML, and the disarm it does has
    // nothing to disarm before buildViews() runs. Only the rail has to be revealed -- the
    // markup ships it hidden, since the panel ships open.
    else {
      this.sidebarRailBtn.hidden = false;
      this.syncTabEffects();
    }
    // Fill whichever pane came up showing. Deferred to a microtask so the whole sidebar is
    // painted first: refreshSettings and refreshRecordings are network round trips, and an
    // exception in one must not abort the rest of the build.
    queueMicrotask(() => this.tabActivated(this.sidebarTab));
  }

  /** @param {TabId} id  is this the pane on screen? */
  tabActive(id) {
    return this.sidebarTab === id;
  }

  /** The DOM half of a tab switch -- no fetching, no polling. */
  paintTabs() {
    for (const { id, tab, body } of this.tabs.values()) {
      const active = id === this.sidebarTab;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      body.hidden = !active;
    }
  }

  /** @param {TabId} id */
  setSidebarTab(id) {
    if (this.sidebarTab === id) return;
    // Leaving Landmarks disarms. An armed landmark that outlives the operator's attention
    // turns the next canvas click into a silent placement, and the pane that explains what
    // is armed is no longer on screen to say so.
    if (this.sidebarTab === "marks") this.armLandmark(-1);
    this.sidebarTab = id;
    this.paintTabs();
    localStorage.setItem(SIDEBAR_TAB_KEY, id);
    this.tabActivated(id);
    this.syncTabEffects();
  }

  // What a pane needs in order to have anything in it. Activating a tab IS the fetch
  // trigger, so an operator who never opens Settings never pays for /api/config.
  //
  // Called from two places, which is the whole reason it is a method: a tab the operator
  // clicks, and the tab that is ALREADY showing at boot (the default, or the persisted
  // one). Missing the second is invisible in the DOM -- the pane renders empty, which
  // reads as "there is nothing to show".
  /** @param {TabId} id */
  tabActivated(id) {
    if (id === "recordings") this.refreshRecordings();
    if (id === "suggest") this.refreshSuggestions();
    if (id === "marks") this.renderLandmarks();
    if (id === "settings") this.refreshSettings();
    if (id === "bundle") this.refreshBundleAdjust();
    // Showing a list makes it the one ↑/↓ step, and scrolls the current frame into it.
    if (id === "labeled" || id === "suggest") {
      this.setNavList(id);
      this.updateActiveFrameRow();
    }
  }

  // The one place that decides whether the jobs poll runs: its tab is showing AND the panel
  // is open. That second half closes a real leak -- closing the panel on the Jobs tab used
  // to leave it polling every two seconds behind a hidden aside, forever.
  syncTabEffects() {
    if (this.framesOpen && this.tabActive("jobs")) this.startJobsPolling();
    else this.stopJobsPolling();
  }

  // Show a pane from the keyboard: open the panel if it is shut, then activate the tab.
  /** @param {TabId} id */
  revealTab(id) {
    if (!this.framesOpen) this.openFrames();
    this.setSidebarTab(id);
  }

  /** @param {"labeled"|"suggest"} list @param {number} dir */
  stepList(list, dir) {
    this.setNavList(list);
    if (list === "suggest") this.jumpSuggested(dir);
    else this.jumpLabeled(dir);
  }

  // Which list ↑/↓ step: normally the list tab on screen, and from Jobs or Settings the last
  // list shown -- so the keys never go dead and never silently change meaning. Each list
  // tab's own tooltip says that it is steppable; there are no ↑/↓ buttons to keep in sync.
  /** @param {"labeled"|"suggest"} list */
  setNavList(list) {
    this.navList = list;
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
    // The queue knows exactly where the same keystroke goes next, and its entries are
    // scattered through the recording -- each one a jump the decoder has to seek for, so
    // warming it is worth more here than for a step.
    const after = walk[(next + dir + walk.length) % walk.length];
    this.goToFrame(walk[next].frame, { then: after.frame });
  }

  // -- close / shutdown -------------------------------------------------------

  // The Close button: stop the server outright when nothing is at stake, else ask whether
  // to save first. This is the ONLY prompt about unsaved labels the editor has -- a switch
  // keeps the recording it leaves in memory, so closing the server is the single act that
  // can lose hand work, and the question covers every recording at once.
  requestClose() {
    if (this.projectDirty) this.openCloseConfirm();
    else this.shutdown();
  }

  openCloseConfirm() {
    // Named, not counted: "unsaved labels" on its own reads as "in front of me", and after
    // an afternoon of switching the operator's next click depends on knowing it is flyC.
    const slugs = this.dirtySlugs();
    this.closeListEl.textContent = slugs.length ? ` in ${slugs.join(", ")}` : "";
    this.closeOverlay.hidden = false;
    this.closeConfirmOpen = true;
  }

  closeCloseConfirm() {
    this.closeOverlay.hidden = true;
    this.closeConfirmOpen = false;
  }

  // "Save & close": only stop the server once every save actually lands, so a failed write
  // leaves the editor open with the labels intact -- in whichever recording they belong to.
  async saveAndShutdown() {
    try {
      await this.save();
    } catch (_) {
      this.closeCloseConfirm();
      this.statusEl.textContent = "save failed — the editor is still open";
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
    const epoch = this.epoch;
    const s = await fetchScene(this.frame);
    if (epoch !== this.epoch) return; // a recording switch landed mid-flight
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
      // suggestion queue in rank order. They are the only way to step a list now (the head's
      // ↑/↓ pair is gone), so the frame row's tooltip and each list tab's own tooltip are
      // what teach them; Home/End jump to the first/last frame. All are registered as real
      // (non-global) bindings so `matches`->preventDefault suppresses the browser's scroll/history
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
      b.push({ key: "l", group: "cam", label: "l", desc: "Layout — flip between Grid (every camera) and Focus (one big + thumbnails)", run: () => this.toggleLayout() });
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
    b.push({ key: "s", group: "show", label: "s", desc: "Where an unplaced joint is drawn — the reprojection of its 3D, or the seed the skeleton started from", run: () => this.setNongtDisplay(this.nongtDisplay === "seed" ? "reprojection" : "seed") });
    b.push({ key: "g", group: "frame", label: "g", desc: "Create the annotation skeleton for this frame, seeded from the detections", run: () => this.createInstance() });
    b.push({ key: "G", shift: true, group: "frame", label: hint("G", ["shift"]), desc: "Reseed this frame from the detections, keeping every ground-truth pixel", run: () => this.reseedInstance() });
    b.push({ key: "t", group: "show", label: "t", desc: "Detected — the detector's own output, as a read-only reference", run: () => this.toggleDetected() });
    b.push({ key: "n", group: "show", label: "n", desc: "Keypoint names", run: () => this.toggleCheck(this.labelsCheck, () => this.applyLabels()) });
    if (has3d) {
      b.push({ key: "p", group: "show", label: "p", desc: "Reprojected skeleton (3D reprojection)", run: () => this.toggleCheck(this.projectedCheck, () => this.applyProjected()) });
      b.push({ key: "w", group: "show", label: "w", desc: "Reprojection-distance warning", run: () => this.toggleCheck(this.warnCheck, () => this.applyWarn()) });
    }
    // Gated on the camera count, not on `has_3d`: counting labeled views needs neither a rig nor a
    // solve, so this check is live on a fresh uncalibrated project -- but "two views" is
    // unsatisfiable with one camera, so the key would only ever flag every joint.
    if (multi) {
      b.push({ key: "u", group: "show", label: "u", desc: "Under-labeled joints — a gauge on every keypoint you have not yet labeled in two views of this frame", run: () => this.toggleCheck(this.coverCheck, () => this.applyCover()) });
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
    // Acting on the selection. Each key sets one orthogonal fact and each has a toggle in the
    // card that shows whether it is set: Enter / Backspace place and clear the GT pixel, e holds
    // the cell out of the training loss, x marks the joint absent from the animal, r retracts both
    // of the per-cell facts. There used to be 1 / 2 / 3 setting one of three mutually exclusive
    // "states", which is not the shape the data has: a cell can carry a pixel AND be held out.
    // Verbs, not states. A cell is not "set to Ground truth"; a GT pixel is created at the
    // position already drawn, or deleted, and the loss switch is flipped independently.
    b.push({ key: "Enter", group: "edit", label: "⏎", desc: "Create ground truth for the selection, at the position shown", run: () => this.createSelectionGt() });
    b.push({ key: "Backspace", group: "edit", label: "⌫", desc: "Delete the selection's ground truth (the detection / reprojection shows through again)", run: () => this.deleteSelectionGt() });
    b.push({ key: "Delete", hidden: true, label: "Delete", desc: "", run: () => this.deleteSelectionGt() });
    b.push({ key: "e", group: "edit", label: "e", desc: "Hidden (toggle) — hold the selected cell(s) out of the training loss. Its own switch: it does not move the joint, hide its marker, or affect triangulation, and it leaves any pixel you placed standing", run: () => this.toggleSelectionExclude() });
    b.push({ key: "r", group: "edit", label: "r", desc: "Reset the selection — retract both the ground truth and the exclusion", run: () => this.resetSelection() });
    b.push({ key: "x", group: "edit", label: "x", desc: "Absent — this keypoint is not on this animal (amputated / ablated). This frame, every view; press again to un-mark", run: () => this.toggleAbsentSelection("frame") });
    // Uppercase key rather than `shift: true`: for a non-mod binding this keymap takes
    // shift as implied by the key itself (see `matches`), the same way Shift+M works.
    b.push({ key: "X", group: "edit", label: hint("X", ["shift"]), desc: "Absent for the whole recording — the usual case, an animal that arrives with a leg already missing", run: () => this.toggleAbsentSelection("recording") });
    b.push({ key: "d", group: "frame", label: "d", desc: "Reviewed — mark this frame checked (done); press again to un-mark", run: () => this.toggleReviewedCurrent() });
    b.push({ key: "z", mod: true, group: "hist", label: hint("Z", ["mod"]), desc: "Undo", run: () => this.undo() });
    // Redo answers to both ⌘Y and ⇧⌘Z; the help shows whichever the platform expects
    // (⇧⌘Z is the macOS idiom, Ctrl+Y the Windows/Linux one) while the other stays a
    // hidden alias that still works.
    b.push({ key: "y", mod: true, group: "hist", hidden: IS_MAC, label: hint("Y", ["mod"]), desc: "Redo", run: () => this.redo() });
    b.push({ key: "z", mod: true, shift: true, group: "hist", hidden: !IS_MAC, label: hint("Z", ["mod", "shift"]), desc: "Redo", run: () => this.redo() });
    b.push({ key: "s", mod: true, global: true, group: "hist", label: hint("S", ["mod"]), desc: "Save labels (every unsaved recording)", run: () => this.save() });
    b.push({ key: "c", group: "panel", label: "c", desc: "Show / hide the 3D scene", run: () => this.toggleScene() });
    b.push({ key: "j", group: "panel", label: "j", desc: "Show / hide the side panel — the recording, labeled frames, the suggested queue, landmarks, jobs and settings", run: () => this.toggleFrames() });
    // Only a project session has other recordings to browse, so the key is not
    // advertised in the help of a bare results.h5 session that could not honor it.
    if (this.meta.project_root) {
      b.push({ key: "b", group: "panel", label: "b", desc: "Browse this project's recordings — and switch to another", run: () => this.revealTab("recordings") });
    }
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
    // The per-frame verbs, beside the navigation ladder rather than among the cell verbs:
    // they have frame scope, and grouping them with the selection taught the wrong one.
    out.push(section("This frame", this.bindingRows("frame")));

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

    // Marker vocabulary. What a marker says is now one of two things about the annotation
    // skeleton -- did you place this pixel, and can a human see the joint here -- plus two
    // read-only reference layers. It used to name four "point sources" for one cell; provenance
    // was deleted in labels v7 and the instance owns every position, so there is one skeleton
    // whose joints are yours or derived.
    const markers = [
      [
        `<i class="mk m-gt"></i>`,
        `<b>Ground truth</b> — a pixel you placed. Drag a joint to place one, or select and press <kbd>Enter</kbd>; the <b>GT</b> toggle in the card is pressed whenever the selection carries yours.`,
      ],
      [
        `<i class="mk m-pred"></i>`,
        `<b>Detected</b> — the detector's own output, straight from the 2D network (the fill fades as its confidence drops). A read-only reference: it is what your skeleton was seeded from, and it hides itself once the frame has one (<kbd>t</kbd> brings it back).`,
      ],
    ];
    if (this.meta.has_3d) {
      markers.push([
        `<i class="mk m-proj"></i>`,
        `<b>Derived</b> — a joint you have not placed, drawn where the multi-view 3D puts it. Label it in two views and the other cameras move to where the geometry says it is; that is the whole point of labeling across views. The reprojection is also its own dashed overlay (<kbd>p</kbd>), and <kbd>s</kbd> switches these joints to the seed the skeleton started from instead.`,
      ]);
    }
    // The one thing a marker has to say that nothing else can: that its position is a guess the
    // editor made, not something the geometry produced. With an ipsilateral-only detector every
    // contralateral keypoint is in this state, so it is the common case, not an edge one.
    markers.push([
      `<i class="mk m-placeholder"></i>`,
      `<b>Invented</b> — faint and dashed: nothing in this frame can place this joint (the detector predicted it in no view, and there is no 3D to reproject), so its position is only a guess — a neighbouring joint, the centre of the view. It is drawn so you can find and drag it; it never feeds the 3D solve, and <kbd>Enter</kbd> skips it rather than recording a made-up pixel as yours. Drag it to where the joint really is.`,
    ])
    // The one marker that annotates another rather than replacing it -- so it is the one whose
    // legend row has to say what it leaves alone, not just what it adds.
    markers.push([
      `<i class="mk m-hidden"></i>`,
      `<b>Hidden</b> — a bar struck through the joint: this <i>(keypoint, camera)</i> cell is <b>held out of the training loss</b>. Select and press <kbd>e</kbd>. It is its own switch, independent of everything else: the joint stays exactly where it was, keeps its own marker, keeps its bones, stays draggable, and the 3D does not move. Placing a pixel here is still worth doing — "this is where the joint is" and "do not train on it here" are two separate things to record.`,
    ]);
    markers.push([
      `<i class="mk m-absent"></i>`,
      `<b>Absent</b> — this keypoint is not on this animal: an amputated leg, an ablated antenna. A claim about the <i>animal</i>, so one gesture covers every frame and every view. Select the joint and press <kbd>x</kbd>. It draws as a dim grey ✕ with no bones, contributes nothing to the 3D, and is excluded from the training export. Press <kbd>x</kbd> again, or <kbd>Ctrl+Z</kbd>, to un-mark it — nothing underneath is lost.`,
    ]);
    const markerRows = markers
      .map(([m, d]) => `<div class="legend-row">${m}<span>${d}</span></div>`)
      .join("");
    const markerBlock = `<h3 class="legend-title">What a marker tells you</h3>`
      + `<p class="legend-note">There is one annotation skeleton per frame. Each of its joints is either <b>yours</b> — a pixel you placed — or <b>derived</b> from the joints you have placed in other views. That is <i>where</i> the joint is. On a separate axis, any cell can be marked <b>Hidden</b> (<kbd>e</kbd>) — <i>held out of the training loss</i> — and the two never interfere: a hidden cell is drawn, dragged and solved exactly as it would be without the mark, and a cell can carry your pixel and the mark at once.</p>`
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
    setTitle(
      "save",
      "Save the ground-truth labels of every recording holding unsaved work " +
        `(${hint("S", ["mod"])})`
    );
    // The frame row advertises the whole navigation ladder.
    const frameLabel = document.querySelector("#controls .frame-row label");
    if (frameLabel instanceof HTMLElement) {
      frameLabel.title =
        `Jump to a frame — ← / → step 1 · ${hint("←", ["shift"])} steps 10 · PgUp / PgDn jump 100 · ↑ / ↓ step the side panel's list · Home / End first / last`;
    }
    // Overlay toggles: the "Show" button's summary and the NMF-mesh chip.
    const mesh = hint("M", ["shift"]);
    setTitle(
      "show-toggle",
      `What is drawn on each camera (keyboard: h Hide all · n Names · s Unplaced-joint positions · t Detected${this.meta.has_3d ? " · p Reprojected 3D · w Reproj. warning" : ""}${this.meta.n_views > 1 ? " · u Under-labeled" : ""}${this.meta.has_nmf ? " · m NMF skeleton · Shift+M NMF mesh" : ""} · l Grid/Focus · 0 Fit every camera)`,
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
      `The selected point(s). Each button is a state: GT (a pixel you placed), Hidden (held out of the training loss), Absent (not on this animal) — pressed means it is set, and Reset retracts them. GT and Hidden are independent, so both can be pressed at once. Select with click / Ctrl+click (add) / double-click / Shift+drag (new set) / Ctrl+drag (add); a = all, v = this view, Esc clears.`,
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
    // An operator who stepped by `d` almost always steps by `d` again, so warm that one.
    this.goToFrame(this.frame + d, { step: d });
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
      if (this.showMenuOpen) {
        this.closeShowMenu();
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
      } else if (this.armedLandmark >= 0) {
        // Escape is the way out of an armed landmark that needs no aim: the others are
        // clicking its row again, leaving the Landmarks tab, and `j`.
        this.armLandmark(-1);
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
