# Plan — in-place recording switch + SLEAP-style stacked sidebar

> ## Revision — the `#switch-*` prompt is gone; unsaved work is project-wide
>
> Everywhere below that assumes a switch must be *guarded* — the `#switch-overlay` modal and
> its five ids, `POST /api/recordings/open`'s 409, `test_switching_with_unsaved_labels_
> asks_before_dropping_them` — is superseded. The server now **keeps every session it has
> opened** (`opened` in `create_app`), so a switch loses neither labels nor undo history:
> there is nothing to prompt about, and the modal was deleted. What replaced it: a `● unsaved`
> cue in the toolbar (+ a dot per affected row of the recording list), `POST /api/save-all`
> behind the one Save, and a close prompt that names every unsaved recording. Section 1's
> in-place rebuild is unaffected and still accurate.

> ## Revision — it shipped as a TAB STRIP, not stacked sections
>
> The stacked sidebar below was built and then replaced, on the operator's preference, by a
> strip of tabs showing one pane at a time. **Section 1 (the in-place rebuild) shipped as
> written and is still accurate. Sections 2, 4 and 5 describe the stacked layout and are now
> historical** — read them for the *reasoning* (what each pane costs to fill, which effects
> had to move), not for the shape.
>
> What the tab strip changes, and nothing else does:
>
> - **One pane, full column.** `#sidebar-tabs` (a wrapping strip of `.side-tab` chips) plus
>   `#sidebar-panes` (the seven `.sidebar-body` panes, one visible). The strip **wraps** to two
>   or three rows rather than scrolling: seven names do not fit a 240–340px column, and a tab
>   scrolled off the end is a panel nobody finds.
> - **`sidebarTab` replaces `openSections`** (a single id, not a set), persisted under
>   `localStorage["deeperfly.sidebar.tab"]`; `SIDEBAR_DEFAULT_TAB = "labeled"`. The panel's
>   own open/shut still lives in `deeperfly.sidebar.open` and still defaults to OPEN.
> - **"Leaving a tab" is a real event again**, which is where two things belong that the
>   stacked layout had to hang off *collapse*: the landmark disarm (`setSidebarTab`, plus
>   `closeFrames` and Escape) and the jobs poll's stop (`syncTabEffects` — its tab showing AND
>   the panel open, which is still the fix for the poll that outlived a hidden panel).
> - **Activating a tab is the lazy-load trigger** (`tabActivated`), exactly as expanding was.
> - **The counts moved onto the chips** (`labeled-count`, `suggest-tally`, `marks-count`,
>   `jobs-count`), so they stay readable from whichever pane is open — the job the section
>   headers used to do.
> - **The two per-list ↑/↓ pairs collapse back to one** in the panel head. It steps the list
>   tab on screen, and from Jobs/Settings it steps the last list shown (`navList`), which its
>   tooltip names.
> - **Section 0's conclusion survives, its reason does not.** The aside still defaults to open,
>   but not because a section header carries the open recording's slug: with one pane showing,
>   the slug is a `.pane-head` line inside the Recording pane (`#recording-name`), and the
>   browser tab's title is the only other place it appears. That was chosen knowingly.

Files (absolute): `/home/tlam/deeperfly/src/deeperfly/gui/web/index.html` (**H**), `.../static/app.js` (**J**), `.../static/styles.css` (**C**), `.../static/api.js`, `.../static/poseView.js`, `.../static/scene3d.js`, `.../static/types.js`, `/home/tlam/deeperfly/src/deeperfly/gui/server.py`, `/home/tlam/deeperfly/tests/test_gui_browser.py`, `.../test_gui_server.py`, `.../test_gui_recordings.py`.

**No server change is required.** `/ws` reads `session` out of the enclosing scope on every message (server.py:834) and `POST /api/recordings/open` rebinds that variable (server.py:667), so the already-open socket talks to the new session from the next message onward. Do not construct a new `EditSocket` — that re-runs the writer-slot handshake (server.py:806-812) and can silently flip this tab's role.

---

## 0. The one decision the brief did not make

Removing `#recording-wrap` (H:145-152) deletes the only on-screen readout of which recording is open. `test_the_toolbar_names_the_open_recording` (test_gui_browser.py:853) exists precisely because "the browser tab's title" was judged not good enough. With the aside hidden by default (H:184) the name would go back into the tab title.

**Plan's answer: the aside defaults to OPEN and its open/shut persists** (`localStorage["deeperfly.sidebar.open"]`), while the *Recordings section itself* stays collapsed — so the slug is always on screen in its header at zero I/O cost. This also matches SLEAP (docks visible by default) and is the same argument J:2067-2074 already makes about the `reviewed` flag: five of the seven sections would otherwise sit behind a keystroke nobody discovers.

Revert lever if unwanted: one constant, `SIDEBAR_DEFAULT_OPEN = false` in J. Everything else in this plan is unaffected.

---

# 1. The in-place rebuild

## 1.1 Split `buildControls()` (J:615-809)

`buildControls()` stays, runs **exactly once** from `init()` (J:562), and keeps every `addEventListener`, every `segmented()` construction + `append`, the `_detectedTitle` capture (J:666) and the localStorage warn restore (J:712-715).

New method **`applyMeta()`** takes only the meta-derived statements. It must be idempotent and is called by `buildControls()` (last statement) and by the rebuild.

```js
  // Everything in the editor's chrome that is DERIVED from /api/meta -- ranges, names,
  // and which controls exist at all. Split out of buildControls so a recording switch can
  // re-apply it without re-running the one-time wiring, which would double every listener
  // (two edits per click on undo/gt/absent) and append a second copy of every segmented
  // switch, still wired to `this`. Every statement here must be idempotent.
  applyMeta() {
    const last = Math.max(0, this.meta.n_frames - 1);
    // .max is set BEFORE .value: a range input clamps its value to its max, so doing it
    // the other way round would silently pin a longer recording to the old last frame.
    for (const input of [this.slider, this.number]) {
      input.min = "0";
      input.max = String(last);
      input.value = "0";
    }
    this.totalEl.textContent = `/ ${last}`;

    const multiCam = this.meta.n_views > 1;
    this.layoutArrangeSection.style.display = multiCam ? "" : "none";
    this.layoutArrangeRow.style.display = multiCam ? "" : "none";
    // A single-camera recording hides the Grid/Focus segment (above) and registers no `l`
    // binding (buildBindings), so a session left in focus would be stuck there with no
    // control that could get it out.
    if (!multiCam) { this.layout = "grid"; this.focused = 0; }
    this.layoutSwitch.set(this.layout);

    this.uncalBanner.hidden = this.meta.has_cameras !== false;
    this.projectedWrap.style.display = this.meta.has_3d ? "" : "none";
    const warnAvailable = this.meta.has_3d;
    this.warnSection.style.display = warnAvailable ? "" : "none";
    this.warnWrap.style.display = warnAvailable ? "" : "none";
    this.warnThresholdWrap.style.display = warnAvailable ? "" : "none";
    this.referenceSection.style.display = this.meta.has_nmf ? "" : "none";
    this.nmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.meshWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.sceneNmfWrap.style.display = this.meta.has_nmf ? "" : "none";
    this.sceneMeshWrap.style.display = this.meta.has_nmf ? "" : "none";

    // Which recording is open. The toolbar picker is gone, so this heading is the only
    // place outside the tab title that names it -- and it is the single most visible
    // stale-data bug if a switch forgets to re-run this.
    this.recordingsSection.hidden = !this.meta.project_root;
    this.recordingNameEl.textContent = this.meta.recording ?? "recording";

    // Server-mirrored editor settings: a fresh EditorState reverts both server-side
    // (state.py:183, state.py:189), so the switches follow the reset fields.
    this.nongtSwitch.set(this.nongtDisplay);
    this.seedSwitch.set(this.seedMode);
    // The widest the status readout can ever get is fixed by the point/camera names.
    this.reserveStatusNameWidth();
  }
```

Edits to `buildControls()`:

| line(s) | action |
|---|---|
| J:616-622 | **move** to `applyMeta` |
| J:629-631, 638 | **move** |
| J:634-637, 639 | keep (construction) — but the three `segmented()` blocks (J:634-639, 670-674, 676-680) must all run **before** the `applyMeta()` call at the end |
| J:651-660 | **delete** (tab strip) — replaced by `buildSidebar()`, §2 |
| J:663 | **move** |
| J:675, 681 | **move** (the `.set()` calls only; the `append` stays) |
| J:697-698, 705-708, 725-726, 731, 787-788 | **move** |
| J:754-755 | **move**, rewritten (`recordingsSection.hidden`, section-header name) |
| J:756-767 | **delete** (recording popover toggle + outside-click; the second document-scoped listener leak goes with it) |
| J:777-780 | **move** to `buildSidebar()` (§2) |
| end of method | add `this.applyMeta();` |

**Also meta-derived, outside `buildControls`** — the rebuild must re-run each: `applyOsHints()` (J:3242-3276, tooltip at 3263 is `has_3d`/`has_nmf`-conditional), `buildBindings()` (J:2967-3057), `updateDirty()` (J:1970-1973, `document.title` from `results_path`), `helpBuilt = false` (J:262; `buildHelp`/`buildLegend` read `has_3d`/`has_nmf`/`limbs`/`n_views`), `this.jobsActions.replaceChildren()` (J:2469-2470 is `built once` and J:2471 **captures `meta.recording` into every submit handler**), `renderLandmarks()` (J:2351, per-recording `observations`), `buildViews()` (`--cols` at J:813).

## 1.2 `buildViews()` teardown (J:811-850)

`buildViews` **appends** (`this.cells.push` J:840, `this.views.push` J:848). Running it twice gives `relayout()` (J:883) both generations of canvases, and `applyPoints`'s `p.points[v]` for `v >= n_views` throws.

**poseView.js — add a real teardown.** Replace the constructor's line 303:

```js
    // Stored, so a recording switch can actually stop it. An unstored observer cannot be
    // disconnected, which is why the old canvases could only be abandoned and hoped about.
    this.ro = new ResizeObserver(() => this.layoutAndDraw());
    this.ro.observe(canvas);
```

and add (after `loadFrame`, ~poseView.js:352):

```js
  /**
   * Drop this view. The nine listeners registered on the canvas in the constructor die
   * with the element, which the caller detaches -- but the ResizeObserver, a mid-drag
   * rAF and an in-flight image decode all outlive it, and the rAF would call
   * `cb.onDragging` with THIS recording's (view, point) against the NEXT one's session.
   */
  destroy() {
    this.ro.disconnect();
    if (this.rafId) { cancelAnimationFrame(this.rafId); this.rafId = 0; }
    this.dragging = null;
    this.pendingDrag = null;
    this.loadToken++;          // any decode still in flight is dropped by the guard at :348
    this.cb = NO_CALLBACKS;    // a captured pointer can still fire; it must reach nothing
  }
```

with, at module scope in poseView.js:

```js
/** A dead callback set for a destroyed view -- see PoseView.destroy. */
const NO_CALLBACKS = Object.freeze({
  onDragging() {}, onDragged() {}, onToggleFixed() {}, onSelect() {},
  onSelectRegion() {}, onSelectKeypointAllViews() {}, onBackground() {},
  onActiveView() {}, onHover() {}, onPlaceLandmark() {},
});
```

**Do not** re-`new Scene3D` on `#scene-canvas`: scene3d.js:112-119 registers 7 canvas listeners + an unstored `ResizeObserver` per instance, so that leaks a whole set per switch. Re-seed in place (step 7 below).

## 1.3 New fields on `App`

Add beside the existing timer fields (near J:286):

```js
  //: Bumped by every recording rebuild. Checked after every `await` in the refresh*
  //: methods, because there is no AbortController anywhere in this codebase: without it
  //: an /api/points fetch issued against the OLD session resolves after the swap and
  //: applyPoints' only defence (the frame guard, app.js:982) passes -- the rebuild
  //: navigates to frame 0, and the stale reply is for frame 0.
  epoch = 0;
  //: The in-flight rebuild, or null. The server broadcasts the reload to every socket
  //: INCLUDING the initiator (server.py:677), so the switching tab enters twice.
  /** @type {Promise<void> | null} */
  rebuilding = null;
  //: `flash`'s self-dismiss timer. Never declared before; it must be cancellable, or a
  //: notice from the old recording blanks the new one's status line four seconds in.
  _noticeTimer = 0;
```

**Also fix `save()` (J:1963-1964)** — its 3 s `setTimeout` handle is not stored, and `saveAndSwitch()` (J:1421-1431) awaits `save()` then switches, so that timer blanks the post-switch status:

```js
    this.flash("saved");   // was: statusEl.textContent = "saved" + an untracked setTimeout
```

## 1.4 Guard every `await` with `epoch`

```js
  async refreshPoints() {
    const epoch = this.epoch;
    const p = await fetchPoints(this.frame, this.mode, true);
    if (epoch !== this.epoch) return; // a recording switch landed while this was in flight
    this.applyPoints(p);
  }
```

Same three-line shape in `refreshScenePoints` (J:2925-2931), `refreshCorrected` (J:1980-1993), `refreshSuggestions` (J:2556-2565), `refreshRecordings` (J:1316-1327), `refreshJobs` (J:2448-2467), `refreshSettings` (J:2191-2216), `ensureSceneMesh` (J:2902-2922).

## 1.5 `resetRecordingState()`

```js
  // Everything the App holds that belongs to ONE recording. Anything missing here is the
  // previous animal's data presented as this one's -- and the [view][point] masks are
  // worse than misleading: a stale (view, point) whose view >= the new n_views makes
  // updateStatusWidget throw (app.js:1692, 1700-1704).
  //
  // Deliberately NOT reset:
  //   readOnly (523)        the writer slot is per SOCKET and the socket survives the
  //                         swap (server.py:794-812); resetting it silently promotes a reader
  //   closing (271)         setting it would permanently disable the unsaved-changes guard
  //                         (init, app.js:582-587) for the NEW recording's edits
  //   configSchemaCache(472) project-scoped, and the switch cannot leave the project
  //                         (server.py:604-618)
  //   meshGL / meshAssetLoaded (368-369)  /api/nmf/asset is the packaged mesh, identical
  //                         for every recording (server.py:424-434, lru_cache at :932)
  //   framesOpen / openSections           the operator's arrangement, not the recording's data
  resetRecordingState() {
    this.frame = 0;
    this.focused = 0;              // else relayout (app.js:884) hands replaceChildren undefined
    this.selection.clear();
    this.selAnchor = null;
    this.activeView = 0;
    this.hoverCell = null;
    this.fixedMask = null;
    this.projectedMask = null;
    this.absentMask = null;
    this.absentRecording = [];     // else the previous animal's amputations are announced
    this.detectedMask = null;
    this.excludedMask = null;
    this.hasInstance = false;
    this.skeletonCreateBtn.disabled = this.readOnly;
    this.chirality = null;
    this.reviewed = false;
    this.reviewedBtn.setAttribute("aria-pressed", "false");
    this.reviewedBtn.classList.remove("is-on");
    // The new session's undo stack is empty. The next payload will say so, but an enabled
    // Undo in the meantime offers history that belongs to the recording just closed.
    this.undoBtn.disabled = true;
    this.redoBtn.disabled = true;
    this.seedMode = "triangulate";        // state.py:183 -- the new EditorState's default
    this.nongtDisplay = "reprojection";   // state.py:189
    this.correctedFrames = [];
    this.frameRows.clear();
    this.suggestions = null;              // labels_suggest.json is per recording
    this.suggestRows.clear();
    // The listing marks the OPEN recording active and disables its row (app.js:1369-1371);
    // a kept payload marks the wrong one and offers a click-to-itself on the new one.
    this.recordings = null;
    this.recordingListEl.replaceChildren();
    this.navList = "labeled";
    this.addMod = false;
    this.overViews = false;
    document.body.classList.remove("adding");
  }
```

## 1.6 The new `rebuildForNewRecording()`

Rename `reloadForNewRecording` (J:1437-1443) → `rebuildForNewRecording`; update both call sites, **J:573** and **J:1418**. Rewrite the comment at J:1433-1436 and the `ReloadMessage` doc at types.js:250-254 ("The page rebuilds in place; it must not `location.reload()`, because the socket carries the writer slot").

```js
  // The open recording changed underneath this page (switched here, or in another tab).
  // Everything the editor built at boot came from /api/meta, so all of it is rebuilt --
  // in place, with no page load. The SOCKET is deliberately kept: /ws reads `session` out
  // of its enclosing scope on every message (server.py:834) and the switch handler rebinds
  // that variable (server.py:667), so the open socket already talks to the new session --
  // while reconnecting would re-run the writer-slot handshake (server.py:806-812) and
  // could turn this tab from writer to reader without telling anyone.
  //
  // `closing` is never set here. It permanently disables the beforeunload guard
  // (app.js:582-587), and the NEW recording's edits still need protecting.
  async rebuildForNewRecording() {
    // The server broadcasts the reload to every socket including the initiator
    // (server.py:677), so the switching tab arrives here twice: once from the push
    // (app.js:573) and once from doSwitch (app.js:1418). A page load made the second call
    // a no-op; an async rebuild does not.
    if (this.rebuilding) return this.rebuilding;
    this.rebuilding = (async () => {
      try {
        // 1 -- quiesce. Every one of these would otherwise fire against the new recording
        // carrying the old one's intent.
        clearTimeout(this._noticeTimer);
        clearTimeout(this.correctedTimer);
        clearTimeout(this.meshTimer);
        clearTimeout(this.sceneMeshTimer);
        this.stopJobsPolling();
        this.epoch++;      // invalidates every in-flight fetch (§1.4)
        this.editSeq++;    // BUMPED, never zeroed: zeroing risks an old reply matching again
        this.meshReq++;    // invalidates an in-flight refreshMesh (guard at app.js:1219)
        this.closeShowMenu();
        this.closeSkeletonMenu();
        this.closeHelp();
        this.closeSwitchConfirm();
        this.closeCloseConfirm();
        this.pendingSwitch = "";
        this.armLandmark(-1);   // fans out to the OLD views, and clears body.arming-landmark

        // 2 -- tear the canvases down. buildViews APPENDS (app.js:840, 848), so without
        // this both rigs would be live: relayout re-attaches the old canvases
        // (app.js:883) and applyPoints indexes p.points past its end and throws.
        for (const view of this.views) view.destroy();
        this.stageEl.replaceChildren();
        this.stripEl.replaceChildren();
        this.views = [];
        this.cells = [];

        // 3 -- new meta. This also re-stamps api.js's module-level cacheVersion
        // (api.js:25), which frameUrl reads at CALL time (api.js:126-133), so it must
        // precede goToFrame or the new frames go out under the old recording's token.
        this.meta = await fetchMeta();
        this.dirty = this.meta.dirty;

        // 4 -- per-recording state (§1.5)
        this.resetRecordingState();

        // 5 -- meta-derived chrome. Never the wiring.
        this.applyMeta();
        this.bindings = this.buildBindings();   // l, [, ], p, w, m, Shift+M, b are meta-gated.
                                                // onKey reads this.bindings at dispatch time
                                                // (app.js:3332), so reassigning is enough --
                                                // the keydown listener must NOT be re-added.
        this.applyOsHints();
        this.helpBuilt = false;
        this.jobsActions.replaceChildren();     // renderJobActions is `built once` and closed
                                                // over the OLD meta.recording (app.js:2471)

        // 6 -- rebuild the canvases, then fan out EVERY display toggle. New PoseViews take
        // their constructor defaults (poseView.js:209-236), which match the HTML at boot
        // but not the operator's current checkboxes.
        this.buildViews();
        this.applyHideAll();
        this.applyLabels();
        this.applyDetected();
        this.applyProjected();
        this.applyWarn();
        this.applyWarnThreshold();
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

        // 8 -- repaint. The lists are emptied BEFORE the fetch, so the old recording's
        // frame numbers are never on screen under the new recording's name.
        this.renderFrameList();
        this.renderSuggestList();
        this.renderLandmarks();
        this.renderInstance();
        await this.goToFrame(0);
        this.updateSelected();
        this.updateDirty();
        this.updateAbsentBadge();
        this.updateChiralityBadge();
        await this.refreshCorrected();
        this.refreshSuggestions();
        if (this.sectionOpen("recordings")) this.refreshRecordings();
        if (this.sectionOpen("settings")) this.refreshSettings();
        this.syncSectionEffects();              // restarts the jobs poll iff Jobs is expanded
        if (this.sceneOpen) await this.refreshScene();
        this.flash(`opened ${this.meta.recording}`);
      } finally {
        this.rebuilding = null;
      }
    })();
    return this.rebuilding;
  }
```

Also delete `this.closeRecordingMenu()` from `requestSwitch` (J:1387), `openShowMenu` (J:1268) and `openSkeletonMenu` (J:1459).

---

# 2. The stacked sidebar

## 2.1 New DOM (replaces H:175-243 wholesale)

```html
      <!-- The side panel: a stack of collapsible sections, not a tab strip, so several are
           on screen at once. Order is the order of the working day -- which recording, what
           you have labeled, what to label next, this frame's skeleton, the calibration
           landmarks, the jobs you can run, the project's settings. Each section keeps its
           OWN scroller, because `position: sticky` table headers pin to the nearest
           scrollport: one shared scroller and the two tables' headers would stack on top of
           each other. The three sections that cost real work to fill start collapsed and
           fetch nothing until expanded -- Recordings opens every recording's labels.h5
           (server.py:583-584), Settings re-composes the project config on every read
           (server.py:496-543), Jobs polls every two seconds. Which sections are open
           persists across reloads AND across a recording switch: it is the operator's
           arrangement, not the recording's data. `j` toggles the whole panel. -->
      <aside id="sidebar" hidden>
        <div class="sidebar-head">
          <span class="sidebar-title">Panels</span>
          <button id="frames-collapse" class="icon-btn" aria-label="Hide" title="Hide the panel (j)">✕</button>
        </div>
        <div id="sidebar-sections" class="sidebar-sections">

          <!-- Which recording is open, and the project's others with the counts that decide
               which is worth opening next. The HEADING carries the open slug and stays
               readable while the list is collapsed: after the toolbar picker was retired
               this is the only place outside the browser tab's title that names it. -->
          <section class="side-sec is-compact" id="sec-recordings" hidden>
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-recordings-toggle"
                      aria-expanded="false" aria-controls="recording-pane"
                      title="This project's recordings — and switch to another without restarting the editor (b)">
                <span class="sec-caret" aria-hidden="true">▸</span>
                <span class="sec-name">Recording</span>
                <span id="recording-name" class="sec-value">recording</span>
              </button>
            </h2>
            <div class="sidebar-body" id="recording-pane" hidden>
              <div id="recording-list" class="rec-list"></div>
              <div id="recording-empty" class="frames-empty" hidden></div>
            </div>
          </section>

          <!-- Every frame carrying a ground-truth label, in time order. -->
          <section class="side-sec" id="sec-labeled">
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-labeled-toggle"
                      aria-expanded="true" aria-controls="labeled-pane">
                <span class="sec-caret" aria-hidden="true">▾</span>
                <span class="sec-name">Labeled frames</span>
                <span id="labeled-count" class="count-badge">0</span>
              </button>
              <span class="sec-nav" id="labeled-nav">
                <button id="frames-prev" class="icon-btn" title="Previous labeled frame">↑</button>
                <button id="frames-next" class="icon-btn" title="Next labeled frame">↓</button>
              </span>
            </h2>
            <div class="sidebar-body" id="labeled-pane">
              <table id="frames-table" class="frames-table">
                <thead><tr><th>Frame</th><th class="reviewed-col" title="Tick once you have finished reviewing a frame">Reviewed</th></tr></thead>
                <tbody></tbody>
              </table>
              <div id="frames-empty" class="frames-empty">No labels yet — frames you annotate will be listed here.</div>
            </div>
          </section>

          <!-- The ranked queue written by `deeperfly labels-suggest`: which frames to
               correct NEXT, worst multi-view disagreement first, in rank order. A different
               set in a different order from the list above, which is why it is its own
               section with its own columns and its own ↑/↓. The status strip carries the two
               facts most likely to mislead a reader -- how stale the queue is, and any
               caveat about how it was scored. -->
          <section class="side-sec" id="sec-suggest">
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-suggest-toggle"
                      aria-expanded="true" aria-controls="suggest-pane">
                <span class="sec-caret" aria-hidden="true">▾</span>
                <span class="sec-name">Suggested</span>
                <span id="suggest-tally" class="count-badge is-suggest" hidden>0</span>
              </button>
              <span class="sec-nav" id="suggest-nav">
                <button id="suggest-prev" class="icon-btn" title="Previous suggested frame, in rank order">↑</button>
                <button id="suggest-next" class="icon-btn" title="Next suggested frame, in rank order">↓</button>
              </span>
            </h2>
            <div class="sidebar-body" id="suggest-pane">
              <div id="suggest-status" class="sidebar-status" hidden></div>
              <table id="suggest-table" class="frames-table suggest-table">
                <thead><tr>
                  <th class="rank-col" title="Rank in the queue — 1 is the frame most worth your time">#</th>
                  <th>Frame</th>
                  <th class="t-col" title="Time into the recording, in seconds">t (s)</th>
                  <th class="score-col" title="Multi-view disagreement of the detector's own 2D: how far each view's prediction sits from the reprojection of the triangulated joint, averaged over the worst joints. Ranks WITHIN this recording only — the absolute level tracks how many cells the detector fired, so it is not comparable between recordings.">Score</th>
                </tr></thead>
                <tbody></tbody>
              </table>
              <div id="suggest-empty" class="frames-empty"></div>
            </div>
          </section>

          <!-- This frame's annotation skeleton. One per frame is all the format can carry:
               the labels COO index reserves an instance axis (labels.py:171-180) but the
               loader REFUSES any row with instance != 0, and EditorState keeps a single
               per-frame bool (state.py:115). So this reports; it does not list, and it
               carries no verbs -- Create and Reseed stay on the frame row, where the muscle
               memory is and where they deliberately do not share a hotspot (index.html:53-59). -->
          <section class="side-sec is-compact" id="sec-instances">
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-instances-toggle"
                      aria-expanded="true" aria-controls="instances-pane">
                <span class="sec-caret" aria-hidden="true">▾</span>
                <span class="sec-name">Instance</span>
                <span id="instance-state" class="sec-value">—</span>
              </button>
            </h2>
            <div class="sidebar-body" id="instances-pane">
              <div id="instances-list" class="inst-list"></div>
            </div>
          </section>

          <!-- Calibration landmarks. Non-skeleton points that make a from-scratch rig
               solvable: a STATIC one (a coverslip scratch, the tether tip) is three unknowns
               however many frames observe it, while a keypoint is three unknowns PER FRAME
               because the animal moved. Arm one, then click it into place in each view --
               and COLLAPSING this section disarms, because with every section on screen at
               once there is no "leaving the tab" event to do it. -->
          <section class="side-sec is-compact" id="sec-marks">
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-marks-toggle"
                      aria-expanded="true" aria-controls="marks-pane">
                <span class="sec-caret" aria-hidden="true">▾</span>
                <span class="sec-name">Landmarks</span>
                <span id="marks-count" class="count-badge">0</span>
              </button>
            </h2>
            <div class="sidebar-body" id="marks-pane">
              <div id="marks-list" class="marks-list"></div>
              <div id="marks-empty" class="frames-empty"></div>
            </div>
          </section>

          <!-- Pipeline jobs. Each row IS a CLI command, shown verbatim so a GUI action that
               fails is reproducible in a terminal without reverse-engineering what the GUI
               did. There is no percentage: these commands emit human log lines, and a number
               synthesized from those would be a fiction with a spinner attached, so the last
               log line is shown instead. Collapsed by default because expanding it starts a
               two-second poll -- 1800 requests an hour on an idle editor. -->
          <section class="side-sec" id="sec-jobs">
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-jobs-toggle"
                      aria-expanded="false" aria-controls="jobs-pane">
                <span class="sec-caret" aria-hidden="true">▸</span>
                <span class="sec-name">Jobs</span>
                <span id="jobs-count" class="count-badge" hidden>0</span>
              </button>
            </h2>
            <div class="sidebar-body" id="jobs-pane" hidden>
              <div id="jobs-actions" class="jobs-actions"></div>
              <div id="jobs-list" class="jobs-list"></div>
              <div id="jobs-empty" class="frames-empty"></div>
            </div>
          </section>

          <!-- Project settings. Every field, its default, its type and its documentation are
               DERIVED from the *Params dataclasses (GET /api/schema), so this panel cannot
               drift from the code and a new option appears here with nothing to update.
               Writes go to the project's profile, which holds only what differs from the
               defaults -- so "reset" genuinely removes the key. Collapsed by default: each
               read is a Project.load + compose_config + two TOML parses (server.py:496-543),
               and it re-reads after every knob write. -->
          <section class="side-sec" id="sec-settings">
            <h2 class="sec-head">
              <button type="button" class="sec-toggle" id="sec-settings-toggle"
                      aria-expanded="false" aria-controls="settings-pane">
                <span class="sec-caret" aria-hidden="true">▸</span>
                <span class="sec-name">Settings</span>
              </button>
            </h2>
            <div class="sidebar-body" id="settings-pane" hidden>
              <div id="settings-list" class="settings-list"></div>
              <div id="settings-empty" class="frames-empty"></div>
            </div>
          </section>

        </div>
      </aside>
```

Deleted from H: `#sidebar-tabs` (186), the aside-level `.sidebar-nav` (187-190), the orphaned Jobs doc comment (216-220) — now above its own section — and the whole `#recording-wrap` group (145-152). `#frames-toggle` (H:157) stays with **both** badges: it is the collapsed-panel summary, and the "how much have I labeled" number must not vanish when the panel is shut.

## 2.2 CSS (C)

| C lines | change |
|---|---|
| **669** | `flex: 0 0 clamp(240px, 20vw, 340px);` — the Suggested width becomes the one width |
| **677-679** | **delete** `#sidebar.tab-suggest` (the per-tab override, toggled at J:2747) |
| **681-683** | keep `#sidebar[hidden]{display:none}` |
| **685-693** | keep `.sidebar-head`; **reuse its shape** for `.sec-head` |
| **697-699** | keep `.sidebar-title { flex: 1 }` |
| **701-704** | **delete** `.sidebar-title .segmented .seg-btn` (orphaned with the strip) |
| **706-709** | **delete** the aside-level `.sidebar-nav`; replaced by `.sec-nav` |
| **711-715** | **rewrite** — the core layout change |
| 717-796, 803-958, 1303-1443, 1464-1498 | keep unchanged (tables, sticky headers, chips, jobs, marks, settings, `.rec-*`) |
| **1451-1463** | **delete** `#recording-menu` and `#recording-toggle #recording-name` (the popover positioning + the toolbar ellipsis) |
| 500-525 | keep `.segmented` — still used by Layout / Unplaced-joints / Seed |
| 626-662 | keep `.count-badge`; now used on both `#frames-toggle` and the section headers |

New block, replacing C:706-715:

```css
/* The section column. Each open section takes a share of the aside's height and keeps its
   OWN scroller -- `.frames-table th` is `position: sticky; top: 0` (below), which pins to
   the nearest scrollport, so a single shared scroller would stack both tables' headers on
   top of each other. When enough sections are open that their minimums no longer fit, this
   column is the overflow valve. */
.sidebar-sections {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.side-sec {
  display: flex;
  flex-direction: column;
  flex: 0 0 auto;
  border-bottom: 1px solid var(--border);
}

.side-sec[hidden] { display: none; }

/* An open LIST section shares the leftover height with its peers but never shrinks below a
   usable list. A `.is-compact` one (Recordings, Instance, Landmarks: short, bounded content)
   sizes to its content instead, so it cannot squeeze the lists that actually need the room. */
.side-sec.is-open { flex: 1 1 auto; min-height: 140px; }
.side-sec.is-open.is-compact { flex: 0 0 auto; min-height: 0; }
.side-sec.is-open.is-compact > .sidebar-body { max-height: 32vh; }

/* The section header: the same 32-ish px bar the aside head is, with a disclosure button
   filling it and any per-section tools to its right. The disclosure is its own <button>
   rather than the whole row, because a nested button (the ↑/↓ pair) is invalid inside one. */
.sec-head {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  padding: 0 6px 0 0;
  background: #333;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
}

.sec-toggle {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  padding: 8px 4px 8px 10px;
  background: none;
  border: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.sec-toggle:hover { background: #3a3a3a; }
.sec-caret { width: 10px; color: #999; flex: none; }
.sec-name { flex: none; }

/* The header's own readout slot: the open recording's slug, the frame's GT-cell count.
   Ellipsised rather than wrapped -- the header must stay one line high. */
.sec-value {
  margin-left: auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #9aa7b4;
  font-weight: 400;
  font-size: 11px;
}

/* A list section's own ↑/↓ pair. The pair that currently owns the ↑/↓ KEYS is marked, so
   which list a keystroke steps is shown rather than remembered. */
.sec-nav { display: inline-flex; gap: 2px; flex: none; }
.sec-nav.is-nav .icon-btn { color: #cfe3f7; box-shadow: inset 0 -2px 0 #4a90d9; }

.sidebar-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

/* `.sidebar-body` is a flex child with no `display` of its own, so the UA `[hidden]` rule
   would work -- this is here because every other collapsible surface in this file needed
   the author rule (.count-badge C:648, .popover-menu C:201) and the next edit to add a
   `display` here would break the collapse silently. */
.sidebar-body[hidden] { display: none; }

/* The Instance readout: label/value pairs, no verbs. */
.inst-list { padding: 6px 10px 10px; }
.inst-row { display: flex; gap: 8px; padding: 3px 0; font-size: 12px; }
.inst-row .inst-key { color: #8a97a4; flex: none; }
.inst-row .inst-val { margin-left: auto; text-align: right; }
```

One `.rec-row` adjustment (C:1469-1482), since `#recording-menu`'s `min-width: 320px` is gone and rows now live at 240-340px:

```css
.rec-row { flex-wrap: wrap; }          /* the stats line drops below the name at 240px */
```

## 2.3 JS — what replaces `setSidebarTab`

**Delete:** `SidebarTab` typedef (J:64), the `sidebarTab` field + its comment (J:286-295), `sidebarTabs` (J:382-384, including the duplicated stray JSDoc at 382), `sidebarTabsEl` (J:449-450), the strip construction (J:651-660), `setSidebarTab` (J:2735-2760), and the six `sidebarTab` reads at J:2124, 2152, 2402, 2421, 2764 and 2747. Also delete `recordingWrap`/`recordingToggle`/`recordingMenu` fields (J:399-403), `recordingMenuOpen` (J:410), `openRecordingMenu`/`closeRecordingMenu`/`toggleRecordingMenu` (J:1293-1313) and the Escape branch at J:3307-3309. **Keep** `segmented()` (J:104-135) — four other users.

Module scope, near the `WARN_*` keys (J:77-80):

```js
/** @typedef {"recordings"|"labeled"|"suggest"|"instances"|"marks"|"jobs"|"settings"} SectionId */

// The sidebar's sections, in DOM order. `open` is the DEFAULT disclosure, and the three
// that are shut cost real work to fill: Recordings opens every recording's labels.h5
// (server.py:583-584), Settings re-composes the project config and parses two TOML
// documents per read (server.py:496-543), Jobs starts a 2 s poll (app.js:2437). The
// expand hook IS the lazy-load contract, which is why the sections that render from `meta`
// alone (Instance, Landmarks) have nothing to run.
const SIDEBAR_SECTIONS = [
  { id: "recordings", pane: "recording-pane", open: false },
  { id: "labeled",    pane: "labeled-pane",   open: true },
  { id: "suggest",    pane: "suggest-pane",   open: true },
  { id: "instances",  pane: "instances-pane", open: true },
  { id: "marks",      pane: "marks-pane",     open: true },
  { id: "jobs",       pane: "jobs-pane",      open: false },
  { id: "settings",   pane: "settings-pane",  open: false },
];
const SECTIONS_KEY = "deeperfly.sidebar.sections";
const SIDEBAR_OPEN_KEY = "deeperfly.sidebar.open";
const SIDEBAR_DEFAULT_OPEN = true;
```

New fields (replacing J:286-299's tab block):

```js
  /** @type {Map<SectionId, {id: SectionId, pane: string, open: boolean,
   *   root: HTMLElement, toggle: HTMLButtonElement, body: HTMLElement}>} */
  sections = new Map();
  /** @type {Set<SectionId>} which sections are expanded. Persisted, and preserved across a
   *  recording switch -- it is the operator's arrangement, not the recording's data. */
  openSections = new Set();
  //: Which list the ↑/↓ KEYS step. With both lists on screen there is no "active tab" to
  //: infer it from, so it is explicit, set by the last list the operator actually used, and
  //: SHOWN (that list's nav pair carries `.is-nav`) rather than guessed at.
  /** @type {"labeled" | "suggest"} */
  navList = "labeled";
```

New methods:

```js
  // One-time: bind each section's disclosure, restore the persisted arrangement, and wire
  // the panel-level controls. Called from init() in place of the tab strip.
  buildSidebar() {
    const saved = this.readOpenSections();
    for (const spec of SIDEBAR_SECTIONS) {
      const root = el(`sec-${spec.id}`);
      const toggle = el(`sec-${spec.id}-toggle`);
      const body = el(spec.pane);
      this.sections.set(spec.id, { ...spec, root, toggle, body });
      toggle.addEventListener("click", () => this.toggleSection(spec.id));
      if (saved ? saved.has(spec.id) : spec.open) this.openSections.add(spec.id);
      // Painted without the side effects: restoring four open sections must not fire four
      // rounds of fetch/poll mid-build. syncSectionEffects (below) does that once.
      this.paintSection(spec.id);
    }
    this.framesToggleBtn.addEventListener("click", () => this.toggleFrames());
    this.framesCollapseBtn.addEventListener("click", () => this.closeFrames());
    this.framesPrevBtn.addEventListener("click", () => this.stepList("labeled", -1));
    this.framesNextBtn.addEventListener("click", () => this.stepList("labeled", 1));
    this.suggestPrevBtn.addEventListener("click", () => this.stepList("suggest", -1));
    this.suggestNextBtn.addEventListener("click", () => this.stepList("suggest", 1));
    this.updateSidebarNavTitles();
    const stored = localStorage.getItem(SIDEBAR_OPEN_KEY);
    if (stored === null ? SIDEBAR_DEFAULT_OPEN : stored === "1") this.openFrames();
    else this.syncSectionEffects();
  }

  /** @returns {Set<SectionId> | null} the persisted arrangement, or null for "never set" */
  readOpenSections() {
    try {
      const raw = localStorage.getItem(SECTIONS_KEY);
      return raw === null ? null : new Set(JSON.parse(raw));
    } catch (_) {
      return null; // a corrupt key falls back to the defaults rather than blanking the panel
    }
  }

  /** @param {SectionId} id */
  sectionOpen(id) { return this.openSections.has(id); }

  /** The DOM half of a disclosure -- no fetching, no polling. @param {SectionId} id */
  paintSection(id) {
    const sec = this.sections.get(id);
    const open = this.openSections.has(id);
    sec.root.classList.toggle("is-open", open);
    sec.body.hidden = !open;
    sec.toggle.setAttribute("aria-expanded", String(open));
    sec.toggle.querySelector(".sec-caret").textContent = open ? "▾" : "▸";
  }

  /** @param {SectionId} id @param {boolean} open */
  setSection(id, open) {
    if (this.openSections.has(id) === open) return;
    if (open) this.openSections.add(id);
    else this.openSections.delete(id);
    this.paintSection(id);
    localStorage.setItem(SECTIONS_KEY, JSON.stringify([...this.openSections]));
    if (open) {
      // Expanding is the fetch trigger -- the contract setSidebarTab used to carry
      // (app.js:2749-2755). It is per-section now, so an operator who never opens Settings
      // never pays for /api/config.
      if (id === "recordings") this.refreshRecordings();
      if (id === "suggest") this.refreshSuggestions();
      if (id === "marks") this.renderLandmarks();
      if (id === "settings") this.refreshSettings();
      if (id === "labeled" || id === "suggest") this.updateActiveFrameRow();
    } else if (id === "marks") {
      // Collapsing disarms, and nothing else can: with every section on screen at once
      // there is no "leaving the tab" event. An armed landmark that outlives the
      // operator's attention turns the next canvas click into a silent placement.
      this.armLandmark(-1);
    }
    this.syncSectionEffects();
  }

  /** @param {SectionId} id */
  toggleSection(id) { this.setSection(id, !this.openSections.has(id)); }

  // The one place that decides whether the jobs poll runs: the section is expanded AND the
  // panel is open. The second half closes a leak the tab strip had -- closeFrames
  // (app.js:2136-2139) never stopped the poll, so a hidden panel left on the Jobs tab
  // polled forever.
  syncSectionEffects() {
    if (this.framesOpen && this.sectionOpen("jobs")) this.startJobsPolling();
    else this.stopJobsPolling();
  }

  // Show a section from the keyboard: open the panel if shut, expand, and scroll its header
  // into the column's scroller -- without that, `b` on a tall arrangement expands something
  // off screen and looks like it did nothing.
  /** @param {SectionId} id */
  revealSection(id) {
    if (!this.framesOpen) this.openFrames();
    this.setSection(id, true);
    this.sections.get(id).root.scrollIntoView({ block: "nearest" });
  }
```

`openFrames`/`closeFrames` (replacing J:2130-2139):

```js
  openFrames() {
    this.sidebarEl.hidden = false;
    this.framesOpen = true;
    localStorage.setItem(SIDEBAR_OPEN_KEY, "1");
    this.syncSectionEffects();
    this.updateActiveFrameRow(); // scroll the current frame into view now it is shown
  }

  closeFrames() {
    this.sidebarEl.hidden = true;
    this.framesOpen = false;
    localStorage.setItem(SIDEBAR_OPEN_KEY, "0");
    // A landmark armed from a panel that is now hidden is exactly the surprise the old
    // "leaving the tab disarms" rule guarded against.
    this.armLandmark(-1);
    this.syncSectionEffects();
  }
```

`armLandmark` (J:2402) and `onPlaceLandmark` (J:2421): `if (this.sidebarTab === "marks")` → `if (this.sectionOpen("marks"))`. This also **fixes a live bug** the map found: `setSidebarTab`'s `else this.armLandmark(-1)` (J:2756-2758) binds to the `settings` test, not the `marks` one, so today selecting Landmarks disarms itself and switching Landmarks→Settings does *not* disarm.

`init()` (J:557-575): replace `this.buildControls()` ordering so `buildSidebar()` runs after `buildControls()` and before `buildViews()`; add `this.renderInstance();` after `await this.goToFrame(0)`.

Badges — two targets each, single call sites:

```js
  // renderFrameList, replacing app.js:2005-2006
  const n = this.correctedFrames.length;
  for (const badge of [this.framesCountEl, this.labeledCountEl]) {
    badge.textContent = String(n);
    badge.classList.toggle("is-zero", n === 0);
  }
```
```js
  // renderSuggestStatus, replacing app.js:2673-2678
  for (const badge of [this.suggestCountEl, this.suggestTallyEl]) {
    badge.hidden = !s?.present;
    badge.textContent = `${done} / ${total}`;
    badge.classList.toggle("is-zero", total === 0);
    badge.title = s?.present ? `${done} of ${total} suggested frames labeled` : "";
  }
```
`renderLandmarks` (J:2362) sets `#marks-count`; `renderJobs` (J:2497) sets `#jobs-count` to the queued+running count and `hidden` when zero.

## 2.4 The cost, stated

| endpoint | handler | cost | when it runs now |
|---|---|---|---|
| `/api/corrected` | server.py:458-465 → state.py:385-406 | in-memory numpy over `(V,T,P)`, no I/O | unchanged: init + debounced per settled edit |
| `/api/suggestions` | server.py:467-476, `_read_suggestions` :1427-1446 | **re-reads + JSON-parses the sidecar from disk**, plus a `corrected_frames()` | init, every save, and on Suggested expand — same order as today (2-3 per session) |
| `/api/jobs?tail=3` | server.py:692-706 | cheap rows, **but every 2 s** | only while Jobs is expanded **and** the panel is open. Collapsed by default; the `closeFrames` leak is closed |
| `/api/schema` | server.py:330-363 | pure reflection, client-cached (J:472) | once, on first Settings expand |
| `/api/config` | server.py:496-543 | **`Project.load` + `compose_config` + 2 TOML parses + `profile_values` per call** | only on Settings expand and after each knob write. Collapsed by default |
| `/api/recordings` | server.py:576-602 → project.py:1186-1211 | **one HDF5 open per recording** (26 on this lab's corpus) | only on Recordings expand, same as the popover did (J:1300). Never polled |

Net at boot with the defaults: exactly what loads today (`/api/meta`, `/api/points/0`, `/api/corrected`, `/api/suggestions`) plus zero. The persisted arrangement can re-open an expensive section; that is the operator's explicit choice, and each still fetches on expand rather than on a timer.

---

# 3. Instances — there is nothing to list

Plainly: **there is no instance UI to move, and no data to build one from.** The whole surface is:

- `hasInstance`, one boolean for the current frame (J:249-251), set from `p.has_instance` (J:1004-1009).
- Two verbs, on the frame row, not the sidebar: `#skeleton-create` (H:66 / J:682) and `#reseed` (H:72 / J:683-686), keys `g` (J:3004) and `Shift+G` (J:3005), plus the implicit create on double-click (J:1551-1553).
- The seed-mode switch, `#seed-switch` (H:70 / J:676-681).
- Server: `p.instance` is returned (server.py:1161) and **never read by app.js**; `EditorState` keeps `instance: bool` per frame (state.py:115).
- Storage: the labels COO index is `[view, frame, instance, point]` (labels.py:171-180) with the instance column reserved but always 0, and **a row carrying a different instance is refused on load** (labels.py:179-180).

A SLEAP Instances dock lists animals per frame with tracks and add/delete. There is no identity, no name, no color, no track, no second instance — and the loader actively rejects one. Re-homing Create/Reseed into the sidebar would only duplicate `#skeleton-wrap` and give Reseed (which rewrites every seed in every view) a second hotspot, which H:53-59 explicitly reasons against.

**Recommendation: ship the section as a readout, with no verbs.** It occupies the requested slot honestly, documents the one-instance constraint, and answers one question nothing currently answers — how much of this frame you have actually placed. `renderInstance()` is called from `applyPoints` (after J:1064) and from `setSeedMode` (J:1091-1095):

```js
  // This frame's one annotation skeleton, as a readout. There is exactly one per frame and
  // there cannot yet be more -- the labels format reserves the instance axis but refuses
  // any row with instance != 0 (labels.py:171-180) -- so this does not pretend to be a
  // list, and it carries no verbs: Create (g) and Reseed (Shift+G) stay on the frame row.
  // What it adds that nothing else answers is how much of the frame is actually yours.
  renderInstance() {
    const V = this.meta.n_views;
    const P = this.meta.n_points;
    let placed = 0;
    if (this.fixedMask) {
      for (let v = 0; v < V; v++) for (let p = 0; p < P; p++) if (this.fixedMask[v][p]) placed++;
    }
    this.instanceStateEl.textContent = this.hasInstance ? `${placed} / ${V * P}` : "none";
    const seed = this.seedMode === "copy" ? "each view's own detection" : "the triangulated detections";
    const rows = this.hasInstance
      ? [["Skeleton", "created for this frame"],
         ["Your pixels", `${placed} of ${V * P} cells`],
         ["Seeded from", seed]]
      : [["Skeleton", "none yet — drag a joint, double-click one, or press g"],
         ["A new one seeds from", seed]];
    this.instancesList.replaceChildren(...rows.map(([k, v]) => {
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
    }));
  }
```

If the slot is not wanted, deleting it is one `<section>` in H, one entry in `SIDEBAR_SECTIONS`, `renderInstance` and its two call sites.

---

# 4. What the frame-nav arrows and `j` do now

**Arrows.** The single aside-level pair (H:187-190) is gone. Each list section carries its own: `#frames-prev`/`#frames-next` in the Labeled header (always the labeled frames, in time order — which is what the help already says at J:3098 and what the static titles at H:188-189 already said), and new `#suggest-prev`/`#suggest-next` in the Suggested header (the queue in rank order, `jumpSuggested`, J:2773-2783).

**The ↑/↓ keys** need one owner. `navList` is that owner, set by the last list the operator actually used and shown by `.sec-nav.is-nav` on that header. It replaces the two `sidebarTab` reads at J:2152 and J:2764.

```js
  /** @param {"labeled"|"suggest"} list @param {number} dir */
  stepList(list, dir) {
    this.setNavList(list);
    if (list === "suggest") this.jumpSuggested(dir);
    else this.jumpLabeled(dir);
  }

  /** @param {"labeled"|"suggest"} list */
  setNavList(list) {
    if (this.navList === list) return;
    this.navList = list;
    this.updateSidebarNavTitles();
  }

  // The ↑/↓ KEYS (buildBindings, app.js:2986-2987): step whichever list the operator last
  // used. There is no active tab to read it off any more, so it is explicit state -- and it
  // is marked in the owning section's header, so it is never a guess which list moves.
  /** @param {number} dir */
  jumpCorrected(dir) { this.stepList(this.navList, dir); }

  /** @param {number} dir  -1 for the previous labeled frame, +1 for the next */
  jumpLabeled(dir) {
    /* verbatim body of app.js:2156-2165 */
  }

  // Each list's pair names its own list; the pair that owns the keys says so.
  updateSidebarNavTitles() {
    const own = (l) => (this.navList === l ? " — ↑ / ↓ step this list" : "");
    this.framesPrevBtn.title = `Previous labeled frame${own("labeled")}`;
    this.framesNextBtn.title = `Next labeled frame${own("labeled")}`;
    this.suggestPrevBtn.title = `Previous suggested frame, in rank order${own("suggest")}`;
    this.suggestNextBtn.title = `Next suggested frame, in rank order${own("suggest")}`;
    this.labeledNavEl.classList.toggle("is-nav", this.navList === "labeled");
    this.suggestNavEl.classList.toggle("is-nav", this.navList === "suggest");
  }
```

Row clicks set the owner too: J:2019 → `tr.addEventListener("click", () => { this.setNavList("labeled"); this.goToFrame(frame); })`; J:2658 → the same with `"suggest"`.

`updateActiveFrameRow` (J:2115-2128) rewritten — both lists get the `.is-current` highlight, but only the **nav owner's** row is scrolled, because each list is now its own scrollport and two `scrollIntoView` calls would fight over the shared `#sidebar-sections` scroller:

```js
  updateActiveFrameRow() {
    this.frameRows.forEach((tr, f) => tr.classList.toggle("is-current", f === this.frame));
    this.suggestRows.forEach((trs, f) => {
      for (const tr of trs) tr.classList.toggle("is-current", f === this.frame);
    });
    if (!this.framesOpen) return;
    // Both lists are on screen, so both are highlighted -- but only the list the operator
    // is walking is SCROLLED. Two scrollIntoView calls would each also scroll the shared
    // section column, and the second would undo the first.
    if (!this.sectionOpen(this.navList)) return;
    const row = this.navList === "suggest"
      ? this.suggestRows.get(this.frame)?.[0]
      : this.frameRows.get(this.frame);
    row?.scrollIntoView({ block: "nearest" });
  }
```

**`j`** keeps its verb (toggle the whole aside) and gains persistence + the disarm + the poll stop via `openFrames`/`closeFrames` above. Its description (J:3048) and `#frames-toggle`'s title (H:157) are rewritten:

```js
  b.push({ key: "j", group: "panel", label: "j",
    desc: "Show / hide the side panel — the recording, labeled frames, the suggested queue, landmarks, jobs and settings",
    run: () => this.toggleFrames() });
```

**`b`** (J:3051-3053) no longer has a popover to open:

```js
  if (this.meta.project_root) {
    b.push({ key: "b", group: "panel", label: "b",
      desc: "Browse this project's recordings — and switch to another",
      run: () => this.revealSection("recordings") });
  }
```

**Escape** (J:3303-3328): delete the `recordingMenuOpen` branch (3307-3309); insert a disarm branch before the `selection.size` one, so Escape is a third way out of an armed landmark:

```js
      } else if (this.armedLandmark >= 0) {
        this.armLandmark(-1);
        e.preventDefault();
      } else if (this.selection.size) {
```

`applyOsHints` (J:3255) keeps its "↑ / ↓ step the side panel's list" phrasing; J:3258's `updateSidebarNavTitles()` call stays.

---

# 5. Tests

## 5.1 THE CATEGORY THAT MATTERS — green while proving nothing

Every assertion in these is `count()` / `inner_text()` / `get_attribute()` / `is_checked()`, all of which read through `display: none`. And because the section headers are `<button>`s carrying the same words, `_open_suggest`'s `button:has-text("Suggested")` (there is **no `data-tab` anywhere** — grep confirms; the text half has always done 100% of the work) will still match, still click, and now **collapse** the section under test. Every one of these stays green.

| test | file:line | why it proves nothing | fix |
|---|---|---|---|
| `test_the_suggestion_queue_carries_its_own_reviewed_tick` | :226, assertion :236 | `.reviewed-tick` **unscoped**; `#labeled-pane` renders them too. Passes even if the queue renders nothing. **Worst offender.** | scope to `#suggest-pane .reviewed-tick`, + `expect(pane).to_be_visible()` |
| `test_ticking_reviewed_in_the_queue_sticks` | :242, :245/:249/:251 | `.first` is now whichever pane is first in DOM order — the Labeled list | scope all three to `#suggest-pane` |
| `test_the_labeled_tab_still_renders_after_the_shared_tick` | :257, :263/:265 | `#labeled-pane tbody tr` is populated the whole time; degenerates into "no JS errors", already covered by :178 | delete the tab click at :263; rename off "tab"; keep as a `#labeled-pane` row-count check |
| `test_the_jobs_panel_renders_its_actions` | :552, :555 | passes whether or not the section can be expanded | `_section(page,"jobs")` (asserts visible) |
| `test_a_session_without_a_project_explains_the_empty_jobs_panel` | :572, :576 | `inner_text()` reads through `display:none` | same |
| `test_the_landmarks_panel_lists_the_declared_landmarks` | :632, :635-638 | ditto; also blind to `renderLandmarks` losing its trigger | same |
| `test_a_project_with_no_landmarks_explains_the_empty_panel` | :674, :677 | ditto | same |
| `test_the_settings_panel_is_generated_from_the_schema` | :735, :739-746 | the `refreshSettings` trigger is exactly what is at risk and this cannot see it | same |
| `test_a_session_without_a_project_explains_the_settings_panel` | :774, :777 | ditto | same |
| `test_switching_reloads_the_editor_onto_the_new_recording` | :890, :900-905 | **every assertion is satisfied identically by `location.reload()` and by an in-place rebuild.** It cannot tell you whether step 1 landed or silently reverted | navigation sentinel, §5.4 |

The structural fix is the helper asserting visibility — that is the only thing the current assertions cannot fake.

## 5.2 Helpers

Add `from playwright.sync_api import expect  # noqa: E402` beside :32.

**Delete:** `_open_suggest` (:172-175), `_open_jobs` (:546-549), `_open_marks` (:626-629), `_open_settings` (:729-732), `_open_recording_menu` (:848-850), and the inline tab clicks at :263 and :665. **Keep** `_open_panel` (:165-170) but make it honor the new default:

```python
def _open_panel(page):
    """The panel is the operator's arrangement, so it persists -- open it if it is shut."""
    if page.locator("#sidebar[hidden]").count():
        page.locator("#frames-toggle").click()
        page.wait_for_timeout(300)


def _close_panel(page):
    if not page.locator("#sidebar[hidden]").count():
        page.locator("#frames-toggle").click()
        page.wait_for_timeout(300)


def _section(page, name, want_open=True):
    """Put the named sidebar section in the wanted disclosure state, and PROVE it.

    The visibility assertion is the point. Every count()/inner_text() assertion in this
    file reads through `display: none`, so a section that could no longer be expanded at
    all would leave the whole suite green -- which is exactly what the old
    `button:has-text("Suggested")` clicks would do against a stacked layout, where they
    still match a header and now COLLAPSE the section under test.
    """
    _open_panel(page)
    toggle = page.locator(f"#sec-{name}-toggle")
    if (toggle.get_attribute("aria-expanded") == "true") != want_open:
        toggle.click()
        page.wait_for_timeout(500)
    body = page.locator(f"#sec-{name} .sidebar-body")
    expect(body).to_be_visible() if want_open else expect(body).to_be_hidden()
```

Also: drop the pointless `_open_panel(page)` at :479 (that test works entirely in the `#show-toggle` popover), fix the dead `... or True` assertion at :766-770, and fold `jobs_page_and_errors`'s inline server boot (:508-519) into `_serve_app`.

Fixtures: **none change.** `recording_page_and_errors` (:784-845) is the only one that can prove an in-place rebuild works — keep it exactly as-is, and its currently-unused `port` (:842) is now the handle for the navigation sentinel.

## 5.3 Rewrites

- `test_leaving_the_landmarks_tab_disarms` (:660) → **`test_collapsing_the_landmarks_section_disarms`**. Do not delete it: the surprise it guards against gets *more* likely when the panel is permanently on screen. `_section(page,"marks")`; arm; `_section(page,"marks",want_open=False)`; click a canvas; assert `session.state.landmarks.observed.sum() == 0`. Add siblings **`test_closing_the_panel_disarms`** (press `j`) and **`test_escape_disarms`**.
- `test_the_toolbar_names_the_open_recording` (:853) → **`test_the_sidebar_names_the_open_recording`**: `#recording-name` reads "flyA" **while `#sec-recordings` is collapsed** — that is the property the picker's removal put at risk.
- `test_the_picker_lists_the_projects_recordings` (:860) → open via `_section(page,"recordings")`; the `.rec-row` assertions (:863-872) are unchanged.
- `test_the_picker_opens_from_the_keyboard` (:876) → **`test_b_reveals_the_recordings_section`**: assert `#recording-pane` hidden, press `b`, assert visible and `aria-expanded="true"`; drop the Escape half (there is no popover) or replace it with "pressing `b` again is idempotent".
- `test_switching_with_unsaved_labels_asks_before_dropping_them` (:908) → only the opening gesture (`_section(page,"recordings")`) and the `#recording-name` reads change; the `#switch-*` modal is untouched by all of this.
- `test_gui_server.py:921` `test_suggest_panel_ids_agree_across_the_assets` → rename to `test_the_sidebar_ids_agree_across_the_assets`. **Drop `sidebar-tabs`.** Ids become: `sidebar-sections`, `recording-pane`, `labeled-pane`, `suggest-pane`, `instances-pane`, `marks-pane`, `jobs-pane`, `settings-pane`, plus every `sec-<id>-toggle`, plus `labeled-count`, `suggest-tally`, `suggest-count`, `frames-count`, `suggest-status`, `suggest-table`, `suggest-empty`, `suggest-prev`, `suggest-next`, `instance-state`, `instances-list`. Classes (:937) gain `side-sec`, `sec-head`, `sec-toggle`, `sec-caret`, `sec-value`, `sec-nav`, `sidebar-sections`, `inst-row`. **This is the cheapest guard against a blank editor at boot** (`el()` throws on a missing id) — extend it, do not shrink it.
- `test_gui_recordings.py:400` `test_the_picker_ids_agree_across_the_assets` → drop `recording-wrap`, `recording-toggle`, `recording-menu`; keep `recording-name`, `recording-list`, `recording-empty` and all six `switch-*`; add `recording-pane`, `sec-recordings-toggle`. Keep the CSS list at :429. Reword the comment at :433-434 — the push now drives a rebuild, not a reload.
- `test_gui_recordings.py:333` `test_switching_tells_every_open_browser_to_reload` → the server behavior and the `type`/`recording`/`reason` assertions (:356-357) are unchanged; the **docstring at :335-338 becomes false** ("a full page load is the only correct response") and must be rewritten to "so every tab rebuilds from the new /api/meta".
- `test_gui_recordings.py:362` `test_a_reload_storm_does_not_stop_the_server_mid_switch` → **retitle** `test_a_refresh_storm_does_not_stop_the_server_mid_switch` and say in the docstring that the in-place rebuild keeps sockets alive, so this now guards a *manual* refresh (and old clients), not the switch path. Keep the server behavior.
- `src/deeperfly/gui/web/static/types.js:250-254` — rewrite the `ReloadMessage` doc.

## 5.4 New tests

1. **`test_switching_recordings_rebuilds_in_place`** — the sentinel that stops step 1 silently reverting:
```python
def test_switching_recordings_rebuilds_in_place(recording_page_and_errors):
    """A reload and an in-place rebuild are indistinguishable to every other assertion in
    this file, so this one pins the page identity: a marker set before the switch has to
    still be there afterwards, and no navigation may have happened."""
    page, errors, _ = recording_page_and_errors
    navigated = []
    page.on("framenavigated", lambda f: navigated.append(f.url))
    page.evaluate("() => { window.__pageId = 'sentinel'; }")
    _section(page, "recordings")
    page.locator(".rec-row:not(.is-active)").first.click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    assert page.evaluate("() => window.__pageId") == "sentinel", "the page reloaded"
    assert not navigated, f"the page navigated: {navigated}"
    assert page.evaluate("() => Number(document.getElementById('frame-number').max)") == 8
    assert not errors
```
2. **`test_the_switch_rebuilds_one_canvas_per_camera`** — `page.locator("#stage canvas, #strip canvas").count() == n_views` after the switch. Catches `buildViews` appending.
3. **`test_the_rebuilt_editor_requests_the_new_recordings_frames`** — collect `/api/frame/` request URLs via `page.on("request")`; assert the post-switch ones carry a `?v=` token equal to the new `/api/meta` `cache_v` and different from the pre-switch one. Nothing asserts this today; `location.reload()` got it for free.
4. **`test_the_open_sections_survive_a_recording_switch`** — expand Jobs + collapse Suggested, switch, assert `aria-expanded` on both is unchanged.
5. **`test_every_sidebar_section_expands_and_collapses`** — loop the seven ids; assert `aria-expanded` flips and the body's visibility follows.
6. **`test_the_sidebar_holds_exactly_the_seven_sections_in_order`** — the structural net, mirroring `test_the_toolbar_is_one_row` (:183), which is what will catch a stray `</div>` in this rewrite:
```python
    ids = page.evaluate(
        "() => [...document.getElementById('sidebar-sections').children].map(e => e.id)"
    )
    assert ids == ["sec-recordings", "sec-labeled", "sec-suggest", "sec-instances",
                   "sec-marks", "sec-jobs", "sec-settings"]
```
7. **`test_the_toolbar_no_longer_carries_a_recording_picker`** — `#recording-wrap` count 0, and `#controls` children still `["control-row frame-row", "control-row toolbar-row"]`. **Run `test_the_toolbar_is_one_row` (:183) first** — it is the safety net for the H:145-152 deletion.
8. **`test_the_expensive_sections_fetch_nothing_until_expanded`** — with the default arrangement, no request to `/api/recordings`, `/api/config` or `/api/jobs` during load; then `_section(page,"settings")` and assert `/api/config` arrives.
9. **`test_the_jobs_poll_stops_when_the_panel_closes`** — count `/api/jobs` requests over 4 s with Jobs expanded, close with `j`, count again over 4 s: zero. This closes a pre-existing leak, so it is a genuinely new guarantee.
10. **`test_which_sections_are_open_survives_a_reload`** — collapse Labeled, `page.reload()`, assert still collapsed.
11. **`test_stepping_the_suggested_list_moves_the_arrow_keys_to_it`** — click `#suggest-next`, assert `#suggest-nav` has `.is-nav`, then press `ArrowDown` and assert the frame moved to the next *queue* frame, not the next labeled one.
12. **`test_the_instance_section_reports_the_frames_skeleton`** — `#instance-state` reads "none", press `g`, assert it reads `"0 / <V*P>"`.
13. Amend **`test_d_marks_the_current_frame_reviewed_with_no_panel_open`** (:269): its `#sidebar[hidden]` assertion at :277 must become `_close_panel(page)` first (the default is now open). Keep everything else — its pane-scoped selectors at :283/:286 are already correct and it is the model the other tests should copy.

---

# 6. Ranked risks

**Anything that can show one recording's data while claiming to be another is at the top.**

| # | risk | mitigation |
|---|---|---|
| **1** | **Stale meta-derived chrome after a switch.** Missing `applyMeta()` leaves the old recording's name (J:755), frame count (J:616-622 — the scrubber still ranges over the old length), banners and toggles. The picker's removal makes the *name* the loudest of these. | §1.1 `applyMeta()`, one method both paths call; `.max` set before `.value`. New test 1 pins `#recording-name` + `#frame-number.max`; test 3 pins the frame URLs. |
| **2** | **Jobs submitted against the wrong recording.** `renderJobActions` (J:2469-2493) is `built once` (2470) and closes over `const recording = this.meta.recording` (2471), used in `submitJob` at 2485. After a switch every button runs a pipeline command on the *previous* recording — it does not merely display wrong data, it writes to the wrong outputs. | `this.jobsActions.replaceChildren()` in the rebuild, step 5. Add a test: switch, expand Jobs, click "Suggest frames", assert `.job-cmd` names flyB. |
| **3** | **Stale `[view][point]` masks and `focused`.** `fixedMask`/`excludedMask`/`absentMask`/`detectedMask` are sized by the old rig. `updateStatusWidget` (J:1692, 1700-1704) indexes `this.fixedMask[v][p]` with a stale `v` → `undefined[p]` **TypeError**; `relayout` (J:884) with `focused >= n_views` passes `undefined` to `replaceChildren` → throws. And `absentRecording` announces the previous animal's amputations. | `resetRecordingState()` (§1.5), called before `applyMeta`; browser tests fail on any console error, so a switch onto a rig with fewer views is now covered by test 1 alone. |
| **4** | **Frames served from the old recording's cache.** `cacheVersion` is api.js's only module-level mutable state (api.js:18), set in `fetchMeta` (api.js:25) and read by `frameUrl` at call time (api.js:126-133). | `fetchMeta()` is step 3, strictly before `goToFrame(0)` in step 8. New test 3 asserts the token on the actual `<img>` requests. |
| **5** | **In-flight fetches resolving after the swap.** There is **no `AbortController` anywhere in the codebase**. `refreshPoints` (J:947-953) applies with `fromEdit=false`, whose only defense is the frame guard (J:982) — and the rebuild navigates to frame 0, so an old `fetchPoints(0)` passes it. `refreshCorrected`/`refreshSuggestions` have no guard at all. | `this.epoch`, bumped in step 1 and checked after every `await` (§1.4). |
| **6** | **Doubled `PoseView`s.** `buildViews` pushes (J:840, 848); `relayout` (J:883) then re-attaches both generations and `applyPoints` (J:1031) indexes past `p.points`. | Step 2's teardown + `PoseView.destroy()` (§1.2). New test 2 counts canvases. |
| **7** | **A pre-switch edit reply applied to the new session.** Zeroing `editSeq` would let an old reply match again at J:988. | `this.editSeq++`, never `= 0`. Same for `meshReq` (guard at J:1219). |
| **8** | **Double entry into the rebuild.** The server broadcasts to every socket including the initiator (server.py:677), so J:573 and J:1418 both fire. A `location.reload()` made the second a no-op; an async rebuild does not — two concurrent rebuilds would interleave teardown and construction. | `this.rebuilding` promise guard, step 0. |
| **9** | **Silent role / guard changes.** Resetting `readOnly` would promote a reader (the writer slot is per socket, server.py:794-812, and the socket survives). Setting `closing = true` (today's J:1441) would permanently kill the beforeunload guard (J:582-587) for the new recording's edits. | Both listed as "deliberately NOT reset" in `resetRecordingState`'s own comment; **do not** construct a new `EditSocket`. |
| **10** | **Leaks per switch.** Re-running `buildControls` would re-add ~50 element listeners (two edits per click on undo/gt/absent), two document-scoped listeners that can never be removed, four duplicated `segmented` widgets, and four `#scene-head` listeners via `initSceneDrag` (J:2842-2862). Re-`new Scene3D` leaks 7 listeners + a `ResizeObserver` per switch (scene3d.js:112-119). Unstored `PoseView` `ResizeObserver`s (poseView.js:303). | The one-time/per-recording split; scene re-seeded in place (step 7); `PoseView.destroy()`. Manual check: switch 10× and assert `getEventListeners` counts are flat, or at minimum that `#stage canvas` stays at `n_views`. |
| **11** | **`_detectedTitle` corruption.** J:666 is a one-time *capture*; re-running it while the auto-hide note is showing (J:1148-1150) makes that note the permanent base tooltip. | Explicitly left in `buildControls`, never in `applyMeta`. |
| **12** | **Cost blow-up from several panels visible.** Naive "expand everything by default" adds an HDF5 open per recording, two TOML parses and a 2 s forever-poll to every editor load. | Defaults in `SIDEBAR_SECTIONS`: Recordings / Jobs / Settings shut. `syncSectionEffects()` is the single poll gate and also closes the pre-existing `closeFrames` leak. New tests 8 and 9. |
| **13** | **Tests green, proving nothing.** §5.1 in full — ten tests, the worst being an unscoped `.reviewed-tick` count and a switch test that cannot distinguish a reload from a rebuild. | Scope every pane assertion; `_section()` asserts visibility; navigation sentinel. |
| **14** | **An armed landmark outliving the operator's attention.** `setSidebarTab`'s disarm (J:2756-2758) disappears, and it is *already broken* — the `else` binds to the `settings` test, so Landmarks→Settings never disarms today. | Disarm on section collapse (`setSection`), on `closeFrames`, and on Escape. Three new tests. |
| **15** | **Layout regressions.** `.sidebar-body { flex: 1 }` (C:711-715) with several visible makes the panes fight; a single aside scroller would overlap both tables' sticky `th` (C:722-724); `.rec-row`'s `margin-left:auto` stats (C:1487) reflow badly below `#recording-menu`'s old `min-width: 320px`. | Per-section scrollers + `.is-open`/`.is-compact` flex rules (§2.2); `.rec-row { flex-wrap: wrap }`; widen `#sidebar` to `clamp(240px, 20vw, 340px)` and delete `#sidebar.tab-suggest`. |
| **16** | **Section arrangement silently reset by a switch.** Nothing today asserts any sidebar disclosure survives anything. | `openSections`/`framesOpen` explicitly excluded from `resetRecordingState`; new tests 4 and 10. |