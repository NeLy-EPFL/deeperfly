# Annotation GUI

`deeperfly gui` opens an interactive **web** viewer for a result and turns it into a
**ground-truth annotation** tool. It serves every camera view with its 2D skeleton
overlay to a browser canvas and lets you author the ground-truth 2D pose, with the
run's prediction as a starting point. Your labels are written to a `labels.h5`
sidecar next to the result and **never modify `results.h5`** — re-running the
pipeline is always safe.

Because it is a browser app it needs no GUI toolkit, runs headless, and can be
reached from another machine (see [Remote use](#remote-use)).

## The idea: 2D is the source, 3D is derived

The 2D observations are the only source of truth; the 3D pose is a *pure function* of
them (triangulation over the cameras). The 3D that `deeperfly run` wrote is a cache —
the editor recomputes it live from whatever 2D you settle on. So you never edit 3D
directly: you author **2D ground truth**, and the 3D follows.

Per `(view, point)` — one *cell* — you author two things, on two **independent** axes:

- **Where the keypoint is.** Either **ground truth** (a 2D pixel you placed, by dragging or
  by confirming what is drawn) or nothing, in which case the cell follows the detector's
  prediction, or the 3D reprojection where the detector fired nothing, as a *suggestion*
  you can accept or move. This is what a future training run learns from.
- **Whether the cell is included in the training loss.** That is the **Hidden** flag
  (`e`) and it is all it means: marked = held out, unmarked = used. It is a binary switch
  of its own, not a third value of the first axis.

Because they are separate axes, all four combinations are meaningful, and the useful one
is the pair: place the pixel *and* mark it Hidden when you are confident where a joint is —
inferred from the views that can see it, say — but do not want the detector trained on
pixels that do not show it. Hiding a cell does exactly one thing. It does **not** move the
joint, remove its marker, break its bones, make it undraggable, or change the 3D by a single
coordinate; the joint is drawn exactly as it would be, with a bar struck through it.

The displayed point resolves by precedence **ground truth → prediction →
reprojection**. How ground truth and predictions combine into the live 3D is
configurable (`[annotation]`, default "ground truth wins"); see
[configuration](../reference/configuration.md#annotation).

## Launch

```bash
deeperfly gui RESULT                     # a results.h5, or a dir containing one
deeperfly gui recording/deeperfly_outputs
deeperfly gui myproject/                 # a project: every recording, plus Jobs and Bundle adjust
```

`RESULT` is a `results.h5` file, a directory holding one (e.g. a recording's
`deeperfly_outputs/`), or a **project** directory — which is what puts the project's other
recordings, the pipeline Jobs and the Bundle-adjust tab in the side panel. The footage is
resolved from the paths recorded in `results.h5`; if those no longer exist, pass
`--footage-dir` to point at it (views with no footage still draw their overlays on blank
frames). A browser opens automatically; the server stops a few seconds after the last tab
closes. See the [CLI reference](cli.md#deeperfly-gui) for every flag (`--host`, `--port`,
`--no-browser`, `--keep-alive`).

## Layout

- **Views.** Every camera is shown with its overlay. Two arrangements share the same
  canvases: **Grid** (all cameras equally, the default) and **Focus** (one big view
  plus thumbnails) — toggle with the layout switch or `l`. In Focus, `[` / `]`
  cycle which camera is enlarged.
- **Frame strip + slider.** Scrub with the slider, the `←` / `→` keys (`Shift` for
  ±10), or type a frame number. The slider drives the whole editor, including the 3D
  view, so everything stays on the same frame.

## Reading the markers

There is no 2D/3D mode to choose — each keypoint is drawn once, and its **marker style
tells you where it came from** so you know at a glance what still needs attention:

| Marker | Source | Meaning |
| --- | --- | --- |
| Filled disc, **lime** ring | **Ground truth** | you authored it (dragged or confirmed) — trusted |
| Filled disc, thin **dark** ring (fill fades when faint) | **Prediction** | the detector's raw 2D; the fainter the fill, the lower its confidence |
| **Hollow** circle in the point's limb color | **Projection** | no direct observation in this view — the 3D reprojected here (a suggestion) |
| A **bar struck through** any of the above | **Hidden** | this cell is held out of the training loss — see below. It *annotates* the marker rather than replacing it, because it says nothing about where the keypoint is |

The **`?` button** (or the `?` key) opens the **Help panel**, which carries the full
legend — the keypoint colors (one swatch per limb, taken from your skeleton's
`limb_palette`, so it matches whatever config you loaded), the marker vocabulary above,
and the reference-overlay line styles — alongside the keyboard shortcuts. The displayed
point resolves by precedence **ground truth → prediction → projection**, and the Hidden
bar rides on top of whichever of those the cell landed on.

## Annotating the pose

**Dragging** a keypoint **creates ground truth** at the drop pixel — one gesture,
whatever the marker was. When the result carries a 3D pose the 3D **re-solves live** as
you drag, so every view's projection markers follow; the dragged view is authored as
ground truth and lands exactly under the cursor. (With a 2D-only result there is no 3D
to re-solve — the drop is simply that view's ground-truth pixel.)

Most predictions are already good — you don't need to drag them, just **select** the
points you want and **act** on them. Annotation is two steps: build a selection, then
apply one verb to all of it.

### Selecting points

A selection is a set of `(point, view)` cells; the **Selection** toolbar group shows the
live count. Build one with:

- **Click** a point — select just that one (replaces the selection).
- **Shift+click** a point — add it to (or remove it from) the selection.
- **Double-click** a point — select that keypoint in **every** view.
- **Shift+drag** a box (a rubber-band marquee) — add every point inside it. A plain drag
  still moves a point / pans the view, so hold Shift to box-select.
- **`a`** (or `Ctrl`/`Cmd`+`A`) — select every point in every view.
- **`v`** — select every point in the view under the cursor.
- **`Esc`** — clear the selection.

The selection persists as you step between frames (the indices are the same), so you can
act on the same joints frame after frame.

### Acting on the selection

The verbs act on whatever is selected — the **GT** / **Hidden** / **Absent** toggles and
**Reset**, or their keys. Each is a **single undo step**, however large the selection. Each
toggle is also its own readout: it is *pressed* when the fact is set, and **GT and Hidden can
be pressed at the same time**, which is the whole shape of the data.

- **GT** (`Enter` places, `Backspace` clears) — place a ground-truth pixel for each
  selected cell at the position already drawn there, so the joint becomes yours and can be
  nudged. Cells whose position the editor *invented* (a neighbor mean, the view center —
  see **Invented** in the Help legend) are skipped rather than recorded as your pixel.
- **Hidden** (`e`) — hold each selected cell **out of the training loss**, or put it back
  (a toggle). That is the entire effect. The joint does not move, keeps its marker and its
  bones, stays draggable and hoverable, and the 3D does not change — a bar is struck through
  it, and the export reports it in its own mask beside the ground truth. It is available on
  cells that already carry your pixel, and that pairing is the point: *"this is where the
  joint is"* and *"do not train on it here"* are two separate things worth recording.
- **Reset** (`r`, or `Delete`) — retract **both** facts for each selected cell, back to
  nothing authored. The only verb that spans both axes, which is why it is named for
  starting over rather than for either of them.

- **Absent** (`x`, or `Shift`+`X` for the whole recording) — mark the selected
  keypoint(s) **not on this animal**: an amputated leg, an ablated antenna. Unlike every
  other action it is not per *view* — an amputated joint is missing from every camera
  at once, which is exactly what separates it from Hidden. `x` marks the current frame;
  **`Shift`+`X` applies it to every frame**, which is what almost every real declaration
  wants (an animal that arrives with a leg already missing). Per-frame exists for the case
  where it genuinely changes: a leg lost to autotomy part-way through a recording.
  The joint then draws as a dim grey ✕ with no bones, drops out of the 3D solve, and is
  excluded from the training export in *both* directions (it is neither ground truth nor
  hidden). A header badge lists what is declared and says whether it covers the whole
  recording or just this frame. Press `x` again or `Ctrl`/`Cmd`+`Z` to lift it — nothing
  you labeled underneath is lost, it is only vetoed while the declaration stands.

Everyday flows: click a point and press `Enter` to place one keypoint everywhere; `a`
then `Enter` for the whole frame; double-click a mislocated joint and press `r` to
reset it across all views; Shift+drag a box around the far-side joints and press `e` to hold
them out of the loss; double-click an amputated joint and press `Shift`+`X` once for the
whole recording.

### Hidden vs Absent vs Unplaced

Three different things, and they must not be confused. Note that only **Absent** removes a
keypoint's *position*:

| | means | scope | position | on export |
|---|---|---|---|---|
| **Hidden** (`e`) | do not include this cell in the training loss | one (view, frame, point) | unchanged — still drawn, still draggable | its own mask, beside the GT one |
| **Absent** (`x` / `Shift`+`X`) | it is not on this animal | every view; this frame, or the whole recording | gone — no 2D, no 3D, no bones | excluded from GT *and* from hidden |
| **Unplaced** (`i`) | nobody has placed it yet | one (view, frame, point) | the editor's guess, so you can grab it | nothing (unlabeled) |

So **Absent is not a bulk Hidden**. Both keep a cell out of the loss, but Absent also
deletes the keypoint from the reconstruction, in every view, and (for the IK body plan and
bundle adjustment) for the whole recording. Declaring a joint absent because it is awkward to
see in a few frames throws away its position too; marking an amputated joint hidden view by
view leaves a phantom limb standing in every canvas, reconstructed from detector peaks that
fire on whatever looks leg-like. Absence is a claim about the **animal**; Hidden is a claim
about **what to train on**.

The **Hidden** flag is deliberately *only* that claim. It is not read by the 3D solve — a bad
observation is down-weighted on its merits by the robust estimator, so there is nothing to
exclude by hand — and it is not read by the display beyond drawing its own bar. That
independence is what makes it safe to mark a whole frame at once (`a` then `e`): nothing on
screen moves.

### The reprojection check

**Reproj. warning** (`w`, under **Checks**) flags a keypoint whose 2D disagrees with the
multi-view 3D: an amber→red ring on the point plus a connector to where the 3D reprojects
it — amber at the threshold (**Warn above**, 8 px by default), full red at twice it.

What it measures is the position a view *claims*, which is what makes it actionable:

- **A view you labeled** claims your pixel. Two labels can always be satisfied exactly by
  some 3D point, so it takes a **third** disagreeing view to light up — and then the rings
  sit on your own labels, which is the one error no other view can reveal.
- **A view you have not labeled** is drawn *at* the reprojection, so it has nothing to
  disagree with and is never flagged. Label a joint in two views and the other cameras move
  onto the geometry, quietly. (Switch the unplaced joints to their **seeds** with `s` and the
  check comes back — a frozen seed is a claim of its own.)
- **Before the frame has a skeleton** the detections are what is on screen, so the
  detector's pixel is what gets checked.

It deliberately does **not** flag the *detector* for disagreeing with the geometry once you
are annotating. A keypoint on the far side of the animal — a left-hind joint in a right-side
camera — is routinely a hundred pixels out there, and no amount of labeling moves it: that
is a red flag you could never clear. Turn the **Detected** layer on (`t`) when the detector's
opinion is what you want to see.

### The label-coverage check

**Under-labeled** (`u`, under **Checks**) flags every keypoint you have labeled in fewer than
**two** views of this frame. It draws a violet gauge ring outside the joint's own marker: a
dashed track for the two labels it needs, and a solid arc for the ones it has. So *nothing
labeled* reads as an empty dashed ring, *one of two* as a half-swept one, and a joint that has
its two views is drawn nothing at all — the rings empty out as you work, which is what makes it
usable as a "what is left here?" pass rather than permanent decoration.

Off by default, and deliberately: a ring on all 38 joints of an untouched frame would bury the
skeleton it is describing. Turn it on when you want the audit.

Two is not an arbitrary bar. A single pixel fixes only a *viewing ray* — it says where the joint
is on a line through the camera, not where it is in space — so until a second view lands, the
joint's depth still comes from the detector rather than from you, and that is precisely where
the annotation solve hands the point over: at
[`min_gt_for_exclusive`](../reference/configuration.md#annotation) labeled views (2 by default) your pixels
become authoritative for the whole point. Raise **Want at least** if you want redundancy beyond
the minimum; it is capped at the number of cameras, since more than that could never be satisfied.

Two labeled views is the floor, though, not always enough — and *which* two matters. A camera
cannot see distance along its own optical axis, so two cameras that face each other leave that
one direction nearly free: their viewing rays are almost the same line, and two nearly identical
lines have no sharp crossing point. `rm` and `lm` of the standard rig are exactly opposed, and
labeling only those two used to leave the joint hundreds of microns out and tens of pixels off in
every view you had *not* labeled. By default the editor now lets the unlabeled views supply that
one direction and nothing else, so your pixels still decide everything they have an opinion about
— see **Skeleton ▸ Derive the 3D from**, which also offers the older *My pixels only* behavior.
Flipping between the two is the quickest read on what the other views are contributing: the
joints whose two labels have a poor baseline visibly move, and the rest do not budge. If a joint
does move a lot, the real fix is a third label in a view that looks from a different direction —
worth roughly 89× on the test rig.

What it does and does not count:

- **Your pixels, wherever they are.** The count is over the whole frame, so the ring shows in
  every view — including the one you already labeled, whose own label is the arc you can see.
  The joint's problem is not local to a camera.
- **Hidden cells still count.** [Hidden](#hidden-vs-absent-vs-unplaced) decides what the
  *training loss* reads; a pixel
  you placed is a pixel you placed, and it constrains the 3D either way.
- **Absent joints are never flagged.** They are not on the animal, so there is nothing to label —
  a coverage flag there would be a demand you could never meet.

Unlike the reprojection check it needs no rig, no detections and no 3D — only a second camera to
count over. On a fresh uncalibrated project it is the only check there is, and "nothing labeled
in two views yet" is exactly the state such a project is in.

### Undo / redo

Every edit is undoable: `Ctrl`/`Cmd`+`Z` undoes, `Ctrl`/`Cmd`+`Y` (or
`Ctrl`/`Cmd`+`Shift`+`Z`) redoes; the ↶ / ↷ buttons do the same. A whole drag — and a
whole batched GT / Hidden / Reset — is a single undo step, and undo jumps back to
the frame the edit was on.

## Frame lists

A retractable side panel holds a strip of tabs showing one pane at a time, of which
**Labeled** and **Suggested** are the frame lists below. The rest are **Instance** (this
frame's annotation skeleton), **Jobs** ([pipeline commands](#running-the-pipeline-from-the-editor)), **Bundle adjust**
(solve the rig from the ground truth you placed, save it as a new calibration, and choose
which calibration the editor derives from), **Settings** (generated from the pipeline's own
parameter dataclasses) and **Recording** (the project's others) — the last of which appear
only in a project session. Which tab you left it on is remembered, as is whether it was
open.

`j` toggles the whole panel. Otherwise the ✕ in its head closes it and the slim rail down
the right edge — which is there only while it is closed — opens it again; the rail also
carries the count of labeled frames, so that number stays visible with the panel shut.

Click a row in either list to jump to that frame; `↑` / `↓` step through the list on
screen, wrapping at the ends — from a tab that is not a list they keep stepping the last
one you used. The current frame stays highlighted as you scrub.

### Labeled

Every frame carrying a label, in time order, with a per-frame **Reviewed** tick box —
your "I have finished checking this" flag, which persists with the labels and keeps the
frame listed even if its point labels are later reset. The list updates live and includes
labels loaded from a previous session; the blue badge on the tab (and on the right-edge
rail, when the panel is shut) shows the total from wherever you are.

### Suggested

The ranked queue of frames worth correcting **next**, written by
[`deeperfly labels-suggest`](cli.md#deeperfly-labels-suggest). The panel only *reads* that
sidecar — ranking triangulates the whole recording, so it is a command you run, not a
button — and the tab is empty (showing the exact command to run) until you have.

Each entry gives its rank, frame, time, disagreement score, and **why** it was picked: a
`most wrong` chip for a frame the cameras disagree about most, or `diversity` for one
drawn from a uniform time grid so the round still sees typical poses. A frame you have
since labeled is struck through and chipped `labeled` the moment you drag its first
point, so `↓` walks you through what is left. The amber badge reads *done / total*.

Scores rank **within one recording only** — the absolute level tracks how many keypoints
the detector fired, not how bad the recording is — so never compare them across files.

The strip above the list carries the queue's own caveats, and they are worth reading: how
many frames it actually delivered against what was requested (the minimum spacing between
picks routinely runs out of room), whether it is stale, and whether the scores may be
tracking calibration error rather than the detector's mistakes. If the queue was computed
for a different recording, or from predictions that have since been replaced, the panel
says so instead of quietly navigating a list that no longer applies.

## NeuroMechFly overlays

When the result carries a fitted inverse-kinematics model — which
[`do_inverse_kinematics`](../reference/configuration.md#inverse_kinematics) now produces by
default — three extra overlays are available:

- **Model skeleton** (`m`) — the fitted model joints, reprojected onto each view.
- **Model mesh** (`Shift+M`) — the posed model mesh, smooth-shaded on the client
  GPU at the view's resolution.
- **3D view** (`c`) — a floating panel showing the cameras, the derived pose, and the
  model skeleton/mesh together in 3D. Drag to orbit, `Shift`/right-drag to pan, scroll to
  zoom; it overlays the editor without blocking it, so the frame slider still scrubs.

Both overlays **re-fit to your labels**: as the derived 3D moves under your edits the
model is re-solved for that frame. The live re-fit runs on the **body plan the pipeline
solved**, read back from `results.h5` — this recording's measured segment lengths and its
registration to the model, so the live overlay and the rendered one describe the same
animal. (An older result file carries no plan, and one whose stored plan cannot be read
says so; either way one is rebuilt from the `config.toml` snapshot beside `results.h5`, and
the live overlay may then differ slightly from the rendered one.) Which parts the mesh
draws is set by [`[gui].mesh_hide`](../reference/configuration.md#gui).

Re-fitting a frame depends only on that frame and its labels, so scrubbing away and back,
or undoing and redoing, always returns the same pose. It also needs the optional
[`ik` extra](../reference/configuration.md#inverse_kinematics), since it is the same
solver the pipeline uses: without it the overlays still draw the stored fit, they just
stop following your edits (the editor says so once, on startup).

## Other overlays

| Toggle | Key | Shows |
| --- | --- | --- |
| Skeleton | `s` | The editable 2D skeleton overlay. |
| Labels | `n` | Each keypoint's name. |
| 3D estimate | `p` | The derived 3D ghosted onto each view. |
| Keypoints ↗ | `k` | The reference [keypoint viewer](../explanation/keypoints.md) (new tab). |

## Saving & exporting

Edits live in memory until you save, and **they live there for the whole project**:
switching to another recording keeps the one you left open behind the scenes, with its
unsaved labels and its undo history intact, so you can move between a project's
recordings the way you move between one recording's frames. Nothing asks you to save on
the way.

While anything is unsaved the toolbar shows an amber **● unsaved** cue (with a count when
more than one recording is involved), the Save button matches it, the browser tab's title
gains a `*`, and every affected recording is dotted in the **Recording** pane's list. The
cue is clickable: it does what Save does.

**Save** (`Ctrl`/`Cmd`+`S`, or the cue) writes the `labels.h5` sidecar of *every* recording
holding unsaved labels — not just the open one. **Close** stops the server and is the only
act that can lose hand work, so it is the only thing that prompts: it names the recordings
still unsaved and offers to save them all first. (Closing the browser tab gets the
browser's own "leave site?" dialog for the same reason.)

Each sidecar is stamped with its recording's fingerprint — the **index domain** its
`(view, frame, point)` keys are keys into (`point_names`, `camera_names`, the frame count)
plus a **recording fingerprint** (each view's image size, since ground truth is stored in
footage pixels, and the footage file basenames). Predictions are deliberately not part of
it, so a re-run of the *same* recording — new detector weights, retuned triangulation —
keeps your labels valid: ground truth is absolute.

`point_names` and the frame count must match **exactly**. A point reorder is a project-wide
migration (`deeperfly project skeleton`) with its own dry run and confirmation, and remapping
it silently here would bypass both. The camera axis is the one that gets remapped rather than
refused: this rig names the same cameras three ways — file stems from a bare directory, source
names from a config, view names from a detection plan — so labels authored *before* a
recording was ever run used to be refused after its first run, which walked the
label-first-then-calibrate workflow into a wall. Sizes and footage are compared per camera
*through* that correspondence, so a rename is not mistaken for a different recording.

Two consequences worth knowing:

- A recording listed with unsaved labels shows its **live** counts, not its sidecar's — the
  numbers already include what is only in memory. That is what the dot is telling you.
- A **job** (the Jobs pane, or any `deeperfly` command) reads the files on disk. Save before
  running one over labels you have just placed, in whichever recording they are in.

To use the labels as training/eval data, export them to an `.npz`:

```bash
deeperfly labels-export RESULT           # writes labels_gt.npz beside results.h5
```

This writes the ground-truth pixels in footage pixel space, plus two masks that are
**separate axes, not filters already applied**: `occluded` (the **Hidden** flag — the cells
you held out of the loss; the array keeps its old name) and `absent` (the keypoints that are
not on this animal). Your loss mask is `gt_mask & ~occluded`, and absent channels should be
**masked** — contributing no gradient. (Supervising them with an all-zero target heatmap is
the stronger claim, "learn that nothing is here"; it needs a decode that can abstain and
enough amputee animals to calibrate one.)

To declare absence without opening the editor — across every clip of one animal at once,
before any labeling — use
[`deeperfly labels-absent`](cli.md#deeperfly-labels-absent).

## Remote use

The server binds loopback by default, so the editor is private. To annotate on a
remote machine, tunnel the port over SSH and open the browser locally:

```bash
ssh -L 8000:localhost:8000 user@host
# on the remote host:
deeperfly gui RESULT --no-browser
# then open http://localhost:8000/ in your local browser
```

Binding a routable address (`--host 0.0.0.0`) is possible but the server is
**unauthenticated** — only do so behind a trusted network. Prefer the tunnel.

## Keyboard shortcuts

Press `?` in the editor for the full, context-aware list. The essentials:

| Key | Action |
| --- | --- |
| `←` / `→` (`Shift` ±10) | Previous / next frame |
| `l`, `[` / `]` | Toggle Grid / Focus layout; cycle the focused camera |
| Click / `Shift`+click / double-click | Select a point / add-remove / that keypoint in every view |
| `Shift`+drag | Rubber-band box: add every enclosed point |
| `a` (`Ctrl`/`Cmd`+`A`) / `v` / `Esc` | Select all / all in the current view / clear the selection |
| `Enter` / `Backspace` | Place / clear the selection's ground-truth pixels |
| `e` / `r` | Hidden (hold out of the training loss) / Reset both facts |
| `x` / `Shift`+`X` | Absent, this frame / the whole recording |
| `Ctrl`/`Cmd`+`Z` / `Ctrl`/`Cmd`+`Y` | Undo / redo |
| `s` / `n` / `p` | Toggle skeleton / labels / 3D estimate |
| `w` / `u` | Toggle the two checks: reprojection distance / under-labeled joints |
| `m` / `Shift+M` / `c` | Toggle the model skeleton / model mesh / 3D view |
| `j` | Show / hide the labeled-frames list |
| `Ctrl`/`Cmd`+`S` | Save labels — every recording holding unsaved work |
| `?` / `Esc` | Show shortcuts / close an overlay |

## The animal is the calibration target { #calibration-target }

A from-scratch project has no camera rig, and the rig is solved from the labels you place
— the tracked keypoints, which is what every rig here was in fact solved from. There is
**no separate landmark namespace**: no `landmarks.toml`, no Landmarks tab, and no amber
diamonds. Nothing ever shipped one.

The cost of that is worth knowing while you label. A skeleton keypoint at frame *t* is a
**different 3D point** from the same keypoint at *t+1*, because the animal moved, so *N*
labeled frames of *P* keypoints add `3·N·P` unknowns — all inside a ~3 mm blob near the
middle of the field, where a static feature would have been one 3D point observed
throughout the scene volume. What follows is practical: **spread the labeled frames**, and
watch the readiness meter's weakest-view-pair row rather than the raw frame count.

`deeperfly calibrate --dry-run` (and the **Bundle adjust** tab) report that meter, phrased
as the labeling that would fix each shortfall.

## Running the pipeline from the editor

Open a **project** (`deeperfly gui myproject/`) and the **Jobs** tab can run pipeline
commands: suggest frames, export labels, check or solve the calibration.

Each row *is* the CLI command, printed verbatim and click-to-copy — so a GUI action that fails
is reproducible in a terminal, and there is no behavior only the GUI can reach. Jobs run one
at a time in their own process (two stages writing one `results.h5` would corrupt it, and a
CUDA OOM must not take your unsaved labels with it).

There is no progress percentage, on purpose: these commands emit human log lines, and a number
synthesized from those would be a fiction with a spinner attached. The last log line is shown
instead, which is the honest signal.

A session opened on a bare `results.h5` has no queue and says so — jobs need a project to run
in.
