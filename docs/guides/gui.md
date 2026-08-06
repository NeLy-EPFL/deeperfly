# Annotation GUI

`deeperfly gui` opens an interactive **web** viewer for a result and turns it into a
**ground-truth annotation** tool. It serves every camera view with its 2D skeleton
overlay to a browser canvas and lets you author the ground-truth 2D pose, with the
run's prediction as a starting point. Your labels are written to a `labels.h5`
sidecar next to the result and **never modify `results.h5`** — re-running the
pipeline is always safe (an older `corrections.h5` is migrated to `labels.h5` on
open).

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
deeperfly gui RESULT            # a results.h5, or a dir containing one
deeperfly gui recording/deeperfly_outputs
```

`RESULT` is a `results.h5` file or a directory holding one (e.g. a recording's
`deeperfly_outputs/`). The footage is resolved from the paths recorded in
`results.h5`; if those no longer exist, pass `--footage-dir` to point at it (views
with no footage still draw their overlays on blank frames). A browser opens
automatically; the server stops a few seconds after the last tab closes. See the
[CLI reference](cli.md#deeperfly-gui) for every flag (`--host`, `--port`,
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
| **Hollow** circle in the point's limb colour | **Projection** | no direct observation in this view — the 3D reprojected here (a suggestion) |
| A **bar struck through** any of the above | **Hidden** | this cell is held out of the training loss — see below. It *annotates* the marker rather than replacing it, because it says nothing about where the keypoint is |

The **`?` button** (or the `?` key) opens the **Help panel**, which carries the full
legend — the keypoint colours (one swatch per limb, taken from your skeleton's
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
  nudged. Cells whose position the editor *invented* (a neighbour mean, the view centre —
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
  other action it is not per *view* — an amputated joint is missing from all seven cameras
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

### The left/right check

A swapped symmetry pair — the left claw dragged onto the right leg and vice versa — is
the one labeling error that costs **nothing** in every metric that looks at a point
cloud. Both pixels are on a real joint, the reprojection error stays small, and
triangulation converges; only the identity is wrong. So the editor looks for it directly.

When the frame's derived 3D puts a pair on the wrong side of the body, an amber
**⇄ left/right?** badge appears next to the save controls, naming the worst pair. Click it
to select that pair in every view, so you land on the joints instead of hunting for them;
hover it for the full list and each pair's margin.

The check is judged on the **3D**, not per view, and that is not a shortcut. In 3D "left"
is a fixed halfspace whatever the animal is doing, so a disagreement really is a swap —
measured over 500 independently-posed flies it emits 0.002 false flags per pose and
recovers 98–99% of planted swaps exactly. The same test on a *side view* emits 4–11 false
flags per pose, because an asymmetric posture genuinely puts the left claw right-of the
right claw in projection. That is real 3D structure, not an error, and no threshold
separates the two.

Two things the badge deliberately does not do:

- **It stays silent when it could not judge** — no 3D in the session, too few co-visible
  pairs, or a collapsed left-right axis. A badge that read "checked, all clear" when it
  actually meant "could not tell" would be worse than no badge.
- **It cannot catch a wholesale flip.** If *every* pair is swapped, the labeling is
  self-consistent: it is a valid labeling of a mirror-image fly, and with no unpaired
  landmark there is nothing inside the frame that says which side is which.

It needs the skeleton's [`symmetries`](../reference/configuration.md#symmetries). The
packaged fly38 skeleton declares them; a skeleton that does not gets pairs inferred from
its point names.

### Undo / redo

Every edit is undoable: `Ctrl`/`Cmd`+`Z` undoes, `Ctrl`/`Cmd`+`Y` (or
`Ctrl`/`Cmd`+`Shift`+`Z`) redoes; the ↶ / ↷ buttons do the same. A whole drag — and a
whole batched Confirm / Reset / Occlude — is a single undo step, and undo jumps back to
the frame the edit was on.

## Frame lists

The **Labels** button (`j`) opens a retractable side panel holding two lists as tabs.
Click a row in either to jump to that frame; `↑` / `↓` step through the **active** tab's
list (wrapping at the ends), and the current frame stays highlighted as you scrub.

### Labeled

Every frame carrying a label, in time order, with a per-frame **Reviewed** tick box —
your "I have finished checking this" flag, which persists with the labels and keeps the
frame listed even if its point labels are later reset. The list updates live and includes
labels loaded from a previous session; the button's blue badge shows the total even while
collapsed.

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

When the result carries a fitted inverse-kinematics model (the run enabled
[`do_inverse_kinematics`](../reference/configuration.md#inverse_kinematics)), three
extra overlays are available:

- **NMF skeleton** (`m`) — the fitted model joints, reprojected onto each view.
- **NMF mesh** (`Shift+M`) — the posed NeuroMechFly mesh, smooth-shaded on the client
  GPU at the view's resolution.
- **3D view** (`c`) — a floating panel showing the cameras, the derived pose, and the
  NMF skeleton/mesh together in 3D. Drag to orbit, `Shift`/right-drag to pan, scroll to
  zoom; it overlays the editor without blocking it, so the frame slider still scrubs.

Both overlays **re-fit to your labels**: as the derived 3D moves under your edits the
model is re-solved for that frame. The live re-fit runs on the **body plan the pipeline
solved**, read back from `results.h5` — this recording's measured segment lengths and its
registration to the model, so the live overlay and the rendered one describe the same
animal. (Older result files carry no plan; one is rebuilt from the `config.toml` snapshot
beside `results.h5` instead.) Which parts the mesh draws is set by
[`[gui].mesh_hide`](../reference/configuration.md#gui).

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

Edits live in memory until you save. **Save** (`Ctrl`/`Cmd`+`S`) writes the `labels.h5`
sidecar; **Close** stops the server (it offers to save first if there are unsaved
labels). The sidecar is stamped with the recording's fingerprint (skeleton, cameras,
frame count, image sizes, footage) so it is refused if pointed at a different
recording — but a re-run of the *same* recording (new detector weights, retuned
triangulation) keeps your labels valid, since ground truth is absolute.

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
| `m` / `Shift+M` / `c` | Toggle NMF skeleton / NMF mesh / 3D view |
| `j` | Show / hide the labelled-frames list |
| `Ctrl`/`Cmd`+`S` | Save labels — every recording holding unsaved work |
| `?` / `Esc` | Show shortcuts / close an overlay |

## Calibration landmarks

A from-scratch project has no camera rig, and the rig is solved from labels — but *which*
labels decides whether it converges:

> A skeleton keypoint at frame *t* is a **different 3D point** from the same keypoint at
> *t+1*, because the animal moved. So *N* labeled frames of *P* keypoints add `3·N·P`
> unknowns, all inside a 3 mm blob near the middle of the field. A **static** landmark — a
> scratch on the coverslip, the tip of the tether, a dust speck on the glass — is **one** 3D
> point observed in `V·N` images, spread through the scene *volume*. That is what conditions
> the solve.

Declare them in the project's `landmarks.toml`, then place them in the **Landmarks** tab:

1. Click a landmark row to **arm** it (the row highlights and the cursor becomes a crosshair
   over every view).
2. Click the same feature in as many views as can see it. Each click places one observation.
3. Click the row again — or leave the tab — to disarm.

Landmarks draw as **amber diamonds**, deliberately a different *shape* from keypoints rather
than just a different colour: a landmark is a different kind of thing, and shape survives
colour blindness and a busy frame.

The gesture is a click, not a drag, for a structural reason: a landmark has no detection to
grab and no reprojection to nudge, so until one exists there is nothing on the canvas to
start a drag from.

A **static** landmark keeps one 3D position for the whole recording, so re-placing it in more
frames sharpens it — but always on *the same feature*. `deeperfly calibrate` reports each
static landmark's pixel **scatter**, and a large value means it was not actually static, or
was placed on a different speck in a different frame. Both corrupt the solve.

Landmarks live in their own namespace in `labels.h5` and never touch the skeleton: they cannot
reach the detector, the IK body plan, the bone-length priors or the training export.

## Running the pipeline from the editor

Open a **project** (`deeperfly gui myproject/`) and the **Jobs** tab can run pipeline
commands: suggest frames, export labels, check or solve the calibration.

Each row *is* the CLI command, printed verbatim and click-to-copy — so a GUI action that fails
is reproducible in a terminal, and there is no behaviour only the GUI can reach. Jobs run one
at a time in their own process (two stages writing one `results.h5` would corrupt it, and a
CUDA OOM must not take your unsaved labels with it).

There is no progress percentage, on purpose: these commands emit human log lines, and a number
synthesized from those would be a fiction with a spinner attached. The last log line is shown
instead, which is the honest signal.

A session opened on a bare `results.h5` has no queue and says so — jobs need a project to run
in.
