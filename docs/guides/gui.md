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

Per `(view, point)` you author at most one of:

- **Ground truth** — an affirmed 2D pixel. Placed by dragging, or by confirming a
  suggestion (below). This is what gets saved and what a future training run consumes.
- **Occluded** — *"a human cannot place this point from this view."* The view is
  dropped from the 3D solve. This is a positive judgement, not "the pixel is hidden":
  if you can infer the location (e.g. the intersection of two visible segments), place
  ground truth instead.
- **nothing** — the view follows the detector's prediction, or (where the detector
  fired nothing) the 3D reprojection, as a *suggestion* you can accept or move.

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
  plus thumbnails) — toggle with the layout switch or `f` / `g`. In Focus, `[` / `]`
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
| **Hollow** circle in the point's left/right colour | **Projection** | no direct observation in this view — the 3D reprojected here (a suggestion) |

The legend in the toolbar (GT / Pred / Proj) mirrors these. The displayed point resolves
by precedence **ground truth → prediction → projection**. A view you **occlude** (below)
has its observation deleted, so it too shows as a projection — there is deliberately no
separate marker for it.

## Annotating the pose

**Dragging** a keypoint **creates ground truth** at the drop pixel — one gesture,
whatever the marker was. When the result carries a 3D pose the 3D **re-solves live** as
you drag, so every view's projection markers follow; the dragged view is authored as
ground truth and lands exactly under the cursor. (With a 2D-only result there is no 3D
to re-solve — the drop is simply that view's ground-truth pixel.)

### Confirming suggestions (fast path)

Most predictions are already good — you don't need to drag them, just accept them:

- **Confirm point** (`Enter`) — promote the selected joint's suggestion to ground
  truth in **every** view.
- **Confirm whole frame** (`a`) — promote every suggested point in the frame (all
  views) to ground truth at once.
- Per view: `l` confirms (or clears) ground truth for the selected point in the
  selected view.

Confirming a *prediction* snapshots the detector's pixel; confirming a
*reprojection* (a view where the detector fired nothing) snapshots the 3D's
reprojected pixel and is tagged as such, so it can be filtered out on export.

### Occluding a view

When a point cannot be read from a view, mark it **occluded** (`o`, or the point-status
widget): its observation is deleted, so it is dropped from the 3D solve and the other
views carry the reconstruction. On the canvas it then looks like any other projection (a
hollow palette circle) — occluding is just *"delete this view's observation"*, and a
deleted observation is indistinguishable from one the detector never made. It is fully
reversible — toggle it back, drag to place ground truth (which un-occludes it), or
**undo**. Occluded views are still recorded on export as a positive "unplaceable" label
(useful negative training signal), so the state is kept even though it has no marker of
its own.

The point-status widget shows the selected `(point, view)`'s state — **Predicted** /
**Ground truth** / **Occluded** — and sets it.

### Undo / redo

Every edit is undoable: `Ctrl`/`Cmd`+`Z` undoes, `Ctrl`/`Cmd`+`Y` (or
`Ctrl`/`Cmd`+`Shift`+`Z`) redoes; the ↶ / ↷ buttons do the same. A whole drag is a
single undo step, and undo jumps back to the frame the edit was on.

### Discarding labels

Three discards (the **Discard** group), each back to the pipeline's suggestion:

- **Point in view** (`r`) — the selected point's label in its view.
- **Point in all views** (`Shift+R`) — the selected point's labels everywhere.
- **Whole frame** — every label in the current frame, all views.

## Labelled-frames list

The **Labels** button (`j`) opens a retractable panel listing every frame carrying a
label, with the number of labelled keypoints in each. It updates live and includes
labels loaded from a previous session. Click a row to jump there, or use `↑` / `↓` to
step through labelled frames (wrapping at the ends); the current frame stays
highlighted as you scrub. The button's badge shows the total even while collapsed.

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
model is re-solved for that frame. The live re-fit uses the **same model the pipeline
did** — template, joint bounds, fitted legs, `fit_head`/`fit_abdomen`, and any
[custom marker placement](../reference/configuration.md#inverse_kinematics) all carry
over (read from the `config.toml` snapshot beside `results.h5`). Which parts the mesh
draws is set by [`[gui].mesh_hide`](../reference/configuration.md#gui).

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

This writes the provenance-filtered ground-truth pixels + occluded mask in footage
pixel space (reprojection-confirmed GT is excluded unless `--include-projection`).

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
| `f` / `g`, `[` / `]` | Focus / Grid layout; cycle the focused camera |
| `Enter` / `a` | Confirm the selected point (all views) / the whole frame |
| `l` / `o` | Confirm-or-clear ground truth / occlude-or-reveal the selected point (this view) |
| `r` / `Shift+R` | Discard the selected point's label in its view / all views |
| `Ctrl`/`Cmd`+`Z` / `Ctrl`/`Cmd`+`Y` | Undo / redo |
| `s` / `n` / `p` | Toggle skeleton / labels / 3D estimate |
| `m` / `Shift+M` / `c` | Toggle NMF skeleton / NMF mesh / 3D view |
| `j` | Show / hide the labelled-frames list |
| `Ctrl`/`Cmd`+`S` | Save labels |
| `?` / `Esc` | Show shortcuts / close an overlay |
