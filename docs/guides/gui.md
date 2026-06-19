# Correction GUI

`deeperfly gui` opens an interactive **web** viewer/corrector for a result. It
serves every camera view with its 2D skeleton overlay to a browser canvas and lets
you drag keypoints to fix the pose. Corrections are written to a `corrections.h5`
sidecar next to the result and **never modify `results.h5`** — re-running the
pipeline is always safe.

Because it is a browser app it needs no GUI toolkit, runs headless, and can be
reached from another machine (see [Remote use](#remote-use)).

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
[CLI reference](cli.md#deeperfly-gui-correct-a-result) for every flag
(`--host`, `--port`, `--no-browser`, `--keep-alive`).

## Layout

- **Views.** Every camera is shown with its overlay. Two arrangements share the
  same canvases: **Focus** (one big view plus thumbnails) and **Grid** (all
  cameras equally) — toggle with the layout switch or `f` / `g`. In Focus, `[` /
  `]` cycle which camera is enlarged.
- **Frame strip + slider.** Scrub with the slider, the `←` / `→` keys (`Shift` for
  ±10), or type a frame number. The slider drives the whole editor, including the
  3D view, so everything stays on the same frame.

## Editing the pose

The mode switch chooses **what a drag does** (keys `2` / `3`):

- **Edit 2D** — drag a keypoint to move it **in that view only**. Each view is
  independent; use it to fix a single bad detection.
- **Edit 3D** — drag a *reprojected 3D* point. The 3D point is re-solved from the
  drag and **every other view updates live** to its new reprojection. This is the
  fast way to fix a point everywhere at once. (Only available when the result has a
  3D pose.)

To just inspect, don't drag — both modes are read-only until you grab a point.

### Fixed and obscured points (Edit 3D)

Calibration is never perfect, so one 3D point rarely reprojects exactly onto every
view. Each per-view point therefore has a state, shown in the **point status**
widget for the selected point/view and set from there or by key:

| State | Key | Meaning |
| --- | --- | --- |
| **plain** | — | The view follows the shared 3D point's reprojection. |
| **fixed** | `l` | The view is *pinned* at its pixel and acts as a constraint: the 3D point is re-triangulated from the fixed views, so the rest agree with your finalized pixels. Dropping a drag also fixes that view at the release pixel. |
| **obscured** | `o` | The camera genuinely cannot see the point: it is dropped from triangulation entirely and just follows the reprojection (it can't be dragged). The 3D point re-solves from the remaining visible views. |

A point is at most one of fixed / obscured / plain. Fresh sessions start a view
**obscured** wherever the detector returned no point (a `NaN`); drag it in to
reveal it. **Pin-on-tap** (`x`) makes a tap fix/unfix a point instead of dragging
it.

### Resetting

Three reverts (in the controls bar), each back to the pipeline's original pose:

- **Point in view** (`r`) — just the selected point in its view.
- **Point in all views** (`Shift+R`) — the selected point everywhere (2D, fixed
  flags, and the 3D point).
- **Whole frame** — every point in the current frame, all views.

## Corrected-frames list

The **Corrections** button (`j`) opens a retractable panel on the right listing
every frame you have touched, with the number of corrected keypoints in each. It
updates live as you edit, obscure, or reset. Click a row to jump to that frame, or
use the panel's `↑` / `↓` buttons to step through corrected frames (wrapping at the
ends); the current frame stays highlighted as you scrub. A frame is listed when a
keypoint's 2D was moved, its 3D was re-solved, or its visibility differs from the
detector's own — so it mirrors exactly what the `corrections.h5` sidecar stores,
including corrections loaded from a previous session. The button's badge shows the
total count even while the panel is collapsed.

## NeuroMechFly overlays

When the result carries a fitted inverse-kinematics model (the run enabled
[`do_inverse_kinematics`](../reference/configuration.md#inverse_kinematics)), three
extra overlays are available:

- **NMF skeleton** (`m`) — the fitted model joints, reprojected onto each view.
- **NMF mesh** (`Shift+M`) — the posed NeuroMechFly mesh, smooth-shaded on the
  client GPU at the view's resolution.
- **3D view** (`c`) — a floating panel showing the cameras, the triangulated pose,
  and the NMF skeleton/mesh together in 3D. Drag to orbit, `Shift`/right-drag to
  pan, scroll to zoom; it overlays the editor without blocking it, so the frame
  slider still scrubs everything.

Both overlays **re-fit to your corrections**: as you edit the 3D latent skeleton,
the model is re-solved for that frame and the overlay follows. The legs skin to the
corrected keypoints; the head and abdomen are fixed model geometry sized to the fly
by a per-recording scale the IK stage estimates from the data. The live re-fit uses
the **same model the pipeline did** — the run config's template, joint bounds,
fitted legs, `fit_head`/`fit_abdomen`, and any
[custom marker placement](../reference/configuration.md#inverse_kinematics) all
carry over (read from the `config.toml` snapshot beside `results.h5`).

Which body parts the mesh draws is set by
[`[gui].mesh_hide`](../reference/configuration.md#gui) (default: hide the wings);
the rendered videos use `[visualization].mesh_hide` instead.

## Other overlays

| Toggle | Key | Shows |
| --- | --- | --- |
| Skeleton | `s` | The editable 2D skeleton overlay. |
| Labels | `n` | Each keypoint's name. |
| Latent 3D | `p` | The triangulated 3D estimate ghosted onto each view (no fixed overrides). |
| Keypoints ↗ | `k` | The reference [keypoint viewer](../explanation/keypoints.md) (new tab). |

## Saving

Edits live in memory until you save. **Save** (`Ctrl`/`Cmd`+`S`) writes the
`corrections.h5` sidecar; **Close** stops the server (it offers to save first if
there are unsaved edits). The sidecar records its source `results.h5`, so a later
session re-loads your corrections.

## Remote use

The server binds loopback by default, so the editor is private. To correct a
result on a remote machine, tunnel the port over SSH and open the browser locally:

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
| `2` / `3` | Edit 2D / Edit 3D mode |
| `f` / `g`, `[` / `]` | Focus / Grid layout; cycle the focused camera |
| `s` / `n` / `p` | Toggle skeleton / labels / latent 3D |
| `m` / `Shift+M` / `c` | Toggle NMF skeleton / NMF mesh / 3D view |
| `x` | Pin-on-tap (Edit 3D) |
| `l` / `o` | Fix / obscure the selected point (Edit 3D) |
| `r` / `Shift+R` | Reset the selected point in its view / all views |
| `j` | Show / hide the corrected-frames list |
| `Ctrl`/`Cmd`+`S` | Save corrections |
| `?` / `Esc` | Show shortcuts / close an overlay |
