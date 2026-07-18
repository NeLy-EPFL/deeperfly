# Keypoint editor redesign — from "correction editor" to "ground-truth annotation tool"

**Status:** design / plan (pre-implementation).
**Scope:** the `deeperfly gui` keypoint editor (`src/deeperfly/gui/**` + `web/**`).
**Date:** 2026-07-17.

This document was drafted from a full map of the current editor and then hardened against a
five-lens adversarial review (schema soundness, solve correctness, latency/concurrency,
UX faithfulness, completeness). Findings from that review are woven in; the ones that changed a
decision are flagged inline as **[review]**.

---

## 1. Guiding philosophy

1. **2D observations are the only source of truth; 3D is a pure derived function.**
   `pts3d = triangulate(active 2D observations, cameras, method, hyperparams)`. The 3D that
   `deeperfly run` writes is *a cache* computed from the initial 2D predictions. The editor
   never edits 3D directly — it edits 2D observations and *recomputes* 3D.

2. **The GUI creates new ground truth (GT); it does not "correct" predictions.**
   Predictions and triangulation-projections are *free initial seeds* so the operator's mouse
   starts near the right place. The moment the operator affirms/moves a point it becomes a
   first-class GT annotation, independent of what the network predicted. The eventual consumer of
   these labels is detector re-training and evaluation (see §8.2).

The system as built is already ~80% aligned with idea (1) (see §11). The redesign is a
*re-conceptualization + consolidation* of existing machinery plus genuinely new pieces (undo,
multi-select/mass-confirm, durable-GT identity, an export seam).

---

## 2. Conceptual data model

### 2.1 Two orthogonal axes per `(frame, point, view)`

**[review]** The draft's single 5-value "source" enum conflated *what the operator authored*
with *what gets displayed*. Split them:

- **Authored state** (persisted, the only thing the operator controls): a tri-state
  `unset | gt(x,y) | occluded`.
- **Resolved display source** (derived at request time): `gt | prediction | projection | none`.

Resolution (default precedence `gt → prediction → projection`):

| authored | detector predicted here? | 3D exists? | display source | feeds 3D solve? |
|---|---|---|---|---|
| `gt(x,y)` | — | — | **gt** | **yes** |
| `occluded` | — | yes | **projection** (badged "excluded") | no |
| `occluded` | — | no | **none** (badged "excluded") | no |
| `unset` | yes (finite `pose2d` peak) | — | **prediction** (styled by `conf`) | yes |
| `unset` | no | yes | **projection** (unreviewed suggestion) | no |
| `unset` | no | no | **none** | no |

Key invariant: **projection is display-only.** A view showing a projection contributes *nothing*
to the 3D — it is the solve's own output reprojected, so feeding it back would be circular. The
solve reads only the *authored/predicted* view sources, never the payload's resolved `points`
field. **[review]** This invariant must be guarded in code (never wire the solve to the effective
`points` array).

### 2.2 The only authored (persisted) state — a sparse tri-state

Everything the operator authors reduces to the per-`(frame,point,view)` tri-state above. `gt`
and `occluded` are **mutually exclusive** (if you can place it, it isn't occluded). Everything
else — predictions, projections, 3D, reprojection error, review progress — is **derived**. This
is the "minimal, no-duplication" storage (§4).

### 2.3 Occluded semantics (operator-facing)

`occluded` means *"I cannot read this point's location from this view at reasonable confidence."*
It is **not** a geometric "is it hidden" flag. A point at the intersection of two visible
segments is occluded-in-pixels but the operator *can* place it — so they place GT, not occlude.
Occlusion is a positive human assertion that this view carries no usable information.

`occluded` (a human decision) is distinct from `absent` (no pathway ever targets this
`(view,point)` — e.g. the front-camera hind-leg keypoints have 0% training labels and are
*structurally* absent). Both are excluded from the solve, but only `occluded` counts as
"reviewed", and the two get visibly different markers so the operator knows a blank view is
"structurally impossible here" vs "I decided this is unusable".

### 2.4 Review progress (derived, free)

A `(point, view)` is **done** if it has GT or is occluded (a human made a call), else **pending**
(an unconfirmed prediction or a bare projection). This drives progress counts and a "jump to next
pending" review loop. **[review]** This must be reconciled with the existing corrected-frames
sidebar — see §5.7.

---

## 3. Resolution & triangulation policy (configurable)

Per the answer to fork #1, precedence, weighting, method, and hyperparameters are all
config-driven with sensible defaults. The GUI live-solve and the batch pipeline share the *same*
`[triangulation]` method/hyperparams so a zero-GT re-solve reproduces the run's cache.

> **[review] — this parity does not exist today and the redesign must create it.** The batch
> default is `method="ransac"` (`config.py:109`, `pipeline/core.py` `reconstruct_ransac`), but the
> live GUI imports only plain `triangulate` (`state.py:45`) and solves with unweighted DLT
> (`state.py:469`). So today, the instant you edit a point, its 3D is recomputed by a *different*
> estimator than produced the cache — and DLT silently re-includes views RANSAC had rejected as
> outliers (RANSAC-rejected-but-finite predictions are **not** seeded occluded). Fixing this is
> part of Phase A, not a pre-existing guarantee.

### 3.1 New `[annotation]` config section (proposed)

Complements the existing `[gui]` (`GuiParams`, `config.py:187`) and shares `[triangulation]`.

```toml
[annotation]
precedence = ["gt", "prediction", "projection"]   # projection is display-only

solve_policy = "gt_wins"          # "gt_wins" | "equal_weight" | "weighted_blend"
min_gt_for_exclusive = 2          # gt_wins: >= this many GT views ⇒ solve from GT only
gt_weight = 1000.0                # relative weight of GT rows when GT is mixed with predictions
prediction_weight = "uniform"     # "uniform" | "confidence" | <float>   [review] default uniform

confirm_default = "all"           # "all" | "predictions" | "projections"   (fork #2 answer)
low_conf = 0.2                    # predictions below this are visually de-emphasized (not hidden)
undistort_before_solve = true     # [review] undistort pixels before the linear DLT (see §3.4)
```

**[review] `prediction_weight` defaults to `"uniform"`**, not `"confidence"`: the batch default is
`weigh_by_confidence=false` (`config.py:114`) and this project's own finding is that heatmap-peak
confidence does not track correctness (see the `fusion-detector-eval` note). Confidence weighting
stays opt-in.

`[triangulation]` (unchanged, shared): `method="ransac"`, `ransac_threshold=15.0`,
`min_inliers=2`, `reproj_threshold=40.0`, `weigh_by_confidence=false`.

### 3.2 The three solve policies — corrected

Let `G` = views with a GT pixel, `P` = views with a usable prediction (finite `pose2d` peak, not
occluded); occluded views are excluded entirely. **No policy may ever discard a GT observation.**

- **`gt_wins` (default):**
  - `|G| ≥ min_gt_for_exclusive` → triangulate from **GT only** via the configured method.
  - `1 ≤ |G| < min_gt_for_exclusive` → **weighted DLT**: GT rows carry `gt_weight` ≫ prediction
    rows so the GT views define the point and predictions only stabilise depth. **[review]** This
    *requires* a weight — an unweighted `triangulate()` cannot "lock" a single GT (it is just two
    equal rows among N); `gt_wins` therefore carries a weight in the mixed case, contradicting the
    draft's "unweighted" wording.
  - `|G| = 0` → predictions only, through the **configured `[triangulation]` method** (RANSAC by
    default) so the result equals the run's cache byte-for-byte.
- **`equal_weight`:** per view use GT if present else prediction, drop occluded, feed to the run's
  RANSAC. **[review]** RANSAC inlier *voting* is unweighted (`triangulation.py:165`), so a GT that
  disagrees with the prediction consensus by > `ransac_threshold` would be **voted out and NaN'd**
  from its own solve (`core.py:436`) — silently discarding ground truth. Mitigation: GT views are
  **forced permanent inliers** (never subject to the reprojection vote); if that is not
  implemented, `equal_weight` must be documented as unsafe for GT and must *warn* when a GT view is
  dropped by consensus.
- **`weighted_blend`:** one weighted DLT — GT rows `gt_weight`, prediction rows
  `conf`/uniform — **no RANSAC voting** (avoids the vote-out problem). `|G| = 0` behaves exactly
  like `gt_wins |G| = 0` (configured method). **[review] cost:** bypassing RANSAC when GT is
  present loses outlier rejection over the prediction rows, so a confidently-mislocated prediction
  is only down-weighted, not rejected; gate prediction rows through a reprojection sanity check
  before blending.

**[review] Degenerate geometry:** `gt_wins` with `|G| ≥ min` throws away prediction views even
when the GT views have a short baseline / near-collinear rays (unstable depth). Either keep
predictions as low-weight depth stabilisers, or warn when the GT triangulation angle is small so
the operator adds a third view.

### 3.3 Two distinct solve entry points — **[review]** (the single-helper idea was wrong)

The draft claimed one `solve_point_3d(view_sources)` covers everything "and the fallback". It
can't: the `<2`-view ray fallback (`state.py:369-382`) needs the *dragged view* and the *prior
3D* (`closest_point_on_ray`, `geometry.py:396`), neither of which is in `view_sources`. Split:

```python
# Pure recompute — used by mass-confirm, occlude/un-occlude, navigation, batch.
# Below 2 usable views ⇒ returns NaN (no fallback).
def solve_point_3d(cameras, view_sources, policy, tri_params) -> xyz: ...

# Drag-time solve — owns the ray fallback. Below 2 usable views, back-projects the
# dragged pixel and slides the prior 3D the least-distance onto that ray.
def solve_point_3d_drag(cameras, view_sources, dragged_view, prior_xyz, ...) -> xyz: ...
```

`view_sources[v] = ("gt", xy) | ("prediction", xy, conf) | ("occluded",) | ("absent",)`. The
three current near-identical solve blocks (`state.py:361-368, 427-431, 468-475`) collapse into
these two. Mass-confirm / occlude solves do **not** ray-fallback (a bulk op has no single anchor).

### 3.4 Distortion — **[review]** (was understated as "sub-pixel, out of scope")

`triangulate_dlt` uses linear `pmats = K@[R|t]` with **no** lens distortion (`cameras.py:558`),
but the operator clicks GT in the *distorted* image. Consequences: (a) the DLT 3D from GT pixels
is distortion-biased; (b) that 3D reprojected through the full model does **not** land on the
clicked pixel, so a perfectly-placed GT carries a nonzero residual purely from distortion —
which in `equal_weight` eats the RANSAC inlier budget; (c) the "drag lands under the cursor"
guarantee only holds in the ray-fallback branch (full model), *not* the ≥2-view DLT branch.
**Fix (cheap, real):** undistort GT and prediction pixels to normalised coordinates before the
DLT (`undistort_one` already exists, `geometry.py:227`), gated by `undistort_before_solve`.

### 3.5 Mid-drag vs settle — **[review]**

The live mid-drag re-solve runs ~60×/s (`poseView.js:798` rAF throttle → `onDragging`). It must
**always** use the cheap incremental path (`solve_point_3d_drag`: weighted DLT / ray-fallback)
**regardless of `solve_policy`** — routing `equal_weight`'s RANSAC (a Python loop over C(7,2)=21
jit dispatches per re-solve) into the drag loop would reintroduce the exact drag lag the
`has_nmf` switch fights. `solve_policy` governs the **settle/release** solve only.

---

## 4. Persistence — sparse, identity-stamped labels file

### 4.1 Schema (replaces the dense `corrections.h5`)

Store only the authored deltas, sparse (COO), plus provenance and a real identity stamp. Proposed
file `labels.h5` next to `results.h5`.

```
attrs["meta"] = json {
    deeperfly_labels_format_version: 1,
    created_utc,
    # --- index domain (for name-based remap) ---
    skeleton_name, point_names: [...],      # point index ↔ name
    camera_names: [...],                    # view index ↔ camera
    n_frames,
    # --- recording identity (for refuse) --- [review]
    results_uuid,                           # stable id written into results.h5 meta (§4.3)
    footage: [basename_or_relpath, ...],    # per source, from results.h5
    image_sizes: {camera_name: [h, w]},     # footage space that GT xy lives in
}
gt/
    index       (N, 3) int32      # [frame, point, view]           [review] explicit int32 cast
    xy          (N, 2) float64    # affirmed 2D pixel (footage space)
    provenance  (N,)   uint8      # 0=dragged, 1=confirmed_prediction, 2=confirmed_projection
occluded/
    index       (M, 3) int32      # [frame, point, view]
```

No dense NaN arrays, no `fixed`, no `pts3d` (derived), no prediction-NaN-copy. Empty datasets are
written with shape `(0,3)`/`(0,2)`/`(0,)`. This also **eliminates the full `(V,T,P)` `isfinite`
scan** currently in `corrected_frames()` (`state.py:169`).

**[review] Provenance solves the fork-2 tension.** Fork #2's answer keeps `confirm_default="all"`
(promote predictions *and* projections). But a projection-sourced GT is the model's own guess
snapshotted as "truth", which (a) is one-step-circular when fed back to the solve and (b) pollutes
a store whose whole purpose is trustworthy GT. Recording *provenance* per GT row lets the default
stay "all" while making projection-sourced GT **filterable**: it can be excluded from export and
optionally down-weighted/excluded from the solve, and audited later. Drag = `dragged`,
confirm-a-prediction = `confirmed_prediction`, confirm-a-projection = `confirmed_projection`.

### 4.2 Load-time invariants — **[review]** (two independent COO lists can violate them)

The mutual-exclusion and uniqueness invariants of §2.2 are enforced in-memory by the edit ops but
are **not** guaranteed on disk (hand-merge, partial write, remap collision). On load, normalise:

1. **Dedup** each list; on a duplicate `(f,p,v)` in `gt` with conflicting `xy`, last-write-wins
   with a warning (or hard-error under a `--strict` flag).
2. **Disjointness:** `gt ∩ occluded` — **gt wins** (it carries an authored pixel); drop+warn the
   occluded dup.
3. **Fail-fast** with a clear message on anything unresolved.
4. **[review]** Filter migration/loaded GT by `np.isfinite(xy).all(-1)` — a GT that asserts "truth
   is nowhere" is dropped.

### 4.3 Identity check — **[review]** (the draft's "semantic identity" was oversold)

`point_names + camera_names + n_frames` are the *index domain*, not the *recording*. Two different
recordings from the same rig with the same frame count would be indistinguishable — reintroducing
the very stale-overlay hazard we set out to kill, and *weaker* than the old file, which at least
stored a `source` path (`corrections.py:167`). Two-tier check:

- **Recording identity (refuse on mismatch):** `results_uuid` and, as fallback, `footage`
  basenames + `image_sizes`. Add a stable `results_uuid` to `results.h5` meta at write time
  (small change in `results.py`); older results without one fall back to a hash of
  footage-basenames + image_sizes + `created_utc`. A re-run of the *same* recording with new
  detector weights or retuned triangulation keeps the same identity → **labels stay valid**
  (the whole point: GT is absolute, not relative to predictions). A different recording, or a
  changed footage resolution / front-camera crop, **fails loudly** — footage-space GT must never
  be silently reinterpreted in a new coordinate space.
- **Index-domain compatibility (remap, don't refuse):** if `point_names` / `camera_names` are the
  same *set* in a different *order*, remap both `gt/index` and `occluded/index` on both the point
  and camera axes by name. For a camera **subset/superset**, remap surviving views and
  **drop-with-warning** labels on removed cameras — don't discard the whole precious file because
  one camera was added.

### 4.4 Migration from `corrections.h5` (schema v3) — **[review]** honest, not "lossless"

Migration preserves authored 2D *pixels* as GT but does **not** preserve the old solve semantics
or the exact 3D — the 3D re-derives under the new policy. Report counts for every lossy step.

- `pts2d_edited` → **GT** at stored `xy`, provenance `dragged`. `fixed` collapses into GT.
  **[review] behaviour change to disclose:** an `edit_2d`-only move (`apply_2d_edit`, non-fixed)
  today does *not* constrain triangulation; as GT under `gt_wins` it now does, and a single-GT
  drag now re-solves 3D and shifts other views' suggestions. This is intended (GT is truth) but
  must be stated, not hidden under "lossless".
- `pts2d_invisible & isfinite(prediction)` → **occluded** (confident: a human deleted a real
  prediction).
- `pts2d_invisible & ~isfinite(prediction)` → **[review] ambiguous and lossy.** The old overlay
  *seeds* `invisible` from NaN predictions (`state.py:102`), and a human occlude on a NaN view
  sets the identical bit — the two are indistinguishable in the source. Default: **drop** these
  (treat as `absent`), but **count and warn** with the affected `(frame,point,view)` list; offer a
  `--keep-all-occluded` migration flag for operators who prefer to keep them and un-occlude the
  false positives by hand.
- `pts3d_edited` with no fixed 2D → **dropped**, warned (a pure-3D edit cannot be faithfully
  expressed as 2D GT without fabricating it).

**[review] CLI vs auto:** provide `deeperfly labels migrate <results_dir>` (explicit, reviewable,
testable) *and* auto-migrate on GUI load; auto-migration surfaces its lossy-drop report as a
**dismissable banner in the UI** (a server log is invisible to a browser operator). Never
overwrite `corrections.h5`; write `labels.h5` alongside. Decide and document whether a subsequent
save orphans `corrections.h5` (a downgraded `deeperfly` would then silently lose labels) — default:
leave it, warn once.

### 4.5 In-memory representation — **[review]** disk-sparse ≠ memory-sparse

The hot path resolves the effective `(V,P)` grid on every reply (~60×/s during a drag). Keep the
**in-memory** overlay either dense per-`(V,T,P)` (as today) or a `dict[frame] → records`, so
per-reply resolution stays `O(V·P)` and independent of total authored count. The COO form is
**disk-only**. Precedence resolution builds the frame's effective grid from `result.pts2d[:,t]`
overlaid with that frame's GT/occluded only.

**[review] confirm-all vs "minimal":** a whole-frame `confirm_default="all"` snapshots up to
`V·P` GT rows (~280 on a 7×40 rig), most of them projection pixels — the sidecar for that frame
becomes near-dense. This is an accepted trade (durability over sparsity); the re-solve after a
projection-only confirm is a no-op fast path (GT equals the current reprojection). Consider
keeping the *default confirm scope* narrow (selection/joint), reserving whole-frame confirm for an
explicit action.

---

## 5. GUI / UX

### 5.1 Default (clean) view — one point per `(point, view)`, distinct channels

**[review]** Encode source primarily by **marker/ring pattern (a strong perceptual channel)**, and
reserve **opacity solely for confidence** — the draft triple-booked opacity (confidence +
projection + occluded) and its new rings collided with the existing
hover(white)/selected(cyan)/fixed(lime)/invisible(dashed) vocabulary (`poseView.js:484-518`). Hue
stays reserved for body side (L/R), as today.

| display source | marker (proposal) | opacity |
|---|---|---|
| gt | filled disc, **solid bold** ring | full |
| prediction (unconfirmed) | disc, **thin** ring | scaled by `conf` (≤`low_conf` → faint) |
| projection (suggestion) | **open, dashed** ring | reduced |
| occluded | **hollow + slash** badge at projection location | reduced |
| none / absent | nothing, or a faint `+` hint if a 3D exists | — |
| hover / selected / multi-select | existing white / cyan + marquee highlight (compose on top) | — |

Final marker set must be checked for pairwise distinguishability against the retained
hover/selected/side-hue channels (no two states opacity-only distinguishable).

### 5.2 Verbose overlay mode (toggle, off by default)

Reveals the raw prediction ghost + projection ghost + `conf` labels simultaneously (clones the
existing `drawReference` latent/nmf ghost machinery, `poseView.js:534`). **[review]** The raw
prediction is **not** derivable from the effective `points` once a point is GT, so verbose needs a
separate raw-prediction array shipped only in verbose mode (§7.2). **[review]** Reconcile with the
existing "3D estimate" (`p`) latent overlay, which already ghosts the reprojected 3D: fold it into
the verbose toggle (or retire it in favour of the default per-view projection) so there aren't two
names for the same ghost.

### 5.3 Interactions

- **Drag** a prediction *or* projection → creates/moves **GT** at the drop (`solve_point_3d_drag`
  live re-solve, reusing the rAF-throttled `edit_3d` stream + cursor-pin, `poseView.js:308-310,
  798`); commit GT on release.
- **[review] Single-point confirm-in-place** (the highest-frequency action) → `Enter`/`Space` on
  the selected point (or a click on a prediction) snapshots its pixel as GT without dragging.
  Distinct from drag and from mass-confirm; it is what makes the §2.4 "jump to next pending" loop
  usable.
- **Occlude / un-occlude** selected point(s) → a key (see keymap §5.6) or the status widget;
  reversible (undo, toggle back, or drag to place GT). Replaces `toggle_invisible`.
- **Multi-select / box-select** → modifier+drag marquee (empty-drag stays pan), plus "select this
  joint across all views" and "select all in this view". **[review]** Box-select is a **client-only
  local selection** — no socket round-trip; only the bulk *action* on the selection is a message.
- **Mass-confirm** → promotes the selection's suggested positions to GT. **Default = everything
  shown** (predictions + projections), with **predictions-only / projections-only** subsets (fork
  #2). Provenance is recorded per row (§4.1). Scopes: box, whole view, whole joint, whole frame.
- **Undo / redo** → `Ctrl+Z` / `Ctrl+Shift+Z` (§6).

### 5.4 Mode simplification — **[review]** keep the collapse, fix the rationale

Collapse `edit_2d`/`edit_3d` into one editing model (drag always creates GT and re-solves). This
is faithful, not over-reach — there is only one thing to edit (2D GT); 3D always re-derives. But
the draft's rationale ("edit_2d is subsumed by the ray fallback") was **wrong**: `apply_2d_edit`
moves one view's pixel and touches neither the 3D nor other views, whereas the ray fallback
re-solves 3D and shifts every other view's projection. Correct rationale: *GT is inherently local
to the view where the pixel was dropped; 3D always re-derives; the old no-3D-coupling `edit_2d`
contradicted the vision.* Disclose the behaviour change (a single GT drag now moves other views'
suggestions). **[review]** This orphans the `/api/points` `mode` param, the `EditMode` type, and
the mode switch — drop `mode` or repurpose it as the verbose flag.

### 5.5 Status widget & Reset group — **[review]** (silently dropped in the draft)

- The per-view status widget (Normal/Fixed/Obscured, `index.html:33-36`, `app.js:680-726`)
  becomes a **source/state inspector-and-actor** for the selected `(point,view)`: shows
  `gt|prediction|projection|occluded|absent` and offers **Confirm / Occlude / Clear**.
- The Reset group (`Point in view` / `Point in all views` / `Whole frame`, `state.py:480-506`)
  becomes **clear authored GT/occluded → `unset`** at those scopes; relabel from "Revert"/"reset
  to prediction" (which now means "discard my GT") and define its relationship to Undo (Reset is a
  scoped clear that is itself undoable).

### 5.6 Keyboard map — **[review]** (must be enumerated; there is a real conflict)

Provide a full old→new keymap and rebuild `buildBindings` + the `?` help overlay
(`app.js:1106-1151`). Notes: `x` is already bound (pin-on-tap, `docs/guides/gui.md`) — pick a
non-colliding key for occlude or repurpose `x` deliberately; `l` (fix) becomes meaningless once
`fixed` collapses; new bindings needed for confirm-in-place, mass-confirm, verbose toggle,
jump-to-next-pending, and select mode.

### 5.7 Vocabulary & the corrected-frames sidebar — **[review]**

- **Relabel the visible surface** from "corrections" to "labels" / "ground truth": the sidebar
  title and empty state (`index.html:57-71`), the Save/close modal ("unsaved corrections"), button
  tooltips, and the header comment. The reframe is load-bearing, not cosmetic.
- **Redefine the corrected-frames sidebar** (`/api/corrected`, `state.py:154`) under the GT model:
  per-frame count = GT + occluded decisions; host the §2.4 pending/done surface and "jump to next
  pending" here rather than a separate widget.

---

## 6. Undo / redo — **[review]** corrected

A bounded **server-side** command stack on the authored overlay (the server holds authoritative
state and already replies with a full-frame payload). Each authored mutation is invertible:
`SetGT` / `MoveGT` / `ClearGT`, `SetOccluded` / `ClearOccluded`, `Confirm(batch)` (one unit).
Corrections to the draft:

- **One command per drag, not per mid-drag re-solve.** A drag streams ~50 `fix=false` re-solves
  before the `fix=true` commit (`poseView.js:798`, `app.js:747/767`). Only the **release commit**
  pushes an undo entry; mid-drag re-solves mutate the working 3D but never touch the stack (else
  `Ctrl+Z` steps through sub-drag frames and the 200-op ring fills in ~4 drags).
- **Route every socket mutation through the seq-stamp choke point** (see §7.4) — including undo /
  redo — or the reply is silently dropped.
- **Cross-frame:** undo can pop a command authored on a *different* frame than the one being
  viewed. `applyPoints` drops replies for a non-current frame (`app.js:506`), so undo/redo carries
  its target frame and the client navigates there before applying. (Simpler alternative consistent
  with per-frame scope: scope the stack to the current frame and seal it on navigation.)
- **Lifecycle:** stack lives in `EditorState` (mutated under the single lock), in-memory per
  session, unaffected by a browser refresh/reconnect, lost on server stop. Undo may cross the last
  save; `dirty` reflects "differs from last-saved snapshot" (§8.3).

---

## 7. Backend shape & latency

### 7.1 jit-stable solve — **[review]**

`triangulate_dlt` is `@jax.jit` (`geometry.py:538`) and recompiles when the point-axis size
changes. Today the GUI is thrash-free because it always solves one point at fixed `(V,1,2)`
(`state.py:367`). A mass-confirm must **not** solve a variable-`K` `(V,K,2)` batch (every distinct
`K` triggers a fresh XLA compile — tens-to-hundreds of ms each). Instead: loop per point at
`(V,1,2)` (reuses the one cached compile; ≤~40 dispatches is a few ms) **or** solve the whole
frame at fixed `(V,n_points,2)` with NaN padding. The `weights` None-vs-array branch
(`geometry.py:579`) adds one bounded one-time compile when weighting is first used.

### 7.2 Wire contract — **[review]** mode-aware payload

`source`/`conf`/raw-prediction are static within a frame but the mid-drag reply is rebuilt ~60×/s
(`server.py:539`, already omits `nmf` mid-drag). Make the payload mode-aware:

- **mid-drag reply:** `points` + `proj` only (as today, minus the extras).
- **settle / plain reply:** add per-`(view,point)` **authored-state** (`unset|gt|occluded`) and
  **display-source** (`gt|prediction|projection|none`) — two fields per §2.1, not one enum — plus
  `conf`.
- **verbose mode** (a request flag like the existing `mode`): additionally the raw `prediction`
  array (`V×P×2`) for the ghost.

New edit types (all seq-stamped): `set_gt`, `clear_gt`, `toggle_occluded`,
`confirm{scope, sources, targets}`, `undo`, `redo`. Old types kept as aliases during transition.
Precedence is resolved server-side, so the client draw stays `O(P)` styling with no per-frame
recompute.

### 7.3 Bulk ops = one message, one refit

Mass-confirm is **one** WebSocket message carrying the whole batch → the handler mutates the
overlay for all targets, calls `_invalidate_nmf(t)` **exactly once**, and builds the reply payload
**once** (so the expensive `nmf_fit` — 6 legs + chains, `max_nfev=100`, `nmf_live.py:111-141` —
runs once, not per point). It must **not** delegate to the per-edit `EditorState` methods in a
loop (each calls `_invalidate_nmf`). The single `asyncio.Lock` (`server.py:69,223`) makes it
atomic; the mesh follows via one debounced `scheduleMeshRefresh` → one `fetchNmfVerts` → one GPU
render. Keep the `include_nmf=False` mid-drag suppression (`server.py:516,539`) — the documented
`has_nmf` drag-lag switch.

### 7.4 The seq-stamp choke point is load-bearing — **[review] (blocker)**

Every WS reply is applied with `fromEdit=true` and dropped unless `p.seq === editSeq`
(`app.js:291,512`); the server echoes `msg.get("seq")` (`server.py:543`). So **every** new
mutation (`set_gt`, `clear_gt`, `toggle_occluded`, `confirm`, `undo`, `redo`) MUST go through
`sendEdit` (`app.js:735`, which stamps `++editSeq`), never a raw `socket.send` — otherwise the
edit mutates server state but the reply arrives with `seq=null`, is dropped, and the canvas never
repaints (looks broken). State this as a hard rule in the wire contract.

---

## 8. Downstream consumers & seams — **[review]** (missing from the draft)

### 8.1 3D scene panel + NMF overlays

`/api/scene` (`_scene_payload`), `scene3d.js`, and `NmfLive.refit` all consume `display_pts3d`.
The redesign re-points that derivation at `solve_point_3d` over effective view sources; specify
that `display_pts3d` / `_scene_payload` / `nmf_fit` read the new solve output and that the 3D panel
reflects the live GT-derived 3D during annotation.

### 8.2 GT export / training seam (the redesign's *purpose*)

Even if implementation is deferred, define the seam so §4's schema is validated against a real
consumer: a `labels.h5` → detector-training/eval contract with per-`(frame,point,view)` GT +
occluded semantics for the loss, provenance filtering (exclude `confirmed_projection` by default),
and — critically — the **coordinate transform**. GT is stored in **footage space**; the detector
trains in mirrored/cropped/resized **model-input space**. Export must invert each pathway's
`FrameTransform` (`normalized_peaks_to_original_pixels`, `pose2d/inference.py:384`) to move GT into
model-input space. This is also why `image_sizes` and a preprocessing fingerprint belong in the
identity stamp (§4.3).

### 8.3 Save / dirty / close flow

- **Save** writes `labels.h5` to the resolved path (`session` wiring updated from
  `corrections_path`; §12.4). Decide corrections.h5 retirement vs dual-write.
- **`dirty`** = "authored state differs from last-saved snapshot" (not a per-mutation boolean);
  undo-to-saved clears it; a migration-on-load counts as dirty until saved.
- The `beforeunload` guard and close-confirm modal (`app.js:299`, `index.html:76-86`) are preserved
  unchanged against the new `dirty` definition.

---

## 9. Testing strategy — **[review]** (the ~1200-line GUI suite is otherwise silently invalidated)

`tests/test_gui.py` and `tests/test_gui_server.py` assert directly on the machinery this redesign
deletes/renames (dense `Corrections` arrays; `apply_2d_edit`/`apply_3d_edit`/`toggle_*`/`reset_*`;
`save/load_corrections` shape-match; payload `fixed`/`invisible`; WS `edit_2d`/`edit_3d`). Per
phase, classify each test:

- **Port:** dense roundtrip → sparse `labels.h5` roundtrip; `load_corrections` shape-mismatch →
  identity refuse/remap; `_points_payload` shapes → new source/state fields.
- **Delete:** `fixed`/`invisible` mechanics; edit_2d/edit_3d split; invisible-seed test.
- **New:** `solve_point_3d` per policy (`gt_wins` |G|=0/1/≥min, `equal_weight` GT-inlier
  protection, `weighted_blend`), zero-GT-reproduces-run-cache, undistort-before-solve, migration
  (all lossy branches with counts), semantic-identity refuse/remap/subset, undo/redo (one-per-drag,
  cross-frame), mass-confirm batching (one refit, jit-stable), occluded reversibility, provenance.

---

## 10. Implementation phases (each independently shippable)

- **A — Model + storage core (backend).** `Labels` sparse dataclass + save/load + invariant
  normalisation + two-tier identity + `results_uuid` in `results.h5` + `corrections.h5→labels.h5`
  migration (CLI + auto). Rework `EditorState` to resolve effective 2D by precedence and to solve
  via `solve_point_3d` / `solve_point_3d_drag` (default `gt_wins`, configured method for |G|=0,
  undistort-before-solve). Add `[annotation]` config. *UI behaviour unchanged; model cleaned and
  the DLT-vs-RANSAC mismatch fixed underneath.* Port/add backend tests.
- **B — Wire contract.** Mode-aware payload (authored-state + display-source + `conf`; verbose raw
  prediction); new seq-stamped edit types; old types aliased. Payload/server tests.
- **C — Clean default rendering.** One point per `(point,view)` by distinct marker channels +
  confidence opacity; occluded/absent markers; projection style; collapse edit_2d/edit_3d (drop
  `mode`); relabel vocabulary.
- **D — Undo/redo** (server command stack, one-per-drag, frame-carry) + reversible occlude +
  status-widget/Reset respec.
- **E — Multi-select + box-select + mass-confirm** (all / predictions / projections; box / view /
  joint / frame; one message, one refit, jit-stable) + single-point confirm-in-place.
- **F — Verbose overlay + confidence surfacing + review-progress** (pending/done, jump-to-next,
  corrected-sidebar redefinition) + reconcile the `p` latent toggle.
- **G — Config polish + docs** (`[annotation]`; update `web/README.md`, `docs/guides/gui.md`,
  `docs/guides/cli.md`, `docs/reference/configuration.md`, `docs/reference/output-format.md` — add
  the `labels.h5` schema entry) + full keymap + `?` help rebuild.
- **H — GT export seam** (footage→model-input transform, provenance filter) — may be deferred but
  its contract is fixed in Phase A so the schema supports it.
- **Later — cross-frame propagation** (copy-to-neighbour, interpolate between keyframes) — the
  deferred fork-#3 phase.

---

## 10a. Implementation status & decisions made in code

**Done + tested (Phase A foundation; additive, the pre-existing suite stays green):**

- **`[annotation]` config** (`config.py` `AnnotationParams`, `Config.annotation`). Two
  refinements vs the plan above: `prediction_weight` defaults to `"uniform"` (not
  `"confidence"`) to match the batch and the peak-conf-≠-correctness finding; and
  `undistort_before_solve` defaults **False** — undistorting only in the GUI would break
  the `gt_wins |G|=0` run-cache parity, so it is an opt-in accuracy trade. Extra
  "build-both" knobs: `equal_weight_protect_gt`, `gt_wins_keep_stabilizers`.
- **`gui/labels.py`** — the sparse `labels.h5` store (§4). Refinements: identity is a
  **fingerprint** (skeleton points + camera names + `n_frames` + `image_sizes` + footage
  basenames), not a stored `results_uuid` — no `results.h5` change, and it works for
  existing files; it deliberately excludes predictions and `created_utc` so a
  prediction-only re-run keeps labels valid. GT rows carry a **provenance** byte
  (`dragged`/`confirmed_prediction`/`confirmed_projection`). In-memory is dense
  `(V,T,P)`, disk is COO. Load normalises invariants (GT wins GT/occluded ties).
  `migrate_from_corrections` maps `edited`+`fixed`→GT and confident `invisible`→occluded,
  and reports every lossy drop.
- **`gui/solve.py`** — `solve_point_3d` (all three policies; `|G|=0` routes through the
  configured `[triangulation]` method for run parity; GT is never voted out under
  `equal_weight`) and `solve_point_3d_drag` (cheap weighted-DLT / ray-fallback,
  policy-independent per §3.5).
- Tests: `tests/test_gui_labels.py`, `tests/test_gui_solve.py`.

**Done + tested (Phase A5, the rewire).** `EditorState` now holds `Labels` (not
`Corrections`), resolves effective 2D by precedence, and derives 3D via `solve.py`.
**Decision implemented:** the 3D is *purely derived* — no stored `pts3d`.
`display_pts3d` returns `solve_point_3d(gt, pred)` where it yields ≥2 usable views, else
falls back to `result.pts3d` (the run cache); a single GT view with no predictions can't
move the 3D (documented edge case §10.3). `server.py` `_points_payload`/`save` use the
labels masks + `save_labels`; `session.py`/`__init__.py` wire a `labels.h5` path with an
identity fingerprint and **auto-migration** of a legacy `corrections.h5` on load (lossy
drops logged). The server **wire contract is unchanged** (`edit_2d`/`edit_3d`/
`toggle_fixed`→confirm-GT/`toggle_invisible`→occlude/`reset_*`; payload keeps
`points`/`fixed`(=has-GT)/`invisible`(=occluded)/`proj`/`nmf`/`seq`), so the existing
front-end runs against the new model. `tests/test_gui.py` rewritten for the labels model,
`test_gui_server.py` updated; the **full suite is green** plus an end-to-end smoke
(build → serve → drag → save `labels.h5` → reload → GT persists).

**Done + tested (Phases B–H).**
- **B (wire contract):** new seq-stamped edit types `set_gt` / `clear_gt` /
  `toggle_occluded` / `confirm{targets,sources}` / `undo` / `redo` (old types aliased);
  mode-aware payload — `conf` + `can_undo/can_redo` on settle replies, raw `pred` behind a
  `verbose` flag, `goto` on undo/redo.
- **C (rendering + relabel):** the visible surface is relabelled `corrections`→`labels` /
  ground-truth (index.html, app.js, docs); plain predictions fade with detector confidence.
  **Mode collapse DONE** — the `Edit 2D`/`Edit 3D` selector and `2`/`3` keys are gone; there
  is one unified editor (a drag authors GT and, when the result has 3D, re-solves it live;
  routing keys on `meta.has_3d`, not a user mode). The client always requests the plain
  per-view overlay (`display_pts2d`); the server's `edit_3d` refine display is retained for
  tests/back-compat. **Per-source markers DONE** — each joint's marker encodes its source.
  Refined (2026-07-17, per user) to **three** visual sources, not four: ground truth
  (solid lime ring over a filled disc), prediction (thin dark ring over a confidence-faded
  filled disc), and a **derived** point reprojected from the 3D (a *hollow circle in the
  point's own left/right palette colour* — no fill, no amber). The derived style covers
  both "the detector fired nothing here" and an **occluded** view, since occluding NaNs the
  observation out of `display_pts2d` exactly like an absent detection — a deleted
  observation is indistinguishable from one never made, so it gets no separate marker. The
  occlude *action/state/export* is unchanged (it still suppresses a confident-wrong
  prediction from the solve and is written as a positive "unplaceable" label). Toolbar
  legend is now GT/Pred/Proj; `drawLeashes` no longer double-draws the projection the
  skeleton shows inline. Grid is now the default layout (focus stays a `f`/switch away).
- **D (undo/redo):** a bounded server-side command stack (one step per drag via
  coalescing; carries the target frame so the client navigates on undo across frames);
  `Ctrl/⌘+Z` / `Ctrl/⌘+Y` + ↶/↷ buttons; occlude is reversible; the status widget reads
  Predicted / Ground truth / Occluded and the Reset group became "Discard".
- **E (confirm):** `EditorState.confirm(targets, sources)` (one undo step, one IK refit)
  wired to "Confirm point" (all views, `Enter`) and "Confirm whole frame" (`a`), plus
  single-view confirm-in-place (`l`). *Deferred:* box/marquee multi-select.
- **F (confidence):** `conf` shipped and rendered as a low-confidence fade. *Deferred:*
  the verbose prediction-ghost overlay and the pending/done review-progress panel.
- **G (docs):** `guides/gui.md` rewritten; `guides/cli.md` (+ `labels-export`);
  `reference/configuration.md` (`[annotation]`); `reference/output-format.md` (`labels.h5`);
  `web/README.md`; keymap + `?` help updated.
- **H (export seam):** `labels.export_gt()` (provenance-filtered, footage space) +
  `deeperfly labels-export` CLI.

Verification: the full test suite is green, with an end-to-end wire smoke exercising
confirm / occlude / undo / redo / verbose / save / reload / export.

**Remaining:** a browser-verification pass to finish the still-deferred rendering polish
(verbose prediction-ghost overlay, box/marquee multi-select, pending/done review-progress
panel) with live feedback — there is no `node`/browser in this environment. The mode
collapse and per-source markers, previously on this list, are now built (see Phase C);
they need a visual check but were implemented as data/routing + self-contained canvas
drawing, so the risk of editing them without a browser was low.

## 11. What already exists (so we build, not rewrite)

- **2D-source / 3D-derived**: `results.h5` stores 2D and 3D in separate stage groups
  (`results.py:11-33`); the GUI already re-triangulates live from a dynamic per-view obs set via
  the NaN convention (`state.py apply_3d_edit:310`, `_resolve_3d_from_visible:455`,
  `_resolve_3d_from_fixed:419` → `triangulation.py triangulate:26`).
- **GT-over-prediction**: `display_pts2d = np.where(edited, GT, prediction)` (`state.py:179`).
- **Projection pseudo-point**: the `proj` payload / "latent skeleton" ghost (`state.py:199`,
  `poseView.js:534,611`).
- **Per-view disable**: the `invisible` mask — to be disentangled from "detector-missed".
- **Confidence**: computed (`results.py:87`, `pose2d/inference.py`) but never shipped or used in
  the live solve.
- **Shared triangulation config**: `TriangulationParams` (`config.py:105`).
- **Two proven latency mitigations**: the single `asyncio.Lock` and the `include_nmf=False`
  mid-drag switch.

New pieces with no existing scaffold: **undo/redo**, **multi-select/box-select/mass-confirm**,
**recording-identity stamp**, the **sparse+provenance schema**, and the **export seam**.

---

## 12. Open risks & decisions to confirm

1. **`equal_weight` GT protection:** implement GT-as-permanent-inlier, or ship `equal_weight` as
   "unsafe for GT, warns when GT dropped"? (§3.2)
2. **`gt_wins |G|≥min` depth stability:** GT-only vs keep low-weight prediction stabilisers under
   degenerate geometry? (§3.2)
3. **Confirm default scope:** keep `confirm_default="all"` at *frame* scope (densifies the
   sidecar), or narrow the default scope to selection/joint and reserve frame-confirm for an
   explicit action? (§4.5)
4. **File rename `corrections.h5→labels.h5`** and `session.corrections_path` rewiring
   (`__init__.py:93`); corrections.h5 retirement vs dual-write for downgrade safety. (§8.3)
5. **Undo cross-frame** (carry-frame-and-navigate) vs **per-frame-sealed** stack. (§6)
6. **Multi-annotator / locking** — explicitly out of scope (single lock, one operator).

---

## Appendix — provenance of this plan

Grounded in a five-reader subsystem map (frontend pose-view, frontend app-state, backend
persistence, triangulation, pipeline data model) and hardened by a five-lens adversarial critique
(schema soundness, solve correctness, latency/concurrency, UX faithfulness, completeness). All
**[review]** annotations mark a place where that critique corrected or extended the first draft.
