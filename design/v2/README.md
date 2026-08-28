# deeperfly v2 -- a breaking refactor, not a config rewrite

**Status:** design / plan (pre-implementation).
**Scope:** the run config; the skeleton domain model; footage discovery and decode; the
calibration solve and its GUI surface; every example config and every doc page that names
a config key.
**Release:** 0.3.0. Hard break, no migrator -- a v1 config fails to load with an error
naming the v2 table that replaced the offending one.
**Date:** 2026-08-27.

Read [config-schema.md](config-schema.md) for the schema itself and its before/after
tables; [PLAN.md](PLAN.md) for how it lands, workstream by workstream.
[ik-model-packs.md](ik-model-packs.md) for W9, which lets the IK stage fit a model
other than NeuroMechFly and makes the skeleton, the model and the binding between them
three separate artifacts. [default_config.toml](default_config.toml),
[example_scape_E49.toml](example_scape_E49.toml) and [fly38.toml](fly38.toml) are the
target files.

## Why this stopped being a config rewrite

It began as "replace the v1 config schema". Three things pushed it wider, and all three
are reasons the schema gets to be as small as it is:

1. **The schema enables new behavior.** `[cameras.<name>].video` as a list of regexes
   means *concatenation*, which changes how frames are **read**, not how a run is
   described. Every other item here is a surface change; this one is a new feature with a
   silent failure mode (a frame-index bug after a part boundary succeeds and mislabels
   every subsequent frame), so it needs its own byte-identical gate.
2. **The schema requires domain-model surgery.** The limb/chain concept exists in
   `Skeleton`, in `results.h5` and in 22 places in the GUI's JS *because the v1 config
   declared it*. Deleting the declaration means deleting the concept -- ~144 references in
   `src/`. `pictorial.py` already derives chains from the bone graph, which is what makes
   this safe rather than lossy.
3. **Features are being retired on their own merits.** Calibration landmarks, sparse and
   multi-pathway detection, `mirror`, the preprocessing op grammar. None of these are
   schema questions. All of them are why the schema has nothing to say about them.

So v2 is the release where deeperfly's **domain model narrows to what it actually does** --
one dense detector run once per camera, an animal-as-calibration-target rig, a skeleton
that is points and edges -- and the config schema is the visible face of that narrowing.

## The removals, and the evidence for each

The pattern is the same every time: a generalization was built for a case that either
never arrived or was solved another way, and it is now paid for at every call site.

| Removed | Why it existed | Why it goes |
| --- | --- | --- |
| **Calibration landmarks** (`landmarks.py`, the GUI panel, the whole namespace) | `project-system-plan.md` item 3: a from-scratch rig conditioned by static non-skeleton points spread through the scene volume | Never used. **Zero `landmarks.toml` and zero `landmarks/` groups exist** anywhere under `~/fly-pose-data`, `~/deeperfly-runs` or `examples/`, so every rig ever solved here was already animal-as-target. See below. |
| **Sparse / multi-pathway detection** (`output_points`, mirrored pathways, `[[sources]]`) | DeepFly3D's 19-channel checkpoints, and one model serving two views through a flip | The detector is dense-38 and one-to-one with cameras. A 19-channel checkpoint runs under a v1 tag, not a resurrected code path. |
| **The limb/chain concept** (`limb_names`, `limb_id`, `n_limbs`, the limb-keyed palette) | Grouping points for color, symmetry inference and chain consistency | Color becomes a per-point selector; symmetry becomes an automorphism check on `edges`; chains are derived from the bone graph where they are needed. |
| **The preprocessing op grammar** (`fliplr`, `flipud`, `rot90`, config-level `resize`) | Arbitrary per-pathway frame ops | After the mirrored pathways die, **`fliplr` has no consumer outside a deleted test fixture**, and `resize` as a config op never had one (`pathways.py` constructs `Resize` internally to fit the detector input). What survives is `Crop` + `Resize` + `FrameTransform` as internal machinery. |
| **`[cameras.<name>].mirror`** | Flip augmentation during training needed the id of the camera a mirrored sample looks like | Its in-repo consumer went in 0.2.0; dfpose hardcodes its own `NEWRIG_MIRROR_CAMERA`. Derivable from `azimuth_deg` if ever wanted. |
| **The body-frame alignment state** (`Alignment.r_body`, `.head_origin`, `align.to_local`, `align.to_world`) | Rotating the leg subtree into the recording's coxa-derived body frame | That rotation was **removed** when the leg subtree went identity (`bodyplan.py:40`); the four survivors have no production consumer, only `tests/test_inverse_kinematics.py`. They also carry the IK stage's only hardcoded leg names. See [ik-model-packs.md](ik-model-packs.md). |
| **Calibration scale pinning** (`--scale-from`, `scale_pair`/`scale_distance`, `UNITS`' `"mm"`, `SCALE_SOURCES`' `known_distance`) | Pinning physical units at the calibration stage from a measured landmark-to-landmark distance | **Physical scale is not a calibration-stage concern.** Everything through triangulation is arbitrary units by design; scale first becomes physical at IK, where `body_scale` fits the point cloud to the NeuroMechFly model's defined dimensions. See below. |

## Landmarks: what the removal actually costs

Landmarks were never a bookkeeping convenience -- they changed the geometry. A static
landmark is **one** 3D unknown observed in `V*N` images; a keypoint at frame *t* is a
different 3D point from the same keypoint at *t+1*, so it is 3 unknowns per frame. The
`landmarks.py` docstring makes two arguments from this. Only one of them was ever
binding here:

- **Equation efficiency -- not binding on this rig.** A keypoint track observed in `V`
  views contributes `2V` equations against 3 unknowns, so the ratio per track is `2V/3`.
  At `V = 8` that is **5.33x**, against a `MIN_EQUATION_RATIO` of **1.5**
  ([calibration_solve.py:65](../../src/deeperfly/calibration_solve.py#L65)). Keypoints
  were never starved.
- **Conditioning -- real, and empirically it did not bite.** Keypoints all sit inside a
  ~3 mm blob near the middle of the field, where a landmark is spread through the scene
  volume. This is the substantive argument, and the honest cost of the removal is that
  cold starts lose it. But cold start is a live path (the GUI when `not has_rig`, the CLI
  `--cold`) that has **only ever been exercised without landmarks**, because none exist.

So this removal ratifies what already happens rather than betting on it. Two guidance
strings that currently recommend adding a static landmark
([calibration_solve.py:462](../../src/deeperfly/calibration_solve.py#L462) and
[:468](../../src/deeperfly/calibration_solve.py#L468)) must be rewritten to name the
remedy that exists: label the same point in a view from each disconnected group, and
label more frames.

Downstream, `Track` loses `kind` and `static`, `merge_observations` loses `share` and
the rig-scoped cross-recording tie, and multi-recording solves become plain
concatenation.

**The word survives in a different sense and must not be swept up.** A chain's **base
landmark** in `inverse_kinematics/` (the head's `neck`) is an IK marker, unrelated to
calibration; so is the prose in `results.py:471`, `config.py:584` and
`pathways.py:330`. `visualization/bird.py`'s local `_landmarks` helper is an
orientation-group lookup and should be renamed to stop the collision.

## Scale: where physical units actually enter

Worth stating once, because the removal above is easy to misread as losing metric units:

- Images cannot determine scale -- a rig twice as large viewing a fly twice as large
  produces pixel-identical images. A correspondence-only solve has 7 gauge freedoms, and
  the scale one needs an external measurement.
- `_normalize_initial_scale`
  ([calibration_solve.py:1041](../../src/deeperfly/calibration_solve.py#L1041)) does
  **not** supply one. `cv2.recoverPose` returns a unit baseline, landing a cold start
  ~180x too small; that function rescales so the median camera-scene distance equals the
  focal length in pixels. It changes no reprojection. The result is still
  `units = "arbitrary"`, and that is correct.
- **Physical scale enters at inverse kinematics.** `IKResult.body_scale` is "the
  recording's body size relative to the model", fitted from the anchor registration, and
  the body plan lives in model units. The fitted model -- NeuroMechFly by default, any
  pack under W9 -- has defined dimensions, so that factor is the first place a length in
  this pipeline means anything physical. Which model supplied them is therefore part of
  the result, and W9 records the pack name beside the angles.

Under v2, then, `units` is `"arbitrary"` or `"config"` (whatever `[default_camera]
distance` was in), `SCALE_SOURCES` keeps `none`, `orbit_prior`, `board` and `imported`,
and nothing in the calibration path claims millimeters.

## Workstreams

Ordered for landing, but the dependency structure is looser than the numbering suggests
-- **W1, W7 and W9 all touch the GUI's JS**, and W4, W7 and W9 are independent of
everything else, so any of them can go first if it is convenient. W8 stays last.

| # | Workstream | Depends on | Nature |
| --- | --- | --- | --- |
| W0 | Test suite: restore the dead browser gate, document the inner loop | -- | prerequisite for W1 and W7 |
| W1 | Skeleton: points, edges, symmetries, colors; limb/chain deleted | -- | domain model, wide + mechanical |
| W2 | Rig: `[calibration]`, `[default_camera]`, `[cameras.<name>]`; `mirror` deleted | W1 | surface |
| W3 | Detection: one detector, per-camera crop, synthesized plan; op grammar deleted | W2 | surface + dead-code removal |
| W4 | Footage: regex matching and concatenation | -- | **behavioral** -- the only one |
| W5 | Pipeline flags, named collections, point selectors | W1 | surface |
| W6 | Visualization layers and defaults | W1 | surface |
| W7 | Landmarks removed, scale pinning with them | -- | feature removal |
| W9 | IK model packs, and the skeleton-to-model adaptor that holds the offsets | -- | domain model + one behavioral thread |
| W8 | The configs, the docs, the release | all | churn + the real gate |

## Gates

The suite is necessary and nowhere near sufficient, because most of v2 is a surface
change whose failure mode is "runs fine, describes a different run". Three gates carry
the confidence:

0. **The GUI browser test, restored.** `tests/test_gui_browser.py` -- 1859 lines, 53 tests
   -- runs in no environment that exists: `playwright` is in no dependency group and no CI
   job, and the module-level `importorskip` reports all 53 as one skip. It is the only gate
   over the GUI's JavaScript, which W1 (22 `limb` references) and W7 (88 landmark references
   across five web assets) both rewrite. Injected with `uv run --with playwright` all 53
   pass, so this is a packaging omission, not rot. See W0a in PLAN.md for what restoring
   costs (168 s serial, or ~16 s parallel with two reproducible races to settle).
1. **`ConcatReader` against the single-file reader**, byte-identical for random access,
   slices and a full decode, over a recording split with `ffmpeg -f segment`. No example
   recording is split, so the fixture has to be made. This is the only gate that can
   catch a frame-index bug, because nothing downstream can.
2. **One full `deeperfly run` per example directory** (eight, ~3000 frames x 8 cameras),
   checked for exit 0, every enabled stage present in `results.h5`, reprojection error in
   the same band as the cached `deeperfly_outputs_*` beside it, and videos at the expected
   size. This is the long pole; the runs go in the background while the suite proceeds.
3. **`mkdocs build --strict`, plus a grep for every deleted key name across `docs/`.**
   A removed key that survives in prose is how the next reader learns something false.

## Candidates -- not in the plan, your call

Found while auditing, in the same spirit as the removals above, and deliberately kept
out of PLAN.md until accepted:

| # | Candidate | The case for | The case against |
| --- | --- | --- | --- |
| C1 | Delete `skeleton_migrate.py` (+ its CLI command and test) | It exists for the fly38 -> fly38b migration, which is **done** across the corpus. Its only caller is `cli/project.py:578`. | The next skeleton change would want it again -- though it would want it rewritten for that change, not resurrected. |
| C2 | Declare `bird` in the schema | After v2 it is the **only name in the config that means something without being declared anywhere** -- a grid cell that gets no footage because the parser recognizes the string. | It works, and declaring it costs a table for one value. |
| C3 | Rename `limb_id` / `limb_names` in `results.h5` | Otherwise the file format says `limb` while the config and code say nothing of the kind -- one documented mismatch in `_write_skeleton`/`_read_skeleton`. | A corpus repack of 351 files / 16.9 GB, unless done lazily on the next write. The v3 repack is already owed, so these could ride together. |
| C4 | Drop the v1 output snapshots in `examples/*/deeperfly_outputs*/config.toml` | They are v1 configs inside the repo that v2 cannot load, and they are what a reader greps into. | They are provenance for the cached outputs the run gate compares against. |
| C5 | Consolidate `import_outputs.py` and `cli/merge.py` | Two label-merging paths, each with its own `_snapshot`, both merging by name. | Modest duplication; they merge different *things* (a whole outputs dir vs. one label file). Lower value than the rest. |
| C6 | Do the `nmf_*` -> `model_*` rename (W9e) in this release rather than the next | 288 identifier occurrences over 34 names say `nmf` where the config, the stage and `IKResult.model_pts3d` say nothing of the kind. Deferring it means shipping a model-agnostic stage under a model-specific vocabulary. | It renames `results.h5` datasets, so it wants the v3 repack -- the same argument as C3, and the same answer: ride them together or do it lazily on the next write. |
| C7 | Cap the worker count (`--maxprocesses=8` in `addopts`) | `-n auto` is flat past 8 workers here: 23.7 s wall / 270 s CPU at `-n 8` against 21.4 s / 490 s at `-n 32`. The extra 24 workers buy 2.3 s of wall for 220 s of CPU, nearly all of it each worker re-importing torch, jax, scipy and fastapi to collect. | 2.3 s is 2.3 s, and CI runners have few enough cores that `auto` already caps itself. Pure preference -- it is here because it was measured, not because it is owed. |

## Relationship to the earlier design record

This retracts **item 3 of [project-system-plan.md](../project-system-plan.md)** -- "projects
from scratch ... optionally using dedicated non-skeleton calibration landmarks". The
project system, GUI-first operation and from-scratch rigs all stay; only the dedicated
landmarks go, and `design/README.md` now says so.
