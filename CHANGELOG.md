# Changelog

Notable changes to `deeperfly`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[semantic](https://semver.org/spec/v2.0.0.html).

**0.2.0 is the first release.** `0.1.0` was a version number in `pyproject.toml` and never
a git tag, so it has no entry of its own and 0.2.0 spans the whole distance from it. It is
deliberately not 1.0: the release is a real one, but 1.0 is a promise about stability and
this one breaks the skeleton name, the detector registry, three config keys and the public
API. A version that says so is more useful than one that flatters.

## [Unreleased]

### Changed

- **The default 2D detector now has a PADDED heatmap field.** `[pose2d].models` names
  `mvt_r28_pad48_gray_fly38.pth` (sha `ae482d3a…`, 86,082,205 bytes) in place of
  `mvt_alt8_r27_gray_fly38.pth`. Every earlier MVT was structurally unable to report a joint
  outside the frame — its decode is a soft-argmax, an expectation over in-image pixel
  coordinates, so a joint the crop cut off saturated against the border and came back at high
  confidence up to 200 px from the truth, straight into RANSAC and the bundle adjustment. The
  new checkpoint declares `arch.hm_margin_px = 48` and gets that margin by padding the
  network's *input* (352×608, field 88×152 at unchanged stride 4), so the extra cells are
  computed from real tokens rather than from zeros.

  **Nothing a config can see has changed.** `input_size` still reports the 256×512 reported
  frame; the margin lives inside `pose2d/mvt.py`, which pads in `prepare_images` and subtracts
  it again in `decode_points`. What is new is that a returned coordinate may legitimately fall
  outside `[0, 1]` — an off-frame joint, not an error, and it must not be clipped.
  `deeperfly.triangulation` already consumes such points.

  Costs and caveats, stated because they are real: the padded model runs at **24.4 frames/s
  against 46.3** for r27 on an RTX 4080 (1.9×), and its **accuracy has not yet been
  evaluated** — the mechanism is confirmed to fire (0.13% of detections on a full recording
  land outside the box, reaching 43 model px, where r27 produces exactly zero) and no in-frame
  regression was detected, but there is no measurement yet that off-frame points land in the
  *right* place. Pin `weights = "mvt_alt8_r27_gray_fly38.pth"` to stay on the previous
  detector.

- **The top-K candidate path (`pictorial_structures = true`) decodes through the model's own
  cell geometry.** `inference.detect_candidates_sequence` briefly *refused* a model with a
  padded field rather than decode it through a normalization that assumes the field spans the
  input; `LoadedModel.cells_to_normalized` replaces that refusal with a dispatch, so both
  shipped classes work. The shared `(c + 0.5) / W_field` convention is right only when the
  field spans the reported frame, which neither shipped class does — the dense HRNet pads its
  head's output by 25% a side, the MVT pads the network's input by 48 model px — so it was off
  by both the margin and the `(w + 2m) / w` scale, ~46 model px at the frame's edge on r28.
  That is three times the 15 px a candidate may sit from its hypothesis
  (`pictorial.DEFAULT_INLIER_PX`), so every edge candidate was silently discarded. This path
  was already wrong this way for the dense HRNet before r28 existed.

  The same path's *arg-max* also now comes from the model's own readout rather than the shared
  decode, so it matches what `detect_sequence` writes. For the MVT the two genuinely differ —
  its readout is a global soft-argmax over the 16×-upsampled field where the shared decode is a
  windowed centroid over the raw cells — so enabling `pictorial_structures` used to move every
  point before recovery had considered anything. Both come from **one** forward
  (`predict_points_and_heatmaps`) rather than two.

- `pose2d.models.LoadedModel.prepare` passes a margin through to a class that declares one, and
  raises if the class cannot accept it — preparing a padded checkpoint without its pad would
  shift every point by the margin with nothing to notice.

- **`pose2d/autocrop.py` now applies its exact clipping test to the MVT.** `LoadedModel.padded_field`
  is what picks the test, and it is now true for the packaged detector, so "does this crop cut the
  animal?" is answered by counting detections outside `[0, 1]` rather than by counting ones within
  3 px of the border (`BORDER_BAND_PX`). The band was a stand-in for a field that could not
  represent an off-frame joint; a searched crop (`{op = "crop", auto = true}`) may therefore pick a
  different box than it did on r27.

- **The packaged `[eks].inflate_threshold` is `30.0`**, up from `15.0` (the dataclass default is
  still `5.0`). Reprojection cannot choose this number — the smoother is *supposed* to leave the 2D
  where it judges the 2D unreliable, and a 5→30 sweep barely moves the reprojection distribution —
  so it was chosen against ground truth instead: on 54 recordings and 63,921 labeled cells, mean 2D
  error falls 4.916 → 4.628 px and p90 9.103 → 8.343 going from 5 to 30, while frame-to-frame jumps
  grow monotonically. Accuracy is flat from 30 to 100, so 30 is where the gain is realized at the
  smallest temporal cost. Turning the inflation *off* is worse than any threshold. The full table is
  in the configuration reference's `[eks]` section. The five `examples/*/config.toml`
  move with it.

### Added

- **A confidence floor, off unless the artifact asks for it.** `mvt.predict_points` reads
  `arch.conf_floor` and reports a point below it as `NaN` — which is what `deeperfly.triangulation`
  already means by "this camera cannot see this point" — instead of as a confident location pinned
  to the border. It is meaningful only for a checkpoint trained to answer "not here" with a flat
  map, where the confidence separates the two answers by ~500×; on one that was not, confidence does
  not track off-frame-ness and a floor would drop good points. The packaged `mvt_r28_pad48` ships
  `conf_floor = 0.0`, i.e. the gate is **off**.

- `load_mvt` refuses an artifact whose `arch.hm_margin_px` is negative, is not a whole number of
  patches (a fractional token grid is truncated by the backbone, shifting every point), or whose
  `arch.hm_margin_fill` disagrees with what this module pads with — the fill is a train/test
  contract, not a cosmetic choice.

### Fixed

- `results.py`'s schema docstring described `eks/posterior_var` as `(T, P)`. It is `(T, P, 3)` —
  the smoother's variance is per axis.

The one thing the tree names as planned is the **trainer**: `deeperfly.training`
was removed here (see below) and training will return written directly rather than absorbed.
The multiview transformer is trained today through a patched Lightning Pose fork pinned to
python 3.12 in a 7.5 GB environment of its own, which is why it is not an extra — behind one,
`nvidia-dali-cuda110` (linux-x86_64 only) would have to resolve in metadata that also
installs on macOS.

## [0.2.0] - 2026-08-19

### Upgrading from 0.1.0

Ten things a 0.1.0 tree can trip over. Most configs need one edit or none.

1. **`[skeleton]`, if it spells out `point_names`: nothing to do.** A self-contained skeleton
   table is left exactly as written, and `name` goes on being a free-text label for it — so a
   0.1.0 config keeps its own point order and means what it always meant. Only a table that
   *references* a skeleton resolves a packaged one, which 0.1.0 had no way to write.
   `name = "fly38b"` (written by any dev build) still resolves, with an INFO line saying so.
   A reference reading `name = "fly38"` now gets the **midline-abdomen** set, not the
   DeepFly3D one — spell the old points out, or point `file` at `tests/data/fly38_deepfly3d.toml`.
2. **`class = "hourglass"` / `"deepfly2d"` is refused.** Two classes remain, both dense:
   `"mvt"` and `"hrnet"`. Selecting one is two lines and one of them is the filename:

   ```toml
   [pose2d]
   model = "dense38mv"
   models = [{ name = "dense38mv", class = "mvt", weights = "mvt_alt8_r27_gray_fly38.pth" }]
   ```

   Then delete every `[pose2d.output_points]` row — a dense plan needs no mapping (channel *i*
   is point *i* of the pathway's view), and each view needs one pathway rather than a pathway
   and a mirrored twin. **Nothing downloads.** Set `$DEEPERFLY_MODELS` to the directory holding
   the checkpoints (several, `os.pathsep`-separated, like `PATH`), name the weights as a bare
   filename, and run `deeperfly doctor` — it prints the variable, each search directory with
   what is in it, and whether the default config's checkpoint resolves. A dev-build config
   naming the RGB `mvt_alt8_fly38b.pth` must be repointed; that artifact no longer loads.
3. **`[cameras.<name>].preprocess` is a hard error**, and the message names its replacement.
   It had been parsed, validated and thrown away for several releases, so a config cropping a
   view this way was already detecting on the uncropped frame. Move the ops to the detection
   pathway, where they are inverted on the way back:

   ```toml
   [pose2d]
   preprocessors = [{ name = "crop_f", ops = [{ op = "crop", x = 400, y = 290, width = 800, height = 400 }] }]
   pathways = [{ name = "f", source = "vid_f", preprocessor = "crop_f" }]
   ```

   …or `auto = true` in place of the box to have one measured per recording.
4. **`[inverse_kinematics].constant_points` is removed.** Move its points into the
   `[postprocess]` chain, which is on by default now:
   `[[postprocess.ops]]` with `op = "static"`, `method = "median"`, `points = [...]`. The old
   key was the solver's private pin — it collapsed the listed points for the fit alone, so the
   stored angles and the stored 3D disagreed about exactly those points, silently. The op
   applies the correction to the *result*, in 2D and 3D, and logs how far it moved what it
   touched.
5. **`[inverse_kinematics]` scipy knobs are rejected, not ignored**: `max_nfev`, `loss`,
   `f_scale` and `regularization` tuned a solver that no longer exists. The QuickIK knobs are
   `n_iterations`, `neutral_weight`, `damping`, `position_tolerance`, `angle_tolerance`,
   `fixed_body`, `weigh_by_confidence`, `parallel`, `segment_len`, `overlap_len`.
6. **Three more stages now run by default** — `eks`, `postprocess`, `inverse_kinematics` — and
   `symmetric_segments` with them. To decline one, `deeperfly config set pipeline.do_eks false`
   now works on any config that already states the key. A 0.1.0 `[pipeline]` table does not
   state these three, and appending a bare key after an existing header would reparent the keys
   below it, so `config set` refuses and tells you to add the line by hand.
7. **Public API removals** (each has an entry under *Removed*): `deeperfly.load_detector`,
   `deeperfly.pose2d.model`, `deeperfly.pose2d.weights`, `deeperfly.training`,
   `deeperfly.gui.corrections`, `deeperfly.inverse_kinematics.core` / `.kinematics`,
   `Config.frame_transforms`, `preprocessing.parse_frame_transforms`,
   `FrameTransform.map_intrinsics` and `Camera.from_spec(transform=…)`.
   `solve_inverse_kinematics` keeps its name and changes its keywords.
8. **Caches.** The skeleton's *name* left every stage fingerprint (the ordered `point_names`
   and the `bones` are what decide a stage's answer), so the rename alone re-detects nothing.
   Two things do recompute on purpose, and both matter to anyone who has been tracking dev
   builds rather than to a 0.1.0 tree, which had no IK stage at all: any stored **leg or
   abdomen angle**, because `IK_SOLVER_REVISION` went 1 → 2 for a convention change that used
   to slip through the digest; and a fit whose **marker placement** changed, now that
   `[inverse_kinematics.head]` / `[inverse_kinematics.abdomen]` retargets digest. A run also refuses an output directory
   whose stored pose is on a different point set, naming the points each side has that the
   other does not, and calling out a pure reordering explicitly.
9. **`results.h5` from 0.1.0 (schema v2) still reads.** New writes are v3; `deeperfly repack`
   converts an existing file without recomputing anything.
10. **`corrections.h5` has no reader any more.** The editor stopped writing one in the rewrite
    and the migration-on-open is gone, so anything still held in that format has to be read
    with an older checkout.

### Added

**Detectors, both dense** — every tracked point in every view, so a contralateral point
arrives as a prediction to correct rather than a gap to author from nothing (measured
visibility 1.000 against the retired plan's ~0.5).

- `class = "hrnet"`, the per-view detector, which also runs the HGNetV2-B4 checkpoint.
- `class = "mvt"`, the multiview transformer: the views of a frame are encoded together, so a
  joint only one camera can see informs the cameras that cannot. Verified against the
  Lightning Pose route on a real recording at 0.0000 px in footage pixels, and permutation-
  equivariant with its views to 4.8e-05 of output range.
- Three checkpoints ship, one generation, trained on 55 recordings / 465 moments / 138,708
  label cells, each recording the `fly38` point order and taking one grayscale plane:

  | file | class | sha256 | bytes |
  | --- | --- | --- | --- |
  | `mvt_alt8_r27_gray_fly38.pth` | `mvt` | `d4ca455b…` | 86,082,077 |
  | `hrnet_w32_r27_gray_fly38.pth` | `hrnet` | `13ceb937…` | 127,076,045 |
  | `hgnetv2_b4_r27_gray_fly38.pth` | `hrnet` | `fa427062…` | 62,452,371 |

  They live at `/mnt/upramdya/data/TL/deeperfly-models/260819_*`, each a directory carrying
  the `.pth`, a README, `SHA256SUMS` and `fly38.toml`. The packaged default names the MVT.
- `{ op = "crop", auto = true }`: the detector's box, measured per recording. Fed a whole
  1600×1008 axial frame the fly lands at a scale no detector saw — 255 px of error, a collapse
  — and the search reaches 2.9 px, better than the 3.4 px a person placed by hand. The
  **search** is a grid over (center, width) scored by detector confidence and needs no rig;
  what needs a solved rig is the **accept gate**, which is reprojection agreement with the
  other cameras' 3D. Against a nominal orbit the gate measures its own reference ~250 px out,
  refuses rather than choose with a broken ruler, and falls back to confidence alone — a box
  ~1.7× too wide, still far better than the whole frame. Export a calibration, point
  `[cameras].calibration` at it, and the gate engages. `deeperfly auto-crop` runs the search
  alone and prints the TOML that freezes it.

**Two stages after triangulation**, both on by default.

- `[eks]` — the nonlinear multi-view ensemble Kalman smoother of Lightning Pose 3D, fitting one
  3D trajectory per keypoint to the whole recording through the rig's own projection. Median
  frame-to-frame acceleration drops **39%** (0.0062 → 0.0038), at the cost of lagging a claw
  mid-swing on 0.13% of cells. Variance inflation — the component that repairs a blown
  detection — needs two *views* rather than two models, so it is fully active with a single
  detector. Omitting `smooth_param` fits it per keypoint by maximum marginal likelihood, which
  is the right default: a fly's claw and its thorax do not move alike.
- `[postprocess]` — an ordered chain of corrections that come from knowing the animal rather
  than the pixels, applied where a reader can see them and each logging how far it moved what
  it touched. `op = "static"` (five centers: `median`, `mean`, `trimmed_mean`, `mode`,
  `geometric_median`) and `op = "symmetrize"`. The packaged config freezes the neck and the six
  thorax-coxae and symmetrizes the three body-fixed pairs — never the legs, whose left/right
  asymmetry at any instant *is* the behavior.

**Inverse kinematics**, now a default stage.

- QuickIK (Rust, PyO3) replaces the per-limb scipy/JAX fit: one solve for the whole body — six
  legs, head and abdomen against every tracked keypoint at once, warm-started frame to frame —
  at 0.5 ms/frame. Optional extra `deeperfly[ik]`, because quickik publishes no wheels; the
  stage now **skips with a logged reason** when it is absent instead of raising, which also
  costs the videos that come after it.
- Articulated head and abdomen chains, retargeted onto the packaged skeleton's five dorsal
  midline stripes: a plan covers 38 of 38 points and the abdomen fits in 100% of frames (it
  returned NaN for every DOF before). Each chain's size is measured with a ruler its own
  posture cannot move.
- `symmetric_segments = true` (default): each mirror pair of leg segments takes the mean of the
  two data-measured lengths, derived from the skeleton's declared `symmetries` rather than from
  an `l`/`r` prefix. It constrains the animal, not its pose. The cost is measured and real —
  3D residual 0.0120 → 0.0146 (+22%), reprojection 4.45 → 4.59 px, worse in 7 of 8 views — and
  what it buys is one animal with angles comparable across sides instead of two half-animals,
  plus the knowledge that a 5% femur difference which is not anatomy is a detector bias
  otherwise reported as biology.
- The posed NeuroMechFly mesh overlays the 2D views in rendered videos and in the editor,
  with a headless moderngl rasterizer (~10× the numpy/OpenCV painter's-order path, exact
  depth-buffer occlusion) and a software fallback where there is no GL context.

**A run narrows itself to the footage present** instead of refusing the recording. A source
with no footage invalidates the pathways reading it; a view no surviving pathway feeds leaves
the rig, so `V` shortens; grid cells naming it are blanked, an orphaned `auto = true`
preprocessor and its `[pose2d.output_points]` rows are dropped; one warning names the source,
the pathways and the views. `Config.narrowed_to_sources` does it in `run_recording` **before**
the snapshot and the fingerprints, so a seven-view run records a seven-view fingerprint and
recomputes when the eighth camera turns up, while the snapshot still describes the rig that was
configured. Below `config.MIN_VIEWS_FOR_3D` (two) it refuses, because one view fails silently
everywhere — RANSAC gives a lone observation zero inliers and then erases it, and bundle
adjustment reports success at a cost near 1e-26.

**The annotation editor, the project layer and the from-scratch rig.** `deeperfly gui` is a
ground-truth annotation tool: 2D is the source, 3D is derived live from it. Labels live in a
sparse `labels.h5` sidecar (schema v8) whose unit is an *instance* — one skeleton, created in a
frame, owning a position for every (view, point) — with `hidden` narrowed to a pure training-loss
switch, orthogonal to `gt`, and `absent` for a keypoint that is not on this animal. Around it:

- `deeperfly project` — group recordings under one skeleton and one rig **without moving a
  label**: `results.h5` and `labels.h5` stay where they are, adopted by symlink. Plus
  `project config` / `project rig` (compose a run config from fragments — the rig belongs to the
  setup, so changing one triangulation knob no longer means copying a whole config per
  recording), `project export|import` (one shareable `.dfpkg`), `project skeleton` (a
  skeleton edit as a typed migration over every label), `project import-outputs`.
- `deeperfly calibrate` — solve a rig from hand labels and calibration landmarks, with a
  readiness meter (`--dry-run`) that phrases every shortfall as the labeling that would fix it.
  Validated cold against a known rig: recovered to 0.01% of its radius from landmarks alone,
  0.02% from keypoints alone, 0.00% from both. `calibration.toml` is a portable solved rig
  (`deeperfly calibration show|export`), and `[cameras].calibration` wins over the orbit.
- In the editor: a Landmarks tab that places calibration points by hand, a Bundle adjust tab
  that solves the rig from the labels in the session and writes it as a new calibration, a Jobs
  panel that runs pipeline commands as isolated serialized subprocesses, and recording tabs
  that switch in place (`deeperfly gui --recording` is retired; the picker answers it live).
- `deeperfly labels-suggest` (a ranked queue of which frames to label next),
  `deeperfly labels-merge` (reconcile label sets that ended up in two places),
  `deeperfly labels-export`, `deeperfly labels-absent`.
- `[skeleton].symmetries` — left/right mirror pairs, with `flip_perm()`, a chirality check that
  flags hand labels whose left/right identities look swapped, and a refusal for a mirrored
  pathway that lands on the wrong body side.

**The eight-camera rig is the packaged default.** `[cameras.h]`, the axial hind camera, is no
longer a commented-out example: it is the rig's only left/right **bridge** — without it no
camera sees both body sides, so a contralateral joint is triangulated from one side's cameras
alone — and it is what the shipped detector was trained on (the MVT checkpoint records
`provenance/view_names_at_training = [rh, rm, rf, f, lf, lm, lh, h]`). Its numbers are not the side
cameras': that view is on a different lens, so **both** its focal length and its distance
differ, and no residual can detect it afterwards — bundle adjustment holds the intrinsics fixed
and absorbs a focal error into the distance while reporting a clean solve. The shipped values
are the example rig's; measure them per rig. A seven-camera recording needs no edit (see the
narrowing above).

**Config discovery and visualization.**

- `deeperfly config show` / `deeperfly config set` — every key, its default and its
  documentation derived from the code, so they cannot drift from it, and one key settable
  without reading the other four hundred lines.
- `deeperfly doctor` reports the weights search path (see the upgrade note), and `deeperfly
  repack` rewrites result files in the current schema.
- Videos: `grid` layouts, `stage = "…"` to pin which stage's points a panel draws (unset means
  the most-derived stage present, so a video's meaning otherwise changes when a later stage is
  enabled), `crop = [x, y, w, h]` or `crop = "pose2d"` to frame each panel through the window
  its own view detected through, `view = "bird"` (a synthetic dorsal plan view fitted to the
  animal's own body axes — the one viewpoint showing all six legs with no body in the way),
  `line_dash` so a before/after pair reads as two skeletons rather than one thick one, and
  joints drawn as a full-opacity ring whose fill opacity is the detector's confidence.
- `[annotation]` — how the editor turns 2D labels into a live 3D estimate (`solve_policy`,
  `min_gt_for_exclusive`, `gt_weight`, …).

### Changed

- **`fly38` now names the midline-abdomen 38-point set** — six 5-point legs, two antennae,
  `neck` and `abdomen0..4` — and is the only packaged skeleton. `name = "fly38b"` resolves to
  it (`config.SKELETON_ALIASES`), so every config and output snapshot written before the rename
  goes on loading. The historical DeepFly3D set that `fly38` named before (two 3-marker abdomen
  *side* chains, no neck) is retired to `tests/data/fly38_deepfly3d.toml`. The two are not
  interchangeable: both are 38 points and they share 32 of them in a different order, every
  right-side point shifted by three.
- **One input plane.** The corpus is monochrome, so every shipped detector takes a single
  channel and the three it used to be handed were three copies of one image.
  `LoadedModel.prepare` emits `(…, 1, H, W)`, HRNet's `in_chans` is 1 unconditionally, and
  `mvt.ARTIFACT_FORMATS` is `("deeperfly-mvt-2",)`. No accuracy is lost: a `-2` artifact is the
  same *function* as the `-1` it was folded from, since the stem fold sums filters that were
  three copies of each other.
- **Every stage on by default except `pictorial_structures`**, in both places that decide it
  (`config.STAGE_DEFAULTS` and the packaged `[pipeline]`). `pictorial_structures` stays off and
  not for symmetry: it recovers a joint from the top-K peaks of a one-body-side detector, and
  switching it on rewires triangulation *and* the smoother onto its committed 2D — NaN in every
  view with no candidate within 15 px — so it would silently un-densify a dense run.
- **A dense plan needs no `[pose2d.output_points]`.** Channel *i* → point *i* of the pathway's
  view is the default, `n_out_channels` resolves to the skeleton's point count, and
  `input_size` plus the normalization come from the checkpoint, whose loader refuses a config
  that disagrees. A generated `[pose2d]` section for a 7-camera rig went from 343 lines to 63;
  what was deleted carried no information, and a single transposition in it was a wrong limb
  rather than a crash. Verified as the same mapping unwritten and not a different one:
  bit-identical 2D. The mapping table still exists and is still supported (it is how one view
  can be fed by several pathways) and is required for anything not dense; the shipped configs
  declare none.
- **An unknown detector class is refused** by `class_defaults`, naming the classes the build
  has, instead of inheriting DeepFly2D's 19 channels and `mean = 0.22` and failing later on a
  channel-count mismatch that says nothing about the word that was wrong. `DEFAULT_MEAN` and
  `DEFAULT_N_OUT_CHANNELS` go with it. **A checkpoint recording no channel names is refused**
  too: that check is the one thing standing between a mis-stamped config and a fly with its
  limbs on the wrong joints, and a count check cannot substitute.
- **A calibration covering only some of the config's cameras subsets, with a warning**, rather
  than refusing: a calibration is a measurement of which cameras exist, so an unmeasured view
  is dropped like a view with no footage. Covering *none* still refuses — that is the wrong-rig
  case, which a subset cannot explain away. `deeperfly calibrate --from-calibration` stays
  strict, because that is where the intrinsics to solve *with* come from.
- **`results.h5` schema v3**: a stage stores only what cannot be rebuilt. The smoother's 2D
  *is* the projection of its 3D, the correction chain's differs on columns it holds constant
  over time, and every reprojection error is recomputable — so the classification is measured
  per write (`derived` / `override` / `full`) rather than tabulated per stage. Point arrays are
  float32, which resolves 6.1e-05 px against the ~1 px a detector can localize. An eight-view
  2007-frame recording went 66 MB → **15.6 MB (4.23×)**, handing back the same pose to
  1.7e-05 px. Triangulation keeps its `reproj_error` on purpose: it is the only witness when an
  outside tool overwrites that stage's 2D, worth 6.4 MB not saved.
- **Triangulation defaults to RANSAC** (`t15`, `min_inliers = 2`, unweighted). Each leg joint is
  triangulated from only its 2–3 ipsilateral views, and on an occluded joint the 2D detector
  emits confident wrong guesses that unweighted DLT averages in.
- **Forward precision defaults to `float16`** (was bfloat16): identical keypoint accuracy to
  fp32 (<0.02 px) with higher inference throughput on CUDA, and `[[pose2d.models]].precision`
  overrides it per model.
- **Faster, at the same numbers.** Decode and compositing run concurrently: on the 7-camera
  4009-frame example, pose2d 40.7 → 107 fps, warm render 110.7 → 16.7 s, full run 313 → 105 s
  (3.0×), bit-identical output. Monochrome full-range footage is read as its luma plane, a
  memcpy against swscale's YUV→RGB — which is ~90% of what a frame costs — measured 100–300×;
  TV-range footage still converts, because there the plane is a different set of numbers.
  The next window is prepared on a worker thread while the GPU still has this one, and a
  window's pathways prepare concurrently. In the editor a held-open `FrameCursor` takes a step
  forward from 2224 ms to 5 ms and a jump from 2664 ms to 195 ms (the GOP walk itself).
- The test suite runs across processes with each worker's thread pools capped to one thread:
  115 s serial → 39 s. `-n auto` alone is worthless and measurably so (120 s), because N
  workers each ask torch for one intra-op thread per core.
- The three pre-implementation plan documents moved from `docs/` to `design/`. MkDocs publishes
  every `.md` beneath `docs/` whether or not it is in the nav, and an unlisted page is an INFO
  rather than a warning, so all three had been live on the public site — superseded designs
  read as documentation.

### Removed

- **The stacked-hourglass / DeepFly2D detector**: `class = "hourglass"` / `"deepfly2d"`,
  `deeperfly.pose2d.model` (HourglassNet), `deeperfly.pose2d.weights`, the public
  `deeperfly.load_detector`, and the sh8 **auto-download** — with it, any auto-download at all.
  It was already off every path that matters (the packaged config and all six examples are
  dense) and was the last thing keeping several DeepFly2D-shaped assumptions alive in code the
  dense detectors run through. `pose2d/detector.py` survives, trimmed to what every class
  shares and no class owns; the device, input-coercion and precision plumbing moved to
  `pose2d/runtime.py`.
- **`deeperfly dense-config` and `deeperfly.pose2d.dense_plan`.** The command existed to write
  a table that no longer exists — 122 mapping rows for a 7-view rig, 304 for an 8-view one. Its
  one remaining job, pointing a config at a newly trained checkpoint, is now editing `weights`;
  the check behind it (a checkpoint's ordered point names against the config's skeleton) did
  not go with it and runs on every run.
- **`deeperfly.training`** — `mirror_decisions`, `mirror_sample`, `mirror_view_names` and the
  heatmap-target helpers: a trainer's numeric contract with no trainer, 520 lines with zero
  importers anywhere in the package, whose own documentation named a `deeperfly[train]` extra
  that `pyproject.toml` has never declared.
  `[cameras.<name>].mirror` and `Config.mirror_views()` stay — the key is a fact about how the
  rig was built.
- **`deeperfly.gui.corrections`** (`Corrections`, `load_corrections`, `save_corrections`,
  `migrate_from_corrections`) and with it the dense `corrections.h5` format. The editor stopped
  writing one in the rewrite; a `$HOME` sweep found no file left to read.
- **`[inverse_kinematics].constant_points`** (item 4 above) and the scipy-era solver knobs
  (item 5), plus `deeperfly.inverse_kinematics.core` / `.kinematics` — 502 lines of numerics —
  and the legacy antenna-vector head method with its `c_head-<side>_pedicel-{pitch,yaw}` angles.
- **`[cameras.<name>].preprocess`** (item 3), along with the dead machinery behind it:
  `Config.frame_transforms`, `preprocessing.parse_frame_transforms`,
  `FrameTransform.map_intrinsics` and `Camera.from_spec(transform=…)`. `map_intrinsics` — the
  function that would have made the key correct — was reachable only from the dead path, so
  that intrinsics-shifting capability has never run in production. Worth knowing before anyone
  tries to revive the key rather than refuse it: a pathway's ops are inverted on the way back,
  so a detection meets its camera in raw footage pixels; the retired key moved the camera into
  cropped-pixel space instead, and honoring both would double-correct by exactly the crop
  offset.
- **The `deeperfly-mvt-1` artifact format** (three channels), and the packaged DeepFly3D
  skeleton preset (moved to `tests/data/`, see *Changed*).

### Fixed

- **The heatmap decode was biased half a cell toward the origin.** It divided the arg-max cell
  index by the heatmap size, a top-left-corner convention, against training targets that peak
  at `floor(keypoint * heatmap_size)` — about **3.75 px up-and-left** on this rig's 960×480
  frames, on every decoded point, and not recoverable by sub-pixel refinement.
- **The leg chains are parameterized in flygym's own frame.** The template had two DOF axes
  swapped and the plan rotated each leg subtree by the wrong body frame (the coxa-centroid
  frame, pitched ~23° away), which left the fit unable to reproduce NeuroMechFly's own resting
  posture within NeuroMechFly's own joint limits. Fitting the model's neutral keypoints, the
  residual goes 0.0340 → 0.0011 and the angles land 1.02° from the model's spring reference
  (was 32.0°). On 287 frames of the example recording: all 24 leg DOFs 0.0334 → 0.0198
  (**−41%**), reprojection 4.68 → 2.89 px, worst pinned DOF 71.1% → 0.7% of frames. The
  residual now sits at the rigid-segment-length floor. **Breaking:** every stored leg angle
  changes convention, which is what `IK_SOLVER_REVISION = 2` forces to recompute.
- **The abdomen markers ride the segment behind the fold.** `abdomen3` and `abdomen4` used to
  share a body, so no joint lay between them and the model held them 18% short of what the
  animal measured. Re-anchoring one body proximal: `abdomen3` −72%, `abdomen4` −60%, the
  five-marker chain 0.0170 → 0.0098 (−42%), legs unchanged as a control. **Breaking:** the
  lateral DOF is flygym's `-roll`, not its `-yaw` — five stored angles and five
  `[inverse_kinematics.bounds]` keys are renamed. The twist stays unfitted because it *aliases*
  the lateral bend (displacement fields agreeing to cos 0.87–0.97), not because its lever arm
  vanishes.
- **The committed `nmf_mesh.npz` was the old skeleton's asset**, so the mesh overlay skinned the
  right-side legs between the wrong keypoints and registered the body on three wrong points.
  Both asset generators now reproduce their output, and the viewer table, the keypoints JSON
  and the IK articulation are checked to agree.
- **Grayscale input was not actually in effect for any `hrnet`/`hgnet` plan.** The gray decode
  requires every model in a plan to declare `accepts_gray`, and `_HRNetPose` did not — so those
  two paid the full YUV→RGB conversion, ~90% of what a frame costs, to feed a network that
  immediately took one channel back.
- **A retarget could validate a stale fit.** The IK fingerprint recorded per-DOF bounds and
  nothing else, so retargeting a chain with `[inverse_kinematics.head]` /
  `[inverse_kinematics.abdomen]` reused the previous fit while the config on disk described a
  different one. Marker placement now digests, as resolved neutral positions, depths and base
  point rather than as the config table, so a retarget expressed any of the three ways it can be
  is one comparison.
- **`deeperfly run` resolved a recording's footage against the discovery config**, not against
  the config the run would use. The packaged default was a seven-camera rig, so an eight-camera
  recording came back missing its hind view — and nothing noticed until a frame was wanted,
  which with a cached `pose2d` meant dying in visualization with advice to do what had just
  been done.
- Three foundation bugs under the narrowing: `source_sources` was all-or-nothing, so one absent
  key discarded the whole map and a seven-of-eight recording resolved to *zero* footage;
  `source_image_sizes` opened every source unconditionally, and opening an empty one raises;
  `find_recording` returned `None` for a partial directory, which is the gate that runs first.
- **`deeperfly config set pipeline.do_<stage>`** never worked: it refused any file already
  declaring the table — correctly, because appending a bare key after an existing header
  reparents what follows — and the packaged config states every `[pipeline]` flag, so that
  refusal was the whole surface. An existing key is now rewritten in place; a genuinely new key
  under an existing table is still refused, with the reason. Its validation also looked up a
  `Config.pipeline` attribute that does not exist.
- **Feature maps are selected by stride, not by index.** `out_indices=(1,2,3,4)` is a fact
  about HRNet (whose index 0 is a stride-2 stem), and on a ResNet or an HGNet the same tuple
  hands the head strides 8/16/32/32 — a model that builds, runs and is quietly wrong. The head
  type and keypoint count are likewise read from the checkpoint rather than defaulted; a `unet`
  checkpoint rebuilt as `concat` used to lose every arm to a `load_state_dict` mismatch that a
  caller saw as an arm with no rows.
- `assemble_result` read the IK arrays but never the meta, so a run's videos drew the fit at
  `body_scale = 1.0` with model-size head and abdomen while the editor on the same file used
  the fitted ones.
- Fragment extraction read any continuation line of a multi-line array beginning with `[` as a
  new table header, so `extract_section` cut a config off *inside* an array and produced a
  fragment that did not parse at all.
- `Project.create` defaulted to `fly38b` while `deeperfly project new --skeleton` defaulted to
  `fly38` — the CLI seeded the old point set and the library the new one.
