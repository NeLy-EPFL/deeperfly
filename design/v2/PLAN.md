# deeperfly v2 -- implementation plan

Read [README.md](README.md) for what v2 *is* -- the workstreams, the removals and the
evidence for each; [config-schema.md](config-schema.md) for the schema itself and its
before/after tables. [default_config.toml](default_config.toml) and
[example_scape_E49.toml](example_scape_E49.toml) are the target files. This document is
only *how* it lands.

## Decisions taken

- **Skeleton**: OUT of the run config. It is four things (points, edges, symmetries,
  colors) in a version-controlled file, resolved from the detector's recorded skeleton
  name; `[skeleton] include = "<name or path>"` in a config overrides that, per key. No
  chains, no groups: what a group name was for is now a `*` point pattern (W5c).
- **Footage**: `[cameras.<name>].video` is a regex (`re.fullmatch` on the filename,
  case-insensitive), or a list of them concatenated in order. Nothing is inferred from
  the camera name or its index. Everything one pattern matches is one stream, in
  natural order, so a split recording and an image sequence are the same rule; the
  matches must be parts of one series (identical with digit runs masked), which is what
  keeps `camera_(RH|0)` from concatenating two naming schemes in a directory holding
  both. Alternates therefore live inside the regex, and the list means concatenation --
  the one v2 key whose *v1 shape still parses but now means something else*, so it gets
  an explicit check (below) rather than the generic v1 error.
- **Landmarks**: the whole calibration-landmark namespace is REMOVED, GUI included. The
  animal is the calibration target -- which is already what every rig here was solved
  from, since no `landmarks.toml` and no `landmarks/` group exists anywhere. This retracts
  item 3 of `design/project-system-plan.md`.
- **Scale**: physical units are not a calibration-stage concern. Everything through
  triangulation is arbitrary units by design; scale first becomes physical at inverse
  kinematics, where `body_scale` fits the point cloud to the NeuroMechFly model's defined
  dimensions. So calibration-stage scale pinning goes with the landmarks, and `units` is
  `"arbitrary"` or `"config"`.
- **Model**: the mechanical model the IK stage fits stops being wired into Python. A
  `[inverse_kinematics] model` names a *pack* (leg template + articulation + overlay
  mesh) resolved by name or path, the way `[skeleton] include` resolves a skeleton. The
  config does not grow -- `template` is renamed and widened, `fit_head`/`fit_abdomen`
  collapse into `chains`, so the table loses a key. **Where each tracked point sits on the
  model is a third artifact**, a `bindings/<skeleton>@<model>.toml` resolved from the pair,
  so a pack names no skeleton point and a skeleton names no model body -- a greppable
  invariant. That binding exists today as `docs/keypoints/assets/keypoints.json`; W9 gives
  it a home, and gives the IK all 38 of its rows instead of 7. Authoring the flybody pack
  is explicitly outside the release (W9g). See [ik-model-packs.md](ik-model-packs.md).
- **Postprocess**: the stage STAYS -- before the IK, dependency-free, on by default,
  both ops (`static`, `symmetrize`) intact. It looks redundant, because the IK's
  `body_alignment` already takes the same medians, and it is not: the two produce the
  same *number* for different *destinations*. The stage writes a corrected `points3d`;
  the alignment bakes an `offset_pos` into the body plan. A pose and a model parameter
  are not the same output, and only the pose exists on an install without the Rust
  solver (`require_quickik` gates the solve; `align.py` is pure numpy, so the alignment
  could run without it, but its output is six numbers per leg, not a pose).
- **The two agree by ordering, not by merging.** The alignment's median over an
  already-constant point returns that constant exactly, and `_body_axes` builds `y` from
  the left-minus-right coxa centroids and `x` from front-minus-hind, so mirror-symmetric
  origins *induce* a frame whose x-z plane IS the stage's midline plane -- not a
  competing estimate of it that overwrites it. No coordination between the two is
  needed, and the recomputation must not be "deduplicated" later.
- **`symmetric_segments` is a different axis, not the other half of `symmetrize`.** It
  symmetrizes LENGTHS (a model parameter, consumed by the plan); the op moves POSITIONS
  (a pose). Different layers, different consumers, so they stay two knobs. Rigid-bone
  rescaling -- forcing every frame's femur to the median length -- would be a correction
  by the same test and is deliberately NOT an op: it rewrites *moving* keypoints, where
  both ops here touch only the seven body-fixed ones, and it is only ever right if the
  median length is. The `postprocess` -> `corrections` rename is DEFERRED: the code
  already calls these corrections in 57 places of prose, but the `results.h5` group
  would have to follow, which means a v4 plus a read alias for the ~351 unrepacked v3
  files. Not worth a name.
- **Compatibility**: hard break, no migrator. A v1 config fails to load with an error
  naming the v2 table that replaced the offending one (`[[sources]]` ->
  `[cameras.<name>].video`, `[pose2d.output_points]` -> gone, ...). Output-dir snapshots
  written by v1 fail the same way; the fix is to re-point `-c` at a v2 config.

## Strategy: new surface, same domain objects

`Config` keeps handing out exactly what it hands out today -- a `DetectionPlan`, a
`CameraGroup`, a `Skeleton`, a `list[VideoSpec]`, and the frozen `*Params` dataclasses.
Everything downstream of those (`pipeline/`, `results.py`, the GUI, the fingerprints)
therefore needs no change at all. The detection plan is *synthesized* from the camera
table (one source, one preprocessor, one identity-mapped pathway per camera), and
`layers` expand into the same `Panel` list `_expand_grid` already produces.

Two consequences worth stating up front:

- Fingerprints keep their shape but change their *values* (pathway and preprocessor
  names are now camera names). A first v2 run recomputes from `pose2d` down. That is
  correct, not a regression, and it is why the autocrop sidecar version is bumped: its
  keys are preprocessor names, which are now camera names.
- `*Params` defaults, and every measured number in their docstrings, are untouched.
  This is a surface change; no stage's behavior moves.

## Workstreams

Each numbered item below is one commit, ends green on the tests named, and leaves the
tree runnable. W8a flips the packaged default and every example, so between W3 and W8a the
suite is expected red on config *fixtures*; that churn is listed in the item that causes it.

**W4 and W7 depend on no *schema* work here** and can land first, last, or in parallel --
W4 because footage discovery is orthogonal to the schema, W7 because it removes a feature
the schema never described. Both W1 and W7 edit the GUI's JS, so landing them adjacently
keeps the rebase cheap -- and both want **W0a** ahead of them, since the browser test is
the only gate over that JS and it does not currently run.

### W0. Test-suite prerequisites

Not a schema change, but W1 and W7 both rest on a gate that does not currently exist, so
this lands first. The numbers below are measured on the 32-core dev box, `-n auto`, and
they are the reason the rest of this section is short: **the suite is 1728 tests in 21.4 s
wall, spread over 86.9 s of serial `call` time with a 4.25 s maximum.** There is no hot
spot to cut and no session-scoped-fixture win to find -- `setup` is 4.0 s in total. The
cost that is worth anything is startup: 5.5 s of the 21.4 s (and 93 of the 490 CPU-seconds)
is spent importing before the first test runs, once per worker, and parallelism is flat
past `-n 8` (23.7 s / 270 s CPU there against 21.4 s / 490 s at `-n 32`).

#### W0a. `tests/test_gui_browser.py` runs in no environment that exists

`playwright` appears nowhere in `pyproject.toml`, `uv.lock`, or either CI job. Locally the
module-level `importorskip` collapses all 53 tests into a single reported skip, which is
why this has gone unnoticed. **1859 lines and 53 tests are currently dead**, and PLAN/README
lean on them twice: as a W7a gate, and in the risk table's claim that the 144 + 22 reference
`limb` -> `chain` rename is "fully covered by `test_skeleton` + the GUI browser test". That
second gate is the one v2 most needs, because W1 and W7 are the two workstreams that edit
the GUI's JS -- 22 limb references in `gui/web/static/*.js`, and 88 landmark references
across five web assets.

Restore it rather than delete it. Injected with `uv run --with playwright`, all **53 pass**,
so the file is not rotted, only unreachable. What restoring costs, measured:

- **Serially the file is 168 s** -- eight times the entire rest of the suite -- against 34 s
  of CPU. It is browser wait, not work.
- Under `-n auto` it drops to ~16 s but **goes flaky: 3 of 4 runs failed**, always in
  `test_hidden_draws_a_mark_and_nothing_else_moves` and/or `test_l_toggles_the_camera_layout`
  (both keystroke-then-redraw races that lose under CPU contention). The full suite with
  playwright is 57.3 s wall, up from 21.4 s.

So the work is: add `playwright` to the `test` dependency group; add a CI job that runs
`playwright install --with-deps chromium` and this file only (the browsers are already
cached locally, so the dev cost is just the wheel); and settle the two racy tests -- either
fix the wait or mark the file `xdist_group` and run it under `--dist loadgroup` so it stays
serial. The stale `_cached_chromium` docstring saying "1223 present, 1234 wanted" should go
with it; this machine now has 1234.

Until this lands, **the limb rename has no JS gate**, and W1 should not be treated as covered.

#### W0b. Inner-loop and contributor docs

- `CONTRIBUTING.md`: document `-n0` for single-file runs. One 6-test file costs **2.6 s and
  35 CPU-seconds with the default `-n auto`** (32 workers each import torch, jax, scipy and
  fastapi to collect a suite they will not run) against **0.67 s with `-n0`**. That 4x is on
  the loop this refactor lives in for ten workstreams.
- Same file, line 32: "Some tests download the detector weights on first run" is stale --
  `ci.yml:24` records that the download is mocked and the suite runs fully offline.
- Gate: none needed; `mkdocs build --strict` in W8b covers the prose.

### W1. Skeleton: points, edges, symmetries, colors -- and out of the config

- `skeleton.py`: `Skeleton.from_config` reads a skeleton FILE (`[skeleton] include`, a
  packaged name or a path relative to the config) or, with no `[skeleton]` table, the
  skeleton name recorded with the detector's weights; neither present is an error naming
  both places. Points equality against the checkpoint stays the real check
  (`pose2d/stream.py` already does it).
- The definition is four keys: `points` (ordered -- the channel contract), `edges` (point
  pairs -- the whole topology), `symmetries` (point pairs) and `colors` (point name or
  `*` pattern -> hex, optional). `Skeleton.bones` comes from `edges` directly.
- **Delete the limb/chain concept**: `limb_names`, `limb_id`, `n_limbs`, the limb-keyed
  `palette`, `_parse_limb_points`, and `infer_symmetries_by_name`. `Skeleton` gains a
  per-point color array instead, resolved through the selector (exact name, then pattern,
  then colormap-by-index) plus a per-bone array (an edge takes its source point's color,
  overridden by a layer's `bone_color`, W6). This is ~144 references in `src/` and 22 in
  `gui/web/static/*.js`, mechanical except the two below.
- `gui/server.py` `_limb_legend`: groups by distinct COLOR rather than by limb name --
  ten swatches for `fly38`, labeled by the shared prefix of the points in each group.
- `results.py`: stop writing `limb_names` / `limb_id`; write the per-point colors. A file
  written before this reads back with a colormap palette, which is acceptable because
  colors are cosmetic -- no corpus repack.
- **New symmetry check**: the point permutation must be an automorphism of `edges` (apply
  it and get the same edge set back). Verified against `fly38` while drafting. Replaces
  the chain-consistency check, needs no chains, and catches both realistic typos.
- `pictorial.py` `skeleton_chains` is UNCHANGED -- it derives chains from the bone graph,
  which is what makes chains safe to delete from the schema.
- `data/skeletons/fly38.toml`: rewritten in the new shape (drafted as
  `design/v2/fly38.toml`; 107 lines, verified to parse, to have every point matched
  by exactly one color key, and to satisfy the automorphism check).
- `project.py`: the skeleton template it writes, the preset detection at line 1468.
- `skeleton_migrate.py`: its emitter and its `limbs` / `symmetries` change-kinds move to
  `edges`.
- Gate: `test_skeleton`, `test_skeleton_migrate`, `test_project`,
  `test_gui_server`, plus `test_gui_browser` for the 22 JS references -- which means **W0a
  first**, since that file runs nowhere today.

### W2. Rig: `[calibration]`, `[default_camera]`, `[cameras.<name>]`

- `config.py` `camera_table()`: returns `([default_camera], [cameras.*])`. `[cameras]`
  may hold no bare keys and no `defaults` sub-table -- both are refused with a message
  pointing at `[default_camera]`.
- `config.py` `calibration_path()`: reads `[calibration].path`. Its own table, so
  `project compose_config` can inject it as a fragment without overlapping the rig
  fragment -- which deletes the `[cameras]`-before-`[cameras.defaults]` exception in the
  fragment guard (see W8a).
- `[cameras.<name>].mirror` and `config.py` `mirror_views()` are DELETED, along with
  `tests/test_config_mirror.py`. The key exists for flip augmentation during training --
  a mirrored sample must carry the id of the camera it now looks like, or
  `ipsi_contra_split` calls every swapped channel by the wrong side. Its in-repo consumer
  (`deeperfly.training`) went in 0.2.0 and the key was kept as "a fact about how the rig
  was built"; the trainer that actually exists, dfpose, hardcodes its own
  `NEWRIG_MIRROR_CAMERA` (`dfpose/datasets/labels_h5.py`) and has never called
  `mirror_views()`. Nothing else reads it. (Unrelated to `check_mirror_consistency`,
  which reads pathway `fliplr` ops and dies in W3 for its own reason.)
  If a consumer ever wants it back it is derivable, no config surface: under v2 every
  camera declares `azimuth_deg`, and the mirror is the camera at the negated azimuth
  (`rh` -120 <-> `lh` 120, `f` and `h` to themselves), which reproduces the default rig's
  table exactly and, unlike the extrinsics, is known before there is a calibration.
  A leftover `mirror` in a camera table is refused by name like the other removed keys,
  not ignored -- today `_NON_RIG_KEYS` swallows it, so silence is exactly what a reader
  of a v1 config would misread as "still honored".
- `cameras.py`: `_NON_RIG_KEYS` becomes `("video",)`; `from_config` takes the shared
  values from `[default_camera]`.
- Gate: `test_cameras`, `test_config_narrowing`.

### W3. Detection: one detector, per-camera crop, synthesized plan

- `config.py`: `[pose2d] class` / `weights` / `auto_crops` (+ the existing `precision`,
  `batch_size`, `decode_buffer`), `[pose2d.crops]` (camera -> `[x, y, w, h]`, one value
  type), and `[pose2d.crop_search]` (was `[pose2d.autocrop]`). A camera in `auto_crops`
  with a box in `crops` searches from that box, which is what `crop_seed` would have
  been; there is no third form.
- `pose2d/pathways.py`: `DetectionPlan.from_config` synthesizes sources/preprocessors/
  pathways from the camera table. Delete `_parse_sources`, `_parse_preprocessors`,
  `_parse_models`, `_parse_pathways`, `_parse_output_points`, `_identity_triples`,
  `_default_model`, `_resolve_view`, `check_mirror_consistency`. `Pathway` keeps its
  fields; the mapping is always the identity but stays materialized, because `pts2d`
  assembly and the fingerprint read it.
- `config.py` `source_patterns()` -> reads `[cameras.<name>].video`, keeping its name and
  return shape (`str | list[str]`) so `recordings.py`, `jobs.py`, `merge.py` and the GUI
  are untouched. The v1-shape check lives here: a value that is a list of literal
  filenames with no regex metacharacter is refused by name, because under v2 it would
  concatenate the alternates it used to choose between -- silently doubling a recording
  is the one migration failure that produces a plausible result instead of an error.
- `config.py` `narrowed_to_sources()` / `narrowed_to_covered_views()`: rewritten on the
  camera table, and much shorter -- source, pathway and view are one name now, so the
  three-way bookkeeping and the `output_points` re-derivation collapse to "drop the
  camera, and blank the grid cells that named it".
- `pose2d/autocrop.py`: bump `SIDECAR_VERSION`; `Resolution.preprocessor` -> `camera`.
- `preprocessing.py`: the **op grammar loses its last consumer**. `_parse_op`,
  `_normalize_ops`, `frame_transform_from_ops` and the `Fliplr` / `Flipud` / `Rot90`
  classes go; `Crop`, `Resize` and `FrameTransform` stay as internal machinery, because
  a crop still has to map detections back to raw pixels and `pathways.py:137` still
  builds a `Resize` to fit the detector input. `fliplr` was only ever reachable through
  a mirrored pathway (deleted above) and the `fly38_sparse_config.toml` fixture (deleted
  below); `resize` as a *config* op never had a caller at all. `visualization/compose.py`
  imports `preprocessing` for a video's `crop`, which is unaffected.
- Gate: `test_pathways`, `test_autocrop`, `test_config_narrowing`, `test_pose2d`,
  `test_fingerprint`. Fixture churn: `tests/data/fly38_sparse_config.toml` and
  `fly38_deepfly3d.toml` are deleted; the tests behind them (partial per-view
  visibility, the mirrored-pathway left/right check) either move to a hand-built
  `pts2d` with NaN columns -- which is what they are actually about -- or go.

### W4. Footage: regex matching and concatenation

Independent of the schema work above and testable on its own, but it is what makes the
`video` key mean what W3 says it means.

- `recordings.py`: `_camera_glob`, `_case_insensitive_glob`, `_as_alternates` and
  `_first_if_video` (with its "using only the first" warning) are deleted. `camera_files`
  becomes: for each entry in order, `re.fullmatch` every name in `root` (case-insensitive,
  filenames only -- a pattern never traverses into a subdirectory), natural-sort the
  matches, check they are one series, concatenate. The extension-priority tiebreak in
  `_footage_exts` goes with it: the pattern names the extension, and a directory holding
  both `camera_RH.mp4` and `camera_RH.avi` is now an error rather than a silent pick.
- **The series check** is the whole safety of the change: mask every run of digits in each
  matched name and require one distinct result. `camera_RH_0.mp4` / `camera_RH_1.mp4` pass
  (`camera_RH_#.mp4`); `camera_RH.mp4` / `camera_0.mp4` do not, and the error names both
  files and the camera. It cannot catch a pattern loose enough to match two real cameras
  (`camera_\d\.mp4`) -- the frame-count gate below is what catches that.
- `io/__init__.py`: `open_reader` stops keeping `files[0]` for video and returns a new
  `ConcatReader` when a video list has more than one entry. `ConcatReader` holds one
  `VideoReader` per part plus the exclusive-prefix-sum of their lengths, and maps a global
  index to `(part, local)` by `bisect`. Its `cursor()` keeps the underlying
  `VideoCursor`s open (one per part, opened lazily) so the seek-not-walk and held-open-cursor
  work is not undone by a boundary; a forward `stream_frames` walks the parts in order
  and never seeks at all.
- **Exact lengths are the hard requirement.** A global index is undefined unless every
  part but the last has an exact frame count, and `VideoReader.count()` already returns
  `None` when the container header has no `nb_frames`. `ConcatReader` therefore probes
  each part's header; a part that will not answer is counted once by a full decode, logged
  with what it cost, and cached in the run's sidecar so it is paid once per recording, not
  once per stage. Do NOT estimate from `duration * fps` -- an off-by-one there shifts every
  frame index after the boundary, and nothing downstream can detect it.
- Parts must agree on frame size and frame rate; a mismatch is an error naming the two
  files, because a concatenated stream with two resolutions has no single intrinsics.
- **`_frame_counts_match` inverts under this change and must be rewritten, not extended.**
  It compares *file* counts first and returns early when they differ -- which is precisely
  the legal case now, since the acquisition splits each camera at its own byte threshold
  and one camera can be two files while another is three. Drop the file-count comparison
  (keep it only for image sequences, where it is still the cheap proxy it was) and compare
  the concatenated frame count, which `ConcatReader.count()` returns as the sum over parts.
  Left as it is, every split recording is silently skipped at discovery. This is also the
  check that catches a runaway pattern -- a camera matching two real cameras comes out at
  twice the others' T.
- Gate: `test_recordings` (new cases: split parts, mixed schemes, mixed extensions, an
  image sequence unchanged), `test_io` (a `ConcatReader` against the single-file reader
  over a video split with `ffmpeg -f segment`, asserting byte-identical frames for random
  access, slices and full decode), `test_fingerprint`.

### W5. Pipeline flags, named collections, point selectors

Three commits, all narrow, all independent of each other.

#### W5a. Stage flags

- `config.py` `stage_flags()`: `[pipeline] <stage>`, no `do_` prefix; a `do_*` key is
  refused by name.
- `config_schema.py` `stage_flags_spec()`, `cli/config.py`, `gui/server.py`'s
  `undescribable` list.
- Gate: `test_pipeline`, `test_cli_run`, `test_cli_config`, `test_config_schema`,
  `test_gui_server`.

#### W5b. Named-collection cleanups

- `[inverse_kinematics.head]` / `[abdomen]` -> `[inverse_kinematics.markers.head]` /
  `.abdomen`, so the marker tables are not two fixed names sitting in the stage's knob
  namespace (which is also what `_params`' strict validation special-cases today). These
  are the fitted body's articulation chains, not `[skeleton.chains]` entries -- the plan
  said "chain" here and that was ambiguous.
- Gate: `test_inverse_kinematics`, `test_ik_forward_bodyplan`.

#### W5c. Point selectors

- One grammar, used by `[bundle_adjustment] points`, the postprocess ops' `points` /
  `midline`, and `[skeleton.colors]`: an entry is a point name or a `*` pattern
  (`fnmatch`, case-sensitive). Resolution refuses two patterns matching one point (naming
  both), refuses a pattern matching nothing (always a typo), and logs the resolved set,
  because over-matching is the one failure a selector cannot detect for itself. An exact
  name beats a pattern.
- `BundleAdjustmentParams.points_to_use` and `SymmetrizeParams.pairs` stay the RESOLVED
  fields, so `bundle_adjustment.py` and `postprocess.py` are untouched.
- `symmetrize` takes `points` (either half of a pair) and looks the partner up in the
  skeleton's `symmetries` instead of restating pairs.
- Gate: `test_bundle_adjustment`, `test_ba_weights_and_bones`, `test_postprocess`, plus a
  selector unit test for the three refusals.

### W6. Visualization layers and defaults

- `visualization/compose.py`: a video is `[visualization.videos.<name>]` + `grid` +
  `layers` -- keyed by name, so `video_name` goes and a duplicate is a TOML error. Each layer
  `{ draw, stage, ...style }`. `_expand_grid` runs once per layer and appends, so layer
  order is draw order. Footage is implicit under camera cells (`footage = false` drops
  it). Delete `panels`, the per-op `kwargs` tables, video-level `plot`/`stage`, and
  `video_name` (-> `name`).
- New layer style key `bone_color` (DeepLabCut's `skeleton_color`): every bone in one
  color, joints keeping theirs. Falls out of the per-bone color array added in W1.
- `[visualization.default_video]` (background, crop, cell, output_fps, speed) and
  `[visualization.default_layer]` (style) replace `[visualization]`'s bare keys and the
  three-level kwargs merge: resolution is now exactly `default -> explicit`, once per
  collection, so `_op_kwargs` and `_layout_key`'s merge order go away. `[visualization]`
  itself holds nothing.
- `VideoSpec`/`Panel` keep their fields; only the parser changes.
- Gate: `test_visualization_panels`, `test_visualization_opencv`, `test_gui`.

### W7. Landmarks removed, and calibration scale pinning with them

Independent of W1-W6: this removes a feature the v2 schema never described. See
[README.md](README.md#landmarks-what-the-removal-actually-costs) for why it is safe --
the short version is that **no `landmarks.toml` and no `landmarks/` group exists
anywhere**, so every rig ever solved here was already animal-as-target.

Two commits, because the second is only reachable once the first has removed the caller.

#### W7a. The landmark namespace

- **Delete `src/deeperfly/landmarks.py`** (`Landmark`, `LandmarkSet`, `SCOPES`,
  `LANDMARKS_FILENAME`, `LANDMARKS_FORMAT_VERSION`).
- `gui/labels.py`: `LandmarkLabels`, `_write_landmarks`, `load_landmark_labels`, and
  `save_labels`' `landmarks=` parameter. The parameter's whole reason to exist was that
  `save_labels` is a whole-file rewrite and a caller who loaded landmarks had to pass
  them back; with the group gone, four callers stop having to remember that
  (`cli/gui.py:261`, `cli/merge.py:164`, `skeleton_migrate.py:414`, `gui/server.py:1410`
  -- the last carries a comment explaining the hazard, which goes with it).
- `gui/state.py`: the `landmarks` field, `display_landmarks`, `set_landmark`,
  `clear_landmark`, `landmark_counts`, `landmark_names`, `has_landmarks`, and the
  landmark half of the dirty accounting at `state.py:372`.
- `gui/server.py`: `_landmarks_meta`, the `landmarks` key in the meta and frame payloads,
  and the `set_landmark` / `clear_landmark` message handlers.
- `gui/__init__.py`: `_load_landmarks` and its carry-observations-across-by-name logic
  for a changed landmark set.
- `gui/web/`: the landmark panel and its rendering -- `app.js` (45 references),
  `poseView.js` (32), `index.html` (6), `styles.css` (4), `types.js` (1).
- `import_outputs.py`: `_plan_landmarks`, `dest_landmarks`, `landmarks_taken`,
  `landmarks_only_source`; and the two report rows in `cli/project.py:532,547`.
- `package.py`: the `landmarks` archive entry, the `landmarks/index` read, `has_landmarks`
  in the manifest, and the `("landmarks", "landmarks.toml")` extraction pair. **Bump
  `PACKAGE_FORMAT_VERSION` to 2** -- the archive layout changes, and `package.py:354`
  already refuses a version it does not understand, so a v1 archive gets a named error
  rather than a silent missing member.
- A `landmarks.toml` still sitting in a project root is refused by name when the project
  loads, not ignored -- silence is what a reader would misread as "still honored".
- Gate: `test_gui_uncalibrated` (63 references -- the largest single body of landmark
  tests; what survives is the uncalibrated-session behavior itself, which is the point of
  the file), `test_import_outputs`, `test_package`, `test_project`, and `test_gui_browser`
  for the 88 references across the five web assets -- again only real once W0a lands.

#### W7b. The solve

- `calibration_solve.py` `build_observations`: drop `landmark_xy`, `landmark_names`,
  `landmark_static` and the `use` parameter (`"landmarks"` / `"keypoints"` / `"both"`).
  Every track is now one keypoint at one frame, so **`Track.kind` and `Track.static` lose
  their reason to exist**, `Observations.summary()`'s `landmark_tracks` count goes, and
  `Track.scatter_px` (per-view pixel scatter of a static landmark averaged over frames)
  goes with them.
- `merge_observations`: drop `share` and the rig-scoped cross-recording merge
  (`shared_cols` / `shared_meta` and the across-recording spread that was the drift
  signal). Multi-recording solves become plain concatenation.
- **The two guidance strings that recommend a landmark must be rewritten**, not deleted:
  the disconnected-co-visibility remedy at `calibration_solve.py:462` and the
  equation-ratio remedy at `:468`. Both must name a remedy that exists -- label the same
  point in a view from each group; label more frames.
- **Scale pinning goes** (see [README.md](README.md#scale-where-physical-units-actually-enter)):
  `bundle_adjust`'s `scale_pair` / `scale_distance`, `_scale_label`, the
  `bone_pairs`/`bone_targets` scale-bar construction at `:903-906`, `--scale-from` in
  `cli/app.py` and `cli/calibrate.py`, and the `units="mm" if scale_distance else
  "arbitrary"` branches at `cli/calibrate.py:437` and `gui/ba.py:778`. `UNITS` loses
  `"mm"` and `SCALE_SOURCES` loses `"known_distance"`. `_normalize_initial_scale` is
  UNCHANGED -- it is a legibility rescale, not a unit, and it stays honest about that.
- `cli/app.py`: the `--points`, `--share` and `--scale-from` options on `calibrate`, and
  the "landmarks" mention in the `package` help at `:932`.
- `gui/ba.py`: the `source` knob collapses from `"gt"` / `"gt+landmarks"` / `"landmarks"`
  to always-GT -- the field, its validation at `:219`, and the `:365` branch.
- `visualization/bird.py`: rename the local `_landmarks` helper (an orientation-group
  lookup, nothing to do with calibration) so the word stops colliding.
- **Do not touch** the IK sense of the word: a chain's **base landmark** in
  `inverse_kinematics/{__init__,articulation,bodyplan}.py`, `results.py:471`,
  `config.py:584`, and the prose in `pathways.py:330`.
- Gate: `test_calibration_solve` (38 references), `test_calibrate_cli` (32),
  `test_cli_suggest` (13), `test_gui_ba`. Plus one end-to-end cold-start solve from
  keypoints alone on an example recording, checked for a reprojection error in the same
  band as the cached calibration -- the removal's one real risk is cold-start
  conditioning, and this is what measures it.

### W9. IK model packs: NeuroMechFly stops being wired in

Independent of W1-W8 except for `config.py` and `results.py`. The full design, the
evidence for each hardcode and the measured flybody comparison are in
[ik-model-packs.md](ik-model-packs.md); this is the landing order.

The IK core is already chain-name agnostic -- `unfittable_branches`,
`estimate_chain_scale`, `calibrate_chain` and the solve name no body part in any
branch. What is hardcoded is nine specific things, five of which are names and one of
which is dead.

#### W9a. Delete the dead body-frame alignment state

- `Alignment.r_body`, `.head_origin`, `align.to_local`, `align.to_world` and
  `align._body_axes` / `_head_origin`. No production consumer: the leg subtree stopped
  being rotated by `r_body` when it went identity (`bodyplan.py:40`), and grep finds the
  four only in `tests/test_inverse_kinematics.py`. `_body_axes` carries the stage's only
  hardcoded leg names (`["rf","rm","rh"]` / `["lf","lm","lh"]`,
  [align.py:153-156](../../src/deeperfly/inverse_kinematics/align.py#L153-L156)), so this
  removes the hardcode rather than parameterizing it.
- Gate: `test_inverse_kinematics` (two tests deleted with it).

#### W9b. The registration and the chain list stop naming fly parts

- `Articulation.coxa_points` / `coxa_neutral` -> `anchor_*`; `_coxa_world` /
  `_coxa_similarity` -> `_anchor_*`; the mesh asset's `coxa_idx` -> `anchor_idx`. The
  solve is already generic (`body_similarity` needs three finite anchors, not six coxae).
- More than a rename: `coxa_points` is `["lf_thorax_coxa", ...]` -- **skeleton point names
  inside a model asset**. The pack declares `anchors` as model *bodies* (`lf_coxa`, ...)
  and their neutral frames come from `bodies`; which tracked point observes each is read
  out of the binding in W9f. Until W9f lands, the lookup keeps a shim that resolves an
  anchor body through the current name convention, so this commit stands alone.
- `fit_head` / `fit_abdomen` -> `chains: list[str] | None` (`None` = every chain the
  pack defines, `[]` = legs only). Five sites: `config.py:620`, `:1188`, `:1276` and
  `results.py:497`, `:502`, the last two becoming `chain_scale(name)`.
- Gate: `test_inverse_kinematics`, `test_ik_forward_bodyplan`, `test_config_schema`.

#### W9c. The template carries what only the model knows

- A leg declares its `side` instead of it being inferred from the point name's first
  letter ([template.py:225](../../src/deeperfly/inverse_kinematics/template.py#L225)) --
  the one hardcode here that **fails silently**, since a leg named `T1_left` currently
  resolves to side `"r"` and gets mirrored axes and the wrong bounds with no error. Once
  W9f lands the binding, `side` is the only thing left that a leg id was carrying.
- A DOF declares its `angle` name, defaulting to flygym's `<joint>-<dof>`, so the
  convention stops being written into `template.py:106` and `bodyplan.py:434`.
- A joint declares an optional `quat`, and the pack manifest a `rest_axis`. **This is
  the only place W9 changes behavior rather than names.** Every NeuroMechFly leg body is
  `quat="1 0 0 0"` with segments along `-z`, which is what licensed the identity
  `offset_quat` at [bodyplan.py:421](../../src/deeperfly/inverse_kinematics/bodyplan.py#L421)
  and `REST_AXIS` at [forward.py:47](../../src/deeperfly/inverse_kinematics/forward.py#L47).
  flybody's are neither -- its segments run along `+y` and every leg body carries a
  rotation -- so `_leg_joints` and `forward.leg_fk` have to thread both. QuickIK already
  accepts `offset_quat`; deeperfly never fills it.
- Gate: the packaged pack sets all three to today's values, so the fit must come out
  **bit-identical** -- `test_inverse_kinematics_quickik`'s
  `test_the_leg_parameterisation_is_flygyms`, plus a fitted-angle comparison against a
  cached example run. A rest-axis or quat bug is otherwise a plausible-looking fit.

#### W9d. `model` selects a pack

- `[inverse_kinematics] model = "<name or path>"` resolving
  `data/models/<name>/model.toml`, which names the template, the articulation and the
  mesh beside it. `template` is refused by name like the other removed keys.
- Closes the gap that `Articulation.load` takes a `ref` the config never passes
  ([config.py:1282](../../src/deeperfly/config.py#L1282)), and makes the overlay mesh
  selectable at all.
- A pack naming points the skeleton lacks is refused at load rather than degrading into
  NaN observations.
- Gate: `test_config`, `test_config_schema`, `test_inverse_kinematics`.

#### W9e. `nmf_*` -> `model_*`

- 288 identifier occurrences over 34 distinct names (`nmf_angles`, `nmf_pts3d`,
  `has_nmf`, `nmf_body_plan`, `skeleton_nmf`, `mesh_nmf`, ...) across `results.py`,
  `visualization/compose.py`, `pipeline/stages.py`, `pipeline/fingerprint.py`, the GUI
  server and five JS modules. `model_*` is already the internal vocabulary
  (`IKResult.model_pts3d`, `BodyPlan.to_model`). Draw ops become `skeleton_model` /
  `mesh_model`.
- `results.h5` dataset names change, so this rides the v3 repack with candidate C3
  rather than forcing one of its own.
- Gate: the full suite, `test_gui_server`, and the headless GUI check -- a renamed key
  the JS still reads by its old name is a runtime throw Python tests cannot see.

#### W9f. The binding: one artifact per (skeleton, model) pair

The half of the decoupling W9a-W9e do not do. Today the placement of a tracked point on
the model is stated in three places and in two forms -- 30 leg points as a name convention
inside `template.toml`, 7 head/abdomen markers as `body` + `offset` inside
`nmf_articulation.json`, and all 38 as rows in `docs/keypoints/assets/keypoints.json`,
which is where they are actually authored and which the bake script already reads. The
consequence is that a pack reaches into the skeleton's namespace and a skeleton cannot
reach a second model.

- `src/deeperfly/data/bindings/<skeleton>@<model>.toml`: one row per tracked point,
  `{ body, offset = [0,0,0], approximate = false, base = false }`. **The offsets live
  here and nowhere else** -- `abdomen0 = { body = "c_abdomen12", offset = [-0.37, 0.0,
  0.34], approximate = true }` is a fact about how fly38 was labelled against
  NeuroMechFly, true of neither alone, so the pair is what it is keyed on.
- Whether a row becomes a fitted plan joint or a zero-DOF pseudo-joint is *derived* from
  whether its offset is zero on a body whose origin is its parent joint -- one schema, not
  two, and `depth` stops being authored because the pack knows how deep each body is. The
  one exception is a point at a chain's own origin (`neck` on `c_head`), which the
  derivation would call a fitted joint and which must stay a `base = true` nomination: the
  head's DOFs rotate about that point, so it constrains none of them. The binding carries
  `base`, and the derivation reads it first.
- **`nmf_articulation.json` loses its `markers` arrays** -- they are binding rows
  (`point` + `body` + `offset` + `depth` + a precomputed `neutral`) living in a model
  asset. `bodies` stays and becomes mandatory, since it is what a `(body, offset)`
  resolves against, and `neutral` becomes derived rather than baked; the "this asset
  predates `bodies`" error path goes with it. `Articulation.load(marker_overrides=)`
  becomes "resolve markers from the binding", and `[inverse_kinematics.markers.<chain>]`
  stays a per-run patch over it -- which is what it already is.
- The W9b anchor shim is deleted here: `anchors` are model bodies, and the point
  observing each is the binding row naming that body at offset zero.
- `template.toml` loses `suffix`; `align.py`'s `leg.joints[0].point` and `bodyplan`'s
  `x-deeperfly-point` resolve through the binding. `[inverse_kinematics] binding`
  defaults to `<skeleton>@<model>`, so a normal config writes nothing, and an unbound pair
  is a load error naming both halves.
- `keypoints.json` becomes derived from binding + pack; `map_keypoint`'s name conventions,
  its computed `distal_tip_offset` and its hand-tuned `MIDLINE_POINTS` / `ABDOMEN_POINTS`
  stop being three kinds of rule in a docs script and become one kind of row. A
  `deeperfly ik bind <skeleton> <model>` generator expands the conventions and resolves the
  geometry, leaving the ambiguous rows for review -- **part of this commit, not a
  follow-up**, since the claw offset is computed from mesh geometry and cannot be
  hand-authored.
- `approximate` (today only `abdomen0..4`, which have no exact NeuroMechFly counterpart)
  reaches the IK and is **carried and reported, never acted on**: the residual splits
  exact from approximate so a reader can tell a bad fit from a bad retarget. Down-weighting
  them or fitting their offsets is explicitly not proposed -- no measurement supports it,
  and a fitted offset would absorb real error into the retarget.
- Gate: the packaged `fly38@neuromechfly.toml` reproduces the committed `keypoints.json`
  and the committed `nmf_articulation.json` markers **exactly** -- both are in git, which
  makes this a byte comparison rather than a residual band -- and the fit stays
  bit-identical. Plus the invariant as a test: no skeleton point name appears in any pack
  file, no model body name in any skeleton file, no offset anywhere but a binding.

#### W9g. The flybody pack -- after the release

- `scripts/build_nmf_mesh_asset.py` reads one hardcoded MJCF and knows which bodies are
  the head chain, which the abdomen, where each marker attaches and that the coxae are
  the anchors. Generalizing it means moving that into a per-model bake spec; that plus
  transcribing flybody's axes and limits is the bulk of the work, and it needs MuJoCo and
  the flybody meshes.
- It needs a `fly38@flybody` binding as well as the pack. That is the cheap half: the leg
  rows are a name convention over flybody's `{part}_T{n}_{side}` bodies, and only the
  abdomen rows are a judgement -- flybody has 7 tergites where NeuroMechFly has 5, so
  `abdomen0..4` land on different segments and the offsets are new hand-tuned numbers
  carrying `approximate = true`, exactly as they do today.
- The topology is the encouraging part: flybody's leg is the same seven DOFs at the same
  four anatomical joints as NeuroMechFly's, and its head chain is the same three, so no
  new plan machinery is needed -- see the comparison in
  [ik-model-packs.md](ik-model-packs.md#what-flybody-actually-needs).
- Gate: a fit of flybody's own neutral keypoints returns flybody's own joint angles, the
  way `test_the_leg_parameterisation_is_flygyms` pins NeuroMechFly's.

### W10. Postprocess stays, and the IK inherits it

Nothing moves in `src/`, which is the decision. What is owed is two doc notes and one
measurement.

- `default_config.toml`'s postprocess header and `config-schema.md`'s "deliberately
  kept" line both gain the reason the op chain survives next to `body_alignment` -- same
  medians, different destinations -- so it is not deleted later as a duplicate.
- **The measurement owed.** With `fixed_body = true` (the default) a leg's thorax-coxa
  is joint 0, at a baked `offset_pos` with no upstream DOF, so no setting of the angle
  vector can move it -- yet `_leg_joints` still sets its `x-deeperfly-point`, so its
  per-frame detection noise enters the reported residual as a floor no parameter can
  reduce. It cannot bias the fitted angles. It can inflate every residual number quoted
  for the stage, including the ones in these docs. Check whether they include it; mask
  joint 0 out of the reported residual if they do.
- `fixed_body = false` is the one mode where staticness is *not* structural: a free
  6-DOF root is pinned by exactly those six coxa observations, so per-frame coxa noise
  jitters the root. The answer is that a tethered recording belongs on `fixed_body =
  true`, not that the prior gets applied a second time inside the solve.
- Gate: `test_postprocess` and `test_inverse_kinematics`, both unchanged -- which is the
  point.

**Half settled (2026-08-29): "2D after triangulation is always a reprojection."** The
`postprocess` half is done -- `freeze_2d` is deleted, `{ op = "static" }` corrects the 3D
only, and the `points2d_override` write branch is gone (the read stays for old files). The
measured artifact it had been storing was 1.4 px median on the seven frozen columns. The
`triangulation` half below is still open, with both prerequisites intact.

**What remains open:** applying the same rule to `triangulation`, which would make
`reproj_error` mean one comparable thing at every stage -- this stage's 3D against the
detections. Two prerequisites, both real:

1. Triangulation's stored 2D carries the outlier rejector's decisions in its NaN
   pattern, and a total reprojection cannot express that. The support has to move to an
   explicit `(V, T, P)` inlier mask, which is smaller than the array it replaces and
   says what it means instead of encoding it in absences.
2. `acquisition.stored_vs_pose2d` guards a degenerate-signal trap that works *because* a
   stage's 2D may differ from `pose2d/points`; `results.py`'s schema note records that
   the trap was reintroduced once already. Confirm the inlier mask plus the detections
   can carry that witness before the branch is deleted.

`freeze_2d`'s stated reason -- stay a pixel measurement rather than absorb the rig's
residual -- did not hold with `eks = true`, since the smoother's 2D is stored `"derived"`
and is therefore already a reprojection. That is what settled the `postprocess` half above.
None of this blocks the release.

### W8. The configs, the docs, the release

#### W8a. The configs

- `src/deeperfly/data/default_config.toml` <- `design/v2/default_config.toml`.
- All six example configs rewritten (`examples/data`, `flywheel_260507_g25_Fly1_002`,
  `scape_260417_IN07B001_Fly4_{004,005,007}` plus 005's `config2.toml`,
  `scape_260810_E49_Fly3_002`), keeping each one's measured commentary.
- `project.py`: `RIG_TABLES = ("default_camera", "cameras", "io")`, and the injected
  calibration is its own `[calibration]` fragment -- so the
  `[cameras]`-before-`[cameras.defaults]` legal-overlap exception in the fragment guard
  is deleted rather than reworded.
- `tests/helpers.py`: `seven_camera_default_text()`'s line surgery shrinks to one
  `[cameras.h]` table and one grid cell.
- Gate: the full suite, then **one full `deeperfly run` per example directory**, checked
  for (a) exit 0, (b) `results.h5` carrying every stage the config enables, (c)
  reprojection error in the same band as the cached `deeperfly_outputs_*` beside it, (d)
  videos rendered at the expected size. This is the "robust" requirement, and the only
  gate that can prove it.

#### W8b. Docs

- `docs/reference/configuration.md` (rewritten -- it is organized around the v1 tables),
  `docs/guides/configuration.md`, `docs/explanation/{pipeline,detectors,conventions}.md`,
  `docs/guides/cli.md` (it still documents a `dense-config` command that no longer
  exists), `docs/getting-started.md`, `CHANGELOG.md`.
- Gate: `mkdocs build --strict`, plus a grep for every deleted key name across `docs/`.

## Risks

- **Concatenation is the only change here that is not a surface change.** Everything
  else in v2 rewrites how a run is *described*; `ConcatReader` changes how frames are
  *read*, and a frame-index bug in it is silent -- the run succeeds and every keypoint
  after the boundary belongs to the wrong frame. That is why its gate is a byte-identical
  comparison against the single-file reader rather than a shape check, and why no example
  recording exercises it (all eight are single-file), so the split fixture has to be made.
- **The example run gate is the long pole**: eight cameras x ~3000 frames per recording.
  The runs go in the background while the suite proceeds.
- **Cold-start conditioning is the landmark removal's one real cost.** Static landmarks
  were spread through the scene volume where keypoints sit in a ~3 mm blob, and cold start
  is a live path (`gui/server.py:777` when `not has_rig`, `cli/calibrate.py` `--cold`). The
  equation-count half of the argument was never binding here -- a track seen in 8 views is
  5.33x against a `MIN_EQUATION_RATIO` of 1.5 -- and the conditioning half has only ever
  been tested without landmarks, because none exist. Still: the cold-start gate in W7b is
  the only thing that turns that from an inference into a measurement.
- **The landmark removal is the widest single deletion**: 598 references across 40 files,
  5 of them web assets -- but **~35 of those are the unrelated IK sense** of the word (a
  chain's base landmark). A blanket rename or a global grep-and-delete would break the IK
  stage silently, which is why W7 names both sides explicitly.
- **`bird`** stays a magic view name in a grid (a non-camera cell gets no footage).
  Untouched by this change, but it becomes the only name in the schema that means
  something without being declared anywhere.
- **The `limb` -> `chain` rename is wide** (144 + 22 references) and lands in the same
  commit as the skeleton parser, so a bisect over step 1 is coarse. It is mechanical and
  fully covered by `test_skeleton` + the GUI browser test -- **but the browser test does not
  run in any environment that exists** (no `playwright` in the test group or in CI, 53 tests
  collapsed into one reported skip), so that coverage is on paper until W0a restores it.
- **`results.h5` keeps saying `limb_id`** while the config and the code say `chain`. One
  documented mismatch, in `_write_skeleton` / `_read_skeleton`, which is where a
  format-versus-code naming difference belongs -- but it is a mismatch.
- **The rest axis and the leg quaternions are the one silent thing in W9.** Every other
  item is a rename or a deletion whose failure is loud. Getting `rest_axis` or a leg
  joint's `quat` wrong produces a fit that converges, reprojects plausibly and reports
  wrong angles -- which is why W9c's gate is bit-identity against a cached run and not a
  residual band.
- **W9f is the widest part of W9 and the only one that touches the docs pipeline.** It
  moves a file out of `docs/` that two scripts and a browser viewer read, and it changes
  `template.toml`'s schema. Its safety comes from both of its targets being committed:
  `keypoints.json` and `nmf_articulation.json` must come back byte-for-byte out of the
  generated binding, which is a much stronger gate than the fit's residual.
- **W9e renames `results.h5` datasets**, so it is the second thing (with `limb_id`)
  waiting on the v3 repack. Shipping v2 without it means a model-agnostic stage under a
  model-specific vocabulary; shipping it means the repack is no longer optional.
- **Sparse detection is gone for good.** If a 19-channel checkpoint ever has to run
  again, it runs under a v1 tag, not a resurrected code path.
