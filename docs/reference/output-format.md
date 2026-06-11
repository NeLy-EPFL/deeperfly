# Output format

A run writes everything to its output directory (`<input>/deeperfly_outputs/` by
default, or `-o`):

```
deeperfly_outputs/
├── poses.h5        # the result: cameras, skeleton, per-stage 2D/3D data
├── config.toml     # byte-for-byte snapshot of the config this run used
├── run.json        # per-stage fingerprints (drives cache reuse)
└── *.mp4           # one per [[visualization.videos]] entry
```

## `poses.h5`

A self-contained HDF5 file (schema **version 2**). Each pipeline stage writes its
own group, so a stage never overwrites another's data and any downstream stage
can be re-run later from pristine upstream outputs. The file fully reconstructs
the cameras and skeleton, so results are portable without the original config.

Arrays use the [view-leading layout](../explanation/conventions.md#array-layouts)
in float64; `NaN` encodes missing observations / un-triangulated points.

```text
attrs["meta"]               json: {deeperfly_format_version: 2, created_utc, ...}
skeleton/                   point_names, limb_names, limb_id, bones, palette/
pose2d/
    points                  (V, T, P, 2)  arg-max 2D detections (visibility-masked)
    conf                    (V, T, P)     detection confidences
    cameras/                the config rig as built at detect time
    attrs["image_sizes"]    json {camera_name: [h, w]} of the raw footage frames
    candidates/             top-K peaks (xy, score) -- only if pictorial_structures
                            was enabled at detect time
bundle_adjustment/
    cameras/                the BA-refined rig
pictorial_structures/
    points                  (V, T, P, 2)  PS-corrected 2D
    points3d                (T, P, 3)     initial 3D estimate
    reproj_error            (V, T, P)
triangulation/
    points                  (V, T, P, 2)  cleaned 2D (outlier-rejecting methods)
    points3d                (T, P, 3)
    reproj_error            (V, T, P)
```

A `cameras/` group stores `names`, `rvecs`, `tvecs`, `intrs` (`[fx, fy, cx, cy]`),
and `dists`. The `skeleton/` group stores `point_names`, `limb_names`, `limb_id`,
`bones`, and a `palette/` subgroup of limb → hex color.

Which groups are present depends on which stages ran. A group exists only once its
stage completed; only the stages that were enabled (and whose inputs were
available) appear.

### What the library reads back

`PoseResult.load(path)` assembles the **most-derived data present**, so you get
the best result without knowing which stages ran:

| Field | Preference order |
| --- | --- |
| `pts2d` | `triangulation` → `pictorial_structures` → `pose2d` |
| `pts3d` | `triangulation` → `pictorial_structures` |
| `reproj_error` | `triangulation` → `pictorial_structures` |
| `cameras` | `bundle_adjustment` → `pose2d` (config rig) |
| `conf` | `pose2d` |

`PoseResult.save(path)` is the library one-shot (no staged groups): it writes
`pts2d`/`conf` to `pose2d/` and, when a 3D pose is present, the 2D/3D/error to
`triangulation/`, so `load` round-trips the assembled view.

A file in an older schema version is rejected on `load` (re-run to regenerate)
and simply read as empty by the staged run (so it recomputes).

## `config.toml` (snapshot)

The exact config text that drove the run, copied byte-for-byte for
reproducibility. On a later run, `-c` wins when given (and refreshes this
snapshot); without `-c`, this snapshot is reused — so you can edit it in place and
re-run with just `-o`.

## `run.json`

A small JSON sidecar recording, per stage, the **fingerprint** (the
result-affecting config subset) and the completion time. It is outdir-local run
state, kept out of the portable `poses.h5`; deleting it merely recomputes
everything.

```json
{
  "format_version": 1,
  "stages": {
    "pose2d":        { "fingerprint": { "...": "..." }, "completed_utc": "2026-..." },
    "triangulation": { "fingerprint": { "...": "..." }, "completed_utc": "2026-..." }
  }
}
```

On a re-run a stage is reused only when its recorded fingerprint still matches the
current config **and** its output is present. Comparison is *subset* semantics: a
key dropping out of the expected fingerprint (e.g. `candidates` when
`pictorial_structures` is disabled again) does not invalidate the cache, while a
changed or newly-appearing key does. Performance-only knobs (`batch_size`,
`decode_buffer`, `[io.image]`) are deliberately excluded. Fingerprints are stored
verbatim (not hashed) so a mismatch can be reported as a readable diff. See
[caching and re-runs](../explanation/pipeline.md#caching-and-re-runs) for the full
model.
