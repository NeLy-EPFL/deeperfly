# The v2 config schema

One part of [deeperfly v2](README.md) -- its visible face, and the reason most of the rest
of the refactor is possible. This document is the schema; [README.md](README.md) is what
v2 is, and [PLAN.md](PLAN.md) is how it lands.

One rule per section, no indirection, and the skeleton written out. Concretely:
[default_config.toml](default_config.toml) is the new packaged default (compare
`src/deeperfly/data/default_config.toml`), and [example_scape_E49.toml](example_scape_E49.toml)
is the hairiest example rewritten (compare `examples/scape_260810_E49_Fly3_002/config.toml`).

## What the schema is

The whole schema:

| Table | Says |
| --- | --- |
| (no skeleton table) | which points exist is the checkpoint's; `[skeleton] include` overrides |
| `[inverse_kinematics] model` | which mechanical model is fitted; its pack carries the chains, the anchors and the mesh |
| `[inverse_kinematics] binding` | where each tracked point sits on that model; defaults to `<skeleton>@<model>`, so a run normally omits it |
| `[inverse_kinematics.markers.<chain>]` | per point: where its marker sits on the fitted body |
| `[calibration]` | the solved rig, when there is one |
| `[default_camera]` | what every camera is unless it says otherwise |
| `[cameras.<name>]` | one camera: its footage, and where it differs |
| `[pipeline]` | which stages run |
| `[<stage>]` | that stage's knobs, table named exactly after the stage |
| `[visualization.default_video]` / `.default_layer` | what every video / every layer is unless it says otherwise |
| `[visualization.videos.<name>]` | one video: a `grid` of cells, `layers` drawn over them |
| `[io]`, `[gui]`, `[annotation]` | not stages: decode, editor, solver-in-the-GUI |

Three invariants carry most of the simplification:

1. **Detection is dense and one-to-one.** One detector, run once per camera, emitting
   every skeleton point for that camera: channel `i` is `[skeleton] points[i]`. So a
   camera IS a source IS a pathway IS a view, and only the camera is named.
2. **A name exists to be written once.** Nothing is declared in one table to be
   referenced from another. Every former reference is either inlined (the model, the
   crop, a video's styling) or keyed directly by the thing it belongs to
   (`[pose2d.crops]` is keyed by camera name).
3. **A table whose keys are names YOU choose holds nothing else**, and shared values
   live in a `default_<thing>` table beside the collection, holding *the same keys the
   thing itself takes*. `[cameras.*]` and `[pose2d.crops]` are pure name -> value maps:
   no reserved key, and no sub-table that could be mistaken for an entry.

   Three collections have shared values, and all three get the same treatment:

   | Collection | Its defaults |
   | --- | --- |
   | `[cameras.<name>]` | `[default_camera]` |
   | `[visualization.videos.<name>]` | `[visualization.default_video]` |
   | a video's `layers` | `[visualization.default_layer]` |

   The last two are the same wart the camera table had, found by looking for it:
   `[visualization]`'s bare keys (`background`, `crop`, `cell`, `output_fps`, `speed`)
   were all *per-video* keys serving as fallbacks, and the deleted `[visualization]
   kwargs` / per-video `kwargs` tables were per-layer style serving as fallbacks across
   three merge levels. Now each is one `default_*` table and one override site, and
   `[visualization]` itself holds nothing.

   Everything else in the schema is a flat table of knobs with no collection under it,
   so `default_*` applies exactly where a collection exists and nowhere else. The one
   near-miss: `[pose2d.crops]` is camera-keyed but has no default, because a camera with
   no entry means "full frame", which is an absence rather than a value.

## Where a table header is a schema choice, and where it is only syntax

`[pose2d.crops]` is a table rather than a key under `[pose2d]` because TOML has no other
way to write a multi-line map: an inline table cannot span lines. The two forms parse to
the *same data* --

    [pose2d.crops]                   [pose2d]
    rh = [737, 0, 960, 512]   ==     crops = { rh = [737, 0, 960, 512] }

-- so the schema says "crops is a map of camera to box" and the header is just the
readable spelling of it. Same for `[[postprocess.ops]]`, which is identical to
`ops = [{ op = "static" }, ...]`; a config may write either, and multi-line arrays take
comment lines between their entries, so nothing is lost by inlining a short one.

That gives one rule for the whole schema: **a list is a key, a map of names is a table**.
Long items get their own header, short ones stay inline.

What *is* a schema choice is whether a named thing carries its name as a key or IS the
key, and there the schema had it both ways. Fixed:

| Was | Now |
| --- | --- |
| `[[visualization.videos]]` with `name = "pose3d"` inside | `[visualization.videos.pose3d]` |
| `[inverse_kinematics.head]` / `.abdomen`, two fixed names in the stage's knob namespace | `[inverse_kinematics.markers.<chain>]` (the fitted body's chains, not the skeleton's) |
| `[inverse_kinematics] template`, naming the leg file only -- the articulation asset unreachable from the config, the overlay mesh not selectable at all | `[inverse_kinematics] model`, naming a **pack**: leg template + articulation + mesh as one unit ([ik-model-packs.md](ik-model-packs.md)) |
| `[inverse_kinematics] fit_head` / `fit_abdomen`, two chain names hardcoded as booleans | `[inverse_kinematics] chains = ["head", "abdomen"]`; omit for every chain the pack defines |
| `[pose2d.crop]`, whose value was a box **or** the string `"auto"` | `[pose2d.crops]` (always a box) + `[pose2d] auto_crops = ["f", "h"]` |

Keying the videos by name also makes two duplicate `name`s a TOML error rather than two
videos racing to write the same `.mp4`, and it drops a line from every video. Splitting
the searched cameras out of `crops` leaves one value type in the map -- and makes a
*seeded* search fall out for free (a camera in `auto_crops` with a box in `crops`
searches from that box), which in v1 needed a third form,
`{ op = "crop", auto = true, x = ..., y = ... }`.

## The skeleton: out of the config, and four things

A skeleton is **points, edges, symmetries, colors**. Nothing else -- no groups, no
chains, no names for subsets of points. It lives in a version-controlled file of its own
([fly38.toml](fly38.toml)), and a run config normally says *nothing at all* about it:
which points exist and in which order is a property of the checkpoint, whose artifact
already carries the point names (`hrnet.py`, `mvt.py`) and whose loader already refuses a
config that disagrees ([stream.py:80-123](../../src/deeperfly/pose2d/stream.py#L80-L123)).
You cannot pick an animal your detector does not detect. `[skeleton] include = "fly38"`
is the override, not the norm.

Getting here took four revisions, and each one deleted something:

1. `point_names` was the concatenation of the chains -- so rearranging what was also the
   color-grouping table silently changed which channel was which. **`points` is written
   out.**
2. Symmetry was chain pairs (4 rows), which needed equal-length chains and positional
   correspondence to say something about points. **`symmetries` is point pairs**, v1's
   grammar, and the safety of 16 rows comes from a check instead: the permutation must be
   an **automorphism of the edge set** -- apply it to `edges` and you must get `edges`
   back. A row carrying the wrong side, or two joints of one leg exchanged, breaks that.
   This check is strictly stronger than the chain-consistency one it replaces, and it
   needs no chains.
3. One-point chains (`neck = ["neck"]`) existed only to be a `colors` key. **Gone.**
4. Chains themselves were a compaction: 7 rows instead of 28, plus a name three other
   sections referred to. Once the skeleton is a file rather than a config section, 28 rows
   cost nothing -- and a chain can only express a *path*, so it could not attach an
   antenna to a head or link the neck to the first tergite. **`edges` is the whole
   topology**, which is what SLEAP and DeepLabCut both declare directly.

What the chain name was doing for the rest of the schema is now done by a **point
selector**: an entry in any point set may be a name or a `*` pattern.

    [bundle_adjustment] points = ["lf_*", "lm_*", "lh_*", "rf_*", "rm_*", "rh_*"]   # 30 points
    static             points = ["neck", "*_thorax_coxa"]                           # 7 points
    symmetrize         points = ["l*_thorax_coxa"]                                  # 3 points
    [skeleton.colors]  "lf_*" = "#0f7399"                                           # 5 points

One mechanism, four users, and it needs the skeleton to declare no groups at all -- which
is the point: a group existed only so another section could name it. Two patterns
matching one point is an **error naming both**, not a precedence rule; an exact name beats
a pattern; a pattern matching nothing is an error (it is always a typo). The resolved set
is logged, because a pattern that matches too much is the one failure a selector cannot
detect for itself.

Colors are per point, and an edge takes the color of the point it is written **from**, so
one table colors both. The table is optional: without it, points take a colormap by index
(DeepLabCut's default). Fully independent per-edge colors are still not expressible --
an edge has no name -- and neither reference tool offers them; DeepLabCut's answer is one
flat color for every bone, which here is a layer's `bone_color`.

Colors stay with the skeleton rather than with the render stage: `results.h5` stores them
inside the skeleton group, and the editor needs them with `[pipeline] visualization =
false`. The line that holds is keyed-by-point-name -> skeleton, keyed-by-draw-op ->
`[visualization]`, which is why layer style (`line_thickness`, `point_radius`,
`line_dash`, `alpha`, `bone_color`) sits in `[visualization.default_layer]`.

**What this costs in code.** `Skeleton` loses `limb_names`, `limb_id`, `n_limbs` and the
limb-keyed `palette`, and gains a per-point color array; `results.h5` stops writing
`limb_names` / `limb_id` (a reader falls back to a colormap for an old file, since colors
are cosmetic). The editor's legend can no longer group by limb name -- it groups by
distinct color instead, which for `fly38` is the same ten swatches, labeled by the
patterns' common prefix. `pictorial.skeleton_chains` is unaffected: it always derived
chains from the bone graph, which is exactly what says chains were never primitive.

## What SLEAP and DeepLabCut do, and what v2 takes from them

Both declare the same two things v2 does, which is reassuring, and neither declares the
third:

| | SLEAP (sleap-io YAML) | DeepLabCut (`config.yaml`) | v2 |
| --- | --- | --- | --- |
| points | `nodes:` (ordered) | `bodyparts:` (ordered) | `points` |
| topology | `edges:` -- explicit `source`/`destination` pairs | `skeleton:` -- explicit `[[a, b], ...]` pairs | `edges` -- the same |
| symmetry | `symmetries:` -- node pairs | none | `symmetries` -- point pairs |
| point color | not in the skeleton; a view palette by INDEX | `colormap:` over the bodypart index | `colors`, by name or pattern, optional |
| edge color | not in the skeleton | `skeleton_color:` -- ONE color for all bones | a layer's `bone_color` |

Three things taken:

1. **`edges`, and nothing else, for topology.** Both reference tools declare edges
   directly and have no chain concept at all, and `pictorial.skeleton_chains` already
   re-derived ours from the bone graph -- so chains were a compaction, not a primitive.
   With the skeleton in its own file the compaction buys nothing, and it cost
   expressiveness (a chain can only say *path*).
2. **`colors` is optional, with a colormap-by-index fallback** -- DeepLabCut's default,
   and what `_palette.py`'s TAB10 fallback already does per limb. So a hand-written
   skeleton names no colors at all, and the packaged palette exists only because a
   deliberate one reads better than a rainbow.
3. **`bone_color`: one flat color for every bone**, DeepLabCut's `skeleton_color`. This is
   the edge-color demand that is actually real -- colored joints on gray bones is the
   standard figure look -- and it is a layer style key, not skeleton data.

And one thing NOT taken: per-edge colors. Neither tool offers them, which is the best
evidence available that the `edge_colors` table stays documented-as-possible and unbuilt.

Sources: [sleap-io skeleton format](https://io.sleap.ai/v0.6.3/formats/slp/),
[sleap.skeleton](https://sleap.ai/_modules/sleap/skeleton.html),
[DeepLabCut user guide](https://deeplabcut.github.io/DeepLabCut/docs/main-workflows/user-guide.html),
[DeepLabCut plotting.py](https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/utils/plotting.py).

## What was removed, and what replaced it

| v1 | v2 |
| --- | --- |
| `[[sources]]` name + glob, referenced by pathways | `[cameras.<name>].video`, a regex over the recording's filenames |
| a list of `input` globs = alternate names, first match wins | alternates go inside the regex (`camera_(RH\|0)...`); a **list** now concatenates |
| several video files matched -> keep the first, warn | they are the camera's stream, decoded back to back |
| `[[pose2d.models]]` + `[pose2d].model` reference | `[pose2d].class` / `.weights` -- one detector per run |
| `[[pose2d.preprocessors]]` named op lists, referenced by pathways | `[pose2d.crops]`, keyed by camera: always a box |
| the skeleton, restated in every config | a version-controlled file, resolved from the detector |
| `[[pose2d.pathways]]` (name, source, preprocessor, model) | gone: implied by the camera table |
| `[pose2d.output_points.<view>]` (38 x V rows of channel -> point) | gone: channel `i` -> point `i` |
| `[skeleton] name = "fly38"` preset reference | no table at all: the checkpoint names its skeleton; `include` overrides |
| `[skeleton] point_names` | `points`, in the skeleton FILE -- the only ordered thing in it |
| `[skeleton] symmetries` (16 point pairs) | unchanged -- plus a check that they are an automorphism of `edges` |
| `[skeleton] limb_points` (10 entries, 3 of them one point) | `edges` (28 pairs): the topology, with no grouping concept |
| `[skeleton] limb_palette` (per limb, and edges took an endpoint's) | `colors`: per POINT, a `*` pattern as shorthand |

| `[cameras.defaults]` sub-table (a reserved camera name) | `[default_camera]`, a sibling table |
| `[cameras].calibration` | `[calibration].path`, its own table |
| `[cameras.<n>].mirror` (training-only, declared on both sides) | gone: no reader left, and `azimuth_deg` already says it |
| `[visualization]` bare keys as per-video fallbacks | `[visualization.default_video]` |
| `[[pose2d.preprocessors]]` `{ op = "crop", auto = true, x = ... }` | a box in `[pose2d.crops]` + the camera in `auto_crops` |
| `[pipeline] do_<stage> = true` | `[pipeline] <stage> = true` |
| `[visualization] kwargs` + per-video `kwargs`, keyed by draw op | per-layer style, inline in the layer |
| per-video `plot` + `stage` + hand-offset `panels` | `layers = [{ draw, stage, ...style }]`, drawn in order |
| `[bundle_adjustment].points_to_use` (30 names) | `points = ["lf_*", ...]` -- six patterns |
| `symmetrize` `pairs` (restating the skeleton's pairs) | `points`: name either half, `symmetries` says the partner |

Deliberately kept: the orbit vocabulary, the calibration winning over it, the
`static`/`symmetrize` op chain, every `*Params` knob and its default, and the two
narrowings (missing footage, uncovered views) -- they are what makes one config run a
recording that has less rig than it declares.

The op chain is kept against the *appearance* of being redundant. The IK's
`body_alignment` takes the same per-recording medians of the same seven body-fixed
points, but it bakes them into the body plan as joint offsets, where these ops write a
corrected `points3d` -- and only the corrected pose exists on an install without the
optional Rust solver. See W10 in [PLAN.md](PLAN.md).

## What it costs

- **Sparse detection dies.** The 19-channel DeepFly3D-era plan -- one body side, front
  camera run twice through a `fliplr`, 132 mapping rows -- is not expressible. It
  survives today only as `tests/data/fly38_sparse_config.toml`; the tests written
  against it (partial per-view visibility, the mirrored-pathway left/right check) need
  new fixtures or deletion. `check_mirror_consistency` goes with it: with no mirrored
  pathway there is nothing to be inconsistent.
- **Frame ops other than crop die.** `fliplr` / `flipud` / `rot90` / `resize` as
  per-pathway preprocessing. Nothing in the repo uses them outside the sparse fixture;
  the model resizes to its own `input_size`.
- **Two pathways over one source die** (the front camera detected twice).
- **`[cameras.<n>].mirror` dies**, with `Config.mirror_views()` and
  `tests/test_config_mirror.py`. It named the camera that sees this one's mirror image,
  for flip augmentation during training: a mirrored sample must carry the id of the
  camera it now *looks like*, or an ipsilateral/contralateral split calls every swapped
  channel by the wrong side. That is a real thing to know -- but the in-repo consumer
  (`deeperfly.training`) went in 0.2.0 and only the key stayed, and dfpose, the trainer
  that exists, hardcodes its own table and has never read this one. Should a reader
  appear, the pairing is the camera at the negated `azimuth_deg`, which every v2 camera
  declares and which, unlike the extrinsics, is known before there is a calibration.
- **Every existing config, and every output-dir snapshot, stops loading.** Fingerprints
  change too, so a first run under v2 recomputes from `pose2d` down.

## How it is built

The internal domain objects do not change shape: `Config` still hands out a
`DetectionPlan`, a `CameraGroup`, a `Skeleton` and a `list[VideoSpec]`. v2 is a new
*surface* over the same objects -- the plan is synthesized (one source, one
preprocessor, one identity-mapped pathway per camera), and `layers` expand into the
`Panel` list `_expand_grid` already produces. The one domain object that does change is
`Skeleton`, which loses its limb fields (see above). So `pipeline/`, `results.py`, the GUI and
the fingerprints keep working on the same values, and the churn is confined to the
parsers: `config.py`, `pose2d/pathways.py`, `cameras.py`, `skeleton.py`,
`visualization/compose.py`, plus `project.py`'s fragment composition (`RIG_TABLES`
becomes `("default_camera", "cameras", "io")`, with `[calibration]` injected as its own
fragment) and the config-writing CLI paths.
