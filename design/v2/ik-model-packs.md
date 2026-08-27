# Model packs: the IK stage stops being NeuroMechFly-shaped

**Status:** design, folded into v2 as workstream **W9**.
**Scope:** `inverse_kinematics/`, the `[inverse_kinematics]` table, the three baked
assets, `scripts/build_nmf_mesh_asset.py`, and the `nmf_*` names in `results.h5`, the
visualization ops and the GUI.
**Date:** 2026-08-27.

The goal: a run can fit **flybody** (or any other MJCF-derived fly model) instead of
NeuroMechFly by naming it, and nothing about the fly's anatomy or flygym's naming
conventions is spelled out in Python.

## The headline: the config does not grow

The natural fear is that supporting a second model means a large new config surface. It
does not. `[inverse_kinematics]` **loses one key**:

| v1 / current v2 draft | W9 |
| --- | --- |
| `template = "neuromechfly"` (leg chains only; the articulation asset is unreachable from the config, and the mesh is not selectable at all) | `model = "neuromechfly"` -- a *pack*: leg template + articulation + overlay mesh, selected as one unit, by name or path |
| `fit_head = true` / `fit_abdomen = true` -- two chain names hardcoded as booleans in the schema | `chains = ["head", "abdomen"]` -- omit for every chain the pack defines; `[]` fits legs only |
| `legs`, `bounds`, `markers`, solver knobs | unchanged |

Net: one key removed, one renamed and widened. The expansion is entirely in the **asset
format**, which is not config and not user-authored per run.

This is the same move v2 makes for the skeleton: the thing that varies moves into a
version-controlled file resolved by name, and the run config names it.

## What a pack is

A pack is a directory with a `model.toml` manifest pointing at three siblings. All three
files already exist for NeuroMechFly; the manifest and the selection mechanism are new.

```
src/deeperfly/data/models/neuromechfly/
    model.toml          # the manifest: name, rest axis, the three asset paths
    template.toml       # leg chains: joints, DOF axes, bounds, angle names   (exists)
    articulation.json   # non-keypoint chains + the registration anchors      (exists)
    mesh.npz            # the overlay mesh                                    (exists)
```

| File | Declares | Authored by |
| --- | --- | --- |
| `model.toml` | the pack's name, `rest_axis`, the registration **anchor bodies**, and where the other three live | by hand, ~15 lines |
| `template.toml` | per-leg serial chains: each joint's **model** joint name, its DOF axes / bounds / **angle names**, its **`quat`**, and each leg's **`side`** | by hand from the MJCF |
| `articulation.json` | chains whose joints are *not* keypoints (head, abdomen, and anything else): their ordered joints, the **body frames** a marker can attach to, and the leg spring references | baked from the MJCF by a script |
| `mesh.npz` | neutral vertices, faces, colors and the slot each vertex is posed by | baked from the MJCF by the same script |

**A pack names no skeleton point, and carries no offset.** Both are the *binding*'s -- the
adaptor between one skeleton and one model, keyed by the pair, described
[below](#the-binding-is-a-third-thing-and-it-should-say-so). That is the half of the
decoupling that does not exist today, and it moves two things out of the pack:

- **`articulation.json` loses its `markers` arrays.** They are binding rows -- `point` +
  `body` + `offset` + `depth` + a precomputed `neutral` -- sitting inside a model asset.
  What stays is the `bodies` map, which stops being optional: it is exactly what a
  binding's `(body, offset)` resolves against, and `neutral` becomes a derived value
  rather than a baked one. The "this asset predates `bodies`" error path goes with it.
- **The registration anchors become model bodies.** `coxa_points` today is
  `["lf_thorax_coxa", ...]` -- deeperfly skeleton names, in a NeuroMechFly asset, which is
  the violation in its purest form. A pack declares `anchors = ["lf_coxa", ...]` instead,
  their neutral frames come from `bodies`, and which tracked point observes each anchor is
  read back out of the binding (the row whose `body` is that anchor at offset zero). A
  skeleton that tracks only four of the six still registers, exactly as today.

Segment *lengths* stay out of the pack, as today: they are measured per recording from
`pts3d`, which is why the fit reprojects tightly on any animal.

## What has to stop being hardcoded

Nine things, all small, all with a specific site. The IK core is already far more generic
than its docstrings suggest -- `unfittable_branches`, `estimate_chain_scale`,
`calibrate_chain` and the whole solve are chain-name agnostic, and head/antennae appear
only in their comments.

| # | Hardcode | Site | Fix |
| --- | --- | --- | --- |
| 1 | **The DOF angle-name convention** `f"{joint}-{dof}"`, flygym's `<parent_body>-<child_body>-<dof>` scheme, written twice | [template.py:106](../../src/deeperfly/inverse_kinematics/template.py#L106), [bodyplan.py:434](../../src/deeperfly/inverse_kinematics/bodyplan.py#L434) | a DOF declares `angle = "coxa_abduct_{leg}"`; the current form stays the default, so the packaged template does not change |
| 2 | **A leg's side**, inferred as `leg.lower().startswith("l")` | [template.py:225](../../src/deeperfly/inverse_kinematics/template.py#L225) | declared: `legs = [{ name = "lf", side = "l" }, ...]`, with the bare string keeping the prefix rule. This one **fails silently** today -- a skeleton naming its legs `T1_left` gets side `"r"`, mirrored axes and the wrong bounds, with no error |
| 3 | **`REST_AXIS = [0, 0, -1]`**, the direction a segment extends along | [forward.py:47](../../src/deeperfly/inverse_kinematics/forward.py#L47) | `rest_axis` in `model.toml`. flybody's is `[0, 1, 0]` -- see below |
| 4 | **Identity `offset_quat` on every leg joint** | [bodyplan.py:421](../../src/deeperfly/inverse_kinematics/bodyplan.py#L421) | a per-joint `quat` in the template, threaded into the plan and into `forward.leg_fk`. The plan field already exists; it is hardwired. **The only structural change in W9** -- see below |
| 5 | **The six leg names** `["rf","rm","rh"]` / `["lf","lm","lh"]` | [align.py:153-156](../../src/deeperfly/inverse_kinematics/align.py#L153-L156) | **delete.** `r_body`, `head_origin`, `align.to_local` and `align.to_world` have **no production consumer** -- `bodyplan.py:40` records that the leg subtree stopped being rotated by `r_body`, and grep finds them only in `tests/test_inverse_kinematics.py`. This is dead state carrying the stage's only leg-name hardcode |
| 6 | **`coxa_points` / `coxa_neutral`** -- not just fly anatomy in the registration API, but **skeleton point names inside a model asset** (`["lf_thorax_coxa", ...]`) | [articulation.py:155-156](../../src/deeperfly/inverse_kinematics/articulation.py#L155-L156), `_coxa_world`, `_coxa_similarity`, `body_similarity`, the mesh's `coxa_idx` | the pack declares `anchors` as **model bodies**; the binding says which point observes each. Renames to `anchor_*` / `_anchor_similarity`. The solve is already generic (`good.sum() < 3`, not `== 6`) |
| 7 | **`fit_head` / `fit_abdomen`** as two named booleans | [config.py:620](../../src/deeperfly/config.py#L620), [:1188](../../src/deeperfly/config.py#L1188), [:1276](../../src/deeperfly/config.py#L1276), [results.py:497](../../src/deeperfly/results.py#L497), [:502](../../src/deeperfly/results.py#L502) -- 5 sites, all shallow | `chains = [...]`; `IKResult.head_scale` / `.abdomen_scale` become `chain_scale("head")` |
| 8 | **The articulation asset is unreachable from the config** -- `Articulation.load` takes a `ref`, `ik_articulation()` never passes one | [config.py:1282](../../src/deeperfly/config.py#L1282) | the pack supplies it |
| 9 | **`nmf_*` names**: 288 identifier occurrences over 34 distinct names (`nmf_angles`, `nmf_pts3d`, `has_nmf`, `skeleton_nmf`, `mesh_nmf`, ...) across `results.h5`, `visualization/compose.py`, the GUI server and 5 JS modules | `model_*` -- which is *already* the internal vocabulary (`IKResult.model_pts3d`, `plan.to_model`). Draw ops become `skeleton_model` / `mesh_model` |

Item 9 is the bulk of the churn and none of the risk: a rename with no behavior change.
It does rename `results.h5` datasets, so it wants to ride the v3 repack alongside
candidate C3 rather than force one of its own.

## What flybody actually needs

flybody is checked out at `/home/tlam/flybody`; its MJCF is
`flybody/fruitfly/assets/fruitfly.xml`. Reading it changes the estimate in both
directions.

**The good news: the leg topology is identical.** Per leg, thorax to tarsus:

| | NeuroMechFly | flybody |
| --- | --- | --- |
| ThC | `c_thorax-{leg}_coxa`, 3 DOF | `coxa_abduct_T1_left`, `coxa_twist_T1_left`, `coxa_T1_left` -- 3 DOF |
| CTr | `{leg}_coxa-{leg}_trochanterfemur`, 2 DOF | `femur_twist_T1_left`, `femur_T1_left` -- 2 DOF |
| FTi | `{leg}_trochanterfemur-{leg}_tibia`, 1 DOF | `tibia_T1_left` -- 1 DOF |
| TiTa | `{leg}_tibia-{leg}_tarsus1`, 1 DOF | `tarsus_T1_left` -- 1 DOF |
| distal tarsus | `tarsus2..5`, modelled as one rigid segment | `tarsus2..5`, modelled as one rigid segment |

Seven DOFs to the tarsus either way, in the same order, at the same anatomical joints.
flybody has no trochanter *body* -- its coxa attaches straight to the femur -- but that
boundary is anatomically the trochanter, so the fly38b `{leg}_coxa_trochanter` keypoint
still lands on a real joint and the mapping is 1:1. **No new topology machinery, and no
pseudo-joints.** The head is the same shape too (`head_abduct` / `head_twist` / `head`
against NeuroMechFly's three head DOFs); only the abdomen differs, 7 segments x 2 DOFs
against a 10-joint chain, which is already declarative.

**The bad news, and the reason W9 is not purely a rename:**

```
coxa_T1_left    pos = 0.0317 0.0209 -0.0272   quat = -0.532 0.787 -0.311 -0.0229
 femur_T1_left  pos = 0      0.0437  0        quat = 0 0 0.252 0.968
  tibia_T1_left pos = 0      0.0697  0        quat = 0.186 0.162 0.677 -0.694
   tarsus_T1_left pos = 0   -0.051   0.00175  quat = 0.039 0.998 0.00674 0.0474
```

Segments extend along **+y**, not `-z` (hardcode 3), and **every leg body carries a
non-identity quaternion** (hardcode 4). NeuroMechFly's leg bodies are all
`quat="1 0 0 0"`, which is exactly what licensed
[bodyplan.py](../../src/deeperfly/inverse_kinematics/bodyplan.py)'s "each leg subtree
gets an identity `offset_quat` and the template's axes apply verbatim". flybody does not
license it. So the template grows an optional per-joint `quat`, `_leg_joints` writes it
into the plan instead of `_IDENTITY_QUAT`, and `forward.leg_fk` carries the same rotation.
QuickIK already accepts the field; deeperfly just never fills it.

**What is left is the bake.** `scripts/build_nmf_mesh_asset.py` reads one hardcoded MJCF
(`docs/keypoints/assets/model/fly.xml`) and knows which bodies are the head chain, which
are the abdomen chain, which keypoints attach where, and that the coxae are the
registration anchors. Generalizing it means moving that knowledge into a per-model bake
spec. **This is the real work in W9 and it should not be undersold** -- the mechanism
below is maybe a third of the effort, the flybody bake spec the other two thirds, and it
needs the flybody meshes (cached at
`~/.cache/flygym_assets/flybody_fullsize_meshes_20260623a`) plus a MuJoCo run.

## Where a keypoint's placement on the model lives

Worth answering precisely, because it is the question that decides whether the skeleton
and the pack are really independent.

**Today, for the IK stage: `data/nmf_articulation.json`, in `chains[].markers[]`.** Each
marker is `{point, body, offset, depth, neutral}` -- the skeleton point, the model body it
is rigidly attached to, its offset in that body's frame, how many of the chain's joints
carry it, and the precomputed model-frame neutral position. `abdomen0` reads:

```json
{"point": "abdomen0", "body": "c_abdomen12", "offset": [-0.37, 0.0, 0.34],
 "depth": 2, "neutral": [-0.81163647, 0.0, 1.53218125]}
```

A run overrides any of it with `[inverse_kinematics.markers.abdomen]`, and
`Articulation.load` recomputes `neutral` as `body_frame @ offset` from the baked `bodies`
map -- so a marker moves without re-running MuJoCo.

**But that is 7 of 38 points**, and the placement of the other 31 is stated somewhere
else, in a different form. The full picture:

| Points | Where the placement lives | Form |
| --- | --- | --- |
| 30 leg points | `template.toml`'s `legs` x `suffix`, composed as `f"{leg}_{suffix}"` | a **name convention**: no body, no offset. It works because a leg keypoint *is* a joint of the model, so the plan needs only which joint |
| `l_antenna`, `r_antenna`, `neck`, `abdomen0..4` | `nmf_articulation.json` markers | body + offset + depth |
| **all 38** | `docs/keypoints/assets/keypoints.json` | body + offset + an `approximate` flag |

The third row is the real one. `keypoints.json` is the **complete skeleton-to-model
binding**, and `build_nmf_mesh_asset.py` already reads it
([:301](../../scripts/build_nmf_mesh_asset.py#L301)) rather than duplicating it -- so
there is one source of truth, and it is authored by `map_keypoint`
([build_keypoint_viewer_assets.py:239](../../scripts/build_keypoint_viewer_assets.py#L239))
out of three kinds of rule:

- **name convention** -- `LEG_SUFFIX_TO_BODY`, e.g. `femur_tibia -> {leg}_tibia`, offset zero;
- **geometry** -- `distal_tip_offset`, which *computes* the claw's offset from the mesh's
  most distal vertex;
- **hand-tuned tables** -- `MIDLINE_POINTS`
  ([:166-170](../../scripts/build_keypoint_viewer_assets.py#L166-L170)) and
  `ABDOMEN_POINTS`, the abdomen markers, which have **no exact NeuroMechFly counterpart**
  and carry `approximate = true`. Their comment records why each offset is what it is --
  the `-x` in every row, `abdomen2` moved onto a hinge, `abdomen3` and `abdomen4` split
  across two bodies so a joint lies between them.

So the answer to "where does this live" is: **in a docs asset, generated by a Python
function that needs flygym and MuJoCo, consumed by a bake script, and reaching the IK for
only the 7 points that are not legs.**

## The binding is a third thing, and it should say so

That layout is what stops the skeleton and the pack from being independent, in both
directions the question asks about:

- **The pack reaches into the skeleton's namespace.** `template.toml` composes skeleton
  point names from `legs` x `suffix`. A skeleton that names its points differently needs a
  different *template*, even for the same animal on the same model.
- **The skeleton cannot reach a second model.** Nothing declares `fly38` against flybody;
  the only binding that exists is baked into a NeuroMechFly asset and a docs file whose
  filename names neither.
- **The same fact is stated twice**, in two forms -- a leg point's placement as a name
  rule in the template, every point's placement as a row in `keypoints.json`.

The fix is to name the thing that already exists: the **adaptor between one skeleton and
one model**. Called a *binding* here for the key and the filename, because "adapter" is
already an overloaded word in a codebase with readers and pathways -- but it is the
adaptor, and it is the one artifact where a skeleton point name and a model body name are
allowed to appear together:

```
src/deeperfly/data/
    skeletons/fly38.toml               # points, edges, symmetries, colors. Names no model.
    models/neuromechfly/               # bodies, chains, DOFs, mesh.  Names no skeleton.
    bindings/fly38@neuromechfly.toml   # where each tracked point sits on that model.
```

```toml
# bindings/fly38@neuromechfly.toml -- the adaptor: fly38's points onto NeuroMechFly's bodies
skeleton = "fly38"
model = "neuromechfly"

[points]
lf_thorax_coxa    = { body = "lf_coxa" }                                 # offset defaults to 0
lf_femur_tibia    = { body = "lf_tibia" }
lf_claw           = { body = "lf_tarsus5", offset = [0.0, 0.0, -0.107] } # distal tip, computed once
neck              = { body = "c_head", base = true }                     # places the head, fits none of it
abdomen0          = { body = "c_abdomen12", offset = [-0.37, 0.0, 0.34], approximate = true }
```

One row per tracked point, one uniform schema: **where on the model does this point sit.**
Everything else is derived, which is what makes the schema uniform rather than two schemas
wearing a trenchcoat:

- offset `[0,0,0]` on a body whose origin is its parent joint means the point **is** that
  joint, so it becomes a fitted plan joint with the pack's DOFs -- today's leg case;
- anything else means the point is rigidly carried by that body, so it becomes a zero-DOF
  pseudo-joint at the body's chain depth -- today's marker case. `depth` stops being
  authored, because the pack knows how deep each body is.

One row needs a third field, and it is worth stating because the derivation above would
otherwise get it wrong. `neck` sits at offset zero on `c_head`, whose origin is the
`c_thorax-c_head` pivot -- so the rule makes it a fitted joint. It must not be. The head's
three DOFs are *rotations about that very point*, so `neck` constrains none of them, and
nominating it `base = true` is what places the pivot far better than the anchor
registration can (that fit extrapolates 4.7x along the anchors' worst-determined axis and
lands ~11 degrees of pitch off) **without** counting as evidence the head was observed.
That nomination already exists on `[inverse_kinematics.markers.head]`; the binding carries
it, and the derivation reads it before deciding. A point at a chain's own origin is the
one case where "is it a joint" is not a geometric question.

The invariant is greppable, which is the point: **no skeleton point name appears anywhere
in a pack, and no model body name appears anywhere in a skeleton.** A test can assert it.

Everything that today states a placement moves into it, and nothing else does:

| Moves in from | What | Was |
| --- | --- | --- |
| `template.toml` | the 30 leg points | `legs` x `suffix` composed as `f"{leg}_{suffix}"` -- a name convention, offsets implicit |
| `nmf_articulation.json` | the 7 head/abdomen markers, **offsets included** | `chains[].markers[]`, baked into a model asset |
| `nmf_articulation.json` | which point observes each registration anchor | `coxa_points`, skeleton names in a model asset |
| `keypoints.json` | the `approximate` flag, and the committed offsets themselves | a docs asset, read by a bake script |
| `build_keypoint_viewer_assets.py` | `MIDLINE_POINTS`, `ABDOMEN_POINTS`, `LEG_SUFFIX_TO_BODY` | hand-tuned tables in a Python script needing flygym + MuJoCo |

The offsets are the substance of it. `abdomen0 = { body = "c_abdomen12", offset = [-0.37,
0.0, 0.34], approximate = true }` is a statement about *how fly38 was labelled against
NeuroMechFly*, and it is true of neither fly38 alone nor NeuroMechFly alone. Binding fly38
to flybody means writing five different numbers there and changing nothing else; relabelling
the abdomen means changing them for every model. That is exactly the axis the file is keyed
on.

Consequences:

- `template.toml` loses `suffix`; a joint names its own model joint and the binding says
  which point observes it. `legs = ["lf", ...]` in a run config then selects the pack's
  own leg ids.
- `[inverse_kinematics] binding` resolves `<skeleton>@<model>` by default, so a normal run
  still writes nothing. An unbound pair fails at load naming both halves, instead of
  degrading into all-NaN observations.
- `keypoints.json` becomes **derived** from the binding plus the pack, not its source, and
  the viewer builder stops being where a placement is decided. `map_keypoint`'s three rule
  kinds collapse into: name conventions expand when the binding is *generated*, geometry
  is resolved once and frozen into a number, hand-tuned rows are just rows.
- `[inverse_kinematics.markers.<chain>]` keeps working unchanged -- it is a per-run patch
  over the binding, which is exactly what it already is.

### What `approximate` should and should not do

`abdomen0..4` are the reason the flag exists: they are dorsal-midline points a human can
label on video, and NeuroMechFly has no counterpart for them, so their placement is a
*modeling decision* whose error is not the animal's. Today the flag reaches the docs
viewer and stops there.

It should reach the IK, and be **carried and reported, not acted on**. Separating the
residual of approximate markers from exact ones costs nothing and tells a reader which
part of an abdomen residual is the fit and which is the retarget. Down-weighting them, or
freeing their offsets as fitted parameters, is a tempting next step and is deliberately
*not* proposed here -- there is no measurement saying it helps, and a fitted offset would
absorb real error into the retarget. That stays a candidate.

### What this costs

- It is wider than the rest of W9: it touches `template.toml`'s schema, `align.py`'s
  `leg.joints[0].point`, `bodyplan`'s `x-deeperfly-point`, and both bake scripts.
- **A binding is not fully hand-authorable.** The claw offset is computed from mesh
  geometry, not chosen, so a new binding needs a generator that expands the conventions
  and resolves the geometry, leaving only the ambiguous rows for a human. Without that,
  binding a new skeleton means hand-editing numbers against a mesh -- so the generator
  (`deeperfly ik bind <skeleton> <model>`, emitting a binding to review) is part of the
  work, not a follow-up.
- The packaged `fly38@neuromechfly.toml` must reproduce today's `keypoints.json` and
  today's `nmf_articulation.json` markers exactly. That is the gate, and it is a strong
  one: both files are committed.

## What this does not buy

- **It is still a fly.** The pack format assumes serial revolute chains rooted at a set
  of body-fixed anchor points, and one similarity transform from world to model. A model
  with prismatic joints, a floating multi-root body, or no rigid anchor set does not fit.
  That limit is QuickIK's and the registration's, not the config's.
- **It does not make the two models comparable.** A NeuroMechFly angle and a flybody
  angle for "the same" joint are different numbers under different conventions. Nothing
  in W9 attempts a correspondence, and `results.h5` should keep the pack name beside the
  angles so a downstream reader cannot mix them.
- **A binding does not make a bad retarget good.** It makes the placement explicit and
  reviewable; whether `abdomen0` belongs on `c_abdomen12` at `[-0.37, 0, 0.34]` is still a
  judgement, and `approximate = true` is how it admits that.
- **It does not validate a pack against a skeleton by itself.** The binding is what makes
  the pair checkable at all: without one, a pack naming points the skeleton lacks degrades
  quietly into NaN observations. W9d turns that into a load error.

## Commits

| | Does | Gate |
| --- | --- | --- |
| **W9a** | Delete `r_body`, `head_origin`, `align.to_local`/`to_world` and their tests (hardcode 5). Pure removal, no consumer. | `test_inverse_kinematics` |
| **W9b** | `coxa_*` -> `anchor_*` through `articulation.py`, `__init__.py` and the mesh asset key, with the anchors becoming model *bodies* behind a name-convention shim (hardcode 6); `fit_head`/`fit_abdomen` -> `chains` (7). | `test_inverse_kinematics`, `test_ik_forward_bodyplan`, `test_config_schema` |
| **W9c** | Template gains `side`, per-DOF `angle`, per-joint `quat`; `model.toml` gains `rest_axis`; `bodyplan` and `forward` thread `offset_quat` and the rest axis (hardcodes 1-4). The packaged NeuroMechFly pack sets them to today's values, so the fit is **bit-identical** -- that is the gate. | `test_inverse_kinematics_quickik` (`test_the_leg_parameterisation_is_flygyms`), plus a fitted-angle comparison against a cached run |
| **W9d** | `model = "<name or path>"` resolving a pack manifest; `template` refused by name. | `test_config`, `test_config_schema` |
| **W9e** | `nmf_*` -> `model_*` across `results.py`, `visualization/`, `gui/` and the JS (hardcode 9). Rides the `results.h5` v3 repack with candidate C3. | full suite, `test_gui_server`, the headless GUI check |
| **W9f** | **The adaptor.** `bindings/<skeleton>@<model>.toml` carries every point's `body` + `offset` + `approximate` + `base`; `articulation.json` loses its `markers`, `template.toml` its `suffix`, and the W9b shim goes. A `deeperfly ik bind` generator expands the conventions and resolves the claw geometry. The widest commit in W9. | the generated `fly38@neuromechfly.toml` reproduces the committed `keypoints.json` and `nmf_articulation.json` markers **byte for byte**, the fit stays bit-identical, and the greppable invariant becomes a test |
| **W9g** | *Separate, and after v2 ships:* generalize the bake script and author the flybody pack. | a fit of flybody's own neutral keypoints returning flybody's own joint angles, the way `test_the_leg_parameterisation_is_flygyms` pins NeuroMechFly's |

W9a-W9f are independent of W1-W8 and touch no file they touch except `config.py` and
`results.py`. W9g is the one that needs MuJoCo and is deliberately outside the release.

W9a-W9e are worth doing even if W9f slips: they are net deletions plus one threaded field,
and they leave the pack selectable. But **W9f is the commit that answers the question** --
without it the offsets still live in a model asset and a docs script, and "fly38 on
flybody" remains unexpressible no matter how many packs exist.

## Relationship to the rest of v2

This is the fourth narrowing in the same spirit as the README's removals: the IK stage
currently states, in Python, facts about *one* animal model that belong in the asset that
model already ships. W9a and W9b are net deletions. W9c is the only place new capability
costs new code, and it is ~40 lines threading two fields that the plan format already has.

W9f is a different move, and closer to W1's. W1 takes the skeleton out of the run config
because it is a property of the detector, not the run; W9f takes the *placement* out of
both the skeleton and the model because it is a property of the pair. In each case the fact
already existed and was simply written down somewhere that could not hold it -- for the
skeleton, six copies across example configs; for the placement, a docs asset generated by a
script that needs MuJoCo.
