---
hide:
  - toc          # no headings to list; give the freed column to the viewer
---

# Keypoint locations

deeperfly tracks a **38-point skeleton** on the fly. The packaged one is `fly38b`:
five points per leg (thorax–coxa, coxa–trochanter, femur–tibia, tibia–tarsus, claw)
for each of the six legs, one antenna per side, the **neck**, and a five-point
**dorsal-midline abdomen chain** — 30 + 2 + 1 + 5. The names and their ordering are
the [`[skeleton]`](../reference/configuration.md#skeleton) table (see also
[Conventions & glossary](conventions.md)).

The viewer below shows where those keypoints sit on the body. The fly is the
**NeuroMechFly** biomechanical model; the colored balls and connecting sticks are
deeperfly's keypoints, colored by limb (left = blue, right = red, lightening front →
hind; the midline neck and abdomen are green, since neither is a side structure, and
lighten the same way, neck → abdomen). Drag to orbit, scroll to zoom, and move the
joint sliders to see how each keypoint tracks the body as the pose changes. The
sliders start at the model's resting pose; under **Pose**, **Neutral** returns to it
and **Zero** sets every joint angle to 0. The **View** buttons snap the camera to the
seven rig angles (RH–LH) plus hind, bottom and top.

<iframe src="../../keypoints/viewer.html" title="Interactive NeuroMechFly keypoint viewer"
        loading="lazy" width="100%" height="720" style="border:1px solid var(--md-default-fg-color--lightest); border-radius:6px;">
</iframe>

[Open the viewer full-screen ↗](../keypoints/viewer.html){:target="_blank" rel="noopener"}

!!! note "Leg, claw, antenna and neck keypoints sit exactly on the model"

    Each leg keypoint is the *joint between two segments*, which coincides with a
    NeuroMechFly body origin, so those points sit exactly on the model; the claw is the
    distal tip of the fifth tarsal segment, each antenna sits at the pedicel–head joint,
    and the neck is the `c_thorax`–`c_head` pivot. Those four are model geometry, not a
    labeling choice — which is also why the neck cannot be dragged in the viewer: it is
    the center the head rotates *about*, so no head angle moves it. The same is true of
    the six thorax–coxa points on a tethered fly's fixed thorax.

!!! note "The abdomen is one midline chain"

    For annotation consistency the abdomen is labeled on the **top of its silhouette**,
    at the tergite stripes, and `abdomen0`…`abdomen4` run anterior to posterior along
    that dorsal midline. On the model, `abdomen0`–`abdomen3` sit directly above the four
    abdominal hinges — one per intersegmental boundary, which is where a stripe falls —
    and `abdomen4` is the tip marker on the last segment, there being no fifth hinge.
    Each rides the segment it sits on, so the chain follows the abdomen as it curls.

    `abdomen0` and `abdomen4` are exactly the historical DeepFly3D per-side markers
    collapsed onto the midline: the two sides differ only laterally, so the midpoint of a
    labeled pair *is* the midline point — which is how a `fly38` corpus migrates.

    These five are the only points with no exact NeuroMechFly counterpart — their heights
    were set by eye against the model's dorsal crest — and the viewer's legend calls them
    out. They mark a **labeling convention**, not model geometry: a real abdomen is not a
    scaled copy of this one, so expect a labeled point and the marker here to sit close
    rather than coincide.

!!! tip "Using these placements for inverse kinematics"

    Every placement above is **already baked in**: the packaged articulation carries
    `l_antenna` / `r_antenna` with `neck` as the head chain's
    [base landmark](../reference/configuration.md#ik-head-base), and `abdomen0`…`abdomen4`
    at exactly the bodies and offsets this page lists. A `fly38b` body plan therefore
    fits **38 of 38** points with no config at all. The asset bake reads this same
    `keypoints.json`, so what the viewer draws and what the IK fits cannot drift apart —
    and `scripts/build_keypoint_viewer_assets.py` prints the table when it rebuilds the
    viewer, so a placement change reports what this page should now say.

    Retarget a chain only if your labeling scheme differs, via
    [`[inverse_kinematics.abdomen]`](../reference/configuration.md#ik-markers) — and note
    that a table **replaces** its chain's whole marker set, so list every marker you
    track.

    Because these five markers sit on the abdomen's dorsal *surface* rather than on its
    hinges, the distance between neighbouring ones grows as the abdomen curls — the
    outside of a bend is longer. The
    [size estimate](../reference/configuration.md#ik-chain-size) is built only from
    separations the chain's joints cannot change, so a curled abdomen is not mistaken for
    a bigger one.

    The neck sits *on* the head hinge, so it constrains where the head chain is anchored
    rather than how it is rotated — which is exactly why the fit treats it as a base
    landmark and not as evidence about the head angles. Anchoring the head there instead
    of at the coxa registration's extrapolation is worth about 11° of head pitch; see
    [The head's base](../reference/configuration.md#ik-head-base). If you do redeclare
    `[inverse_kinematics.head]` for some other reason, carry the nomination over —
    `neck = { body = "c_head", offset = [0.0, 0.0, 0.0], base = true }` — because a table
    replaces its chain's whole marker set.

The model is rendered with [MuJoCo](https://mujoco.org/) compiled to WebAssembly,
running entirely in your browser — no data is uploaded. It is the
[NeuroMechFly v2](https://neuromechfly.org/) model from
[flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0); the bundled model and
keypoint mapping are generated by `scripts/build_keypoint_viewer_assets.py`, which
reads the skeleton from the packaged config so the page cannot drift from the library.
Pass `--skeleton fly38` to rebuild it for the historical DeepFly3D point set instead.
