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

    The packaged articulation still carries the `fly38` abdomen markers, so a `fly38b`
    body plan fits **32 of 38** points out of the box — the legs and antennae — and
    leaves the neck and the midline abdomen unfitted (nothing fails; they are simply
    absent from the plan). Retargeting it to the placements above, via
    [`[inverse_kinematics.head]` / `[inverse_kinematics.abdomen]`](../reference/configuration.md#ik-markers),
    brings that to **38 of 38**. Each table *replaces* its chain's markers, so list the
    antennae again alongside the neck:

    ```toml
    [inverse_kinematics.head]
    l_antenna = { body = "l_pedicel", offset = [0.0, 0.0, 0.0] }
    r_antenna = { body = "r_pedicel", offset = [0.0, 0.0, 0.0] }
    neck      = { body = "c_head",    offset = [0.0, 0.0, 0.0] }

    [inverse_kinematics.abdomen]
    abdomen0 = { body = "c_abdomen3", offset = [ 0.0,  0.0, 0.3   ] }
    abdomen1 = { body = "c_abdomen4", offset = [ 0.0,  0.0, 0.285 ] }
    abdomen2 = { body = "c_abdomen5", offset = [ 0.0,  0.0, 0.27  ] }
    abdomen3 = { body = "c_abdomen6", offset = [ 0.0,  0.0, 0.243 ] }
    abdomen4 = { body = "c_abdomen6", offset = [-0.23, 0.0, 0.2   ] }
    ```

    `scripts/build_keypoint_viewer_assets.py` prints this exact table when it rebuilds
    the viewer, so a placement change reports what this page should now say.

    The neck sits *on* the head hinge, so it constrains where the head chain is anchored
    rather than how it is rotated.

The model is rendered with [MuJoCo](https://mujoco.org/) compiled to WebAssembly,
running entirely in your browser — no data is uploaded. It is the
[NeuroMechFly v2](https://neuromechfly.org/) model from
[flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0); the bundled model and
keypoint mapping are generated by `scripts/build_keypoint_viewer_assets.py`, which
reads the skeleton from the packaged config so the page cannot drift from the library.
Pass `--skeleton fly38` to rebuild it for the historical DeepFly3D point set instead.
