"""Bake the NeuroMechFly overlay assets (mesh + articulation) for inverse kinematics.

The ``inverse_kinematics`` stage fits a NeuroMechFly-style model to the 3D pose and
can overlay that model -- as a reprojected skeleton and as the shaded body mesh --
onto the 2D views. Rather than carry MuJoCo/flygym at runtime, this *one-time*
script reads the same self-contained model the docs keypoint viewer ships
(``docs/keypoints/assets/model/fly.xml`` + its simplified ``*.stl`` meshes, built
by ``scripts/build_keypoint_viewer_assets.py``) and bakes two compact, runtime-only
assets:

``data/nmf_mesh.npz`` -- every mesh's neutral-pose vertices in world coordinates,
its faces and color, and which "slot" poses it:

- a **leg segment** (coxa / trochanter+femur / tibia / tarsus), posed by mapping
  its two endpoint keypoints (from the fit) onto the segment's neutral endpoints;
- an **articulated node** of the head or abdomen, posed by the fitted joint angles
  through the baked chain forward kinematics (so the head turns / abdomen curls); or
- the rigid **body** (thorax / wings / halteres), posed by one global similarity
  transform fit from the six thorax-coxa keypoints.

``data/nmf_articulation.json`` -- the head and abdomen kinematic chains the IK
solver fits: each chain's ordered revolute joints (neutral world anchor + axis +
angle name + default bounds) and its markers (the tracked keypoints rigidly
attached at a given depth in the chain, with the attachment ``body`` + ``offset``
they were built from), plus a ``bodies`` map of every chain body's neutral world
frame (so a run config can move a marker to a new offset without re-running this
script -- see ``Articulation.load``), and the neutral thorax-coxa positions that
register the whole rigid body. The chain forward kinematics is a serial
product of rotations about the neutral anchors -- it reproduces MuJoCo's frames
exactly (validated to ~1e-16), so the baked angles and overlay are consistent.

So the overlay + head/abdomen IK need only the fitted joints/angles plus these
assets -- no MuJoCo, no MJCF, no STL parsing at runtime. Run it by hand (not in CI)
whenever the bundled model changes::

    uv run --with mujoco --python 3.12 python scripts/build_nmf_mesh_asset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "keypoints" / "assets"
OUT_MESH = REPO / "src" / "deeperfly" / "data" / "nmf_mesh.npz"
OUT_ARTIC = REPO / "src" / "deeperfly" / "data" / "nmf_articulation.json"

# deeperfly skeleton order (must match Skeleton.fly().point_names / keypoints.json).
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
JOINTS = ["thorax_coxa", "coxa_trochanter", "femur_tibia", "tibia_tarsus", "claw"]
# mesh-name part -> the leg segment index it belongs to (its bone spans
# keypoints[seg] -> keypoints[seg + 1]).
PART_SEGMENT = {
    "coxa": 0,
    "trochanterfemur": 1,
    "tibia": 2,
    "tarsus1": 3,
    "tarsus2": 3,
    "tarsus3": 3,
    "tarsus4": 3,
    "tarsus5": 3,
}

# The articulated head/abdomen chains, in the order the assets index them
# (chain 0 = head, chain 1 = abdomen). Each joint is a MuJoCo hinge whose neutral
# world anchor + axis are read from the model; ``angle`` is the fitted DOF's name
# and ``bounds`` its default limits in degrees (overridable from the run config).
# ``markers`` lists the tracked keypoints rigidly attached at a chain depth
# (number of proximal joints that move them). ``base_point`` names the marker that
# measures where the chain's base sits on the real animal -- the counterpart of a leg's
# thorax-coxa, without which the chain is placed by the coxa registration alone.
HEAD_CHAIN = {
    "name": "head",
    "joints": [
        {
            "joint": "c_thorax-c_head-yaw",
            "angle": "c_thorax-c_head-yaw",
            "bounds": [-60, 60],
        },
        {
            "joint": "c_thorax-c_head-pitch",
            "angle": "c_thorax-c_head-pitch",
            "bounds": [-60, 60],
        },
        {
            "joint": "c_thorax-c_head-roll",
            "angle": "c_thorax-c_head-roll",
            "bounds": [-60, 60],
        },
    ],
    # `neck` is the head-thorax pivot: it sits ON `c_head`'s origin, which is exactly
    # this chain's rotation anchor, so its depth is 0 (no head DOF can move a point on
    # the rotation center) and it constrains no angle. What it measures is *where the
    # pivot is* -- which the six near-coplanar coxae cannot say: they are singular along
    # dorsal (0.693 / 0.325 / 0.085) and the neck sits 0.404 above their centroid, a
    # 4.7x extrapolation. Across the 55-recording corpus that puts the fitted pivot a
    # median 0.125 model units too dorsal (29% of the head's own radius, same sign in
    # every recording), which the antennae then absorb as ~11 deg of spurious pitch.
    "markers": [("l_antenna", 3), ("r_antenna", 3), ("neck", 0)],
    "base_point": "neck",
    # model bodies whose meshes ride this node (the whole head, at chain depth 3).
    "node_root": "c_head",
    "node_depth": 3,
}
# The abdomen joints are limited to bend *downwards only* (ventral flexion), up to
# 30 deg per hinge: the markers (a few points near the dorsal midline) under-constrain
# the 5-DOF sagittal chain, so symmetric limits let the solver fold it into a
# non-physical zig-zag. A downward-only, monotone range keeps the fit a smooth ventral
# curl. The model's +pitch raises the tip (dorsal), so "down" is the negative range.
#
# Re-measured after the fly38b retarget, because five midline markers might have
# constrained the chain better than fly38's six paraxial ones did. They do not: opened to
# [-90, 30] the fit averages 2.72 sign changes along the five hinges -- the fold, not a
# curl -- for a residual gain of about a third. So the range stays a REGULARIZER, and it
# binds. Widening the ventral side does not help (it is the wrong end); over [-45,0],
# [-60,0] and a [-60,0] waist the residual moves 0.7% and the pinned fraction 2%.
#
# Re-measured again after the markers were re-anchored one segment proximal (below) and
# the lateral DOF added. The bend is no longer all at the waist: over 287 frames of one
# recording the fitted pose runs about [-19, -0.5, -1, -8, -10] deg, so the waist and the
# two distal hinges share the curl while hinges 1-2 sit against the UPPER bound wanting to
# go dorsal (pinned 89% and 22% of frames; the other three do not pin at all). Expect
# `_warn_about_pinned_limits` to name those two on every run. That pinning is NOT what the
# old anchoring's abdomen residual was: relaxing these bounds to [-60, 30] moved the
# five-marker residual by +3% (it got slightly worse, redistributing between markers),
# while re-anchoring moved it -42%. A saturated limit can be a symptom of geometry
# elsewhere rather than a cause -- the lateral wall, pinned 54% of frames before, stopped
# pinning entirely once the geometry was right, without being touched.
ABDOMEN_PITCH_BOUNDS = [-30, 0]
# Each hinge also gets a second DOF, so the abdomen can swing sideways as well as curl
# ventrally. A tethered fly does swing it laterally, and a pitch-only chain has to absorb
# that in the one place it can -- the root placement -- which drags the whole chain off the
# midline instead of bending it. The range is half the pitch's: lateral swing is the
# smaller motion, and a wide one lets it stand in for the ventral curl.
#
# That DOF is flygym's `-roll`, NOT its `-yaw`. The names are anatomically rotated on this
# chain, and it is worth spelling out because both alternatives are plausible-looking and
# only one is observable. Measured off the MJCF at the neutral pose, every abdomen hinge's
# axes are, in the child segment's own frame:
#
#     -pitch  ->  local y  ->  sagittal bend (the ventral curl)     <- wanted
#     -roll   ->  local z  ->  lateral swing                        <- wanted
#     -yaw    ->  local x  ->  axial TWIST about the long axis      <- unusable
#
# The twist is excluded on purpose, and the reason is ALIASING rather than invisibility.
# Every marker on this chain lies in the sagittal plane, and such a point is swung sideways
# by both a twist about local x and a bend about local z -- the two differ only in lever arm
# (the marker's height above the axis, against its distance behind the hinge). Measured on
# this model the two displacement fields agree to cos 0.87-0.97 per hinge, i.e. they are
# only 14-29 deg apart, so a twist DOF would largely repeat the lateral one and the fit
# would split the motion between them arbitrarily. The markers are NOT on the twist axis --
# they clear it by 0.22-0.34 model units, and a 20 deg twist moves them 41-88% as far as
# the same lateral bend does -- so it is the near-collinearity that rules the twist out,
# not a vanishing lever arm. `scripts/build_keypoint_viewer_assets.py` withholds its slider
# for the same reason, with the same numbers.
#
# Reading a `-roll` axis off the model rather than writing `[0, 0, 1]` by hand also keeps
# the 13-15 deg of segment pitch: each segment's local z is tilted out of world dorsal by
# its own place in the resting curl, and the hand-written vector loses that.
ABDOMEN_LATERAL_BOUNDS = [-15, 15]
ABDOMEN_CHAIN = {
    "name": "abdomen",
    "joints": [
        {
            "joint": "c_thorax-c_abdomen12-pitch",
            "angle": "c_thorax-c_abdomen12-pitch",
            "bounds": ABDOMEN_PITCH_BOUNDS,
        },
        {
            "joint": "c_thorax-c_abdomen12-roll",
            "angle": "c_thorax-c_abdomen12-roll",
            "bounds": ABDOMEN_LATERAL_BOUNDS,
        },
        {
            "joint": "c_abdomen12-c_abdomen3-pitch",
            "angle": "c_abdomen12-c_abdomen3-pitch",
            "bounds": ABDOMEN_PITCH_BOUNDS,
        },
        {
            "joint": "c_abdomen12-c_abdomen3-roll",
            "angle": "c_abdomen12-c_abdomen3-roll",
            "bounds": ABDOMEN_LATERAL_BOUNDS,
        },
        {
            "joint": "c_abdomen3-c_abdomen4-pitch",
            "angle": "c_abdomen3-c_abdomen4-pitch",
            "bounds": ABDOMEN_PITCH_BOUNDS,
        },
        {
            "joint": "c_abdomen3-c_abdomen4-roll",
            "angle": "c_abdomen3-c_abdomen4-roll",
            "bounds": ABDOMEN_LATERAL_BOUNDS,
        },
        {
            "joint": "c_abdomen4-c_abdomen5-pitch",
            "angle": "c_abdomen4-c_abdomen5-pitch",
            "bounds": ABDOMEN_PITCH_BOUNDS,
        },
        {
            "joint": "c_abdomen4-c_abdomen5-roll",
            "angle": "c_abdomen4-c_abdomen5-roll",
            "bounds": ABDOMEN_LATERAL_BOUNDS,
        },
        {
            "joint": "c_abdomen5-c_abdomen6-pitch",
            "angle": "c_abdomen5-c_abdomen6-pitch",
            "bounds": ABDOMEN_PITCH_BOUNDS,
        },
        {
            "joint": "c_abdomen5-c_abdomen6-roll",
            "angle": "c_abdomen5-c_abdomen6-roll",
            "bounds": ABDOMEN_LATERAL_BOUNDS,
        },
    ],
    # The five dorsal-midline tergite stripes of `fly38b`. Unlike every other tracked
    # keypoint these are not model joints: they are points on the abdomen's dorsal
    # SURFACE, whose body + offset were chosen in `docs/keypoints/assets/keypoints.json`
    # so the labeling reference draws them where a human sees the stripes. `point_meta`
    # reads that same file, so the offsets here and the ones the docs viewer shows can
    # never disagree. Each marker's depth is its attachment body's -- asserted below
    # against the model's own tree rather than trusted.
    #
    # No `base_point`: the chain's root anchor (c_thorax-c_abdomen12) is buried inside
    # the thorax, and no keypoint sits on it. The abdomen is therefore placed by the
    # coxa registration -- see `_chain_offsets`, which shares the head's measurement
    # because the two anchors sit at the same dorsal height on the same rigid thorax.
    # `keypoints.json` anchors each stripe one segment PROXIMAL of where fly38b's first
    # retarget put it, and gives every offset an -x component: the stripes sit on the
    # dorsal surface of the segment BEHIND the intersegmental fold, not in front of it.
    # The old anchoring put abdomen3 and abdomen4 on the same body (c_abdomen6), which
    # made their separation rigid -- 18% short of a measured fly and unreachable by any
    # angle, the single largest abdomen residual. One joint now lies between them.
    "markers": [
        ("abdomen0", 2),
        ("abdomen1", 4),
        ("abdomen2", 6),
        ("abdomen3", 8),
        ("abdomen4", 10),
    ],
    "base_point": None,
    # abdomen segment body -> chain depth, used to assign each segment mesh a node and
    # to check every marker's declared depth. Depth counts CHAIN JOINTS proximal to the
    # body (the head's `node_depth` = 3 for its three DOFs at one anchor), so two per
    # hinge here -- pitch and yaw.
    "segment_depth": {
        "c_abdomen12": 2,
        "c_abdomen3": 4,
        "c_abdomen4": 6,
        "c_abdomen5": 8,
        "c_abdomen6": 10,
    },
}
CHAINS = [HEAD_CHAIN, ABDOMEN_CHAIN]


def main() -> None:
    # Disable fusestatic so every keypoint-bearing body keeps its own frame.
    xml = (ASSETS / "model" / "fly.xml").read_text()
    xml = xml.replace('fusestatic="true"', 'fusestatic="false"')
    model = mujoco.MjModel.from_xml_string(xml, _asset_dict())
    data = mujoco.MjData(model)

    pose = json.loads((ASSETS / "pose.json").read_text())
    data.qpos[:] = np.asarray(pose["neutral_qpos"], dtype=float)
    mujoco.mj_forward(model, data)

    geom_rgb = np.asarray(json.loads((ASSETS / "colors.json").read_text())["geom_rgb"])
    kp = json.loads((ASSETS / "keypoints.json").read_text())
    point_names = [p["name"] for p in kp["points"]]
    kp_index = {name: i for i, name in enumerate(point_names)}
    # Per-point attachment body + offset (the labeling-scheme choice: where each
    # keypoint sits relative to its NeuroMechFly body). Baked alongside each marker
    # so a run config can move a marker to a new offset without re-running MuJoCo.
    point_meta = {
        p["name"]: {"body": p["body"], "offset": list(p["offset"])}
        for p in kp["points"]
        if "body" in p and "offset" in p
    }

    # Neutral world keypoints (body world pose * offset), in skeleton order.
    kp_neutral = np.full((len(point_names), 3), np.nan)
    for i, p in enumerate(kp["points"]):
        try:
            bid = model.body(p["body"]).id
        except KeyError:
            continue
        kp_neutral[i] = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.asarray(
            p["offset"]
        )

    _write_mesh(model, data, geom_rgb, kp_index, kp_neutral, point_names)
    _write_articulation(model, data, kp_neutral, point_names, point_meta)


def _part_for(mesh_name: str) -> str:
    """Coarse body part a geom's mesh belongs to (for the overlay hide-by-part lists).

    Maps each NeuroMechFly geom mesh name to one of a small, stable vocabulary --
    ``wings``, ``halteres``, ``eyes``, ``antennae``, ``head``, ``thorax``,
    ``abdomen``, ``legs`` -- so a config list like ``["wings"]`` can drop those faces
    from the rendered overlay (the video and the GUI carry their own list).
    """
    seg = mesh_name.lower().rsplit("_", 1)[
        -1
    ]  # 'l_wing'->'wing', 'lf_tarsus1'->'tarsus1'
    if seg == "thorax":
        return "thorax"
    if seg in ("head", "rostrum", "haustellum"):
        return "head"
    if seg == "eye":
        return "eyes"
    if seg in ("pedicel", "funiculus", "arista"):
        return "antennae"
    if seg == "wing":
        return "wings"
    if seg == "haltere":
        return "halteres"
    if seg.startswith("abdomen"):
        return "abdomen"
    if seg in ("coxa", "trochanterfemur", "femur", "tibia") or seg.startswith("tarsus"):
        return "legs"
    return "other"


def _write_mesh(model, data, geom_rgb, kp_index, kp_neutral, point_names) -> None:
    """Bake ``nmf_mesh.npz``: neutral verts/faces/colors + each mesh's posing slot."""
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    face_rgb: list[np.ndarray] = []
    vert_slot: list[np.ndarray] = []
    vert_part: list[np.ndarray] = []  # per-vertex body-part index (into part_names)
    part_names: list[str] = []
    part_index: dict[str, int] = {}
    # slot 0 = rigid body. Leg-bone slots carry (prox, dist) keypoints; node slots
    # carry (chain, depth); the other pair is -1.
    slot_prox = [-1]
    slot_dist = [-1]
    slot_chain = [-1]
    slot_depth = [-1]
    bone_slot: dict[tuple[str, int], int] = {}
    node_slot: dict[tuple[int, int], int] = {}
    voff = 0

    for g in range(model.ngeom):
        did = int(model.geom_dataid[g])
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or did < 0:
            continue
        name = model.mesh(did).name
        va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
        fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
        local = model.mesh_vert[va : va + vn].astype(float)
        face = model.mesh_face[fa : fa + fn].astype(np.int64)
        world = data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T

        slot = _slot_for(
            model,
            g,
            name,
            kp_index,
            bone_slot,
            node_slot,
            slot_prox,
            slot_dist,
            slot_chain,
            slot_depth,
        )
        part = _part_for(name)
        if part not in part_index:
            part_index[part] = len(part_names)
            part_names.append(part)

        verts.append(world)
        faces.append(face + voff)
        face_rgb.append(np.tile(geom_rgb[g], (fn, 1)))
        vert_slot.append(np.full(vn, slot, dtype=np.int64))
        vert_part.append(np.full(vn, part_index[part], dtype=np.int64))
        voff += vn

    np.savez_compressed(
        OUT_MESH,
        vertices=np.concatenate(verts).astype(np.float32),
        faces=np.concatenate(faces).astype(np.int64),
        face_rgb=(np.concatenate(face_rgb) * 255).round().astype(np.uint8),
        vert_slot=np.concatenate(vert_slot),
        vert_part=np.concatenate(vert_part),
        part_names=np.asarray(part_names, dtype=object),
        slot_prox=np.asarray(slot_prox, dtype=np.int64),
        slot_dist=np.asarray(slot_dist, dtype=np.int64),
        slot_chain=np.asarray(slot_chain, dtype=np.int64),
        slot_depth=np.asarray(slot_depth, dtype=np.int64),
        kp_neutral=kp_neutral.astype(np.float32),
        coxa_idx=np.asarray(
            [kp_index[f"{leg}_thorax_coxa"] for leg in LEGS], dtype=np.int64
        ),
        point_names=np.asarray(point_names, dtype=object),
    )
    n_v = sum(len(v) for v in verts)
    n_f = sum(len(f) for f in faces)
    n_nodes = sum(1 for c in slot_chain if c >= 0)
    print(
        f"wrote {OUT_MESH.relative_to(REPO)}  ({len(verts)} meshes, {n_v} verts, "
        f"{n_f} faces, {len(slot_prox)} slots, {n_nodes} articulated nodes)"
    )


def _write_articulation(model, data, kp_neutral, point_names, point_meta) -> None:
    """Bake ``nmf_articulation.json``: the head/abdomen chains + neutral coxae.

    Each marker records the skeleton ``point`` it predicts, its chain ``depth``, its
    neutral world position, *and* the attachment ``body`` + ``offset`` it was built
    from. The top-level ``bodies`` map gives every chain body's neutral world frame
    (``pos`` + row-major ``mat``) and its chain/depth, so a run config can redefine a
    marker's offset (the labeling-scheme choice) and the loader recomputes its neutral
    position at runtime -- no MuJoCo needed (see ``Articulation.load``).
    """

    def anchor_axis(jname: str) -> tuple[list[float], list[float]]:
        jid = model.joint(jname).id
        return data.xanchor[jid].round(8).tolist(), data.xaxis[jid].round(8).tolist()

    kp_index = {n: i for i, n in enumerate(point_names)}
    bodies = _chain_bodies(model, data)
    chains = []
    for chain in CHAINS:
        joints = []
        for j in chain["joints"]:
            anchor, axis = anchor_axis(j["joint"])
            joints.append(
                {
                    "angle": j["angle"],
                    "anchor": anchor,
                    "axis": axis,
                    "bounds_deg": j["bounds"],
                }
            )
        markers = []
        for name, depth in chain["markers"]:
            meta = point_meta.get(name, {})
            _check_depth(chain, name, meta.get("body"), depth, bodies)
            markers.append(
                {
                    "point": name,
                    "depth": depth,
                    "neutral": kp_neutral[kp_index[name]].round(8).tolist(),
                    "body": meta.get("body"),
                    "offset": meta.get("offset"),
                }
            )
        chains.append(
            {
                "name": chain["name"],
                "joints": joints,
                "markers": markers,
                "base_point": chain.get("base_point"),
            }
        )

    coxa_points = [f"{leg}_thorax_coxa" for leg in LEGS]
    coxa_neutral = [kp_neutral[kp_index[p]].round(8).tolist() for p in coxa_points]
    OUT_ARTIC.write_text(
        json.dumps(
            {
                "coxa_points": coxa_points,
                "coxa_neutral": coxa_neutral,
                "chains": chains,
                "bodies": bodies,
            },
            indent=1,
        )
    )
    print(
        f"wrote {OUT_ARTIC.relative_to(REPO)}  "
        f"({len(chains)} chains: {[c['name'] for c in chains]}, "
        f"{len(bodies)} attachment bodies)"
    )


def _check_depth(chain, point: str, body: str | None, depth: int, bodies: dict) -> None:
    """A marker's declared depth must be its attachment body's, read from the model.

    The depth decides which joints carry the marker, so getting it wrong silently fits
    the wrong DOFs -- and it is written twice: once in the chain's ``markers`` list here,
    once implicitly by the ``body`` that ``keypoints.json`` assigns the point. This
    asserts the two agree against MuJoCo's own tree rather than trusting either.

    The chain's ``base_point`` is exempt: it names the marker that measures where the
    chain *sits*, which is forced to depth 0 because it lies on the rotation center, and
    its attachment body's depth would say otherwise (``c_head``'s is 3).
    """
    if point == chain.get("base_point"):
        return
    if body is None:
        raise ValueError(
            f"{chain['name']} marker {point!r} has no attachment body in "
            "docs/keypoints/assets/keypoints.json; add its `body` + `offset` there"
        )
    frame = bodies.get(body)
    if frame is None:
        raise ValueError(
            f"{chain['name']} marker {point!r} attaches to {body!r}, which is not a "
            f"chain body ({sorted(bodies)})"
        )
    if int(frame["depth"]) != int(depth):
        raise ValueError(
            f"{chain['name']} marker {point!r} is declared at depth {depth}, but its "
            f"attachment body {body!r} sits at depth {frame['depth']} in the model"
        )


def _chain_bodies(model, data) -> dict[str, dict]:
    """Neutral world frame + chain/depth of every body a marker may attach to.

    A marker is rigidly attached to a model body and rides the chain at that body's
    depth (the same ``_node_for_body`` rule that assigns the segment meshes). Baking
    each such body's frame lets the run config place a marker by an offset in the
    body frame: ``neutral = pos + mat @ offset`` at load time.
    """
    bodies: dict[str, dict] = {}
    for bid in range(1, model.nbody):
        node = _node_for_body(model, bid)
        if node is None:
            continue
        chain_idx, depth = node
        bodies[model.body(bid).name] = {
            "pos": data.xpos[bid].round(8).tolist(),
            "mat": data.xmat[bid].round(8).tolist(),  # 3x3 row-major
            "chain": CHAINS[chain_idx]["name"],
            "depth": int(depth),
        }
    return bodies


def _slot_for(
    model,
    g,
    name,
    kp_index,
    bone_slot,
    node_slot,
    slot_prox,
    slot_dist,
    slot_chain,
    slot_depth,
) -> int:
    """The render slot for a mesh: a leg bone (skin), an articulated node, or rigid."""
    leg, _, part = name.partition("_")
    seg = PART_SEGMENT.get(part)
    if leg in LEGS and seg is not None:
        key = (leg, seg)
        if key not in bone_slot:
            bone_slot[key] = len(slot_prox)
            slot_prox.append(kp_index[f"{leg}_{JOINTS[seg]}"])
            slot_dist.append(kp_index[f"{leg}_{JOINTS[seg + 1]}"])
            slot_chain.append(-1)
            slot_depth.append(-1)
        return bone_slot[key]

    node = _node_for_body(model, int(model.geom_bodyid[g]))
    if node is not None:
        if node not in node_slot:
            node_slot[node] = len(slot_prox)
            slot_prox.append(-1)
            slot_dist.append(-1)
            slot_chain.append(node[0])
            slot_depth.append(node[1])
        return node_slot[node]

    return 0  # thorax / wings / halteres -> rigid body


def _node_for_body(model, bid: int) -> tuple[int, int] | None:
    """``(chain_index, depth)`` for a body's mesh, or ``None`` if it is rigid."""
    ancestors = set()
    b = bid
    while b > 0:
        ancestors.add(model.body(b).name)
        b = model.body_parentid[b]
    if HEAD_CHAIN["node_root"] in ancestors:
        return 0, HEAD_CHAIN["node_depth"]
    # the deepest (most distal) abdomen segment that is an ancestor sets the depth.
    for seg, depth in sorted(
        ABDOMEN_CHAIN["segment_depth"].items(), key=lambda kv: -kv[1]
    ):
        if seg in ancestors:
            return 1, depth
    return None


def _asset_dict() -> dict[str, bytes]:
    """In-memory ``filename -> bytes`` for every STL the MJCF references."""
    return {p.name: p.read_bytes() for p in (ASSETS / "model").glob("*.stl")}


if __name__ == "__main__":
    main()
