"""Generate the static assets for the interactive keypoint-locations docs page.

The docs page at ``docs/explanation/keypoints.md`` embeds a browser viewer
(``docs/keypoints/viewer.html``) that renders the NeuroMechFly biomechanical
model with MuJoCo compiled to WebAssembly and overlays deeperfly's 38 tracked
keypoints as a stick-and-ball skeleton. This script produces everything that
viewer loads, so the docs build itself stays lightweight -- it never imports
flygym or mujoco; it only copies the committed files under ``docs/keypoints/``.

It is therefore run *by hand* (not in CI) whenever the NeuroMechFly model or the
deeperfly skeleton changes. It needs ``flygym`` and ``mujoco``, which are heavy
and deliberately not part of any project dependency group. Run it in a throwaway
environment, e.g.::

    uv run --with flygym --with mujoco --python 3.12 \
        python scripts/build_keypoint_viewer_assets.py

Outputs (all under ``docs/keypoints/assets/``):

``model/fly.xml`` + ``model/*.stl``
    A flattened, self-contained MJCF and the simplified (<=2000 faces) meshes it
    references, written by ``dm_control.mjcf.export_with_assets``. The browser
    loads this exact file via ``mj_loadXML``.
``pose.json``
    The *controllable* joint DOFs (the 7 actuated DOFs of each of the 6 legs plus
    the 3 head DOFs): name, ``qpos`` address, neutral (resting) angle, slider
    range, a human label and a UI group. Drives the slider panel and its
    defaults. Also carries the full neutral ``qpos`` vector for all DOFs, so the
    rest of the body stays posed at its resting angles.
``colors.json``
    A representative RGB per geom, derived from flygym's ``visuals.yaml`` (the
    "Colours" toggle paints the mesh with these instead of a flat grey).
``keypoints.json``
    The 38 deeperfly points (read from the packaged ``default_config.toml`` so
    they stay in lockstep with the library), each mapped to a NeuroMechFly body
    plus a local offset, with the limb colours and within-limb bones. The
    overlay is read from these at runtime.
``ATTRIBUTION.txt``
    Upstream licence/attribution for the redistributed model. Kept as ``.txt`` so
    MkDocs serves it as a static file rather than rendering an orphan page.

Everything is resolved against the *exported* model (reloaded standalone, exactly
as the browser sees it) and asserted to exist, so a bad mapping fails here rather
than silently in the browser.
"""

from __future__ import annotations

import fnmatch
import json
import math
import shutil
import sys
import tomllib
from pathlib import Path

import dm_control.mjcf as mjcf
import mujoco as mj
import numpy as np
import yaml
from flygym import assets_dir
from flygym.anatomy import ALL_SEGMENT_NAMES, AxisOrder, JointPreset, Skeleton
from flygym.compose import Fly
from flygym.compose.pose import KinematicPosePreset

# --- repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_TOML = REPO_ROOT / "src/deeperfly/data/default_config.toml"
OUT_DIR = REPO_ROOT / "docs/keypoints/assets"
MODEL_DIR = OUT_DIR / "model"

# --- keypoint -> NeuroMechFly body mapping ----------------------------------
# deeperfly tracks the *joint between* two segments; NeuroMechFly defines each
# body's origin at its joint to the parent, so a leg keypoint is just the origin
# of the distal body. The claw and antenna are the distal *tips* of tarsus5 and
# the arista (computed from geometry below); the abdomen markers have no exact
# NeuroMechFly counterpart and are placed on the midline segments (approximate).
LEG_PREFIXES = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_SUFFIX_TO_BODY = {
    "thorax_coxa": "{leg}_coxa",
    "coxa_trochanter": "{leg}_trochanterfemur",
    "femur_tibia": "{leg}_tibia",
    "tibia_tarsus": "{leg}_tarsus1",
    "claw": "{leg}_tarsus5",  # + distal tip offset
}
# The abdomen markers have no exact NeuroMechFly counterpart; these are the
# hand-tuned (body, body-frame offset in mm) placements per point.
ABDOMEN_POINTS = {
    "l_abdomen0": ("c_abdomen3", [0.0, 0.05, 0.30]),
    "l_abdomen1": ("c_abdomen5", [-0.06, 0.05, 0.27]),
    "l_abdomen2": ("c_abdomen6", [-0.23, 0.05, 0.20]),
    "r_abdomen0": ("c_abdomen3", [0.0, -0.05, 0.30]),
    "r_abdomen1": ("c_abdomen5", [-0.06, -0.05, 0.27]),
    "r_abdomen2": ("c_abdomen6", [-0.23, -0.05, 0.20]),
}


def build_model() -> mj.MjModel:
    """Compose the NeuroMechFly fly with every biological DOF and a neutral pose,
    export it to a self-contained MJCF under ``MODEL_DIR``, and return the model
    reloaded standalone (exactly what the browser's ``mj_loadXML`` will see)."""
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL
    )
    fly = Fly()  # SIMPLIFIED_MAX2000FACES meshes by default
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    fly.compile()  # bakes the "neutral" keyframe

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True)
    mjcf.export_with_assets(fly.mjcf_root, str(MODEL_DIR), "fly.xml")

    model = mj.MjModel.from_xml_path(str(MODEL_DIR / "fly.xml"))
    assert model.nkey >= 1, "expected a baked 'neutral' keyframe in the exported model"
    return model


def body_id(model: mj.MjModel, short_name: str) -> int:
    """Resolve a bare segment name (e.g. ``lf_coxa``) to its body id, tolerating
    any model-name prefix the exporter might add."""
    for i in range(model.nbody):
        if model.body(i).name.split("/")[-1] == short_name:
            return i
    raise KeyError(f"body {short_name!r} not found in exported model")


def distal_tip_offset(model: mj.MjModel, short_name: str) -> np.ndarray:
    """Body-frame offset to the most distal point of a body's geometry.

    Used for the claw (a capsule on ``*_tarsus5``) and the antenna (the arista).
    Considers mesh vertices and capsule end-caps, transformed from geom frame to
    body frame, and returns the candidate farthest from the body origin.
    """
    bid = body_id(model, short_name)
    candidates: list[np.ndarray] = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        rot = np.zeros(9)
        mj.mju_quat2Mat(rot, model.geom_quat[g])
        rot = rot.reshape(3, 3)
        pos = model.geom_pos[g]
        gtype = int(model.geom_type[g])
        if gtype == mj.mjtGeom.mjGEOM_MESH:
            mi = int(model.geom_dataid[g])
            adr, num = model.mesh_vertadr[mi], model.mesh_vertnum[mi]
            verts = model.mesh_vert[adr : adr + num].reshape(-1, 3)
            candidates.append(verts @ rot.T + pos)
        elif gtype == mj.mjtGeom.mjGEOM_CAPSULE:
            half = model.geom_size[g][1]
            caps = np.array([[0, 0, half], [0, 0, -half]])
            candidates.append(caps @ rot.T + pos)
        else:
            candidates.append(pos[None, :])
    if not candidates:
        return np.zeros(3)
    pts = np.vstack(candidates)
    return pts[int(np.argmax(np.linalg.norm(pts, axis=1)))]


def map_keypoint(model: mj.MjModel, name: str) -> tuple[str, np.ndarray, bool]:
    """Map a deeperfly point name to ``(body, body_frame_offset_mm, approximate)``."""
    parts = name.split("_")
    if parts[0] in LEG_PREFIXES:
        leg, suffix = parts[0], "_".join(parts[1:])
        body = LEG_SUFFIX_TO_BODY[suffix].format(leg=leg)
        offset = distal_tip_offset(model, body) if suffix == "claw" else np.zeros(3)
        return body, offset, False
    if name in ("l_antenna", "r_antenna"):
        # The pedicel–head joint, i.e. the origin of the pedicel body.
        return f"{name[0]}_pedicel", np.zeros(3), False
    if "abdomen" in name:
        body, offset = ABDOMEN_POINTS[name]
        return body, np.array(offset), True
    raise ValueError(f"no NeuroMechFly mapping rule for keypoint {name!r}")


# --- slider grouping --------------------------------------------------------
def joint_group(child: str) -> tuple[str, str]:
    """Return ``(group_key, group_label)`` for the child segment of a joint."""
    for leg in LEG_PREFIXES:
        if child.startswith(leg + "_"):
            side = "Left" if leg[0] == "l" else "Right"
            pos = {"f": "front", "m": "mid", "h": "hind"}[leg[1]]
            return f"{leg}_leg", f"{side} {pos} leg"
    if child in ("l_pedicel", "l_funiculus", "l_arista"):
        return "l_antenna", "Left antenna"
    if child in ("r_pedicel", "r_funiculus", "r_arista"):
        return "r_antenna", "Right antenna"
    if child.startswith("c_abdomen"):
        return "abdomen", "Abdomen"
    if child in ("l_wing", "r_wing", "l_haltere", "r_haltere"):
        return "wings", "Wings & halteres"
    return "head", "Head & proboscis"  # head, rostrum, haustellum, eyes


# Only these DOFs get a slider: the 7 actuated leg DOFs (per NeuroMechFly) for
# each of the 6 legs, plus the 3 head DOFs. Everything else stays at its neutral
# angle (still in neutral_qpos, so the body remains posed).
GROUP_ORDER = [
    "lf_leg",
    "lm_leg",
    "lh_leg",
    "rf_leg",
    "rm_leg",
    "rh_leg",
    "head",
    "abdomen",
]
HEAD_DOFS = {"c_thorax-c_head-yaw", "c_thorax-c_head-pitch", "c_thorax-c_head-roll"}
# The abdomen kinematic chain, pitch DOFs only (c_thorax -> 12 -> 3 -> 4 -> 5 -> 6).
ABDOMEN_DOFS = {
    "c_thorax-c_abdomen12-pitch",
    "c_abdomen12-c_abdomen3-pitch",
    "c_abdomen3-c_abdomen4-pitch",
    "c_abdomen4-c_abdomen5-pitch",
    "c_abdomen5-c_abdomen6-pitch",
}


def controllable_leg_dofs(leg: str) -> set[str]:
    return {
        f"c_thorax-{leg}_coxa-yaw",
        f"c_thorax-{leg}_coxa-pitch",
        f"c_thorax-{leg}_coxa-roll",
        f"{leg}_coxa-{leg}_trochanterfemur-pitch",
        f"{leg}_coxa-{leg}_trochanterfemur-roll",
        f"{leg}_trochanterfemur-{leg}_tibia-pitch",
        f"{leg}_tibia-{leg}_tarsus1-pitch",
    }


CONTROLLABLE = HEAD_DOFS.union(
    ABDOMEN_DOFS, *(controllable_leg_dofs(leg) for leg in LEG_PREFIXES)
)


def build_pose_json(model: mj.MjModel) -> dict:
    """Controllable-DOF metadata + the full neutral qpos, for the slider panel."""
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)  # the "neutral" keyframe
    neutral_qpos = data.qpos.copy()

    joints = []
    for j in range(model.njnt):
        name = model.joint(j).name.split("/")[-1]
        if name not in CONTROLLABLE:
            continue
        parent, child, axis = (
            name.rsplit("-", 2)
            if name.count("-") >= 2
            else (
                name,
                name,
                "",
            )
        )
        adr = int(model.jnt_qposadr[j])
        neutral = float(neutral_qpos[adr])
        # Joints are unlimited; give a generous symmetric range that contains the
        # neutral angle so every slider can swing at least +/-180 deg from zero.
        lo = min(-math.pi, neutral - 0.1)
        hi = max(math.pi, neutral + 0.1)
        key, label = joint_group(child)
        joints.append(
            {
                "name": name,
                "qposadr": adr,
                "neutral": neutral,
                "range": [lo, hi],
                "label": f"{child.split('_', 1)[-1]} · {axis}" if axis else child,
                "group": key,
            }
        )

    groups = [{"key": k, "label": joint_group_label(k)} for k in GROUP_ORDER]
    return {
        "nq": int(model.nq),
        "neutral_qpos": [float(x) for x in neutral_qpos],
        "groups": groups,
        "joints": joints,
    }


def joint_group_label(key: str) -> str:
    """Human label for a group key (inverse of the keys produced by joint_group)."""
    labels = {
        "lf_leg": "Left front leg",
        "lm_leg": "Left mid leg",
        "lh_leg": "Left hind leg",
        "rf_leg": "Right front leg",
        "rm_leg": "Right mid leg",
        "rh_leg": "Right hind leg",
        "l_antenna": "Left antenna",
        "r_antenna": "Right antenna",
        "abdomen": "Abdomen",
        "wings": "Wings & halteres",
        "head": "Head",
    }
    return labels[key]


def build_keypoints_json(model: mj.MjModel) -> dict:
    """The 38 deeperfly points, their NeuroMechFly targets, colours and bones."""
    with open(CONFIG_TOML, "rb") as fh:
        skel = tomllib.load(fh)["skeleton"]
    point_names: list[str] = skel["point_names"]
    limb_points: dict[str, list[str]] = skel["limb_points"]
    palette: dict[str, str] = skel.get("limb_palette", {})

    point_to_limb = {p: limb for limb, pts in limb_points.items() for p in pts}
    index = {name: i for i, name in enumerate(point_names)}

    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)

    points, approx = [], []
    for name in point_names:
        body, offset, is_approx = map_keypoint(model, name)
        bid = body_id(model, body)  # asserts existence
        world = (
            np.array(data.body(bid).xpos)
            + np.array(data.body(bid).xmat).reshape(3, 3) @ offset
        )
        assert np.isfinite(world).all(), f"non-finite neutral position for {name}"
        limb = point_to_limb.get(name, "")
        points.append(
            {
                "name": name,
                "limb": limb,
                "color": palette.get(limb, "#888888"),
                "body": model.body(bid).name,
                "offset": [float(v) for v in offset],
            }
        )
        if is_approx:
            approx.append(name)

    bones = []
    for pts in limb_points.values():
        idxs = [index[p] for p in pts]
        bones.extend([a, b] for a, b in zip(idxs, idxs[1:]))

    return {
        "limbs": [
            {"name": limb, "color": palette.get(limb, "#888888")}
            for limb in limb_points
        ],
        "points": points,
        "bones": bones,
        "approximate": approx,
    }


def segment_colors() -> dict[str, list[float]]:
    """Map each body segment to a representative RGB from flygym's visuals.yaml.

    Textured materials have no flat colour, so we take the texture's base colour
    (``rgb1``, or the mean of ``rgb1``/``rgb2`` for gradients); plain materials use
    their ``rgba``. Wildcards in ``apply_to`` match segment names as in flygym.
    """
    with open(assets_dir / "model/visuals.yaml") as fh:
        vis = yaml.safe_load(fh)
    colors: dict[str, list[float]] = {}
    for params in vis.values():
        tex = params.get("texture")
        if tex:
            rgb1 = tex.get("rgb1", [0.6, 0.6, 0.6])
            rgb = (
                [(a + b) / 2 for a, b in zip(rgb1, tex.get("rgb2", rgb1))]
                if tex.get("builtin") == "gradient"
                else rgb1
            )
        else:
            rgb = params["material"]["rgba"][:3]
        patterns = params["apply_to"]
        for pattern in [patterns] if isinstance(patterns, str) else patterns:
            for seg in fnmatch.filter(ALL_SEGMENT_NAMES, pattern):
                colors[seg] = [round(float(c), 4) for c in rgb]
    return colors


def build_colors_json(model: mj.MjModel) -> dict:
    """A representative RGB per geom (matched by geom/segment name)."""
    seg_color = segment_colors()
    geom_rgb = []
    for g in range(model.ngeom):
        seg = model.geom(g).name.split("/")[-1]
        rgb = seg_color.get(seg)
        if rgb is None:  # fall back to the geom's body name
            body = model.body(int(model.geom_bodyid[g])).name.split("/")[-1]
            rgb = seg_color.get(body, [0.7, 0.7, 0.7])
        geom_rgb.append(rgb)
    return {"geom_rgb": geom_rgb}


ATTRIBUTION = """\
# Model attribution

The fly model (`model/fly.xml` and `model/*.stl`) is the **NeuroMechFly v2**
biomechanical model, generated from **flygym** (https://github.com/NeLy-EPFL/flygym),
which is distributed under the **Apache License 2.0**. The meshes are the
simplified (<=2000 faces) set. The flattened MJCF and the keypoint/pose metadata
in this directory are produced by `scripts/build_keypoint_viewer_assets.py`.

If you use the NeuroMechFly model, please cite the NeuroMechFly v2 publication
(see https://neuromechfly.org/). Approximate keypoint placements (antennae and
abdomen markers, listed under `approximate` in `keypoints.json`) have no exact
NeuroMechFly counterpart and are positioned for illustration only.
"""


def main() -> int:
    if not CONFIG_TOML.exists():
        sys.exit(f"cannot find deeperfly config at {CONFIG_TOML}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Composing + exporting NeuroMechFly model ...")
    model = build_model()

    print("Building pose.json (controllable DOFs + neutral) ...")
    pose = build_pose_json(model)
    (OUT_DIR / "pose.json").write_text(json.dumps(pose, indent=1))

    print("Building colors.json (per-geom flygym colours) ...")
    (OUT_DIR / "colors.json").write_text(json.dumps(build_colors_json(model)))

    print("Building keypoints.json (38 points -> bodies) ...")
    keypoints = build_keypoints_json(model)
    (OUT_DIR / "keypoints.json").write_text(json.dumps(keypoints, indent=1))

    (OUT_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION)

    n_stl = len(list(MODEL_DIR.glob("*.stl")))
    size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*")) / 1e6
    print(
        f"\nDone -> {OUT_DIR.relative_to(REPO_ROOT)}\n"
        f"  model/fly.xml + {n_stl} STL meshes\n"
        f"  {len(pose['joints'])} controllable DOFs (nq={pose['nq']} total)\n"
        f"  {len(keypoints['points'])} keypoints, "
        f"{len(keypoints['bones'])} bones, "
        f"{len(keypoints['approximate'])} approximate\n"
        f"  total {size_mb:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
