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
environment **outside the project** -- ``uv run`` inside it would rebuild
``.venv`` against this script's pins::

    uv venv --python 3.12 /tmp/nmf && \
    uv pip install --python /tmp/nmf/bin/python 'flygym<2.1' mujoco dm_control pyyaml && \
    /tmp/nmf/bin/python scripts/build_keypoint_viewer_assets.py

``flygym`` is pinned below 2.1 on purpose: 2.1.0 migrated ``Fly.mjcf_root`` from
dm_control's PyMJCF to ``mujoco.MjSpec``, and the export below is PyMJCF's. The
pin is also what keeps a rebuild honest -- ``flygym==2.0.2`` reproduces the
committed ``model/`` byte for byte, so a run that only changes the skeleton shows
up as a one-file diff in ``keypoints.json``.

By default the skeleton is whichever one the packaged ``default_config.toml``
names (``fly38``); ``--skeleton <name>`` builds the page for another packaged one.

Outputs (all under ``docs/keypoints/assets/``):

``model/fly.xml`` + ``model/*.stl``
    A flattened, self-contained MJCF and the simplified (<=2000 faces) meshes it
    references, written by ``dm_control.mjcf.export_with_assets``. The browser
    loads this exact file via ``mj_loadXML``.
``pose.json``
    The *controllable* joint DOFs (the 7 actuated DOFs of each of the 6 legs, the
    3 head DOFs, and the pitch + roll of each of the 5 abdominal hinges): name,
    ``qpos`` address, neutral (resting) angle, slider range, a human label, a UI
    group, and for the few DOFs whose axis name misdescribes them a ``hint`` the
    panel shows as a tooltip. Drives the slider panel and its defaults. Also
    carries the full neutral ``qpos`` vector for all DOFs, so the rest of the body
    stays posed at its resting angles.
``colors.json``
    A representative RGB per geom, derived from flygym's ``visuals.yaml`` (the
    "Colors" toggle paints the mesh with these instead of a flat grey).
``keypoints.json``
    The 38 deeperfly points (read from the packaged skeleton so they stay in
    lockstep with the library), each mapped to a NeuroMechFly body plus a local
    offset, with the limb colors and within-limb bones. The overlay is read from
    these at runtime.
``ATTRIBUTION.txt``
    Upstream licence/attribution for the redistributed model. Kept as ``.txt`` so
    MkDocs serves it as a static file rather than rendering an orphan page.

Everything is resolved against the *exported* model (reloaded standalone, exactly
as the browser sees it) and asserted to exist, so a bad mapping fails here rather
than silently in the browser.
"""

from __future__ import annotations

import argparse
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
SKELETON_DIR = REPO_ROOT / "src/deeperfly/data/skeletons"
OUT_DIR = REPO_ROOT / "docs/keypoints/assets"
MODEL_DIR = OUT_DIR / "model"


def load_skeleton(name: str | None = None) -> dict:
    """The ``[skeleton]`` table, presets expanded -- ``deeperfly.config`` without importing it.

    A run config only *names* its skeleton (``[skeleton] name = "fly38"``); the point
    names, mirror pairs, limb chains and colors live in ``data/skeletons/<name>.toml``.
    This mirrors :func:`deeperfly.config._resolve_skeleton`: a table that already spells
    out ``point_names`` is self-contained and used as it is, otherwise the named preset is
    loaded and the config's own keys override it **wholesale, per key**. Reimplemented
    rather than imported because this script runs in a throwaway flygym environment that
    has no torch/jax, so it cannot import the library.
    """
    with open(CONFIG_TOML, "rb") as fh:
        skel = tomllib.load(fh)["skeleton"]
    if name is None and "point_names" in skel:
        return skel
    preset = SKELETON_DIR / f"{name or skel.get('name')}.toml"
    if not preset.is_file():
        available = ", ".join(sorted(p.stem for p in SKELETON_DIR.glob("*.toml")))
        sys.exit(f"no packaged skeleton {preset.stem!r} (have: {available})")
    with open(preset, "rb") as fh:
        base = tomllib.load(fh)["skeleton"]
    # An explicit --skeleton asks for that preset as written; otherwise the config's keys
    # win over the preset's, per key, exactly as a real run resolves them.
    return base if name else {**base, **skel}


# --- keypoint -> NeuroMechFly body mapping ----------------------------------
# deeperfly tracks the *joint between* two segments; NeuroMechFly defines each
# body's origin at its joint to the parent, so a leg keypoint is just the origin
# of the distal body. The pretarsus and antenna are the distal *tips* of tarsus5 and
# the arista (computed from geometry below); the abdomen markers have no exact
# NeuroMechFly counterpart and are placed on the midline segments (approximate).
LEG_PREFIXES = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_SUFFIX_TO_BODY = {
    "thorax_coxa": "{leg}_coxa",
    "coxa_trochanter": "{leg}_trochanterfemur",
    "femur_tibia": "{leg}_tibia",
    "tibia_tarsus": "{leg}_tarsus1",
    "pretarsus": "{leg}_tarsus5",  # + distal tip offset
}
# The abdomen markers have no exact NeuroMechFly counterpart; these are the
# hand-tuned (body, body-frame offset in mm) placements per point, and they are the
# same ones the packaged `data/nmf_articulation.json` carries for the IK.
ABDOMEN_POINTS = {
    "l_abdomen0": ("c_abdomen3", [0.0, 0.05, 0.30]),
    "l_abdomen1": ("c_abdomen5", [-0.06, 0.05, 0.27]),
    "l_abdomen2": ("c_abdomen6", [-0.23, 0.05, 0.20]),
    "r_abdomen0": ("c_abdomen3", [0.0, -0.05, 0.30]),
    "r_abdomen1": ("c_abdomen5", [-0.06, -0.05, 0.27]),
    "r_abdomen2": ("c_abdomen6", [-0.23, -0.05, 0.20]),
}

# fly38's dorsal-midline abdomen chain, which replaced the DeepFly3D set's two lateral
# ones. Each point
# sits on the sagittal plane (y = 0) directly above an abdominal hinge -- `abdomen_k` over
# the hinge at the head of its own segment -- except `abdomen4`, which is the tip marker
# on the last segment, there being no sixth hinge. Attaching a point to the segment it
# sits ON is what makes it ride that tergite when the abdomen curls.
#
# The heights are a table rather than a formula because no closed form put all five where
# an annotator would: interpolating the two unlabeled stripes along the chord between
# their neighbors (which is how the label corpus derives them, at fractions 0.5652 and
# 0.3303) bunches `abdomen3` against `abdomen2` on this model and sinks it ~0.049 below
# the dorsal crest. These were set by eye against the crest instead, and land at an even
# spacing with a uniform ~0.03 gap under it. `abdomen0` and `abdomen4` are still exactly
# the DeepFly3D side-pair midpoints; `abdomen2` is that marker moved onto the hinge.
#
# Each stripe hangs off the segment BEHIND its intersegmental fold, one body proximal of
# where the first midline retarget put it -- hence the -x in every offset. The earlier
# anchoring had `abdomen3` and `abdomen4` on the SAME body (`c_abdomen6`), so no joint lay
# between them and the model held them rigidly 0.234 apart while a real fly measured 0.287:
# 18% short and unreachable by any angle, which was the largest single abdomen residual.
# One joint now lies between every neighbouring pair, and on a re-fit of the same recording
# every abdomen segment length lands within 2% of the animal's and the chain's residual
# falls 42%. Keep this table and `docs/keypoints/assets/keypoints.json` in step: this is
# what generates that file, and `scripts/build_nmf_mesh_asset.py` then reads it into the
# IK's own asset, so an edit made only to the JSON is reverted by the next run of this.
MIDLINE_POINTS = {
    "abdomen0": ("c_abdomen12", [-0.37, 0.0, 0.340]),
    "abdomen1": ("c_abdomen3", [-0.22, 0.0, 0.320]),
    "abdomen2": ("c_abdomen4", [-0.23, 0.0, 0.300]),
    "abdomen3": ("c_abdomen5", [-0.24, 0.0, 0.280]),
    "abdomen4": ("c_abdomen6", [-0.25, 0.0, 0.220]),
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

    # Export to a scratch directory and swap it in only once the export SUCCEEDS.
    # MODEL_DIR is committed (the viewer and scripts/build_nmf_mesh_asset.py both read
    # it), and deleting it up front means any failure downstream -- a flygym/dm_control
    # version drift is enough -- leaves the repo without its MJCF and its meshes.
    staging = MODEL_DIR.with_name(MODEL_DIR.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    mjcf.export_with_assets(fly.mjcf_root, str(staging), "fly.xml")
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    staging.rename(MODEL_DIR)

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

    Used for the pretarsus (a capsule on ``*_tarsus5``) and the antenna (the arista).
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
        offset = (
            distal_tip_offset(model, body) if suffix == "pretarsus" else np.zeros(3)
        )
        return body, offset, False
    if name in ("l_antenna", "r_antenna"):
        # The pedicel–head joint, i.e. the origin of the pedicel body.
        return f"{name[0]}_pedicel", np.zeros(3), False
    if name == "neck":
        # The c_thorax-c_head pivot, i.e. the origin of the head body. Exact, like a leg
        # joint -- it is a joint of the model, not a placement convention.
        return "c_head", np.zeros(3), False
    if name in MIDLINE_POINTS:  # fly38's abdomen0..4
        body, offset = MIDLINE_POINTS[name]
        return body, np.array(offset), True
    if "abdomen" in name:  # fly38's l_abdomen0..2 / r_abdomen0..2
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
# each of the 6 legs, the 3 head DOFs, and 2 of the 3 DOFs of each of the 5
# abdominal hinges. Everything else stays at its neutral angle (still in
# neutral_qpos, so the body remains posed).
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
# The abdomen kinematic chain (c_thorax -> 12 -> 3 -> 4 -> 5 -> 6), two DOFs per hinge --
# the same pair `scripts/build_nmf_mesh_asset.py` bakes into the IK's abdomen chain, so the
# sliders here move the fly through exactly the angles a fit reports. In the child
# segment's own frame every abdominal hinge's axes are:
#
#     -pitch  ->  local y  ->  sagittal bend (the ventral curl)
#     -roll   ->  local z  ->  LATERAL swing, despite the name
#     -yaw    ->  local x  ->  axial twist about the long axis
#
# flygym's axis names are anatomically rotated on this chain, so the lateral DOF is `-roll`
# and not the `-yaw` one would reach for -- worth spelling out because both readings look
# plausible and only one is right.
#
# The twist gets no slider, because the IK does not fit it, and the reason is aliasing
# rather than invisibility: every marker on this chain lies in the sagittal plane, and such
# a point is swung sideways by BOTH a twist about local x and a bend about local z -- the
# two differ only in lever arm (the marker's height above the axis, vs its distance behind
# the hinge). Measured on this model the two displacement fields agree to cos 0.87-0.97 per
# hinge, so a twist slider would largely repeat the roll one. The markers are not actually
# *on* the twist axis -- they clear it by 0.22-0.34 model units, and a 20 deg twist moves
# them 41-88% as far as the same roll does -- so it is the near-collinearity that rules the
# twist out, not a vanishing lever arm.
ABDOMEN_DOFS = {
    "c_thorax-c_abdomen12-pitch",
    "c_thorax-c_abdomen12-roll",
    "c_abdomen12-c_abdomen3-pitch",
    "c_abdomen12-c_abdomen3-roll",
    "c_abdomen3-c_abdomen4-pitch",
    "c_abdomen3-c_abdomen4-roll",
    "c_abdomen4-c_abdomen5-pitch",
    "c_abdomen4-c_abdomen5-roll",
    "c_abdomen5-c_abdomen6-pitch",
    "c_abdomen5-c_abdomen6-roll",
}
# Slider tooltips, keyed by model joint name. Only a DOF whose flygym axis name does not
# describe what the reader sees the slider do needs one, which on this model is exactly
# the abdominal `-roll` set above: unexplained, a "roll" slider that bends the abdomen
# sideways reads as a bug in the viewer rather than a naming convention of the model.
JOINT_HINTS = {
    name: "flygym's roll axis — but on the abdomen the axis names are anatomically "
    "rotated, and this one is the segment's local z, so it swings the abdomen "
    "LATERALLY (left/right) rather than twisting it. The twist axis (flygym's yaw) "
    "gets no slider: on a marker that sits on the midline a twist is almost the same "
    "motion as this lateral swing, so the two could not be told apart."
    for name in ABDOMEN_DOFS
    if name.endswith("-roll")
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

# Per-DOF joint-angle limits in DEGREES, given in the legacy naming convention
# ({LEG}_{joint}_{axis}) and converted to model joint names. Legs only; head and
# abdomen DOFs keep a generous default range.
_LIMIT_JOINT_BODIES = {
    "ThC": ("c_thorax", "{leg}_coxa"),
    "CTr": ("{leg}_coxa", "{leg}_trochanterfemur"),
    "FTi": ("{leg}_trochanterfemur", "{leg}_tibia"),
    "TiTa": ("{leg}_tibia", "{leg}_tarsus1"),
}


def _limit_entry(legacy: str, lo_hi: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    """Map a legacy ``LF_ThC_yaw`` limit to ``(model_name, (lo, hi))``.

    flygym negates the rotation axis for right-side roll and yaw, so the angle
    sign -- and therefore the limit bounds -- flip relative to the legacy
    convention (left ``roll (0, 180)`` mirrors to right ``roll (-180, 0)``).
    """
    leg, joint, axis = legacy.split("_")
    parent, child = _LIMIT_JOINT_BODIES[joint]
    leg = leg.lower()
    name = f"{parent.format(leg=leg)}-{child.format(leg=leg)}-{axis}"
    lo, hi = lo_hi
    if leg[0] == "r" and axis != "pitch":
        lo, hi = -hi, -lo
    return name, (lo, hi)


JOINT_LIMITS_DEG = dict(
    _limit_entry(k, v)
    for k, v in {
        "LF_ThC_yaw": (-180, 180),
        "LF_ThC_pitch": (-90, 90),
        "LF_ThC_roll": (-180, 180),
        "LF_CTr_pitch": (-180, 180),
        "LF_CTr_roll": (-180, 180),
        "LF_FTi_pitch": (-180, 180),
        "LF_TiTa_pitch": (-180, 0),
        "LM_ThC_yaw": (-50, 50),
        "LM_ThC_pitch": (-180, 180),
        "LM_ThC_roll": (0, 180),
        "LM_CTr_pitch": (-180, 180),
        "LM_CTr_roll": (-180, 180),
        "LM_FTi_pitch": (-180, 180),
        "LM_TiTa_pitch": (-180, 0),
        "LH_ThC_yaw": (-50, 50),
        "LH_ThC_pitch": (-50, 50),
        "LH_ThC_roll": (0, 180),
        "LH_CTr_pitch": (-180, 0),
        "LH_CTr_roll": (-180, 180),
        "LH_FTi_pitch": (-180, 180),
        "LH_TiTa_pitch": (-180, 0),
        "RF_ThC_yaw": (-180, 180),
        "RF_ThC_pitch": (-90, 90),
        "RF_ThC_roll": (-180, 180),
        "RF_CTr_pitch": (-180, 180),
        "RF_CTr_roll": (-180, 180),
        "RF_FTi_pitch": (-180, 180),
        "RF_TiTa_pitch": (-180, 0),
        "RM_ThC_yaw": (-50, 50),
        "RM_ThC_pitch": (-180, 180),
        "RM_ThC_roll": (-180, 0),
        "RM_CTr_pitch": (-180, 180),
        "RM_CTr_roll": (-180, 180),
        "RM_FTi_pitch": (-180, 180),
        "RM_TiTa_pitch": (-180, 0),
        "RH_ThC_yaw": (-50, 50),
        "RH_ThC_pitch": (-50, 50),
        "RH_ThC_roll": (-180, 0),
        "RH_CTr_pitch": (-180, 0),
        "RH_CTr_roll": (-180, 180),
        "RH_FTi_pitch": (-180, 180),
        "RH_TiTa_pitch": (-180, 0),
    }.items()
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
        if name in JOINT_LIMITS_DEG:
            lo, hi = (math.radians(d) for d in JOINT_LIMITS_DEG[name])
        else:
            # Otherwise unlimited; give a generous symmetric range that contains
            # the neutral angle so every slider can swing at least +/-180 deg.
            lo = min(-math.pi, neutral - 0.1)
            hi = max(math.pi, neutral + 0.1)
        key, label = joint_group(child)
        entry = {
            "name": name,
            "qposadr": adr,
            "neutral": neutral,
            "range": [lo, hi],
            "label": f"{child.split('_', 1)[-1]} · {axis}" if axis else child,
            "group": key,
        }
        if name in JOINT_HINTS:
            entry["hint"] = JOINT_HINTS[name]
        joints.append(entry)

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


def build_keypoints_json(model: mj.MjModel, skel: dict) -> dict:
    """The 38 deeperfly points, their NeuroMechFly targets, colors and bones."""
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
        # Named so the viewer can say *which* skeleton it is showing -- the page is a
        # labeling reference, and a stale one that does not admit it is worse than none.
        "skeleton": skel.get("name", "?"),
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

    Textured materials have no flat color, so we take the texture's base color
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
(see https://neuromechfly.org/). Approximate keypoint placements (the abdomen
markers, listed under `approximate` in `keypoints.json`) have no exact
NeuroMechFly counterpart and are positioned for illustration only.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skeleton",
        metavar="NAME",
        help="a packaged skeleton to build the page for (default: whatever "
        "default_config.toml names)",
    )
    args = parser.parse_args()

    if not CONFIG_TOML.exists():
        sys.exit(f"cannot find deeperfly config at {CONFIG_TOML}")
    skel = load_skeleton(args.skeleton)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Composing + exporting NeuroMechFly model ...")
    model = build_model()

    print("Building pose.json (controllable DOFs + neutral) ...")
    pose = build_pose_json(model)
    (OUT_DIR / "pose.json").write_text(json.dumps(pose, indent=1))

    print("Building colors.json (per-geom flygym colors) ...")
    (OUT_DIR / "colors.json").write_text(json.dumps(build_colors_json(model)))

    print(f"Building keypoints.json ({skel.get('name')} -> bodies) ...")
    keypoints = build_keypoints_json(model, skel)
    (OUT_DIR / "keypoints.json").write_text(json.dumps(keypoints, indent=1))

    (OUT_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION)

    n_stl = len(list(MODEL_DIR.glob("*.stl")))
    size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*")) / 1e6
    print(
        f"\nDone -> {OUT_DIR.relative_to(REPO_ROOT)}\n"
        f"  model/fly.xml + {n_stl} STL meshes\n"
        f"  {len(pose['joints'])} controllable DOFs (nq={pose['nq']} total)\n"
        f"  {keypoints['skeleton']}: {len(keypoints['points'])} keypoints, "
        f"{len(keypoints['bones'])} bones, "
        f"{len(keypoints['approximate'])} approximate\n"
        f"  total {size_mb:.2f} MB"
    )
    print_ik_markers(keypoints)
    return 0


def print_ik_markers(keypoints: dict) -> None:
    """Echo the head/abdomen placements as an ``[inverse_kinematics.*]`` table.

    The same offsets appear in ``docs/explanation/keypoints.md``, where a reader can
    paste them into a run config to take a ``fly38b`` body plan from 32 of 38 fitted
    points to 38 of 38. Printing them here is what keeps that page honest: if a
    placement above ever changes, the rebuild that changes it also says what the page
    should now read.
    """
    # Only the points the packaged articulation does not already carry are worth a
    # table: it bakes the antennae and fly38's side chains, so what a fly38b config
    # has to supply is the head chain re-listed with `neck` (a table replaces the
    # chain's markers wholesale) plus the midline abdomen.
    wanted = {
        "head": lambda n: n.endswith("_antenna") or n == "neck",
        "abdomen": lambda n: n.startswith("abdomen"),
    }

    # A point the baked articulation cannot already reach. Without one of these the
    # skeleton needs no table at all, and an empty heading would only mislead.
    def novel(name: str) -> bool:
        return name == "neck" or name.startswith("abdomen")

    points = keypoints["points"]
    tables = {
        chain: [p for p in points if keep(p["name"])] for chain, keep in wanted.items()
    }
    tables = {
        c: rows for c, rows in tables.items() if any(novel(p["name"]) for p in rows)
    }
    if not tables:
        return
    print(
        "\nMarker placements for [inverse_kinematics] (docs/explanation/keypoints.md):"
    )
    for chain, rows in tables.items():
        print(f"\n    [inverse_kinematics.{chain}]")
        for p in rows:
            body = p["body"].split("/")[-1]
            cells = ", ".join(f"{v:.6g}" for v in p["offset"])
            print(f'    {p["name"]} = {{ body = "{body}", offset = [{cells}] }}')


if __name__ == "__main__":
    raise SystemExit(main())
