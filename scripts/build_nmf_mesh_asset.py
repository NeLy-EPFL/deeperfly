"""Bake the NeuroMechFly mesh overlay asset (``deeperfly/data/nmf_mesh.npz``).

The ``inverse_kinematics`` stage can overlay the NeuroMechFly body mesh, posed to
match the fitted pose, onto the 2D views (the ``mesh_nmf`` video op and the GUI's
mesh overlay). Rather than carry MuJoCo/flygym at runtime, this *one-time* script
reads the same self-contained model the docs keypoint viewer ships
(``docs/keypoints/assets/model/fly.xml`` + its simplified ``*.stl`` meshes, built
by ``scripts/build_keypoint_viewer_assets.py``) and bakes a compact, runtime-only
``.npz``: every mesh's neutral-pose vertices in world coordinates, its faces and
color, and which "slot" it is posed by --

- a **leg segment** (coxa / trochanter+femur / tibia / tarsus), posed at render
  time by mapping its two endpoint keypoints (from the fitted pose) onto the
  segment's neutral keypoint endpoints (a per-bone similarity transform); or
- the rigid **body** (thorax / head / abdomen / wings / antennae), posed by one
  global similarity transform fit from the six thorax-coxa keypoints.

So the overlay needs only the fitted model joints (``inverse_kinematics`` already
stores them) plus this asset -- no MuJoCo, no MJCF, no STL parsing at runtime.

Run it by hand (not in CI) whenever the bundled model changes::

    uv run --with mujoco --python 3.12 python scripts/build_nmf_mesh_asset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "keypoints" / "assets"
OUT = REPO / "src" / "deeperfly" / "data" / "nmf_mesh.npz"

# deeperfly skeleton order (must match Skeleton.fly().point_names / keypoints.json).
# Each leg's five keypoints, in chain order, name the four segment bones below.
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

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    face_rgb: list[np.ndarray] = []
    vert_slot: list[np.ndarray] = []
    # slot 0 is the rigid body; slots >= 1 are leg bones (prox/dist keypoint indices).
    slot_prox = [-1]
    slot_dist = [-1]
    bone_slot: dict[tuple[str, int], int] = {}
    voff = 0

    for g in range(model.ngeom):
        did = int(model.geom_dataid[g])
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or did < 0:
            continue
        name = model.mesh(did).name
        va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
        fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
        local = model.mesh_vert[va : va + vn].astype(float)
        face = model.mesh_face[fa : fa + fn].astype(np.int64)  # local 0-based
        world = data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T

        slot = _slot_for(name, kp_index, bone_slot, slot_prox, slot_dist)
        verts.append(world)
        faces.append(face + voff)
        face_rgb.append(np.tile(geom_rgb[g], (fn, 1)))
        vert_slot.append(np.full(vn, slot, dtype=np.int64))
        voff += vn

    np.savez_compressed(
        OUT,
        vertices=np.concatenate(verts).astype(np.float32),
        faces=np.concatenate(faces).astype(np.int64),
        face_rgb=(np.concatenate(face_rgb) * 255).round().astype(np.uint8),
        vert_slot=np.concatenate(vert_slot),
        slot_prox=np.asarray(slot_prox, dtype=np.int64),
        slot_dist=np.asarray(slot_dist, dtype=np.int64),
        kp_neutral=kp_neutral.astype(np.float32),
        coxa_idx=np.asarray(
            [kp_index[f"{leg}_thorax_coxa"] for leg in LEGS], dtype=np.int64
        ),
        point_names=np.asarray(point_names, dtype=object),
    )
    n_v = sum(len(v) for v in verts)
    n_f = sum(len(f) for f in faces)
    print(
        f"wrote {OUT.relative_to(REPO)}  ({len(verts)} meshes, {n_v} verts, {n_f} faces, "
        f"{len(slot_prox)} slots)"
    )


def _slot_for(name, kp_index, bone_slot, slot_prox, slot_dist) -> int:
    """The render slot for a mesh: a leg bone (>= 1) or the rigid body (0)."""
    leg, _, part = name.partition("_")
    seg = PART_SEGMENT.get(part)
    if leg not in LEGS or seg is None:
        return 0  # thorax / head / abdomen / wings / antennae -> rigid body
    key = (leg, seg)
    if key not in bone_slot:
        bone_slot[key] = len(slot_prox)
        slot_prox.append(kp_index[f"{leg}_{JOINTS[seg]}"])
        slot_dist.append(kp_index[f"{leg}_{JOINTS[seg + 1]}"])
    return bone_slot[key]


def _asset_dict() -> dict[str, bytes]:
    """In-memory ``filename -> bytes`` for every STL the MJCF references."""
    return {p.name: p.read_bytes() for p in (ASSETS / "model").glob("*.stl")}


if __name__ == "__main__":
    main()
