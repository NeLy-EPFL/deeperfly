"""``deeperfly ik bind``: generate a (skeleton, model) binding for review.

A binding says where each tracked point sits on a model, and it is **not fully
hand-authorable**: a pretarsus's offset is the most distal vertex of the last tarsus, which
is computed from the model's geometry rather than chosen. So a new pair needs a
generator that expands the name conventions, resolves the geometry, and leaves the
genuinely ambiguous rows -- the ones no rule can decide -- marked ``approximate`` for a
human to review.

The three kinds of rule are the same three that used to live in
``scripts/build_keypoint_viewer_assets.py`` as a docs concern:

- **name conventions** -- deeperfly tracks the *joint between* two segments, and an
  MJCF body's origin is its joint to its parent, so a leg keypoint is just the origin
  of the distal body;
- **geometry** -- the pretarsus is the distal tip of ``*_tarsus5``, computed once and frozen
  into a number;
- **judgement** -- the abdomen midline points have no exact counterpart on the model, so
  their placement is a modelling decision. Those rows come out ``approximate = true``
  and are exactly what a reviewer should be reading.

This needs MuJoCo, which deeperfly does not depend on. Install it for the one run
(``uv run --with mujoco deeperfly ik bind ...``); the generated file is what ships.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .console import LogLevel, LogLevelOption, _configure_logging

log = logging.getLogger("deeperfly")

#: Which model body a leg keypoint's name suffix sits at. A name convention over the
#: model's bodies, expanded once here rather than composed at load time.
LEG_SUFFIX_TO_BODY = {
    "thorax_coxa": "{leg}_coxa",
    "coxa_trochanter": "{leg}_trochanterfemur",
    "femur_tibia": "{leg}_tibia",
    "tibia_tarsus": "{leg}_tarsus1",
    "pretarsus": "{leg}_tarsus5",  # + the distal tip offset, computed from the mesh
}

#: Points whose placement is a judgement, per model: the dorsal-midline abdomen chain
#: has no exact NeuroMechFly counterpart.
#:
#: Each point sits on the sagittal plane (y = 0) directly above an abdominal hinge --
#: ``abdomen_k`` over the hinge at the head of its own segment -- except ``abdomen4``,
#: the tip marker on the last segment, there being no sixth hinge. Attaching a point to
#: the segment it sits ON is what makes it ride that tergite when the abdomen curls.
#:
#: The heights are a table rather than a formula because no closed form put all five
#: where an annotator would: interpolating the two unlabeled stripes along the chord
#: between their neighbours (which is how the label corpus derives them, at fractions
#: 0.5652 and 0.3303) bunches ``abdomen3`` against ``abdomen2`` on this model and sinks
#: it ~0.049 below the dorsal crest. These were set by eye against the crest instead,
#: and land at an even spacing with a uniform ~0.03 gap under it. ``abdomen0`` and
#: ``abdomen4`` are still exactly the DeepFly3D side-pair midpoints; ``abdomen2`` is
#: that marker moved onto the hinge.
#:
#: Each stripe hangs off the segment BEHIND its intersegmental fold, one body proximal
#: of where the first midline retarget put it -- hence the ``-x`` in every offset. The
#: earlier anchoring had ``abdomen3`` and ``abdomen4`` on the SAME body, so no joint lay
#: between them and the model held them rigidly 0.234 apart while a real fly measured
#: 0.287: 18% short and unreachable by any angle, the largest single abdomen residual.
#: One joint now lies between every neighbouring pair, and on a re-fit of the same
#: recording every abdomen segment length lands within 2% of the animal's and the
#: chain's residual falls 42%.
JUDGED = {
    "neuromechfly": {
        "abdomen0": ("c_abdomen12", [-0.37, 0.0, 0.340]),
        "abdomen1": ("c_abdomen3", [-0.22, 0.0, 0.320]),
        "abdomen2": ("c_abdomen4", [-0.23, 0.0, 0.300]),
        "abdomen3": ("c_abdomen5", [-0.24, 0.0, 0.280]),
        "abdomen4": ("c_abdomen6", [-0.25, 0.0, 0.220]),
        # The DeepFly3D set's two lateral abdomen chains, kept so that skeleton can be
        # bound too. Same kind of row, same reason.
        "l_abdomen0": ("c_abdomen3", [0.0, 0.05, 0.30]),
        "l_abdomen1": ("c_abdomen5", [-0.06, 0.05, 0.27]),
        "l_abdomen2": ("c_abdomen6", [-0.23, 0.05, 0.20]),
        "r_abdomen0": ("c_abdomen3", [0.0, -0.05, 0.30]),
        "r_abdomen1": ("c_abdomen5", [-0.06, -0.05, 0.27]),
        "r_abdomen2": ("c_abdomen6", [-0.23, -0.05, 0.20]),
    }
}

#: Points that are a chain's base landmark rather than an observation of its angles.
BASE = {"neuromechfly": {"neck"}}

LEG_PREFIXES = ("lf", "lm", "lh", "rf", "rm", "rh")


def _body_id(model, short_name: str) -> int:
    """A bare segment name to its body id, tolerating any model-name prefix."""
    for i in range(model.nbody):
        if model.body(i).name.split("/")[-1] == short_name:
            return i
    raise KeyError(f"body {short_name!r} not found in {model.nbody} model bodies")


def _distal_tip_offset(model, short_name: str) -> np.ndarray:
    """Body-frame offset to the most distal point of a body's geometry.

    The pretarsus (a capsule on ``*_tarsus5``). Considers mesh vertices and capsule end-caps,
    transformed from geom frame to body frame, and returns the candidate farthest from
    the body origin. This is the number a binding cannot be hand-authored without.
    """
    import mujoco as mj

    bid = _body_id(model, short_name)
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
            candidates.append(np.array([[0, 0, half], [0, 0, -half]]) @ rot.T + pos)
        else:
            candidates.append(pos[None, :])
    if not candidates:
        return np.zeros(3)
    pts = np.vstack(candidates)
    return pts[int(np.argmax(np.linalg.norm(pts, axis=1)))]


def _row(model, model_name: str, point: str) -> tuple[str, np.ndarray, bool]:
    """``(body, offset, approximate)`` for one tracked point on one model."""
    judged = JUDGED.get(model_name, {})
    if point in judged:
        body, offset = judged[point]
        return body, np.asarray(offset, dtype=float), True
    parts = point.split("_")
    if parts[0] in LEG_PREFIXES:
        leg, suffix = parts[0], "_".join(parts[1:])
        if suffix not in LEG_SUFFIX_TO_BODY:
            raise ValueError(f"no rule for leg keypoint suffix {suffix!r} ({point!r})")
        body = LEG_SUFFIX_TO_BODY[suffix].format(leg=leg)
        offset = (
            _distal_tip_offset(model, body) if suffix == "pretarsus" else np.zeros(3)
        )
        return body, offset, False
    if point.endswith("antenna"):
        # The pedicel-head joint, i.e. the origin of the pedicel body.
        return f"{point[0]}_pedicel", np.zeros(3), False
    if point == "neck":
        # The c_thorax-c_head pivot, i.e. the origin of the head body. Exact, like a
        # leg joint -- it is a joint of the model, not a placement convention.
        return "c_head", np.zeros(3), False
    raise ValueError(
        f"no rule maps skeleton point {point!r} onto model {model_name!r}. Add it to "
        "JUDGED (with its offset) if its placement is a judgement, or write the row by "
        "hand into the generated file."
    )


def render(skeleton, model_name: str, rows: dict) -> str:
    """The binding file's text, in the skeleton's own point order."""
    out = [
        f"# {skeleton.name} on {model_name}: where each tracked point sits on the "
        "model.",
        "#",
        "# GENERATED by `deeperfly ik bind` and then REVIEWED. Every row marked",
        "# `approximate` is a modelling decision no rule can make -- read those.",
        "",
        f'skeleton = "{skeleton.name}"',
        f'model = "{model_name}"',
        "",
        "[points]",
    ]
    width = max(len(n) for n in skeleton.point_names)
    for name in skeleton.point_names:
        body, offset, approximate = rows[name]
        parts = [f'body = "{body}"']
        if np.any(offset):
            parts.append("offset = [" + ", ".join(repr(float(v)) for v in offset) + "]")
        if approximate:
            parts.append("approximate = true")
        if name in BASE.get(model_name, set()):
            parts.append("base = true")
        out.append(f"{name:<{width}} = {{ {', '.join(parts)} }}")
    return "\n".join(out) + "\n"


ik_app = typer.Typer(
    no_args_is_help=True,
    help="Model packs and the bindings that adapt a skeleton to one.",
)


@ik_app.command("bind")
def ik_bind(
    skeleton: Annotated[
        str | None,
        typer.Argument(help="the skeleton to bind (default: the packaged one)"),
    ] = None,
    model: Annotated[
        str, typer.Argument(help="the model pack to bind it to")
    ] = "neuromechfly",
    mjcf: Annotated[
        str | None,
        typer.Option("--mjcf", help="the model's MJCF, which the rules are read from"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("-o", "--output", help="where to write (default: data/bindings/)"),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Generate a (skeleton, model) binding, for review.

    A binding says where each tracked point sits on the model, and it is the one
    artifact where a skeleton point name and a model body name may appear together. It
    is not fully hand-authorable -- a pretarsus's offset is the most distal vertex of the
    last tarsus, computed from the geometry -- so this expands the name conventions,
    resolves that geometry, and marks the rows no rule can decide `approximate = true`.

    **Read the approximate rows.** They are placements a human chose, and this command
    only reproduces the choice that was made before; binding a new skeleton means
    deciding them again.

    Needs MuJoCo, which deeperfly does not depend on:
    `uv run --with mujoco deeperfly ik bind fly38 neuromechfly --mjcf model/fly.xml`.
    """
    _configure_logging(log_level.value)
    try:
        import mujoco  # noqa: F401
    except ImportError:
        sys.exit(
            "`deeperfly ik bind` needs MuJoCo to resolve the geometry a binding "
            "cannot be hand-authored without (the pretarsus's distal tip). deeperfly does "
            "not depend on it; run this one command with it installed, e.g.\n"
            "    uv run --with mujoco deeperfly ik bind ..."
        )
    import mujoco as mj

    from ..config import Config
    from ..inverse_kinematics.binding import BINDING_DIR
    from ..inverse_kinematics.pack import ModelPack

    spec = {} if not skeleton else {"skeleton": {"include": skeleton}}
    skeleton = Config.from_dict(spec).skeleton()
    pack = ModelPack.load(model)
    if not mjcf:
        sys.exit(
            "give --mjcf: the model's MJCF is what the conventions and the geometry "
            "are read from, and a pack ships only its baked assets"
        )
    model = mj.MjModel.from_xml_path(str(mjcf))

    rows = {n: _row(model, pack.name, n) for n in skeleton.point_names}
    text = render(skeleton, pack.name, rows)
    out = Path(output or BINDING_DIR / f"{skeleton.name}@{pack.name}.toml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    n_approx = sum(1 for r in rows.values() if r[2])
    log.info(
        "wrote %s (%d points, %d approximate -- review those)",
        out,
        len(rows),
        n_approx,
    )
