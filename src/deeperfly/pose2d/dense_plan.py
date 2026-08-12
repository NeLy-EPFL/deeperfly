"""Build the ``[pose2d]`` plan for a dense-38 detector, and stamp it into a config.

The shipped plan is shaped by a **19-channel, one-side** detector: eight pathways for
seven cameras (the front one runs twice, mirrored), and ``[pose2d.output_points]`` maps
each pass's 19 channels into the near-side points of the views that see them. Half of
every side camera's skeleton is therefore ``NaN`` by construction -- there is no channel
that could have filled it.

A dense detector (:mod:`deeperfly.pose2d.hrnet`) predicts *every* point in *every* view,
so the plan collapses: **one pathway per camera, channel ``i`` -> point ``i`` of that
camera's view**, and nothing is ``NaN`` because nothing is unrepresented. The mirrored
twins disappear with it, which also halves the forward passes.

Two things this module does not guess:

* **The crop.** Each pathway's preprocessor is the *training* crop for that camera, and
  it must be the same box the detector was trained through or the fly arrives at the
  wrong scale. Boxes come from a crop-plan directory (``<dir>/<slug>.json``, the format
  the training repo writes) or from explicit ``--crop`` arguments; a view with neither
  gets the full frame, which is correct for the six side cameras of this rig and wrong
  for an axial one, so the summary prints what each view resolved to.
* **The skeleton.** The channel order is the checkpoint's own ``point_names``, and a
  config whose ``[skeleton] point_names`` disagree is rejected rather than silently
  routed through the wrong channels. That check is the whole reason this is generated
  code and not a config someone edits by hand: 38 channels x 8 views is 304 mappings, and
  a transposition in any one of them is a wrong limb, not a crash.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("deeperfly")

#: The dense model's input ``(height, width)``.
INPUT_HW: tuple[int, int] = (256, 512)


def crops_from_plan(
    plan_path: str | Path,
) -> dict[str, tuple[int, int, int, int] | None]:
    """Per-view session boxes from a crop-plan JSON (``None`` = the frozen full frame).

    The plan's ``session`` table maps a view to ``[x, y, w, h]`` or ``null``; a ``null``
    means "whatever the frozen per-camera policy already says", which for this rig's six
    side cameras is the full frame.
    """
    plan = json.loads(Path(plan_path).read_text())
    out: dict[str, tuple[int, int, int, int] | None] = {}
    for view, box in (plan.get("session") or {}).items():
        out[view] = None if box is None else tuple(int(v) for v in box)  # type: ignore[assignment]
    if plan.get("per_frame"):
        raise SystemExit(
            f"{plan_path} carries PER-FRAME crops, which a static config cannot express; "
            "re-derive a session-scoped plan or pass --crop explicitly"
        )
    return out


def checkpoint_points(ckpt: str | Path) -> list[str]:
    """The channel order a checkpoint was trained in, by point name."""
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    names = list(ck.get("point_names") or [])
    if not names:
        raise SystemExit(
            f"{ckpt} records no point_names, so its channel order cannot be verified "
            "against the skeleton; refusing to generate a mapping from a guess"
        )
    return [str(n) for n in names]


def dense_pose2d(
    *,
    views: list[str],
    point_names: list[str],
    weights: str,
    sources: dict[str, str],
    crops: dict[str, tuple[int, int, int, int] | None] | None = None,
    precision: str = "float16",
    batch_size: int = 16,
    decode_buffer: int = 4,
) -> dict[str, Any]:
    """The whole ``[pose2d]`` table for a dense detector: models, pathways, mappings.

    ``sources`` maps a view to the ``[[sources]]`` name that carries its footage.
    """
    crops = crops or {}
    missing = [v for v in views if v not in sources]
    if missing:
        raise SystemExit(f"no source named for view(s) {missing}")

    preprocessors = []
    pathways = []
    output_points: dict[str, dict[str, dict[str, Any]]] = {}
    for view in views:
        box = crops.get(view)
        prep_name = None
        if box is not None:
            x, y, w, h = box
            prep_name = f"crop_{view}"
            preprocessors.append(
                {
                    "name": prep_name,
                    "ops": [{"op": "crop", "x": x, "y": y, "width": w, "height": h}],
                }
            )
        pw: dict[str, Any] = {"name": view, "source": sources[view], "model": "dense38"}
        if prep_name:
            pw["preprocessor"] = prep_name
        pathways.append(pw)
        # channel i -> point i of this view. One pathway, no mirrored twin, so there is
        # no left/right decision to get wrong here -- the detector already made it.
        output_points[view] = {
            name: {"pathway": view, "out_channel": i}
            for i, name in enumerate(point_names)
        }

    return {
        "precision": precision,
        "batch_size": batch_size,
        "decode_buffer": decode_buffer,
        "preprocessors": preprocessors,
        "models": [
            {
                "name": "dense38",
                "class": "hrnet",
                "weights": str(weights),
                "input_size": list(INPUT_HW),
                # 0.0 and not DeepFly2D's 0.22: this network carries its own mean/std in
                # the checkpoint and applies them itself. `load_hrnet` refuses anything
                # else rather than shifting every input by a quarter of its range.
                "mean": 0.0,
                "n_out_channels": len(point_names),
            }
        ],
        "pathways": pathways,
        "output_points": output_points,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_fmt(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot serialize {value!r} to TOML")


def pose2d_toml(plan: dict[str, Any]) -> str:
    """Render :func:`dense_pose2d`'s result as the ``[pose2d]`` section of a config."""
    lines = [
        "# ===========================================================================",
        "# 2D detection -- DENSE plan: one pathway per camera, every point in every",
        "# view. Generated by `deeperfly dense-config`; see deeperfly.pose2d.dense_plan.",
        "#",
        "# The 19-channel default runs each side camera twice (once mirrored) and leaves",
        "# the far side of every view NaN. This detector emits all",
        f"# {len(plan['output_points'][next(iter(plan['output_points']))])} channels per",
        "# view, so a contralateral point arrives as a prediction to correct rather than",
        "# as a gap to author from nothing.",
        "# ===========================================================================",
        "[pose2d]",
    ]
    for key in ("precision", "batch_size", "decode_buffer"):
        lines.append(f"{key} = {_fmt(plan[key])}")

    if plan["preprocessors"]:
        lines += [
            "",
            "# Per-camera training crops. A detector is trained through a box; feeding it",
            "# a different one changes the fly's apparent scale, which is the one thing",
            "# no augmentation in this recipe undoes. A view with no entry here is run",
            "# full-frame, which is this rig's frozen policy for the six side cameras.",
        ]
    for prep in plan["preprocessors"]:
        lines += [
            "[[pose2d.preprocessors]]",
            f"name = {_fmt(prep['name'])}",
            f"ops = {_fmt(prep['ops'])}",
        ]

    lines += ["", "[[pose2d.models]]"]
    for k, v in plan["models"][0].items():
        lines.append(f"{k} = {_fmt(v)}")

    lines.append("")
    for pw in plan["pathways"]:
        lines.append("[[pose2d.pathways]]")
        for k, v in pw.items():
            lines.append(f"{k} = {_fmt(v)}")
        lines.append("")

    for view, table in plan["output_points"].items():
        lines.append(f"[pose2d.output_points.{view}]")
        width = max(len(n) for n in table)
        for name, entry in table.items():
            lines.append(f"{name:<{width}} = {_fmt(entry)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _replace_section(
    config_text: str, prefixes: tuple[str, ...], new: str, what: str
) -> str:
    """Swap every top-level table whose header starts with one of ``prefixes``.

    Everything before and after is preserved byte-for-byte, because the rest of the file
    is this recording's own: its ``[[sources]]``, its calibration, its stage knobs.
    Rewriting the file from a parsed dict would silently drop comments and reorder
    tables, and one of those tables holds the only copy of a camera rig.
    """
    lines = config_text.splitlines(keepends=True)

    def mine(s: str) -> bool:
        return s.startswith(prefixes)

    heads = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("[")]
    ours = [i for i in heads if mine(lines[i].lstrip())]
    if not ours:
        raise SystemExit(f"this config has no {what} section to replace")
    first, last = ours[0], ours[-1]
    after = next((i for i in heads if i > last), len(lines))

    # Pull back the comment block that introduces the section; it describes what is
    # being replaced, and leaving it would document a plan that is no longer there.
    start = first
    while start - 1 >= 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    # ...and push the cut back past any comment block introducing the NEXT table,
    # which documents that table and must survive.
    stop = after
    while stop - 1 > last and lines[stop - 1].lstrip().startswith("#"):
        stop -= 1
    while stop - 1 > last and not lines[stop - 1].strip():
        stop -= 1
    return "".join(lines[:start]) + new.rstrip("\n") + "\n\n" + "".join(lines[stop:])


def replace_skeleton_section(config_text: str, skeleton_toml: str) -> str:
    """Swap the ``[skeleton]`` tables for those of a standalone ``skeleton.toml``.

    The dense detector's channels ARE a skeleton: a checkpoint trained on ``fly38b``
    routed through a ``fly38`` config would put the abdomen chain on points that no
    longer exist. Stamping both in one edit is what keeps them from disagreeing.
    """
    return _replace_section(
        config_text, ("[skeleton]", "[skeleton."), skeleton_toml, "[skeleton]"
    )


def replace_pose2d_section(config_text: str, pose2d: str) -> str:
    """Swap a config's whole ``[pose2d]`` block (and its sub-tables) for a new one.

    Everything before and after is preserved byte-for-byte -- see :func:`_replace_section`.
    """
    return _replace_section(
        config_text, ("[pose2d]", "[pose2d.", "[[pose2d."), pose2d, "[pose2d]"
    )
