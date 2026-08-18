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

#: A ``[pose2d].batch_size`` for each dense class.
#:
#: The unit is IMAGES for both, and it is not per-class after all:
#: :func:`~deeperfly.pose2d.inference.detect_sequence` forwards ``batch_size // pathways``
#: whole frames at a time, so with eight pathways anything below 8 is one frame per
#: forward. ``mvt`` was set to 2 here on the theory that its item was one *moment*; it is
#: not, and 2 therefore meant "one frame per forward" -- measured at 49.6 fps against 59.4
#: at 16 on an RTX 4090 (8 views, 256x512). 16 is the first value giving two frames.
DEFAULT_BATCH: dict[str, int] = {"hrnet": 16, "mvt": 16}


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
    """The channel order a checkpoint was trained in, by point name.

    Works for both dense classes because both record it at the top level: a dfpose
    ``.pt`` writes ``point_names`` directly, and an exported multiview-transformer
    artifact carries the same key (which is most of why the export step exists -- a raw
    Lightning checkpoint has no channel names, so nothing could be verified).
    """
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
    precision: str | None = None,
    batch_size: int = 16,
    decode_buffer: int = 4,
    model_class: str = "hrnet",
    model_name: str = "dense38",
) -> dict[str, Any]:
    """The whole ``[pose2d]`` table for a dense detector: models, pathways, mappings.

    ``sources`` maps a view to the ``[[sources]]`` name that carries its footage.

    The plan is the SAME SHAPE for both dense classes -- one pathway per camera, channel
    ``i`` -> point ``i``, no mirrored twins -- so ``model_class`` selects between them
    rather than forking this function. That is not a coincidence: the multiview
    transformer needs its views grouped by frame, and
    :func:`~deeperfly.pose2d.inference.detect_sequence` already batches one model's
    pathways that way. What changes between the two is which channels come out of a shared
    computation, not the plan.
    """
    crops = crops or {}
    missing = [v for v in views if v not in sources]
    if missing:
        raise SystemExit(f"no source named for view(s) {missing}")

    preprocessors = []
    pathways = []
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
        pw: dict[str, Any] = {"name": view, "source": sources[view]}
        if prep_name:
            pw["preprocessor"] = prep_name
        pathways.append(pw)
        # No [pose2d.output_points] table. Channel i -> point i of the view a pathway is
        # named after is what the plan DEFAULTS to (see
        # `deeperfly.pose2d.pathways._identity_triples`), so writing it out would be
        # 38 x V lines of the identity -- and the check that used to justify generating
        # them now runs on every load instead, where it also catches a hand-edited config
        # and a swapped weights file.

    return {
        "precision": precision,
        "batch_size": batch_size,
        "decode_buffer": decode_buffer,
        "preprocessors": preprocessors,
        # Just the class and the checkpoint. `input_size`, `mean`, `n_out_channels` and
        # (for the multiview transformer) `precision` are all properties of the artifact
        # whose loader already refuses a config that disagrees, so the class states them
        # -- see `deeperfly.pose2d.models.CLASS_DEFAULTS`. Writing them here would be the
        # generator copying the checkpoint into a file that gets rejected if it copies
        # wrong.
        "models": [
            {
                "name": model_name,
                "class": model_class,
                "weights": str(weights),
            }
        ],
        # The pathways carry no `model` key: one dense detector serves every camera, so
        # naming it per pathway was the same string written V times to point at the one
        # entry above. `[pose2d].model` is that default (see
        # `deeperfly.pose2d.pathways._default_model`).
        "model": model_name,
        "pathways": pathways,
        "n_out_channels": len(point_names),
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


#: ``[pose2d]`` knobs this generator only writes when they differ from the config
#: default. Restating a default is noise a reader has to check against the docs before
#: they can ignore it, and it silently freezes today's default into every generated file.
_KNOB_DEFAULTS: dict[str, Any] = {
    "precision": "bfloat16",
    "batch_size": 16,
    "decode_buffer": 4,
}


def pose2d_toml(plan: dict[str, Any]) -> str:
    """Render :func:`dense_pose2d`'s result as the ``[pose2d]`` section of a config.

    Written as one table with inline arrays rather than ``[[pose2d.models]]`` /
    ``[[pose2d.pathways]]`` blocks. The two parse identically; the difference is that a
    dense plan's pathway is three short keys, so a block per camera spent four lines and
    a blank on what reads as one row of a table.
    """
    n = plan["n_out_channels"]
    v = len(plan["pathways"])
    lines = [
        "# == 2D detection ============================================================",
        f"# A DENSE plan ({v} pathways, one per camera), generated by `deeperfly",
        "# dense-config`. The model emits every tracked point for every view, so the",
        "# mapping is channel i -> point i of the pathway's view and no",
        f"# [pose2d.output_points] table is written: the {n} x {v} rows it would take carry no",
        "# information, and a transposition in any one is a wrong limb rather than a crash.",
        "# That the model's channels really are this skeleton's, in this order, is checked",
        "# when the weights load.",
        "#",
        "# `model` is the default for every pathway below; write `model` on one to override",
        "# it. Everything else about the model -- input_size, mean, n_out_channels, and the",
        "# multiview transformer's float32 -- is a property of the checkpoint whose loader",
        "# refuses a config that disagrees, so the class states it and this does not.",
        "[pose2d]",
    ]
    for key, default in _KNOB_DEFAULTS.items():
        if plan.get(key) is not None and plan[key] != default:
            lines.append(f"{key} = {_fmt(plan[key])}")

    lines += [f"model = {_fmt(plan['model'])}", "models = ["]
    lines += [f"    {_fmt(m)}," for m in plan["models"]]
    lines.append("]")

    if plan["preprocessors"]:
        lines += [
            "",
            "# Per-camera training crops. A detector is trained through a box, and a",
            "# differently framed camera puts the animal at the wrong apparent SCALE --",
            "# the one thing no augmentation undoes. A view with no entry here runs",
            "# full-frame.",
            "preprocessors = [",
            *(f"    {_fmt(p)}," for p in plan["preprocessors"]),
            "]",
        ]

    lines += ["", "pathways = ["]
    lines += [f"    {_fmt(pw)}," for pw in plan["pathways"]]
    lines.append("]")

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
    # The blank lines between the two sections are NOT preserved -- they are re-emitted
    # below. Keeping them and adding a separator was adding one blank line per rewrite,
    # so running the generator twice on one file produced two different files.
    tail = "".join(lines[stop:])
    # Two blank lines between top-level sections, which is what these configs use, so a
    # replaced section leaves a file that still looks written rather than patched.
    gap = "\n\n\n" if tail.strip() else "\n"
    return "".join(lines[:start]) + new.rstrip("\n") + gap + tail


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
