"""``deeperfly dense-config`` -- stamp a dense-38 detector into a recording's config.

The packaged plan is built for the 19-channel one-side detector and cannot express a
dense one: it names a mirrored twin pathway per side camera and routes 19 channels into
half a view. Swapping in a dense checkpoint therefore means rewriting the whole
``[pose2d]`` section, all 304 channel-to-point entries of it, which is exactly the kind
of edit a human should not be doing by hand.

So this generates it -- from the checkpoint's own recorded channel order, the config's
own view list, and the training crop boxes -- and replaces only that section, leaving
the rest of the file byte-for-byte. The rest of the file matters: a recording's
``config.toml`` carries its own camera geometry.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..pose2d.dense_plan import (
    DEFAULT_BATCH,
    checkpoint_points,
    crops_from_plan,
    dense_pose2d,
    pose2d_toml,
    replace_pose2d_section,
    replace_skeleton_section,
)

log = logging.getLogger("deeperfly")


def _parse(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


def _parse_crop(spec: str) -> tuple[str, tuple[int, int, int, int]]:
    view, _, box = spec.partition("=")
    parts = [p for p in box.replace(",", " ").split() if p]
    if not view or len(parts) != 4:
        raise SystemExit(f"--crop wants view=x,y,w,h; got {spec!r}")
    return view, tuple(int(p) for p in parts)  # type: ignore[return-value]


#: ``--detector`` -> (registry class, the name the model gets in the config). Both are
#: dense-38 plans and differ only in what computes the channels: ``hrnet`` predicts each
#: view alone, ``mvt`` encodes a frame's views together so a joint one camera cannot see
#: still gets a prediction from the ones that can.
_DETECTORS: dict[str, tuple[str, str]] = {
    "hrnet": ("hrnet", "dense38"),
    "mvt": ("mvt", "dense38mv"),
}


def _cmd_dense_config(args: argparse.Namespace) -> None:
    from ..config import Config
    from ..skeleton import Skeleton

    cfg_path = Path(args.config)
    text = cfg_path.read_text()
    if args.skeleton:
        text = replace_skeleton_section(text, Path(args.skeleton).read_text())
    config = Config.from_dict(_parse(text))

    # Views come from [cameras.*], not from the OLD detection plan: that plan is what
    # is being replaced, and after a --skeleton swap it no longer parses (its channel
    # mappings name points the new skeleton does not have).
    _defaults, per_view = config.camera_table()
    views = list(per_view)
    point_names = checkpoint_points(args.weights)

    skel = list(Skeleton.from_config(config).point_names)
    if skel != point_names:
        extra = [n for n in point_names if n not in skel]
        gone = [n for n in skel if n not in point_names]
        raise SystemExit(
            "the checkpoint's channel order is not this config's skeleton -- routing "
            "its channels would attach points to the wrong joints.\n"
            f"  config skeleton : {len(skel)} points\n"
            f"  checkpoint      : {len(point_names)} points\n"
            f"  only in ckpt    : {extra or 'none (order differs)'}\n"
            f"  only in config  : {gone or 'none (order differs)'}\n"
            "Pass --skeleton <skeleton.toml> to stamp the right one in the same edit."
        )

    sources = dict(args.source_map or {})
    for view in views:
        sources.setdefault(view, _guess_source(config, view))

    crops: dict[str, tuple[int, int, int, int] | None] = {}
    if args.crop_plan:
        crops.update(crops_from_plan(args.crop_plan))
    for spec in args.crop or []:
        view, box = _parse_crop(spec)
        crops[view] = box

    detector = str(getattr(args, "detector", None) or "hrnet")
    if detector not in _DETECTORS:
        raise SystemExit(
            f"unknown detector {detector!r}; expected one of {sorted(_DETECTORS)}"
        )
    model_class, model_name = _DETECTORS[detector]
    batch = args.batch_size
    if batch is None:
        batch = DEFAULT_BATCH.get(model_class, 16)

    plan = dense_pose2d(
        views=views,
        point_names=point_names,
        weights=str(Path(args.weights).resolve()),
        sources=sources,
        crops=crops,
        precision=args.precision,
        batch_size=batch,
        model_class=model_class,
        model_name=model_name,
    )
    out_text = replace_pose2d_section(text, pose2d_toml(plan))

    out = Path(args.output) if args.output else cfg_path
    if out.exists() and not args.overwrite and out != cfg_path:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    out.write_text(out_text)

    print(f"wrote {out}")
    print(f"  model    {Path(args.weights).name}  ({len(point_names)} channels)")
    for view in views:
        box = crops.get(view)
        where = "full frame" if box is None else f"crop {tuple(box)}"
        print(f"  view {view:<3} source {sources[view]:<10} {where}")


def _guess_source(config, view: str) -> str:
    """The ``[[sources]]`` name that carries ``view``'s footage.

    Matches by name (``vid_rh`` for view ``rh``, or a source literally named ``rh``);
    an unmatched view is a config the generator must not guess at, because the wrong
    source silently detects one camera and stores it as another.
    """
    names = list(config.source_patterns())
    for candidate in (f"vid_{view}", view, f"camera_{view}"):
        if candidate in names:
            return candidate
    raise SystemExit(
        f"cannot tell which source feeds view {view!r} (sources: {names}); "
        f"pass --source {view}=<source-name>"
    )
