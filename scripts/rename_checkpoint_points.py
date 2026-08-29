"""Rewrite the point names a detector checkpoint records, for a declared rename.

A detector's channels ARE a skeleton, and every run compares the names the checkpoint
carries against the config's skeleton
(:func:`deeperfly.pose2d.stream._check_channel_names`) -- a count check cannot tell two
38-point skeletons apart, so that comparison is strict and a checkpoint trained under
`*_claw` is refused by a config whose skeleton says `*_pretarsus`. It is refused for a
good reason: nothing in either file says the two names denote one point. Only a person
can say that, which is what this script is for -- the checkpoint counterpart of
``deeperfly project skeleton --rename``, which does the same for a project's labels.

It rewrites NAMES ONLY. The channel order, the weights and every other key are copied
through untouched, so the network computes exactly what it computed before; a rename
that also reordered would be a different model and is refused here.

    uv run python scripts/rename_checkpoint_points.py model.ckpt '*_claw=*_pretarsus'

Writes in place after a `.bak` copy, or elsewhere with ``-o``. Both the HRNet
(``point_names`` beside the state dict) and the MVT (``point_names`` in the artifact
dict) store the names the same way, so one script covers both.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deeperfly.project.migrate import expand_renames  # noqa: E402


def _load(path: Path):
    """Read a checkpoint under ``weights_only=True``, with ``pathlib`` allowlisted.

    The HRNet exports record the run directory they came from as a `PosixPath` and the
    torch version they were written under as a `TorchVersion`, neither of which the
    safe unpickler allows by default. Both are inert data, so they are allowlisted
    here rather than dropping the whole file's guard to ``weights_only=False``.
    """
    import pathlib

    import torch

    allowed = [
        pathlib.PosixPath,
        pathlib.WindowsPath,
        pathlib.PurePosixPath,
        torch.torch_version.TorchVersion,
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument(
        "rename",
        nargs="+",
        help="OLD=NEW, one '*' allowed on each side (e.g. '*_claw=*_pretarsus')",
    )
    ap.add_argument("-o", "--out", type=Path, help="write here instead of in place")
    ap.add_argument(
        "-n", "--dry-run", action="store_true", help="report the renames, write nothing"
    )
    args = ap.parse_args()

    import torch

    ck = _load(args.checkpoint)
    names = list(ck.get("point_names") or [])
    if not names:
        raise SystemExit(
            f"{args.checkpoint} records no point names, so there is nothing to rename "
            "-- and nothing that would tell a run what its channels mean either"
        )

    # No new skeleton to land on here, so the guard is that every pattern matches a
    # channel and that no two channels end up sharing a name.
    mapping = expand_renames(list(args.rename), names)
    renamed = [mapping.get(n, n) for n in names]
    if len(set(renamed)) != len(renamed):
        raise SystemExit(
            "that rename would give two channels the same name: "
            f"{sorted({n for n in renamed if renamed.count(n) > 1})}"
        )

    for was, now in mapping.items():
        print(f"  {was} -> {now}")
    print(f"{len(mapping)} of {len(names)} channels renamed")
    if args.dry_run:
        return

    out = args.out or args.checkpoint
    if out == args.checkpoint:
        backup = args.checkpoint.with_suffix(args.checkpoint.suffix + ".bak")
        shutil.copy2(args.checkpoint, backup)
        print(f"backup: {backup}")
    ck["point_names"] = renamed
    torch.save(ck, out)
    print(f"wrote:  {out}")


if __name__ == "__main__":
    main()
