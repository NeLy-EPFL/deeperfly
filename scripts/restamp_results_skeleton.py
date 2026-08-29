"""Rewrite the skeleton a `results.h5` records, for a declared rename or a new palette.

A `results.h5` is portable: every array in it is `(..., P, ...)` with no names beside
it, so the file carries its own skeleton and that record is what the editor colors by
and what `deeperfly run` compares a config against
(`deeperfly.pipeline.run._refuse_a_foreign_skeleton`). A file written before a rename
therefore keeps refusing the config that renamed the point, and one written before
colors and symmetries were stored reads back on the colormap with no mirror pairs --
cosmetic for the palette, not cosmetic for flip augmentation.

This re-stamps that record IN PLACE, and only where re-stamping is honest: the stored
point ORDER must equal the new skeleton's, modulo the renames declared on the command
line. A file whose points are in a different order is a migration, not a re-stamp, and
is refused -- the P axis would then mean something else.

    uv run python scripts/restamp_results_skeleton.py results.h5 \
        --skeleton project/skeleton.toml --rename '*_claw=*_pretarsus'

    uv run python scripts/restamp_results_skeleton.py --project ~/fly-pose-data/project \
        --rename '*_claw=*_pretarsus' --apply

Reports and writes nothing without --apply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deeperfly.project.migrate import expand_renames  # noqa: E402
from deeperfly.results.core import _write_skeleton  # noqa: E402
from deeperfly.skeleton import Skeleton  # noqa: E402


def _stored_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        raw = f["skeleton/point_names"][()]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _restamp(path: Path, skeleton: Skeleton) -> None:
    with h5py.File(path, "r+") as f:
        del f["skeleton"]
        _write_skeleton(f.create_group("skeleton"), skeleton)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path, nargs="*", help="results.h5 file(s)")
    ap.add_argument(
        "--project",
        type=Path,
        help="a project: take its skeleton, and every recording's results.h5",
    )
    ap.add_argument(
        "--skeleton",
        type=Path,
        help="TOML with a [skeleton] table (default: --project's)",
    )
    ap.add_argument(
        "--rename",
        action="append",
        default=[],
        help="OLD=NEW, one '*' allowed on each side (e.g. '*_claw=*_pretarsus')",
    )
    ap.add_argument(
        "--apply", action="store_true", help="write (otherwise this only reports)"
    )
    args = ap.parse_args()

    paths = list(args.results)
    skeleton = None
    if args.project is not None:
        from deeperfly.project.core import Project

        proj = Project.load(args.project)
        skeleton = proj.skeleton()
        paths += [
            p
            for p in (proj.results_path(e) for e in proj.recordings)
            if p is not None and Path(p).exists()
        ]
    if args.skeleton is not None:
        from deeperfly.config import Config

        skeleton = Config.from_toml(args.skeleton).skeleton()
    if skeleton is None:
        raise SystemExit("give --skeleton, or --project to take the project's")
    if not paths:
        raise SystemExit("no results.h5 to re-stamp")

    print(
        f"skeleton: {skeleton.label}  ({skeleton.n_points} points, "
        f"{skeleton.n_edges} edges, {skeleton.n_symmetries} symmetry pairs)"
    )
    written = skipped = 0
    foreign: list[tuple[Path, int]] = []
    for path in sorted({Path(p).resolve() for p in paths}):
        stored = _stored_names(path)
        try:
            mapping = expand_renames(
                list(args.rename), stored, list(skeleton.point_names)
            )
        except ValueError:
            # A rename that matches nothing in THIS file -- it is already on the new
            # spelling, or on another skeleton entirely. Both are decided below, by
            # comparing the names; neither is a reason to stop the batch.
            mapping = {}
        renamed = [mapping.get(n, n) for n in stored]
        if renamed != list(skeleton.point_names):
            first = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(renamed, skeleton.point_names))
                    if a != b
                ),
                min(len(renamed), skeleton.n_points),
            )
            # Another skeleton entirely -- a file on the historical DeepFly3D points, say.
            # Re-stamping it would relabel its P axis, so it is left exactly as it is.
            foreign.append((path, first))
            continue
        with h5py.File(path, "r") as f:
            g = f["skeleton"]
            same_names = stored == list(skeleton.point_names)
            has_colors = "point_colors" in g
            has_sym = (
                "point_symmetries" in g and np.asarray(g["point_symmetries"][()]).size
            )
        if same_names and has_colors and has_sym:
            skipped += 1
            continue
        need = [
            what
            for what, ok in (
                ("names", same_names),
                ("colors", has_colors),
                ("symmetries", has_sym),
            )
            if not ok
        ]
        print(f"  {path}  <- {', '.join(need)}")
        if args.apply:
            _restamp(path, skeleton)
        written += 1
    for path, first in foreign:
        print(f"  SKIPPED {path}: another skeleton (first difference at point {first})")
    verb = "re-stamped" if args.apply else "would re-stamp"
    print(
        f"{verb} {written} file(s); {skipped} already current; "
        f"{len(foreign)} on another skeleton, untouched"
    )
    if not args.apply and written:
        print("dry run -- nothing written. Re-run with --apply")


if __name__ == "__main__":
    main()
