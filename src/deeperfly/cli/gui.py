"""The ``gui`` command worker: launch the interactive web viewer/corrector.

Kept thin and free of any web import at module load -- the FastAPI/uvicorn
import happens inside :func:`deeperfly.gui.serve`, so importing ``deeperfly``
(and this module) stays cheap for every command other than ``gui``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("deeperfly")

#: Bind addresses that keep the editor on the local machine (no warning).
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _find_results(path: Path) -> Path:
    """Resolve ``path`` to a ``results.h5`` file.

    Accepts the file directly, or a directory containing ``results.h5`` (or a
    ``deeperfly_outputs/results.h5`` beneath it, i.e. a recording directory).

    Parameters
    ----------
    path
        A ``results.h5`` file or a directory to search.

    Returns
    -------
    Path
        The resolved ``results.h5`` path.

    Raises
    ------
    SystemExit
        If no ``results.h5`` can be found at ``path``.
    """
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (
            path / "results.h5",
            path / "deeperfly_outputs" / "results.h5",
        ):
            if candidate.exists():
                return candidate
    raise SystemExit(
        f"no results.h5 found at {path} -- pass a results.h5 file or a directory "
        "containing one (e.g. <recording>/deeperfly_outputs)"
    )


def _cmd_gui(args: argparse.Namespace) -> None:
    """Serve the web GUI on the result resolved from ``args.path``.

    Parameters
    ----------
    args
        The ``gui`` namespace (``path``, ``footage_dir``, ``host``, ``port``,
        ``no_browser``, ``keep_alive``).

    Raises
    ------
    SystemExit
        If no result is found, or the web stack fails to import (an incomplete
        install -- FastAPI + uvicorn are core dependencies).
    """
    # A project directory resolves inside `serve` (it may hold several recordings, and a
    # recording with no results.h5 opens uncalibrated); anything else is resolved here so
    # a bad path fails before the server starts.
    from ..project import PROJECT_FILENAME

    target = Path(args.path)
    is_project = (target / PROJECT_FILENAME).exists()
    results_path = target if is_project else _find_results(target)
    if args.host not in _LOOPBACK:
        log.warning(
            "binding %s exposes the editor on the network without authentication; "
            "prefer the default localhost and an `ssh -L` tunnel for remote use",
            args.host,
        )
    try:
        from ..gui import serve
    except ImportError as exc:  # pragma: no cover -- exercised manually
        raise SystemExit(str(exc)) from exc
    try:
        serve(
            results_path,
            footage_dir=args.footage_dir,
            recording=getattr(args, "recording", None),
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            exit_on_close=not args.keep_alive,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_labels_export(args: argparse.Namespace) -> None:
    """Export the saved ground-truth labels (``labels.h5``) as an ``.npz`` dataset.

    Resolves ``results.h5`` (a file or a directory holding one), loads the
    ``labels.h5`` beside it (refusing a sidecar from a different recording), and writes
    the provenance-filtered GT + occluded masks in footage pixel space. Raises
    ``SystemExit`` when there are no saved labels to export.
    """
    import numpy as np

    from ..gui.labels import export_absent, export_gt, labels_identity, load_labels
    from ..results import PoseResult, StageStore

    results_path = _find_results(Path(args.path))
    result = PoseResult.load(results_path)
    store = StageStore(results_path)
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes=store.read_image_sizes(),
        footage=store.read_footage(),
    )
    labels_path = results_path.parent / "labels.h5"
    labels = load_labels(labels_path, identity=identity)
    if labels is None:
        raise SystemExit(
            f"no labels.h5 next to {results_path} -- annotate with 'deeperfly gui' "
            "and Save first"
        )
    gt_xy, gt_mask, occluded = export_gt(labels)
    absent = export_absent(labels)
    out = Path(args.output) if args.output else results_path.parent / "labels_gt.npz"
    np.savez(
        out,
        gt_xy=gt_xy,
        gt_mask=gt_mask,
        occluded=occluded,
        # (T, P) "not on this animal", per frame (a limb can be lost part-way through a
        # recording). Strictly additive: a consumer that predates this key should default
        # it to all-False. The intended training semantics is MASKING -- an absent keypoint
        # contributes no gradient -- not negative supervision.
        absent=absent,
        point_names=np.array(list(result.skeleton.point_names)),
        camera_names=np.array(list(result.cameras.names)),
    )
    whole = absent.all(axis=0) if absent.size else absent
    names = [str(n) for n, a in zip(result.skeleton.point_names, whole) if a]
    n_partial = int((absent.any(axis=0) & ~whole).sum()) if absent.size else 0
    log.info(
        "exported %d ground-truth point(s), %d occlusion(s), %d keypoint(s) not on this "
        "animal%s%s (footage space) -> %s",
        int(gt_mask.sum()),
        int(occluded.sum()),
        int(whole.sum()),
        f" ({', '.join(names)})" if names else "",
        f", plus {n_partial} absent in some frames only" if n_partial else "",
        out,
    )


def _cmd_labels_absent(args: argparse.Namespace) -> None:
    """Declare keypoints absent (not on this animal) in one or more label sidecars.

    The batch, pre-GUI counterpart of the editor's ``x`` gesture. The scientist usually
    knows a leg is gone before the camera rolls, and one animal is typically recorded
    several times -- so this takes many recordings at once and writes the same
    point-indexed declaration to each. Point-indexing is what makes that safe: the
    declaration carries no frame or view indices, so it means the same thing in every clip
    of the same animal.

    Refuses to touch a sidecar whose ``labels.h5`` is being edited elsewhere is *not*
    something it can detect -- ``save_labels`` is a whole-file rewrite -- so it warns
    loudly instead. Close the GUI first.
    """
    import numpy as np

    from ..gui.labels import (
        Labels,
        labels_identity,
        load_labels,
        resolve_point_names,
        save_labels,
    )
    from ..results import PoseResult, StageStore

    names = [n.strip() for n in str(args.points).split(",") if n.strip()]
    if not names:
        raise SystemExit(
            "--points is empty; pass e.g. --points 'lf_femur_tibia,lf_claw'"
        )

    frame_spec = getattr(args, "frames", None)
    span: tuple[int, int] | None = None
    if frame_spec:
        # "900:" (from 900 to the end), "0:900", ":900", "900" (that frame alone).
        try:
            if ":" in str(frame_spec):
                lo_s, _, hi_s = str(frame_spec).partition(":")
                span = (int(lo_s) if lo_s else 0, int(hi_s) if hi_s else -1)
            else:
                span = (int(frame_spec), int(frame_spec) + 1)
        except ValueError:
            raise SystemExit(
                f"--frames {frame_spec!r} is not a frame or a RANGE like '900:' or "
                "'0:900' (end exclusive)"
            ) from None

    for raw in args.paths:
        results_path = _find_results(Path(raw))
        result = PoseResult.load(results_path)
        store = StageStore(results_path)
        identity = labels_identity(
            point_names=list(result.skeleton.point_names),
            camera_names=list(result.cameras.names),
            n_frames=result.n_frames,
            image_sizes=store.read_image_sizes(),
            footage=store.read_footage(),
        )
        try:
            idx = resolve_point_names(names, list(result.skeleton.point_names))
        except ValueError as exc:
            raise SystemExit(f"{results_path}: {exc}") from exc

        labels_path = results_path.parent / "labels.h5"
        labels = load_labels(labels_path, identity=identity)
        if labels is None:
            labels = Labels.empty(
                result.n_views, result.n_frames, len(result.skeleton.point_names)
            )
        before = labels.absent.copy()
        if span is None:
            labels.set_absent(idx, not args.clear)  # the whole recording
        else:
            lo, hi = span
            hi = result.n_frames if hi < 0 else min(hi, result.n_frames)
            if not 0 <= lo < hi:
                raise SystemExit(
                    f"{results_path}: --frames {frame_spec!r} is empty or out of range "
                    f"for a {result.n_frames}-frame recording"
                )
            labels.set_absent(idx, not args.clear, frames=range(lo, hi))
        after = labels.absent
        if np.array_equal(before, after):
            log.info("%s: already as requested, not rewritten", labels_path)
            continue

        # Quarantine counts, so the operator sees exactly what the declaration displaced.
        vetoed = after[None, :, :]
        vetoed_gt = int((labels.gt_authored & vetoed).sum())
        vetoed_occ = int((labels.occluded & vetoed).sum())
        save_labels(
            labels_path,
            labels,
            identity=identity,
            subject_id=args.subject or labels.subject_id,
        )
        whole = after.all(axis=0)
        partial = after.any(axis=0) & ~whole
        declared = [str(n) for n, a in zip(result.skeleton.point_names, whole) if a]
        some = [str(n) for n, a in zip(result.skeleton.point_names, partial) if a]
        log.info(
            "%s: %d keypoint(s) absent in every frame (%s)%s%s",
            labels_path,
            len(declared),
            ", ".join(declared) if declared else "none",
            f"; {len(some)} in some frames only ({', '.join(some)})" if some else "",
            (
                f"; quarantined {vetoed_gt} GT row(s) and {vetoed_occ} occlusion(s) "
                "(restored if you un-declare)"
                if vetoed_gt or vetoed_occ
                else ""
            ),
        )
