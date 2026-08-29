"""``deeperfly calibration`` -- inspect and extract solved camera rigs.

Two verbs, both cheap and read-only where they can be:

``show``
    Print a calibration's cameras, provenance and residuals. The residuals are the
    point: a rig is not a set of numbers to be trusted on sight, and the whole reason
    :class:`~deeperfly.rig.calibration.Calibration` carries a quality block is so a human
    (or a script) can refuse one.

``export``
    Lift the rig out of a ``results.h5`` into a portable ``calibration.toml``. This is
    the migration path for every recording processed before calibrations existed: the
    bundle-adjusted cameras have been sitting in ``bundle_adjustment/cameras`` all
    along, reachable only through HDF5 and usable only by the recording that produced
    them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..results import PoseResult, StageStore
from ..rig.calibration import CALIBRATION_FILENAME, Calibration, quality_from_errors

log = logging.getLogger("deeperfly")


def _resolve_results(path: str | Path) -> Path:
    """A ``results.h5`` from a file or a directory containing one.

    Mirrors ``deeperfly gui``'s argument handling, so the same thing a user pastes at
    the editor works here.

    Raises
    ------
    SystemExit
        If no result file is there.
    """
    p = Path(path)
    if p.is_dir():
        for candidate in (p / "results.h5", p / "deeperfly_outputs" / "results.h5"):
            if candidate.exists():
                return candidate
        raise SystemExit(f"no results.h5 under {p}")
    if not p.exists():
        raise SystemExit(f"{p} does not exist")
    return p


def _cmd_calibration_show(args) -> None:
    """Print a summary of a calibration file (``deeperfly calibration show``)."""
    print(Calibration.load(args.path).summary())


def _cmd_calibration_export(args) -> None:
    """Write a ``calibration.toml`` from a result's stored rig.

    Prefers the bundle-adjusted rig and falls back to the config rig ``pose2d``
    recorded, saying which it used -- an unrefined rig is still worth exporting (it is
    what the run actually projected with), but the provenance must not claim it was
    solved.
    """
    results_path = _resolve_results(args.path)
    store = StageStore(results_path)

    cameras = store.read_cameras("bundle_adjustment")
    method = "labels_ba"
    if cameras is None:
        cameras = store.read_cameras("pose2d")
        method = "orbit_prior"
        log.warning(
            "%s has no bundle_adjustment/cameras; exporting the un-refined config rig "
            "pose2d recorded (run with do_bundle_adjustment = true to solve one)",
            results_path,
        )
    if cameras is None:
        raise SystemExit(f"{results_path} stores no camera rig to export")

    # Residuals of the rig being exported, measured on the result's own 2D. Recomputed
    # rather than read from `triangulation/reproj_error`, which may describe a
    # *different* rig (or a substituted 2D layer) than the one going into this file.
    quality: dict = {}
    try:
        from ..rig.triangulation import reprojection_error, triangulate

        result = PoseResult.load(results_path)
        pts2d = result.pts2d
        quality = quality_from_errors(
            reprojection_error(cameras, triangulate(cameras, pts2d), pts2d),
            cameras.names,
        )
    except Exception:  # a 2D-only or unreadable result still exports its rig
        log.warning(
            "could not measure this rig's reprojection error; exporting without it"
        )

    out = (
        Path(args.output) if args.output else results_path.parent / CALIBRATION_FILENAME
    )
    # Pass the rig's OWN units/scale/intrinsics through when the result recorded them, and
    # only fall back to the config-orbit answer when it did not. Hardcoding them re-labelled
    # a millimeter board calibration as an arbitrary-scale orbit guess -- a false claim on
    # the one field that says whether the numbers mean anything physical.
    stored_meta = store.read_camera_meta(
        "bundle_adjustment" if method == "labels_ba" else "pose2d"
    )
    stored_provenance = dict(stored_meta.get("provenance") or {})
    calibration = Calibration.from_camera_group(
        cameras,
        name=results_path.parent.parent.name or results_path.parent.name,
        image_sizes=stored_meta.get("image_sizes") or store.read_image_sizes(),
        units=stored_meta.get("units", "config"),
        scale_source=stored_meta.get("scale_source", "orbit_prior"),
        provenance={
            "method": method,
            "intrinsics": stored_provenance.get("intrinsics", "config"),
            "source": str(results_path),
            "exported_by": "deeperfly calibration export",
            **{
                k: v
                for k, v in stored_provenance.items()
                if k not in ("method", "intrinsics", "source", "exported_by")
            },
        },
        quality=quality or stored_meta.get("quality") or {},
    )
    written = calibration.save(out)
    log.info("wrote %s", written)
    print(calibration.summary())
