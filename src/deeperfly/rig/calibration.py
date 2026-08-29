"""The calibration artifact: a solved camera rig as a standalone, shareable file.

A deeperfly config describes cameras as an **orbit** -- ``distance`` away from a
``look_at`` target at some azimuth/elevation (:func:`deeperfly.rig.cameras.resolve_extrinsics`).
That is a good way for a human to *specify* a rig they built, and a deliberately bad way
to write down one a *solver* produced: bundle adjustment returns an arbitrary
``(rvec, tvec)`` per camera, and the orbit parser rejects those keys outright rather than
accept a half-specified rig. So until now a refined rig could only live inside one
recording's ``results.h5`` (``bundle_adjustment/cameras``), where it could not be shared
with the next recording, compared against a later solve, or reviewed before use.

This module is the missing file. A :class:`Calibration` is a solved rig plus the three
things that make one safe to reuse:

- the **pixel frame** its intrinsics describe (``image_sizes``), so applying it to
  rescaled or cropped footage fails loudly instead of silently misprojecting;
- its **provenance** -- how it was obtained, from what, with which solver settings;
- its **quality** -- the reprojection residuals, because a rig without its residuals is a
  number you cannot refuse.

.. code-block:: text

    [calibration]                 format_version, name, created_utc, units, scale_source
    [calibration.provenance]      method, recordings, frames, solver, intrinsics
    [calibration.quality]         rms_reproj_px, p90_reproj_px, per_camera_rms_px, ...
    [calibration.cameras.<name>]  rvec, tvec, intr, dist, image_size

**Which pixel frame.** Intrinsics are in **raw footage pixels** -- the frame the operator
sees and clicks in, and the frame every ``(V, T, P, 2)`` array in deeperfly is expressed
in. This is worth stating because :class:`~deeperfly.preprocessing.FrameTransform` *can*
map intrinsics into a preprocessed frame (:meth:`~deeperfly.preprocessing.FrameTransform.map_intrinsics`)
and :meth:`deeperfly.rig.cameras.Camera.from_spec` accepts a ``transform`` -- but
:meth:`deeperfly.rig.cameras.CameraGroup.from_config` never passes one, and a pathway's
preprocessing is inverted back into the view's raw frame before any point is stored
(:func:`deeperfly.pose2d.pathways.normalized_peaks_to_original_pixels`). Raw footage
pixels is therefore the one coordinate system a calibration ever needs to name.

**Units.** A rig solved from correspondences alone is determined only up to scale, so
``units`` is ``"arbitrary"`` unless something fixed it (a known distance, a bone-length
prior, a calibration board) -- recorded in ``scale_source``. Arbitrary units are fine for
angles and useless for velocities, so the distinction is stored rather than assumed.

TOML is written by a small writer in this module rather than a dependency: the schema is
fixed and shallow, floats round-trip exactly through ``repr``, and a hand-rolled writer
can emit the comments that make the file self-explanatory to whoever opens it next.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .._toml import key as _key
from .._toml import table_lines as _table_lines
from .._toml import value as _value
from .cameras import Camera, CameraGroup

__all__ = [
    "Calibration",
    "CALIBRATION_FORMAT_VERSION",
    "CALIBRATION_FILENAME",
    "UNITS",
    "SCALE_SOURCES",
    "METHODS",
    "quality_from_errors",
]

log = logging.getLogger("deeperfly")

#: Bumped when the on-disk schema changes incompatibly. A file written by a *newer*
#: deeperfly is refused rather than silently misread (see :meth:`Calibration.load`).
CALIBRATION_FORMAT_VERSION = 1

#: The conventional filename, so a directory can be handed around instead of a file.
CALIBRATION_FILENAME = "calibration.toml"

#: What a length in this rig means, and there are only two answers.
#: ``"arbitrary"`` is not a defect -- it is the honest state of a rig solved from
#: correspondences, which cannot determine scale at all: a rig twice as large viewing a
#: fly twice as large produces pixel-identical images. ``"config"`` means "whatever unit
#: ``[default_camera] distance`` was written in": the scale is determined, but deeperfly
#: has never been told what it measures.
#:
#: ``"mm"`` went in 0.3.0 with calibration-stage scale pinning. Physical scale is not a
#: calibration-stage concern -- everything through triangulation is arbitrary units by
#: design, and scale first becomes physical at inverse kinematics, where
#: :attr:`~deeperfly.inverse_kinematics.IKResult.body_scale` fits the point cloud to the
#: fitted model's defined dimensions. A calibration claiming millimeters was claiming
#: something the images could not have told it.
UNITS = ("arbitrary", "config")

#: What fixed the scale (``"none"`` leaves ``units = "arbitrary"``). ``"orbit_prior"`` is
#: bundle adjustment started from a hand-specified config rig: the seven gauge freedoms
#: are unconstrained directions the solver has no reason to move along, so the scale
#: stays where the orbit put it. ``"known_distance"`` went with ``"mm"`` (see
#: :data:`UNITS`).
SCALE_SOURCES = ("none", "orbit_prior", "bone_prior", "board", "imported")

#: How the rig was obtained. ``"orbit_prior"`` is an unrefined config rig promoted to a
#: calibration; ``"labels_ba"`` is bundle adjustment over observations.
METHODS = ("orbit_prior", "labels_ba", "board", "imported")


# -- the artifact --------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """A solved camera rig plus the metadata that makes it safe to reuse.

    Attributes
    ----------
    cameras
        The rig itself. Intrinsics are in **raw footage pixels** (see the module
        docstring).
    image_sizes
        ``camera_name -> (height, width)`` of the raw footage the intrinsics describe.
        Checked against the footage a consumer actually has
        (:meth:`check_image_sizes`); may be empty for a rig whose spec carried an
        explicit ``principal_point_px`` and whose footage size was never recorded.
    name
        A human-facing label, conventionally the file's stem.
    units
        One of :data:`UNITS`.
    scale_source
        One of :data:`SCALE_SOURCES`.
    provenance
        Free-form record of how this rig was produced (method, inputs, solver settings).
    quality
        Free-form record of how well it fits (see :func:`quality_from_errors`).
    created_utc
        ISO-8601 timestamp; filled in on :meth:`save` when absent.
    """

    cameras: CameraGroup
    image_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    name: str = "calibration"
    units: str = "arbitrary"
    scale_source: str = "none"
    provenance: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    created_utc: str | None = None

    @property
    def camera_names(self) -> list[str]:
        return self.cameras.names

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_camera_group(
        cls,
        cameras: CameraGroup,
        *,
        name: str = "calibration",
        image_sizes: dict[str, tuple[int, int]] | None = None,
        units: str = "arbitrary",
        scale_source: str = "none",
        provenance: dict | None = None,
        quality: dict | None = None,
    ) -> Calibration:
        """Wrap a solved :class:`~deeperfly.rig.cameras.CameraGroup` as an artifact.

        Parameters
        ----------
        cameras
            The rig to record (raw-footage-pixel intrinsics).
        name
            A human-facing label for the calibration.
        image_sizes
            ``camera_name -> (height, width)`` of the raw footage. Entries for cameras
            not in ``cameras`` are dropped, so a caller can pass a whole run's map.
        units, scale_source
            See :data:`UNITS` / :data:`SCALE_SOURCES`; validated.
        provenance, quality
            Free-form metadata blocks.

        Returns
        -------
        Calibration
            The artifact, ready to :meth:`save`.
        """
        _check_choice("units", units, UNITS)
        _check_choice("scale_source", scale_source, SCALE_SOURCES)
        sizes = {
            str(n): (int(hw[0]), int(hw[1]))
            for n, hw in (image_sizes or {}).items()
            if n in cameras.cameras
        }
        return cls(
            cameras=cameras,
            image_sizes=sizes,
            name=str(name),
            units=str(units),
            scale_source=str(scale_source),
            provenance=dict(provenance or {}),
            quality=dict(quality or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        """Read a calibration TOML.

        Parameters
        ----------
        path
            A ``calibration.toml``, or a directory containing one.

        Returns
        -------
        Calibration
            The loaded artifact.

        Raises
        ------
        FileNotFoundError
            If ``path`` (or ``path/calibration.toml``) does not exist.
        ValueError
            If the file was written by a newer deeperfly, defines no cameras, or a
            camera is missing a required key.
        """
        p = resolve_path(path)
        data = tomllib.loads(p.read_text())
        cal = data.get("calibration")
        if not isinstance(cal, dict):
            raise ValueError(f"{p} has no [calibration] table")
        version = int(cal.get("format_version", 1))
        if version > CALIBRATION_FORMAT_VERSION:
            raise ValueError(
                f"{p} was written by a newer deeperfly (calibration format v{version}, "
                f"this build understands v{CALIBRATION_FORMAT_VERSION}); refusing to "
                "read it rather than silently dropping state it carries"
            )
        specs = cal.get("cameras")
        if not isinstance(specs, dict) or not specs:
            raise ValueError(f"{p} defines no cameras ([calibration.cameras.<name>])")

        cameras: dict[str, Camera] = {}
        image_sizes: dict[str, tuple[int, int]] = {}
        for cam_name, spec in specs.items():
            if not isinstance(spec, dict):
                raise ValueError(f"{p}: camera {cam_name!r} is not a table")
            cameras[cam_name] = Camera(
                rvec=_vec(p, cam_name, spec, "rvec", 3),
                tvec=_vec(p, cam_name, spec, "tvec", 3),
                intr=_vec(p, cam_name, spec, "intr", 4),
                dist=np.asarray(spec.get("dist", []), dtype=float).reshape(-1),
                name=cam_name,
            )
            size = spec.get("image_size")
            if size is not None:
                if len(size) != 2:
                    raise ValueError(
                        f"{p}: camera {cam_name!r} image_size must be [height, width]"
                    )
                image_sizes[cam_name] = (int(size[0]), int(size[1]))
        return cls(
            cameras=CameraGroup(cameras),
            image_sizes=image_sizes,
            name=str(cal.get("name", p.stem)),
            units=str(cal.get("units", "arbitrary")),
            scale_source=str(cal.get("scale_source", "none")),
            provenance=dict(cal.get("provenance", {})),
            quality=dict(cal.get("quality", {})),
            created_utc=cal.get("created_utc"),
        )

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the calibration TOML (overwriting ``path``).

        Parameters
        ----------
        path
            Destination file, or a directory (which gets a
            :data:`CALIBRATION_FILENAME`). Parent directories are created.

        Returns
        -------
        Path
            The file written.
        """
        out = Path(path)
        if out.is_dir() or not out.suffix:
            out = out / CALIBRATION_FILENAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_toml())
        return out

    def to_toml(self) -> str:
        """The artifact as TOML text (what :meth:`save` writes)."""
        created = self.created_utc or datetime.now(timezone.utc).isoformat()
        lines = [
            "# deeperfly camera calibration -- a solved rig, portable across recordings.",
            "#",
            "# Intrinsics are in RAW FOOTAGE PIXELS: `intr` is [fx, fy, cx, cy] in the",
            "# pixels of the frame named by `image_size` ([height, width]). Applying this",
            "# rig to differently-sized footage is refused, not rescaled.",
            "#",
            "# Extrinsics map world -> camera as `R(rvec) @ X + tvec`. Use it with:",
            "#",
            "#     [cameras]",
            '#     calibration = "calibration.toml"   # in a run config, or',
            "#",
            "#     CameraGroup.from_calibration(path)  # from the library",
            "",
            "[calibration]",
            f"format_version = {CALIBRATION_FORMAT_VERSION}",
            f"name = {_value(self.name)}",
            f"created_utc = {_value(created)}",
        ]
        lines += [
            "# 'arbitrary': the rig is determined only up to scale (correspondences alone",
            "# cannot fix it). Fine for angles, meaningless for velocities.",
            f"units = {_value(self.units)}",
            f"scale_source = {_value(self.scale_source)}",
        ]
        if self.provenance:
            lines += ["", "# How this rig was produced."]
            lines += _table_lines(["calibration", "provenance"], self.provenance)
        if self.quality:
            lines += [
                "",
                "# How well it fits. A rig without its residuals is a number you cannot",
                "# refuse -- read these before trusting the 3D that comes out of it.",
            ]
            lines += _table_lines(["calibration", "quality"], self.quality)
        for cam_name, cam in self.cameras.cameras.items():
            lines += ["", f"[calibration.cameras.{_key(cam_name)}]"]
            lines.append(f"rvec = {_value(np.asarray(cam.rvec, dtype=float))}")
            lines.append(f"tvec = {_value(np.asarray(cam.tvec, dtype=float))}")
            lines.append(f"intr = {_value(np.asarray(cam.intr, dtype=float))}")
            lines.append(f"dist = {_value(np.asarray(cam.dist, dtype=float))}")
            size = self.image_sizes.get(cam_name)
            if size is not None:
                lines.append(f"image_size = [{int(size[0])}, {int(size[1])}]")
        return "\n".join(lines) + "\n"

    # -- use ------------------------------------------------------------------

    def check_image_sizes(self, image_sizes: dict[str, tuple[int, int]] | None) -> None:
        """Raise if this rig does not describe the footage it is about to be used on.

        The analogue of the labels sidecar's identity check
        (:func:`deeperfly.labels.store._check_identity`), and for the same reason: the
        intrinsics are pixel quantities, so footage that was rescaled, cropped or
        rotated since the solve would be *silently* misprojected. A size this
        calibration does not record cannot be checked and is skipped, so a rig whose
        footage size was never recorded still loads.

        Parameters
        ----------
        image_sizes
            ``camera_name -> (height, width)`` of the footage in hand (``None`` skips
            the check entirely).

        Raises
        ------
        ValueError
            If a camera present in both maps has a different frame size.
        """
        if not image_sizes or not self.image_sizes:
            return
        bad = [
            f"{name}: calibrated for {tuple(self.image_sizes[name])}, "
            f"footage is {(int(hw[0]), int(hw[1]))}"
            for name, hw in image_sizes.items()
            if name in self.image_sizes
            and tuple(self.image_sizes[name]) != (int(hw[0]), int(hw[1]))
        ]
        if bad:
            raise ValueError(
                f"calibration {self.name!r} describes different footage than this run "
                "has (" + "; ".join(bad) + ") -- its intrinsics are pixel quantities, "
                "so using it here would silently misproject every point. Re-solve the "
                "rig for this footage, or point at the calibration that matches it."
            )

    def check_camera_names(self, names) -> list[str]:
        """Raise if the rig does not cover exactly the cameras ``names`` asks for.

        A calibration missing a camera cannot project it; a calibration carrying an
        extra one is a sign it belongs to a different rig. Both are reported by name
        rather than by count, because "6 of 7" is not actionable and "missing: f" is.

        Parameters
        ----------
        names
            The camera names the caller needs.

        Returns
        -------
        list of str
            The requested cameras this calibration actually covers, in the order asked
            for. A run uses THAT set: a view the rig never measured cannot be placed, so
            it is dropped like a view with no footage rather than refused -- one config
            routinely describes more rig than one solve covers (a project whose recordings
            predate a camera being added, say).

        Raises
        ------
        ValueError
            If the calibration covers NONE of the requested cameras. That is the
            wrong-rig case, and it is the one a subset cannot explain away.
        """
        wanted, have = list(names), set(self.cameras.cameras)
        missing = [n for n in wanted if n not in have]
        kept = [n for n in wanted if n in have]
        if missing and not kept:
            raise ValueError(
                f"calibration {self.name!r} covers none of the cameras this run needs "
                f"({wanted}); it covers {sorted(have)} -- it belongs to a different rig"
            )
        if missing:
            log.warning(
                "calibration %s does not cover camera(s) %s (it covers %s), so this run "
                "drops them and uses %d view(s): %s",
                self.name,
                missing,
                sorted(have),
                len(kept),
                kept,
            )
        extra = sorted(have - set(wanted))
        if extra:
            log.info(
                "calibration %s also covers camera(s) %s, which this run does not use",
                self.name,
                extra,
            )
        return kept

    def summary(self) -> str:
        """A human-readable report (what ``deeperfly calibration show`` prints)."""
        q, prov = self.quality, self.provenance
        head = [
            f"calibration: {self.name}",
            f"  created:      {self.created_utc or 'unknown'}",
            f"  cameras:      {len(self.cameras)}  ({', '.join(self.camera_names)})",
            f"  units:        {self.units}"
            + _scale_note(self.units, self.scale_source),
        ]
        if prov:
            head.append(f"  method:       {prov.get('method', 'unknown')}")
            if prov.get("intrinsics"):
                head.append(f"  intrinsics:   {prov['intrinsics']}")
            if prov.get("frames") is not None:
                head.append(f"  frames:       {prov['frames']}")
            if prov.get("recordings"):
                head.append(f"  recordings:   {', '.join(prov['recordings'])}")
        if q:
            head.append("  reprojection:")
            for label, key in (
                ("rms", "rms_reproj_px"),
                ("median", "median_reproj_px"),
                ("p90", "p90_reproj_px"),
                ("max", "max_reproj_px"),
            ):
                if q.get(key) is not None:
                    head.append(f"    {label:<7}{float(q[key]):8.3f} px")
            per_cam = q.get("per_camera_rms_px") or {}
            for cam in self.camera_names:
                if cam in per_cam:
                    head.append(f"    {cam:<7}{float(per_cam[cam]):8.3f} px rms")
        else:
            head.append("  reprojection: not recorded")
        for cam_name, cam in self.cameras.cameras.items():
            pos = np.asarray(cam.position, dtype=float)
            intr = np.asarray(cam.intr, dtype=float)
            size = self.image_sizes.get(cam_name)
            head.append(
                f"  {cam_name:<6} pos ({pos[0]:7.3f},{pos[1]:7.3f},{pos[2]:7.3f})  "
                f"f ({intr[0]:.1f}, {intr[1]:.1f})  c ({intr[2]:.1f}, {intr[3]:.1f})"
                + (f"  frame {size[1]}x{size[0]}" if size else "")
            )
        return "\n".join(head)


def _scale_note(units: str, scale_source: str) -> str:
    """The parenthetical that says what a length in this rig is worth.

    Spelled out rather than left to the reader, because the failure it guards against is
    silent: a rig with an unfixed scale reprojects perfectly and still yields velocities
    that are wrong by a constant nobody measured.
    """
    if scale_source == "none":
        return "  (nothing fixed the scale; angles are valid, lengths are not)"
    if units == "config":
        return (
            f"  (scale from {scale_source}, in whatever unit the config's "
            "[cameras] distance was written in)"
        )
    return f"  (scale from {scale_source})"


# -- quality -------------------------------------------------------------------


def quality_from_errors(errors, camera_names) -> dict:
    """The ``[calibration.quality]`` block from a ``(V, T, P)`` reprojection error array.

    A pure function of an array the bundle-adjustment stage already computes, so
    recording quality costs no extra solve. NaN entries (a point a view never observed)
    are ignored throughout rather than counted as zero error, which would make a sparse
    rig look better than a dense one.

    Parameters
    ----------
    errors
        Per-``(view, frame, point)`` reprojection error in pixels, NaN where unobserved.
    camera_names
        The view names labelling ``errors``' leading axis.

    Returns
    -------
    dict
        ``rms_reproj_px``, ``median_reproj_px``, ``p90_reproj_px``, ``max_reproj_px``,
        ``n_observations``, and ``per_camera_rms_px``. Empty when nothing was observed
        -- an empty block is honest, whereas zeros would read as a perfect fit.
    """
    err = np.asarray(errors, dtype=float)
    finite = np.isfinite(err)
    if not finite.any():
        return {}
    vals = err[finite]
    per_camera = {}
    for i, name in enumerate(camera_names):
        if i >= err.shape[0]:
            break
        cam = err[i][np.isfinite(err[i])]
        if cam.size:
            per_camera[str(name)] = float(np.sqrt(np.mean(cam**2)))
    return {
        "rms_reproj_px": float(np.sqrt(np.mean(vals**2))),
        "median_reproj_px": float(np.median(vals)),
        "p90_reproj_px": float(np.percentile(vals, 90)),
        "max_reproj_px": float(vals.max()),
        "n_observations": int(vals.size),
        "per_camera_rms_px": per_camera,
    }


def resolve_path(path: str | Path) -> Path:
    """A calibration file from a file *or* directory path.

    Parameters
    ----------
    path
        A ``.toml`` file, or a directory holding a :data:`CALIBRATION_FILENAME`.

    Returns
    -------
    Path
        The resolved file.

    Raises
    ------
    FileNotFoundError
        If nothing is there.
    """
    p = Path(path)
    if p.is_dir():
        p = p / CALIBRATION_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"no calibration at {p}")
    return p


# -- validation helpers --------------------------------------------------------


def _check_choice(what: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {list(allowed)}, got {value!r}")


def _vec(path: Path, cam: str, spec: dict, key: str, size: int) -> np.ndarray:
    """A required fixed-length float vector from a camera table."""
    if key not in spec:
        raise ValueError(f"{path}: camera {cam!r} is missing {key!r}")
    arr = np.asarray(spec[key], dtype=float).reshape(-1)
    if arr.size != size:
        raise ValueError(
            f"{path}: camera {cam!r} {key!r} must have {size} entries, got {arr.size}"
        )
    return arr
