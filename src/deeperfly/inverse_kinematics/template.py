"""The kinematic template: the NeuroMechFly-style leg/head model the IK fits.

A :class:`KinematicTemplate` describes *what* to fit, independent of any recording:
the per-leg serial chains (ordered joints, each with its revolute DOF axes and
angle bounds), which skeleton point marks each joint, and the head/antenna spec.
Segment *lengths* are deliberately not part of the template -- they are measured
from the data during alignment (:mod:`deeperfly.inverse_kinematics.align`) so the
fitted model reprojects tightly onto the real fly.

The packaged template is ``data/neuromechfly_template.toml`` (derived from
NeLy-EPFL/sequential-inverse-kinematics' ``body_config``); a run selects it by
name (``"neuromechfly"``) or a path and overrides any joint's bounds from the
``[inverse_kinematics.bounds]`` config table. The generic single leg chain in the
file is expanded over the six legs, with left/right bounds chosen per side.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Dof",
    "Joint",
    "LegChain",
    "HeadSpec",
    "KinematicTemplate",
    "DEFAULT_TEMPLATE_PATH",
    "TEMPLATES",
]

#: Packaged NeuroMechFly template selected by ``template = "neuromechfly"``.
DEFAULT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "data" / "neuromechfly_template.toml"
)

#: Named packaged templates (extend as more body models are added).
TEMPLATES = {"neuromechfly": DEFAULT_TEMPLATE_PATH}


@dataclass(frozen=True)
class Dof:
    """One revolute degree of freedom: a rotation about ``axis`` bounded to ``[lo, hi]``.

    ``axis`` is a unit vector in the leg-local frame; ``lo`` / ``hi`` are the angle
    limits in **radians** (the file gives degrees; the loader converts).
    """

    name: str
    axis: tuple[float, float, float]
    lo: float
    hi: float


@dataclass(frozen=True)
class Joint:
    """A joint in a leg chain: the skeleton ``point`` it sits on and its DOFs.

    ``segment`` names the segment that leads *into* this joint from its parent
    (``""`` for the root ThC, which sits at the chain origin); the segment's length
    is measured from the data at solve time. ``joint`` is the flygym joint base name
    ``<parent_body>-<child_body>`` for this leg (e.g. ``"c_thorax-rf_coxa"``); each
    DOF's angle name is ``<joint>-<dof>`` (``"c_thorax-rf_coxa-roll"``).
    """

    name: str
    point: str
    segment: str
    joint: str
    dofs: tuple[Dof, ...]


@dataclass(frozen=True)
class LegChain:
    """One leg's serial chain (ThC -> CTr -> FTi -> TiTa -> Claw)."""

    name: str  # "rf", "lm", ...
    side: str  # "r" or "l"
    joints: tuple[Joint, ...]

    @property
    def point_names(self) -> tuple[str, ...]:
        return tuple(j.point for j in self.joints)

    @property
    def dof_counts(self) -> tuple[int, ...]:
        return tuple(len(j.dofs) for j in self.joints)

    @property
    def dof_names(self) -> list[str]:
        """``<parent_body>-<child_body>-<dof>`` (the flygym joint name) for every DOF.

        E.g. ``c_thorax-rf_coxa-roll``, ``rf_coxa-rf_trochanterfemur-pitch``.
        """
        return [f"{j.joint}-{d.name}" for j in self.joints for d in j.dofs]

    @property
    def axes(self) -> np.ndarray:
        """``(D, 3)`` unit rotation axes, one per DOF in chain order."""
        out = [d.axis for j in self.joints for d in j.dofs]
        return np.asarray(out, dtype=float) if out else np.zeros((0, 3))

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` radian bound arrays of shape ``(D,)``."""
        lo = [d.lo for j in self.joints for d in j.dofs]
        hi = [d.hi for j in self.joints for d in j.dofs]
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


@dataclass(frozen=True)
class HeadSpec:
    """The head/antenna model: the antenna-tip skeleton points fit by the vector method."""

    antennae: tuple[str, ...]


@dataclass(frozen=True)
class KinematicTemplate:
    """A loaded, validated kinematic template (the legs + head spec)."""

    name: str
    legs: tuple[LegChain, ...]
    head: HeadSpec | None

    # -- construction --------------------------------------------------------

    @classmethod
    def load(
        cls,
        ref: str | Path = "neuromechfly",
        *,
        legs: list[str] | None = None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> "KinematicTemplate":
        """Load a template by name or path, optionally restricting legs / overriding bounds.

        Parameters
        ----------
        ref
            A packaged template name (``"neuromechfly"``) or a path to a template
            TOML file.
        legs
            Which legs to fit (subset of the file's ``legs``); ``None`` = all.
        bounds_overrides
            ``"<parent>-<child>-<dof>" -> (lo_deg, hi_deg)`` degree overrides keyed by
            the flygym joint name (e.g. ``{"rf_trochanterfemur-rf_tibia-pitch": (10,
            160)}``), applied after the per-side defaults. Case-insensitive.

        Returns
        -------
        KinematicTemplate
            The expanded per-leg template.
        """
        path = TEMPLATES.get(str(ref), None)
        path = Path(path) if path is not None else Path(ref)
        if not path.exists():
            raise FileNotFoundError(
                f"inverse-kinematics template {ref!r} not found "
                f"(known: {sorted(TEMPLATES)}, or a path to a template TOML)"
            )
        spec = tomllib.loads(path.read_text())
        return cls.from_spec(spec, legs=legs, bounds_overrides=bounds_overrides)

    @classmethod
    def from_spec(
        cls,
        spec: dict,
        *,
        legs: list[str] | None = None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> "KinematicTemplate":
        """Expand a parsed template mapping into per-leg chains."""
        overrides = {k.lower(): v for k, v in (bounds_overrides or {}).items()}
        all_legs = list(spec.get("legs", []))
        chosen = all_legs if legs is None else list(legs)
        unknown = [leg for leg in chosen if leg not in all_legs]
        if unknown:
            raise ValueError(
                f"[inverse_kinematics].legs has unknown leg(s) {unknown}; "
                f"the template defines {all_legs}"
            )
        bounds_by_side = {
            side: dict(spec.get("bounds", {}).get(side, {})) for side in ("l", "r")
        }
        chains = tuple(
            _build_leg(leg, spec["joints"], bounds_by_side, overrides) for leg in chosen
        )
        head_spec = spec.get("head")
        head = (
            HeadSpec(antennae=tuple(head_spec.get("antennae", [])))
            if head_spec
            else None
        )
        return cls(name=spec.get("name", "template"), legs=chains, head=head)

    # -- views ---------------------------------------------------------------

    @property
    def dof_names(self) -> list[str]:
        """Every leg DOF name, in template order."""
        return [name for leg in self.legs for name in leg.dof_names]

    @property
    def model_point_names(self) -> list[str]:
        """The skeleton points the model predicts (every leg joint, in order)."""
        out: list[str] = []
        for leg in self.legs:
            for name in leg.point_names:
                if name not in out:
                    out.append(name)
        for name in self.head.antennae if self.head else ():
            if name not in out:
                out.append(name)
        return out


def _build_leg(
    leg: str,
    joints_spec: list[dict],
    bounds_by_side: dict[str, dict],
    overrides: dict[str, tuple[float, float]],
) -> LegChain:
    """Expand the generic joint list into one leg's chain with side-specific bounds."""
    side = "l" if leg.lower().startswith("l") else "r"
    side_bounds = bounds_by_side.get(side, {})
    joints: list[Joint] = []
    for jspec in joints_spec:
        jname = jspec["name"]
        point = f"{leg}_{jspec['suffix']}"
        joint = jspec.get("joint", "").format(
            leg=leg
        )  # flygym base, "c_thorax-rf_coxa"
        dofs: list[Dof] = []
        for dspec in jspec.get("dofs", []):
            side_key = f"{jname}_{dspec['name']}"  # side-default key, e.g. "ThC_roll"
            dof_name = f"{joint}-{dspec['name']}"  # flygym DOF, "c_thorax-rf_coxa-roll"
            lo_deg, hi_deg = _resolve_bounds(
                dof_name, side_bounds.get(side_key), overrides
            )
            dofs.append(
                Dof(
                    name=dspec["name"],
                    axis=tuple(float(a) for a in dspec["axis"]),
                    lo=float(np.deg2rad(lo_deg)),
                    hi=float(np.deg2rad(hi_deg)),
                )
            )
        joints.append(
            Joint(
                name=jname,
                point=point,
                segment=str(jspec.get("segment", "")),
                joint=joint,
                dofs=tuple(dofs),
            )
        )
    return LegChain(name=leg, side=side, joints=tuple(joints))


def _resolve_bounds(
    dof_name: str,
    default: list | tuple | None,
    overrides: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Per-DOF degree bounds: a config override (keyed by the flygym DOF name) wins."""
    override = overrides.get(dof_name.lower())
    if override is not None:
        return float(override[0]), float(override[1])
    if default is None:
        # A DOF with no bound is left effectively free (the solver still needs a box).
        return -180.0, 180.0
    return float(default[0]), float(default[1])
