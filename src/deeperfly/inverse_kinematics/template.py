"""The kinematic template: the NeuroMechFly-style leg model the IK fits.

A :class:`KinematicTemplate` describes *what* to fit, independent of any recording:
the per-leg serial chains (ordered joints, each with its revolute DOF axes and
angle bounds) and which skeleton point marks each joint. It is one of the three
sources :mod:`deeperfly.inverse_kinematics.bodyplan` assembles into the body plan
QuickIK solves; the head and abdomen come from the baked
:mod:`deeperfly.inverse_kinematics.articulation` instead, since their joints are not
keypoints.

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

from .binding import Binding
from .pack import MODELS

__all__ = [
    "Dof",
    "Joint",
    "LegChain",
    "KinematicTemplate",
    "DEFAULT_TEMPLATE_PATH",
    "TEMPLATES",
]

#: The leg template of each packaged model pack, ``pack name -> path``. A template is
#: selected through its pack (``[inverse_kinematics] model``), not on its own; this
#: mapping is what lets a bare name still resolve.
TEMPLATES = {name: path.parent / "template.toml" for name, path in MODELS.items()}

#: The packaged NeuroMechFly leg template.
DEFAULT_TEMPLATE_PATH = TEMPLATES["neuromechfly"]


@dataclass(frozen=True)
class Dof:
    """One revolute degree of freedom: a rotation about ``axis`` bounded to ``[lo, hi]``.

    ``axis`` is a unit vector in the model frame; ``lo`` / ``hi`` are the angle limits in
    **radians** (the file gives degrees; the loader converts). A DOF marked ``mirror`` in
    the template has its axis negated on the right side, because the model does: the right
    legs' hinges carry ``axis="-1 0 0"`` / ``"0 0 -1"`` where the left carry ``+1``. That
    keeps a fitted angle equal to *flygym's own* angle on both sides, rather than its
    negation on one of them.
    """

    name: str
    axis: tuple[float, float, float]
    lo: float
    hi: float
    #: The model's own name for this DOF's angle -- the key a fitted angle is reported
    #: under, a ``[inverse_kinematics.bounds]`` override is looked up by, and the
    #: model's spring reference is read by. Declared rather than composed, so the
    #: ``<joint>-<dof>`` convention is flygym's and not deeperfly's; the file's default
    #: is that convention, which is why the packaged template writes none.
    angle: str = ""


@dataclass(frozen=True)
class Joint:
    """A joint in a leg chain: the model ``body`` it sits at, and its DOFs.

    ``point`` is the tracked point that observes it, resolved through the binding at
    load time -- the template itself names no skeleton point.

    ``segment`` names the segment that leads *into* this joint from its parent
    (``""`` for the root ThC, which sits at the chain origin); the segment's length
    is measured from the data at solve time. ``joint`` is the flygym joint base name
    ``<parent_body>-<child_body>`` for this leg (e.g. ``"c_thorax-rf_coxa"``), and each
    DOF names its own angle (:attr:`Dof.angle`).

    ``quat`` is the constant rotation from the parent joint's post-DOF frame into this
    joint's own, in ``(w, x, y, z)`` -- the model body's own orientation, which the DOF
    axes are then expressed in. NeuroMechFly's leg bodies all carry ``quat="1 0 0 0"``,
    so the packaged template declares none and every joint is identity; flybody's carry
    real rotations, which is why this is a field rather than an assumption.
    """

    name: str
    body: str
    point: str
    segment: str
    joint: str
    dofs: tuple[Dof, ...]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LegChain:
    """One leg's serial chain (ThC -> CTr -> FTi -> TiTa -> Pretarsus)."""

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
        """Every DOF's declared angle name, in chain order.

        E.g. ``c_thorax-rf_coxa-roll``, ``rf_coxa-rf_trochanterfemur-pitch`` for the
        packaged template, which leaves them on flygym's ``<joint>-<dof>`` convention.
        """
        return [d.angle for j in self.joints for d in j.dofs]

    @property
    def quats(self) -> np.ndarray:
        """``(J, 4)`` each joint's constant ``(w, x, y, z)`` offset rotation."""
        return np.asarray([j.quat for j in self.joints], dtype=float)

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
class KinematicTemplate:
    """A loaded, validated kinematic template (the leg chains)."""

    name: str
    legs: tuple[LegChain, ...]
    #: The direction a segment extends along in its parent joint's frame, from the
    #: pack manifest. ``-z`` for NeuroMechFly, whose leg bodies sit at a pure ``-z``
    #: offset from their parent; flybody's run along ``+y``.
    rest_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)

    # -- construction --------------------------------------------------------

    @classmethod
    def load(
        cls,
        ref: str | Path = "neuromechfly",
        *,
        legs: list[str] | None = None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
        rest_axis: tuple[float, float, float] | None = None,
        binding: "Binding | str | Path | None" = "fly38@neuromechfly",
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
        rest_axis
            The model's segment rest direction, from its pack manifest. ``None`` keeps
            the default, which is NeuroMechFly's ``-z``.
        binding
            The (skeleton, model) binding, which names the tracked point observing each
            joint's body -- a :class:`~deeperfly.inverse_kinematics.binding.Binding`, or
            a ``"<skeleton>@<model>"`` reference to load. Defaults to the packaged pair,
            which is what a caller reading the model's structure wants; a run resolves
            its own through :meth:`deeperfly.config.Config.ik_binding`. ``None`` leaves
            every joint's ``point`` empty -- enough to read the structure, not to fit.

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
        if binding is not None and not isinstance(binding, Binding):
            binding = Binding.load(binding)
        spec = tomllib.loads(path.read_text())
        return cls.from_spec(
            spec,
            legs=legs,
            bounds_overrides=bounds_overrides,
            rest_axis=rest_axis,
            binding=binding,
        )

    @classmethod
    def from_spec(
        cls,
        spec: dict,
        *,
        legs: list[str] | None = None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
        rest_axis: tuple[float, float, float] | None = None,
        binding: "Binding | None" = None,
    ) -> "KinematicTemplate":
        """Expand a parsed template mapping into per-leg chains."""
        overrides = {k.lower(): v for k, v in (bounds_overrides or {}).items()}
        # A leg entry is its id, or a table declaring the side too. Inferring the side
        # from the id's first letter is what a template named `T1_left` gets silently
        # wrong -- mirrored axes and the wrong bounds, with no error -- so the fallback
        # is kept only for a template that predates the field.
        declared = {}
        all_legs: list[str] = []
        for entry in spec.get("legs", []):
            if isinstance(entry, dict):
                name = str(entry["name"])
                declared[name] = str(entry["side"])
            else:
                name = str(entry)
            all_legs.append(name)
        chosen = all_legs if legs is None else list(legs)
        unknown = [leg for leg in chosen if leg not in all_legs]
        if unknown:
            raise ValueError(
                f"[inverse_kinematics].legs has unknown leg(s) {unknown}; "
                f"the template defines {all_legs}"
            )
        # A flat [bounds] table applies to both sides; [bounds.l] / [bounds.r] override
        # it per side. The packaged template needs only the flat one now that a mirrored
        # DOF negates its axis instead of its range -- the per-side tables it used to
        # carry were exactly each other's negation, compensating for a shared axis.
        raw = dict(spec.get("bounds", {}))
        flat = {k: v for k, v in raw.items() if k not in ("l", "r")}
        bounds_by_side = {
            side: {**flat, **dict(raw.get(side, {}))} for side in ("l", "r")
        }
        chains = tuple(
            _build_leg(
                leg,
                declared.get(leg),
                spec["joints"],
                bounds_by_side,
                overrides,
                binding,
            )
            for leg in chosen
        )
        rest = rest_axis if rest_axis is not None else (0.0, 0.0, -1.0)
        return cls(
            name=spec.get("name", "template"),
            legs=chains,
            rest_axis=tuple(float(v) for v in rest),  # type: ignore[arg-type]
        )

    # -- views ---------------------------------------------------------------

    @property
    def dof_names(self) -> list[str]:
        """Every leg DOF name, in template order."""
        return [name for leg in self.legs for name in leg.dof_names]

    @property
    def model_point_names(self) -> list[str]:
        """The skeleton points the legs predict (every leg joint, in chain order)."""
        out: list[str] = []
        for leg in self.legs:
            for name in leg.point_names:
                if name not in out:
                    out.append(name)
        return out


def _build_leg(
    leg: str,
    side: str | None,
    joints_spec: list[dict],
    bounds_by_side: dict[str, dict],
    overrides: dict[str, tuple[float, float]],
    binding: "Binding | None" = None,
) -> LegChain:
    """Expand the generic joint list into one leg's chain with side-specific bounds."""
    side = side if side is not None else ("l" if leg.lower().startswith("l") else "r")
    side_bounds = bounds_by_side.get(side, {})
    joints: list[Joint] = []
    for jspec in joints_spec:
        jname = jspec["name"]
        body = str(jspec["body"]).format(leg=leg)
        point = "" if binding is None else (binding.point_for(body) or "")
        joint = jspec.get("joint", "").format(
            leg=leg
        )  # flygym base, "c_thorax-rf_coxa"
        dofs: list[Dof] = []
        for dspec in jspec.get("dofs", []):
            side_key = f"{jname}_{dspec['name']}"  # side-default key, e.g. "ThC_roll"
            # The model's own name for the angle. `{joint}-{dof}` is flygym's scheme
            # and stays the default, so declaring it is only needed for a model that
            # names its DOFs some other way.
            dof_name = str(dspec.get("angle", "{joint}-{dof}")).format(
                joint=joint, dof=dspec["name"], leg=leg
            )
            lo_deg, hi_deg = _resolve_bounds(
                dof_name, side_bounds.get(side_key), overrides
            )
            # The model mirrors these axes on the right; mirroring here too is what
            # makes a right leg's fitted angle flygym's own value and not its negation.
            flip = -1.0 if (dspec.get("mirror") and side == "r") else 1.0
            dofs.append(
                Dof(
                    name=dspec["name"],
                    axis=tuple(flip * float(a) for a in dspec["axis"]),
                    lo=float(np.deg2rad(lo_deg)),
                    hi=float(np.deg2rad(hi_deg)),
                    angle=dof_name,
                )
            )
        quat = jspec.get("quat", (1.0, 0.0, 0.0, 0.0))
        joints.append(
            Joint(
                name=jname,
                body=body,
                point=point,
                segment=str(jspec.get("segment", "")),
                joint=joint,
                dofs=tuple(dofs),
                quat=tuple(float(v) for v in quat),  # type: ignore[arg-type]
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
