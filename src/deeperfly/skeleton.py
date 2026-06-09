"""Tracked-point skeleton for multi-view pose (Drosophila by default).

A :class:`Skeleton` is the rig-independent description of *what* is tracked: the
ordered tracked points, their grouping into limbs, the bones (edges) connecting
them, and -- for a known camera rig -- which points each named camera can see.
It carries no geometry; it is consumed by triangulation (to mask unobservable
points), bundle adjustment (bone-length priors), correction (per-side Procrustes)
and visualization (drawing bones).

The default fly skeleton is packaged as ``data/skeleton_fly.toml``; it tracks the
same 38 points as NeLy-EPFL/DeepFly3D's ``skeleton_fly.py`` but orders the body
sides left-first (left ``0..18``, right ``19..37``), with 10 limbs and 28
within-leg/stripe bones. Load it with :meth:`Skeleton.fly`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Int

if TYPE_CHECKING:
    from .config import Config

_DATA_DIR = Path(__file__).parent / "data"
_FLY_TOML = _DATA_DIR / "skeleton_fly.toml"


@dataclass(frozen=True)
class Skeleton:
    """An ordered set of tracked points with limb/bone structure and visibility.

    Attributes
    ----------
    name
        Identifier for the skeleton (e.g. ``"fly38"``).
    joint_names
        Human-readable name per tracked point, in order (length ``n_points``).
    limb_names, limb_id, bones
        Limb structure derived from the config's ``limb_joints`` mapping (see
        :func:`_parse_limb_joints`): the limb names (length ``n_limbs``), each
        point's limb index (shape ``(n_points,)``), and the within-view 2D edges
        as point-index pairs (shape ``(n_bones, 2)``).
    palette
        Mapping ``limb_name -> hex color`` for plotting. Limbs absent from the
        mapping fall back to a default colormap in the visualization helpers.
    visibility
        Mapping ``camera_name -> array of visible point indices`` for a known
        rig. Cameras absent from the mapping are treated as seeing every point.
    """

    name: str
    joint_names: tuple[str, ...]
    limb_names: tuple[str, ...]
    limb_id: Int[np.ndarray, "N"]
    bones: Int[np.ndarray, "B 2"]
    palette: dict[str, str]
    visibility: dict[str, Int[np.ndarray, "M"]]

    # -- construction --------------------------------------------------------

    @classmethod
    def fly(cls) -> Skeleton:
        """The default 38-point Drosophila skeleton (DeepFly3D 7-camera rig)."""
        return cls.from_config(_FLY_TOML)

    @classmethod
    def from_config(cls, config: "Config | dict | str | Path") -> Skeleton:
        """Build a skeleton from a :class:`~deeperfly.config.Config`, a config dict
        or a path to a TOML file."""
        from .config import Config

        spec = Config.coerce(config).data["skeleton"]
        n = len(spec["joint_names"])
        limb_names, limb_id, bones = _parse_limb_joints(spec.get("limb_joints", {}), n)
        palette = {str(k): str(v) for k, v in spec.get("palette", {}).items()}
        visibility = {
            name: np.asarray(idx, dtype=np.int64)
            for name, idx in spec.get("visibility", {}).items()
        }
        for name, idx in visibility.items():
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"visibility[{name!r}] has out-of-range indices")
        return cls(
            name=spec.get("name", "skeleton"),
            joint_names=tuple(spec["joint_names"]),
            limb_names=limb_names,
            limb_id=limb_id,
            bones=bones,
            palette=palette,
            visibility=visibility,
        )

    # -- basic views ---------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.joint_names)

    @property
    def n_limbs(self) -> int:
        return len(self.limb_names)

    def __len__(self) -> int:
        return self.n_points

    # -- derived structure ---------------------------------------------------

    def visibility_mask(self, camera_names: list[str]) -> Bool[np.ndarray, "V N"]:
        """Boolean ``(V, N)`` mask: can camera ``v`` see point ``n``?

        Cameras without an entry in :attr:`visibility` are taken to see every
        point (all ``True``), so an unknown rig degrades to "everything visible".
        """
        mask = np.ones((len(camera_names), self.n_points), dtype=bool)
        for v, name in enumerate(camera_names):
            if name in self.visibility:
                mask[v] = False
                mask[v, self.visibility[name]] = True
        return mask

    def bone_index_pairs(
        self,
    ) -> tuple[Int[np.ndarray, "B"], Int[np.ndarray, "B"]]:
        """Endpoint index arrays ``(i, j)`` for vectorized bone-length maths."""
        return self.bones[:, 0], self.bones[:, 1]


def _parse_limb_joints(
    limb_joints: dict[str, list[int]], n_points: int
) -> tuple[tuple[str, ...], Int[np.ndarray, "N"], Int[np.ndarray, "B 2"]]:
    """Expand a ``{limb_name: [joint_indices]}`` mapping into limb structure.

    ``limb_joints`` is the single source of truth for a skeleton's limbs: each
    entry lists a limb's points in kinematic-chain order. From it we derive

    * ``limb_names`` -- the mapping keys, in order;
    * ``limb_id`` -- each point's limb (its key's position), shape ``(n_points,)``;
      points absent from every limb get ``-1``;
    * ``bones`` -- the within-limb 2D edges, i.e. consecutive points of each
      chain (so a single-point limb such as an antenna contributes none).
    """
    limb_names = tuple(limb_joints)
    limb_id = np.full(n_points, -1, dtype=np.int64)
    bones: list[list[int]] = []
    for lid, joints in enumerate(limb_joints.values()):
        joints = [int(j) for j in joints]
        for j in joints:
            if not 0 <= j < n_points:
                raise ValueError(
                    f"limb {limb_names[lid]!r} references point index {j} "
                    f"outside [0, {n_points})"
                )
            limb_id[j] = lid
        bones.extend([a, b] for a, b in zip(joints, joints[1:]))
    return limb_names, limb_id, _edges(bones, n_points, "bones")


def _edges(raw: list, n_points: int, what: str) -> Int[np.ndarray, "E 2"]:
    """Validate and pack a list of index pairs into an ``(E, 2)`` int array."""
    arr = (
        np.asarray(raw, dtype=np.int64).reshape(-1, 2)
        if raw
        else np.empty((0, 2), np.int64)
    )
    if arr.size and (arr.min() < 0 or arr.max() >= n_points):
        raise ValueError(f"{what} reference a point index outside [0, {n_points})")
    return arr
