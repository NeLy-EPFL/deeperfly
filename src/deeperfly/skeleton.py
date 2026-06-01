"""Tracked-point skeleton for multi-view pose (Drosophila by default).

A :class:`Skeleton` is the rig-independent description of *what* is tracked: the
ordered tracked points, their grouping into limbs, the bones (edges) connecting
them, and -- for a known camera rig -- which points each named camera can see.
It carries no geometry; it is consumed by triangulation (to mask unobservable
points), bundle adjustment (bone-length priors), correction (per-side Procrustes)
and visualization (drawing bones).

The default fly skeleton is packaged as ``data/skeleton_fly.toml`` and mirrors
NeLy-EPFL/DeepFly3D's ``skeleton_fly.py``: 38 points (right ``0..18``, left
``19..37``), 10 limbs, 28 within-leg/stripe bones, and one cross-body antenna
bone. Load it with :meth:`Skeleton.fly`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Int

_DATA_DIR = Path(__file__).parent / "data"
_FLY_TOML = _DATA_DIR / "skeleton_fly.toml"


@dataclass(frozen=True)
class Skeleton:
    """An ordered set of tracked points with limb/bone structure and visibility.

    Attributes
    ----------
    name
        Identifier for the skeleton (e.g. ``"drosophila"``).
    joint_names
        Human-readable name per tracked point, in order (length ``n_points``).
    limb_names
        Name per limb (length ``n_limbs``).
    limb_id
        Limb index for each tracked point, shape ``(n_points,)``.
    bones
        Within-view 2D edges as point-index pairs, shape ``(n_bones, 2)``.
    bones3d
        Cross-body edges meaningful only in 3D (e.g. antenna-antenna), shape
        ``(n_bones3d, 2)``.
    left_idx, right_idx
        Point indices of the left / right leg joints, used for separate-side
        Procrustes alignment.
    visibility
        Mapping ``camera_name -> array of visible point indices`` for a known
        rig. Cameras absent from the mapping are treated as seeing every point.
    """

    name: str
    joint_names: tuple[str, ...]
    limb_names: tuple[str, ...]
    limb_id: Int[np.ndarray, "N"]
    bones: Int[np.ndarray, "B 2"]
    bones3d: Int[np.ndarray, "B3 2"]
    left_idx: Int[np.ndarray, "L"]
    right_idx: Int[np.ndarray, "R"]
    visibility: dict[str, Int[np.ndarray, "M"]]

    # -- construction --------------------------------------------------------

    @classmethod
    def fly(cls) -> Skeleton:
        """The default 38-point Drosophila skeleton (DeepFly3D 7-camera rig)."""
        return cls.from_config(_FLY_TOML)

    @classmethod
    def from_config(cls, config: dict | str | Path) -> Skeleton:
        """Build a skeleton from a config dict or a path to a TOML file."""
        if not isinstance(config, dict):
            with open(config, "rb") as f:
                config = tomllib.load(f)
        spec = config["skeleton"]
        n = len(spec["joint_names"])
        limb_id = np.asarray(spec["limb_id"], dtype=np.int64)
        if limb_id.shape != (n,):
            raise ValueError(
                f"limb_id has {limb_id.size} entries but there are {n} joints"
            )
        bones = _edges(spec.get("bones", []), n, "bones")
        bones3d = _edges(spec.get("bones3d", []), n, "bones3d")
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
            limb_names=tuple(spec.get("limb_names", ())),
            limb_id=limb_id,
            bones=bones,
            bones3d=bones3d,
            left_idx=np.asarray(spec.get("left_points", []), dtype=np.int64),
            right_idx=np.asarray(spec.get("right_points", []), dtype=np.int64),
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
        self, include_3d: bool = False
    ) -> tuple[Int[np.ndarray, "B"], Int[np.ndarray, "B"]]:
        """Endpoint index arrays ``(i, j)`` for vectorized bone-length maths.

        With ``include_3d`` the cross-body :attr:`bones3d` edges are appended.
        """
        edges = self.bones
        if include_3d and self.bones3d.size:
            edges = np.concatenate([self.bones, self.bones3d], axis=0)
        return edges[:, 0], edges[:, 1]


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
