"""Tracked-point skeleton for multi-view pose (Drosophila by default).

A :class:`Skeleton` is the rig-independent description of *what* is tracked: the
ordered tracked points, their grouping into limbs, the bones (edges) connecting
them, and -- for a known camera rig -- which points each named camera can see.
It carries no geometry; it is consumed by triangulation (to mask unobservable
points), bundle adjustment (bone-length priors), pictorial-structures recovery
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
from jaxtyping import Int

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

    Which view sees which point is no longer carried here: it is intrinsic to the
    detection plan (the pathways' ``(channel, view, point)`` mappings), and an
    unobserved ``(view, point)`` is simply ``NaN`` in the points array.
    """

    name: str
    joint_names: tuple[str, ...]
    limb_names: tuple[str, ...]
    limb_id: Int[np.ndarray, "N"]
    bones: Int[np.ndarray, "B 2"]
    palette: dict[str, str]

    # -- construction --------------------------------------------------------

    @classmethod
    def fly(cls) -> Skeleton:
        """The default 38-point Drosophila skeleton (DeepFly3D 7-camera rig)."""
        from .config import Config

        return cls.from_config(Config.from_toml(_FLY_TOML))

    @classmethod
    def from_config(cls, config: "Config") -> Skeleton:
        """Build a skeleton from a config.

        Parameters
        ----------
        config
            A :class:`~deeperfly.config.Config` with a ``[skeleton]`` table.

        Returns
        -------
        Skeleton
            The skeleton described by the config's ``[skeleton]`` table.

        Raises
        ------
        ValueError
            If a ``visibility`` entry has out-of-range point indices.
        """
        spec = config.data["skeleton"]
        limb_names, limb_id, bones = _parse_limb_joints(
            spec.get("limb_joints", {}), len(spec["joint_names"])
        )
        palette = {str(k): str(v) for k, v in spec.get("palette", {}).items()}
        return cls(
            name=spec.get("name", "skeleton"),
            joint_names=tuple(spec["joint_names"]),
            limb_names=limb_names,
            limb_id=limb_id,
            bones=bones,
            palette=palette,
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

    def bone_index_pairs(
        self,
    ) -> tuple[Int[np.ndarray, "B"], Int[np.ndarray, "B"]]:
        """Endpoint index arrays ``(i, j)`` for vectorized bone-length maths.

        Returns
        -------
        i, j : np.ndarray
            The first and second endpoint index of each bone (shape ``(B,)``).
        """
        return self.bones[:, 0], self.bones[:, 1]


def _parse_limb_joints(
    limb_joints: dict[str, list[int]], n_points: int
) -> tuple[tuple[str, ...], Int[np.ndarray, "N"], Int[np.ndarray, "B 2"]]:
    """Expand a ``{limb_name: [joint_indices]}`` mapping into limb structure.

    ``limb_joints`` is the single source of truth for a skeleton's limbs: each
    entry lists a limb's points in kinematic-chain order.

    Parameters
    ----------
    limb_joints
        Mapping ``limb_name -> [point indices]`` in kinematic-chain order.
    n_points
        Total number of tracked points (for index validation).

    Returns
    -------
    limb_names : tuple of str
        The mapping keys, in order.
    limb_id : np.ndarray
        Each point's limb index (shape ``(n_points,)``); points absent from
        every limb get ``-1``.
    bones : np.ndarray
        The within-limb 2D edges, i.e. consecutive points of each chain (a
        single-point limb such as an antenna contributes none).

    Raises
    ------
    ValueError
        If a limb references a point index outside ``[0, n_points)``.
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
    """Validate and pack a list of index pairs into an ``(E, 2)`` int array.

    Parameters
    ----------
    raw
        A list of ``[i, j]`` index pairs (or empty).
    n_points
        Total number of tracked points (for index validation).
    what
        Label naming the edge kind, used in the error message.

    Returns
    -------
    np.ndarray
        The packed ``(E, 2)`` int64 edge array.

    Raises
    ------
    ValueError
        If any index is outside ``[0, n_points)``.
    """
    arr = (
        np.asarray(raw, dtype=np.int64).reshape(-1, 2)
        if raw
        else np.empty((0, 2), np.int64)
    )
    if arr.size and (arr.min() < 0 or arr.max() >= n_points):
        raise ValueError(f"{what} reference a point index outside [0, {n_points})")
    return arr
