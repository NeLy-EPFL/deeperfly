"""Tracked-point skeleton for multi-view pose (Drosophila by default).

A :class:`Skeleton` is the rig-independent description of *what* is tracked: the
ordered tracked points, their grouping into limbs, the bones (edges) connecting
them, and -- for a known camera rig -- which points each named camera can see.
It carries no geometry; it is consumed by triangulation (to mask unobservable
points), bundle adjustment (bone-length priors), pictorial-structures recovery
and visualization (drawing bones).

The default fly skeleton is the ``[skeleton]`` section of the packaged
``data/default_config.toml``; it tracks the same 38 points as NeLy-EPFL/DeepFly3D's
``skeleton_fly.py`` but orders the body sides left-first (left ``0..18``, right
``19..37``), with 10 limbs and 28 within-leg/abdomen bones. Load it with
:meth:`Skeleton.fly`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Int

if TYPE_CHECKING:
    from .config import Config

__all__ = ["Skeleton"]


@dataclass(frozen=True)
class Skeleton:
    """An ordered set of tracked points with limb/bone structure and visibility.

    Attributes
    ----------
    name
        Identifier for the skeleton (e.g. ``"fly38"``).
    point_names
        Human-readable name per tracked point, in order (length ``n_points``).
    limb_names, limb_id, bones
        Limb structure derived from the config's ``limb_points`` mapping (see
        :func:`_parse_limb_points`): the limb names (length ``n_limbs``), each
        point's limb index (shape ``(n_points,)``), and the within-view 2D edges
        as point-index pairs (shape ``(n_bones, 2)``).
    palette
        Mapping ``limb_name -> hex color`` for plotting. Limbs absent from the
        mapping fall back to a default colormap in the visualization helpers.

    Which view sees which point lives in the detection plan (the pathways'
    ``(channel, view, point)`` mappings), not here: an unobserved ``(view, point)``
    is simply ``NaN`` in the points array.
    """

    name: str
    point_names: tuple[str, ...]
    limb_names: tuple[str, ...]
    limb_id: Int[np.ndarray, "P"]
    bones: Int[np.ndarray, "B 2"]
    palette: dict[str, str]

    # -- construction --------------------------------------------------------

    @classmethod
    def fly(cls) -> Skeleton:
        """The default 38-point Drosophila skeleton (DeepFly3D 7-camera rig)."""
        from .config import Config

        return cls.from_config(Config.default())

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
            If a ``limb_points`` entry names an unknown point or an
            out-of-range point index.
        """
        spec = config.data["skeleton"]
        point_names = tuple(spec["point_names"])
        limb_names, limb_id, bones = _parse_limb_points(
            spec.get("limb_points", {}), point_names
        )
        palette = {str(k): str(v) for k, v in spec.get("limb_palette", {}).items()}
        return cls(
            name=spec.get("name", "skeleton"),
            point_names=point_names,
            limb_names=limb_names,
            limb_id=limb_id,
            bones=bones,
            palette=palette,
        )

    # -- basic views ---------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.point_names)

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


def _parse_limb_points(
    limb_points: dict[str, list], point_names: tuple[str, ...]
) -> tuple[tuple[str, ...], Int[np.ndarray, "P"], Int[np.ndarray, "B 2"]]:
    """Expand a ``{limb_name: [points]}`` mapping into limb structure.

    ``limb_points`` is the single source of truth for a skeleton's limbs: each
    entry lists a limb's points in kinematic-chain order. A point may be given by
    its name (resolved against ``point_names``) or by its integer index.

    Parameters
    ----------
    limb_points
        Mapping ``limb_name -> [points]`` in kinematic-chain order; each point is
        a name in ``point_names`` or an integer index.
    point_names
        The ordered tracked-point names (for name resolution + index validation).

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
        If a limb names an unknown point or references an index outside
        ``[0, n_points)``.
    """
    n_points = len(point_names)
    index = {name: i for i, name in enumerate(point_names)}
    limb_names = tuple(limb_points)
    limb_id = np.full(n_points, -1, dtype=np.int64)
    bones: list[list[int]] = []
    for lid, points in enumerate(limb_points.values()):
        resolved = [_point_index(p, index, limb_names[lid]) for p in points]
        for j in resolved:
            if not 0 <= j < n_points:
                raise ValueError(
                    f"limb {limb_names[lid]!r} references point index {j} "
                    f"outside [0, {n_points})"
                )
            limb_id[j] = lid
        bones.extend([a, b] for a, b in zip(resolved, resolved[1:]))
    return limb_names, limb_id, _edges(bones, n_points, "bones")


def _point_index(point, index: dict[str, int], limb_name: str) -> int:
    """A limb point given by name or integer index -> its integer index."""
    if isinstance(point, str):
        if point not in index:
            raise ValueError(
                f"limb {limb_name!r} references unknown point name {point!r}"
            )
        return index[point]
    return int(point)


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
