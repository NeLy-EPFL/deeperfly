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

    # -- point selection / merging -------------------------------------------

    #: Maps a category name to the substring that identifies its limbs by name.
    _CATEGORY_KEYWORD = {"legs": "leg", "antennae": "antenna", "stripes": "stripe"}

    def points_in_category(
        self, categories: str | tuple[str, ...] | list[str]
    ) -> Int[np.ndarray, "M"]:
        """Point indices belonging to the requested limb categories.

        ``categories`` is any subset of ``{"legs", "antennae", "stripes"}``
        (a single string is accepted too). A limb belongs to a category when its
        name contains the category keyword (``leg`` / ``antenna`` / ``stripe``,
        case-insensitive), so this works both before and after
        :meth:`merge_lr_stripes` (the merged stripe limb is still named
        ``stripe``). Raises ``ValueError`` for an unknown category.
        """
        if isinstance(categories, str):
            categories = (categories,)
        keywords = []
        for c in categories:
            if c not in self._CATEGORY_KEYWORD:
                raise ValueError(
                    f"unknown keypoint category {c!r}; "
                    f"expected a subset of {sorted(self._CATEGORY_KEYWORD)}"
                )
            keywords.append(self._CATEGORY_KEYWORD[c])
        limb_sel = [
            i
            for i, name in enumerate(self.limb_names)
            if any(kw in name.lower() for kw in keywords)
        ]
        return np.flatnonzero(np.isin(self.limb_id, limb_sel))

    def merge_lr_stripes(self) -> tuple[Skeleton, Int[np.ndarray, "N"]]:
        """Merge left/right abdominal stripe points into shared markers.

        ``r_stripe*`` and ``l_stripe*`` track the same physical markers on the
        abdomen but are stored as separate points seen by disjoint camera sets.
        This returns ``(merged_skeleton, remap)`` where the two stripe sets are
        collapsed into single prefix-stripped points (``stripe0/1/2``) so they
        can be triangulated from every camera that sees either side, and
        ``remap`` maps each old point index to its new index.

        Only stripes are merged (matched by ``"stripe"`` in the joint name);
        antennae and legs are untouched. If there is nothing to merge (e.g. an
        already-merged or non-fly skeleton) the skeleton is returned unchanged
        with an identity ``remap``, so the operation is idempotent.
        """
        n = self.n_points
        # Merge key: stripe points collapse by their prefix-stripped name; every
        # other point is its own singleton (keyed by index) so it never merges.
        keys: list = []
        for idx, name in enumerate(self.joint_names):
            if "stripe" in name.lower():
                keys.append(_strip_side_prefix(name))
            else:
                keys.append(idx)

        new_of_key: dict = {}
        remap = np.empty(n, dtype=np.int64)
        new_names: list[str] = []
        rep_old: list[int] = []  # representative old index per new point
        for old, key in enumerate(keys):
            if key not in new_of_key:
                new_of_key[key] = len(new_names)
                new_names.append(
                    _strip_side_prefix(self.joint_names[old])
                    if isinstance(key, str)
                    else self.joint_names[old]
                )
                rep_old.append(old)
            remap[old] = new_of_key[key]

        if len(new_names) == n:  # nothing merged
            return self, np.arange(n, dtype=np.int64)

        rep_old_arr = np.asarray(rep_old, dtype=np.int64)
        # Each new point inherits its representative's limb; then compact limb
        # ids so limbs that lost all their points (e.g. the left stripe limb) are
        # dropped, and rename the surviving stripe limb to strip its L/R prefix.
        rep_limb = self.limb_id[rep_old_arr]
        used = np.unique(rep_limb)
        limb_remap = {int(old): new for new, old in enumerate(used)}
        new_limb_id = np.array(
            [limb_remap[int(lid)] for lid in rep_limb], dtype=np.int64
        )
        new_limb_names = [
            _strip_side_prefix(name) if "stripe" in name.lower() else name
            for name in (self.limb_names[u] for u in used)
        ]

        new_bones = _unique_edges(remap[self.bones])
        new_bones3d = _unique_edges(remap[self.bones3d])
        new_visibility = {
            cam: np.unique(remap[idx]) for cam, idx in self.visibility.items()
        }
        merged = Skeleton(
            name=self.name,
            joint_names=tuple(new_names),
            limb_names=tuple(new_limb_names),
            limb_id=new_limb_id,
            bones=new_bones,
            bones3d=new_bones3d,
            left_idx=remap[self.left_idx] if self.left_idx.size else self.left_idx,
            right_idx=remap[self.right_idx] if self.right_idx.size else self.right_idx,
            visibility=new_visibility,
        )
        return merged, remap


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


def _strip_side_prefix(name: str) -> str:
    """Drop a leading ``l_`` / ``r_`` body-side prefix from a name (case-insensitive)."""
    return name[2:] if name[:2].lower() in ("l_", "r_") else name


def _unique_edges(edges: Int[np.ndarray, "E 2"]) -> Int[np.ndarray, "E2 2"]:
    """Deduplicate undirected edges (kept in first-seen order)."""
    if edges.size == 0:
        return edges.reshape(-1, 2)
    seen: dict[tuple[int, int], None] = {}
    for i, j in edges:
        seen.setdefault((int(min(i, j)), int(max(i, j))), None)
    return np.asarray(list(seen), dtype=np.int64).reshape(-1, 2)
