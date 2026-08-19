"""Tracked-point skeleton for multi-view pose (Drosophila by default).

A :class:`Skeleton` is the rig-independent description of *what* is tracked: the
ordered tracked points, their grouping into limbs, the bones (edges) connecting
them, their left/right symmetry pairs, and -- for a known camera rig -- which
points each named camera can see. It carries no geometry; it is consumed by
triangulation (to mask unobservable points), bundle adjustment (bone-length
priors), pictorial-structures recovery and visualization (drawing bones).

The default fly skeleton is the ``[skeleton]`` section of the packaged
``data/default_config.toml``; it tracks the same 38 points as NeLy-EPFL/DeepFly3D's
``skeleton_fly.py`` but orders the body sides left-first (left ``0..18``, right
``19..37``), with 10 limbs, 28 within-leg/abdomen bones and 19 symmetry pairs.
Load it with :meth:`Skeleton.fly`.

**Symmetry** (``[skeleton].symmetries``) is the same relation SLEAP models as a
``type 2`` skeleton edge: an unordered pair of points that mirror each other
across the animal's sagittal plane. Which side comes first carries no meaning, so
a pair is stored sorted. Three things read it, and they are worth naming because
each fails differently without it:

:meth:`Skeleton.flip_perm`
    The full-length channel permutation a horizontal mirror implies -- the
    equivalent of SLEAP's ``Skeleton.get_flipped_node_inds()``. Mirroring an image
    without applying it trains every left channel on a right joint, which costs no
    error, emits no warning, and looks exactly like a model that will not converge.
    Read by flip augmentation, which lives outside this package.
:mod:`deeperfly.pose2d.pathways`
    Validates that a mirrored detection pathway lands on the *mirrored* points.
    Without the pairs a one-word typo in ``[pose2d.output_points]`` silently swaps
    a side and still reconstructs a plausible-looking skeleton.
:mod:`deeperfly.chirality`
    Flags hand labels whose left/right identities look swapped.

Declaring no pairs is legal and switches all three off (``flip_perm`` becomes the
identity, the pathway check and the chirality QC skip). That is the right default
for a genuinely asymmetric subject; it is the *wrong* one for a fly, which is why
the packaged skeleton declares them and :func:`infer_symmetries_by_name` exists to
propose them for a skeleton that does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Int

if TYPE_CHECKING:
    from .config import Config

__all__ = ["Skeleton", "infer_symmetries_by_name"]


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
    symmetries
        Left/right mirror pairs as point-index pairs (shape ``(n_symmetries, 2)``),
        each row sorted ascending and the rows sorted -- so two skeletons that
        declare the same pairs in different orders compare equal. Empty when the
        config declares none. Each point appears in at most one pair.
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
    #: Last and defaulted, so a skeleton read from a ``results.h5`` written before
    #: symmetry existed still constructs -- it simply declares no pairs.
    symmetries: Int[np.ndarray, "S 2"] = field(
        default_factory=lambda: np.empty((0, 2), np.int64)
    )

    def __post_init__(self) -> None:
        """Canonicalize and validate the symmetry pairs, whatever built the skeleton.

        Here rather than only in :func:`_parse_symmetries` because a skeleton also arrives
        from ``results.h5`` and from the editor, and :meth:`flip_perm`'s promise to be an
        involution is only true while no point sits in two pairs. Canonicalizing (rows
        sorted, then sorted) here is what lets :func:`diff_skeletons` compare two
        skeletons' pairs without caring how either was spelled.
        """
        pairs = np.asarray(self.symmetries, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            if pairs.min() < 0 or pairs.max() >= len(self.point_names):
                raise ValueError(
                    f"symmetries reference a point index outside "
                    f"[0, {len(self.point_names)})"
                )
            if (pairs[:, 0] == pairs[:, 1]).any():
                raise ValueError("a symmetry pairs a point with itself")
            flat = pairs.reshape(-1)
            if len(np.unique(flat)) != len(flat):
                dupes = sorted(
                    {self.point_names[i] for i in flat if (flat == i).sum() > 1}
                )
                raise ValueError(
                    f"these points sit in more than one symmetry pair: {dupes}; "
                    "each point mirrors exactly one other"
                )
            pairs = np.sort(pairs, axis=1)
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        object.__setattr__(self, "symmetries", pairs)

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
            out-of-range point index, or if ``symmetries`` is malformed (see
            :func:`_parse_symmetries`).
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
            symmetries=_parse_symmetries(spec.get("symmetries"), point_names),
        )

    # -- basic views ---------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.point_names)

    @property
    def n_limbs(self) -> int:
        return len(self.limb_names)

    @property
    def n_symmetries(self) -> int:
        return int(np.asarray(self.symmetries).reshape(-1, 2).shape[0])

    def __len__(self) -> int:
        return self.n_points

    # -- symmetry ------------------------------------------------------------

    @property
    def symmetry_names(self) -> tuple[tuple[str, str], ...]:
        """The symmetry pairs as ``(name, name)`` tuples, in stored order."""
        pairs = np.asarray(self.symmetries).reshape(-1, 2)
        return tuple(
            (self.point_names[int(i)], self.point_names[int(j)]) for i, j in pairs
        )

    def partner(self, point) -> int | None:
        """The index mirroring ``point`` (name or index), or ``None`` if unpaired.

        Raises ``ValueError`` for a name the skeleton does not have -- an unknown point
        is a caller bug, distinct from a known point that simply has no partner.
        """
        if isinstance(point, str):
            if point not in self.point_names:
                raise ValueError(f"{point!r} is not a point of skeleton {self.name!r}")
            i = self.point_names.index(point)
        else:
            i = int(point)
        for a, b in np.asarray(self.symmetries).reshape(-1, 2):
            if int(a) == i:
                return int(b)
            if int(b) == i:
                return int(a)
        return None

    def flip_perm(self) -> Int[np.ndarray, "P"]:
        """The point permutation a left-right mirror implies (shape ``(n_points,)``).

        ``mirrored[i] = original[flip_perm()[i]]``: the value that belongs in slot ``i``
        of a mirrored sample is the one that sat in the partner slot before the mirror.
        Identity at every unpaired point, so a skeleton with no declared symmetries
        returns ``arange(n_points)`` -- and a mirror then permutes nothing, which is
        exactly right for points that carry no side.

        Because each point is in at most one pair, the permutation is an **involution**:
        applying it twice is the identity. Flip augmentation relies on
        that, and so does the round-trip test.

        This is the same lookup table SLEAP exposes as
        ``Skeleton.get_flipped_node_inds()``.
        """
        perm = np.arange(self.n_points, dtype=np.int64)
        pairs = np.asarray(self.symmetries, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            perm[pairs[:, 0]] = pairs[:, 1]
            perm[pairs[:, 1]] = pairs[:, 0]
        return perm

    def symmetries_or_inferred(self) -> Int[np.ndarray, "S 2"]:
        """The declared pairs, or -- if none are declared -- pairs inferred by name.

        The single fallback for consumers that would rather work approximately than not
        at all: the chirality QC wants pairs even for a skeleton loaded from a
        ``results.h5`` written before symmetry existed. Anything that *writes* (config,
        migrations) must use :attr:`symmetries` verbatim instead, so an inference never
        becomes a stored fact behind the operator's back.

        SLEAP does the same thing for skeletons imported without symmetries.
        """
        if self.n_symmetries:
            return np.asarray(self.symmetries, dtype=np.int64).reshape(-1, 2)
        return infer_symmetries_by_name(self.point_names)

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


def _parse_symmetries(raw, point_names: tuple[str, ...]) -> Int[np.ndarray, "S 2"]:
    """A ``[skeleton].symmetries`` list of 2-element pairs -> a sorted ``(S, 2)`` array.

    ``raw`` is ``[[a, b], ...]``, each element a point name or an integer index; ``None``
    or ``[]`` means "no symmetry declared". Pairs are canonicalized (each row sorted, then
    the rows sorted) so declaration order carries no meaning -- matching SLEAP, where a
    symmetry is an unordered node *set*.

    Raises
    ------
    ValueError
        If an entry is not a 2-element pair, names an unknown point, indexes outside
        ``[0, n_points)``, pairs a point with itself, or puts one point in two pairs. The
        last is the one worth being strict about: a point with two partners has no
        well-defined mirror, so :meth:`Skeleton.flip_perm` would silently stop being an
        involution and a mirror-then-unmirror round trip would not return the input.
    """
    if not raw:
        return np.empty((0, 2), np.int64)
    if not isinstance(raw, list):
        raise ValueError(
            f"[skeleton].symmetries must be a list of 2-element pairs, got {raw!r}"
        )
    index = {name: i for i, name in enumerate(point_names)}
    n = len(point_names)
    seen: dict[int, int] = {}  # point -> the row that claimed it
    rows: list[tuple[int, int]] = []
    for k, entry in enumerate(raw):
        where = f"[skeleton].symmetries[{k}]"
        if isinstance(entry, str) or not isinstance(entry, (list, tuple)):
            raise ValueError(f"{where} must be a 2-element pair, got {entry!r}")
        if len(entry) != 2:
            raise ValueError(
                f"{where} must name exactly 2 points, got {len(entry)}: {list(entry)!r}"
            )
        pair = []
        for p in entry:
            if isinstance(p, str):
                if p not in index:
                    raise ValueError(f"{where} references unknown point name {p!r}")
                pair.append(index[p])
            else:
                i = int(p)
                if not 0 <= i < n:
                    raise ValueError(f"{where} point index {i} outside [0, {n})")
                pair.append(i)
        a, b = sorted(pair)
        if a == b:
            raise ValueError(
                f"{where} pairs {point_names[a]!r} with itself; a symmetry needs two points"
            )
        for i in (a, b):
            if i in seen:
                raise ValueError(
                    f"{where} puts {point_names[i]!r} in a second symmetry pair "
                    f"(already paired in [skeleton].symmetries[{seen[i]}]); "
                    "each point mirrors exactly one other"
                )
            seen[i] = k
        rows.append((a, b))
    return np.asarray(sorted(rows), dtype=np.int64).reshape(-1, 2)


#: Left/right name tokens, most specific first. Each entry is
#: ``(strip, side)`` where ``strip`` turns a name into its side-free stem or ``None``.
#: Order matters: the single-letter prefix rule is last because it is the loosest, and it
#: would otherwise read ``left_wing`` as side ``l`` with stem ``eft_wing``.
_LR_RULES: tuple[tuple[str, str, str], ...] = (
    ("prefix", "left", "right"),  # left_wing / right_wing, leftWing / rightWing
    ("suffix", "left", "right"),  # wing_left / wing_right
    ("suffix", "l", "r"),  # Ear_L / Ear_R
    ("prefix", "l", "r"),  # lf_claw / rf_claw, l_antenna / r_antenna
)
_LR_SEPARATORS = ("_", "-", "")


def _split_lr(name: str) -> list[tuple[tuple[int, str], bool]]:
    """Every ``((rule, stem), is_left)`` reading of ``name`` under :data:`_LR_RULES`.

    Returns a list rather than one answer because a name can parse several ways and only
    the caller knows which stems have a counterpart -- ``lf_claw`` is unambiguous, but
    ``left_wing`` reads as both the word ``left`` (stem ``wing``) and the letter ``l``
    (stem ``eft_wing``). The rule index rides along in the key so a ``left_x``/``right_x``
    pair can never match an ``l_x``/``r_x`` pair that happens to share a stem.
    """
    out: list[tuple[tuple[int, str], bool]] = []
    low = name.lower()
    for rule, (kind, ltok, rtok) in enumerate(_LR_RULES):
        for tok, is_left in ((ltok, True), (rtok, False)):
            for sep in _LR_SEPARATORS:
                affix = (tok + sep) if kind == "prefix" else (sep + tok)
                if kind == "prefix" and low.startswith(affix):
                    stem = name[len(affix) :]
                elif kind == "suffix" and low.endswith(affix):
                    stem = name[: len(name) - len(affix)]
                else:
                    continue
                if stem:
                    out.append(((rule, stem.lower()), is_left))
    return out


def infer_symmetries_by_name(
    point_names,
) -> Int[np.ndarray, "S 2"]:
    """Propose left/right pairs from point-name tokens (shape ``(S, 2)``, rows sorted).

    Recognizes ``left``/``right`` and ``l``/``r`` as a prefix or a suffix, with ``_``,
    ``-`` or nothing between token and stem: ``lf_claw``/``rf_claw``,
    ``l_antenna``/``r_antenna``, ``Ear_L``/``Ear_R``, ``left_wing``/``right_wing``.

    A reading is only honored when the **counterpart stem actually exists**, which is what
    keeps the loosest rule (a bare leading ``l``/``r``) from inventing pairs -- a lone
    ``rostrum`` stays unpaired because there is no ``lostrum``. Each point lands in at most
    one pair, and a stem claimed by a more specific rule is not reconsidered by a looser
    one.

    This is a **suggestion**, for the skeleton editor and for
    :meth:`Skeleton.symmetries_or_inferred`. Nothing writes its output to a config
    silently: a config states its pairs, so that renaming a point cannot quietly
    re-pair the skeleton.

    SLEAP has the same helper for the same reason
    (``sleap.qc.features.chirality.infer_symmetry_pairs_by_name``).
    """
    names = tuple(point_names)
    # (rule, stem) -> {True: idx, False: idx}; first occurrence per side wins, for
    # determinism when a skeleton repeats a stem.
    groups: dict[tuple[int, str], dict[bool, int]] = {}
    for i, name in enumerate(names):
        for key, is_left in _split_lr(name):
            groups.setdefault(key, {}).setdefault(is_left, i)

    rows: list[tuple[int, int]] = []
    used: set[int] = set()
    # Sorted by (rule, stem), so the specific rules claim their points before the loose
    # single-letter ones get a look.
    for key in sorted(groups):
        bucket = groups[key]
        if True not in bucket or False not in bucket:
            continue
        left, right = bucket[True], bucket[False]
        if left == right or left in used or right in used:
            continue
        rows.append((min(left, right), max(left, right)))
        used |= {left, right}
    return np.asarray(sorted(rows), dtype=np.int64).reshape(-1, 2)


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
