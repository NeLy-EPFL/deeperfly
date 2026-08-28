"""Tracked-point skeleton for multi-view pose (Drosophila by default).

A :class:`Skeleton` is the rig-independent description of *what* is tracked, and it is
**four things**: the ordered ``points`` (the detector's channel contract), the ``edges``
between them (the whole topology), the left/right ``point_symmetries``, and the two
color tables. Nothing else -- no groups, no chains, no names for subsets of points. It
carries no geometry; it is consumed by triangulation (to mask unobservable points),
bundle adjustment (bone-length priors), pictorial-structures recovery and visualization
(drawing the edges). Which view sees which point is not here either: an unobserved
``(view, point)`` is simply ``NaN`` in the points array.

It lives in a version-controlled file of its own (``data/skeletons/fly38.toml``), not in
a run config: which points exist and in which order is a property of the *checkpoint*,
whose artifact carries the point names and whose loader refuses a config that disagrees
(:func:`deeperfly.pose2d.stream._check_channel_names`). A run config normally says
nothing at all about the skeleton; ``[skeleton] include = "fly38"`` is the override.

**A name is a label, not an identity.** ``fly38`` meant the DeepFly3D point set before
1.0 and means the midline-abdomen one now; ``fly38b`` was renamed to ``fly38`` without a
coordinate moving. No alias table can disentangle that, so nothing in this package
compares skeletons by name -- every check is on the ordered ``point_names``
(:func:`deeperfly.pose2d.stream._check_channel_names`,
:func:`deeperfly.pipeline.run._refuse_a_foreign_skeleton`,
:func:`deeperfly.gui.labels.labels_identity`). :attr:`Skeleton.digest` is the printable
form of that identity and :attr:`Skeleton.label` pairs it with the name, so
"same name, different points" reads as one line instead of a 38-item diff. It is only
ever *printed*; the comparisons stay on the names themselves.

**Chains were never primitive.** They existed as ``limb_points``, a compaction of the
edge list that also served as the color-grouping table -- so rearranging the colors
silently changed which channel was which, and a chain could only ever express a *path*
(it could not attach an antenna to a head). :func:`deeperfly.pictorial.skeleton_chains`
already derived the chains it needs from the edge graph, which is what made them safe to
delete. What a chain name did for the rest of the schema is now a **point selector** (see
:func:`resolve_points`): an entry in any point set is a name or a ``*`` pattern.

**Symmetry** (``point_symmetries``) is the same relation SLEAP models as a ``type 2``
skeleton edge: an unordered pair of points that mirror each other across the animal's
sagittal plane. Which side comes first carries no meaning, so a pair is stored sorted.
Two things read it, and they are worth naming because each fails differently without it:

:meth:`Skeleton.flip_perm`
    The full-length channel permutation a horizontal mirror implies -- the
    equivalent of SLEAP's ``Skeleton.get_flipped_node_inds()``. Mirroring an image
    without applying it trains every left channel on a right joint, which costs no
    error, emits no warning, and looks exactly like a model that will not converge.
    Read by flip augmentation, which lives outside this package.
:meth:`Skeleton.partner`
    Which point is the other half of a pair -- what lets the ``symmetrize``
    correction name one side and get both.

16 rows is 16 chances to swap a side silently, so the loader checks the permutation
against the edges: applying it to ``edges`` must give ``edges`` back
(:func:`_check_automorphism`). A row carrying the wrong side, or two joints of one leg
exchanged, breaks that and is named. Declaring no pairs is legal and switches both
consumers off -- the right default for a genuinely asymmetric subject.

**Color** is two index-aligned tables, ``point_colors`` and ``edge_colors``, each one
``#rrggbb`` per item. Both are optional in the file and both are always populated on the
object: an unnamed point takes :data:`TAB10_HEX` by index, and an unnamed edge takes the
**average of its two endpoints' colors**. Averaging is what makes the edge table
genuinely optional -- an edge inside one colored group comes out that group's color
exactly, and only an edge that crosses two groups blends. (It replaces an older rule
where an edge took the color of the point it was written *from*; direction still matters
for :meth:`Skeleton.edge_endpoints`, but no longer for color.)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Sequence

import numpy as np
from jaxtyping import Int

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger("deeperfly")

__all__ = ["Skeleton", "resolve_points"]

#: The characters that make a selector entry a PATTERN rather than a point name. An entry
#: holding none of them is matched by equality, which is what lets an exact name beat a
#: pattern that also covers it.
_GLOB_CHARS = "*?["

#: matplotlib's ``tab10`` as ``#rrggbb`` -- the color a point takes when the skeleton
#: declares none for it. DeepLabCut's default is the same idea (``colormap`` over the
#: bodypart index); the packaged palette exists only because a deliberate one reads
#: better than a rainbow.
TAB10_HEX = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@dataclass(frozen=True)
class Skeleton:
    """An ordered set of points, their edges, their mirror pairs and their colors.

    Attributes
    ----------
    name
        A human label for the skeleton (e.g. ``"fly38"``). **Not** its identity -- see
        :attr:`digest`, and the module docstring for why nothing compares names.
    point_names
        Human-readable name per tracked point, in order (length ``n_points``). This is
        the channel contract: channel ``i`` of the detector is ``point_names[i]``.
    edges
        The edges as point-index pairs (shape ``(n_edges, 2)``), in declaration order.
        Direction is preserved -- it is what :meth:`edge_endpoints` reports and what the
        editor seeds a missing joint along -- but it no longer decides the edge's color.
    point_colors
        ``#rrggbb`` per point, in point order (length ``n_points``). Always populated: a
        point the skeleton names no color for takes :data:`TAB10_HEX` by index.
    edge_colors
        ``#rrggbb`` per edge, in edge order (length ``n_edges``). Always populated: an
        edge the skeleton names no color for takes the average of its two endpoints'.
    point_symmetries
        Left/right mirror pairs as point-index pairs (shape ``(n_symmetries, 2)``),
        each row sorted ascending and the rows sorted -- so two skeletons that
        declare the same pairs in different orders compare equal. Empty when the
        skeleton declares none. Each point appears in at most one pair.
    """

    name: str
    point_names: tuple[str, ...]
    edges: Int[np.ndarray, "E 2"]
    #: Defaulted so a skeleton built by hand -- in a test, or read from a ``results.h5``
    #: written before colors were stored -- still constructs; it then takes the colormap.
    point_colors: tuple[str, ...] = ()
    #: Defaulted for the same reason, and additionally because it is the rarer table: a
    #: skeleton that colors its points has already colored its edges.
    edge_colors: tuple[str, ...] = ()
    #: Last and defaulted, so a skeleton read from a ``results.h5`` written before
    #: symmetry existed still constructs -- it simply declares no pairs.
    point_symmetries: Int[np.ndarray, "S 2"] = field(
        default_factory=lambda: np.empty((0, 2), np.int64)
    )

    def __post_init__(self) -> None:
        """Canonicalize the symmetry pairs and fill in any unnamed color.

        Here rather than only in the parsers because a skeleton also arrives from
        ``results.h5`` and from the editor, and :meth:`flip_perm`'s promise to be an
        involution is only true while no point sits in two pairs. Canonicalizing (rows
        sorted, then sorted) here is what lets :func:`diff_skeletons` compare two
        skeletons' pairs without caring how either was spelled.

        The **automorphism** check is deliberately NOT here: it belongs to the loader of
        a hand-written file (:meth:`from_spec`), and a skeleton read back from an old
        result must stay readable even if its stored pairs and edges disagree.
        """
        pairs = np.asarray(self.point_symmetries, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            if pairs.min() < 0 or pairs.max() >= len(self.point_names):
                raise ValueError(
                    f"point_symmetries reference a point index outside "
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
        object.__setattr__(self, "point_symmetries", pairs)

        colors = tuple(str(c) for c in self.point_colors)
        if len(colors) != len(self.point_names):
            if colors:
                raise ValueError(
                    f"point_colors has {len(colors)} entries for "
                    f"{len(self.point_names)} points"
                )
            colors = tuple(
                TAB10_HEX[i % len(TAB10_HEX)] for i in range(len(self.point_names))
            )
        object.__setattr__(self, "point_colors", colors)

        edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        object.__setattr__(self, "edges", edges)
        edge_colors = tuple(str(c) for c in self.edge_colors)
        if len(edge_colors) != len(edges):
            if edge_colors:
                raise ValueError(
                    f"edge_colors has {len(edge_colors)} entries for {len(edges)} edges"
                )
            edge_colors = tuple(
                _average_hex(colors[int(a)], colors[int(b)]) for a, b in edges
            )
        object.__setattr__(self, "edge_colors", edge_colors)

    # -- construction --------------------------------------------------------

    @classmethod
    def fly(cls) -> Skeleton:
        """The packaged 38-point Drosophila skeleton."""
        from .config import default_skeleton_spec

        return cls.from_spec(default_skeleton_spec())

    @classmethod
    def from_config(cls, config: "Config") -> Skeleton:
        """Build a skeleton from a config.

        The ``[skeleton]`` table has already been resolved by
        :func:`deeperfly.config._resolve_skeleton` -- ``include`` expanded, or the
        packaged skeleton filled in for a config that named none -- so this only ever
        reads a fully-spelled table.

        Raises
        ------
        ValueError
            If the table is malformed (see :meth:`from_spec`).
        """
        return cls.from_spec(config.data["skeleton"])

    @classmethod
    def from_spec(cls, spec: dict) -> Skeleton:
        """Build a skeleton from a ``[skeleton]`` table: points, edges, symmetries, colors.

        Parameters
        ----------
        spec
            The table. ``points`` is required and ordered; ``edges`` and
            ``point_symmetries`` are lists of point pairs (each a name or an index);
            ``point_colors`` maps a point name or a ``*`` pattern to a hex color and
            ``edge_colors`` maps an ``"<a>--<b>"`` endpoint pattern to one. Both color
            tables are optional.

        Raises
        ------
        ValueError
            If ``points`` is missing or holds a duplicate, if an edge or symmetry names
            an unknown point, if the symmetries are not an automorphism of the edges, or
            if a color table holds a key matching nothing or two patterns matching one
            point/edge.
        """
        if "points" not in spec:
            raise ValueError(
                "[skeleton] declares no 'points'. A skeleton is points, edges, "
                "symmetries and colors; the points are the ordered channel contract."
            )
        # An EMPTY list is legal and distinct from a missing key: a fresh project's
        # skeleton starts empty and the operator fills it in.
        point_names = tuple(str(p) for p in spec["points"] or ())
        dupes = sorted({n for n in point_names if point_names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"[skeleton] points repeats {dupes}; a point name is a channel and has "
                "to be unique"
            )
        edges = _pairs(spec.get("edges"), point_names, "[skeleton] edges")
        symmetries = _pairs(
            spec.get("point_symmetries"), point_names, "[skeleton] point_symmetries"
        )
        _check_automorphism(edges, symmetries, point_names)
        point_colors = _resolve_point_colors(
            spec.get("point_colors") or {}, point_names
        )
        skeleton = cls(
            name=str(spec.get("name", "skeleton")),
            point_names=point_names,
            edges=edges,
            point_colors=point_colors,
            edge_colors=_resolve_edge_colors(
                spec.get("edge_colors") or {}, point_names, edges, point_colors
            ),
            point_symmetries=symmetries,
        )
        log.info(
            "skeleton %s: %d points, %d edges, %d symmetry pairs",
            skeleton.label,
            skeleton.n_points,
            skeleton.n_edges,
            skeleton.n_symmetries,
        )
        return skeleton

    # -- basic views ---------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.point_names)

    @property
    def n_edges(self) -> int:
        return int(np.asarray(self.edges).reshape(-1, 2).shape[0])

    @property
    def n_symmetries(self) -> int:
        return int(np.asarray(self.point_symmetries).reshape(-1, 2).shape[0])

    def __len__(self) -> int:
        return self.n_points

    # -- identity ------------------------------------------------------------

    @property
    def digest(self) -> str:
        """8 hex chars over *what* is tracked and how it is connected.

        The printable form of the identity every check in this package actually
        compares: the ordered ``point_names``, the ``edges``, and the
        ``point_symmetries``. Two skeletons with the same digest mean the same thing by
        every index; two with the same :attr:`name` need not.

        **Colors are deliberately out of it**, for the same reason the ``name`` is out
        of :func:`deeperfly.pipeline.fingerprint._skeleton_digest`: recoloring one
        changes no stage's answer and moves no label, so a digest that changed with the
        palette would report a different skeleton where there is none. The visualization
        stage fingerprints the colors separately, which is where that belongs.

        Never used as a *comparison* -- only printed, so a mismatch reads as one line
        instead of a 38-item diff. What is compared stays the point names themselves.
        """
        payload = json.dumps(
            {
                "points": list(self.point_names),
                "edges": np.asarray(self.edges).reshape(-1, 2).tolist(),
                "point_symmetries": np.asarray(self.point_symmetries)
                .reshape(-1, 2)
                .tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]

    @property
    def label(self) -> str:
        """``"<name>@<digest>"`` -- what to print when naming a skeleton to a human."""
        return f"{self.name}@{self.digest}"

    # -- symmetry ------------------------------------------------------------

    @property
    def symmetry_names(self) -> tuple[tuple[str, str], ...]:
        """The symmetry pairs as ``(name, name)`` tuples, in stored order."""
        pairs = np.asarray(self.point_symmetries).reshape(-1, 2)
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
        for a, b in np.asarray(self.point_symmetries).reshape(-1, 2):
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
        pairs = np.asarray(self.point_symmetries, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            perm[pairs[:, 0]] = pairs[:, 1]
            perm[pairs[:, 1]] = pairs[:, 0]
        return perm

    # -- derived structure ---------------------------------------------------

    def edge_endpoints(
        self,
    ) -> tuple[Int[np.ndarray, "E"], Int[np.ndarray, "E"]]:
        """Endpoint index arrays ``(i, j)`` for vectorized bone-length maths.

        Returns
        -------
        i, j : np.ndarray
            The first and second endpoint index of each edge (shape ``(E,)``).
        """
        return self.edges[:, 0], self.edges[:, 1]

    def index(self, name: str) -> int:
        """``name``'s point index, or ``ValueError`` naming the skeleton."""
        if name not in self.point_names:
            raise ValueError(f"{name!r} is not a point of skeleton {self.name!r}")
        return self.point_names.index(name)


# -- the point selector -------------------------------------------------------


def resolve_points(
    entries: Sequence[str] | None,
    point_names: Sequence[str],
    *,
    where: str,
) -> tuple[int, ...]:
    """Resolve a point selector to point indices, in point order.

    One grammar, four users: ``[bundle_adjustment] points``, the ``static`` and
    ``symmetrize`` ops' ``points`` / ``midline``, and ``[skeleton] colors``. An entry is
    a **point name** or a ``*`` **pattern** over the names (``fnmatch``, case-sensitive).
    This is what a chain name used to do -- a group existed only so another section could
    name it -- with none of the cost, since a pattern needs nothing declared.

    Three refusals, and each is a typo that would otherwise pass:

    * an entry with no glob character that is not a point -- a misspelled name;
    * a pattern matching **nothing** -- always a typo, because a selector naming no
      point is never what anyone means;
    * two **patterns** matching one point -- ambiguous for a color table, so it is
      refused everywhere rather than left to mean different things in different tables.
      An exact name is not a pattern and beats one, so ``l_antenna`` alongside ``"l*"``
      is fine.

    The resolved set is logged at INFO, because over-matching is the one failure a
    selector cannot detect for itself.

    Parameters
    ----------
    entries
        The selector. ``None`` or empty resolves to ``()``.
    point_names
        The skeleton's ordered point names.
    where
        The config location, for the error and log messages (e.g.
        ``"[bundle_adjustment] points"``).

    Returns
    -------
    tuple of int
        The matched point indices, ascending and deduplicated.
    """
    if not entries:
        return ()
    if isinstance(entries, str):
        raise ValueError(
            f"{where} must be a list of point names or patterns, not a string"
        )
    names = tuple(str(n) for n in point_names)
    index = {n: i for i, n in enumerate(names)}
    hit: dict[int, str] = {}  # point -> the PATTERN that claimed it
    chosen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(f"{where} entry {entry!r} is not a point name or pattern")
        if not any(c in entry for c in _GLOB_CHARS):
            if entry not in index:
                raise ValueError(
                    f"{where} names {entry!r}, which is not a point of this skeleton"
                )
            chosen.add(index[entry])
            continue
        matched = [i for i, n in enumerate(names) if fnmatchcase(n, entry)]
        if not matched:
            raise ValueError(
                f"{where} pattern {entry!r} matches no point of this skeleton "
                f"(have {len(names)}: {names[0]!r} ... {names[-1]!r})"
            )
        for i in matched:
            if i in hit:
                raise ValueError(
                    f"{where} patterns {hit[i]!r} and {entry!r} both match "
                    f"{names[i]!r}; a point may be claimed by only one pattern"
                )
            hit[i] = entry
        chosen.update(matched)
    out = tuple(sorted(chosen))
    log.info("%s -> %d points: %s", where, len(out), [names[i] for i in out])
    return out


# -- parsing ------------------------------------------------------------------


def _pairs(raw, point_names: tuple[str, ...], where: str) -> Int[np.ndarray, "E 2"]:
    """A list of 2-element point pairs -> an ``(E, 2)`` index array.

    Used for both ``edges`` and ``symmetries``, which are the same shape: each element is
    a point name or an integer index. ``None`` or ``[]`` gives an empty array.
    """
    if not raw:
        return np.empty((0, 2), np.int64)
    if not isinstance(raw, list):
        raise ValueError(f"{where} must be a list of 2-element pairs, got {raw!r}")
    index = {name: i for i, name in enumerate(point_names)}
    n = len(point_names)
    rows: list[tuple[int, int]] = []
    for k, entry in enumerate(raw):
        at = f"{where}[{k}]"
        if isinstance(entry, str) or not isinstance(entry, (list, tuple)):
            raise ValueError(f"{at} must be a 2-element pair, got {entry!r}")
        if len(entry) != 2:
            raise ValueError(
                f"{at} must name exactly 2 points, got {len(entry)}: {list(entry)!r}"
            )
        pair = []
        for p in entry:
            if isinstance(p, str):
                if p not in index:
                    raise ValueError(f"{at} references unknown point name {p!r}")
                pair.append(index[p])
            else:
                i = int(p)
                if not 0 <= i < n:
                    raise ValueError(f"{at} point index {i} outside [0, {n})")
                pair.append(i)
        if pair[0] == pair[1]:
            raise ValueError(f"{at} pairs {point_names[pair[0]]!r} with itself")
        rows.append((pair[0], pair[1]))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def _check_automorphism(
    edges: Int[np.ndarray, "E 2"],
    symmetries: Int[np.ndarray, "S 2"],
    point_names: tuple[str, ...],
) -> None:
    """Refuse symmetry pairs that are not an automorphism of the edge set.

    The mirror permutation is applied to every edge; the result must be the edge set
    again (as unordered pairs). This is what makes 16 hand-written rows safe without any
    grouping concept: a row carrying the wrong side, or two joints of one leg exchanged,
    maps some edge onto a pair that is not an edge -- and it is strictly stronger than the
    chain-consistency check it replaces, which only ever compared chain *lengths*.

    Points with no partner map to themselves, so a midline point needs no row and a
    skeleton declaring no symmetries at all passes trivially.
    """
    pairs = np.asarray(symmetries, dtype=np.int64).reshape(-1, 2)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not pairs.size or not edges.size:
        return
    perm = np.arange(len(point_names), dtype=np.int64)
    perm[pairs[:, 0]] = pairs[:, 1]
    perm[pairs[:, 1]] = pairs[:, 0]
    have = {frozenset((int(a), int(b))) for a, b in edges}
    for a, b in edges:
        want = frozenset((int(perm[a]), int(perm[b])))
        if want not in have:
            i, j = sorted(want)
            raise ValueError(
                f"[skeleton] point_symmetries are not a mirror of [skeleton] edges: "
                f"the edge {point_names[int(a)]!r} -- {point_names[int(b)]!r} maps to "
                f"{point_names[i]!r} -- {point_names[j]!r}, which is not an edge. "
                "Applying the mirror to every edge has to give the edge set back, so "
                "either a pair carries the wrong side or an edge is missing."
            )


def _resolve_point_colors(raw: dict, point_names: tuple[str, ...]) -> tuple[str, ...]:
    """A ``[skeleton.point_colors]`` table -> one hex color per point.

    Keys are point names or ``*`` patterns, resolved by :func:`resolve_points`'s rules --
    an exact name beats a pattern, two patterns over one point is an error, a pattern
    matching nothing is an error. A point no key covers takes :data:`TAB10_HEX` by index,
    so the table is optional and may be partial.
    """
    where = "[skeleton] point_colors"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a table, got {raw!r}")
    out: list[str | None] = [None] * len(point_names)
    patterns = {k: v for k, v in raw.items() if any(c in k for c in _GLOB_CHARS)}
    exact = {k: v for k, v in raw.items() if k not in patterns}
    for pattern, color in patterns.items():
        for i in resolve_points([pattern], point_names, where=where):
            if out[i] is not None:
                raise ValueError(
                    f"{where}: two patterns both match "
                    f"{point_names[i]!r}; a point takes exactly one color"
                )
            out[i] = _hex(color, pattern)
    for name, color in exact.items():
        if name not in point_names:
            raise ValueError(
                f"{where} names {name!r}, which is not a point of this skeleton"
            )
        out[point_names.index(name)] = _hex(color, name)
    return tuple(
        c if c is not None else TAB10_HEX[i % len(TAB10_HEX)] for i, c in enumerate(out)
    )


#: What separates the two endpoint selectors of an ``[skeleton.edge_colors]`` key. Two
#: characters rather than one because a point name may legitimately hold a single ``-``,
#: and a doubled one cannot be mistaken for part of a name.
_EDGE_SEP = "--"


def _resolve_edge_colors(
    raw: dict,
    point_names: tuple[str, ...],
    edges: Int[np.ndarray, "E 2"],
    point_colors: tuple[str, ...],
) -> tuple[str, ...]:
    """A ``[skeleton.edge_colors]`` table -> one hex color per edge.

    A key is ``"<a>--<b>"``, each side a point name or a ``*`` pattern over the names,
    and it claims every edge whose endpoints the two sides cover **in either
    orientation** -- so ``"abdomen1--abdomen0"`` is not a silent miss on an edge stored
    the other way round. That is the whole grammar: an edge has no name of its own to
    key on, and keying on the *index* would let a reorder silently recolor the drawing.

    An edge no key covers takes the average of its two endpoints' colors, which is what
    makes the table optional -- an edge inside one colored group comes out that group's
    color exactly.

    Three refusals, matching :func:`_resolve_point_colors`: a key that is not one
    ``--``-separated pair, a key matching **no** edge (always a typo), and two keys
    claiming one edge (ambiguous -- an edge takes exactly one color).
    """
    where = "[skeleton] edge_colors"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a table, got {raw!r}")
    rows = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out: list[str | None] = [None] * len(rows)
    for key, color in raw.items():
        sides = str(key).split(_EDGE_SEP)
        if len(sides) != 2 or not sides[0] or not sides[1]:
            raise ValueError(
                f"{where} key {key!r} is not an edge selector; write "
                f"'<a>{_EDGE_SEP}<b>', each side a point name or a '*' pattern "
                f'(e.g. "abdomen*{_EDGE_SEP}abdomen*")'
            )
        left = set(resolve_points([sides[0]], point_names, where=f"{where} {key!r}"))
        right = set(resolve_points([sides[1]], point_names, where=f"{where} {key!r}"))
        matched = [
            k
            for k, (a, b) in enumerate(rows)
            if (int(a) in left and int(b) in right)
            or (int(b) in left and int(a) in right)
        ]
        if not matched:
            raise ValueError(
                f"{where} key {key!r} matches no edge of this skeleton "
                f"(it has {len(rows)})"
            )
        value = _hex(color, key)
        for k in matched:
            if out[k] is not None:
                a, b = rows[k]
                raise ValueError(
                    f"{where}: two keys both match the edge "
                    f"{point_names[int(a)]!r} -- {point_names[int(b)]!r}; "
                    "an edge takes exactly one color"
                )
            out[k] = value
    return tuple(
        c if c is not None else _average_hex(point_colors[int(a)], point_colors[int(b)])
        for c, (a, b) in zip(out, rows)
    )


def _average_hex(a: str, b: str) -> str:
    """The per-channel mean of two ``#rgb`` / ``#rrggbb`` colors, as ``#rrggbb``.

    Averaged in 8-bit sRGB rather than linear light: this is a label color, and "halfway
    between these two swatches" is what a reader expects to see. Two equal inputs come
    back unchanged, which is what makes the derivation a no-op for every edge inside one
    colored group.
    """
    ar, ag, ab = _rgb(a)
    br, bg, bb = _rgb(b)
    return "#%02x%02x%02x" % (
        (ar + br + 1) // 2,
        (ag + bg + 1) // 2,
        (ab + bb + 1) // 2,
    )


def _rgb(value: str) -> tuple[int, int, int]:
    """``#rgb`` / ``#rrggbb`` -> three 0-255 channels. Assumes :func:`_hex` passed."""
    body = value[1:] if value.startswith("#") else value
    if len(body) == 3:
        body = "".join(c * 2 for c in body)
    return int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16)


def _hex(value, key: str) -> str:
    """Validate a ``#rgb`` / ``#rrggbb`` color, so a typo fails at load, not at draw."""
    text = str(value)
    body = text[1:] if text.startswith("#") else text
    bad = ValueError(
        f"[skeleton] color {key!r} = {value!r} is not a #rgb or #rrggbb color"
    )
    if not text.startswith("#") or len(body) not in (3, 6):
        raise bad
    try:
        int(body, 16)
    except ValueError:
        raise bad from None
    return text
