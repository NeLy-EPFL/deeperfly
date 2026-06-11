"""Packed parameter state for bundle adjustment (backend-agnostic).

The state vector concatenates ``[rvecs, tvecs, intrs, dists, pts3d]`` flat.
Index arrays (``rvecs_idx``, etc.) recover the original shapes; a boolean
``fixed`` mask marks parameters held constant during optimization.

Unlike a naive packing, two index entries are allowed to point at the *same*
slot in ``values`` -- that is how parameters are *shared*. :func:`build_state`
exposes this via a small string grammar (``"f.tvec[2]"``, ``"*.intr"``) so a
caller can fix individual elements and tie parameters across cameras together
without touching index arithmetic.

The same packed state is consumed by the solver in
:mod:`deeperfly.bundle_adjustment.core`.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from ..geometry import intr_to_kmat, rvec_to_rmat, triangulate_dlt

# Parameter names accepted in fixed/shared references, mapped to their group.
# ``kmat`` is accepted as an alias of the packed intrinsics ``intr``.
_PARAM_ALIASES = {
    "rvec": "rvec",
    "tvec": "tvec",
    "intr": "intr",
    "kmat": "intr",
    "dist": "dist",
}


class BAState(NamedTuple):
    """Packed bundle-adjustment state ready for ``bundle_adjust``.

    ``values`` is the flat parameter vector; ``fixed`` is a boolean mask of
    parameters to hold constant. The ``*_idx`` arrays index into ``values`` to
    recover the original ``rvecs`` / ``tvecs`` / ``intrs`` / ``dists`` /
    ``pts3d`` arrays (and may alias to share parameters). ``pts2d`` is the
    observation tensor.
    """

    values: Float[Array, "n_params"]
    fixed: Bool[Array, "n_params"]
    rvecs_idx: Int[Array, "V 3"]
    tvecs_idx: Int[Array, "V 3"]
    intrs_idx: Int[Array, "V P"]
    dists_idx: Int[Array, "V K"]
    pts3d_idx: Int[Array, "N 3"]
    pts2d: Float[Array, "V N 2"]


def initialize_pts3d(
    pts2d: Float[Array, "V *pts 2"],
    rvecs: Float[Array, "V 3"],
    tvecs: Float[Array, "V 3"],
    intrs: Float[Array, "V P"] | Float[Array, "P"],
) -> Float[Array, "*pts 3"]:
    """Triangulate initial 3D points from 2D observations and camera poses.

    Parameters
    ----------
    pts2d
        Observed 2D points of shape ``(V, *pts, 2)``, NaN for missing.
    rvecs, tvecs
        Per-camera extrinsics of shape ``(V, 3)``.
    intrs
        Packed intrinsics of shape ``(V, P)`` or ``(P,)`` (shared).

    Returns
    -------
    Array
        Triangulated 3D points of shape ``(*pts, 3)``.
    """
    kmat = intr_to_kmat(intrs)
    rtmat = jnp.concatenate(
        (jnp.asarray(rvec_to_rmat(rvecs)), tvecs[..., None]), axis=-1
    )
    return triangulate_dlt(pts2d, kmat @ rtmat)


def build_state(
    rvecs: Float[Array, "V 3"],
    tvecs: Float[Array, "V 3"],
    intrs: Float[Array, "V P"],
    dists: Float[Array, "V K"],
    pts2d: Float[Array, "V N 2"],
    names: list[str] | None = None,
    *,
    fixed: list[str] = (),
    shared: list[list[str]] = (),
    pts3d: Float[Array, "N 3"] | None = None,
) -> BAState:
    """Build a :class:`BAState` from per-camera arrays and fix/share specs.

    Parameters
    ----------
    rvecs, tvecs, intrs, dists
        Per-camera parameters of shape ``(V, 3)``, ``(V, 3)``, ``(V, P)`` and
        ``(V, K)``. Intrinsics and distortion are *per camera* -- share them
        explicitly via ``shared``.
    pts2d
        Observed 2D points of shape ``(V, N, 2)`` with NaNs for missing.
    names
        Camera names used to resolve references; defaults to ``["0", "1", ...]``.
    fixed
        References to parameters held constant, e.g. ``["*.intr", "f.rvec",
        "rm.tvec[2]"]``. A reference is ``"<cam>.<param>"`` or
        ``"<cam>.<param>[i]"`` where ``<cam>`` is a name or ``*`` (all) and
        ``<param>`` is one of ``rvec``/``tvec``/``intr``/``dist`` (``kmat`` is an
        alias of ``intr``).
    shared
        Groups of references tied to the same value, e.g.
        ``[["f.tvec[2]", "lf.tvec[2]", "rf.tvec[2]"]]``. Whole-parameter
        references in a group are tied element-wise; all members of a group
        must therefore resolve to the same number of elements.
    pts3d
        Initial 3D points; triangulated from the cameras if omitted.

    Returns
    -------
    BAState
        The packed state ready to splat into ``bundle_adjust``.
    """
    rvecs, tvecs, intrs, dists = (
        np.asarray(a, dtype=float) for a in (rvecs, tvecs, intrs, dists)
    )
    pts2d = np.asarray(pts2d, dtype=float)
    n_views = len(rvecs)
    if names is None:
        names = [str(i) for i in range(n_views)]
    name_to_row = {name: i for i, name in enumerate(names)}

    if pts3d is None:
        pts3d = initialize_pts3d(pts2d, rvecs, tvecs, intrs)
    pts3d = np.asarray(pts3d, dtype=float)

    arrs = [rvecs, tvecs, intrs, dists, pts3d]
    values = np.concatenate([a.ravel() for a in arrs])
    offsets = np.cumsum([0, *(a.size for a in arrs)])

    def make_idx(arr, off):
        return np.arange(arr.size).reshape(arr.shape) + off

    rvecs_idx = make_idx(rvecs, offsets[0])
    tvecs_idx = make_idx(tvecs, offsets[1])
    intrs_idx = make_idx(intrs, offsets[2])
    dists_idx = make_idx(dists, offsets[3])
    pts3d_idx = make_idx(pts3d, offsets[4])

    param_idx = {
        "rvec": rvecs_idx,
        "tvec": tvecs_idx,
        "intr": intrs_idx,
        "dist": dists_idx,
    }

    def ref_indices(ref: str) -> list[np.ndarray]:
        """Resolve one reference to a flat-index array per matched camera."""
        cam, sep, rest = ref.partition(".")
        if not sep or not rest:
            raise ValueError(f"malformed reference {ref!r}; expected 'cam.param'")
        if rest.endswith("]"):
            param_name, _, idx_str = rest[:-1].partition("[")
            index = int(idx_str)
        else:
            param_name, index = rest, None
        if param_name not in _PARAM_ALIASES:
            raise ValueError(
                f"unknown parameter {param_name!r} in {ref!r}; "
                f"expected one of {sorted(_PARAM_ALIASES)}"
            )
        group = _PARAM_ALIASES[param_name]
        cams = names if cam == "*" else [cam]
        out = []
        for c in cams:
            if c not in name_to_row:
                raise ValueError(f"unknown camera {c!r} in {ref!r}")
            row = param_idx[group][name_to_row[c]]
            if index is None:
                out.append(row)
            else:
                if not 0 <= index < row.size:
                    raise ValueError(f"index {index} out of range in {ref!r}")
                out.append(row[index : index + 1])
        return out

    fixed_mask = np.zeros(values.size, dtype=bool)
    for ref in fixed:
        for idxs in ref_indices(ref):
            fixed_mask[idxs] = True

    # Union-find over slots; each shared group ties its members element-wise.
    parent = list(range(values.size))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for group in shared:
        members = [arr for ref in group for arr in ref_indices(ref)]
        if not members:
            continue
        size = members[0].size
        if any(m.size != size for m in members):
            raise ValueError(
                f"shared group {group} mixes references of different lengths"
            )
        for j in range(size):
            idxs = [int(m[j]) for m in members]
            vals = values[idxs]
            if not np.allclose(vals, vals[0]):
                kept = values[min(idxs)]  # matches the compaction tie-break below
                loc = group if size == 1 else f"{group} element {j}"
                warnings.warn(
                    f"shared {loc} have differing initial values "
                    f"{vals.tolist()}; using {kept} from the lowest-indexed member",
                    stacklevel=2,
                )
            for m in members[1:]:
                union(idxs[0], int(m[j]))

    # Compact: collapse each connected component to a single slot.
    roots = np.array([find(i) for i in range(values.size)])
    uniq, old_to_new = np.unique(roots, return_inverse=True)
    old_to_new = old_to_new.reshape(-1)  # numpy version-proof: keep 1D
    new_values = np.empty(uniq.size)
    new_fixed = np.zeros(uniq.size, dtype=bool)
    filled = np.zeros(uniq.size, dtype=bool)
    for i in range(values.size):
        k = old_to_new[i]
        if not filled[k]:  # smallest original index wins the initial value
            new_values[k] = values[i]
            filled[k] = True
        new_fixed[k] |= fixed_mask[i]

    return BAState(
        new_values,
        new_fixed,
        old_to_new[rvecs_idx],
        old_to_new[tvecs_idx],
        old_to_new[intrs_idx],
        old_to_new[dists_idx],
        old_to_new[pts3d_idx],
        pts2d,
    )
