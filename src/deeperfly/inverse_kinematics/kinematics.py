"""Forward kinematics for a serial leg chain, in JAX (grad-friendly).

A leg is a chain of joints from the body-fixed thorax-coxa (ThC) outward to the
claw. Each joint rotates the running frame about its revolute DOF axes (applied in
order) and the *next* joint's segment then offsets the position along the rest
direction ``-z`` of that running frame. With every angle zero the chain reproduces
the rest pose: a straight leg pointing down (``-z``) in the leg-local frame whose
``x`` is anterior, ``y`` is left and ``z`` is dorsal (up).

The model is intentionally tiny (a leg is 5 joints / up to 7 DOFs), so the
per-joint Python loop is fully unrolled inside the jitted kernel. Reverse-mode
autodiff (:func:`jax.jacrev`) over the flat angle vector gives the analytic
Jacobian the bounded least-squares solver (:mod:`deeperfly.inverse_kinematics.core`)
feeds to scipy -- the same JAX-Jacobian-into-scipy pattern as bundle adjustment.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["make_leg_fk", "make_leg_kernels"]

#: The rest direction every segment extends along in its parent joint's frame
#: (straight down in the leg-local frame).
_REST = jnp.array([0.0, 0.0, -1.0])


def _axis_rmat(axis: Float[Array, "3"], angle: Float[Array, ""]) -> Float[Array, "3 3"]:
    """Rotation by ``angle`` about the fixed unit ``axis`` (Rodrigues, sin/cos form).

    ``R = I + sin(t) K + (1 - cos(t)) K^2`` with ``K = skew(axis)``. Unlike the
    general axis-angle :func:`deeperfly.geometry.rvec_to_rmat_one` (which divides by
    the rotation-vector norm and has a ``jnp.where`` whose unused branch poisons the
    gradient at ``t = 0``), this is smooth in ``angle`` everywhere -- essential
    because a joint angle is routinely exactly zero (the rest pose) and the solver
    differentiates through it. ``axis`` must be a unit vector.
    """
    ax, ay, az = axis[0], axis[1], axis[2]
    k = jnp.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    s, c = jnp.sin(angle), jnp.cos(angle)
    return jnp.eye(3) + s * k + (1.0 - c) * (k @ k)


def make_leg_fk(
    dof_counts: tuple[int, ...],
) -> Callable[
    [Float[Array, "D"], Float[Array, "D 3"], Float[Array, "J"]], Float[Array, "J 3"]
]:
    """Build the forward-kinematics function for a chain with a fixed DOF layout.

    ``dof_counts`` is static (it comes from the template, not the data), so the
    returned ``fk`` unrolls the chain and is safe to :func:`jax.jit` and
    :func:`jax.jacrev` over the angle vector.

    Parameters
    ----------
    dof_counts
        Number of revolute DOFs at each joint, in chain order (e.g.
        ``(3, 2, 1, 1, 0)`` for ThC, CTr, FTi, TiTa, Claw). Its length is the
        joint count ``J`` and its sum is the total DOF count ``D``.

    Returns
    -------
    fk : callable
        ``fk(angles, axes, seglens) -> (J, 3)`` joint positions in the leg-local
        frame. ``angles`` is ``(D,)``; ``axes`` is ``(D, 3)`` unit rotation axes
        (one per DOF, in chain order); ``seglens`` is ``(J,)`` the segment length
        leading *into* each joint (``seglens[0]`` is unused -- the root sits at the
        origin).
    """
    n_joints = len(dof_counts)

    def fk(
        angles: Float[Array, "D"],
        axes: Float[Array, "D 3"],
        seglens: Float[Array, "J"],
    ) -> Float[Array, "J 3"]:
        rot = jnp.eye(3)
        pos = jnp.zeros(3)
        positions = []
        d = 0
        for j in range(n_joints):
            # Offset from the previous joint along the parent frame's rest axis.
            pos = pos + rot @ (_REST * seglens[j])
            positions.append(pos)
            # This joint's DOFs rotate the running frame for the segments below it.
            for _ in range(dof_counts[j]):
                rot = rot @ _axis_rmat(axes[d], angles[d])
                d += 1
        return jnp.stack(positions)

    return fk


def make_leg_kernels(dof_counts: tuple[int, ...]):
    """Jitted forward-kinematics value and angle-Jacobian for a DOF layout.

    Returns
    -------
    fk, jac : callable
        ``fk(angles, axes, seglens) -> (J, 3)`` and its Jacobian w.r.t. ``angles``
        ``jac(angles, axes, seglens) -> (J, 3, D)``, both :func:`jax.jit`-wrapped.
    """
    fk = make_leg_fk(dof_counts)
    fk_jit = jax.jit(fk)
    jac_jit = jax.jit(jax.jacrev(fk, argnums=0))
    return fk_jit, jac_jit
