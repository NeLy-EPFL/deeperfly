"""JAX configuration for deeperfly's geometry and bundle-adjustment math.

deeperfly's projection / triangulation / Rodrigues primitives
(:mod:`deeperfly.geometry`) and its bundle adjustment
(:mod:`deeperfly.bundle_adjustment`) are the only JAX in deeperfly -- the detector
is PyTorch. They run on tiny arrays (a handful of cameras and points), so deeperfly
installs only CPU JAX: a GPU's host<->device transfer and kernel-launch overhead
would dwarf the arithmetic. With no accelerator backend to land on, the math runs
on the CPU automatically -- nothing has to pin it there.

Import this module before creating any JAX array: it enables float64, which the
geometry needs and which must be set process-wide before the first array exists.
Decorate a function with :func:`cpu_jit` to JIT it.
"""

from __future__ import annotations

import jax

# The geometry / bundle-adjustment math is closed-form camera algebra that needs
# float64 to stay accurate; enable it process-wide before any array is created.
jax.config.update("jax_enable_x64", True)

#: JIT compiler for the geometry / bundle-adjustment kernels. A thin alias for
#: :func:`jax.jit`: with only CPU JAX installed there is no device to steer away
#: from, so the kernels compile and run on the CPU on their own.
cpu_jit = jax.jit
