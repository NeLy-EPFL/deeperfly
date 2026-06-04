"""Pin deeperfly's JAX geometry and bundle-adjustment math to the CPU.

deeperfly's projection / triangulation / Rodrigues primitives
(:mod:`deeperfly.geometry`) and its bundle adjustment
(:mod:`deeperfly.bundle_adjustment`) run on tiny arrays -- a handful of cameras
and points -- so a GPU's host<->device transfer and kernel-launch overhead
dwarfs the arithmetic and the GPU never helps. Worse, the first GPU computation
makes JAX preallocate ~75% of the VRAM, an unwelcome surprise for anyone who
imported deeperfly only for its host-side geometry. So this module pins that
math to the CPU and leaves the GPU (and its memory) untouched unless the
detector -- the one part that benefits -- explicitly asks for it (see
:mod:`deeperfly.pose2d.backends.jax`, which keeps the default GPU backend).

Import this module before creating any JAX array: it enables float64 and, on
Apple Silicon, steers the default device back to the CPU. Decorate a function
with :func:`cpu_jit` to make it always compile and run on the CPU.
"""

from __future__ import annotations

import functools
import sys

import jax

# The geometry / bundle-adjustment math is closed-form camera algebra that needs
# float64 to stay accurate; enable it process-wide before any array is created.
jax.config.update("jax_enable_x64", True)

#: The CPU device every geometry / bundle-adjustment kernel is pinned to.
CPU = jax.devices("cpu")[0]

# The optional jax-mps plugin (Apple Silicon) registers an experimental Metal
# backend and makes it the *default* device -- but MLX is float32-only, so the
# x64 math here would crash on it. Force the default device back to the CPU so
# stray array creation lands there; the detector opts into Metal explicitly (it
# is float32). No-op without jax-mps, and CUDA / CPU defaults are untouched. Gate
# on macOS first: ``jax.default_backend()`` forces a backend to initialize, and
# only Apple Silicon can ever report "mps", so elsewhere we skip the probe.
if sys.platform == "darwin" and jax.default_backend() == "mps":
    jax.config.update("jax_default_device", CPU)


def cpu_jit(fn):
    """JIT ``fn`` and force its compilation and execution onto the CPU.

    Committing the inputs to the CPU with :func:`jax.device_put` makes XLA
    compile and run ``fn`` on the CPU and keep its output there, regardless of
    the process-wide default device. So callers that import these helpers into a
    GPU program (e.g. alongside the detector) still get CPU execution -- and the
    GPU is never touched, so JAX never preallocates its VRAM. (The ``device=`` /
    ``backend=`` arguments to :func:`jax.jit` that used to express this are
    deprecated; ``device_put`` on the inputs is the supported replacement.)
    """
    jitted = jax.jit(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = jax.device_put(args, CPU)
        kwargs = jax.device_put(kwargs, CPU)
        return jitted(*args, **kwargs)

    return wrapper
