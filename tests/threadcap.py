"""Cap each xdist worker's BLAS/OpenMP thread pools to one thread.

Loaded via ``-p threadcap`` (see ``addopts`` in ``pyproject.toml``) so that it runs
*before* numpy, jax or torch are imported and can still size their thread pools.

Without this, ``-n auto`` is worthless: torch defaults to one intra-op thread per
core (24 here), so N workers ask for N x 24 threads on 24 cores and the machine
spends its time context-switching. Measured on the full suite: ``-n 16`` takes
120 s uncapped versus 39 s capped, against 115 s serial.

The cap is deliberately scoped to workers -- ``PYTEST_XDIST_WORKER`` is unset in a
plain serial run, which genuinely wants the wide thread pools (capping a serial run
makes it *slower*: 146 s versus 115 s).
"""

import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
