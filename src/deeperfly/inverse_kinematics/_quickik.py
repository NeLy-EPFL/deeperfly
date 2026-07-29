"""Access to the optional QuickIK solver, behind one explicit gate.

QuickIK (https://nely-epfl.github.io/quickik/) is the solver the
``inverse_kinematics`` stage runs, but it is an *optional* dependency: it publishes no
wheels, so installing it builds a Rust extension and needs a Rust toolchain -- which
nothing else in deeperfly does. The stage is opt-in and off by default, so a plain
install must stay Rust-free.

Everything that needs the solver calls :func:`require_quickik` **inside a function
body**, never at module scope. That is what keeps ``import deeperfly`` (which re-exports
:func:`~deeperfly.inverse_kinematics.solve_inverse_kinematics`) and
``deeperfly.gui.state`` (which imports the live re-fit) working without the extra. The
body plan and forward kinematics deliberately import nothing from here, so a result file
that already holds a fit still renders its overlays -- in videos and in the editor -- on
an install that cannot solve.
"""

from __future__ import annotations

import functools
from types import ModuleType

__all__ = ["MissingQuickIK", "require_quickik"]

#: How to get it. No wheels are published, so there is no plain ``pip install quickik``.
INSTALL_HINT = (
    'uv sync --extra ik   (or: pip install "quickik @ '
    'git+https://github.com/NeLy-EPFL/quickik#subdirectory=python")'
)


class MissingQuickIK(ImportError):
    """Raised when inverse kinematics is asked for without the ``ik`` extra installed.

    An :class:`ImportError` on purpose: the GUI already treats a failed live-re-fit
    setup as "fall back to the stored fit", so a missing solver degrades there instead
    of breaking the editor.
    """


@functools.cache
def require_quickik() -> ModuleType:
    """The ``quickik`` module, or :class:`MissingQuickIK` explaining how to install it."""
    try:
        import quickik
    except ImportError as exc:  # pragma: no cover -- exercised by the install-less path
        raise MissingQuickIK(
            "inverse kinematics needs the optional QuickIK solver, which is not "
            f"installed. Install the extra with:\n    {INSTALL_HINT}\n"
            "It builds a Rust extension, so a Rust toolchain is required "
            "(https://rustup.rs). Rendering an existing fit does not need it."
        ) from exc
    return quickik
