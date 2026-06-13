"""The editing session shared by the web server: state + footage + paths.

A :class:`Session` bundles everything a request handler needs to serve and edit
one ``results.h5``: the Qt-free :class:`~deeperfly.gui.state.EditorState` (the
corrections overlay and all the edit logic), the :class:`FrameSource` that
decodes footage, the on-disk paths, and the playable frame count. It carries no
web dependency -- :mod:`deeperfly.gui.server` builds the FastAPI app *around* a
session and adds the request handlers and the mutation lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .readers import FrameSource
from .state import EditorState

__all__ = ["Session"]


@dataclass
class Session:
    """One open ``results.h5`` being viewed/corrected over the web.

    Attributes
    ----------
    state
        The editor model (result + corrections overlay); holds every edit op.
    source
        The per-camera frame decoder.
    results_path
        Path to the ``results.h5`` (recorded in the saved sidecar's metadata).
    corrections_path
        Where :func:`~deeperfly.gui.corrections.save_corrections` writes.
    n_frames
        The playable frame count: the result's frames clipped to what the
        footage actually covers (so scrubbing never runs past the video).
    image_sizes
        ``camera_name -> (height, width)`` recorded by ``pose2d`` (or ``{}``),
        used to size the canvases before the first frame loads.
    """

    state: EditorState
    source: FrameSource
    results_path: str
    corrections_path: Path
    n_frames: int
    image_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        state: EditorState,
        source: FrameSource,
        *,
        results_path: str | Path,
        corrections_path: str | Path,
        image_sizes: dict[str, tuple[int, int]] | None = None,
    ) -> Session:
        """Assemble a session, clipping ``n_frames`` to the available footage."""
        n_source = source.n_frames()
        n_frames = state.n_frames if n_source is None else min(state.n_frames, n_source)
        return cls(
            state=state,
            source=source,
            results_path=str(results_path),
            corrections_path=Path(corrections_path),
            n_frames=int(n_frames),
            image_sizes=dict(image_sizes or {}),
        )
