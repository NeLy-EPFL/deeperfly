"""The editing session shared by the web server: state + footage + paths.

A :class:`Session` bundles everything a request handler needs to serve and edit
one ``results.h5``: the Qt-free :class:`~deeperfly.gui.state.EditorState` (the
corrections overlay and all the edit logic), the :class:`FrameSource` that
decodes footage, the on-disk paths, and the playable frame count. It carries no
web dependency -- :mod:`deeperfly.gui.server` builds the FastAPI app *around* a
session and adds the request handlers and the mutation lock.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .labels import labels_identity
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
    labels_path
        Where :func:`~deeperfly.gui.labels.save_labels` writes the ``labels.h5`` sidecar.
    identity
        The recording fingerprint stamped into ``labels.h5`` (see
        :func:`~deeperfly.gui.labels.labels_identity`), so a stale sidecar is refused.
    n_frames
        The playable frame count: the result's frames clipped to what the
        footage actually covers (so scrubbing never runs past the video).
    image_sizes
        ``camera_name -> (height, width)`` recorded by ``pose2d`` (or ``{}``),
        used to size the canvases before the first frame loads.
    nmf_hide_parts
        Body parts hidden from the NMF mesh overlay (``["wings"]`` by default; from
        ``[gui].mesh_hide`` in the run config). The render videos carry their own
        ``[visualization].mesh_hide`` list.
    """

    state: EditorState
    source: FrameSource
    results_path: str
    labels_path: Path
    n_frames: int
    identity: dict = field(default_factory=dict)
    image_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    nmf_hide_parts: tuple[str, ...] = ("wings",)

    @classmethod
    def build(
        cls,
        state: EditorState,
        source: FrameSource,
        *,
        results_path: str | Path,
        labels_path: str | Path,
        identity: dict | None = None,
        footage: dict | None = None,
        image_sizes: dict[str, tuple[int, int]] | None = None,
        nmf_hide_parts: "Sequence[str]" = ("wings",),
    ) -> Session:
        """Assemble a session, clipping ``n_frames`` to the available footage.

        ``identity`` fingerprints the recording for the labels sidecar; when omitted
        it is derived from the state (skeleton points, cameras, frame count) plus
        ``image_sizes`` and ``footage`` -- enough to refuse a labels file from a
        different recording.
        """
        n_source = source.n_frames()
        n_frames = state.n_frames if n_source is None else min(state.n_frames, n_source)
        if identity is None:
            identity = labels_identity(
                point_names=list(state.result.skeleton.point_names),
                camera_names=list(state.camera_names),
                n_frames=state.n_frames,
                image_sizes=image_sizes,
                footage=footage,
            )
        return cls(
            state=state,
            source=source,
            results_path=str(results_path),
            labels_path=Path(labels_path),
            n_frames=int(n_frames),
            identity=identity,
            image_sizes=dict(image_sizes or {}),
            nmf_hide_parts=tuple(nmf_hide_parts),
        )
