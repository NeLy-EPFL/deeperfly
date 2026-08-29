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

from ..labels import labels_identity
from ..labels.suggest import SUGGESTIONS_FILENAME
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
        Where :func:`~deeperfly.labels.store.save_labels` writes the ``labels.h5`` sidecar.
    suggestions_path
        Where ``deeperfly labels-suggest`` writes its ranked-frame sidecar
        (``labels_suggest.json``). Defaults to the file of that name beside
        ``labels_path``. The GUI only ever *reads* it, and its absence is normal --
        the Suggested tab then just says how to produce it.
    identity
        The recording fingerprint stamped into ``labels.h5`` (see
        :func:`~deeperfly.labels.store.labels_identity`), so a stale sidecar is refused.
    n_frames
        The playable frame count: the result's frames clipped to what the
        footage actually covers (so scrubbing never runs past the video).
    image_sizes
        ``camera_name -> (height, width)`` recorded by ``pose2d`` (or ``{}``),
        used to size the canvases before the first frame loads.
    model_hide_parts
        Body parts hidden from the model mesh overlay (``["wings"]`` by default; from
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
    model_hide_parts: tuple[str, ...] = ("wings",)
    # Filled in by __post_init__ when not given, so every Session -- however it was
    # constructed -- has a concrete path to look for the suggestions sidecar at.
    suggestions_path: Path | None = None
    #: The project this recording belongs to, when it was opened through one. ``None`` for a
    #: bare ``results.h5``, which is what decides whether the editor can *run* anything:
    #: a job needs a project root to work in and a recording to name.
    project_root: Path | None = None
    #: The recording's project slug, for job arguments and the window title. ``None`` for a
    #: bare ``results.h5``.
    recording_slug: str | None = None
    #: The calibration the editor is deriving non-GT positions from, when the operator has
    #: chosen one (``POST /api/calibrations/select``). ``None`` means the rig that came out of
    #: ``results.h5``, which is the default and what a pipeline run would use. Recorded so the
    #: tab can show which rig is live -- a residual is only interpretable against a named rig.
    active_calibration: str | None = None

    def __post_init__(self) -> None:
        if self.suggestions_path is None:
            self.suggestions_path = Path(self.labels_path).parent / SUGGESTIONS_FILENAME
        else:
            self.suggestions_path = Path(self.suggestions_path)

    @classmethod
    def build(
        cls,
        state: EditorState,
        source: FrameSource,
        *,
        results_path: str | Path,
        labels_path: str | Path,
        suggestions_path: str | Path | None = None,
        identity: dict | None = None,
        footage: dict | None = None,
        image_sizes: dict[str, tuple[int, int]] | None = None,
        model_hide_parts: "Sequence[str]" = ("wings",),
        project_root: Path | None = None,
        recording_slug: str | None = None,
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
            suggestions_path=None
            if suggestions_path is None
            else Path(suggestions_path),
            n_frames=int(n_frames),
            identity=identity,
            image_sizes=dict(image_sizes or {}),
            model_hide_parts=tuple(model_hide_parts),
            project_root=None if project_root is None else Path(project_root),
            recording_slug=recording_slug,
        )
