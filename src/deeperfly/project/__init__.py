"""Projects: related recordings, one skeleton, shared rigs -- an index, not a re-home.

:class:`~deeperfly.project.core.Project` groups the recordings of an experiment under one
skeleton and one set of camera rigs, and gives them one place to see what is labeled. It
*indexes*: a recording's ``results.h5`` and ``labels.h5`` stay where they are and are
adopted by symlink, so nothing is ever copied or moved to create one.

Around that core sit the operations that act on a whole project:

- :mod:`~deeperfly.project.package` writes it out as a single ``.dfpkg``, labeled frames
  embedded, and reads one back;
- :mod:`~deeperfly.project.import_outputs` adopts a stray ``deeperfly_outputs/`` tree the
  project already indexes;
- :mod:`~deeperfly.project.migrate` applies a skeleton edit as a typed migration -- the
  guard on the sharpest hazard in the project;
- :mod:`~deeperfly.project.jobs` runs the pipeline in the background for the editor.
"""

from __future__ import annotations

from . import import_outputs, jobs, migrate, package  # noqa: E402
from .core import (
    OUTPUTS_DIRNAME,
    PROJECT_FILENAME,
    PROJECT_FORMAT_VERSION,
    SKELETON_FILENAME,
    SKELETON_PRESETS,
    Project,
    RecordingEntry,
    discover_footage,
    label_stats,
    recording_fingerprint,
    recording_id,
)

__all__ = [
    "import_outputs",
    "jobs",
    "migrate",
    "package",
    "Project",
    "RecordingEntry",
    "PROJECT_FORMAT_VERSION",
    "PROJECT_FILENAME",
    "SKELETON_FILENAME",
    "SKELETON_PRESETS",
    "OUTPUTS_DIRNAME",
    "recording_fingerprint",
    "recording_id",
    "label_stats",
    "discover_footage",
]
