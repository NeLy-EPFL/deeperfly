"""Ground truth: the ``labels.h5`` sidecar, and what is done with a set of labels.

A recording's hand labels live in a ``labels.h5`` beside its ``results.h5``, and they
are the source of truth the rest of the project answers to -- the editor writes them,
bundle adjustment and the from-scratch rig solve read them, training exports them.

- :mod:`~deeperfly.labels.store` is the artifact itself: :class:`~deeperfly.labels.store.Labels`,
  its reader and writer, and the identity check that refuses a sidecar written against
  different footage.
- :mod:`~deeperfly.labels.merge` reconciles two label sets for the same recording --
  ground truth that arrived in several places, under possibly different skeletons.
- :mod:`~deeperfly.labels.suggest` ranks the frames worth a human's next pass
  (active-learning acquisition).

This is a *lower* layer than :mod:`deeperfly.project`, not part of it: a ``labels.h5``
exists beside a lone ``results.h5`` with no project anywhere, which is what ``deeperfly
gui`` opens on a bare recording directory. A project is an index over recordings that
own labels, and depends on this package -- never the other way round.
"""

from __future__ import annotations

from . import merge, store, suggest
from .store import (
    LABELS_FORMAT_VERSION,
    Labels,
    absent_to_spans,
    export_absent,
    export_gt,
    labels_identity,
    load_labels,
    resolve_point_names,
    save_labels,
    spans_to_absent,
)

__all__ = [
    "merge",
    "store",
    "suggest",
    "Labels",
    "LABELS_FORMAT_VERSION",
    "load_labels",
    "save_labels",
    "labels_identity",
    "absent_to_spans",
    "spans_to_absent",
    "export_gt",
    "export_absent",
    "resolve_point_names",
]
