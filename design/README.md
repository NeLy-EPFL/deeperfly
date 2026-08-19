# Design record

Pre-implementation design documents, kept for the rationale rather than as
documentation. They are **not** part of the docs site and are not maintained: each
describes what was intended at the time it was written, and several sections have since
been decided differently — the documents say so where they were revised, but they were
not rewritten as the work landed.

They live here rather than under `docs/` because MkDocs publishes every `.md` beneath
that tree whether or not it appears in the nav, and an unlisted page is an INFO rather
than a warning, so `mkdocs build --strict` never flags one. All three were therefore live
on the public site, in its sitemap and its search index, complete with absolute paths
from the machine they were drafted on.

| Document | What it designed |
| --- | --- |
| `project-system-plan.md` | The project system: many recordings under one skeleton and one rig, with a shared label store. |
| `keypoint-editor-redesign.md` | Rebuilding the GUI as a ground-truth annotation tool — 2D as the source, 3D derived. |
| `sidebar-and-live-switch-plan.md` | The editor's sidebar and switching the live overlay between stages. |

For what the code actually does, read the code and `docs/`.
