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

The three shipped ones live in `archive/`; `v2/` is the live one.

| Document | What it designed |
| --- | --- |
| `archive/project-system-plan.md` | The project system: many recordings under one skeleton and one rig, with a shared label store. |
| `archive/keypoint-editor-redesign.md` | Rebuilding the GUI as a ground-truth annotation tool — 2D as the source, 3D derived. |
| `archive/sidebar-and-live-switch-plan.md` | The editor's sidebar and switching the live overlay between stages. |
| `v2/` | **Active.** The 0.3.0 breaking refactor: the config schema, the skeleton domain model, footage concatenation, and the removals that make the schema small. Start at `v2/README.md`. |

`v2/` **retracts item 3 of `archive/project-system-plan.md`** -- the dedicated non-skeleton
calibration landmarks. The project system, GUI-first operation and from-scratch rigs all
stay; only the landmarks go, because the animal turned out to be the better calibration
target in every case that ever ran.

For what the code actually does, read the code and `docs/`.
