"""Projects: related recordings, one skeleton, shared rigs -- an index, not a re-home.

Labels arrive scattered. A recording's ground truth lives in a ``labels.h5`` beside its
``results.h5``, which lives wherever the operator happened to run the pipeline, and the
only thing tying a body of work together is a directory naming convention plus, in
practice, a hand-generated manifest. Nothing says *these recordings share a skeleton*,
nothing says *these share a camera rig*, and nothing can tell one recording from a backup
copy of it.

A project says those things. The load-bearing decision is that it says them **without
moving any data**:

- ``results.h5`` and ``labels.h5`` stay exactly where they are. A recording is *adopted*
  by reference (a symlink), so the file the editor writes is the file the training set
  reads, and no hand label is copied, renamed, or migrated to create a project.
- Footage is referenced by path, never copied.
- Every existing command keeps working on a bare recording. A project is a layer above,
  never a precondition.

.. code-block:: text

    myproject/
        project.toml                  the manifest: identity, settings, recording index
        skeleton.toml                 the project's skeleton (a [skeleton] config fragment)
        calibrations/<name>.toml      solved rigs (deeperfly.calibration)
        recordings/<slug>/
            recording.toml            this recording's footage pointers
            deeperfly_outputs/        -> symlink to the adopted outputs, or a real dir

**Recording identity.** A recording is identified by its *content*, not its path, so the
same recording adopted twice -- or arriving through a merge -- is recognized as one thing.
This matters more than it sounds: in this rig every recording's files are named
``camera_RH.mp4`` ... ``camera_LH.mp4``, so basenames alone identify nothing. The
fingerprint therefore needs something recording-specific, and there are two ways to get
one (see :func:`recording_fingerprint`):

``footage``
    ``camera:basename:size_bytes`` per camera. Preferred, and cheap -- a ``stat`` per
    file, no decode. Byte size means a re-encode yields a *different* id, which is
    correct: ground truth is in footage pixels, so re-encoded footage is a different
    recording to label against (the same reasoning as
    :func:`deeperfly.gui.labels.labels_identity`).

``result``
    ``camera:basename`` plus the frame count, read from ``results.h5``. The fallback for
    footage that no longer resolves -- an unmounted share, an archived recording whose
    labels are still wanted.

The basis is recorded per entry, because the two cannot be compared: a recording adopted
once by each basis gets two ids. That is reported, not silently deduplicated.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import _toml

__all__ = [
    "Project",
    "RecordingEntry",
    "PROJECT_FORMAT_VERSION",
    "PROJECT_FILENAME",
    "SKELETON_FILENAME",
    "SKELETON_PRESETS",
    "recording_fingerprint",
    "recording_id",
    "label_stats",
    "discover_footage",
]

log = logging.getLogger("deeperfly")

#: Bumped when the manifest schema changes incompatibly. A project written by a *newer*
#: deeperfly is refused rather than silently misread.
PROJECT_FORMAT_VERSION = 1

PROJECT_FILENAME = "project.toml"
SKELETON_FILENAME = "skeleton.toml"
RECORDING_FILENAME = "recording.toml"
RIG_FILENAME = "rig.toml"
PROFILE_DIRNAME = "profiles"
DEFAULT_PROFILE = "default.toml"
OUTPUTS_DIRNAME = "deeperfly_outputs"


def skeleton_preset_names() -> tuple[str, ...]:
    """Skeletons ``deeperfly project new --skeleton`` understands, besides a path.

    The packaged skeletons (:func:`deeperfly.config.skeleton_presets`) plus ``blank``.
    Derived rather than listed, so a skeleton added to the package is offered here
    without a second place to remember.
    """
    from .config import skeleton_presets

    return (*sorted(skeleton_presets()), "blank")


#: Backwards-compatible snapshot of :func:`skeleton_preset_names`.
SKELETON_PRESETS = skeleton_preset_names()

_BLANK_SKELETON = """\
# The project's skeleton: FOUR things -- points, edges, symmetries, colors.
#
# `points` is the source of truth for the tracked-point ORDER, which every
# (V, T, P, ...) array and every stored label indexes into positionally -- so adding a
# point is safe, and reordering or renaming one is a migration.
#
# `edges` is the whole topology, as point pairs: the bones drawn in the editor, the
# length prior, pictorial structures' graph. There is no grouping concept -- a point in
# no edge is still tracked, and what a group name was for is a `*` pattern in `colors`.
#
# `symmetries` pairs each point with its mirror; the loader checks the pairs are an
# automorphism of `edges`, which is what makes hand-written rows safe.
[skeleton]
name = "unnamed"
points = []
# edges = [["hip", "knee"], ["knee", "ankle"]]
# symmetries = [["l_hip", "r_hip"]]

[skeleton.colors]
# "l_*" = "#0f7399"
"""

#: Top-level config tables a project's ``rig.toml`` owns: the cameras (their shared
#: values, their footage patterns and their geometry) and decode. Everything else stays in
#: the run config (the detector, the algorithm knobs) or in the project's own files (the
#: skeleton, the calibrations).
RIG_TABLES = ("default_camera", "cameras", "io")

#: What a fresh ``profiles/default.toml`` contains: nothing but an explanation. A profile
#: holds only the keys that DIFFER from the packaged defaults, so an empty one is the
#: correct starting state and its emptiness is the point.
_BLANK_PROFILE = """\
# Algorithm settings for this project -- ONLY the keys that differ from the packaged
# defaults. An empty file means "use the defaults for everything", which is the right
# starting state; `deeperfly config show` lists every key with its default, and marks the
# ones a config actually sets.
#
# The composed config (skeleton + rig + this file) is what a run consumes:
#
#     deeperfly project config .            # print it
#     deeperfly project config . -o run.toml && deeperfly run REC -c run.toml
#
# Do NOT put [[sources]] or [cameras...] here -- those live in rig.toml, and a table
# declared in two fragments would make the composition invalid.

# [triangulation]
# method = "ransac"

# [pipeline]
# do_inverse_kinematics = true
"""

#: A project/recording slug: safe as a directory name on every platform we target, and
#: safe to embed in a URL path (the GUI will address recordings by slug).
_SLUG_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    """A filesystem- and URL-safe slug from arbitrary text (never empty)."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")
    return out or "recording"


# -- identity ------------------------------------------------------------------


def recording_fingerprint(
    footage: dict[str, list[Path]] | None = None,
    *,
    result_footage: dict | None = None,
    n_frames: int | None = None,
) -> tuple[str, str]:
    """``(fingerprint text, basis)`` identifying a recording by content.

    Prefers the ``footage`` basis (a ``stat`` per file, no decode) and falls back to the
    ``result`` basis when the files do not resolve. See the module docstring for why
    basenames alone are not enough on this rig.

    Parameters
    ----------
    footage
        ``camera -> footage files`` on disk. Used when every listed file exists.
    result_footage
        The map :meth:`deeperfly.results.StageStore.read_footage` returns
        (``camera -> {"abs"|"rel": [paths]}``), for the fallback.
    n_frames
        The recording's frame count, for the fallback (it is what makes the fallback
        discriminative, since the basenames repeat across recordings).

    Returns
    -------
    text, basis : str
        The fingerprint text and which basis produced it (``"footage"`` or
        ``"result"``).

    Raises
    ------
    ValueError
        If neither basis has enough to identify the recording.
    """
    if footage:
        parts = []
        resolvable = True
        for camera, files in sorted(footage.items()):
            for path in files:
                p = Path(path)
                try:
                    size = p.stat().st_size
                except OSError:
                    resolvable = False
                    break
                parts.append(f"{camera}:{p.name}:{size}")
            if not resolvable:
                break
        if resolvable and parts:
            return "\0".join(parts), "footage"

    if result_footage:
        parts = []
        for camera, spec in sorted(result_footage.items()):
            names = _basenames(spec)
            parts += [f"{camera}:{name}" for name in names]
        if parts:
            parts.append(f"frames:{'' if n_frames is None else int(n_frames)}")
            return "\0".join(parts), "result"

    raise ValueError(
        "cannot identify this recording: its footage does not resolve and its "
        "results.h5 records no footage paths either. Re-run 'deeperfly run' to embed "
        "them, or pass an explicit --id"
    )


def _basenames(spec) -> list[str]:
    """Sorted footage basenames from any footage pointer (see :mod:`deeperfly.footage`)."""
    from .footage import basenames

    return basenames(spec)


def recording_id(fingerprint: str) -> str:
    """A stable ``rec_<hex>`` id from a :func:`recording_fingerprint` text."""
    return "rec_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def project_id() -> str:
    """A fresh random ``prj_<hex>`` id (stable across renames and moves)."""
    return "prj_" + hashlib.sha256(os.urandom(32)).hexdigest()[:8]


# -- footage discovery ---------------------------------------------------------


def discover_footage(root: Path, config=None) -> dict[str, list[Path]]:
    """``camera -> footage files`` for a recording directory.

    Two modes, because a project has to work *before* it has a rig:

    - With a ``config``, the configured per-source globs are used
      (:func:`deeperfly.recordings.find_recording`) -- the same discovery ``deeperfly
      run`` does, so an adopted recording resolves to exactly the cameras a run would.
    - Without one, each video file in ``root`` becomes a camera named after its stem.
      That is the from-scratch case: the operator drops seven videos in a folder and has
      not described a rig yet, so the *files* are the only statement of what the cameras
      are.

    Image sequences are only found in the ``config`` mode: grouping loose image files
    into per-camera sequences needs a naming convention, and guessing one would silently
    mis-group them. A recording that already has a ``results.h5`` should be fingerprinted
    from *its* recorded footage instead (see :func:`recording_fingerprint`).

    Parameters
    ----------
    root
        The recording directory.
    config
        Optional :class:`~deeperfly.config.Config` supplying the per-source globs.

    Returns
    -------
    dict of str to list of Path
        ``camera -> footage files``; empty when nothing was found.
    """
    root = Path(root)
    if config is not None:
        from .recordings import find_recording

        return find_recording(root, config) or {}

    from natsort import natsorted

    from .io import VIDEO_EXTS

    videos = natsorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    return {p.stem: [p] for p in videos}


# -- label statistics ----------------------------------------------------------


def _coo_rows(raw):
    """A stored COO index as ``(N, width)``, whatever width it was written with.

    ``labels.h5`` widened its cell indices at v6 (an instance column was reserved), so a
    reader that hard-codes three columns raises on a current file. Only the frame column
    is read downstream, and it is column 1 in both layouts.
    """
    import numpy as np

    arr = np.asarray(raw)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return arr.reshape(-1, arr.shape[-1])


def _gt_cells(labels_path: Path) -> set[tuple[int, int, int]]:
    """The ``(view, frame, point)`` cells a ``labels.h5`` holds ground truth for.

    Read straight out of HDF5 for the same reason :func:`label_stats` is: a comparison
    between two label sets has no business requiring either to validate against a result.
    Empty for an absent or unreadable file, which makes a comparison against it degrade to
    "everything is unseen" rather than raising.
    """
    path = Path(labels_path)
    if not path.exists():
        return set()
    try:
        import h5py

        with h5py.File(path, "r") as f:
            rows = _coo_rows(f["gt/index"][()]) if "gt/index" in f else []
        # The point column is last in both index widths (v6 widened the middle), and the
        # view/frame columns are 0 and 1 in both.
        return {(int(r[0]), int(r[1]), int(r[-1])) for r in rows}
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return set()


def label_stats(labels_path: Path) -> dict:
    """Counts from a ``labels.h5``, read straight out of HDF5.

    Deliberately does **not** go through :func:`deeperfly.gui.labels.load_labels`: that
    validates the sidecar's identity against a result, which a status listing has no
    business requiring (and which would make the whole listing fail on one stale file).
    The sparse indices are all that is needed to count.

    Counts *live* rows, so a keypoint declared absent -- whose labels are quarantined
    under ``absent/void_*`` -- is not counted as ground truth.

    ``gt_points`` is every stored row, which is exactly what
    :func:`deeperfly.gui.labels.export_gt` yields: there is one kind of ground truth -- a
    pixel the operator created -- so the progress number and the export cannot disagree.

    Parameters
    ----------
    labels_path
        Path to a ``labels.h5`` (may be absent).

    Returns
    -------
    dict
        ``gt_points``,
        ``occluded``, ``labeled_frames``, ``reviewed_frames``, ``absent_points`` and
        ``format_version``. All zero when the file is absent or unreadable (with a
        warning in the unreadable case).
    """
    empty = {
        "gt_points": 0,
        "occluded": 0,
        "labeled_frames": 0,
        "reviewed_frames": 0,
        "absent_points": 0,
        "format_version": None,
    }
    path = Path(labels_path)
    if not path.exists():
        return empty
    try:
        import json

        import h5py
        import numpy as np

        with h5py.File(path, "r") as f:
            # Width-tolerant: v6+ stores [view, frame, instance, point], earlier versions
            # [view, frame, point]. Only column 1 (frame) is read, and it is column 1 in
            # both -- but the reshape must not assume a width, or a v6 file would raise
            # here and a whole project listing would report zeros.
            gt_index = _coo_rows(f["gt/index"][()])
            occ = _coo_rows(f["occluded/index"][()])
            reviewed = (
                np.asarray(f["reviewed/index"][()]).reshape(-1)
                if "reviewed" in f
                else []
            )
            absent = (
                np.asarray(f["absent/index"][()]).reshape(-1) if "absent" in f else []
            )
            meta = json.loads(f.attrs.get("meta", "{}"))
        return {
            "gt_points": int(len(gt_index)),
            "occluded": int(len(occ)),
            # Frames carrying human work: the distinct frame column of the GT rows.
            "labeled_frames": int(len(set(gt_index[:, 1].tolist())))
            if len(gt_index)
            else 0,
            "reviewed_frames": int(len(reviewed)),
            "absent_points": int(len(absent)),
            "format_version": meta.get("deeperfly_labels_format_version"),
        }
    except Exception as exc:  # a corrupt sidecar must not break a whole listing
        log.warning("could not read %s: %s", path, exc)
        return empty


# -- the manifest --------------------------------------------------------------


@dataclass(frozen=True)
class RecordingEntry:
    """One recording's row in the project index.

    Attributes
    ----------
    id
        Content-derived ``rec_<hex>`` (see :func:`recording_id`). Stable across moves
        and renames; the key merge deduplicates on.
    slug
        Human-facing name, and the directory name under ``recordings/``. Free to rename.
    id_basis
        Which fingerprint produced ``id`` (``"footage"``, ``"result"``, or ``"manual"``).
        Recorded because ids from different bases are not comparable.
    subject
        Optional animal identifier, so one specimen's several clips can be grouped (and
        an absence declaration shared between them).
    calibration
        Project-relative path to this recording's rig, or ``None`` to use the project's
        current one.
    n_frames, fps
        Cached in the index so a listing needs no file opens.
    added_utc
        When it was adopted.
    origin
        How it got here (``kind`` = ``adopted`` / ``created`` / ``merged``, plus the
        source path), for provenance.
    """

    id: str
    slug: str
    path: str
    id_basis: str = "footage"
    subject: str | None = None
    calibration: str | None = None
    n_frames: int | None = None
    fps: float | None = None
    added_utc: str = ""
    origin: dict = field(default_factory=dict)

    def as_table(self) -> dict:
        """The entry as a TOML-writable mapping (dropping unset optionals)."""
        out: dict = {
            "id": self.id,
            "slug": self.slug,
            "path": self.path,
            "id_basis": self.id_basis,
        }
        for name in ("subject", "calibration", "n_frames", "fps", "added_utc"):
            v = getattr(self, name)
            if v is not None and v != "":
                out[name] = v
        if self.origin:
            out["origin"] = dict(self.origin)
        return out


@dataclass
class Project:
    """A loaded project: the manifest, plus path resolution for its recordings."""

    root: Path
    name: str
    id: str
    description: str = ""
    iteration: int = 0
    created_utc: str = ""
    format_version: int = PROJECT_FORMAT_VERSION
    skeleton_file: str = SKELETON_FILENAME
    calibration: str | None = None
    recordings: list[RecordingEntry] = field(default_factory=list)

    # -- construction ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        name: str | None = None,
        skeleton: str = "fly38",
        description: str = "",
        exist_ok: bool = False,
    ) -> Project:
        """Create a new project directory.

        Parameters
        ----------
        root
            Directory to create (created with parents; must not already hold a
            ``project.toml`` unless ``exist_ok``).
        name
            Project name; defaults to the directory's name.
        skeleton
            One of :data:`SKELETON_PRESETS`, or a path to a TOML file holding a
            ``[skeleton]`` table (copied in).
        description
            Free-text description.
        exist_ok
            Allow an existing project to be re-created (leaves its ``skeleton.toml``
            alone).

        Returns
        -------
        Project
            The created project, already saved.

        Raises
        ------
        FileExistsError
            If ``root`` already holds a project and ``exist_ok`` is false.
        """
        root = Path(root)
        manifest = root / PROJECT_FILENAME
        if manifest.exists() and not exist_ok:
            raise FileExistsError(
                f"{manifest} already exists -- pass a different directory, or open the "
                "existing project"
            )
        (root / "recordings").mkdir(parents=True, exist_ok=True)
        (root / "calibrations").mkdir(parents=True, exist_ok=True)

        skeleton_path = root / SKELETON_FILENAME
        if not skeleton_path.exists():
            skeleton_path.write_text(_skeleton_text(skeleton))
        # An empty profile, whose emptiness is the point: it holds only what DIFFERS from
        # the packaged defaults, so a project starts by overriding nothing.
        profile = root / PROFILE_DIRNAME / DEFAULT_PROFILE
        if not profile.exists():
            profile.parent.mkdir(parents=True, exist_ok=True)
            profile.write_text(_BLANK_PROFILE)

        project = cls(
            root=root,
            name=name or root.resolve().name,
            id=project_id(),
            description=description,
            created_utc=_now(),
        )
        project.save()
        return project

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """Open a project from its directory or its ``project.toml``.

        Raises
        ------
        FileNotFoundError
            If there is no project there.
        ValueError
            If the manifest was written by a newer deeperfly, or is malformed.
        """
        manifest = Path(path)
        if manifest.is_dir():
            manifest = manifest / PROJECT_FILENAME
        if not manifest.exists():
            raise FileNotFoundError(
                f"no project at {manifest} -- create one with 'deeperfly project new'"
            )
        data = tomllib.loads(manifest.read_text())
        head = data.get("project")
        if not isinstance(head, dict):
            raise ValueError(f"{manifest} has no [project] table")
        version = int(head.get("format_version", 1))
        if version > PROJECT_FORMAT_VERSION:
            raise ValueError(
                f"{manifest} was written by a newer deeperfly (project format "
                f"v{version}, this build understands v{PROJECT_FORMAT_VERSION}); "
                "refusing to read it rather than silently dropping state it carries"
            )
        root = manifest.parent
        entries = []
        for row in data.get("recordings", []) or []:
            if not isinstance(row, dict) or "id" not in row:
                log.warning("%s: skipping a malformed [[recordings]] entry", manifest)
                continue
            slug = str(row.get("slug") or row["id"])
            entries.append(
                RecordingEntry(
                    id=str(row["id"]),
                    slug=slug,
                    path=str(row.get("path") or f"recordings/{slug}"),
                    id_basis=str(row.get("id_basis", "footage")),
                    subject=row.get("subject"),
                    calibration=row.get("calibration"),
                    n_frames=row.get("n_frames"),
                    fps=row.get("fps"),
                    added_utc=str(row.get("added_utc", "")),
                    origin=dict(row.get("origin") or {}),
                )
            )
        return cls(
            root=root,
            name=str(head.get("name", root.name)),
            id=str(head.get("id", "")),
            description=str(head.get("description", "")),
            iteration=int(head.get("iteration", 0)),
            created_utc=str(head.get("created_utc", "")),
            format_version=version,
            skeleton_file=str(head.get("skeleton", SKELETON_FILENAME)),
            calibration=head.get("calibration"),
            recordings=entries,
        )

    @staticmethod
    def find(start: str | Path = ".") -> Path | None:
        """The nearest enclosing project directory, searching upward from ``start``.

        Lets ``deeperfly project status`` work from inside a project the way ``git``
        does, rather than requiring the root to be typed every time.
        """
        here = Path(start).resolve()
        for candidate in (here, *here.parents):
            if (candidate / PROJECT_FILENAME).exists():
                return candidate
        return None

    # -- persistence ----------------------------------------------------------

    def save(self) -> Path:
        """Write ``project.toml`` (overwriting). Returns the path written."""
        path = self.root / PROJECT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml())
        return path

    def to_toml(self) -> str:
        """The manifest as TOML text."""
        head: dict = {
            "format_version": self.format_version,
            "name": self.name,
            "id": self.id,
            "created_utc": self.created_utc or _now(),
            # Bumped by every merge/import, so a training set is attributable to a
            # project state rather than to "whatever the directory held that day".
            "iteration": self.iteration,
            "skeleton": self.skeleton_file,
        }
        if self.description:
            head["description"] = self.description
        if self.calibration is not None:
            head["calibration"] = self.calibration
        lines = [
            "# deeperfly project -- an INDEX of related recordings, not a copy of them.",
            "#",
            "# Each [[recordings]] entry points at a recording whose results.h5 and",
            "# labels.h5 stay where they already are (adopted by symlink). Nothing here",
            "# owns pixel data; deleting this file loses the index, not the labels.",
            "#",
            "#     deeperfly project status .        # what is labeled, and how much",
            "#     deeperfly gui .                   # open the project in the editor",
            "",
        ]
        lines += _toml.table_lines(["project"], head)
        for entry in self.recordings:
            lines += ["", "[[recordings]]"]
            table = entry.as_table()
            origin = table.pop("origin", None)
            lines += [f"{_toml.key(k)} = {_toml.value(v)}" for k, v in table.items()]
            if origin:
                lines.append("")
                # An inline table would need a writer feature; a sub-table under an
                # array-of-tables entry binds to that entry, which is what we want.
                lines += _toml.table_lines(["recordings", "origin"], origin)
        return "\n".join(lines) + "\n"

    # -- resolution -----------------------------------------------------------

    def skeleton_path(self) -> Path:
        return self.root / self.skeleton_file

    def skeleton(self):
        """The project's :class:`~deeperfly.skeleton.Skeleton`.

        Raises
        ------
        FileNotFoundError
            If the skeleton file is missing.
        """
        from .config import Config

        path = self.skeleton_path()
        if not path.exists():
            raise FileNotFoundError(f"the project's skeleton file {path} is missing")
        return Config.from_toml(path).skeleton()

    # -- config composition ---------------------------------------------------
    #
    # A project owns its skeleton, its rig topology and its algorithm deltas as three
    # separate files, and a run consumes ONE config. They are combined by *concatenating
    # fragments*, not by serializing a merged dict: each fragment declares different
    # top-level tables, so nothing is ever defined twice, no TOML writer is involved (so
    # nothing can be silently mis-serialized -- the detection plan's 132 inline tables are
    # the risk that would carry), and every comment survives byte-for-byte.
    #
    # Layering is therefore an AUTHORING convenience only. What a run receives, snapshots
    # and fingerprints is a single resolved text, exactly as before.

    def rig_path(self) -> Path:
        return self.root / RIG_FILENAME

    def profile_path(self, name: str | None = None) -> Path:
        return self.root / PROFILE_DIRNAME / (name or DEFAULT_PROFILE)

    def compose_config(self, *, profile: str | None = None, base=None) -> str:
        """The project's resolved run config, as TOML text.

        Assembled from, in order: the **skeleton**, the **rig** (``rig.toml``, when the
        project has one), the current **calibration** (as ``[calibration].path``), the
        **profile**'s deltas, and finally whatever tables ``base`` still supplies that none
        of those did -- the detection plan and visualization, which are open-ended and stay
        in the packaged config until a project has reason to override them.

        Parameters
        ----------
        profile
            Profile filename under ``profiles/``; defaults to the project's.
        base
            Config text (or path) to draw the remaining tables from. Defaults to the
            packaged config.

        Returns
        -------
        str
            One TOML document.

        Raises
        ------
        ValueError
            If two fragments would declare the same top-level table -- which TOML forbids
            and which would otherwise produce an invalid config only at parse time.
        """
        from . import _toml
        from .config import DEFAULT_CONFIG_PATH

        if base is None:
            base_text = DEFAULT_CONFIG_PATH.read_text()
        else:
            base_text = (
                Path(base).read_text()
                if not str(base).startswith("#") and Path(str(base)).exists()
                else str(base)
            )

        fragments: list[tuple[str, str]] = []
        skeleton = self.skeleton_path()
        if skeleton.exists():
            fragments.append(("skeleton", skeleton.read_text()))

        # Its own table, which is the whole reason it is one: as a `[cameras]` key it
        # overlapped the rig fragment's `[cameras.*]` sub-tables, and the guard below had
        # to carry an exception for the one legal overlap. Now the fragments are disjoint.
        if self.calibration is not None:
            fragments.append(
                (
                    "calibration",
                    "# The project's current calibration (deeperfly project config).\n"
                    "[calibration]\n"
                    f"path = {_toml.value(str((self.root / self.calibration).resolve()))}\n",
                )
            )

        rig = self.rig_path()
        rig_text = (
            rig.read_text()
            if rig.exists()
            else _toml.extract_tables(base_text, RIG_TABLES)
        )
        if rig_text.strip():
            fragments.append(("rig", rig_text))

        prof = self.profile_path(profile)
        if prof.exists():
            fragments.append(("profile", prof.read_text()))

        # Whatever no fragment claimed: the detection plan, visualization, and any table a
        # project has not taken over. Taken from the base so a project never has to restate
        # 132 detector channel mappings to change a triangulation knob.
        claimed: set[str] = set()
        for _, text in fragments:
            claimed.update(_toml.top_level_tables(text))
        remaining = [
            name for name in _toml.top_level_tables(base_text) if name not in claimed
        ]
        if remaining:
            fragments.append(("base", _toml.extract_tables(base_text, remaining)))

        # Guard the one way concatenation can go wrong. No exception any more: every
        # fragment owns whole top-level tables, so any overlap is a mistake.
        seen: dict[str, str] = {}
        for source, text in fragments:
            for name in _toml.top_level_tables(text):
                if name in seen:
                    raise ValueError(
                        f"both the {seen[name]} and {source} fragments declare "
                        f"[{name}]; a project's files must own disjoint tables. Remove "
                        f"[{name}] from one of them"
                    )
                seen[name] = source

        header = [
            "# GENERATED by 'deeperfly project config' -- do not edit.",
            f"# Composed from project {self.name!r} ({self.root}):",
        ]
        header += [f"#   {source}" for source, _ in fragments]
        header.append("")
        return "\n".join(header) + "\n\n".join(
            text.rstrip() + "\n" for _, text in fragments
        )

    def profile_values(self, profile: str | None = None) -> dict:
        """The profile's raw overrides, as a nested mapping (``{}`` when it has none)."""
        path = self.profile_path(profile)
        return tomllib.loads(path.read_text()) if path.exists() else {}

    def set_profile_key(
        self, section: str, key: str, value, *, profile: str | None = None
    ) -> Path:
        """Set (or clear) one key in the project's profile, and rewrite it.

        A **rewrite** rather than an append, unlike ``deeperfly config set``: a profile holds
        only knob tables -- scalars, flat arrays and nested tables -- every one of which the
        writer in :mod:`deeperfly._toml` handles exactly. There are no arrays-of-tables and
        no inline tables in a profile, which is precisely why rewriting it is safe here and
        is not safe for a full config.

        ``value is None`` removes the key, so the GUI's "reset to default" is the same call
        -- and the resulting file genuinely no longer mentions it, rather than restating the
        default as though someone had chosen it.

        Parameters
        ----------
        section, key
            The config table and key, e.g. ``("triangulation", "method")``.
        value
            The new value, or ``None`` to remove the override.
        profile
            Profile filename; defaults to the project's.

        Returns
        -------
        Path
            The profile written.

        Raises
        ------
        ValueError
            If the result would not load through ``Config``'s own strict validator -- the
            same one a run uses, so the GUI cannot store a key a run would reject.
        """
        from .config import Config
        from .config_schema import SECTIONS

        data = self.profile_values(profile)
        table = dict(data.get(section) or {})
        if value is None:
            table.pop(key, None)
        else:
            table[key] = value
        if table:
            data[section] = table
        else:
            data.pop(section, None)

        # Validate before writing, through the accessor for this section rather than a
        # bespoke check, so a rejected key reads identically however it arrived.
        if section in SECTIONS:
            probe = Config.from_dict({section: data.get(section, {})})
            accessor = {"pictorial_structures": "pictorial"}.get(section, section)
            getattr(probe, accessor)

        lines = [
            "# Algorithm settings for this project -- ONLY the keys that differ from the",
            "# packaged defaults. Managed by 'deeperfly config' and the editor's Settings",
            '# panel; an empty file means "use the defaults for everything".',
        ]
        for name in sorted(data):
            body = data[name]
            if not isinstance(body, dict) or not body:
                continue
            lines.append("")
            lines += _toml.table_lines([name], body)
        out = self.profile_path(profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out

    def write_rig(self, base=None) -> Path:
        """Extract the rig tables out of a config into the project's ``rig.toml``.

        The rig -- footage sources, camera topology, per-camera preprocessing -- is a
        property of the *setup*, shared by every recording on it. Lifting it into the
        project is what stops each recording carrying its own copy.
        """
        from . import _toml
        from .config import DEFAULT_CONFIG_PATH

        text = Path(base).read_text() if base else DEFAULT_CONFIG_PATH.read_text()
        fragment = _toml.extract_tables(text, RIG_TABLES)
        if not fragment.strip():
            raise ValueError(
                f"no rig tables ({list(RIG_TABLES)}) in that config -- nothing to extract"
            )
        out = self.rig_path()
        out.write_text(
            "# This project's camera rig: the cameras, their footage patterns and their\n"
            "# geometry. Shared by every recording on this setup.\n"
            "#\n"
            "# The current calibration is NOT here -- the project injects it as its own\n"
            "# [calibration] table when it composes the run config.\n\n" + fragment
        )
        return out

    def calibration_path(self) -> Path | None:
        """The project's current calibration file, or ``None`` (uncalibrated)."""
        return None if self.calibration is None else self.root / self.calibration

    def recording_dir(self, entry: RecordingEntry) -> Path:
        return self.root / entry.path

    def outputs_dir(self, entry: RecordingEntry) -> Path:
        """Where this recording's ``results.h5`` / ``labels.h5`` live.

        A symlink for an adopted recording, a real directory for one created here --
        resolved identically either way, which is the point of adopting by reference.
        """
        return self.recording_dir(entry) / OUTPUTS_DIRNAME

    def results_path(self, entry: RecordingEntry) -> Path:
        return self.outputs_dir(entry) / "results.h5"

    def labels_path(self, entry: RecordingEntry) -> Path:
        return self.outputs_dir(entry) / "labels.h5"

    def recording(self, key: str) -> RecordingEntry:
        """One recording by id, slug, or unambiguous id prefix.

        Raises
        ------
        KeyError
            If nothing matches, or a prefix is ambiguous.
        """
        for entry in self.recordings:
            if key in (entry.id, entry.slug):
                return entry
        hits = [e for e in self.recordings if e.id.startswith(key)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise KeyError(f"{key!r} is ambiguous: {', '.join(e.slug for e in hits)}")
        raise KeyError(f"no recording {key!r} in project {self.name!r}")

    # -- mutation -------------------------------------------------------------

    def add_recording(
        self,
        source: str | Path,
        *,
        link: bool = True,
        slug: str | None = None,
        subject: str | None = None,
        config=None,
        rec_id: str | None = None,
    ) -> RecordingEntry:
        """Adopt a recording into the project.

        ``source`` may be a recording directory (holding per-camera footage and possibly
        a ``deeperfly_outputs/``), an outputs directory, or a ``results.h5``. All three
        are things a user already has a path to.

        Nothing is copied by default: ``recordings/<slug>/deeperfly_outputs`` becomes a
        **symlink** to the source's outputs, so the file the editor writes is the file
        the training set reads and no hand label is duplicated. ``link=False`` copies
        instead.

        Parameters
        ----------
        source
            The recording, its outputs directory, or its ``results.h5``.
        link
            Symlink the outputs directory (default) rather than copy it.
        slug
            Directory/display name; defaults to the recording directory's name.
        subject
            Animal identifier. Read from an existing ``results.h5`` when omitted.
        config
            Optional config supplying per-source footage globs (see
            :func:`discover_footage`).
        rec_id
            Override the content-derived id (``id_basis`` becomes ``"manual"``). For
            reconciling a recording adopted earlier under a different basis.

        Returns
        -------
        RecordingEntry
            The new entry (already saved into the manifest).

        Raises
        ------
        FileNotFoundError
            If ``source`` does not exist.
        ValueError
            If the recording cannot be identified, or its slug collides with a
            *different* recording already in the project.
        """
        rec_root, outputs = _split_source(Path(source))
        results = outputs / "results.h5" if outputs is not None else None
        store = None
        result_footage: dict | None = None
        n_frames = fps = None
        if results is not None and results.exists():
            from .results import StageStore

            store = StageStore(results)
            try:
                result_footage = store.read_footage()
                n_frames, fps = _frames_and_fps(store)
                if subject is None:
                    subject = store.read_animal()[1]
            except ValueError as exc:
                # A results.h5 this build refuses (e.g. written by a newer deeperfly) must
                # not block adoption: the reason to adopt a recording is usually its
                # labels.h5, which is a separate file and separately versioned. The
                # metadata this would have supplied is optional -- an unknown frame count
                # prints as '?' and is backfilled when the editor opens the footage.
                log.warning(
                    "%s: %s -- adopting anyway, without its recorded metadata",
                    results,
                    exc,
                )

        footage = discover_footage(rec_root, config) if rec_root.is_dir() else {}
        if rec_id is None:
            fingerprint, basis = recording_fingerprint(
                footage, result_footage=result_footage, n_frames=n_frames
            )
            rec_id, basis = recording_id(fingerprint), basis
        else:
            basis = "manual"

        existing = next((e for e in self.recordings if e.id == rec_id), None)
        if existing is not None:
            log.info(
                "%s is already in this project as %r (same content id %s); "
                "leaving it alone",
                rec_root,
                existing.slug,
                rec_id,
            )
            self._warn_unseen_labels(existing, outputs)
            return existing

        slug = _unique_slug(self, slug or _slugify(rec_root.name), rec_id)
        dest = self.root / "recordings" / slug
        dest.mkdir(parents=True, exist_ok=True)
        if outputs is not None and outputs.is_dir():
            _attach_outputs(dest / OUTPUTS_DIRNAME, outputs, link=link)
        else:
            # A recording with no outputs yet -- the from-scratch case -- still needs
            # somewhere for them, because the very first thing that happens to it is a
            # labels.h5 being written. Creating it here means no writer has to guess
            # whether the directory exists.
            (dest / OUTPUTS_DIRNAME).mkdir(exist_ok=True)

        entry = RecordingEntry(
            id=rec_id,
            slug=slug,
            path=f"recordings/{slug}",
            id_basis=basis,
            subject=subject,
            n_frames=n_frames,
            fps=fps,
            added_utc=_now(),
            origin={"kind": "adopted", "from": str(rec_root.resolve())},
        )
        _write_recording_file(dest / RECORDING_FILENAME, entry, footage, result_footage)
        self.recordings.append(entry)
        self.save()
        return entry

    def _warn_unseen_labels(
        self, existing: RecordingEntry, outputs: Path | None
    ) -> None:
        """Warn when a de-duplicated source carries labels the indexed entry cannot see.

        One recording is one entry -- that is what content-based identity buys, and it is
        what stops a backup copy from double-counting. But a recording can genuinely have
        *several* label sets: an earlier round, a superseded pass, a second annotator's
        copy under a different directory. Adopting the second one is then a no-op, and
        its labels would silently read as zero -- which looks exactly like "the labels
        are gone".

        So it is reported, with both counts and both paths, rather than left to be
        discovered by a training set that came out smaller than expected -- and the fix it
        names is ``deeperfly project import-outputs``, which merges the two by name instead
        of making the operator pick one to abandon.

        Compares label *sets*, not totals. Comparing totals meant a second set that was
        merely **smaller** returned early and said nothing -- and two annotators working on
        disjoint frames is exactly that case, with every one of the smaller set's cells
        genuinely unseen.
        """
        if outputs is None:
            return
        incoming_path = outputs / "labels.h5"
        incoming = label_stats(incoming_path)
        if not incoming["gt_points"]:
            return
        indexed_path = self.labels_path(existing)
        indexed = label_stats(indexed_path)
        unseen = len(_gt_cells(incoming_path) - _gt_cells(indexed_path))
        if not unseen:
            return
        log.warning(
            "%s carries %d ground-truth cell(s) this project will NOT count (it holds %d "
            "in %d frame(s); %r is already indexed for the same footage and points at %s, "
            "which holds %d). One recording is one entry, so only the indexed copy is "
            "read -- merge them with 'deeperfly project import-outputs %s %s'",
            incoming_path,
            unseen,
            incoming["gt_points"],
            incoming["labeled_frames"],
            existing.slug,
            indexed_path,
            indexed["gt_points"],
            self.root,
            outputs,
        )

    def update_recording(self, key: str, **fields) -> RecordingEntry:
        """Replace fields on one indexed recording and persist the manifest.

        The index caches ``n_frames`` / ``fps`` so a listing needs no file opens -- which
        means they can be *unknown* for a recording adopted before it was ever run. The
        editor learns the real frame count when it opens the footage, and backfilling it
        here is what stops ``project status`` reporting ``?`` forever.

        Parameters
        ----------
        key
            Slug, id, or unambiguous id prefix.
        **fields
            :class:`RecordingEntry` fields to replace.

        Returns
        -------
        RecordingEntry
            The updated entry.
        """
        from dataclasses import replace

        entry = self.recording(key)
        updated = replace(entry, **fields)
        self.recordings = [updated if e.id == entry.id else e for e in self.recordings]
        self.save()
        return updated

    def remove_recording(self, key: str, *, delete: bool = False) -> RecordingEntry:
        """Drop a recording from the index.

        By default only the index entry and the project's own directory for it go away
        -- the adopted outputs are left untouched, because a symlinked project must not
        be able to delete the originals by accident. ``delete=True`` removes the
        project's directory too, which for a linked recording removes only the link.
        """
        entry = self.recording(key)
        if delete:
            import shutil

            target = self.recording_dir(entry)
            link = target / OUTPUTS_DIRNAME
            if link.is_symlink():
                link.unlink()
            if target.exists():
                shutil.rmtree(target)
        self.recordings = [e for e in self.recordings if e.id != entry.id]
        self.save()
        return entry

    # -- reporting ------------------------------------------------------------

    def status(self) -> list[dict]:
        """Per-recording state: what exists on disk and how much is labeled.

        One row per entry, in index order, each with the entry plus ``has_results``,
        ``has_labels``, ``outputs`` and the :func:`label_stats` counts. Never raises on
        a broken recording -- a missing or corrupt file is *reported*, since a status
        listing that dies on one bad row is useless exactly when it is needed.
        """
        rows = []
        for entry in self.recordings:
            outputs = self.outputs_dir(entry)
            results = self.results_path(entry)
            labels = self.labels_path(entry)
            rows.append(
                {
                    "entry": entry,
                    "outputs": outputs,
                    "linked": (outputs.is_symlink()),
                    "outputs_missing": not outputs.exists(),
                    "has_results": results.exists(),
                    "has_labels": labels.exists(),
                    "calibration": entry.calibration or self.calibration,
                    **label_stats(labels),
                }
            )
        return rows

    def totals(self, rows: list[dict] | None = None) -> dict:
        """Project-wide label totals (the numbers a training round is judged on)."""
        rows = self.status() if rows is None else rows
        keys = (
            "gt_points",
            "occluded",
            "labeled_frames",
            "reviewed_frames",
        )
        out = {k: int(sum(r[k] for r in rows)) for k in keys}
        out["recordings"] = len(rows)
        out["with_labels"] = sum(1 for r in rows if r["gt_points"])
        return out

    def bump_iteration(self) -> int:
        """Increment and persist the iteration counter (a merge/import happened)."""
        self.iteration += 1
        self.save()
        return self.iteration


# -- helpers -------------------------------------------------------------------


def _split_source(source: Path) -> tuple[Path, Path | None]:
    """``(recording dir, outputs dir | None)`` from whatever path the user gave.

    Accepts a recording directory, an outputs directory, or a ``results.h5``. The
    outputs directory is ``None`` for a recording that has never been run -- the
    from-scratch case, which must be adoptable.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    if source.is_file():
        return source.parent.parent, source.parent
    if (source / "results.h5").exists():
        return source.parent, source
    nested = source / OUTPUTS_DIRNAME
    if nested.is_dir():
        return source, nested
    return source, None


def _frames_and_fps(store) -> tuple[int | None, float | None]:
    """``(n_frames, fps)`` from a result store, without loading the arrays.

    Both are cached in the index so a listing needs no file opens, and both are
    best-effort: an unreadable result still adopts (its labels may be the reason it is
    being adopted at all).
    """
    try:
        pose2d = store.read_pose2d()
        n_frames = int(pose2d[0].shape[1]) if pose2d is not None else None
    except Exception:
        n_frames = None
    fps = None
    try:
        import json

        import h5py

        with h5py.File(store.path, "r") as f:
            meta = json.loads(f.attrs.get("meta", "{}"))
        raw = meta.get("fps")
        fps = float(raw) if raw is not None else None
    except Exception:
        fps = None
    return n_frames, fps


def _unique_slug(project: Project, slug: str, rec_id: str) -> str:
    """A slug not already taken by a *different* recording in ``project``."""
    if not _SLUG_OK.match(slug):
        slug = _slugify(slug)
    taken = {e.slug for e in project.recordings}
    if slug not in taken:
        return slug
    # Disambiguate with the content id rather than a counter: the suffix then means
    # something, and re-adopting the same recording lands on the same name.
    candidate = f"{slug}-{rec_id.removeprefix('rec_')[:6]}"
    n = 2
    while candidate in taken:
        candidate = f"{slug}-{rec_id.removeprefix('rec_')[:6]}-{n}"
        n += 1
    return candidate


def _attach_outputs(dest: Path, outputs: Path, *, link: bool) -> None:
    """Point ``dest`` at an existing outputs directory (symlink, else copy)."""
    import shutil

    outputs = outputs.resolve()
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and dest.resolve() == outputs:
            return
        raise FileExistsError(
            f"{dest} already exists and does not point at {outputs}; remove it first"
        )
    if not link:
        shutil.copytree(outputs, dest)
        return
    try:
        dest.symlink_to(outputs, target_is_directory=True)
    except OSError as exc:  # pragma: no cover -- platforms without symlink permission
        log.warning(
            "could not symlink %s -> %s (%s); copying instead. The copy is a SNAPSHOT: "
            "labels authored in the original will not appear here",
            dest,
            outputs,
            exc,
        )
        shutil.copytree(outputs, dest)


def _write_recording_file(
    path: Path, entry: RecordingEntry, footage: dict, result_footage: dict | None
) -> None:
    """Write ``recording.toml``: this recording's footage pointers.

    Footage lives here rather than only in the project manifest because a recording that
    has never been run has no ``results.h5`` to record it in -- and that recording (seven
    videos and nothing else) is exactly the from-scratch starting point.
    """
    table: dict = {
        "id": entry.id,
        "slug": entry.slug,
        "id_basis": entry.id_basis,
        "added_utc": entry.added_utc,
    }
    if entry.subject:
        table["subject"] = entry.subject
    if entry.n_frames is not None:
        table["n_frames"] = entry.n_frames
    if entry.fps is not None:
        table["fps"] = entry.fps

    lines = [
        "# One recording's footage pointers. Written by 'deeperfly project add'.",
        "# The results.h5 / labels.h5 for this recording live in ./deeperfly_outputs,",
        "# which is a symlink to wherever they already were.",
        "",
    ]
    lines += _toml.table_lines(["recording"], table)
    if footage:
        # The canonical pointer (deeperfly.footage): every flavor, always, anchored at this
        # file's own directory. `rel` is what the plan specified and the implementation had
        # dropped -- and it is what survives the recording being moved with its project,
        # where an absolute path does not. `bytes` makes the content id re-derivable.
        from .footage import write_pointer

        lines += ["", "# camera -> the footage files this recording was adopted from."]
        for camera in sorted(footage):
            pointer = write_pointer(footage[camera], path.parent)
            lines += ["", f"[recording.footage.{_toml.key(camera)}]"]
            lines += [f"{k} = {_toml.value(v)}" for k, v in pointer.items()]
    elif result_footage:
        # Footage that no longer resolves: only the names results.h5 knows. Writing a stale
        # absolute path would look like a location and be a fiction.
        lines += ["", "# The footage did not resolve; only its names are known."]
        for camera in sorted(result_footage):
            lines += ["", f"[recording.footage.{_toml.key(camera)}]"]
            lines.append(f"names = {_toml.value(_basenames(result_footage[camera]))}")
    path.write_text("\n".join(lines) + "\n")


def _skeleton_text(skeleton: str) -> str:
    """The text for a new project's ``skeleton.toml``.

    A preset name, or a path to a TOML file holding a ``[skeleton]`` table. A packaged
    preset is copied *with its comments*, so the prose explaining the point ordering and
    the palette convention travels with the project instead of being re-derived.

    The preset is read from the packaged skeleton FILE rather than out of the packaged
    run config. Those are two different questions -- "what is a fly38" and "what does a
    fresh config detect" -- and tying them meant that changing the shipped detector
    changed what a new project's skeleton was.

    Raises
    ------
    ValueError
        If ``skeleton`` is neither a known preset nor a readable file with a
        ``[skeleton]`` table.
    """
    from .config import skeleton_presets

    if skeleton == "blank":
        return _BLANK_SKELETON
    presets = skeleton_presets()
    if skeleton in presets:
        return _toml.extract_section(presets[skeleton].read_text(), "skeleton")
    path = Path(skeleton)
    if not path.exists():
        raise ValueError(
            f"unknown skeleton {skeleton!r}: expected one of "
            f"{list(skeleton_preset_names())} or a path to a TOML file with a "
            "[skeleton] table"
        )
    text = path.read_text()
    parsed = tomllib.loads(text).get("skeleton")
    if parsed is None:
        raise ValueError(f"{path} has no [skeleton] table")
    # A config may only INCLUDE its skeleton (`[skeleton] include = "fly38"`). Seeding a
    # project from one has to write the skeleton out, not copy the reference: a project
    # is a long-lived record of what it tracks, and a reference would let a package
    # upgrade change, retroactively, what its stored labels mean.
    if "points" not in parsed and parsed.get("include") in presets:
        text = presets[parsed["include"]].read_text()
    # Keep only the skeleton section: a whole run config would drag a rig and a
    # detection plan into the project's skeleton file, where a later reader would have
    # no way to know which of the two definitions was authoritative.
    return _toml.extract_section(text, "skeleton")
