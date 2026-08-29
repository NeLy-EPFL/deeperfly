"""The editor server's own state: everything ``create_app`` used to close over.

One :class:`EditorApp` per running server. It holds the session being edited, the
recordings held open behind it, the live browser sockets and the two timers that decide
when a closed tab means "stop the server" -- plus the small number of methods that read
or mutate exactly those. The routes in :mod:`deeperfly.gui.routes` reach it through
:func:`get_editor`, which is the only reason they can be module-level functions instead
of closures.

The split matters for one reason beyond size: this state has real invariants (a session
and its cache token are minted together and must never be paired up from different
recordings; a session with unsaved labels is never evicted), and they are easier to keep
true in one named place than spread across fifty handlers that all closed over the same
variables.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fastapi import HTTPException, WebSocket
from starlette.requests import HTTPConnection

if TYPE_CHECKING:
    from ..project.jobs import JobQueue

from . import payloads
from .session import Session

log = logging.getLogger("deeperfly")

_WEB_DIR = Path(__file__).parent / "web"


def _asset_version() -> str:
    """A short content hash of the entry assets, stamped into the URLs the page loads.

    ``index.html`` references ``app.js`` / ``styles.css`` as ``...?v=<hash>``. The page
    itself is served ``no-cache`` (revalidated every load), so a fresh load always
    carries the current hash and the browser fetches the *matching* JS/CSS. Without
    this, a browser that serves a heuristically-cached ``app.js`` against freshly
    changed HTML gets a half-broken editor -- the stale JS wires to DOM ids the new
    HTML no longer has. Computed per request (cheap) so in-place edits take effect on
    the next reload with no server restart."""
    h = hashlib.sha1()
    for name in ("app.js", "styles.css", "baPanel.js"):
        p = _WEB_DIR / "static" / name
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


#: How long the exit-on-last-tab timer stays disarmed around a recording switch. Every
#: browser reloads at once, so every socket drops at once, and ``exit_on_disconnect``
#: would read that as "the last tab closed". Generous on purpose: the cost of being
#: wrong is the server stopping in the middle of a switch, and the cost of being
#: over-generous is a closed editor lingering for half a minute.
_SWITCH_GRACE = 30.0
#: Rebinding the rig drops the cache of derived 3D, which can hold a hand-placed depth
#: the 2D labels cannot reproduce -- so, unlike a recording switch, selecting a
#: calibration still has to refuse while there are unsaved labels.
_UNSAVED_RIG_MSG = (
    "this recording has unsaved labels; save them first (POST /api/save), or pass "
    '"discard": true to abandon them'
)
#: How many *clean* recordings to keep open behind the active one. A retained session
#: spares the operator re-reading every camera's video header, ``results.h5`` and
#: ``labels.h5`` (a network share, in this lab) when they switch back -- but it also
#: holds that recording's arrays, so the clean ones are an LRU cache and get pruned. A
#: session with **unsaved labels is never pruned**: it is the only copy of hand work.
_SESSION_KEEP = 3


def _live_label_stats(session: Session) -> dict:
    """The label counts of an *open* session, as :func:`~deeperfly.project.label_stats`
    would report them once saved.

    A recording held open with unsaved labels would otherwise be listed from its
    on-disk sidecar -- "unlabeled" next to an hour of work that has not reached disk
    yet. The four counts must mean exactly what the disk ones mean, so each is the
    in-memory twin of the group ``label_stats`` counts rows of: the vetoed masks
    (:attr:`~deeperfly.labels.store.Labels.has_gt`,
    :attr:`~deeperfly.labels.store.Labels.occluded_effective`), because an absent point's
    labels are quarantined out of ``gt/index`` on the way out.
    """
    labels = session.state.labels
    has_gt = np.asarray(labels.has_gt)
    return {
        "gt_points": int(has_gt.sum()),
        "occluded": int(np.asarray(labels.occluded_effective).sum()),
        # Frames carrying human work, in any view -- the frame axis is the middle one.
        "labeled_frames": int(has_gt.any(axis=(0, 2)).sum()),
        "reviewed_frames": int(np.asarray(labels.reviewed).sum()),
    }


def _open_for_switch(root: Path, slug: str) -> tuple[Session, str]:
    """Open ``slug`` from the project at ``root``, with the cache token for its pictures.

    Runs in a worker thread (see ``POST /api/recordings/open``): opening a recording
    reads every camera's video header, loads ``results.h5`` + ``labels.h5`` and may build
    the IK model, and :func:`_session_version` then ``stat``s all that footage again --
    on a network share, for this lab. None of that belongs on the event loop.

    Both values are produced by this one call so a session and its token can never be
    minted from *different* recordings, which is precisely the mix-up the token exists to
    prevent.
    """
    from . import open_target  # deferred: deeperfly.gui imports this module

    session = open_target(root, recording=slug)
    return session, payloads._session_version(session)


def _recording_rows(
    project, active: str | None, open_sessions: dict[str, Session] | None = None
) -> list[dict]:
    """One JSON-safe row per recording of ``project``, in index order.

    :meth:`~deeperfly.project.Project.status` rows carry a ``RecordingEntry`` dataclass
    and ``Path``s, which FastAPI's encoder cannot serialize. Naming the fields here makes
    the payload a contract rather than a dump of whatever ``status()`` happens to return.

    ``open_sessions`` (``slug -> Session``) are the recordings the editor is holding in
    memory. Their counts come from the live session and their ``dirty`` flag says whether
    it has unsaved labels -- so the list reports what the operator has *done*, not what
    has reached disk. Without that, the recording they just labeled 40 frames in and
    switched away from would sit in the list saying "unlabeled".
    """
    rows = []
    for row in project.status():
        entry = row["entry"]
        live = (open_sessions or {}).get(entry.slug)
        counts = {
            "gt_points": int(row["gt_points"]),
            "occluded": int(row["occluded"]),
            "labeled_frames": int(row["labeled_frames"]),
            "reviewed_frames": int(row["reviewed_frames"]),
        }
        if live is not None:
            counts = _live_label_stats(live)
        rows.append(
            {
                "slug": entry.slug,
                "id": entry.id,
                "subject": entry.subject,
                "n_frames": entry.n_frames,
                "fps": entry.fps,
                "active": entry.slug == active,
                # Held in memory by this editor, and whether it has unsaved labels. The
                # front-end marks the dirty ones -- the operator has to be able to see
                # where their unsaved work is without opening each recording to check.
                "open": live is not None,
                "dirty": bool(live is not None and live.state.dirty),
                "has_results": bool(row["has_results"]),
                "has_labels": bool(row["has_labels"]),
                "outputs_missing": bool(row["outputs_missing"]),
                **counts,
            }
        )
    return rows


@dataclass
class EditorApp:
    """The running editor's mutable state, shared by every route.

    Constructed once by :func:`~deeperfly.gui.server.create_app` and reachable from any
    handler as ``request.app.state.editor`` (see :func:`get_editor`).
    """

    #: The recording currently being edited. Rebound by a recording switch.
    session: Session
    #: Stops the running server: `POST /api/shutdown` (the Close button) and, when
    #: `exit_on_disconnect` is set, `disconnect_grace` seconds after the last browser
    #: drops its `/ws` socket. `deeperfly.gui.serve` passes a callback that flips
    #: uvicorn's `should_exit`; tests pass a plain stub.
    on_shutdown: Callable[[], None] | None = None
    exit_on_disconnect: bool = False
    disconnect_grace: float = 0.5
    #: A `deeperfly.project.jobs.JobQueue` the browser may submit work to, or `None`
    #: -- what a bare `results.h5` session gets, which leaves `/api/jobs` reporting
    #: `enabled: false` rather than absent, so the front-end can say *why* the buttons
    #: are missing instead of just not showing them.
    jobs: JobQueue | None = None

    #: Serializes edits against the shared session.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: The cache token stamped into every picture URL. Recomputed only when the open
    #: recording changes: it stats the footage (on a network share, for this lab), and
    #: a single value keeps the token the page stamps into its URLs identical to the
    #: one the frame handler validates against.
    cache_v: str = field(init=False)
    #: Every recording opened this run: `slug -> (session, its cache token)`,
    #: least-recently used first, the open one always last. This is what makes unsaved
    #: labels project-wide: the swap used to DROP the outgoing session -- labels and
    #: undo history together -- so it had to be refused while dirty, and the operator
    #: was prompted on every switch. Keeping the session means a switch loses nothing,
    #: costs no I/O on the way back, and the only question left is at close time. The
    #: token rides along in the same tuple so a session and its cache stamp can never
    #: be paired up wrongly (see :func:`_open_for_switch`).
    #:
    #: Bounded, but only over the sessions it is safe to bound: :meth:`prune_sessions`
    #: drops least-recently-used CLEAN ones past :data:`_SESSION_KEEP` and never touches
    #: a session with unsaved labels. A bare results.h5 has no slug and no project to
    #: switch within, so the registry stays empty and every path falls back to
    #: :attr:`session`.
    opened: OrderedDict[str, tuple[Session, str]] = field(init=False)

    #: Open `/ws` sockets (one per browser tab), and the single "writer" allowed to edit.
    sockets: set[WebSocket] = field(default_factory=set)
    writer: WebSocket | None = None
    #: Fires `on_shutdown` once the last socket has stayed closed for the grace period.
    pending_exit: asyncio.TimerHandle | None = None
    #: Live while a recording switch is in flight. Every browser reloads at once, so
    #: every socket drops at once, and `exit_on_disconnect` must not read that as the
    #: last tab closing. `switch_busy` additionally makes two overlapping switches a
    #: 409 rather than a race between two half-built sessions.
    switching: asyncio.TimerHandle | None = None
    switch_busy: bool = False

    #: Already-JPEG-encoded frames, byte-budgeted. Re-encoding dominates a frame
    #: request and every panel of the page asks for the same frame. Unlike
    #: :attr:`mesh_cache` a recording switch has nothing to clear here: the entries a
    #: switch would drop are keyed by a token that will never be asked for again.
    encoded_frames: payloads._EncodedFrames = field(
        default_factory=payloads._EncodedFrames
    )
    #: Rendered mesh overlays, keyed by (token, camera, frame). Cleared on a switch.
    mesh_cache: dict[tuple[str, str, int], bytes] = field(default_factory=dict)
    #: The one bundle-adjustment run this server will do at a time, and its guard.
    #: There is one per SERVER, not per session, so the tab reporting it survives a
    #: recording switch.
    ba_run: dict = field(default_factory=lambda: {"state": "idle"})
    ba_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.cache_v = payloads._session_version(self.session)
        self.opened = OrderedDict()
        if self.session.recording_slug is not None:
            self.opened[self.session.recording_slug] = (self.session, self.cache_v)

    # -- the open-recording registry ----------------------------------------

    def dirty_slugs(self) -> list[str]:
        """Every recording of this project holding unsaved labels, oldest use first.

        Empty for a bare ``results.h5`` -- it has no slug to name and no project to be
        dirty *within*, so ``dirty`` on the meta payload is the whole story there.
        """
        return [slug for slug, (sess, _) in self.opened.items() if sess.state.dirty]

    def project_dirty(self) -> bool:
        """Whether anything anywhere is unsaved -- the one flag the close prompt reads."""
        return bool(self.session.state.dirty or self.dirty_slugs())

    def prune_sessions(self) -> None:
        """Forget the least-recently-used *clean* recordings past :data:`_SESSION_KEEP`.

        The registry exists to make switching free; it must not become a way to run out
        of memory on a 26-recording project. A clean session can always be rebuilt from
        disk, so it is a cache. A dirty one cannot -- it is the operator's only copy --
        so it is skipped here however old it is, which is the whole point of keeping it.

        The evicted session is *dropped*, not closed: ``FrameSource.close()`` clears the
        decoded-frame cache that a threadpool frame handler pops from outside its ``try``,
        and a handler that snapshotted this session just before the swap may still be in
        it. Nothing leaks -- the readers hold no OS handle, and the last reference going
        away reclaims the arrays, the cache and the undo stacks.
        """
        clean = [
            slug
            for slug, (sess, _) in self.opened.items()
            if not sess.state.dirty and slug != self.session.recording_slug
        ]
        for slug in clean[: max(0, len(clean) - _SESSION_KEEP)]:
            self.opened.pop(slug, None)
            log.debug("dropped the cached session for %s", slug)

    def project(self):
        """The project this session belongs to.

        Raises 409 for a bare ``results.h5``: it has no project, so there is nothing to
        read settings out of or write them back to.
        """
        if self.session.project_root is None:
            raise HTTPException(
                409,
                "this session was opened on a bare results.h5; open a project to change "
                "its settings from the editor",
            )
        from ..project import Project

        return Project.load(self.session.project_root)

    # -- the browsers --------------------------------------------------------

    def role_msg(self, ws: WebSocket) -> dict:
        """The role handshake for a browser: may it edit (writer) or is it read-only?"""
        return {
            "type": "role",
            "role": "writer" if ws is self.writer else "reader",
            "clients": len(self.sockets),
        }

    async def broadcast(self, payload: dict) -> None:
        """Push one message to every open browser, dropping the sockets that have gone.

        Iterates a *copy*: the set is mutated by connects and disconnects, and a send to
        a socket the browser has already closed must not stop the live ones receiving it.
        """
        for cand in list(self.sockets):
            try:
                await cand.send_json(payload)
            except Exception:  # pragma: no cover -- the socket is on its way out
                self.sockets.discard(cand)

    # -- the exit timer ------------------------------------------------------

    def begin_switch(self) -> None:
        """Open the window in which a dropped socket means "reloading", not "closed"."""
        if self.pending_exit is not None:
            self.pending_exit.cancel()
            self.pending_exit = None
        if self.switching is not None:
            self.switching.cancel()
        self.switching = asyncio.get_running_loop().call_later(
            _SWITCH_GRACE, self.end_switch
        )

    def end_switch(self) -> None:
        """The reload window closed. Re-arm the exit timer if no browser came back.

        Without this, closing the last tab shortly after a switch would leave the server
        running forever -- the switch would have disabled the only thing that stops it.
        """
        self.switching = None
        if (
            self.exit_on_disconnect
            and not self.sockets
            and self.on_shutdown is not None
        ):
            log.info("no browser returned after the recording switch; stopping")
            self.pending_exit = asyncio.get_running_loop().call_later(
                self.disconnect_grace, self.on_shutdown
            )

    def arm_exit(self) -> None:
        """The last browser went away: stop the server after the grace period.

        A no-op during a recording switch -- every tab drops its socket to reload, and
        that must not read as the editor being closed.
        """
        if self.sockets or self.switching is not None:
            return
        if not self.exit_on_disconnect or self.on_shutdown is None:
            return
        self.pending_exit = asyncio.get_running_loop().call_later(
            self.disconnect_grace, self.on_shutdown
        )


def get_editor(conn: HTTPConnection) -> EditorApp:
    """The running :class:`EditorApp`, as a FastAPI dependency.

    Annotated ``HTTPConnection`` -- the base of both ``Request`` and ``WebSocket`` --
    rather than ``Request``, because ``/ws`` needs the same dependency and FastAPI hands
    a websocket handler a ``WebSocket``. With ``Request`` the socket route raises a bare
    500 at connect time, which reads as a server bug rather than a wiring one.
    """
    return conn.app.state.editor
