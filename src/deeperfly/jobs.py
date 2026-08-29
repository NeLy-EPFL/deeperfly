"""Background jobs: running the pipeline from the editor without blocking it.

The GUI has to be able to *do* things -- detect, calibrate, triangulate, render, export --
not merely display what a terminal already produced. Three properties decide whether that is
safe, and they are why this is a subprocess queue rather than a thread pool:

**Isolation.** A job runs as ``python -m deeperfly <argv>`` in its own process. A CUDA OOM,
a segfault in a native decoder, or a stray ``SystemExit`` then kills the job and nothing
else; in-process it would take the editor -- and any unsaved labels -- with it.

**Serialization.** One worker, FIFO. Two detections on one GPU do not go faster, and two
stages writing one ``results.h5`` corrupt it. The queue is the lock.

**Legibility.** Every job *is* a CLI invocation, stored verbatim and shown to the operator.
So the GUI teaches the CLI rather than hiding it, a failed job is reproducible by
copy-paste, and there is no second code path that only the GUI can reach -- which is the
usual way a GUI and a CLI drift apart.

**On progress.** These jobs report *state, elapsed time and their log tail* -- not a
percentage. The underlying commands emit human log lines and a rich progress bar sized for a
terminal, and inventing a number from that would be a fiction with a spinner attached. The
last log line is the honest progress signal, and it is what the panel shows.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["Job", "JobQueue", "JOB_KINDS", "TERMINAL_STATES"]

log = logging.getLogger("deeperfly")

#: Job kinds the GUI may submit, each mapped to the ``deeperfly`` subcommand it runs.
#:
#: An allow-list rather than "run any argv": the editor is reachable over HTTP, and a queue
#: that executed arbitrary commands would be a remote shell. Every entry here is a
#: subcommand that already exists, so nothing is reachable through a job that is not
#: reachable from the terminal.
JOB_KINDS: dict[str, str] = {
    "run": "run",
    "calibrate": "calibrate",
    "labels-suggest": "labels-suggest",
    "labels-export": "labels-export",
    "labels-merge": "labels-merge",
    "project-status": "project",
    "train": "train",  # reserved: no in-tree trainer yet (see the plan's fork F3)
}

#: States a job never leaves.
TERMINAL_STATES = ("done", "failed", "cancelled")

#: How many log lines to keep in memory per job for the panel's tail. The full log is on
#: disk; this is only what the UI polls.
_TAIL_LINES = 200


@dataclass
class Job:
    """One queued or finished command."""

    id: str
    kind: str
    argv: list[str]
    log_path: Path
    label: str = ""
    recording: str | None = None
    state: str = "queued"
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_utc: str | None = None
    finished_utc: str | None = None
    returncode: int | None = None
    error: str | None = None
    _tail: deque = field(default_factory=lambda: deque(maxlen=_TAIL_LINES), repr=False)
    _started_at: float | None = field(default=None, repr=False)
    _finished_at: float | None = field(default=None, repr=False)

    @property
    def command(self) -> str:
        """The exact shell command this job runs -- copyable into a terminal.

        The point of storing it: a GUI action that fails should be reproducible without
        anyone reverse-engineering what the GUI did.
        """
        return "deeperfly " + " ".join(shlex.quote(a) for a in self.argv)

    @property
    def elapsed(self) -> float | None:
        if self._started_at is None:
            return None
        end = self._finished_at if self._finished_at is not None else time.monotonic()
        return round(end - self._started_at, 2)

    @property
    def done(self) -> bool:
        return self.state in TERMINAL_STATES

    def tail(self, n: int = 20) -> list[str]:
        """The last ``n`` log lines (the honest progress signal -- see the module docstring)."""
        lines = list(self._tail)
        return lines[-int(n) :] if n else lines

    def as_dict(self, *, tail: int = 5) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "recording": self.recording,
            "state": self.state,
            "command": self.command,
            "created_utc": self.created_utc,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed": self.elapsed,
            "returncode": self.returncode,
            "error": self.error,
            "log": str(self.log_path),
            "tail": self.tail(tail),
        }


class JobQueue:
    """A FIFO queue of subprocess jobs, with one worker.

    Thread-safe for the handful of operations a web server performs (submit, list, cancel).
    The worker thread is started lazily on the first submit and is a daemon, so an
    interpreter exit never waits on it -- a queued-but-unstarted job is simply lost, which
    is correct: it had no effect yet, and its command is recorded if it mattered.

    Parameters
    ----------
    root
        Where job logs go (``<root>/jobs/<id>.log``).
    cwd
        Working directory for the subprocesses. Defaults to ``root``, so a job's relative
        paths mean what they mean in the project.
    python
        Interpreter to run ``-m deeperfly`` with; defaults to the current one, so a job
        inherits the venv the editor is running in rather than whatever is on ``PATH``.
    """

    def __init__(
        self, root: Path, *, cwd: Path | None = None, python: str | None = None
    ):
        self.root = Path(root)
        self.cwd = Path(cwd) if cwd else self.root
        self.python = python or sys.executable
        self.log_dir = self.root / "jobs"
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._pending: deque[str] = deque()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._current: subprocess.Popen | None = None
        self._current_id: str | None = None
        self._stopping = False

    # -- submission -----------------------------------------------------------

    def submit(
        self,
        kind: str,
        argv: list[str],
        *,
        label: str = "",
        recording: str | None = None,
    ) -> Job:
        """Queue a job.

        Parameters
        ----------
        kind
            A key of :data:`JOB_KINDS`.
        argv
            Arguments *after* the subcommand (the subcommand comes from ``kind``).
        label
            Short human-facing description for the panel.
        recording
            Which recording this concerns, for grouping in the UI.

        Returns
        -------
        Job
            The queued job.

        Raises
        ------
        ValueError
            If ``kind`` is not allow-listed, or an argument is not a plain string. Both are
            refusals rather than sanitizations: the queue is reachable over HTTP, so it
            executes only shapes it recognizes.
        """
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}; allowed: {sorted(JOB_KINDS)}")
        if not all(isinstance(a, str) for a in argv):
            raise ValueError("every job argument must be a string")
        job_id = uuid.uuid4().hex[:12]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            id=job_id,
            kind=kind,
            argv=[JOB_KINDS[kind], *argv],
            log_path=self.log_dir / f"{job_id}.log",
            label=label or kind,
            recording=recording,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._pending.append(job_id)
        self._ensure_worker()
        self._wake.set()
        log.info("queued job %s: %s", job_id, job.command)
        return job

    # -- inspection -----------------------------------------------------------

    def list(self) -> list[Job]:
        """Every job, newest first."""
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order)]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current_id is not None or bool(self._pending)

    # -- cancellation ---------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns whether anything was cancelled.

        A queued job is simply dropped. A running one is terminated, then killed if it does
        not exit -- ``terminate`` first so a stage that traps it can close its HDF5 file
        rather than leaving a truncated one.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.done:
                return False
            if job_id in self._pending:
                self._pending.remove(job_id)
                job.state = "cancelled"
                job.finished_utc = datetime.now(timezone.utc).isoformat()
                return True
            if self._current_id == job_id and self._current is not None:
                proc = self._current
            else:
                return False
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover -- a stuck child
            proc.kill()
        with self._lock:
            job.state = "cancelled"
        return True

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop the worker, cancelling anything running (used on server shutdown)."""
        self._stopping = True
        with self._lock:
            self._pending.clear()
            current = self._current
        if current is not None:
            current.terminate()
        self._wake.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)

    # -- the worker -----------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._loop, name="deeperfly-jobs", daemon=True
            )
            self._worker.start()

    def _loop(self) -> None:
        while not self._stopping:
            with self._lock:
                job_id = self._pending.popleft() if self._pending else None
            if job_id is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            job = self.get(job_id)
            if job is None or job.state == "cancelled":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        """Run one job to completion, streaming its output to the log and the tail."""
        argv = [self.python, "-m", "deeperfly", *job.argv]
        job.state = "running"
        job.started_utc = datetime.now(timezone.utc).isoformat()
        job._started_at = time.monotonic()
        env = {
            **os.environ,
            # Unbuffered, or the log tail lags the work by whatever the pipe buffer holds
            # and the panel looks frozen during the slowest stage.
            "PYTHONUNBUFFERED": "1",
        }
        try:
            with open(job.log_path, "w", encoding="utf-8") as sink:
                sink.write(f"$ {job.command}\n\n")
                sink.flush()
                proc = subprocess.Popen(
                    argv,
                    cwd=str(self.cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                with self._lock:
                    self._current, self._current_id = proc, job.id
                assert proc.stdout is not None
                for line in proc.stdout:
                    sink.write(line)
                    stripped = line.rstrip("\n")
                    if stripped:
                        job._tail.append(stripped)
                proc.wait()
                job.returncode = proc.returncode
        except Exception as exc:  # the subprocess could not even be started
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            log.exception("job %s could not run", job.id)
        else:
            if job.state != "cancelled":
                job.state = "done" if job.returncode == 0 else "failed"
                if job.returncode:
                    job.error = f"exited {job.returncode}"
        finally:
            job.finished_utc = datetime.now(timezone.utc).isoformat()
            job._finished_at = time.monotonic()
            with self._lock:
                self._current, self._current_id = None, None
            log.info("job %s %s (%.1fs)", job.id, job.state, job.elapsed or 0.0)
