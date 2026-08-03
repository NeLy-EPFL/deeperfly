"""The background job queue -- running pipeline commands from the editor.

Three properties are what make this safe to expose over HTTP, and each has a test:
the kinds are an **allow-list** (the editor is a web server, so a queue that ran arbitrary
argv would be a remote shell), jobs are **serialized** (two stages writing one results.h5
corrupt it), and they run **out of process** (so a crash cannot take unsaved labels with it).
"""

from __future__ import annotations

import time

import pytest

from deeperfly.jobs import JOB_KINDS, JobQueue


def _wait(job, *, timeout=60.0):
    """Block until ``job`` reaches a terminal state."""
    deadline = time.monotonic() + timeout
    while not job.done and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.done, f"job {job.id} did not finish (state={job.state})"
    return job


@pytest.fixture
def queue(tmp_path) -> JobQueue:
    q = JobQueue(tmp_path)
    yield q
    q.shutdown()


# -- the allow-list ------------------------------------------------------------


def test_an_unknown_kind_is_refused(queue):
    """The editor is reachable over HTTP; an open argv queue would be a remote shell."""
    with pytest.raises(ValueError, match="unknown job kind"):
        queue.submit("rm", ["-rf", "/"])


def test_non_string_arguments_are_refused(queue):
    with pytest.raises(ValueError, match="must be a string"):
        queue.submit("run", [{"not": "a string"}])  # type: ignore[list-item]


def test_every_allowed_kind_maps_to_a_real_subcommand():
    """A kind that named a nonexistent subcommand would fail only at run time."""
    from typer.main import get_command

    from deeperfly.cli.app import app

    known = set(get_command(app).commands)  # type: ignore[attr-defined]
    for kind, subcommand in JOB_KINDS.items():
        if kind == "train":
            continue  # reserved: no in-tree trainer yet (fork F3)
        assert subcommand in known, f"{kind} -> {subcommand} is not a deeperfly command"


# -- running -------------------------------------------------------------------


def test_a_job_runs_out_of_process_and_records_its_command(queue):
    job = queue.submit("project-status", ["--help"], label="help")
    assert job.command.startswith("deeperfly project --help")
    _wait(job)
    assert job.state == "done"
    assert job.returncode == 0
    assert job.elapsed is not None and job.elapsed >= 0
    # The full log is on disk, and starts with the command so the file is self-explaining.
    text = job.log_path.read_text()
    assert text.startswith("$ deeperfly project --help")
    assert job.tail(5)  # ...and the tail is what the panel polls


def test_a_failing_job_is_reported_not_raised(queue):
    """A bad command must mark the job failed, not take the server down."""
    job = _wait(queue.submit("run", ["/definitely/not/a/recording"]))
    assert job.state == "failed"
    assert job.returncode not in (0, None)
    assert job.error


def test_jobs_run_one_at_a_time(queue):
    """Two stages writing one results.h5 corrupt it, so the queue is the lock."""
    jobs = [queue.submit("project-status", ["--help"]) for _ in range(4)]
    for job in jobs:
        _wait(job)
    # Every job finished, and no two overlapped: each start is after the previous finish.
    starts = [j._started_at for j in jobs]
    finishes = [j._finished_at for j in jobs]
    for i in range(1, len(jobs)):
        assert starts[i] >= finishes[i - 1] - 1e-6


def test_the_queue_reports_when_it_is_busy(queue):
    job = queue.submit("project-status", ["--help"])
    _wait(job)
    assert not queue.busy


# -- listing and lookup --------------------------------------------------------


def test_jobs_are_listed_newest_first(queue):
    first = queue.submit("project-status", ["--help"], label="a")
    second = queue.submit("project-status", ["--help"], label="b")
    _wait(second)
    listed = [j.id for j in queue.list()]
    assert listed == [second.id, first.id]
    assert queue.get(first.id) is first
    assert queue.get("nope") is None


def test_a_job_serializes_to_plain_data(queue):
    import json

    job = _wait(queue.submit("project-status", ["--help"], recording="flyA"))
    payload = job.as_dict()
    json.dumps(payload)
    assert payload["recording"] == "flyA"
    assert payload["state"] == "done"
    assert isinstance(payload["tail"], list)


# -- cancellation --------------------------------------------------------------


def test_a_queued_job_can_be_cancelled_before_it_starts(tmp_path):
    q = JobQueue(tmp_path)
    try:
        # Submit a slow job first so the second stays queued behind it.
        slow = q.submit("run", ["/definitely/not/a/recording"])
        queued = q.submit("project-status", ["--help"])
        assert q.cancel(queued.id)
        assert queued.state == "cancelled"
        _wait(slow)
        # A cancelled job never ran, so it wrote no log.
        assert not queued.log_path.exists()
    finally:
        q.shutdown()


def test_cancelling_a_finished_job_is_a_no_op(queue):
    job = _wait(queue.submit("project-status", ["--help"]))
    assert not queue.cancel(job.id)
    assert job.state == "done"


def test_cancelling_an_unknown_job_is_false(queue):
    assert not queue.cancel("nope")


def test_shutdown_clears_the_queue(tmp_path):
    q = JobQueue(tmp_path)
    q.submit("run", ["/definitely/not/a/recording"])
    q.submit("project-status", ["--help"])
    q.shutdown()
    assert not q._pending


# -- the HTTP surface ----------------------------------------------------------


@pytest.fixture
def client(result, tmp_path):
    """A test client whose session HAS a job queue."""
    from fastapi.testclient import TestClient

    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.server import create_app
    from deeperfly.gui.session import Session
    from deeperfly.gui.state import EditorState

    queue = JobQueue(tmp_path / "proj")
    session = Session.build(
        EditorState.from_result(result),
        FrameSource({}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        project_root=tmp_path / "proj",
        recording_slug="flyA",
    )
    yield TestClient(create_app(session, jobs=queue)), queue
    queue.shutdown()


def test_the_api_lists_and_submits(client):
    api, queue = client
    assert api.get("/api/jobs").json()["enabled"] is True

    posted = api.post(
        "/api/jobs",
        json={"kind": "project-status", "argv": ["--help"], "label": "help"},
    )
    assert posted.status_code == 200
    job_id = posted.json()["id"]
    _wait(queue.get(job_id))

    detail = api.get(f"/api/jobs/{job_id}").json()
    assert detail["state"] == "done"
    assert detail["command"].startswith("deeperfly project --help")
    assert api.get("/api/jobs").json()["jobs"][0]["id"] == job_id


def test_the_api_refuses_a_disallowed_kind(client):
    api, _ = client
    bad = api.post("/api/jobs", json={"kind": "rm", "argv": ["-rf", "/"]})
    assert bad.status_code == 400
    assert "unknown job kind" in bad.json()["detail"]


def test_the_api_404s_an_unknown_job(client):
    api, _ = client
    assert api.get("/api/jobs/nope").status_code == 404
    assert api.delete("/api/jobs/nope").status_code == 404


def test_a_session_without_a_project_reports_why_not(result, tmp_path):
    """An absent button must be explained, not merely absent."""
    from fastapi.testclient import TestClient

    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.server import create_app
    from deeperfly.gui.session import Session
    from deeperfly.gui.state import EditorState

    session = Session.build(
        EditorState.from_result(result),
        FrameSource({}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
    )
    api = TestClient(create_app(session))  # no queue
    payload = api.get("/api/jobs").json()
    assert payload["enabled"] is False
    assert "open a project" in payload["reason"]
    assert api.post("/api/jobs", json={"kind": "run", "argv": []}).status_code == 409
    # ...and meta says so too, so the UI can decide before it asks.
    meta = api.get("/api/meta").json()
    assert meta["has_jobs"] is False
    assert meta["project_root"] is None


def test_meta_reports_the_project_context_when_there_is_one(client):
    api, _ = client
    meta = api.get("/api/meta").json()
    assert meta["has_jobs"] is True
    assert meta["recording"] == "flyA"
    assert meta["project_root"]
