"""``python -m deeperfly`` -- the CLI, without depending on the console script.

The ``deeperfly`` entry point on ``PATH`` may belong to a different environment than the
interpreter currently running (a common state with several venvs, and the normal state inside
a subprocess). ``python -m deeperfly`` cannot be wrong about that: it uses the interpreter it
was invoked with, and therefore that interpreter's installed deeperfly.

That guarantee is why :class:`deeperfly.project.jobs.JobQueue` runs its jobs this way -- a job
launched from the editor has to be the same deeperfly the editor is.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
