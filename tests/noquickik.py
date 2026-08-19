"""Make ``quickik`` unimportable, so a plain install can be tested for.

Load it with ``-p noquickik``::

    uv run pytest -p noquickik

The inverse-kinematics solver is an optional extra (``deeperfly[ik]``) and the stage that
uses it is ON BY DEFAULT, so "a plain install still works" is a property of the release
rather than a detail -- and it is one no ordinary run of the suite can check, because this
machine has the extra. The main CI matrix syncs *without* ``--extra ik`` in three of its
four jobs, which is where the property actually gets exercised; this plugin is how to
reproduce that locally before pushing.

Measured 2026-08-19: 1658 passed / 9 skipped without the extra, against 1694 / 1 with it.
Nothing FAILS -- the stage logs a reason and skips, which is the behavior that lets it
default on. A regression here shows up as a failure rather than a skip.

Not in ``addopts``: the default run should exercise the solver where it is available.
"""

import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    """Refuse ``quickik`` at import, the way an install without the extra does."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "quickik" or fullname.startswith("quickik."):
            raise ImportError(f"blocked by tests/noquickik.py: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())
# `require_quickik` is `functools.cache`d, and anything already imported would satisfy it.
for _name in [m for m in sys.modules if m == "quickik" or m.startswith("quickik.")]:
    del sys.modules[_name]
