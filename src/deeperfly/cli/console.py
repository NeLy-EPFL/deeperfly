"""Shared Rich console, logging setup and the frames/second progress bar."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

#: rich output: status/results to stdout, logs and the progress bar to stderr, so
#: piping stdout stays clean and progress never clobbers a log line.
console = Console()
err_console = Console(stderr=True)

log = logging.getLogger("deeperfly")


class _FPSColumn(ProgressColumn):
    """Detection throughput in frames/second (rich ships no built-in FPS column).

    ``task.speed`` is the smoothed completed-per-second rate; since the bar ticks
    once per frame, that is frames/second. ``finished_speed`` holds the final
    average once the bar completes.
    """

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if not speed:
            return Text("  ?.? fps", style="progress.data.speed")
        return Text(f"{speed:5.1f} fps", style="progress.data.speed")


def _frame_progress() -> Progress:
    """A frames/second progress bar on the stderr console (detection, rendering).

    Shown only while INFO logging is on (so ``--log-level warning+`` hides it) and
    stderr is a TTY (tqdm-style); otherwise it is a no-op, so log lines and the bar
    never overwrite each other.

    Returns
    -------
    rich.progress.Progress
        The configured (possibly disabled) progress bar.
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("frames"),
        _FPSColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        disable=not (log.isEnabledFor(logging.INFO) and err_console.is_terminal),
    )


@contextmanager
def _rich_progress(total, description):
    """Rich-backed progress factory for the library stages (one bar per task).

    Implements the ``progress(total, description) -> context manager yielding wrap``
    contract the library stages expect (:func:`deeperfly.pose2d.stream.detect_2d`,
    :func:`deeperfly.pipeline.render_videos`): ``wrap(rng)`` iterates ``rng``,
    advancing a frames/second :func:`_frame_progress` bar once per item. Each call
    opens its own short-lived bar, so log lines emitted by the stages *between*
    progress phases (e.g. bundle adjustment, triangulation) are not held under a
    live display.

    Parameters
    ----------
    total
        The task's total frame count (the bar's denominator).
    description
        The task label shown on the bar.

    Yields
    ------
    wrap : callable
        ``wrap(rng)`` yields each item of ``rng``, advancing the bar per item.
    """
    bar = _frame_progress()
    with bar:
        task = bar.add_task(description, total=total)

        def wrap(rng):
            for item in rng:
                yield item
                bar.advance(task)

        yield wrap


def _configure_logging(level_name: str) -> None:
    """Configure the root log level from a ``--log-level`` name (default ``info``).

    ``info`` surfaces the per-stage messages and the progress bar; ``warning`` or
    higher hides them (the "quiet" mode). Records render through rich's
    :class:`~rich.logging.RichHandler` on the same stderr console as the bar, so
    log lines and the bar never overwrite each other.

    Parameters
    ----------
    level_name
        A logging level name (``"debug"``, ``"info"``, ``"warning"``, ...).
    """
    level = getattr(logging, level_name.upper())
    handler = RichHandler(
        console=err_console,
        show_time=False,
        show_path=False,
        markup=False,  # log messages carry dict/list reprs; don't parse their brackets
        rich_tracebacks=True,
    )
    logging.basicConfig(level=level, format="%(message)s", handlers=[handler])
    log.setLevel(level)
    # JAX warns when the TPU plugin's libtpu.so is absent (the normal case). Mute
    # that noise unless we're at debug.
    if level > logging.DEBUG:
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def _info_line(label: str, value: object) -> None:
    """Print one ``label   value`` row with a colored label.

    Built as :class:`rich.text.Text` (not markup) so values containing brackets
    (e.g. the camera-name list) are never parsed as style tags.

    Parameters
    ----------
    label
        The (colored) row label.
    value
        The value printed after the label (stringified).
    """
    line = Text(label, style="bold cyan")
    line.append(str(value))
    console.print(line)
