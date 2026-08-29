"""The ``init``, ``inspect`` and ``doctor`` command workers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.text import Text

from ..config import DEFAULT_CONFIG_PATH
from ..results import PoseResult
from .console import LogLevel, LogLevelOption, _configure_logging, _info_line, console

log = logging.getLogger("deeperfly")


def init(
    output: Annotated[
        str, typer.Argument(help="destination (defaults to config.toml)")
    ] = "config.toml",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="overwrite an existing file")
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Write a default config.toml to edit (destination defaults to config.toml)."""
    _configure_logging(log_level.value)
    dst = Path(output)
    if dst.exists() and not overwrite:
        console.print(
            f"[yellow]{dst} already exists[/yellow]; pass --overwrite to replace it "
            "(left unchanged)"
        )
        return
    dst.write_text(DEFAULT_CONFIG_PATH.read_text())
    console.print(f"[green]wrote[/green] {dst}")
    # markup=False: the message shows literal [cameras] config sections, which rich
    # would otherwise try to parse as style tags.
    console.print(
        "next: edit [cameras] to match your rig, then "
        f"'deeperfly run <recording> -c {dst}' "
        "(outputs land in <recording>/deeperfly_outputs/; override with -o <dir>)",
        markup=False,
        highlight=False,
    )


def inspect(
    input: Annotated[str, typer.Argument(help="path to a result .h5 file")],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Print a summary of a result .h5 file."""
    _configure_logging(log_level.value)
    result = PoseResult.load(input)
    _info_line("file:     ", input)
    _info_line("views:    ", f"{result.n_views}  {result.cameras.names}")
    _info_line("frames:   ", result.n_frames)
    _info_line(
        "skeleton: ", f"{result.skeleton.label}  ({result.skeleton.n_points} points)"
    )
    _info_line("has 3D:   ", result.pts3d is not None)
    if result.reproj_error is not None:
        _info_line(
            "reproj:   ",
            f"median {np.nanmedian(result.reproj_error):.3f} px"
            f"  max {np.nanmax(result.reproj_error):.3f} px",
        )


# -- repack: rewrite result files in the current schema ------------------------


def _results_files(targets: "list[str]") -> "list[Path]":
    """Every ``results.h5`` named by ``targets``, deduplicated and sorted.

    A target may be the file itself or a directory to search, which is what makes
    repacking a whole corpus one command rather than a shell loop.
    """
    found: set[Path] = set()
    for target in targets:
        path = Path(target)
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("results.h5"))
        elif path.exists():
            found.add(path.resolve())
        else:
            console.print(f"[yellow]skipped[/yellow] {path} (does not exist)")
    return sorted(found)


def repack(
    paths: Annotated[
        list[str],
        typer.Argument(
            help="result .h5 files, or directories to search for results.h5"
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="report what each file would shrink to without replacing it",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Rewrite result .h5 files in the current schema, smaller, without recomputing."""
    _configure_logging(log_level.value)
    import tempfile

    from ..results import FORMAT_VERSION, repack, stored_version

    files = _results_files(list(paths))
    if not files:
        console.print("[yellow]no results.h5 found[/yellow] at the given paths")
        return
    total_before = total_after = 0
    done = skipped = failed = 0
    for path in files:
        if stored_version(path) == FORMAT_VERSION:
            skipped += 1
            log.debug("%s is already schema v%d", path, FORMAT_VERSION)
            continue
        try:
            if dry_run:
                # A dry run still does the work -- it is the only honest way to report
                # the size -- and throws the result away instead of moving it into place.
                with tempfile.TemporaryDirectory(dir=str(path.parent)) as tmp:
                    before, after = repack(path, dst=Path(tmp) / "results.h5")
            else:
                before, after = repack(path)
        except (ValueError, OSError) as e:
            failed += 1
            console.print(f"[red]failed[/red] {path}: {e}")
            continue
        done += 1
        total_before += before
        total_after += after
        console.print(
            f"{'would repack' if dry_run else 'repacked'} {path}  "
            f"{_fmt_bytes(before)} -> {_fmt_bytes(after)}  "
            f"[green]{before / max(after, 1):.2f}x[/green]"
        )
    verb = "would save" if dry_run else "saved"
    _info_line(
        "files:    ", f"{done} repacked, {skipped} already current, {failed} failed"
    )
    if done:
        _info_line(
            "total:    ",
            f"{_fmt_bytes(total_before)} -> {_fmt_bytes(total_after)}  "
            f"({verb} {_fmt_bytes(total_before - total_after)}, "
            f"{total_before / max(total_after, 1):.2f}x)",
        )


# -- doctor: installation / runtime report -----------------------------------


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size (``1.2 GiB``).

    Parameters
    ----------
    n
        A size in bytes.

    Returns
    -------
    str
        The size rendered with a binary (KiB/MiB/...) unit.
    """
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _default_model_specs() -> list[tuple[str, str]]:
    """``(weights name, whether it resolves)`` for each model the default config names.

    The one question a stuck user actually has, answered against the config that will
    actually run rather than against a cache nothing writes to any more.
    """
    from ..config import Config
    from ..pose2d.download import resolve_weights

    try:
        specs = Config.default().detection_plan().models.values()
    except Exception as exc:  # a broken packaged config is its own, louder problem
        return [("(default config)", f"could not be read: {exc}")]
    out: list[tuple[str, str]] = []
    for spec in specs:
        if not spec.weights:
            out.append((f"{spec.name} (class {spec.cls})", "no 'weights' named"))
            continue
        try:
            path = resolve_weights(spec.weights, cls=spec.cls, model_name=spec.name)
        except SystemExit:
            out.append((str(spec.weights), "NOT FOUND on the search path above"))
        else:
            out.append((str(spec.weights), f"found at {path}"))
    return out


def _doctor_header(title: str) -> None:
    """Print a blank line then a section title (its own colored line).

    Parameters
    ----------
    title
        The section title.
    """
    console.print()
    console.print(Text(title, style="bold magenta"))


def _doctor_row(label: str, value: object, *, width: int = 18) -> None:
    """Print one indented ``label   value`` row, label padded to ``width``.

    Built as :class:`~rich.text.Text` (not markup) so values containing brackets
    (e.g. JAX's ``[cuda:0]`` device list) are never parsed as style tags.

    Parameters
    ----------
    label
        The row label (padded to ``width``).
    value
        The value printed after the label (stringified).
    width
        Column width the label is padded to.
    """
    line = Text("  ")
    line.append(f"{label:<{width}}", style="bold cyan")
    line.append(str(value))
    console.print(line)


def _probe_torch() -> dict:
    """PyTorch presence + accelerator availability, without raising.

    Probing CUDA/MPS can fail on a broken install, so every query is guarded and
    missing keys mean "unknown/no".

    Returns
    -------
    dict
        ``{"installed": bool, ...}`` with optional ``version`` / ``cuda`` / ``mps``
        keys when detectable.
    """
    info: dict = {"installed": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        log.debug("torch not importable: %s", exc)
        return info
    info.update(installed=True, version=torch.__version__)
    try:
        if torch.cuda.is_available():
            info["cuda"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        log.debug("torch.cuda probe failed: %s", exc)
    try:
        info["mps"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        log.debug("torch.backends.mps probe failed: %s", exc)
    return info


def doctor(log_level: LogLevelOption = LogLevel.info) -> None:
    """Report installation/runtime: accelerators, frame I/O, weights."""
    _configure_logging(log_level.value)
    import importlib.metadata
    import importlib.util
    import platform

    from ..pose2d import detector, download

    _doctor_header("deeperfly")
    try:
        version = importlib.metadata.version("deeperfly")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (not installed as a package)"
    _doctor_row("version", version)
    _doctor_row("location", Path(__file__).resolve().parent.parent)

    _doctor_header("system")
    _doctor_row(
        "python", f"{platform.python_version()} ({platform.python_implementation()})"
    )
    _doctor_row("platform", platform.platform())

    torch_info = _probe_torch()
    _doctor_header("inference")
    if torch_info["installed"]:
        accel = []
        if "cuda" in torch_info:
            accel.append(f"CUDA: {torch_info['cuda']}")
        if torch_info.get("mps"):
            accel.append("Metal (MPS)")
        _doctor_row(
            "torch",
            f"{torch_info['version']}  ({', '.join(accel) if accel else 'CPU only'})",
        )
    else:
        _doctor_row("torch", "not installed")

    gpu = "cuda" in torch_info or torch_info.get("mps")
    mem = detector.gpu_memory_bytes()
    if gpu:
        _doctor_row(
            "GPU inference",
            f"available ({_fmt_bytes(mem)} memory)" if mem else "available",
        )
    else:
        _doctor_row("GPU inference", "not available -- CPU only")
    _doctor_row("detector", "torch" if torch_info["installed"] else "none")

    _doctor_header("frame I/O")
    have_av = importlib.util.find_spec("av") is not None
    have_cv2 = importlib.util.find_spec("cv2") is not None
    _doctor_row("video read/write", "pyav" if have_av else "av not installed")
    _doctor_row("image read", "opencv" if have_cv2 else "opencv not installed")

    _doctor_header("gui")
    have_web = (
        importlib.util.find_spec("fastapi") is not None
        and importlib.util.find_spec("uvicorn") is not None
    )
    _doctor_row(
        "deeperfly gui",
        "FastAPI + uvicorn available"
        if have_web
        else "missing -- core deps absent, reinstall deeperfly",
    )

    # Nothing auto-provisions, so what a stuck user needs is not "is it downloaded" but
    # "where does deeperfly look, and does the checkpoint my config names turn up there".
    _doctor_header("weights")
    raw = os.environ.get(download.MODELS_ENV, "")
    _doctor_row(
        download.MODELS_ENV,
        raw if raw else "unset -- set it to the directory holding the checkpoints",
    )
    for i, d in enumerate(download.search_path()):
        found = sorted(p.name for p in d.glob("*.pth")) if d.is_dir() else []
        state = f"{len(found)} .pth" if found else ("empty" if d.is_dir() else "absent")
        _doctor_row(f"searched [{i}]", f"{d}  ({state})")
    for name, state in _default_model_specs():
        _doctor_row("default wants", f"{name}  --  {state}")

    _doctor_header("config")
    _doctor_row("default config", DEFAULT_CONFIG_PATH)
