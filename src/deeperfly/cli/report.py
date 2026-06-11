"""The ``init``, ``inspect`` and ``doctor`` command workers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from rich.text import Text

from ..config import DEFAULT_CONFIG_PATH
from ..results import PoseResult
from .console import _info_line, console

log = logging.getLogger("deeperfly")


def _cmd_init(args: argparse.Namespace) -> None:
    """Write the packaged default config to ``args.output``.

    Parameters
    ----------
    args
        The ``init`` namespace (``output``, ``force``).

    Raises
    ------
    SystemExit
        If the destination exists and ``--force`` was not given.
    """
    dst = Path(args.output)
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} already exists (pass --force to overwrite)")
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


def _cmd_inspect(args: argparse.Namespace) -> None:
    """Print a summary of the result file at ``args.input``.

    Parameters
    ----------
    args
        The ``inspect`` namespace (``input``).
    """
    result = PoseResult.load(args.input)
    _info_line("file:     ", args.input)
    _info_line("views:    ", f"{result.n_views}  {result.cameras.names}")
    _info_line("frames:   ", result.n_frames)
    _info_line(
        "skeleton: ", f"{result.skeleton.name}  ({result.skeleton.n_points} points)"
    )
    _info_line("has 3D:   ", result.pts3d is not None)
    if result.reproj_error is not None:
        _info_line(
            "reproj:   ",
            f"median {np.nanmedian(result.reproj_error):.3f} px"
            f"  max {np.nanmax(result.reproj_error):.3f} px",
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


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Report the installation and what this machine can run.

    Covers version + location, Python/OS, CPU/GPU inference (torch CUDA/MPS), the
    frame I/O (PyAV for video, OpenCV for image sequences), whether the detector
    weights are downloaded and where, and the default config path. Imports are lazy
    and each probe guarded, so a missing or broken piece is reported rather than
    crashing.

    Parameters
    ----------
    args
        The ``doctor`` namespace (no fields are read; kept for symmetry with the
        other command workers).
    """
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

    _doctor_header("weights")
    _doctor_row("cache dir", download.cache_dir())
    path = download.torch_weights_path()
    if path.exists():
        state = f"downloaded ({_fmt_bytes(path.stat().st_size)}) -- {path.name}"
    else:
        state = f"not downloaded -- would cache as {path.name}"
    _doctor_row("detector", state)

    _doctor_header("config")
    _doctor_row("default config", DEFAULT_CONFIG_PATH)
