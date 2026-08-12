"""The dense-38 HRNet detector: one channel per tracked point, in every view.

The shipped DeepFly2D network (:mod:`deeperfly.pose2d.model`) is a **19-channel,
one-side** detector: a side camera is run twice -- once mirrored -- and the two passes
are packaged into the near-side points of the two views that face it. That is why
``[pose2d.output_points]`` in the packaged config names a ``*_flip`` pathway for every
left camera, and why the far side of a side camera is simply absent from the result
(``NaN``, which is how visibility is encoded).

This module is the other kind of detector: **38 channels, all of them, in every view**.
One pathway per camera, no mirrored twin, and a contralateral point gets a prediction
instead of a ``NaN``. It is the architecture trained in the ``dfpose`` repo -- an ImageNet
HRNet backbone (``timm``) with a multi-scale stride-4 heatmap head -- and this module
exists so a checkpoint from there is *runnable from a config*, through the same
sources -> preprocessors -> models -> pathways plan as everything else.

Three parts of the contract are not the shipped detector's, and each is a silent
wrong-answer if it is got wrong rather than a crash:

**1. The heatmap is bigger than the image.** It covers the input extended by
:data:`HM_MARGIN` on every side -- 96x192 cells at stride 4 for a 256x512 input -- so a
joint pushed outside the frame still has a cell to peak in. A decode that assumes the
heatmap spans the image is off by 25% of the frame. :func:`decode_points` inverts the
padding, and returns input-normalized coordinates that may legitimately fall outside
``[0, 1]``: that is an off-frame joint, and clamping it throws away the only evidence
there is about where the joint went.

**2. The sub-pixel decode is not the same one.** ``heatmap_to_points`` uses the
cell-centre convention ``(col + 0.5) / W`` because DeepFly2D's targets peak at
``floor(kp * size)``. These targets are rendered at the *continuous* cell coordinate, so
the matched inverse has no half-cell term at all -- applying deeperfly's would bias every
point by half a cell (2 input px). The refinement is the parabolic fit through the peak's
immediate neighbours, clamped to half a cell.

**3. Normalization is the checkpoint's, not the config's.** The network was trained on
``(gray/255 - mean) / std`` with ``mean``/``std`` *stored in the checkpoint*. A
:class:`~deeperfly.pose2d.models.ModelSpec` for this class must therefore declare
``mean = 0.0`` -- the model applies its own -- and :func:`load_hrnet` refuses anything
else rather than quietly shifting every input by DeepFly2D's 0.22.

The class deliberately mirrors the submodule names ``backbone`` / ``laterals`` / ``head``
of the training-side definition, so a checkpoint loads ``strict=True`` with no key
translation. ``dfpose``'s ``tests/test_deeperfly_parity.py`` asserts the two agree
bit-for-bit on the same weights; that test is what makes the duplication safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("deeperfly")

#: Model input ``(height, width)``: grayscale, resized by the pathway.
IMG_HW: tuple[int, int] = (256, 512)

#: Input pixels per heatmap cell.
STRIDE: int = 4

#: The heatmap covers the input extended by this fraction on every side, so a joint
#: outside the frame still has a cell. Rounded to a whole number of cells below.
HM_MARGIN: float = 0.25

#: Padding in INPUT pixels on each side, ``(y, x)`` -- a whole number of cells.
HM_PAD_PX: tuple[int, int] = (
    int(round(HM_MARGIN * IMG_HW[0] / STRIDE)) * STRIDE,
    int(round(HM_MARGIN * IMG_HW[1] / STRIDE)) * STRIDE,
)

#: Heatmap output ``(height, width)`` in cells.
HM_HW: tuple[int, int] = (
    (IMG_HW[0] + 2 * HM_PAD_PX[0]) // STRIDE,
    (IMG_HW[1] + 2 * HM_PAD_PX[1]) // STRIDE,
)

#: Where input pixel ``(0, 0)`` sits inside the heatmap field, in INPUT px, as ``(x, y)``.
HM_ORIGIN_XY: tuple[int, int] = (HM_PAD_PX[1], HM_PAD_PX[0])

#: timm's HRNet ``features_only`` heads bottleneck every branch to these widths at
#: strides 4/8/16/32. Asserted at construction: a timm release that changes it would
#: silently change the ported architecture.
EXPECTED_CHANNELS: tuple[int, ...] = (128, 256, 512, 1024)

#: Checkpoint ``backbone`` string -> timm model name.
MODEL_NAMES: dict[str, str] = {
    "hrnet_timm": "hrnet_w18_small_v2",
    "hrnet_w18_small_v2": "hrnet_w18_small_v2",
    "hrnet_w32": "hrnet_w32",
}


def _torch():
    import torch

    return torch


class HRNetPose:
    """Built lazily so importing :mod:`deeperfly.pose2d` never imports torch/timm.

    :func:`build` returns the real ``nn.Module``; this class only exists to hold the
    construction in one documented place and is never instantiated.
    """

    def __new__(cls, *a: Any, **k: Any):  # pragma: no cover - not a real class
        raise TypeError("use deeperfly.pose2d.hrnet.build()")


def build(
    n_keypoints: int,
    model_name: str = "hrnet_w18_small_v2",
    *,
    pretrained: bool = False,
    lat: int = 96,
    mid: int = 128,
):
    """The dense-38 detector as an ``nn.Module``: ``(N,3,H,W) -> [ (N,K,96,192) ]``.

    Returns a one-element list so the shared ``_forward_last`` seam (which takes the
    LAST stack of a stacked-hourglass output) works unchanged.
    """
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _HRNetPose(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # out_indices (1,2,3,4) -> strides 4/8/16/32; index 0 is the stride-2 stem.
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=3,
                out_indices=(1, 2, 3, 4),
            )
            chs = tuple(self.backbone.feature_info.channels())
            if chs != EXPECTED_CHANNELS:
                raise RuntimeError(
                    f"timm {timm.__version__} gives {model_name} feature channels {chs}, "
                    f"expected {EXPECTED_CHANNELS}; the trained weights assume the latter"
                )
            self.laterals = nn.ModuleList(
                [nn.Conv2d(c, lat, kernel_size=1) for c in chs]
            )
            self.head = nn.Sequential(
                nn.Conv2d(lat * len(chs), mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, n_keypoints, kernel_size=1),
            )
            self.num_classes = int(n_keypoints)
            #: Set by :func:`load_hrnet` from the checkpoint. The model normalizes its
            #: own input, so a ModelSpec for it declares ``mean = 0.0``.
            self.norm_mean = 0.0
            self.norm_std = 1.0
            #: The heatmap field is PADDED, so the shared decode's normalization does
            #: not apply. `LoadedModel` reads this flag and routes to `predict_points`
            #: here instead -- see the module docstring, point 1.
            self.owns_decode = True

        def forward(self, x: "torch.Tensor") -> list["torch.Tensor"]:
            # The pathway hands us (N, 3, H, W) already resized. The three channels are
            # a grayscale frame expanded by `LoadedModel.prepare`; the network was
            # trained on one channel repeated back to three, so take one and repeat --
            # NOT a luminance mix, which would be a different input than training saw.
            g = x[:, :1]
            g = (g - self.norm_mean) / self.norm_std
            fs = list(self.backbone(g.repeat(1, 3, 1, 1)))
            h, w = fs[0].shape[-2:]  # stride-4 == the UNPADDED heatmap grid
            red = [lat(f) for lat, f in zip(self.laterals, fs, strict=True)]
            ups = [red[0]] + [
                F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)
                for r in red[1:]
            ]
            feat = torch.cat(ups, dim=1)
            py, px = HM_PAD_PX[0] // STRIDE, HM_PAD_PX[1] // STRIDE
            if py or px:
                # Zeros, not reflection: "nothing is here" is the truth outside the
                # frame, and reflecting would paste a mirrored fly exactly where an
                # off-image joint is supposed to be inferred.
                feat = F.pad(feat, (px, px, py, py))
            return [self.head(feat)]

    return _HRNetPose()


def refined_argmax(heatmaps):
    """Hard peak + parabolic sub-pixel refinement, ``(..., K, 2)`` in heatmap cells.

    The second difference is clamped strictly negative so a flat neighbourhood gives a
    zero shift rather than a division blow-up, and the shift is clamped to half a cell so
    the refinement can never leave the peak's own cell. Identical arithmetic to the
    training repo's ``dfpose.heatmaps.refined_argmax``.
    """
    torch = _torch()
    *lead, h, w = heatmaps.shape
    hm = heatmaps.reshape(-1, h, w)
    n = hm.shape[0]
    idx = hm.reshape(n, h * w).argmax(dim=1)
    iy, ix = idx // w, idx % w
    ar = torch.arange(n, device=hm.device)

    def gather(yy, xx):
        return hm[ar, yy.clamp(0, h - 1), xx.clamp(0, w - 1)]

    c = gather(iy, ix)
    xl, xr = gather(iy, ix - 1), gather(iy, ix + 1)
    yu, yd = gather(iy - 1, ix), gather(iy + 1, ix)
    dx = 0.5 * (xl - xr) / (xl - 2 * c + xr).clamp(max=-1e-6)
    dy = 0.5 * (yu - yd) / (yu - 2 * c + yd).clamp(max=-1e-6)
    x = ix.to(hm.dtype) + dx.clamp(-0.5, 0.5)
    y = iy.to(hm.dtype) + dy.clamp(-0.5, 0.5)
    return torch.stack([x, y], dim=-1).reshape(*lead, 2)


def cells_to_input_normalized(cells):
    """Heatmap cells ``(..., 2)`` -> input-normalized ``(x, y)``, padding inverted.

    A cell index ``c`` is input pixel ``c * STRIDE - HM_ORIGIN``; dividing by the input
    size gives the ``[0, 1]`` convention the geometry layer expects. Values outside
    ``[0, 1]`` are meaningful and must not be clipped -- they are joints outside the
    pathway's frame, which the padded field exists to represent.
    """
    torch = _torch()
    x = (cells[..., 0] * STRIDE - HM_ORIGIN_XY[0]) / IMG_HW[1]
    y = (cells[..., 1] * STRIDE - HM_ORIGIN_XY[1]) / IMG_HW[0]
    return torch.stack([x, y], dim=-1)


def decode_points(heatmaps):
    """``(..., K, H, W)`` heatmaps -> input-normalized ``(..., K, 2)`` peaks + conf."""
    cells = refined_argmax(heatmaps)
    xy = cells_to_input_normalized(cells)
    conf = heatmaps.flatten(-2).amax(dim=-1)
    return xy, conf


def load_hrnet(weights: str | Path, *, dev: str | None = None, mean: float = 0.0):
    """Load a dense-38 checkpoint and return the eval-mode module on ``dev``.

    ``mean`` is the ``ModelSpec``'s declared mean and must be ``0.0``: this network
    carries its own normalization constants, so a config that also subtracts DeepFly2D's
    0.22 would shift every input by a quarter of its range with nothing to notice.
    """
    import torch

    from .model import device

    if float(mean) != 0.0:
        raise SystemExit(
            "a dense-38 hrnet model must declare mean = 0.0 in its [[pose2d.models]] "
            f"table (got {mean}); the checkpoint carries its own mean/std and applies "
            "them itself"
        )
    path = Path(weights)
    if not path.exists():
        raise SystemExit(f"no detector checkpoint at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    if any(k.startswith(("fusion.", "fuse_in.", "fuse_out.")) for k in sd):
        raise SystemExit(
            f"{path} is a CROSS-VIEW checkpoint: its residual is not the identity once "
            "trained, so running it one view at a time evaluates a different function. "
            "Run it through the grouped multiview path, not a per-view pathway."
        )
    names = list(ck.get("point_names") or [])
    n_kp = len(names) or int(sd["head.3.weight"].shape[0])
    model = build(n_kp, MODEL_NAMES.get(ck.get("backbone", ""), "hrnet_w18_small_v2"))
    model.load_state_dict(sd, strict=True)
    model.norm_mean = float(ck.get("mean", 0.0))
    model.norm_std = float(ck.get("std", 1.0))
    model.point_names = names
    target = dev or device()
    model = model.eval().to(target)
    log.info(
        "dense-38 detector %s: %d channels, mean %.4f std %.4f, on %s",
        path.name,
        n_kp,
        model.norm_mean,
        model.norm_std,
        target,
    )
    return model


def predict_points(model, inputs, *, method: str = "weighted", radius: int = 2):
    """Forward + this model's own decode -> input-normalized peaks and confidence.

    ``method``/``radius`` are accepted for interface parity with the shipped detector
    and ignored: the sub-pixel readout is the parabolic fit the targets were rendered
    for, and substituting the windowed centroid would bias the result.
    """
    import torch

    from .model import _as_torch, _autocast_dtype

    dev = next(model.parameters()).device
    x = _as_torch(inputs).float().to(dev)
    lead = x.shape[:-3]
    flat = x.reshape(-1, *x.shape[-3:])
    dtype = _autocast_dtype(model, dev)
    with torch.inference_mode():
        if dtype is not None:
            with torch.autocast(dev.type, dtype=dtype):
                hm = model(flat)[-1]
        else:
            hm = model(flat)[-1]
        xy, conf = decode_points(hm.float())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    k = hm.shape[1]
    return (
        xy.reshape(*lead, k, 2).cpu().numpy().astype(np.float32),
        conf.reshape(*lead, k).cpu().numpy().astype(np.float32),
    )


def predict_heatmaps(model, inputs) -> np.ndarray:
    """Heatmaps as host NumPy, ``(..., K, 96, 192)`` -- the PADDED field.

    The candidate path decodes these with :func:`deeperfly.pose2d.inference.heatmap_to_points`,
    whose normalization assumes the heatmap spans the image. It does not here, so a
    caller must map the result through :func:`cells_to_input_normalized` instead.
    """
    import torch

    from .model import _as_torch, _autocast_dtype

    dev = next(model.parameters()).device
    x = _as_torch(inputs).float().to(dev)
    lead = x.shape[:-3]
    flat = x.reshape(-1, *x.shape[-3:])
    dtype = _autocast_dtype(model, dev)
    with torch.inference_mode():
        if dtype is not None:
            with torch.autocast(dev.type, dtype=dtype):
                hm = model(flat)[-1]
        else:
            hm = model(flat)[-1]
    return hm.float().reshape(*lead, *hm.shape[-3:]).cpu().numpy()
