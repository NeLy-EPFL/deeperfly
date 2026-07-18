"""A lightweight single-side detector loaded from TorchScript (no timm/smp needed).

The deeperfly-train repo trains compact single-side (19-channel) pose nets
(hrnet_timm / hrnet_w32 / unet_smp) as drop-in replacements for the DeepFly2D
sh8 hourglass, and exports the chosen one to TorchScript + a sidecar
``*.meta.json``. This module loads that artifact with plain torch and adapts it
to the detector interface so the existing pathway routing (flip left cams, front
duplication, ``output_points``) drives it unchanged.

Two things differ from the hourglass and are handled here:
  * input contract: 1-channel grayscale, normalized ``(x/255 - mean) / std`` with
    the training set's mean/std (from the sidecar), at 256x512 -- NOT the sh8
    3-channel ``x - 0.22``;
  * forward output: a single ``(N, 19, Hh, Wh)`` heatmap tensor (not a per-stack
    list), decoded with the shared :func:`~deeperfly.pose2d.inference.heatmap_to_points`.

The channel order is identical to sh8 (front/mid/hind leg x5, antenna, abdomen0-2),
so ``[pose2d.output_points.*]` needs no change.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import LoadedModel, ModelSpec

_PRECISIONS = {"float32": None, "float16": "float16", "bfloat16": "bfloat16"}
_TORCH_DTYPE = {"float16": "float16", "bfloat16": "bfloat16"}


def load_lite(weights: str | None, **kwargs):
    """Load the TorchScript lite detector + its sidecar meta (mean/std/channels)."""
    import torch

    if weights is None or not Path(weights).exists():
        raise SystemExit(
            f"no lite detector checkpoint at {weights!r}; point the model's "
            "'weights' at the exported deepfly2d_lite.ts.pt"
        )
    module = torch.jit.load(weights, map_location="cpu").eval()
    meta_path = Path(weights).with_suffix("").with_suffix(".meta.json")
    if not meta_path.exists():  # allow deepfly2d_lite.ts.pt -> deepfly2d_lite.meta.json
        meta_path = Path(str(weights).replace(".ts.pt", ".meta.json"))
    meta = json.loads(Path(meta_path).read_text()) if meta_path.exists() else {}
    module._deeperfly_meta = meta  # type: ignore[attr-defined]
    module._deeperfly_precision = "float32"  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        module = module.cuda()
    return module


class LiteLoadedModel(LoadedModel):
    """LoadedModel for the TorchScript single-side net (1-ch + per-dataset norm)."""

    def __init__(self, spec: ModelSpec, module):
        super().__init__(spec, module)
        meta = getattr(module, "_deeperfly_meta", {}) or {}
        self.mean = float(meta.get("mean", spec.mean))
        self.std = float(meta.get("std", 1.0))

    def prepare(self, frames):
        """``(..., H, W, 3|1)`` frame(s) -> ``(..., 1, 256, 512)`` normalized 1-ch input.

        RGB is reduced to luma (ITU-R 601-2, matching PIL ``convert('L')`` used in
        training); grayscale-as-RGB (this rig) is unaffected. Resize is bilinear +
        antialias (as the base contract), then ``(x - mean) / std``.
        """
        import torch
        import torch.nn.functional as F

        from .inference import _to_torch_image

        img = _to_torch_image(frames)
        img = img.float() / 255.0 if not torch.is_floating_point(img) else img.float()
        if img.ndim >= 3 and img.shape[-1] == 3:  # RGB -> luma (PIL 'L')
            w = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
            gray = (img[..., :3] * w).sum(-1)  # (..., H, W)
        elif img.shape[-1] == 1:
            gray = img[..., 0]
        else:
            gray = img  # already (..., H, W)
        chw = gray.unsqueeze(-3).contiguous()  # (..., 1, H, W)
        lead = chw.shape[:-3]
        flat = chw.reshape(-1, 1, *chw.shape[-2:])
        resized = F.interpolate(
            flat,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        out = resized.reshape(*lead, 1, *self.input_size)
        return (out - self.mean) / self.std

    def _forward_np(self, inputs):
        import numpy as np
        import torch

        dev = next(self.module.parameters()).device
        x = inputs if torch.is_tensor(inputs) else torch.as_tensor(np.asarray(inputs))
        x = x.float().to(dev)
        lead = x.shape[:-3]  # (..., 1, H, W) -> fold
        flat = x.reshape(-1, *x.shape[-3:])
        prec = getattr(self.module, "_deeperfly_precision", "float32")
        dtype = (
            getattr(torch, _TORCH_DTYPE[prec])
            if (prec in _TORCH_DTYPE and dev.type == "cuda")
            else None
        )
        with torch.inference_mode():
            if dtype is not None:
                with torch.autocast(dev.type, dtype=dtype):
                    hm = self.module(flat)
            else:
                hm = self.module(flat)
        hm = hm.float()
        hm = hm.reshape(*lead, *hm.shape[-3:])  # (..., C, Hh, Wh)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return hm.cpu().numpy()

    def predict_heatmaps(self, inputs):
        return self._forward_np(inputs)

    def predict_points(self, inputs, *, method: str = "weighted", radius: int = 2):
        from .inference import heatmap_to_points

        hm = self._forward_np(inputs)
        return heatmap_to_points(hm, method=method, radius=radius)

    def set_precision(self, precision: str) -> None:
        precision = (precision or "float32").lower()
        if precision not in _PRECISIONS:
            raise ValueError(
                f"unknown detector precision {precision!r}; use one of {sorted(_PRECISIONS)}"
            )
        self.module._deeperfly_precision = precision  # type: ignore[attr-defined]

    def device(self) -> str:

        try:
            return str(next(self.module.parameters()).device)
        except (StopIteration, Exception):  # noqa: BLE001
            return "cpu"
