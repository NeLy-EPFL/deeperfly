"""Stacked-hourglass 2D pose network in PyTorch.

A faithful copy of DeepFly2D's stacked hourglass (NeLy-EPFL/DeepFly2D
``df2d/model.py``) so the original DeepFly2D weights run directly, with no
conversion (load them with :func:`deeperfly.pose2d.weights.load_model`).
Stacked ``(B, V, 3, H, W)`` float inputs in -- the ``V`` views run in parallel and
independent -- final-stack ``(B, V, J, h, w)`` heatmaps out (:func:`predict_heatmaps`);
plain 4D ``(N, 3, H, W)`` is also accepted as the no-view case. The plain detect path
(:func:`deeperfly.pose2d.inference.detect`) instead drives :func:`predict_points`,
which decodes the peaks on-device and returns only those.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Lay the CUDA conv batch out ``channels_last`` (NHWC in memory, same NCHW logical
#: shape) so cuDNN picks its faster Tensor-Core conv kernels. A CUDA-only win;
#: CPU/MPS keep the default contiguous layout regardless. Set ``False`` to force
#: the plain NCHW path (e.g. to A/B the speedup or work around a cuDNN regression).
USE_CHANNELS_LAST = True


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return out + residual


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super().__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        return nn.Sequential(
            *[block(planes * block.expansion, planes) for _ in range(num_blocks)]
        )

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = [self._make_residual(block, num_blocks, planes) for _ in range(3)]
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = self.hg[n - 1][1](F.max_pool2d(x, 2, stride=2))
        low2 = (
            self._hour_glass_forward(n - 1, low1) if n > 1 else self.hg[n - 1][3](low1)
        )
        low3 = self.hg[n - 1][2](low2)
        return up1 + self.upsample(low3)

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Stacked hourglass network (DeepFly2D)."""

    def __init__(
        self,
        block=Bottleneck,
        num_stacks=8,  # the published DeepFly2D weights are "sh8" (8 stacks)
        num_blocks=1,
        num_classes=19,
        inplanes=64,
        num_feats=128,
        init_stride=2,
    ):
        super().__init__()
        self.inplanes = inplanes
        self.num_feats = num_feats
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            3, inplanes, kernel_size=7, stride=init_stride, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True),
            nn.BatchNorm2d(inplanes),
            self.relu,
        )

    def forward(self, x):
        # Standard input is ``(B, V, C, H, W)``: the V views run in parallel and
        # independent, so they're folded into the conv batch here (a future
        # cross-view model would instead keep V and mix across it). Plain 4D
        # ``(N, C, H, W)`` is the degenerate no-V case. Under fp16/bf16 autocast on
        # CUDA the folded batch is laid out ``channels_last`` (NHWC in memory; the
        # logical NCHW shape is unchanged) so cuDNN picks its faster Tensor-Core
        # conv kernels -- ~10% on this net. The float32 path keeps NCHW, where
        # channels_last gives no win (and is a touch slower at small batch).
        lead = x.shape[:-3]
        if x.dim() == 5:
            x = x.reshape(-1, *x.shape[-3:])
        if x.is_cuda and USE_CHANNELS_LAST and torch.is_autocast_enabled():
            x = x.contiguous(memory_format=torch.channels_last)
        out = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer3(self.layer2(x))
        for i in range(self.num_stacks):
            y = self.fc[i](self.res[i](self.hg[i](x)))
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                x = x + self.fc_[i](y) + self.score_[i](score)
        if len(lead) == 2:  # unfold the view axis back out of the batch
            out = [o.reshape(*lead, *o.shape[1:]) for o in out]
        return out


def device() -> str:
    """Best available torch device: NVIDIA CUDA, then Apple Metal (MPS), else CPU.

    On Apple Silicon ``"mps"`` runs the hourglass forward on the GPU via Metal
    Performance Shaders (~6x over CPU for sh8); output matches CPU to float32
    epsilon, so the detector is accelerated on macOS with no setup.

    Returns
    -------
    str
        ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_torch(inputs) -> "torch.Tensor":
    """Coerce ``(N, 3, H, W)`` inputs to a torch tensor, on-device when possible.

    ``inputs`` is usually already a ``torch.Tensor`` on the detector device (so a
    GPU-decoded frame reaches the forward without leaving the GPU) and passes
    straight through. Any other DLPack-capable on-device array is bridged
    zero-copy; host NumPy is copied to a writable tensor.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs
    if hasattr(inputs, "__dlpack__"):  # other on-device array -- zero-copy
        return torch.from_dlpack(inputs)
    return torch.from_numpy(np.array(inputs))  # host array -> writable copy


#: Cache of ``torch.compile``-d models, keyed by ``id(model)`` (a run holds one).
_COMPILED: dict[int, "torch.nn.Module"] = {}


def _forward_fn(model: HourglassNet, dev: "torch.device", batch: int):
    """The model, ``torch.compile``-d for production CUDA runs (else eager).

    ``torch.compile`` roughly halves the eager CUDA forward time (see
    ``dev/bench_video.py``). Gated to CUDA and large batches: on CPU the speedup is
    small, and for tiny test batches the one-off compile latency would dwarf the
    work.
    """
    if dev.type != "cuda" or batch < 16:
        return model
    fn = _COMPILED.get(id(model))
    if fn is None:
        fn = torch.compile(model)
        _COMPILED[id(model)] = fn
    return fn


#: Detector forward precision -> autocast dtype (``None`` = run in float32).
_PRECISIONS = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def set_precision(model: HourglassNet, precision: str = "float32") -> None:
    """Record the forward precision ``predict_heatmaps`` should run the model in.

    ``"float32"`` (default, the reference), ``"float16"`` or ``"bfloat16"`` (CUDA
    autocast). float16 is fastest; bfloat16 trades a touch of that speed for the
    wider exponent range of float32, so it can't overflow. Stored on the model so
    the forward picks it up without threading it through every call.

    Parameters
    ----------
    model
        The detector (the precision is stored on it).
    precision
        ``"float32"``, ``"float16"`` or ``"bfloat16"``.

    Raises
    ------
    ValueError
        On an unknown ``precision`` name.
    """
    precision = (precision or "float32").lower()
    if precision not in _PRECISIONS:
        opts = ", ".join(repr(p) for p in _PRECISIONS)
        raise ValueError(f"unknown detector precision {precision!r}; use one of {opts}")
    # A real detector (nn.Module) carries a __dict__; bare test stubs don't (and
    # never run the real forward), so there's nothing to set on them.
    if hasattr(model, "__dict__"):
        model._deeperfly_precision = precision


def _autocast_dtype(model: HourglassNet, dev: "torch.device"):
    """Autocast dtype for the forward, or ``None`` to run in float32.

    Honors the precision set by :func:`set_precision`, but only on CUDA: fp16
    autocast is where the win is, and CPU/MPS stay on the float32 reference path.
    """
    if dev.type != "cuda":
        return None
    return _PRECISIONS.get(getattr(model, "_deeperfly_precision", "float32"))


def _forward_last(model: HourglassNet, inputs) -> "torch.Tensor":
    """Run the detector and return the final-stack heatmaps, on-device as float32.

    Shared by :func:`predict_heatmaps` (which copies them to host) and
    :func:`predict_points` (which decodes peaks on-device first). Applies the
    ``torch.compile`` gating and autocast precision recorded by
    :func:`set_precision`; the result is upcast to float32 so the heatmap contract
    holds even under fp16/bf16 autocast.
    """
    dev = next(model.parameters()).device
    x = _as_torch(inputs).float().to(dev)
    batch = int(np.prod(x.shape[:-3]))  # folded conv batch (B*V for a 5D input)
    fn = _forward_fn(model, dev, batch)
    dtype = _autocast_dtype(model, dev)
    if dtype is not None:  # fp16/bf16 autocast on CUDA; conv runs low, reductions f32
        with torch.autocast(dev.type, dtype=dtype):
            out = fn(x)[-1]
    else:
        out = fn(x)[-1]
    return out.float()


@torch.inference_mode()
def predict_heatmaps(model: HourglassNet, inputs: np.ndarray) -> np.ndarray:
    """Final-stack heatmaps for ``(B, V, 3, H, W)`` float inputs.

    Parameters
    ----------
    model
        The detector.
    inputs
        Network inputs of shape ``(B, V, 3, H, W)`` (the ``V`` views run in parallel
        and independent); plain 4D ``(N, 3, H, W)`` is also accepted.

    Returns
    -------
    np.ndarray
        The final-stack heatmaps as host NumPy, shaped ``(B, V, J, h, w)`` (or
        ``(N, J, h, w)`` for a 4D input).
    """
    out = _forward_last(model, inputs)
    if out.device.type == "cuda":
        torch.cuda.synchronize()
    return out.cpu().numpy()


@torch.inference_mode()
def predict_points(
    model: HourglassNet,
    inputs: np.ndarray,
    *,
    method: str = "weighted",
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused forward + on-device heatmap decode: ``(points_norm, conf)`` as NumPy.

    The arg-max peak decode (:func:`deeperfly.pose2d.inference.heatmap_to_points`)
    runs on the *same device as the forward*, so only the tiny ``(B, V, J, 2)`` peaks
    and ``(B, V, J)`` confidences leave the GPU -- never the full ``(B, V, J, h, w)``
    heatmap, and never a float64 arg-max over the whole grid on the host. Results
    match the NumPy decode to float32 epsilon; ``method`` / ``radius`` select the
    sub-pixel refinement exactly as the NumPy path does.

    The candidate path (:func:`deeperfly.pose2d.inference.detect_candidates_sequence`)
    still uses :func:`predict_heatmaps`, since it needs the whole heatmap.

    Parameters
    ----------
    model
        The detector.
    inputs
        Network inputs of shape ``(B, V, 3, H, W)`` (or 4D ``(N, 3, H, W)``).
    method, radius
        Sub-pixel refinement options (see
        :func:`~deeperfly.pose2d.inference.refine_peaks`).

    Returns
    -------
    points_norm : np.ndarray
        Normalized ``(B, V, J, 2)`` peaks (``(N, J, 2)`` for a 4D input).
    conf : np.ndarray
        Per-joint confidence of shape ``(B, V, J)`` (``(N, J)`` for a 4D input).
    """
    out = _forward_last(model, inputs)  # (N, J, h, w) on device, float32
    xy, conf = _decode_peaks(out, method=method, radius=radius)
    if out.device.type == "cuda":
        torch.cuda.synchronize()
    return xy.cpu().numpy(), conf.cpu().numpy()


def _decode_peaks(
    heatmaps: "torch.Tensor", *, method: str = "weighted", radius: int = 2
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch port of :func:`deeperfly.pose2d.inference.heatmap_to_points`.

    ``heatmaps`` is ``(*lead, h, w)`` on any device; returns normalized ``(x, y)``
    peaks ``(*lead, 2)`` and raw peak confidence ``(*lead,)``, both on the input's
    device. Mirrors the NumPy decoder's arg-max + sub-pixel refinement so the fused
    GPU path is numerically equivalent.
    """
    *lead, hh, ww = heatmaps.shape
    flat = heatmaps.reshape(-1, hh * ww)  # (M, h*w)
    conf, idx = flat.max(dim=-1)  # (M,) peak value, (M,) flat index
    row, col = idx // ww, idx % ww
    cx, cy = _refine_peaks(
        flat.reshape(-1, hh, ww), row, col, method=method, radius=radius
    )
    xy = torch.stack([cx / ww, cy / hh], dim=-1)  # (M, 2)
    return xy.reshape(*lead, 2), conf.reshape(*lead)


def _refine_peaks(
    hm: "torch.Tensor",
    row: "torch.Tensor",
    col: "torch.Tensor",
    *,
    method: str = "weighted",
    radius: int = 2,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Refine one integer peak per ``(M, h, w)`` heatmap to sub-pixel ``(cx, cy)``.

    The torch counterpart of :func:`deeperfly.pose2d.inference.refine_peaks` (single
    peak per channel): ``"argmax"`` keeps the cell, ``"weighted"`` takes the
    intensity-weighted centroid of the ``(2*radius+1)`` window, ``"taylor"`` does the
    DARK Newton step on the log-heatmap. See that function for the derivation.
    """
    m, hh, ww = hm.shape
    fcol, frow = col.float(), row.float()
    if method == "argmax":
        return fcol, frow
    if method not in ("weighted", "taylor"):
        raise ValueError(f"unknown sub-pixel method {method!r}")
    if method == "taylor" and radius < 2:
        raise ValueError("taylor refinement needs radius >= 2")

    off = torch.arange(-radius, radius + 1, device=hm.device)
    dr, dc = (a.reshape(-1) for a in torch.meshgrid(off, off, indexing="ij"))  # (PP,)
    nr, nc = row[:, None] + dr, col[:, None] + dc  # (M, PP) window cells
    inb = (nr >= 0) & (nr < hh) & (nc >= 0) & (nc < ww)
    flat = hm.reshape(m, hh * ww)
    gidx = nr.clamp(0, hh - 1) * ww + nc.clamp(0, ww - 1)
    patch = flat.gather(1, gidx)  # (M, PP) values around the peak
    drf, dcf = dr.float(), dc.float()

    if method == "weighted":
        w = torch.where(inb, patch.clamp_min(0.0), torch.zeros_like(patch))
        mass = w.sum(-1)
        ok = mass > 0
        denom = torch.where(ok, mass, torch.ones_like(mass))  # avoid 0/0
        offc = torch.where(ok, (w * dcf).sum(-1) / denom, torch.zeros_like(mass))
        offr = torch.where(ok, (w * drf).sum(-1) / denom, torch.zeros_like(mass))
        return fcol + offc, frow + offr

    # "taylor": fit the log-heatmap's local quadratic and Newton-step to its peak.
    p = 2 * radius + 1
    b = torch.log(patch.clamp_min(1e-10)).reshape(m, p, p)  # (M, P_, P_)
    ib = inb.reshape(m, p, p)
    c = radius  # centre tap; b[:, c + i, c + j] is row+i, col+j
    dx = 0.5 * (b[:, c, c + 1] - b[:, c, c - 1])
    dy = 0.5 * (b[:, c + 1, c] - b[:, c - 1, c])
    dxx = 0.25 * (b[:, c, c + 2] - 2 * b[:, c, c] + b[:, c, c - 2])
    dyy = 0.25 * (b[:, c + 2, c] - 2 * b[:, c, c] + b[:, c - 2, c])
    dxy = 0.25 * (
        b[:, c + 1, c + 1]
        - b[:, c + 1, c - 1]
        - b[:, c - 1, c + 1]
        + b[:, c - 1, c - 1]
    )
    det = dxx * dyy - dxy * dxy
    taps = (  # every tap the derivatives touch must be in-bounds
        ib[:, c, c + 1]
        & ib[:, c, c - 1]
        & ib[:, c + 1, c]
        & ib[:, c - 1, c]
        & ib[:, c, c + 2]
        & ib[:, c, c - 2]
        & ib[:, c + 2, c]
        & ib[:, c - 2, c]
        & ib[:, c + 1, c + 1]
        & ib[:, c + 1, c - 1]
        & ib[:, c - 1, c + 1]
        & ib[:, c - 1, c - 1]
    )
    good = taps & (det > 0) & (dxx < 0)  # a real local maximum
    denom = torch.where(good, det, torch.ones_like(det))
    z = torch.zeros_like(det)
    ox = torch.where(
        good, torch.clamp(-(dyy * dx - dxy * dy) / denom, -radius, radius), z
    )
    oy = torch.where(
        good, torch.clamp(-(-dxy * dx + dxx * dy) / denom, -radius, radius), z
    )
    return fcol + ox, frow + oy
