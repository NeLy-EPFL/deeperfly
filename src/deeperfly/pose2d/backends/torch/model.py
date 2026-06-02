"""Stacked-hourglass 2D pose network in PyTorch -- the reference backend.

A faithful copy of DeepFly2D's stacked hourglass (NeLy-EPFL/DeepFly2D
``df2d/model.py``) so the original ``.tar`` weights run directly, with no
conversion (load them with :func:`deeperfly.pose2d.backends.torch.load_model`).
It exposes the same contract as the JAX backend -- stacked ``(N, 3, H, W)`` float
inputs in, final-stack ``(N, J, h, w)`` heatmaps out (:func:`predict_heatmaps`) --
so :func:`deeperfly.pose2d.inference.detect` works with either, and the two can
be benchmarked head to head. JAX is the default (faster on GPU); pick this one
with ``--backend torch``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return out


def device() -> str:
    """Best available torch device: NVIDIA CUDA, then Apple Metal (MPS), else CPU.

    On Apple Silicon ``"mps"`` runs the hourglass forward on the GPU via Metal
    Performance Shaders (~6x over CPU for sh8); output matches CPU to float32
    epsilon. JAX has no comparable Metal path, so this is the only accelerated
    detector backend on macOS.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_torch(inputs) -> "torch.Tensor":
    """Coerce ``(N, 3, H, W)`` inputs to a torch tensor without a writability warning.

    The shared preprocess path stacks on-device, so ``inputs`` is usually a
    ``jax.Array`` (immutable). ``np.array`` materializes a writable host copy --
    ``torch.from_numpy`` on an immutable array warns, and importing a JAX array via
    DLPack yields an inference tensor that the model's autograd path rejects.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs
    return torch.from_numpy(np.array(inputs))  # writable host copy


@torch.inference_mode()
def predict_heatmaps(model: HourglassNet, inputs: np.ndarray) -> np.ndarray:
    """Final-stack heatmaps for ``(N, 3, H, W)`` float inputs (numpy/array in, numpy out)."""
    dev = next(model.parameters()).device
    x = _as_torch(inputs).float().to(dev)
    out = model(x)[-1]
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return out.cpu().numpy()
