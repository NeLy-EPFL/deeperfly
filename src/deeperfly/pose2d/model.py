"""Stacked-hourglass 2D pose network in JAX (Equinox).

A faithful port of DeepFly2D's ``HourglassNet`` (Newell et al. 2016, as used in
NeLy-EPFL/DeepFly2D ``df2d/model.py``) so the original PyTorch weights can be
converted and run in JAX. Modules are plain Equinox PyTrees; batch norm is
folded into a parameter-only :class:`FrozenBatchNorm` (inference only), so the
whole network is a static PyTree that ``jax.jit`` / ``jax.vmap`` straight over a
batch of camera images.

Each module operates on a single ``(C, H, W)`` image (no batch axis); vectorise
with :func:`jax.vmap`. The network returns one heatmap stack per hourglass; the
last is the prediction. The canonical fly config is
:meth:`HourglassNet.deepfly2d` (2 stacks, 1 block, 19 classes, 128 features).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

EXPANSION = 2  # Bottleneck channel expansion (DeepFly2D uses 2, not torchvision's 4)


def _upsample2(x: Float[Array, "C H W"]) -> Float[Array, "C H2 W2"]:
    """Nearest-neighbour 2x upsampling, matching ``nn.Upsample(scale_factor=2)``."""
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


def _apply(blocks: list, x: Array) -> Array:
    """Run a sequential list of callables."""
    for b in blocks:
        x = b(x)
    return x


def to_dtype(model: eqx.Module, dtype) -> eqx.Module:
    """Cast all floating-point leaves of an Equinox module to ``dtype``.

    The detector runs in float32 (standard for inference, and what the original
    weights are) even though the rest of deeperfly enables x64 for geometry.
    """
    arrays, static = eqx.partition(model, eqx.is_inexact_array)
    arrays = jax.tree_util.tree_map(lambda a: a.astype(dtype), arrays)
    return eqx.combine(arrays, static)


class FrozenBatchNorm(eqx.Module):
    """BatchNorm2d in inference mode: affine with frozen running statistics."""

    weight: Array
    bias: Array
    mean: Array
    var: Array
    eps: float = eqx.field(static=True)

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.mean = jnp.zeros(num_features)
        self.var = jnp.ones(num_features)
        self.eps = eps

    def __call__(self, x: Float[Array, "C H W"]) -> Float[Array, "C H W"]:
        scale = self.weight / jnp.sqrt(self.var + self.eps)
        shift = self.bias - self.mean * scale
        return x * scale[:, None, None] + shift[:, None, None]


class Bottleneck(eqx.Module):
    """Pre-activation bottleneck residual block (bn->relu->conv x3, + shortcut)."""

    bn1: FrozenBatchNorm
    conv1: eqx.nn.Conv2d
    bn2: FrozenBatchNorm
    conv2: eqx.nn.Conv2d
    bn3: FrozenBatchNorm
    conv3: eqx.nn.Conv2d
    downsample: eqx.nn.Conv2d | None

    def __init__(self, inplanes, planes, *, stride=1, downsample=False, key):
        k1, k2, k3, kd = jax.random.split(key, 4)
        self.bn1 = FrozenBatchNorm(inplanes)
        self.conv1 = eqx.nn.Conv2d(inplanes, planes, 1, use_bias=True, key=k1)
        self.bn2 = FrozenBatchNorm(planes)
        self.conv2 = eqx.nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, use_bias=True, key=k2
        )
        self.bn3 = FrozenBatchNorm(planes)
        self.conv3 = eqx.nn.Conv2d(planes, planes * EXPANSION, 1, use_bias=True, key=k3)
        self.downsample = (
            eqx.nn.Conv2d(
                inplanes, planes * EXPANSION, 1, stride=stride, use_bias=True, key=kd
            )
            if downsample
            else None
        )

    def __call__(self, x: Float[Array, "C H W"]) -> Float[Array, "C2 H W"]:
        out = self.conv1(jax.nn.relu(self.bn1(x)))
        out = self.conv2(jax.nn.relu(self.bn2(out)))
        out = self.conv3(jax.nn.relu(self.bn3(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


def _make_residual(inplanes, planes, blocks, *, key) -> tuple[list[Bottleneck], int]:
    """A run of bottlenecks; the first adapts channels (matching DeepFly2D)."""
    keys = jax.random.split(key, blocks)
    downsample = inplanes != planes * EXPANSION
    layers = [Bottleneck(inplanes, planes, downsample=downsample, key=keys[0])]
    inplanes = planes * EXPANSION
    for i in range(1, blocks):
        layers.append(Bottleneck(inplanes, planes, key=keys[i]))
    return layers, inplanes


class Hourglass(eqx.Module):
    """Single hourglass: recursive encoder-decoder with skip connections."""

    hg: list  # [depth] x [3 or 4] residual stacks (level 0 has a 4th = bottom)
    maxpool: eqx.nn.MaxPool2d
    depth: int = eqx.field(static=True)

    def __init__(self, planes, num_blocks, depth=4, *, key):
        self.depth = depth
        self.maxpool = eqx.nn.MaxPool2d(2, stride=2)
        inplanes = planes * EXPANSION
        levels = []
        for i in range(depth):
            n_res = 4 if i == 0 else 3
            res = []
            for _ in range(n_res):
                key, sub = jax.random.split(key)
                stack, _ = _make_residual(inplanes, planes, num_blocks, key=sub)
                res.append(stack)
            levels.append(res)
        self.hg = levels

    def _forward(self, n: int, x: Array) -> Array:
        up1 = _apply(self.hg[n - 1][0], x)
        low1 = _apply(self.hg[n - 1][1], self.maxpool(x))
        low2 = self._forward(n - 1, low1) if n > 1 else _apply(self.hg[n - 1][3], low1)
        low3 = _apply(self.hg[n - 1][2], low2)
        return up1 + _upsample2(low3)

    def __call__(self, x: Float[Array, "C H W"]) -> Float[Array, "C H W"]:
        return self._forward(self.depth, x)


class HourglassNet(eqx.Module):
    """Stacked hourglass network with intermediate supervision."""

    conv1: eqx.nn.Conv2d
    bn1: FrozenBatchNorm
    layer1: list
    layer2: list
    layer3: list
    maxpool: eqx.nn.MaxPool2d
    hg: list  # per stack: Hourglass
    res: list  # per stack: residual stack
    fc_conv: list  # per stack: 1x1 conv
    fc_bn: list  # per stack: FrozenBatchNorm
    score: list  # per stack: 1x1 conv -> num_classes
    fc_: list  # per stack (except last): 1x1 conv
    score_: list  # per stack (except last): 1x1 conv num_classes -> ch
    num_stacks: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_stacks=2,
        num_blocks=1,
        num_classes=19,
        inplanes=64,
        num_feats=128,
        init_stride=2,
        key,
    ):
        self.num_stacks = num_stacks
        keys = iter(jax.random.split(key, 64))

        self.conv1 = eqx.nn.Conv2d(
            3, inplanes, 7, stride=init_stride, padding=3, use_bias=True, key=next(keys)
        )
        self.bn1 = FrozenBatchNorm(inplanes)
        ip = inplanes
        self.layer1, ip = _make_residual(ip, inplanes, num_blocks, key=next(keys))
        self.layer2, ip = _make_residual(ip, ip, num_blocks, key=next(keys))
        self.layer3, ip = _make_residual(ip, num_feats, num_blocks, key=next(keys))
        self.maxpool = eqx.nn.MaxPool2d(2, stride=2)

        ch = num_feats * EXPANSION
        self.hg, self.res, self.fc_conv, self.fc_bn = [], [], [], []
        self.score, self.fc_, self.score_ = [], [], []
        for i in range(num_stacks):
            self.hg.append(Hourglass(num_feats, num_blocks, depth=4, key=next(keys)))
            stack, _ = _make_residual(ch, num_feats, num_blocks, key=next(keys))
            self.res.append(stack)
            self.fc_conv.append(eqx.nn.Conv2d(ch, ch, 1, use_bias=True, key=next(keys)))
            self.fc_bn.append(FrozenBatchNorm(ch))
            self.score.append(
                eqx.nn.Conv2d(ch, num_classes, 1, use_bias=True, key=next(keys))
            )
            if i < num_stacks - 1:
                self.fc_.append(eqx.nn.Conv2d(ch, ch, 1, use_bias=True, key=next(keys)))
                self.score_.append(
                    eqx.nn.Conv2d(num_classes, ch, 1, use_bias=True, key=next(keys))
                )

    @classmethod
    def deepfly2d(cls, *, key) -> HourglassNet:
        """The canonical DeepFly2D configuration (2 stacks, 1 block, 19 joints)."""
        model = cls(
            num_stacks=2,
            num_blocks=1,
            num_classes=19,
            inplanes=64,
            num_feats=128,
            init_stride=2,
            key=key,
        )
        return to_dtype(model, jnp.float32)

    def __call__(self, image: Float[Array, "3 H W"]) -> list[Float[Array, "J Hh Wh"]]:
        image = image.astype(self.conv1.weight.dtype)  # detector runs in its own dtype
        x = jax.nn.relu(self.bn1(self.conv1(image)))
        x = _apply(self.layer1, x)
        x = self.maxpool(x)
        x = _apply(self.layer3, _apply(self.layer2, x))

        out = []
        for i in range(self.num_stacks):
            y = _apply(self.res[i], self.hg[i](x))
            y = jax.nn.relu(self.fc_bn[i](self.fc_conv[i](y)))
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                x = x + self.fc_[i](y) + self.score_[i](score)
        return out

    def heatmaps(self, image: Float[Array, "3 H W"]) -> Float[Array, "J Hh Wh"]:
        """Convenience: the final-stack heatmaps only."""
        return self(image)[-1]
