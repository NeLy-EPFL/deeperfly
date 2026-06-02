"""Weight I/O for the JAX backend: native checkpoints and the PyTorch bridge.

The Equinox :class:`~deeperfly.pose2d.backends.jax.model.HourglassNet` mirrors
the PyTorch module names one-to-one, so converting the original DeepFly2D weights
is a key-by-key copy with two shape fixups: convolution kernels share PyTorch's
``(out, in, kh, kw)`` layout (no transpose), but Equinox conv biases are
``(out, 1, 1)`` and BatchNorm running statistics map onto :class:`FrozenBatchNorm`.
:func:`convert_state_dict` produces a model that :func:`save_checkpoint` serializes
with Equinox's native format, so the runtime (:func:`load_model`) never imports
torch. Reading the original ``.tar`` lives in the torch backend
(:func:`deeperfly.pose2d.backends.torch.weights.state_dict_from_torch_checkpoint`).
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .model import Bottleneck, HourglassNet


def _block_modules(block: Bottleneck, prefix: str):
    yield block.bn1, f"{prefix}.bn1", "bn"
    yield block.conv1, f"{prefix}.conv1", "conv"
    yield block.bn2, f"{prefix}.bn2", "bn"
    yield block.conv2, f"{prefix}.conv2", "conv"
    yield block.bn3, f"{prefix}.bn3", "bn"
    yield block.conv3, f"{prefix}.conv3", "conv"
    if block.downsample is not None:
        yield block.downsample, f"{prefix}.downsample.0", "conv"


def _param_modules(model: HourglassNet):
    """Yield ``(module, torch_key_prefix, kind)`` for every parameter module.

    The prefixes match the PyTorch ``HourglassNet`` state-dict naming exactly.
    """
    yield model.conv1, "conv1", "conv"
    yield model.bn1, "bn1", "bn"
    for name, layer in (
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
    ):
        for bi, block in enumerate(layer):
            yield from _block_modules(block, f"{name}.{bi}")
    for i, hg in enumerate(model.hg):
        for level, group in enumerate(hg.hg):
            for ri, stack in enumerate(group):
                for bi, block in enumerate(stack):
                    yield from _block_modules(block, f"hg.{i}.hg.{level}.{ri}.{bi}")
    for i, stack in enumerate(model.res):
        for bi, block in enumerate(stack):
            yield from _block_modules(block, f"res.{i}.{bi}")
    for i in range(model.num_stacks):
        yield model.fc_conv[i], f"fc.{i}.0", "conv"
        yield model.fc_bn[i], f"fc.{i}.1", "bn"
    for i, conv in enumerate(model.score):
        yield conv, f"score.{i}", "conv"
    for i, conv in enumerate(model.fc_):
        yield conv, f"fc_.{i}", "conv"
    for i, conv in enumerate(model.score_):
        yield conv, f"score_.{i}", "conv"


def _leaf_targets(module, kind: str) -> list:
    if kind == "conv":
        return [module.weight, module.bias]
    return [module.weight, module.bias, module.mean, module.var]


def _leaf_keys(prefix: str, kind: str) -> list[str]:
    if kind == "conv":
        return [f"{prefix}.weight", f"{prefix}.bias"]
    return [
        f"{prefix}.weight",
        f"{prefix}.bias",
        f"{prefix}.running_mean",
        f"{prefix}.running_var",
    ]


def convert_state_dict(
    state_dict: dict[str, np.ndarray], model: HourglassNet
) -> HourglassNet:
    """Return ``model`` with weights filled from a HourglassNet ``state_dict``.

    ``state_dict`` keys are the native ``HourglassNet`` names (e.g.
    ``conv1.weight``, ``layer1.0.bn1.running_mean``). PyTorch's
    ``num_batches_tracked`` BatchNorm counters have no inference-time analog in
    :class:`FrozenBatchNorm` and are dropped. Raises if a needed key is missing
    or if any remaining key goes unused. ``model.num_stacks`` must match the
    checkpoint (see :func:`deeperfly.pose2d.backends.infer_num_stacks`).
    """
    sd = {
        k: np.asarray(v)
        for k, v in state_dict.items()
        if not k.endswith("num_batches_tracked")
    }
    used: set[str] = set()
    values = []
    for module, prefix, kind in _param_modules(model):
        for target, key in zip(_leaf_targets(module, kind), _leaf_keys(prefix, kind)):
            if key not in sd:
                raise KeyError(f"missing weight {key!r} in state_dict")
            values.append(
                jnp.asarray(sd[key], dtype=target.dtype).reshape(target.shape)
            )
            used.add(key)
    unused = set(sd) - used
    if unused:
        raise KeyError(
            f"unused state_dict keys: {sorted(unused)[:5]}... ({len(unused)} total)"
        )

    def where(m: HourglassNet):
        leaves = []
        for module, _, kind in _param_modules(m):
            leaves.extend(_leaf_targets(module, kind))
        return leaves

    return eqx.tree_at(where, model, replace=values)


def export_state_dict(model: HourglassNet) -> dict[str, np.ndarray]:
    """Inverse of :func:`convert_state_dict`: model -> PyTorch-style state dict.

    Conv biases are returned flat ``(out,)`` (PyTorch layout); BatchNorm stats
    are written as ``running_mean`` / ``running_var``. Useful for round-tripping
    and for cross-checking against a reference implementation.
    """
    sd: dict[str, np.ndarray] = {}
    for module, prefix, kind in _param_modules(model):
        if kind == "conv":
            sd[f"{prefix}.weight"] = np.asarray(module.weight)
            sd[f"{prefix}.bias"] = np.asarray(module.bias).reshape(-1)
        else:
            sd[f"{prefix}.weight"] = np.asarray(module.weight)
            sd[f"{prefix}.bias"] = np.asarray(module.bias)
            sd[f"{prefix}.running_mean"] = np.asarray(module.mean)
            sd[f"{prefix}.running_var"] = np.asarray(module.var)
    return sd


def save_checkpoint(model: HourglassNet, path: str | Path) -> None:
    """Serialize a converted model with Equinox's native format."""
    eqx.tree_serialise_leaves(str(path), model)


def load_model(
    checkpoint: str | Path | None = None,
    *,
    key=None,
    num_stacks: int = HourglassNet.DEFAULT_NUM_STACKS,
    device="auto",
) -> HourglassNet:
    """Build the JAX detector and (optionally) load a native ``.eqx`` checkpoint.

    With ``checkpoint=None`` a freshly initialized model is returned. ``key``
    seeds that initialization; it is irrelevant when loading (the leaves are
    overwritten) and defaults to a fixed seed. ``num_stacks`` must match the
    architecture the checkpoint was saved from (default 8, the published ``sh8``
    config). No torch is needed.

    ``device`` places the (float32) model: ``"auto"`` (default) uses Apple Metal
    via the optional ``jax-mps`` plugin when installed and is a no-op otherwise,
    so CUDA/CPU are untouched; ``"mps"`` forces Metal; ``"cpu"`` keeps it on CPU.
    See :func:`deeperfly.pose2d.backends.jax.to_device`.
    """
    import jax

    from .model import to_device

    def build() -> HourglassNet:
        k = jax.random.PRNGKey(0) if key is None else key
        model = HourglassNet.deepfly2d(key=k, num_stacks=num_stacks)
        if checkpoint is None:
            return model
        return eqx.tree_deserialise_leaves(str(checkpoint), model)

    # When jax-mps makes Metal the default device, build on the CPU first: MLX is
    # float32-only, so Equinox's float64 random weight init can't run on Metal.
    if jax.default_backend() == "mps":
        with jax.default_device(jax.devices("cpu")[0]):
            model = build()
    else:
        model = build()
    return to_device(model, device)
