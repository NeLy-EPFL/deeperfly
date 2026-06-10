"""The detector-model registry: a named model class + weights + input contract.

A pose pipeline can drive several detector models (see
:mod:`deeperfly.pose2d.pathways`): each is described in the config by a
``class`` (a registry key here), the ``weights`` to load, the ``input_size`` it
expects, the ``mean`` it subtracts, and the ``n_out_channels`` it emits. A
:class:`ModelSpec` is the parsed, torch-free description; :func:`load_model`
turns it into a :class:`LoadedModel` that owns the model-specific input
preparation (resize to ``input_size`` + normalize) and forward/decode, wrapping
the torch-free seam in :mod:`deeperfly.pose2d.detector`.

The geometry of preparing an input -- a left-right mirror, a crop -- belongs to
the *pathway* (:class:`~deeperfly.preprocessing.FrameTransform`), not here; a
model only ever sees an already-oriented frame and resizes/normalizes it to its
own input contract. Keeping the resize here (anti-aliased, as the original
DeepFly2D did) means a pathway's mirror/crop never perturbs the pixels a model
is trained on -- only the coordinate decode is mapped back through the pathway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: DeepFly2D subtracts this scalar from the ``[0, 1]`` image.
DEFAULT_MEAN = 0.22
#: DeepFly2D network input ``(height, width)``.
DEFAULT_INPUT_SIZE = (256, 512)
#: DeepFly2D body-side detector channels.
DEFAULT_N_OUT_CHANNELS = 19


@dataclass(frozen=True)
class ModelSpec:
    """A torch-free description of one detector model (parsed from ``[[models]]``).

    Attributes
    ----------
    name
        The model's name, referenced by a pathway's ``model`` key.
    cls
        The registry key selecting the model class (e.g. ``"hourglass"``).
    weights
        Path to a checkpoint, or ``None`` to use the auto-provisioned cache.
    input_size
        The network input ``(height, width)``.
    mean
        Scalar subtracted from the ``[0, 1]`` image after the resize.
    n_out_channels
        Number of output heatmap channels (validated against the weights).
    kwargs
        Extra class-specific construction kwargs.
    """

    name: str
    cls: str
    weights: str | None
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE
    mean: float = DEFAULT_MEAN
    n_out_channels: int = DEFAULT_N_OUT_CHANNELS
    kwargs: dict = field(default_factory=dict)


def _load_hourglass(weights: str | None, **kwargs):
    """Load the DeepFly2D stacked-hourglass detector from a ``.pth`` (or the cache)."""
    from . import detector
    from .download import download_torch_weights

    if weights is not None and not Path(weights).exists():
        raise SystemExit(
            f"no detector checkpoint at {weights}. Remove the model's 'weights' "
            "to use the auto-provisioned cache, or point it at a valid .pth."
        )
    path = weights or download_torch_weights()
    return detector.load_detector(path, **kwargs)


#: ``class`` name -> loader(weights, **kwargs) -> torch module. New detector
#: architectures register here; ``"deepfly2d"`` is an alias for ``"hourglass"``.
MODEL_CLASSES = {
    "hourglass": _load_hourglass,
    "deepfly2d": _load_hourglass,
}


class LoadedModel:
    """A loaded detector model plus its input contract (resize + normalize).

    Wraps the torch module behind :mod:`deeperfly.pose2d.detector` so the
    orchestration in :mod:`deeperfly.pose2d.inference` stays torch-free. The
    image preparation (resize to :attr:`input_size`, subtract :attr:`mean`,
    CHW) is the model's own; a pathway hands it an already-oriented frame.
    """

    def __init__(self, spec: ModelSpec, module):
        self.spec = spec
        self.module = module

    @property
    def input_size(self) -> tuple[int, int]:
        return self.spec.input_size

    @property
    def n_out_channels(self) -> int:
        return self.spec.n_out_channels

    def prepare(self, frames):
        """``(..., H, W, 3)`` frame(s) -> ``(..., 3, Hh, Ww)`` normalized model input.

        Accepts a NumPy array or an on-device torch tensor and keeps it on its
        device (a GPU-decoded frame is resized and normalized on the GPU). The
        resize is bilinear + anti-aliased to match the original DeepFly2D
        skimage resize closely; ``mean`` is subtracted last.
        """
        import torch
        import torch.nn.functional as F

        from .inference import _to_torch_image

        img = _to_torch_image(frames)
        img = img.float() / 255.0 if not torch.is_floating_point(img) else img.float()
        if img.ndim == 2:  # a single grayscale frame -> 3 channels
            img = img.unsqueeze(-1).expand(-1, -1, 3)
        img = img[..., :3]
        chw = img.movedim(-1, -3).contiguous()  # (..., 3, H, W)
        lead = chw.shape[:-3]
        flat = chw.reshape(-1, *chw.shape[-3:])
        resized = F.interpolate(
            flat,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        out = resized.reshape(*lead, *resized.shape[-3:])
        return out - self.spec.mean

    def predict_points(self, inputs, *, method: str = "weighted", radius: int = 2):
        """Fused forward + decode: normalized ``(N, J, 2)`` peaks and ``(N, J)`` conf."""
        from . import detector

        return detector.predict_points(
            self.module, inputs, method=method, radius=radius
        )

    def predict_heatmaps(self, inputs):
        """Final-stack heatmaps ``(N, J, Hh, Ww)`` (host NumPy) for the candidate path."""
        from . import detector

        return detector.predict_heatmaps(self.module, inputs)

    def set_precision(self, precision: str) -> None:
        """Set the forward precision (``float32``/``float16``/``bfloat16``)."""
        from . import detector

        detector.set_precision(self.module, precision)

    def device(self) -> str:
        """The device the model's parameters live on."""
        from . import detector

        return detector.detector_device(self.module)


def load_model(spec: ModelSpec) -> LoadedModel:
    """Build a :class:`LoadedModel` from a :class:`ModelSpec`.

    Looks up ``spec.cls`` in :data:`MODEL_CLASSES`, loads the weights, and
    validates that the loaded module emits ``spec.n_out_channels`` heatmaps.

    Raises
    ------
    SystemExit
        If ``spec.cls`` is not a known model class, or an explicit ``weights``
        path does not exist.
    ValueError
        If the loaded module's output-channel count disagrees with
        ``spec.n_out_channels``.
    """
    loader = MODEL_CLASSES.get(spec.cls)
    if loader is None:
        raise SystemExit(
            f"unknown model class {spec.cls!r} for model {spec.name!r}; "
            f"known classes: {sorted(MODEL_CLASSES)}"
        )
    module = loader(spec.weights, **spec.kwargs)
    num_classes = getattr(module, "num_classes", None)
    if num_classes is not None and int(num_classes) != int(spec.n_out_channels):
        raise ValueError(
            f"model {spec.name!r} declares n_out_channels={spec.n_out_channels} "
            f"but its weights emit {num_classes} channels"
        )
    return LoadedModel(spec, module)
