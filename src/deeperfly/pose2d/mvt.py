"""The multiview transformer: 38 channels in every view, computed from all views at once.

Two detector kinds already live here. :mod:`deeperfly.pose2d.model` is the shipped
retired 19-channel detector, one side of the animal per pass. :mod:`deeperfly.pose2d.hrnet` is
the dense-38 HRNet: every point in every view, but each view predicted **alone**. This is
the third and it is the first that is not a per-view function at all -- the views of one
frame are encoded together, and a joint that only one camera can see informs the cameras
that cannot. Trained in the ``dfpose`` repo as a Lightning Pose 3D multiview transformer;
this module exists so its weights are runnable from a config.

**Why the pathway plan already fits.** ``[[pose2d.pathways]]`` looks like a per-view
abstraction and is not: :func:`deeperfly.pose2d.inference.detect_sequence` stacks all of
one model's pathways for the same frame into ``(T, Pm, 3, H, W)`` and hands that to
``predict_points``. That is exactly this network's input -- batch of frames, views of each
frame -- so the plan needs no new concept. One pathway per camera, all naming this model.

**Why view order does not matter.** The checkpoint has ZERO per-view parameters
(``view_embed=off``): views are told apart by which attention blocks may see across them,
not by a per-camera embedding. The function is therefore permutation-equivariant over
views by construction, verified to ~5e-08 on the shipped artifact, and V is free -- a
seven-camera rig declares seven pathways and nothing else changes. No camera name is read
anywhere in this file.

Four parts of the contract differ from BOTH other detectors, and each is a wrong answer
rather than a crash:

**1. Views ride the CHANNEL axis of the output.** ``forward`` returns
``(batch, V * K, H, W)``, not the ``(B*V, K, H, W)`` the head produced one step earlier.
Same element count, so reading it the other way attributes every view's channels to the
wrong view. Channel ``c`` is view ``c // K``, keypoint ``c % K``.

**2. The decode is a soft-argmax over an UPSAMPLED heatmap**, not an argmax with a
parabolic refinement. The field is upsampled twice to the full network input -- 88x152 to
352x608 on the shipped padded artifact, 64x128 to 256x512 on an unpadded one -- softmaxed
with temperature 1000, and reduced by spatial expectation, then shifted by -1.5 and by the
margin. Substituting an argmax decode moves every point.

**3. The peaks are in the PURE-SCALE convention, not the half-pixel one.** LP wrote the
labels this model was trained on as ``model_px * crop_wh / model_wh``, with no half-pixel
term. deeperfly's :class:`~deeperfly.preprocessing.Resize` inverts with
``x = (x' + 0.5) / s - 0.5``, which is the geometrically correct inverse of a cv2 resize
and is what the dense-38 HRNet needs (its labels were written that way). Applying it here
biases every point by ``0.5 * (crop_w / model_w - 1)`` -- about 0.44 px in x and 0.50 px in
y on this rig, a uniform skeleton shift that looks like a calibration error. So the module
declares ``peak_convention`` and the pathway inversion honors it.

**4. The normalization is carried by the artifact, per input plane** (the shipped
one-plane artifacts record ``mean = [0.0]``, ``std = [1.0]``, i.e. nothing is subtracted),
and the resize is ``cv2.INTER_AREA``. Not the shared
:meth:`~deeperfly.pose2d.models.LoadedModel.prepare`, whose antialiased bilinear differs
from INTER_AREA by up to 10/255 on ~35% of pixels -- enough to move 1.8% of cells more
than a model pixel and the worst by 65. So this module owns its input preparation too.

Everything above is READ FROM THE ARTIFACT rather than hard-coded: ``scripts/
export_mvt_weights.py`` in dfpose writes the architecture, the channel names, the
normalization and the conventions into the weights file, and :func:`load_mvt` refuses one
whose fields it does not recognize. ``dfpose/tests/test_mvt_parity.py`` holds this
implementation to a reference forward recorded from Lightning Pose itself.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("deeperfly")

#: Formats this loader runs. ``-2`` is a ONE-PLANE artifact: the corpus is monochrome, so
#: the three-channel ``-1`` its patch-embedding stem was folded from carried three copies
#: of one filter in two thirds of that convolution's input weights. The fold is exact (see
#: dfpose's ``scripts/gray_stem.py``), so a ``-2`` is the same function as the ``-1`` it
#: came from rather than a retrained model -- which is why dropping ``-1`` here costs no
#: accuracy, only the ability to load an artifact nobody should still be running.
#:
#: The version is the guard, and it has to be. A ``-1`` fed one plane is a shape error
#: PyTorch would raise, but a ``-2`` whose scalar normalization was applied as if it were
#: ImageNet's per-channel one would RUN and be quietly wrong. So the channel count is read
#: from the artifact's own ``normalization.mean`` and cross-checked against the stem it
#: actually carries.
ARTIFACT_FORMATS: tuple[str, ...] = ("deeperfly-mvt-2",)

#: Soft-argmax temperature. Not in the state dict -- LP holds it as a plain attribute
#: (``self.temperature = torch.tensor(1000.0)``), so it is a property of the decode rather
#: than of the weights, and it has to be reproduced rather than loaded.
SOFTARGMAX_TEMPERATURE: float = 1000.0

#: Confidence is the softmax mass within ``floor(sigma * num_stds)`` cells of the peak --
#: LP's ``evaluate_heatmaps_at_location`` defaults, i.e. a 5x5 window at sigma 1.25.
CONF_SIGMA: float = 1.25
CONF_NUM_STDS: int = 2

#: kornia's ``_get_pyramid_gaussian_kernel``, copied as the literal it is. The upsample in
#: LP's decode is a bicubic 2x followed by this blur; reproducing "a gaussian blur" instead
#: of this exact kernel would move the soft-argmax.
_PYRAMID_KERNEL: tuple[tuple[float, ...], ...] = (
    (1.0, 4.0, 6.0, 4.0, 1.0),
    (4.0, 16.0, 24.0, 16.0, 4.0),
    (6.0, 24.0, 36.0, 24.0, 6.0),
    (4.0, 16.0, 24.0, 16.0, 4.0),
    (1.0, 4.0, 6.0, 4.0, 1.0),
)

#: Channels decoded at once. The decode upsamples the field 4x on each axis -- 88x152 to
#: 352x608 on the shipped padded artifact -- which is 16x the cells: all 304 channels of an
#: 8-view frame at once is ~260 MB for the upsampled field and as much again for the
#: softmax (~160 MB apiece on an unpadded 64x128 one). Chunking is exact -- every channel is
#: decoded independently -- and keeps the peak bounded regardless of view count.
DECODE_CHUNK: int = 38

#: Grayscale value the MARGIN of a padded input is filled with, before normalization.
#:
#: A constant and not ``BORDER_REPLICATE``: the margin's only job is to be the same at
#: training time and at inference time, and a replicated edge is not -- it smears whatever
#: happens to touch the border, so a leg leaving the frame grows a streaked copy of itself
#: that a detector can learn to chase. A flat fill carries no content to chase, which
#: leaves the visible part of the limb (and the other views) as the only evidence for where
#: the joint went. Recorded in the artifact as ``arch.hm_margin_fill`` so a checkpoint
#: trained against a different fill cannot be served with this one.
MARGIN_FILL_U8: int = 0


def _torch():
    import torch

    return torch


# ---------------------------------------------------------------------------------------
# The network. Submodule names mirror the training-side definition (a HuggingFace
# ``ViTModel`` wrapped by LP's ``VisionEncoder``, plus LP's heatmap head) so a checkpoint
# loads with ``strict=True`` and no key translation. Two declarations of one network is a
# drift hazard; the parity test against a recorded LP forward is what makes it safe.
# ---------------------------------------------------------------------------------------


def _build_modules(arch: dict[str, Any]):
    torch = _torch()
    nn = torch.nn

    dim = int(arch["embed_dim"])
    n_heads = int(arch["num_heads"])
    eps = float(arch["layer_norm_eps"])
    mlp_dim = int(arch["mlp_hidden_dim"])
    patch = int(arch["patch_size"])
    act_name = str(arch["hidden_act"])
    if act_name != "gelu":
        raise SystemExit(
            f"unsupported hidden_act {act_name!r}; this module implements gelu"
        )
    qkv_bias = bool(arch.get("qkv_bias", True))

    # One plane, and one normalization constant per plane (the artifact carries them;
    # ImageNet), 1 for a folded `-2`. Read from the artifact rather than assumed, so the
    # two cannot be silently interchanged.
    in_ch = int(arch.get("in_channels", 3))

    class PatchEmbeddings(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        def forward(self, x):
            return self.projection(x).flatten(2).transpose(1, 2)

    class Embeddings(nn.Module):
        """Patch embedding + interpolated position embedding.

        The cls token is a parameter and plays no part: LP drops it immediately after the
        embeddings, so no view's cls token ever enters a global attention block. It is
        declared here only so the load stays strict, and dropped at the end of forward for
        the same reason it is dropped there.
        """

        def __init__(self) -> None:
            super().__init__()
            self.patch_embeddings = PatchEmbeddings()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            n_pos = (
                int(arch["pretrain_grid_hw"][0]) * int(arch["pretrain_grid_hw"][1]) + 1
            )
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_pos, dim))

        def interpolate_pos_encoding(self, h_tok: int, w_tok: int):
            """14x14 learned positions -> this input's token grid.

            Bicubic, ``align_corners=False``, and **no antialias** -- HuggingFace's
            ``ViTEmbeddings.interpolate_pos_encoding``. The cls row is carried through the
            concatenation it does and then discarded, which is not the same as never
            adding it: keeping the ordering identical is what makes the patch rows line up.
            """
            patch_pos = self.position_embeddings[:, 1:]
            grid = int(round(math.sqrt(patch_pos.shape[1])))
            p = patch_pos.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
            p = torch.nn.functional.interpolate(
                p, size=(h_tok, w_tok), mode="bicubic", align_corners=False
            )
            p = p.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((self.position_embeddings[:, :1], p), dim=1)

        def forward(self, x):
            h_tok, w_tok = x.shape[-2] // patch, x.shape[-1] // patch
            tokens = self.patch_embeddings(x)
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            tokens = tokens + self.interpolate_pos_encoding(h_tok, w_tok)
            return tokens[:, 1:]  # LP's `[:, 1:]`: the blocks never see a cls token

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.o_proj = nn.Linear(dim, dim)

        def forward(self, x, attn_mask=None):
            b, n, _ = x.shape
            head_dim = dim // n_heads

            def split(t):
                return t.view(b, n, n_heads, head_dim).transpose(1, 2)

            q, k, v = (
                split(self.q_proj(x)),
                split(self.k_proj(x)),
                split(self.v_proj(x)),
            )
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=head_dim**-0.5
            )
            out = out.transpose(1, 2).reshape(b, n, dim)
            return self.o_proj(out)

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, mlp_dim)
            self.fc2 = nn.Linear(mlp_dim, dim)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Layer(nn.Module):
        """Pre-norm block: attention around ``layernorm_before``, MLP around ``after``."""

        def __init__(self) -> None:
            super().__init__()
            self.attention = Attention()
            self.layernorm_before = nn.LayerNorm(dim, eps=eps)
            self.layernorm_after = nn.LayerNorm(dim, eps=eps)
            self.mlp = Mlp()

        def forward(self, x, attn_mask=None):
            x = x + self.attention(self.layernorm_before(x), attn_mask)
            return x + self.mlp(self.layernorm_after(x))

    class VisionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = Embeddings()
            self.layers = nn.ModuleList([Layer() for _ in range(int(arch["depth"]))])
            self.layernorm = nn.LayerNorm(dim, eps=eps)

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = VisionEncoder()

    final_softmax = bool(arch.get("final_softmax", True))

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # PixelShuffle(2) doubles the token grid and quarters the channels; one
            # stride-2 ConvTranspose2d doubles it again, so the field is the token grid x4
            # whatever its size: 22x38 tokens -> 88x152 cells on the shipped padded
            # artifact, 16x32 -> 64x128 on an unpadded one.
            self.upsampling_layers = nn.Sequential(
                nn.PixelShuffle(2),
                nn.ConvTranspose2d(
                    dim // 4,
                    int(arch["num_keypoints"]),
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    output_padding=(1, 1),
                ),
            )

        def forward(self, x):
            hm = self.upsampling_layers(x)
            if final_softmax:
                # A spatial softmax at temperature 1 IS part of this head: LP's
                # `final_softmax` defaults to True and nothing overrides it, so the
                # network's output is a normalized map (peaks ~0.03 over the field) and not
                # logits. Omitting it leaves the decode's own temperature-1000 softmax to
                # act on raw logits, which moves every point -- measured 251 px at the
                # worst, i.e. half the frame.
                hm = _spatial_softmax(hm, 1.0)
            return hm

    return Backbone(), Head()


class MvtNet:
    """Holder for the two submodules, assembled as an ``nn.Module`` by :func:`load_mvt`."""


def _make_net(arch: dict[str, Any]):
    torch = _torch()
    nn = torch.nn
    backbone, head = _build_modules(arch)
    scopes = list(str(arch["attn_scopes"]))

    class Net(nn.Module):
        #: This module decodes its own heatmaps (soft-argmax, upsampled field) and
        #: prepares its own inputs (INTER_AREA + the artifact's own normalization).
        owns_decode = True
        owns_prepare = True
        #: The preparation is PIL and cv2, i.e. host libraries, so a device tensor handed
        #: to it is copied back down first. See
        #: :attr:`deeperfly.pose2d.models.LoadedModel.prepares_on_host`.
        prepares_on_host = True
        #: The first thing the preparation does is make the frame grayscale, so a
        #: one-channel frame is not a loss -- it is the frame with the redundant copies of
        #: the luma left out. See
        #: :attr:`deeperfly.pose2d.models.LoadedModel.accepts_gray`.
        accepts_gray = True
        #: The views of a frame are encoded TOGETHER, so the ``V`` axis is not spare batch:
        #: a view's output depends on which others were in the tensor. See
        #: :attr:`deeperfly.pose2d.models.LoadedModel.joint_views`.
        joint_views = True
        #: Whether a peak may land outside the REPORTED frame. Overridden per artifact by
        #: `load_mvt`: false when the field is upsampled to exactly the reported frame (every
        #: artifact through r27), where a joint the crop cuts off saturates toward the border
        #: instead of leaving the box; true when the artifact declares `hm_margin_px`, which
        #: pads the network's input so the field covers ground outside the reported frame and
        #: a coordinate beyond [0, 1] is a location rather than an error -- the same contract
        #: the dense HRNet's padded field has.
        padded_field = False
        #: LP's labels were written with no half-pixel term; see the module docstring.
        peak_convention = "pure-scale"

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.arch = arch
            self.attn_scopes = scopes
            self.num_classes = int(arch["num_keypoints"])
            self.point_names: list[str] = []
            self.input_hw: tuple[int, int] = (0, 0)
            self.norm_mean: tuple[float, ...] = ()
            self.norm_std: tuple[float, ...] = ()

        # -- encode ----------------------------------------------------------------
        def encode(self, images):
            """``(B, V, 3, H, W)`` -> ``(B*V, dim, H_tok, W_tok)``, batch-major in V.

            The alternating schedule is a reshape and nothing else -- no parameters, no
            masks, exactly as in VGGT's aggregator:

                view-local  ``(B*V, patch, dim)``        a view attends to itself
                global      ``(B, V*patch, dim)``        every view sees every other

            The ``(B*V)`` axis is batch-major (item 0's views, then item 1's), which is
            what makes the reshape to ``(B, V*patch, dim)`` group a frame's own views.
            Getting that backwards would mix views across frames and still run.
            """
            b, v = images.shape[0], images.shape[1]
            flat = images.reshape(-1, *images.shape[2:])
            h = self.backbone.vision_encoder.embeddings(flat)
            n_patch, dim = h.shape[1], h.shape[2]

            is_global = False
            for layer, scope in zip(
                self.backbone.vision_encoder.layers, self.attn_scopes
            ):
                want = scope == "g"
                if want != is_global:
                    h = (
                        h.reshape(b, v * n_patch, dim)
                        if want
                        else h.reshape(b * v, n_patch, dim)
                    )
                    is_global = want
                h = layer(h)
            if is_global:
                h = h.reshape(b * v, n_patch, dim)
            h = self.backbone.vision_encoder.layernorm(h)

            patch = int(self.arch["patch_size"])
            h_tok, w_tok = images.shape[-2] // patch, images.shape[-1] // patch
            if h_tok * w_tok != n_patch:
                raise RuntimeError(
                    f"{n_patch} tokens for a {images.shape[-2]}x{images.shape[-1]} input "
                    f"at patch {patch}: expected {h_tok * w_tok}"
                )
            return h.reshape(b * v, h_tok, w_tok, dim).permute(0, 3, 1, 2)

        def forward(self, images):
            """``(B, V, 3, H, W)`` -> ``(B, V * num_keypoints, Hm, Wm)``.

            The channel-axis layout is LP's, kept rather than improved so the parity test
            compares like with like. Callers should use :meth:`heatmaps_by_view`.
            """
            v = images.shape[1]
            hm = self.head(self.encode(images))  # (B*V, K, Hm, Wm)
            return hm.reshape(images.shape[0], v * hm.shape[1], *hm.shape[-2:])

        def heatmaps_by_view(self, images):
            """``(B, V, 3, H, W)`` -> ``(B, V, K, Hm, Wm)``, the layout callers want."""
            hm = self.forward(images)
            k = self.num_classes
            return hm.reshape(hm.shape[0], hm.shape[1] // k, k, *hm.shape[-2:])

    return Net()


# ---------------------------------------------------------------------------------------
# The decode. LP's `run_subpixelmaxima`, reproduced: upsample the field to the input size,
# spatial softmax at temperature 1000, spatial expectation, then a fixed -1.5 shift.
# ---------------------------------------------------------------------------------------


def _pyr_up(x):
    """One 2x upsample: bicubic to twice the size, then the pyramid gaussian blur.

    LP's own `upsample`, which is kornia's ``pyrup`` "with better defaults": the
    interpolation is **bicubic** with ``align_corners=False`` (LP's comment says the
    ``align_corners`` choice is what makes the -1.5 offset below hold), and the blur is
    zero-padded. The kernel is symmetric, so correlation and convolution agree and
    kornia's ``behaviour="corr"`` needs no flip here.
    """
    torch = _torch()
    F = torch.nn.functional
    h, w = x.shape[-2], x.shape[-1]
    up = F.interpolate(x, size=(h * 2, w * 2), mode="bicubic", align_corners=False)
    c = up.shape[1]
    kernel = torch.tensor(_PYRAMID_KERNEL, dtype=up.dtype, device=up.device) / 256.0
    kernel = kernel.expand(c, 1, 5, 5)
    return F.conv2d(up, kernel, padding=2, groups=c)


def _spatial_softmax(x, temperature: float):
    torch = _torch()
    b, c, h, w = x.shape
    flat = x.view(b, c, -1)
    return torch.nn.functional.softmax(flat * temperature, dim=-1).view(b, c, h, w)


def _spatial_expectation(prob):
    """Expectation of the pixel coordinate under ``prob``, returned as ``(x, y)``.

    kornia's ``spatial_expectation2d(..., normalized_coordinates=False)``: the grid is
    ``0..W-1`` in x and ``0..H-1`` in y, so the result is in cells of the tensor it was
    given -- which, after the upsampling above, is model input pixels.
    """
    torch = _torch()
    b, c, h, w = prob.shape
    pos_x = torch.arange(w, dtype=prob.dtype, device=prob.device).view(1, 1, 1, w)
    pos_x = pos_x.expand(1, 1, h, w).reshape(-1)
    pos_y = torch.arange(h, dtype=prob.dtype, device=prob.device).view(1, 1, h, 1)
    pos_y = pos_y.expand(1, 1, h, w).reshape(-1)
    flat = prob.view(b, c, -1)
    ex = (pos_x * flat).sum(-1, keepdim=True)
    ey = (pos_y * flat).sum(-1, keepdim=True)
    return torch.cat([ex, ey], dim=-1).view(b, c, 2)


def _confidence_at(prob, locs):
    """Softmax mass in the 5x5 window around each peak -- LP's
    ``evaluate_heatmaps_at_location`` at its defaults.

    Confidence is a WINDOW sum and not the peak value because the targets were rendered
    with sigma 1.25, so a correct prediction spreads its mass over neighbours; reading the
    single cell under-reports it by a factor that varies with how peaked the heatmap is.
    """
    torch = _torch()
    reach = int(np.floor(CONF_SIGMA * CONF_NUM_STDS))
    padded = torch.nn.functional.pad(prob, (reach, reach, reach, reach))
    b, c = prob.shape[0], prob.shape[1]
    i = torch.arange(b, device=prob.device).view(-1, 1)
    j = torch.arange(c, device=prob.device).view(1, -1)
    # `.long()` truncates toward zero, as LP's `.type(torch.int64)` does. Rounding here
    # would shift the window by a cell on half the points.
    cy = locs[..., 1].long() + reach
    cx = locs[..., 0].long() + reach
    total = torch.zeros(b, c, dtype=prob.dtype, device=prob.device)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            total = total + padded[i, j, cy + dy, cx + dx]
    return total


def decode_points(
    heatmaps,
    input_hw: tuple[int, int],
    downsample_factor: int,
    *,
    margin: int = 0,
):
    """``(B, C, Hm, Wm)`` heatmaps -> input-normalized ``(B, C, 2)`` peaks and conf.

    ``input_hw`` is the **reported frame** -- the coordinate system the returned points are
    normalized into, and the one the pathway inverts. ``margin`` is how many model pixels of
    field lie outside it on every side, so the upsampled field is
    ``(h + 2 * margin, w + 2 * margin)`` and a returned coordinate outside ``[0, 1]`` is a
    joint outside the reported frame rather than an error. With ``margin = 0`` (the shipped
    r27 artifact) the field spans the reported frame exactly and the soft-argmax saturates
    against its border, which is the behavior this decode has always had.

    Channels are decoded independently, so this chunks them (see :data:`DECODE_CHUNK`):
    upsampling to the input size costs 16x the cells, and an 8-view frame is 304 channels.

    The ``-1.5`` is LP's grid-offset correction for ``downsample_factor=2`` and is not a
    fudge: two ``align_corners=False`` upsamples put the cell centers half a cell off at
    each step, and the constant is what puts the expectation back on the input's pixel
    grid. It is indexed by the factor, so a checkpoint trained at another factor gets its
    own constant rather than this one.
    """
    torch = _torch()
    offsets = {1: 0.5, 2: 1.5, 3: 2.5}
    if downsample_factor not in offsets:
        raise SystemExit(
            f"downsample_factor {downsample_factor} has no grid-offset correction; "
            f"known: {sorted(offsets)}"
        )
    h_rep, w_rep = input_hw
    if int(margin) < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    # The FIELD's extent in model px. The upsample below must land on exactly this, which is
    # the one check that catches a margin disagreeing with the checkpoint it decodes: a
    # 48 px margin decoded as 0 puts every point 48 px up and left, a whole antenna's worth,
    # with nothing else to notice.
    h_in, w_in = h_rep + 2 * int(margin), w_rep + 2 * int(margin)
    xs, cs = [], []
    for lo in range(0, heatmaps.shape[1], DECODE_CHUNK):
        hm = heatmaps[:, lo : lo + DECODE_CHUNK].float()
        for _ in range(downsample_factor):
            hm = _pyr_up(hm)
        if hm.shape[-2:] != (h_in, w_in):
            raise RuntimeError(
                f"the upsampled field is {tuple(hm.shape[-2:])}, not the "
                f"{(h_in, w_in)} this reported frame {(h_rep, w_rep)} plus margin "
                f"{margin} describes; the decode's coordinates would not be model pixels"
            )
        prob = _spatial_softmax(hm, SOFTARGMAX_TEMPERATURE)
        pts = _spatial_expectation(prob)
        # ORDER MATTERS. LP evaluates the confidence window at the RAW expectation and
        # applies the grid-offset correction afterwards, so the window is centered on the
        # cell the peak actually occupies in this tensor. Subtracting first moves the 5x5
        # window one to two cells off: harmless on a broad peak and not at all on a sharp
        # one -- measured up to 0.37 of a confidence that tops out near 0.1.
        cs.append(_confidence_at(prob, pts))
        # `- margin` moves the origin from the padded field's top-left corner to the
        # REPORTED frame's, so x < 0 means "left of the frame". It comes after
        # `_confidence_at` for the reason stated there: the window has to be centered on the
        # cell the peak occupies in THIS tensor, which is still the padded one.
        xs.append(pts - offsets[downsample_factor] - float(margin))
    pts = torch.cat(xs, dim=1)
    conf = torch.cat(cs, dim=1)
    # Normalized into the REPORTED frame, as every other detector class here returns.
    # Values outside [0, 1] are a joint outside that frame. With `margin > 0` they are
    # meaningful and must not be clipped -- that is what the padded field exists to
    # represent, exactly as in `hrnet.cells_to_input_normalized`. With `margin == 0` the
    # field spans the frame and the soft-argmax can only saturate toward the border, so the
    # values stay within about a pixel and a half of it.
    norm = torch.tensor([w_rep, h_rep], dtype=pts.dtype, device=pts.device)
    return pts / norm, conf


def cells_to_input_normalized(model, cells):
    """Sub-pixel FIELD cells ``(..., 2)`` as ``(cx, cy)`` -> input-normalized ``(x, y)``.

    The piece of this decode's geometry that :func:`decode_points` cannot expose, because
    it reduces a whole channel to ONE soft-argmax: a caller keeping several peaks per
    channel (the top-K candidate path, :func:`deeperfly.pictorial.peak_candidates`) has to
    place the cells itself, and the shared ``(c + 0.5) / W_field`` convention is wrong here
    twice over -- the field is the PADDED input's, not the reported frame's, so it is off by
    both the margin and the ``(w + 2m) / w`` scale.

    A cell ``c`` sits at padded-input pixel ``2**downsample_factor * c``: each
    ``align_corners=False`` upsample puts cell centers half a cell off, which for two steps
    lands cell ``c`` at index ``4c + 1.5`` -- and ``decode_points`` subtracts exactly that
    same 1.5. Subtracting ``hm_margin_px`` then moves the origin from the padded field's
    corner to the REPORTED frame's, so a negative coordinate means "left of / above the
    frame" rather than an error, exactly as it does there.

    Checked against :func:`decode_points` itself (``tests/test_mvt_candidates.py``): a lone
    spike at any cell two or more in from the field's border decodes to this to within
    1e-5 model px. Only the two border cells of each edge disagree -- the outermost by
    0.50 px and the next by 0.014 -- which is the bicubic upsample having no data past the
    edge to spread its mass into, so its expectation is pulled inward. That is the
    upsample clamping, not a disagreement about the mapping, and it is bounded by half a
    model pixel where a candidate is allowed 15 (:data:`deeperfly.pictorial.DEFAULT_INLIER_PX`).

    Values outside ``[0, 1]`` are meaningful and must not be clipped -- representing them is
    what the padded field is for (:attr:`deeperfly.pose2d.models.LoadedModel.padded_field`).
    """
    cells = np.asarray(cells, dtype=float)
    scale = float(2 ** int(model.downsample_factor))
    margin = float(getattr(model, "hm_margin_px", 0) or 0)
    h_rep, w_rep = model.input_hw
    return np.stack(
        [
            (cells[..., 0] * scale - margin) / float(w_rep),
            (cells[..., 1] * scale - margin) / float(h_rep),
        ],
        axis=-1,
    )


def points_from_heatmaps(model, heatmaps):
    """Already-computed heatmaps -> the points :func:`predict_points` would have returned.

    Same decode, same ``conf_floor``, so the candidate path's arg-max is the production
    arg-max rather than a second opinion about it. Takes a device tensor when it can get
    one: the decode upsamples every channel 16x, which is the reason
    :func:`predict_points_and_heatmaps` exists.
    """
    torch = _torch()
    hm = (
        heatmaps
        if isinstance(heatmaps, torch.Tensor)
        else torch.as_tensor(np.asarray(heatmaps))
    )
    lead = tuple(hm.shape[:-3])
    flat = hm.reshape(-1, *hm.shape[-3:]).float()
    with torch.inference_mode():
        xy, conf = decode_points(
            flat,
            model.input_hw,
            model.downsample_factor,
            margin=int(getattr(model, "hm_margin_px", 0) or 0),
        )
        floor = float(getattr(model, "conf_floor", 0.0))
        if floor > 0.0:
            xy = xy.masked_fill(conf[..., None] < floor, float("nan"))
    k = xy.shape[-2]
    return (
        xy.reshape(*lead, k, 2).cpu().numpy().astype(np.float32),
        conf.reshape(*lead, k).cpu().numpy().astype(np.float32),
    )


def predict_points_and_heatmaps(model, inputs):
    """One forward -> ``(points, conf, heatmaps)``: production points AND the host field.

    The candidate path needs both, and running :func:`predict_points` and
    :func:`predict_heatmaps` in turn would forward the network twice. It also keeps the
    arg-max decode on the DEVICE, where the 16x upsample belongs -- decoding a host copy
    costs more than the network does.
    """
    torch = _torch()
    x = (
        inputs
        if isinstance(inputs, torch.Tensor)
        else torch.as_tensor(np.asarray(inputs))
    )
    dev = next(model.parameters()).device
    x = x.to(device=dev, dtype=torch.float32)
    if x.ndim == 4:
        x = x.unsqueeze(0)
    with torch.inference_mode():
        hm = model.heatmaps_by_view(x)  # (B, V, K, Hm, Wm)
        xy, conf = points_from_heatmaps(model, hm)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return xy, conf, hm.float().cpu().numpy()


# ---------------------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------------------


def prepare_images(
    frames, input_hw: tuple[int, int], mean, std, device, *, margin: int = 0
):
    """Oriented ``(T, H, W, 3)`` uint8 frames -> ``(T, C, h, w)`` normalized model input.

    ``input_hw`` is the **reported frame**: the frame the crop is resized to and the one the
    returned coordinates are normalized against. ``margin`` then pads that resized frame by
    the given number of model pixels on every side with :data:`MARGIN_FILL_U8`, so the
    network's actual input is ``(h + 2m, w + 2m)`` and its field covers ground the reported
    frame does not. The resize is unchanged by the margin, which is the point: the animal
    lands on exactly the pixels it lands on today, and the margin is added around it, so a
    padded checkpoint and an unpadded one share a coordinate system.

    The pad happens on the uint8 plane, BEFORE normalization, so the fill is a grayscale
    value and not a post-normalization number that would change meaning with the artifact's
    mean and std.

    Reproduces the training pipeline rather than an equivalent-looking one, because the
    difference was measured and it is not small: the exporter wrote PNGs via PIL's
    ``convert("L")`` and ``cv2.INTER_AREA``, and torch's antialiased bilinear differs from
    INTER_AREA by up to 10/255 on ~35% of pixels -- 1.8% of cells moved more than a model
    pixel and the worst moved 65.

    **This costs a device round trip**, because PIL and cv2 are host libraries and the
    pathway hands over an on-device window. One copy down for the whole window and one
    tensor back up, not per frame. RGB->L is applied after the pathway's crop rather than
    before it, which is exactly equivalent: the conversion is pointwise, so it commutes
    with a slice.
    """
    torch = _torch()
    import cv2
    from PIL import Image

    if isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
    else:
        arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"the multiview transformer prepares uint8 frames, got {arr.dtype}. Its "
            "grayscale conversion and INTER_AREA resize are defined on the decoded "
            "frame, not on a normalized float one."
        )
    if arr.ndim == 3:  # a single frame
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"expected (T, H, W, C) frames, got shape {arr.shape}")

    if int(margin) < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    h_out, w_out = input_hw
    out = np.empty((arr.shape[0], h_out, w_out), dtype=np.uint8)
    for t in range(arr.shape[0]):
        f = arr[t]
        if f.shape[-1] >= 3:
            gray = np.asarray(Image.fromarray(f[..., :3], mode="RGB").convert("L"))
        else:
            gray = f[..., 0]
        out[t] = cv2.resize(gray, (w_out, h_out), interpolation=cv2.INTER_AREA)

    # One plane per element of `mean`: three for a `deeperfly-mvt-1` artifact, whose stem
    # was trained on the grayscale frame replicated to RGB and normalized per channel; one
    # for a folded `-2`, where those three filters were collapsed into a single exact
    # equivalent. The artifact states which, so this never has to guess -- and a mismatch
    # between the plane count and the stem is a shape error at the first conv rather than a
    # silent broadcast.
    if int(margin):
        m = int(margin)
        out = np.pad(
            out,
            ((0, 0), (m, m), (m, m)),
            mode="constant",
            constant_values=MARGIN_FILL_U8,
        )
    c = len(tuple(mean))
    x = torch.from_numpy(out).to(device=device, dtype=torch.float32) / 255.0
    x = x.unsqueeze(1).expand(-1, c, -1, -1)
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, c, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, c, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------------------------------
# Load / run
# ---------------------------------------------------------------------------------------


def load_mvt(
    weights: str | Path,
    *,
    dev: str | None = None,
    mean: float = 0.0,
    precision: str | None = None,
):
    """Load an exported MVT artifact and return the eval-mode module on ``dev``.

    ``weights`` is the file written by dfpose's ``scripts/export_mvt_weights.py``, which
    carries the architecture, the point names and the normalization. There is no
    hard-coded architecture here: an artifact that does not state a field is a refusal, so
    a future checkpoint with a different width or schedule either works or says why not.

    ``mean`` is the :class:`~deeperfly.pose2d.models.ModelSpec`'s declared mean and must be
    ``0.0``: this network's normalization constants live in the ARTIFACT, one per input
    plane, so a config that also subtracted a mean of its own would shift every input by it
    with nothing to notice.

    ``precision`` must be ``float32`` if given. bf16 autocast moved 99.6% of cells against
    fp32 on the held-out project and put 183 of 12,464 more than a pixel out -- a subpixel
    argmax over a nearly-flat ridge picks a different local peak when the logits wobble in
    the third decimal. The whole corpus is minutes of GPU either way.
    """
    torch = _torch()

    if float(mean) != 0.0:
        raise SystemExit(
            f"a multiview-transformer model must declare mean = 0.0, got {mean}. Its "
            "normalization constants are in the artifact and are applied by the model."
        )
    if precision is not None and str(precision) != "float32":
        raise SystemExit(
            f"the multiview transformer runs in float32; this model declares "
            f"precision = {precision!r}. Measured: bf16 moves 99.6% of cells."
        )

    path = Path(weights)
    if not path.exists():
        raise SystemExit(f"no multiview-transformer artifact at {path}")
    # weights_only=True: the artifact holds tensors and plain containers only, and this
    # file is meant to be downloaded, so it is read without unpickling arbitrary objects.
    art = torch.load(path, map_location="cpu", weights_only=True)

    fmt = art.get("format") if isinstance(art, dict) else None
    if fmt not in ARTIFACT_FORMATS:
        raise SystemExit(
            f"{path}: format {fmt!r}, expected one of {ARTIFACT_FORMATS!r}. This is not an "
            "exported multiview-transformer artifact (a raw Lightning .ckpt is not one -- "
            "run dfpose's scripts/export_mvt_weights.py)."
        )
    arch = dict(art["arch"])
    # The input plane count is the artifact's normalization, and the arch entry must agree
    # with it. Both are then checked against the stem the weights actually carry, below --
    # three statements of one fact, because the failure they guard against (an artifact
    # whose scalar normalization is applied as if it were ImageNet's) produces numbers
    # rather than an exception.
    n_norm = len(tuple(art["normalization"]["mean"]))
    if "in_channels" in arch and int(arch["in_channels"]) != n_norm:
        raise SystemExit(
            f"{path}: arch.in_channels={arch['in_channels']} but normalization.mean has "
            f"{n_norm} entr{'y' if n_norm == 1 else 'ies'}"
        )
    arch["in_channels"] = n_norm
    if str(arch.get("view_embed")) != "off":
        raise SystemExit(
            f"artifact declares view_embed={arch.get('view_embed')!r}; only the "
            "identity-free model is runnable here, because the pathway plan does not "
            "promise the views arrive in the order a per-view table was trained on."
        )
    if str(arch.get("output_layout", "")).startswith("(batch, view * num_keypoints"):
        pass
    else:
        raise SystemExit(
            f"artifact declares an output layout this module does not implement: "
            f"{arch.get('output_layout')!r}"
        )

    net = _make_net(arch)
    # The third statement of the fact above, and the only one read off the WEIGHTS. A shape
    # disagreement would also surface from load_state_dict, but as a RuntimeError about a
    # tensor name; this says what is actually wrong with the artifact.
    stem_w = next(
        (
            v
            for k, v in art["state_dict"].items()
            if k.endswith("patch_embeddings.projection.weight")
        ),
        None,
    )
    if stem_w is not None and int(stem_w.shape[1]) != n_norm:
        raise SystemExit(
            f"{path}: the stem takes {int(stem_w.shape[1])} input channel(s) but the "
            f"artifact's normalization describes {n_norm}. A folded (grayscale) artifact "
            f"must declare a single mean/std; see dfpose scripts/gray_stem.py."
        )
    missing, unexpected = net.load_state_dict(art["state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{path}: the artifact does not match this module's definition -- "
            f"{len(missing)} missing {list(missing)[:4]}, "
            f"{len(unexpected)} unexpected {list(unexpected)[:4]}"
        )

    # How `LoadedModel` finds this module's own prepare/predict/decode. A plain attribute,
    # not a submodule, so it never enters the state dict.
    import sys

    net.impl = sys.modules[__name__]

    norm = art["normalization"]
    net.point_names = list(art["point_names"])
    # `input_size` is what the NETWORK takes. `hm_margin_px` says how much of that is
    # margin, so the REPORTED frame -- the coordinate system every returned point is
    # normalized into, and the one `ModelSpec.input_size` must declare -- is the network
    # input less twice the margin. Absent (every artifact through r27) means no margin, so
    # the two are the same and nothing about this model's behavior changes.
    net.model_input_hw = tuple(int(v) for v in art["input_size"])
    net.hm_margin_px = int(arch.get("hm_margin_px", 0) or 0)
    _m, _patch = net.hm_margin_px, int(arch["patch_size"])
    if _m < 0:
        raise SystemExit(
            f"{path}: arch.hm_margin_px is {_m}; a margin cannot be negative"
        )
    if _m % _patch:
        # The margin is realized by padding the network's INPUT, so it moves the token grid.
        # A margin that is not a whole number of patches gives a fractional grid, which the
        # backbone resolves by truncating -- the field then covers less than the margin says
        # and every decoded point is shifted by the difference.
        raise SystemExit(
            f"{path}: arch.hm_margin_px={_m} is not a multiple of the patch size "
            f"{_patch}. A margin has to be a whole number of patches on each side, so the "
            f"token grid stays integral: {_patch * (_m // _patch)} or "
            f"{_patch * (_m // _patch + 1)}."
        )
    net.input_hw = tuple(int(v) - 2 * _m for v in net.model_input_hw)
    if min(net.input_hw) <= 0:
        raise SystemExit(
            f"{path}: a margin of {_m} px leaves nothing of a "
            f"{net.model_input_hw[0]}x{net.model_input_hw[1]} input"
        )
    _fill = int(arch.get("hm_margin_fill", MARGIN_FILL_U8))
    if _m and _fill != MARGIN_FILL_U8:
        raise SystemExit(
            f"{path}: the artifact's margin was trained against a fill of {_fill}, but "
            f"this module pads with {MARGIN_FILL_U8}. The fill is a train/test contract "
            f"(see MARGIN_FILL_U8), not a cosmetic choice."
        )
    #: A joint may land OUTSIDE the reported frame exactly when there is field out there to
    #: land in. See `LoadedModel.padded_field`.
    net.padded_field = _m > 0
    #: Confidence below which a point is reported as NaN rather than as a location. Zero
    #: (no gate) unless the artifact sets one, because it is only meaningful for a
    #: checkpoint trained to answer "not here" with a flat map -- see `predict_points`.
    net.conf_floor = float(arch.get("conf_floor", 0.0) or 0.0)
    net.norm_mean = tuple(float(v) for v in norm["mean"])
    net.norm_std = tuple(float(v) for v in norm["std"])
    net.downsample_factor = int(arch["downsample_factor"])
    if len(net.point_names) != net.num_classes:
        raise SystemExit(
            f"{path}: {len(net.point_names)} point names for {net.num_classes} channels"
        )

    if dev is None:
        # Imported only when the caller did not choose: the backend's auto-selection lives
        # in the shared runtime module, and this class has no other reason to pull it in.
        from .runtime import device as _default_device

        target = _default_device()
    else:
        target = dev
    net = net.eval().to(target)
    n_view_params = sum(
        p.numel()
        for n, p in net.named_parameters()
        if "view_embed" in n or "view_proj" in n
    )
    log.info(
        "multiview transformer: %s, %d channels, input %dx%d (margin %d px -> reported "
        "frame %dx%d, padded field %s), scopes %s "
        "(%d view-local, %d global), %d per-view parameters -- view order and camera "
        "names are not read",
        arch.get("backbone"),
        net.num_classes,
        *net.model_input_hw,
        net.hm_margin_px,
        *net.input_hw,
        net.padded_field,
        arch.get("attn_scopes"),
        str(arch["attn_scopes"]).count("l"),
        str(arch["attn_scopes"]).count("g"),
        n_view_params,
    )
    return net


def predict_points(
    model, inputs, *, method: str = "weighted", radius: int = 2, views=None
):
    """``(B, V, 3, H, W)`` -> input-normalized ``(B, V, K, 2)`` peaks and ``(B, V, K)`` conf.

    ``method``/``radius`` are accepted for interface parity and ignored: this network's
    readout is the soft-argmax its targets were trained for, and substituting a windowed
    centroid over the un-upsampled field would move every point.

    **The view axis is kept, not flattened.** ``hrnet.predict_points`` deliberately folds
    the leading axes into the batch, which is right for a per-view detector and would here
    delete the only thing this architecture computes -- each view would be encoded alone.

    ``views`` decodes only those view indices and returns the ``V`` axis in *their* order;
    the FORWARD is unchanged, so every view still informs every other and the numbers are
    bit-identical to slicing the full result. Worth having because the decode is not a
    rounding error next to the forward: upsampling to the input size costs 16x the cells, so
    an 8-view frame spends ~11 ms there against ~10 ms in the network, and a caller reading
    one view (:mod:`deeperfly.pose2d.autocrop` searching one camera's crop) pays it eight
    times over for nothing. Exact because channels decode independently -- the same property
    :func:`decode_points` already relies on to chunk.
    """
    torch = _torch()
    x = (
        inputs
        if isinstance(inputs, torch.Tensor)
        else torch.as_tensor(np.asarray(inputs))
    )
    dev = next(model.parameters()).device
    x = x.to(device=dev, dtype=torch.float32)
    if x.ndim == 4:  # a single frame's views
        x = x.unsqueeze(0)
    if x.ndim != 5:
        raise ValueError(
            f"the multiview transformer takes (B, V, 3, H, W); got {tuple(x.shape)}. It "
            "is not a per-view function, so a (N, 3, H, W) batch is ambiguous."
        )
    b, v = x.shape[0], x.shape[1]
    k = model.num_classes
    if views is not None:
        wanted = [int(i) for i in views]
        if not wanted or any(not 0 <= i < v for i in wanted):
            raise ValueError(
                f"views {list(views)!r} outside the {v} view(s) of this input"
            )
    # No autocast: float32 is the contract, see load_mvt.
    with torch.inference_mode():
        hm = model.forward(x)
        if views is not None:
            # Channel c is view c // k (the module docstring's point 1), so a view's
            # channels are one contiguous run. Slice before the decode, not after.
            hm = torch.cat([hm[:, i * k : (i + 1) * k] for i in wanted], dim=1)
            v = len(wanted)
        xy, conf = decode_points(
            hm,
            model.input_hw,
            model.downsample_factor,
            margin=int(getattr(model, "hm_margin_px", 0)),
        )
        # A joint the frame cut off has no location to report. Where the checkpoint was
        # trained to answer that with a FLAT map (see dfpose's `off_frame_target=uniform`),
        # the confidence separates the two answers by ~500x, so a floor turns "a confident
        # point pinned to the border" -- which RANSAC and the bundle adjustment both believe
        # -- into NaN, which `deeperfly.triangulation` already means by "this camera cannot
        # see this point". Zero, i.e. off, unless the artifact sets a floor: on a checkpoint
        # trained without flat targets the confidence does NOT track off-frame-ness, and a
        # floor would drop good points on the strength of a number that does not mean what
        # it would need to mean.
        floor = float(getattr(model, "conf_floor", 0.0))
        if floor > 0.0:
            xy = xy.masked_fill(conf[..., None] < floor, float("nan"))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (
        xy.reshape(b, v, k, 2).cpu().numpy().astype(np.float32),
        conf.reshape(b, v, k).cpu().numpy().astype(np.float32),
    )


def predict_heatmaps(model, inputs):
    """``(B, V, 3, H, W)`` -> host ``(B, V, K, Hm, Wm)`` heatmaps, for the candidate path."""
    torch = _torch()
    x = (
        inputs
        if isinstance(inputs, torch.Tensor)
        else torch.as_tensor(np.asarray(inputs))
    )
    dev = next(model.parameters()).device
    x = x.to(device=dev, dtype=torch.float32)
    if x.ndim == 4:
        x = x.unsqueeze(0)
    with torch.inference_mode():
        hm = model.heatmaps_by_view(x)
    return hm.float().cpu().numpy()
