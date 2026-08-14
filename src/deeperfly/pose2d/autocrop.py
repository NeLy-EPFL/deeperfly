"""Search a view's detector crop from the footage, instead of writing the box by hand.

A detector is trained through a box, and feeding it a different one changes the animal's
apparent scale -- the one thing no augmentation in the recipe undoes. On this rig the six
side cameras match training full-frame and the two **axial** ones (front and hind, 1600x1008
against the side cameras' 960x512) do not: dropped in whole, the fly lands too small and
detection collapses. The box that fixes it is a per-recording measurement, and
``{ op = "crop", auto = true }`` (:class:`~deeperfly.preprocessing.AutoCrop`) is the
declaration that it should be *measured* rather than copy-pasted from another recording.

Every choice below was measured (2026-08-05, 13 labelled recordings; see the ``dfpose``
sibling ``scripts/auto_crop.py``), and most of them are the OPPOSITE of the obvious one.

**Search by grid, not by gradient.** A crop-resize is an affine warp, so it *is*
differentiable in ``(cx, cy, log w)`` -- and gradient descent still cannot find the box.
Two fatal reasons: a badly-scaled recording starts inside a **dead plateau** where the
detector emits near-zero heatmaps everywhere and ``|dJ/dlog w| ~ 1e-6`` is pure noise (Adam
normalizes by gradient magnitude, so it takes full steps along that noise and random-walks
the width); and **centre and width are not separable**, so "ladder the width, re-centre on
the detections, ladder again" refuses outright on the recordings that need it most. Measured,
a coarse grid plus a local refine grid (6.0 px error) also beat a grid plus 25 Adam steps
(8.8 px). Gradients pay off in thousands of dimensions, not in three with a smooth value
function.

**The search is a COVER, then a NARROWING.** That split is why bisection-shaped ideas do not
apply: off the informative region the objective is flat, so no local comparison points toward
it. The first round therefore has to be a blind cover of the whole box space (or of a seed's
neighbourhood, when the config gives one); only then is local refinement meaningful, and from
there each round shrinks the bracket geometrically -- which is the multiresolution narrowing
the intuition was reaching for, with a 3-D stencil instead of a 1-D sign test. A **beam** of
the best few cells is carried between rounds rather than only the winner, because the
objective's *value* is smooth while its differences across adjacent widths are not (bilinear
sampling at a 3x downsample is dominated by high-frequency image content).

**Confidence is the search objective and must NEVER be the accept criterion.** Once the
centre is free, confidence decouples from accuracy: a recording measured at 0.338 -> 0.403
confidence went 30.9 -> 37.5 px *worse*. And an absolute confidence floor throws away the
biggest real wins, because a camera can be improvable without ever being confident (268.1 ->
54.6 px at confidence 0.031). So confidence proposes and something independent decides.

**The gate is agreement with the other cameras' 3D**, which won a head-to-head against two
alternatives, scored by "does the signal say the derived box is better exactly when the real
error fell": reprojection **12/13**, self-consistency across scales 9/13, wide-view leak 5/13
-- worse than chance. It works even when the rig is corrupted by the very failure being
fixed, because it compares boxes against the SAME reference and a systematic displacement
cancels in the difference. Biased reference, valid paired test.

Here the reference is stronger than in the original -- the target view is **held out** of the
triangulation, so it is built only from cameras whose framing is not in question -- and that
buys something the paired version could not do. One reference scores *any* number of boxes, so
the gate is not restricted to accepting or refusing a single winner: it starts a small
**compass search** from where confidence left off (:data:`GATE_STEP_CENTRE`) and takes the box
that actually agrees best. Confidence covers, at ~11.5 ms a probe; geometry chooses, at ~210 ms
an evaluation. Measured over a centre-y x width grid on this rig's hind view, that division is
not a nicety -- the two signals are **uncorrelated** there (``r = +0.17``), confidence's
arg-max agrees to 4.9 px, and geometry's optimum reaches 3.1 px, which is where a person put
the box by hand (3.4 px). The ``r = -0.92`` that would justify optimizing confidence holds
only along a width ladder at a fixed good centre, and giving the centre freedom is exactly
what destroys it.

**A gate is only as good as its rig, and that is checkable.** Run against a *nominal orbit*
rig rather than a solved one, the reference for this recording's hind view was ~250 px out,
and the paired comparison then refused a box that was in truth 40x better than the incumbent
-- a false negative from a reference with no relation to the image, which is a different
thing from the systematic displacement that cancels. So before trusting it, the rig is asked
to explain the views the reference was built FROM (:data:`RIG_RESIDUAL_LIMIT`): a rig that
cannot reproject into the cameras it was triangulated from cannot be believed about a
held-out one, and the search says so instead of quietly picking with a broken ruler.

**Containment is checked against the candidate's OWN output, never against that reference.**
Bias cancels in a paired comparison but not in an absolute constraint -- the arm that
constrained the grid by the reprojected cloud sent one recording from 100.8 px to 326.0. What
survives is :data:`CLIP_FRACTION`: refuse a candidate too many of whose own detections fall
outside it. Two failure modes compound without it, and both were observed: the objective is
confidence, and a tighter box raises confidence by filling the frame with animal; the gate is
paired, so on a view that starts broken it accepts "less bad" and cannot veto a box that
still cuts the legs off.

**Not a fill fraction.** On the hind camera the animal legitimately fills ~90% of a correct
box, and vetoing on fill drove the search off a good tight box onto a huge one centred on the
foam ball (confidence up, visibly wrong).

**Frames are spread over the whole recording, never contiguous.** The fly's distance to an
axial camera drifts through a run (a box fitted from 1 frame vs 100 measured 652 vs 808 px)
and at 100 fps adjacent frames are near-duplicates (lag-1 autocorrelation ~+0.6 at 40 ms), so
a contiguous sample is one moment measured several times. The sibling implementation ranked
frames by how much 3D they already had and took the top few; on a healthy recording 5998 of
6000 frames tie, so that criterion carries no information and an unstable sort returns an
arbitrary *contiguous* block -- 110 ms of a 60 s clip. Here both sets come from
:func:`numpy.linspace` and the gate's frames are disjoint from the search's.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..preprocessing import AutoCrop, FrameTransform

log = logging.getLogger("deeperfly")

__all__ = [
    "SIDECAR_NAME",
    "AutoCropTarget",
    "Resolution",
    "targets",
    "resolved_plan",
    "read_sidecar",
    "write_sidecar",
    "search",
    "ensure_resolved",
]

#: Where the resolved boxes are persisted, inside the recording's output directory. They
#: are NOT in the pose2d fingerprint (see :class:`~deeperfly.preprocessing.AutoCrop`), so
#: this file is how a later process -- a re-render reusing the cached 2D -- learns which
#: window the detector actually looked through.
SIDECAR_NAME = "autocrop.json"

#: Sidecar schema version, so a format change is a refusal and not a misread box.
SIDECAR_VERSION = 1

# -- the stencil -------------------------------------------------------------------------
# Round 0 covers; rounds 1+ narrow. The blind cover's widths are fractions of the FRAME and
# its centres range over the frame; a seeded search replaces both with a neighbourhood of
# the seed, which is why a seed is worth writing down when one is known.

#: Blind cover: candidate widths as a fraction of the frame width. Log-spaced (ratio ~1.37)
#: because what the detector cares about is the animal's *scale*, which is multiplicative.
COVER_WIDTHS: tuple[float, ...] = (0.16, 0.22, 0.30, 0.41, 0.56, 0.77, 1.0)
#: Blind cover: candidate centres as a fraction of the frame, x then y. Wider in x because
#: an axial frame is wider; the step (~160 px on a 1600 px frame) sits inside the measured
#: +-50..150 px window where confidence still carries signal, which is what the cover needs.
COVER_X: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
COVER_Y: tuple[float, ...] = (0.15, 0.325, 0.5, 0.675, 0.85)

#: Seeded search: candidate widths as a multiple of the seed's. The measured ladder.
SEED_WIDTHS: tuple[float, ...] = (0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75)
#: Seeded search: centre offsets per axis as a fraction of the candidate box. 5x5.
SEED_OFFSETS: tuple[float, ...] = (-0.22, -0.11, 0.0, 0.11, 0.22)

#: Refine rounds: widths as a multiple of the incumbent's, and centre offsets as a fraction
#: of the candidate. A round is a narrowing, not another cover, so it stays small.
REFINE_WIDTHS: tuple[float, ...] = (0.9, 1.0, 1.11)
REFINE_OFFSETS: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)
#: How many narrowing rounds follow the cover.
REFINE_ROUNDS: int = 3
#: What each round multiplies the previous round's step by. **The narrowing does not happen
#: without this.** The offsets are a fraction of the candidate box, and the box does not
#: shrink as the search converges, so an undecayed step stays the same size every round: the
#: centre then oscillates inside one step forever instead of closing on anything. Measured
#: before the decay existed, three rounds moved a hind-view centre 960 -> 737 -> 796 px and
#: never settled. At 0.4 a round the reach is ~0.19 of the box (enough to close the cover's
#: own step) and the final resolution ~0.016 of it, a few pixels.
REFINE_DECAY: float = 0.4

#: Candidates carried from one round to the next. More than one because the objective's
#: differences across adjacent widths are noisy, so the true basin can score second.
BEAM: int = 2
#: Candidates promoted from the cheap one-frame ranking to the full multi-frame score.
SHORTLIST: int = 6

#: The geometry refinement's first step: centre offsets as a fraction of the box, and the
#: width as a multiplicative factor. Halved whenever a step finds nothing better, which is a
#: compass (pattern) search -- six evaluations per iteration over ``(cx, cy, w)``.
#:
#: **Why geometry and not more confidence.** Mapped over a 4 x 6 grid of centre-y against
#: width on this rig's hind view, with the target view held out of the reference:
#:
#: * the two signals are essentially UNCORRELATED -- pearson ``r = +0.17``, and the wrong
#:   sign at that. The ``r = -0.92`` that justifies a confidence objective holds along a
#:   width ladder at a FIXED GOOD CENTRE and does not survive giving the centre freedom,
#:   which is the same decoupling the original measurements recorded (a box that got more
#:   confident and less accurate).
#: * confidence's arg-max over the grid agrees to 4.9 px; agreement's own optimum reaches
#:   3.1 px, and a hand-tuned box 3.4 -- so geometry's optimum is not merely better than
#:   confidence's, it lands *where a person put the box by hand*.
#: * agreement is repeatable: the same boxes re-scored on a disjoint set of gate frames
#:   moved 3.41 -> 3.37 px and 4.92 -> 4.50, so differences of a pixel are signal.
#:
#: What confidence is genuinely good at is the part geometry cannot afford: covering a 3-D
#: space at ~11 ms a probe, against ~350 ms for an agreement evaluation. So confidence
#: brackets and geometry chooses, and this constant is where the handover happens.
#: Each step is sized to the uncertainty the confidence search actually leaves behind: the
#: centre to the cover's own grid step (~0.1 of a box), and the width generously, because
#: confidence is flat over a 2x width range and its arg-max was measured 2.2x too wide. A
#: timid 1.15 width step needs four consecutive accepts to cross that and stalls at 4.1 px
#: where 1.4 reaches 3.4 -- a hand-tuned box's accuracy.
GATE_STEP_CENTRE: float = 0.10
GATE_STEP_WIDTH: float = 1.4
#: Step below which the compass search stops (a fraction of the box, so ~1% of a 500 px box).
GATE_STEP_FLOOR: float = 0.012

#: A candidate more than this fraction of whose OWN detections land outside it is clipping
#: the animal and is refused whatever its confidence.
CLIP_FRACTION: float = 0.12

#: How far, in pixels, the rig may miss the views the reference was triangulated FROM before
#: the gate refuses to use it. Those views' 2D is what built the 3D, so their reprojection
#: error is the rig's own self-consistency; a rig that cannot explain them says nothing
#: trustworthy about a held-out view. Generous, because the number it guards against is an
#: order of magnitude larger (a nominal orbit rig measured ~250 px on this recording's hind
#: view while a solved one measured 3).
RIG_RESIDUAL_LIMIT: float = 25.0

#: Border band, in model pixels, that counts as "outside" for a model whose heatmap field
#: does not extend past its input (:attr:`~deeperfly.pose2d.models.LoadedModel.padded_field`
#: is False). Such a model cannot place a peak beyond the box -- a cut-off joint saturates
#: *against* the border instead -- so the test is proximity to the edge rather than crossing
#: it. A padded model gets the exact test (outside ``[0, 1]``) and needs no band.
BORDER_BAND_PX: float = 3.0


@dataclass(frozen=True)
class AutoCropTarget:
    """One ``[[pose2d.preprocessors]]`` chain whose crop is to be searched.

    Attributes
    ----------
    preprocessor
        The preprocessor's name -- the key the resolved box is stored under.
    transform
        Its chain, carrying the unresolved :class:`~deeperfly.preprocessing.AutoCrop`.
    source
        The footage source the pathways using it read.
    view
        The view index the pathways using it write into (the ``V`` axis of the points).
    view_name
        That view's name, for logs and the gate.
    model
        The model name the pathways using it forward through.
    pathways
        The names of the pathways using it, in config order.
    """

    preprocessor: str
    transform: FrameTransform
    source: str
    view: int
    view_name: str
    model: str
    pathways: tuple[str, ...]


@dataclass
class Resolution:
    """What the search decided for one target, and the evidence for it."""

    preprocessor: str
    view_name: str
    box: tuple[int, int, int, int]
    incumbent: tuple[int, int, int, int]
    seeded: bool
    conf: float = float("nan")
    conf_incumbent: float = float("nan")
    agreement_px: float = float("nan")
    agreement_incumbent_px: float = float("nan")
    rig_residual_px: float = float("nan")
    gate_evals: int = 0
    accepted: bool = False
    gated: bool = False
    probes: int = 0
    seconds: float = 0.0
    search_frames: tuple[int, ...] = ()
    gate_frames: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        out = {
            "view": self.view_name,
            "box": list(self.box),
            "incumbent": list(self.incumbent),
            "seeded": self.seeded,
            "conf": _round(self.conf, 4),
            "conf_incumbent": _round(self.conf_incumbent, 4),
            "accepted": self.accepted,
            "gated": self.gated,
            "probes": self.probes,
            "seconds": _round(self.seconds, 2),
            "search_frames": list(self.search_frames),
        }
        if self.gated:
            out["agreement_px"] = _round(self.agreement_px, 2)
            out["agreement_incumbent_px"] = _round(self.agreement_incumbent_px, 2)
            out["rig_residual_px"] = _round(self.rig_residual_px, 2)
            out["gate_evals"] = self.gate_evals
            out["gate_frames"] = list(self.gate_frames)
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def _round(value: float, digits: int):
    v = float(value)
    return None if not np.isfinite(v) else round(v, digits)


# -- what needs searching ----------------------------------------------------------------


def targets(plan) -> list[AutoCropTarget]:
    """The plan's unresolved automatic crops, in ``[[pose2d.preprocessors]]`` order.

    A preprocessor may be shared by several pathways -- the rig's mirrored twin is one
    source through one window twice -- which is fine and gives one box. Sharing it across
    different *sources* or *views* is not: one searched window cannot be two cameras'
    framing, and picking either silently mis-scales the other.

    Raises
    ------
    ValueError
        If a preprocessor with an automatic crop is used by pathways that disagree on the
        source or the view, or by no pathway at all.
    """
    out: list[AutoCropTarget] = []
    for name, transform in plan.preprocessors.items():
        if not transform.needs_auto_crop:
            continue
        users = [pw for pw in plan.pathways if pw.preprocessor == name]
        if not users:
            raise ValueError(
                f"[[pose2d.preprocessors]] {name!r} declares `auto = true` but no pathway "
                "uses it, so there is no footage to search and no view to judge it by; "
                "point a pathway at it or delete it"
            )
        sources = {pw.source for pw in users}
        views = {int(v) for pw in users for v in np.unique(pw.mapping[:, 1])}
        models = {pw.model for pw in users}
        if len(sources) > 1 or len(views) > 1:
            raise ValueError(
                f"[[pose2d.preprocessors]] {name!r} has `auto = true` and is shared by "
                f"pathways covering sources {sorted(sources)} and views "
                f"{sorted(plan.view_names[v] for v in views)}; one searched window cannot "
                "be two cameras' framing. Give each its own preprocessor."
            )
        view = views.pop()
        out.append(
            AutoCropTarget(
                preprocessor=name,
                transform=transform,
                source=sources.pop(),
                view=view,
                view_name=plan.view_names[view],
                model=sorted(models)[0],
                pathways=tuple(pw.name for pw in users),
            )
        )
    return out


def resolved_boxes(plan) -> dict[str, tuple[int, int, int, int]]:
    """``preprocessor name -> window`` for every automatic crop the plan has resolved.

    What a later stage needs in order to look through the same box detection did, whether
    this run searched it or read it back from the sidecar -- so the caller does not have to
    remember which of the two happened.
    """
    return {
        name: transform.auto_crop.box  # type: ignore[union-attr]
        for name, transform in plan.preprocessors.items()
        if transform.auto_crop is not None and transform.auto_crop.resolved
    }


def resolved_plan(plan, boxes: dict[str, tuple[int, int, int, int]]):
    """``plan`` with each named preprocessor's automatic crop set to its box.

    Returns the plan unchanged when ``boxes`` names nothing it carries, so a caller can
    apply a sidecar blindly. A name the plan does not have is ignored for the same reason:
    a stale sidecar entry (a preprocessor since renamed or made explicit) should not stop
    a run, it should just not apply.
    """
    import dataclasses

    if not boxes:
        return plan
    preprocessors = dict(plan.preprocessors)
    changed = False
    for name, box in boxes.items():
        transform = preprocessors.get(name)
        if transform is None or transform.auto_crop is None:
            continue
        preprocessors[name] = transform.resolve_auto_crop(tuple(int(v) for v in box))
        changed = True
    if not changed:
        return plan
    pathways = [
        dataclasses.replace(
            pw,
            transform=(
                preprocessors[pw.preprocessor]
                if pw.preprocessor is not None
                else pw.transform
            ),
        )
        for pw in plan.pathways
    ]
    return dataclasses.replace(plan, preprocessors=preprocessors, pathways=pathways)


# -- the sidecar -------------------------------------------------------------------------


def read_sidecar(outdir: Path | str) -> dict[str, tuple[int, int, int, int]]:
    """Resolved boxes recorded beside a recording's results, or ``{}`` if there are none."""
    path = Path(outdir) / SIDECAR_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "ignoring unreadable %s (%s); the crops will be searched again", path, exc
        )
        return {}
    if data.get("version") != SIDECAR_VERSION:
        log.warning(
            "%s was written by a different version (%r, expected %r); the crops will be "
            "searched again",
            path,
            data.get("version"),
            SIDECAR_VERSION,
        )
        return {}
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, box in (data.get("boxes") or {}).items():
        try:
            x, y, w, h = (int(v) for v in box)
        except (TypeError, ValueError):
            log.warning("ignoring malformed box %r for %r in %s", box, name, path)
            continue
        out[str(name)] = (x, y, w, h)
    return out


def write_sidecar(outdir: Path | str, resolutions: list[Resolution]) -> Path:
    """Persist the resolved boxes and the evidence for them; returns the file's path."""
    path = Path(outdir) / SIDECAR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": SIDECAR_VERSION,
                "boxes": {r.preprocessor: list(r.box) for r in resolutions},
                "detail": {r.preprocessor: r.to_json() for r in resolutions},
            },
            indent=1,
        )
        + "\n"
    )
    return path


# -- candidate geometry ------------------------------------------------------------------


def _fit(
    cx: float, cy: float, width: float, aspect: float, frame_hw
) -> tuple[int, int, int, int]:
    """The ``(x, y, w, h)`` box of the given width and aspect nearest ``(cx, cy)``.

    Shrunk uniformly if it does not fit the frame (never squeezed -- an anisotropic squeeze
    is a second scale error on top of the one being searched) and translated to fit rather
    than clipped, so the requested size survives wherever it can.
    """
    fh, fw = int(frame_hw[0]), int(frame_hw[1])
    w = float(width)
    h = w / aspect
    shrink = min(1.0, fw / w, fh / h)
    w, h = w * shrink, h * shrink
    wi = max(1, min(fw, int(round(w))))
    hi = max(1, min(fh, int(round(h))))
    x = int(round(cx - wi / 2.0))
    y = int(round(cy - hi / 2.0))
    return (
        max(0, min(x, fw - wi)),
        max(0, min(y, fh - hi)),
        wi,
        hi,
    )


def _cover_boxes(frame_hw, aspect: float) -> list[tuple[int, int, int, int]]:
    """The blind cover: every :data:`COVER_WIDTHS` at every :data:`COVER_X` x :data:`COVER_Y`."""
    fh, fw = frame_hw
    boxes = [
        _fit(fx * fw, fy * fh, frac * fw, aspect, frame_hw)
        for frac in COVER_WIDTHS
        for fx in COVER_X
        for fy in COVER_Y
    ]
    return list(dict.fromkeys(boxes))


def _local_boxes(
    box, frame_hw, aspect: float, widths: tuple[float, ...], offsets: tuple[float, ...]
) -> list[tuple[int, int, int, int]]:
    """A neighbourhood of ``box``: widths as multiples of its own, centres as fractions of
    the candidate (so a wide candidate scans coarsely and a narrow one finely)."""
    x0, y0, w0, h0 = box
    cx, cy = x0 + w0 / 2.0, y0 + h0 / 2.0
    boxes = []
    for mult in widths:
        w = w0 * mult
        h = w / aspect
        for dx in offsets:
            for dy in offsets:
                boxes.append(_fit(cx + dx * w, cy + dy * h, w, aspect, frame_hw))
    return list(dict.fromkeys(boxes))


def _incumbent_box(
    auto: AutoCrop, frame_hw, aspect: float
) -> tuple[int, int, int, int]:
    """The box the search is measured AGAINST: the seed, or the widest model-aspect box.

    Without a seed the incumbent is what the recording would get if the feature did not
    exist -- as much of the frame as the model's aspect allows -- so the gate's paired
    comparison answers the question the operator actually has: is searching better than
    not cropping?
    """
    if auto.seed is not None:
        return auto.seed
    fh, fw = frame_hw
    return _fit(fw / 2.0, fh / 2.0, fw, aspect, frame_hw)


# -- probing -----------------------------------------------------------------------------


class _Prober:
    """Scores candidate boxes for ONE view, holding everything that does not vary.

    One probe is one candidate on one frame. For a per-view detector that is one image, and
    every candidate rides the batch axis of a single forward. For a **joint-view** model
    (:attr:`~deeperfly.pose2d.models.LoadedModel.joint_views`) it is a whole *moment*: the
    view's output depends on the other views in the tensor, so the other views' inputs are
    prepared once and reused, and only the target's slot varies.

    Measured on the multiview transformer, RTX 4090, float32: **~11.5 ms a probe**, end to
    end, which is the forward and essentially nothing else. Three things get it there, each
    measured rather than assumed:

    * **Batch the candidates.** An 8-view moment forwards in 10 ms at batch >= 2 and 47 ms at
      batch 1, so this matters more than everything else combined.
    * **Prepare the fixed views once**, not per candidate. Preparation is host cv2 INTER_AREA:
      5 ms for a full 1600x1008 frame against 0.2 ms for a small crop, so re-preparing seven
      unchanging views per probe would cost more than the network.
    * **Decode only the view being searched**
      (:meth:`~deeperfly.pose2d.models.LoadedModel.predict_points_for_views`). The decode
      upsamples every channel to the full input, ~11 ms for eight views' 304 channels against
      ~1 for one view's 38 -- as much as the forward, spent on views this discards. Exact, not
      approximate: channels decode independently, so it is the same arithmetic on fewer of
      them, and the searched boxes are bit-identical with and without it.
    """

    def __init__(
        self,
        model,
        target: AutoCropTarget,
        frames,
        *,
        others: list | None = None,
        slots: tuple[int, ...] = (0,),
        batch: int = 8,
    ) -> None:
        self.model = model
        self.target = target
        self.frames = frames
        self.frame_hw = (int(frames.shape[-3]), int(frames.shape[-2]))
        self.others = (
            others  # (T, 3, H, W) per view of the model, target slots included
        )
        # Usually one slot. More than one when several of the model's pathways share this
        # preprocessor -- they then share its window too, so every one of them has to see the
        # candidate; leaving the others at the incumbent would score a mixture of two boxes.
        self.slots = tuple(slots)
        self.batch = max(1, int(batch))
        self.probes = 0
        self._cache: dict[tuple, tuple[float, float]] = {}

    def _prepare(self, box, n_frames: int):
        chain = self.target.transform.resolve_auto_crop(box)
        return self.model.prepare(chain.apply(self.frames[:n_frames]))

    def _forward(self, boxes, n_frames: int):
        """``(len(boxes), n_frames, K, 2)`` peaks and ``(..., K)`` conf for the target view."""
        import torch

        prepared = [self._prepare(box, n_frames) for box in boxes]
        if self.others is None:  # per-view model: candidates are just more batch
            x = torch.cat([p.reshape(-1, *p.shape[-3:]) for p in prepared])
            xy, conf = self.model.predict_points(x)
            n = len(boxes)
            return (
                np.asarray(xy).reshape(n, n_frames, -1, 2),
                np.asarray(conf).reshape(n, n_frames, -1),
            )
        moments = []
        for prep in prepared:
            for t in range(n_frames):
                views = [o[t] for o in self.others]
                for slot in self.slots:
                    views[slot] = prep[t]
                moments.append(torch.stack(views))
        x = torch.stack(moments)  # (N * T, V, 3, H, W)
        # Decode only the view being searched -- every target slot holds the same box, so one
        # of them is the whole answer. The forward is untouched (every view still informs
        # every other) and this halves the per-probe cost; see the class docstring.
        xy, conf = self.model.predict_points_for_views(x, (self.slots[0],))
        n = len(boxes)
        xy = np.asarray(xy)[:, 0].reshape(n, n_frames, -1, 2)  # one view was requested
        conf = np.asarray(conf)[:, 0].reshape(n, n_frames, -1)
        return xy, conf

    def score(self, boxes, n_frames: int) -> np.ndarray:
        """Mean confidence per box, ``-inf`` for a box that clips its own detections.

        The objective is the mean peak confidence over every channel and frame. It is only
        ever compared *between boxes of the same view*, so a channel this camera can never
        see well contributes a constant and does no harm; normalizing per channel would
        instead let the weakest channel dominate.
        """
        out = np.empty(len(boxes), dtype=float)
        todo = [
            i for i, b in enumerate(boxes) if (tuple(b), n_frames) not in self._cache
        ]
        for start in range(0, len(todo), self._chunk(n_frames)):
            idx = todo[start : start + self._chunk(n_frames)]
            chunk = [tuple(boxes[i]) for i in idx]
            xy, conf = self._forward(chunk, n_frames)
            self.probes += len(chunk) * n_frames
            clipped = self._clipped_fraction(xy, chunk)
            for j, box in enumerate(chunk):
                value = (
                    float(np.nanmean(conf[j]))
                    if np.isfinite(conf[j]).any()
                    else float("-inf")
                )
                if clipped[j] > CLIP_FRACTION:
                    value = float("-inf")
                self._cache[(box, n_frames)] = (value, float(clipped[j]))
        for i, box in enumerate(boxes):
            out[i] = self._cache[(tuple(box), n_frames)][0]
        return out

    def clipped_fraction(self, box, n_frames: int) -> float:
        """The recorded clipped fraction for an already-scored box (for the report)."""
        hit = self._cache.get((tuple(box), n_frames))
        return float("nan") if hit is None else hit[1]

    def _chunk(self, n_frames: int) -> int:
        return max(1, self.batch // max(1, n_frames))

    def _clipped_fraction(self, xy, boxes) -> np.ndarray:
        """Fraction of each candidate's own detections that its box cuts off.

        ``xy`` is input-normalized, so the test is in model coordinates and needs no
        mapping back. Which test depends on the model: a padded field can place a peak
        beyond ``[0, 1]``, which is exactly "outside the box"; an unpadded one cannot, so a
        cut-off joint piles up within :data:`BORDER_BAND_PX` of the edge instead.
        """
        h_in, w_in = self.model.input_size
        if self.model.padded_field:
            lo_x = lo_y = 0.0
            hi_x = hi_y = 1.0
        else:
            lo_x, lo_y = BORDER_BAND_PX / w_in, BORDER_BAND_PX / h_in
            hi_x, hi_y = 1.0 - lo_x, 1.0 - lo_y
        u, v = xy[..., 0], xy[..., 1]
        finite = np.isfinite(u) & np.isfinite(v)
        out = (u < lo_x) | (u > hi_x) | (v < lo_y) | (v > hi_y)
        n = finite.reshape(len(boxes), -1).sum(-1)
        hits = (out & finite).reshape(len(boxes), -1).sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n > 0, hits / np.maximum(n, 1), 0.0)
        return frac


# -- the search --------------------------------------------------------------------------


def _search_one(prober: _Prober, target: AutoCropTarget, aspect: float, n_frames: int):
    """Cover, then narrow. Returns ``(finalists, rounds_log)``.

    Each round ranks its candidates on ONE frame (enough to tell a basin from the flat sea)
    and re-scores only a shortlist on all of them (no decision is ever made on a single
    moment). That is a fidelity ladder on the noisy axis rather than on the search axis,
    which is where it is cheapest.

    ``finalists`` is every multi-frame-scored candidate, best confidence first -- not one
    winner. The caller has a second, better-calibrated signal (see the module docstring) and
    can only use it on a handful of boxes, so this hands over the handful.
    """
    auto = target.transform.auto_crop
    assert auto is not None
    frame_hw = prober.frame_hw
    # Round 0's candidates ARE the cover: a neighbourhood of the seed when the config gives
    # one (the seed itself is in it, at mult 1.0 and offset 0), the whole frame otherwise.
    cands = (
        _local_boxes(auto.seed, frame_hw, aspect, SEED_WIDTHS, SEED_OFFSETS)
        if auto.seed is not None
        else _cover_boxes(frame_hw, aspect)
    )
    beam: list[tuple[int, int, int, int]] = []
    rounds: list[str] = []
    scored: dict[tuple[int, int, int, int], float] = {}
    for rnd in range(1 + REFINE_ROUNDS):
        if rnd:
            decay = REFINE_DECAY ** (rnd - 1)
            widths = tuple(1 + (m - 1) * decay for m in REFINE_WIDTHS)
            offsets = tuple(o * decay for o in REFINE_OFFSETS)
            cands = [
                b
                for b in dict.fromkeys(
                    b
                    for member in beam
                    for b in _local_boxes(member, frame_hw, aspect, widths, offsets)
                )
                if b not in scored
            ]
        if not cands:
            break
        rank = prober.score(cands, 1)
        order = np.argsort(-rank)
        shortlist = [cands[i] for i in order[:SHORTLIST] if np.isfinite(rank[i])]
        if not shortlist:
            rounds.append(f"round {rnd}: every one of {len(cands)} candidates clipped")
            break
        full = prober.score(shortlist, n_frames)
        for box, value in zip(shortlist, full):
            if np.isfinite(value):
                scored[box] = float(value)
        forder = [i for i in np.argsort(-full) if np.isfinite(full[i])]
        beam = [shortlist[i] for i in forder[: (BEAM if rnd == 0 else 1)]]
        if not beam:
            break
        rounds.append(
            f"round {rnd}: {len(cands)} candidates -> {beam[0]} at conf {full[forder[0]]:.3f}"
        )
    finalists = sorted(scored.items(), key=lambda kv: -kv[1])
    return finalists, rounds


def _reference(
    plan, models, cameras, windows, target: AutoCropTarget, batch_size: int
) -> tuple[np.ndarray | None, float, str]:
    """Where the OTHER cameras say the target view's points are: ``(ref, rig_px, why_not)``.

    The target view is **held out** of the triangulation, so the reference depends on
    neither the box being judged nor the target camera's own framing, and one reference
    therefore scores any number of candidates.

    ``rig_px`` is the rig's own self-consistency: the median reprojection error of that
    triangulation *in the views that built it*. It is the check that separates "this box is
    bad" from "this rig cannot reproject" -- see :data:`RIG_RESIDUAL_LIMIT`. ``why_not`` is
    non-empty when the reference must not be used.
    """
    from . import inference

    pts2d, _ = inference.detect_sequence(plan, models, windows, batch_size=batch_size)
    held = np.array(pts2d, dtype=float)
    held[target.view] = np.nan
    seen = np.isfinite(held).all(-1).sum(0)  # (T, P) contributing views per point
    if int((seen >= 2).sum()) == 0:
        return (
            None,
            float("nan"),
            (
                "fewer than two other views triangulate on the gate frames, so there is no "
                "reference to compare against"
            ),
        )
    pts3d = cameras.triangulate(held)
    projected = np.asarray(cameras.project(pts3d))  # (V, T, P, 2)
    residual = np.linalg.norm(projected - held, axis=-1)
    residual = residual[np.isfinite(residual)]
    rig_px = float(np.median(residual)) if residual.size else float("inf")
    if not np.isfinite(rig_px) or rig_px > RIG_RESIDUAL_LIMIT:
        return (
            None,
            rig_px,
            (
                f"the rig misses the views the reference was built from by {rig_px:.0f} px "
                f"(limit {RIG_RESIDUAL_LIMIT:.0f}), so it cannot be believed about a held-out "
                "view either. Solve the rig first -- run once with bundle adjustment and point "
                "[cameras].calibration at the exported calibration.toml"
            ),
        )
    return projected[target.view], rig_px, ""


def _agreement(
    plan, models, windows, target: AutoCropTarget, ref: np.ndarray, batch_size: int
) -> float:
    """Median px from this plan's detections in the target view to ``ref``.

    Deliberately the **production** detection path, whole: ~210 ms measured, which is now the
    larger half of a view's search (a ~6 s confidence phase against ~5 s of gate). Most of
    that is waste of a kind already removed from the confidence probe -- every evaluation
    re-prepares all eight views' gate frames on the host and decodes all their channels, while
    only the target view's points are read.

    Not optimized, and the reason is where the risk sits rather than where the seconds are.
    Narrowing this means routing channels to skeleton points and inverting the pathway here
    instead of in :func:`~deeperfly.pose2d.inference.detect_sequence`, i.e. duplicating the
    peak-convention and mapping logic inside the code that *decides* the box. A subtle error
    there does not crash, it silently optimizes the wrong quantity -- for about 15 s on a
    once-per-recording measurement. If it is ever taken: the check is that the recorded
    ``agreement_px`` and the chosen boxes stay bit-identical on a real recording.
    """
    from . import inference

    pts2d, _ = inference.detect_sequence(plan, models, windows, batch_size=batch_size)
    d = np.linalg.norm(pts2d[target.view] - ref, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("inf")


def search(
    plan,
    models: dict,
    *,
    search_windows: dict,
    gate_windows: dict | None = None,
    cameras=None,
    params=None,
    batch_size: int = 16,
    search_frames: tuple[int, ...] = (),
    gate_frames: tuple[int, ...] = (),
    incumbents: dict[str, tuple[int, int, int, int]] | None = None,
) -> list[Resolution]:
    """Search every unresolved automatic crop in ``plan``; one :class:`Resolution` each.

    Parameters
    ----------
    plan
        The detection plan, carrying unresolved automatic crops.
    models
        ``name -> LoadedModel``, already on the device.
    search_windows
        ``source name -> (T, H, W, 3)`` frames the objective is evaluated on. Must cover at
        least every target's source; for a joint-view model, every source of that model.
    gate_windows
        ``source name -> (T', H, W, 3)`` frames the accept gate is scored on -- disjoint
        from the search frames. ``None`` (or no ``cameras``) skips the gate, which is
        logged: the search then keeps whatever confidence proposed, and confidence alone is
        known to accept a "confidently wrong" box.
    cameras
        The rig, for the gate's held-out triangulation.
    params
        A :class:`~deeperfly.config.AutoCropParams` (or ``None`` for its defaults).
    batch_size
        Forward batch for the gate's detection passes.
    search_frames, gate_frames
        The frame indices behind the two windows, recorded in the result for provenance.
    incumbents
        Optional per-preprocessor boxes to measure against instead of the seed -- used by
        the second pass, where "the incumbent" is what the first pass decided.
    """
    from ..config import AutoCropParams

    params = params or AutoCropParams()
    todo = targets(plan)
    if not todo:
        return []
    out: list[Resolution] = []
    for target in todo:
        started = time.perf_counter()
        model = models[target.model]
        frames = search_windows.get(target.source)
        if frames is None or len(frames) == 0:
            raise SystemExit(
                f"auto-crop has no frames for source {target.source!r}, which feeds "
                f"view {target.view_name!r}"
            )
        frame_hw = (int(frames.shape[-3]), int(frames.shape[-2]))
        auto = target.transform.auto_crop
        assert auto is not None
        aspect = (
            auto.seed[2] / auto.seed[3]
            if auto.seed is not None
            else model.input_size[1] / model.input_size[0]
        )
        incumbent = (incumbents or {}).get(
            target.preprocessor, _incumbent_box(auto, frame_hw, aspect)
        )

        others, slots = None, (0,)
        if model.joint_views:
            others, slots = _joint_views(
                plan, models, target, search_windows, incumbents
            )
        n_frames = min(len(frames), max(1, int(params.search_frames)))
        prober = _Prober(
            model, target, frames, others=others, slots=slots, batch=params.probe_batch
        )
        finalists, rounds = _search_one(prober, target, aspect, n_frames)
        conf_incumbent = float(prober.score([incumbent], n_frames)[0])
        res = Resolution(
            preprocessor=target.preprocessor,
            view_name=target.view_name,
            box=incumbent,
            incumbent=incumbent,
            seeded=auto.seed is not None,
            conf_incumbent=conf_incumbent,
            probes=prober.probes,
            search_frames=tuple(int(i) for i in search_frames),
        )
        for line in rounds:
            log.info("auto-crop %s: %s", target.view_name, line)
        if not finalists:
            res.notes.append(
                "every candidate clipped its own detections; keeping the incumbent"
            )
            res.seconds = time.perf_counter() - started
            out.append(res)
            log.warning(
                "auto-crop %s: no candidate survived the clipping check -- keeping %s",
                target.view_name,
                incumbent,
            )
            continue

        res.conf = finalists[0][1]
        gating = cameras is not None and bool(gate_windows) and params.gate
        if gating:
            _choose_by_agreement(
                plan,
                models,
                cameras,
                gate_windows,
                target,
                finalists[: max(1, int(params.gate_candidates))],
                incumbent,
                res,
                batch_size,
                gate_frames,
                incumbents,
                params,
                prober=prober,
                prober_frames=n_frames,
                aspect=aspect,
            )
        else:
            box, conf = finalists[0]
            res.accepted = box != incumbent
            res.box = box if res.accepted else incumbent
            res.conf = conf
            res.notes.append(
                "chosen on confidence alone (no gate: needs a solved rig and gate frames)"
                " -- confidence is known to prefer a box that is confidently wrong, and"
                " measurably prefers one a little too wide, so check the pose2d overlay"
            )
            if res.accepted:
                log.warning(
                    "auto-crop %s: taking %s on CONFIDENCE ALONE (%.3f -> %.3f); the "
                    "geometry gate needs a camera rig and a second frame set",
                    target.view_name,
                    box,
                    conf_incumbent,
                    conf,
                )
        res.seconds = time.perf_counter() - started
        out.append(res)
        log.info(
            "auto-crop %s: %s conf %.3f -> %s conf %.3f (%d probes, %.1f s) -- %s",
            target.view_name,
            incumbent,
            conf_incumbent,
            res.box,
            res.conf,
            res.probes,
            res.seconds,
            "ACCEPTED" if res.accepted else "kept the incumbent",
        )
    return out


def _joint_views(plan, models, target: AutoCropTarget, windows, incumbents):
    """Prepared inputs for every view of a joint-view model, plus the target's slot(s).

    The other views are constant across candidates, so they are prepared once. A view whose
    own crop is still unresolved is prepared through its incumbent -- the coupling is real
    (this model's views inform each other), which is why the caller runs a second pass once
    every box is known.
    """
    import torch

    users = [i for i, pw in enumerate(plan.pathways) if pw.model == target.model]
    prepared, slots = [], []
    for local, pw_idx in enumerate(users):
        pw = plan.pathways[pw_idx]
        transform = pw.transform
        auto = transform.auto_crop
        if auto is not None and not auto.resolved:
            frames = windows[pw.source]
            hw = (int(frames.shape[-3]), int(frames.shape[-2]))
            aspect = (
                auto.seed[2] / auto.seed[3]
                if auto.seed is not None
                else models[pw.model].input_size[1] / models[pw.model].input_size[0]
            )
            box = (incumbents or {}).get(
                pw.preprocessor, _incumbent_box(auto, hw, aspect)
            )
            transform = transform.resolve_auto_crop(box)
        prepared.append(models[pw.model].prepare(transform.apply(windows[pw.source])))
        if pw.name in target.pathways:
            slots.append(local)
    assert isinstance(prepared[0], torch.Tensor)
    return prepared, tuple(slots or (0,))


def _fill_incumbents(plan, models, windows, incumbents):
    """Resolve any automatic crop still unresolved in ``plan`` to its incumbent.

    The gate detects through the WHOLE plan -- that is how it gets the other cameras' 3D --
    so no view may be left without a window, including one whose own search has not run yet.
    """
    base = dict(incumbents or {})
    fill = {}
    for t in targets(resolved_plan(plan, base)):
        auto = t.transform.auto_crop
        assert auto is not None
        frames = windows[t.source]
        hw = (int(frames.shape[-3]), int(frames.shape[-2]))
        aspect = (
            auto.seed[2] / auto.seed[3]
            if auto.seed is not None
            else models[t.model].input_size[1] / models[t.model].input_size[0]
        )
        fill[t.preprocessor] = _incumbent_box(auto, hw, aspect)
    return {**base, **fill}


def _compass_refine(
    evaluate, start, aspect: float, frame_hw, budget: int, log_line=None
):
    """Compass search over ``(cx, cy, width)`` from ``start``, minimizing ``evaluate``.

    Six trial points an iteration (each axis, both directions); move to the best improvement
    or halve the step. A derivative-free pattern search rather than anything cleverer,
    because ``evaluate`` is an expensive black box (a whole detection pass) with a smooth
    unimodal basin already bracketed by the cheap confidence search -- exactly the regime
    where a compass search is the right tool and a gradient is not available.

    ``evaluate(box) -> float`` may return ``inf`` for a refused box (one that clips), which
    simply makes that direction unattractive. Stops on the evaluation ``budget`` or when the
    step falls below :data:`GATE_STEP_FLOOR`, whichever comes first, so the cost is bounded
    and predictable.

    Returns ``(box, value, n_evals, visited)``.
    """
    best = tuple(int(v) for v in start)
    value = evaluate(best)
    visited = {best: value}
    used = 1
    dc, dw = GATE_STEP_CENTRE, GATE_STEP_WIDTH
    while used < budget and dc > GATE_STEP_FLOOR:
        x0, y0, w0, h0 = best
        cx, cy = x0 + w0 / 2.0, y0 + h0 / 2.0
        trials = [
            _fit(cx + dc * w0, cy, w0, aspect, frame_hw),
            _fit(cx - dc * w0, cy, w0, aspect, frame_hw),
            _fit(cx, cy + dc * h0, w0, aspect, frame_hw),
            _fit(cx, cy - dc * h0, w0, aspect, frame_hw),
            _fit(cx, cy, w0 * dw, aspect, frame_hw),
            _fit(cx, cy, w0 / dw, aspect, frame_hw),
        ]
        moved = False
        for box in dict.fromkeys(trials):
            if box == best or box in visited or used >= budget:
                continue
            got = evaluate(box)
            visited[box] = got
            used += 1
            if got < value:
                best, value, moved = box, got, True
        if not moved:
            dc, dw = dc / 2.0, 1 + (dw - 1) / 2.0
            if log_line:
                log_line(
                    f"step -> {dc:.3f} after no improvement on {best} ({value:.1f} px)"
                )
    return best, value, used, visited


def _choose_by_agreement(
    plan,
    models,
    cameras,
    gate_windows,
    target: AutoCropTarget,
    finalists,
    incumbent,
    res: Resolution,
    batch_size: int,
    gate_frames,
    incumbents,
    params,
    *,
    prober: _Prober,
    prober_frames: int,
    aspect: float,
) -> None:
    """Choose the box by agreement with the other cameras' 3D; fills in ``res``.

    One held-out reference scores any number of boxes, so geometry does two jobs here that
    the original paired design could not: it *refuses* (keeping the incumbent when nothing
    beats it, which is what stops a confidently-wrong box) and it *chooses*, walking from
    where confidence left off to the box that actually agrees best -- see
    :data:`GATE_STEP_CENTRE` for why that second job is worth its cost.
    """
    base = _fill_incumbents(plan, models, gate_windows, incumbents)
    ref, rig_px, why_not = _reference(
        resolved_plan(plan, {**base, target.preprocessor: incumbent}),
        models,
        cameras,
        gate_windows,
        target,
        batch_size,
    )
    if ref is None:
        box, conf = finalists[0]
        res.accepted = box != incumbent
        res.box = box if res.accepted else incumbent
        res.conf = conf
        res.notes.append(f"the geometry gate could not run: {why_not}")
        res.notes.append(
            "so this box was chosen on confidence alone -- check the pose2d overlay"
        )
        log.warning("auto-crop %s: gate unusable -- %s", target.view_name, why_not)
        return

    res.gated = True
    res.gate_frames = tuple(int(i) for i in gate_frames)
    res.rig_residual_px = rig_px

    def evaluate(box):
        return _agreement(
            resolved_plan(plan, {**base, target.preprocessor: box}),
            models,
            gate_windows,
            target,
            ref,
            batch_size,
        )

    res.agreement_incumbent_px = evaluate(incumbent)
    # Every confidence finalist is a candidate start; the search then walks from the best of
    # them. Starting from several costs one evaluation each and protects against a
    # confidence winner that is in the wrong basin altogether.
    starts = [(box, evaluate(box)) for box, _ in finalists]
    for (box, _), (_, agree) in zip(finalists, starts):
        log.info(
            "auto-crop %s: confidence finalist %s agrees to %.1f px",
            target.view_name,
            box,
            agree,
        )
    start = min(starts, key=lambda s: s[1])[0]
    budget = max(1, int(params.gate_evals) - len(starts) - 1)
    box, agree, used, _ = _compass_refine(
        evaluate,
        start,
        aspect,
        prober.frame_hw,
        budget,
        log_line=lambda msg: log.debug("auto-crop %s: %s", target.view_name, msg),
    )
    res.gate_evals = len(starts) + 1 + used
    best_conf = float(prober.score([box], prober_frames)[0])
    res.accepted = bool(agree < res.agreement_incumbent_px and box != incumbent)
    res.box = box if res.accepted else incumbent
    res.conf = best_conf if res.accepted else res.conf_incumbent
    res.agreement_px = agree if res.accepted else res.agreement_incumbent_px
    if res.accepted and box != finalists[0][0]:
        res.notes.append(
            f"geometry moved the box from the most confident candidate "
            f"{finalists[0][0]} ({starts[0][1]:.1f} px) to {box} ({agree:.1f} px)"
        )
    log.info(
        "auto-crop %s: agreement with the other cameras' 3D %.1f px -> %.1f px "
        "(%d evaluations, rig self-consistency %.1f px)",
        target.view_name,
        res.agreement_incumbent_px,
        res.agreement_px,
        res.gate_evals,
        rig_px,
    )
    if res.agreement_px > params.agreement_warn_px:
        res.notes.append(
            f"this view still does not frame the animal: it agrees with the other "
            f"cameras' 3D only to {res.agreement_px:.0f} px. The gate can say 'better', "
            f"never 'good' -- set the box by eye or drop the view from the fit."
        )
        log.warning(
            "auto-crop %s: STILL not framing the animal -- %.0f px from the other "
            "cameras' 3D. The gate can say 'better', never 'good'.",
            target.view_name,
            res.agreement_px,
        )


# -- frames ------------------------------------------------------------------------------


def split_frame_indices(count: int, n_search: int, n_gate: int):
    """Two disjoint, evenly spread frame index sets: ``(search, gate)``.

    One :func:`numpy.linspace` spread over the whole recording, split by taking evenly
    spaced members for the search and leaving the rest to the gate. Both therefore span the
    recording -- see the module docstring on why a contiguous sample is one moment measured
    several times, and how ranking frames by "how much 3D they already have" silently
    produced exactly that.
    """
    total = max(1, int(n_search) + max(0, int(n_gate)))
    count = max(1, int(count))
    spread = np.unique(np.linspace(0, count - 1, min(total, count)).round().astype(int))
    n_search = max(1, min(int(n_search), len(spread)))
    pick = np.unique(np.linspace(0, len(spread) - 1, n_search).round().astype(int))
    search = spread[pick]
    gate = np.array([i for i in spread if i not in set(search.tolist())], dtype=int)
    return [int(i) for i in search], [int(i) for i in gate]


def decode_frames(
    files: dict[str, list], indices: list[int], *, gray_ok: bool = False
) -> dict[str, np.ndarray]:
    """``source -> (T, H, W, C)`` for the given frame indices, decoded by SEEK.

    One cursor per source, held open across its reads (:meth:`FrameReader.cursor`), because
    the indices are spread over the whole recording: decoding forward to each of them would
    walk the file several times over.

    ``gray_ok`` keeps a color-free frame at one channel instead of repeating the luma three
    times -- pass it only when every model that will see these frames declares
    :attr:`~deeperfly.pose2d.models.LoadedModel.accepts_gray`, since a search prepares its
    candidates through the same model the detection will use.
    """
    from .. import io

    out: dict[str, np.ndarray] = {}
    for name, src in files.items():
        reader = io.open_reader(src)
        # same_as_rgb: these frames are the search's arithmetic, not a picture -- a
        # range-scaled gray would score boxes on different pixels from the detection.
        cursor = reader.cursor(gray_ok=gray_ok, same_as_rgb=True)
        try:
            frames = [np.asarray(cursor.frame(int(i))) for i in indices]
        finally:
            cursor.close()
        # A cursor drops the channel axis when it hands over gray; the models want it back,
        # as one channel when they asked for gray and as three when they did not.
        stack = np.stack(
            [
                f
                if f.ndim == 3
                else (f[..., None] if gray_ok else np.repeat(f[..., None], 3, axis=-1))
                for f in frames
            ]
        )
        out[name] = stack
    return out


def frame_count(files: dict[str, list], sources: list[str]) -> int | None:
    """The smallest frame count over ``sources`` (``None`` if none of them report one)."""
    from .. import io

    counts = []
    for name in sources:
        src = files.get(name)
        if src is None:
            continue
        try:
            n = io.open_reader(src).count()
        except Exception:  # noqa: BLE001 -- a source that cannot say is not an error here
            n = None
        if n:
            counts.append(int(n))
    return min(counts) if counts else None


# -- the entry point the pipeline uses ---------------------------------------------------


def ensure_resolved(
    config,
    plan,
    *,
    models: dict,
    cameras=None,
    sources: dict[str, list[Path]] | None = None,
    input=None,
    outdir: Path | None = None,
    force: bool = False,
):
    """The plan with every automatic crop resolved: from the sidecar, or by searching.

    The order matters. A recorded box is reused (so a resumed run does not re-search, and
    the box the cached detections were computed through is the one every later stage sees);
    only what is left over is searched, and the result is written back.

    Parameters
    ----------
    config
        The run config (frame counts, ``[pose2d.autocrop]``, the forward batch).
    plan
        The detection plan, possibly carrying unresolved automatic crops.
    models
        ``name -> LoadedModel`` for the plan's models, already on the device.
    cameras
        The rig for the accept gate; ``None`` skips the gate (and says so).
    sources, input
        The footage (see :func:`deeperfly.recordings.source_sources`).
    outdir
        Where to read and write :data:`SIDECAR_NAME`. ``None`` searches without persisting.
    force
        Search even when a box is already recorded (``--overwrite pose2d``).

    Returns
    -------
    tuple
        ``(plan, resolutions)`` -- the resolved plan, and the searches that ran (empty when
        everything came from the sidecar).
    """
    from ..recordings import source_sources

    if not targets(plan):
        return plan, []

    recorded = {} if force else read_sidecar(outdir) if outdir else {}
    if recorded:
        plan = resolved_plan(plan, recorded)
        for name, box in recorded.items():
            log.info("auto-crop %s: reusing the recorded box %s", name, tuple(box))
    todo = targets(plan)
    if not todo:
        return plan, []

    params = config.autocrop
    files = dict(source_sources(config, sources=sources, input=input))
    # Every source of every model that owns a target: a joint-view model needs its other
    # views to probe at all, and the gate needs all of them to triangulate.
    models_involved = {t.model for t in todo}
    needed = sorted(
        {pw.source for pw in plan.pathways if pw.model in models_involved}
        | {t.source for t in todo}
    )
    missing = [s for s in needed if s not in files]
    if missing:
        raise SystemExit(
            f"auto-crop needs footage for source(s) {missing} and the recording has none"
        )
    count = frame_count(files, needed)
    if count is None:
        raise SystemExit(
            "auto-crop cannot tell how long this recording is, so it cannot spread its "
            "sample frames over it; write the crop boxes explicitly instead"
        )
    search_idx, gate_idx = split_frame_indices(
        count, params.search_frames, params.gate_frames if cameras is not None else 0
    )
    log.info(
        "auto-crop: searching %d crop(s) on frames %s of %d, gating on %s",
        len(todo),
        search_idx,
        count,
        gate_idx or "nothing (no rig)",
    )
    wanted = {s: files[s] for s in needed}
    # These frames are probed through the plan's own models, so they may be grayscale on
    # exactly the condition detection uses (see
    # :func:`deeperfly.pose2d.stream.detect_2d`): every model would make them gray anyway.
    gray_ok = bool(models) and all(
        bool(getattr(m, "accepts_gray", False)) for m in models.values()
    )
    search_windows = decode_frames(wanted, search_idx, gray_ok=gray_ok)
    gate_windows = (
        decode_frames(wanted, gate_idx, gray_ok=gray_ok) if gate_idx else None
    )

    # A joint-view model computes its views together, so while one view's box is unknown
    # every other view is probed through a frame that view will not finally use. One extra
    # pass, once every box is known, removes that -- and because each pass only moves a box
    # the gate prefers, the second can only improve on the first.
    passes = (
        2
        if len(todo) > 1 and any(models[m].joint_views for m in models_involved)
        else 1
    )
    resolutions: list[Resolution] = []
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for p in range(passes):
        if p:
            log.info("auto-crop: second pass, now that every view has a box")
        resolutions = search(
            plan,
            models,
            search_windows=search_windows,
            gate_windows=gate_windows,
            cameras=cameras,
            params=params,
            batch_size=config.pose2d.batch_size,
            search_frames=tuple(search_idx),
            gate_frames=tuple(gate_idx),
            incumbents=boxes or None,
        )
        boxes = {r.preprocessor: r.box for r in resolutions}
    plan = resolved_plan(plan, boxes)
    if outdir is not None:
        merged = [*_recorded_resolutions(recorded, resolutions), *resolutions]
        path = write_sidecar(outdir, merged)
        log.info("auto-crop: wrote %s", path)
    return plan, resolutions


def _recorded_resolutions(recorded, fresh) -> list[Resolution]:
    """Sidecar entries this run did not re-search, so writing it back keeps them."""
    done = {r.preprocessor for r in fresh}
    return [
        Resolution(
            preprocessor=name,
            view_name="",
            box=tuple(box),
            incumbent=tuple(box),
            seeded=False,
            accepted=True,
            notes=["carried over from a previous run"],
        )
        for name, box in recorded.items()
        if name not in done
    ]
