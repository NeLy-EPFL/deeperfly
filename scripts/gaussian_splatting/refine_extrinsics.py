"""Photometric camera-extrinsics refinement from a Gaussian splat — a study.

Explores how gsplat's differentiable rendering can refine deeperfly's camera
extrinsics, and *why many synchronized frames help*. Runs in ``.venv-gsplat``.

Setup that makes this work:
  * The 7 cameras are STATIC across all frames, so their extrinsics are shared
    parameters constrained by every frame at once.
  * Each frame's fly geometry is INDEPENDENT (the legs move), fit as its own
    per-frame Gaussian cloud.
  * Therefore a systematic extrinsic error produces a residual that is
    consistent across frames but that per-frame geometry cannot absorb (it
    would need to lie the same way in every independent cloud). Sharing cameras
    across K frames is what stops the "cameras leak into the geometry" failure.

Parameterisation (grounded in the near-orthographic degeneracy analysis):
  * Per camera: a left-multiplied SE(3) delta  T = Exp([omega | u]) . T_init,
    with omega in so(3) (3-DoF rotation) and u the translation.
  * Optical-axis translation u_z is LOCKED by default (dolly is unobservable
    under a ~2.5 deg FOV); only in-plane u_x,u_y are free -> 5-DoF/camera.
  * One camera (default the head-on ``f``) is fully ANCHORED to fix the global
    similarity gauge.

Validation is controlled perturbation-recovery: treat deeperfly's bundle-
adjusted cameras as reference, apply a known extrinsic perturbation, then see
whether photometric refinement pulls the cameras back. Metrics:
  * per-camera geodesic rotation error + camera-centre error (split into the
    in-plane and optical-axis components), refined vs reference;
  * keypoint reprojection RMSE (deeperfly's native metric; needs no Gaussians);
  * all reported for perturbed-init vs refined.

Example:
  .venv-gsplat/bin/python scripts/gaussian_splatting/refine_extrinsics.py \
      --frames <gs_data>/frame_8,<gs_data>/frame_24,<gs_data>/frame_40 \
      --perturb rot_trans --rot-deg 1.0 --trans 0.6 --tag multi4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from fit_gsplat import build_params, load_data, make_optimizers, render, ssim


def maybe_blur(x_vhwc: torch.Tensor, sigma: float) -> torch.Tensor:
    """Coarse-to-fine: Gaussian-blur (V,H,W,3) images to widen the pose basin."""
    if sigma < 0.5:
        return x_vhwc
    x = x_vhwc.permute(0, 3, 1, 2)
    k = int(2 * round(3 * sigma) + 1)
    x = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    return x.permute(0, 2, 3, 1)


# --------------------------------------------------------------------------- #
# SE(3) / so(3) helpers (torch, differentiable)
# --------------------------------------------------------------------------- #
def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Batched so(3) exponential: (...,3) axis-angle -> (...,3,3) rotation."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (...,1)
    k = omega / theta
    K = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    K[..., 0, 1] = -kz
    K[..., 0, 2] = ky
    K[..., 1, 0] = kz
    K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky
    K[..., 2, 1] = kx
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    s = torch.sin(theta)[..., None]
    c = (1 - torch.cos(theta))[..., None]
    return eye + s * K + c * (K @ K)


def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> np.ndarray:
    """Per-matrix geodesic angle (degrees) between two rotation stacks (V,3,3)."""
    rel = Ra @ Rb.transpose(-1, -2)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    ang = torch.arccos(((tr - 1) / 2).clamp(-1, 1))
    return torch.rad2deg(ang).detach().cpu().numpy()


class CameraDeltas(torch.nn.Module):
    """Shared per-camera SE(3) deltas over a fixed reference rig."""

    def __init__(self, viewmats_ref: torch.Tensor, anchor: int, lock_tz: bool):
        super().__init__()
        V = viewmats_ref.shape[0]
        self.register_buffer("R_ref", viewmats_ref[:, :3, :3].clone())
        self.register_buffer("t_ref", viewmats_ref[:, :3, 3].clone())
        free = torch.ones(V, 1)
        free[anchor] = 0.0  # anchor camera: delta forced to zero (gauge fix)
        self.register_buffer("free", free)
        self.register_buffer("lock_tz", torch.tensor(float(lock_tz)))
        self.omega = torch.nn.Parameter(torch.zeros(V, 3))
        self.u = torch.nn.Parameter(torch.zeros(V, 3))

    def viewmats(self) -> torch.Tensor:
        omega = self.omega * self.free
        u = self.u * self.free
        if float(self.lock_tz) > 0.5:
            u = u * torch.tensor([1.0, 1.0, 0.0], device=u.device)  # lock optical axis
        Rd = so3_exp(omega)  # (V,3,3)
        R_new = Rd @ self.R_ref
        t_new = (Rd @ self.t_ref.unsqueeze(-1)).squeeze(-1) + u
        V = R_new.shape[0]
        vm = torch.eye(4, device=R_new.device).repeat(V, 1, 1)
        vm[:, :3, :3] = R_new
        vm[:, :3, 3] = t_new
        return vm


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def camera_centers(viewmats: torch.Tensor) -> torch.Tensor:
    R = viewmats[:, :3, :3]
    t = viewmats[:, :3, 3]
    return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)


def center_error_split(vm: torch.Tensor, vm_ref: torch.Tensor):
    """Camera-centre error, split into optical-axis vs in-plane components."""
    C = camera_centers(vm)
    Cr = camera_centers(vm_ref)
    d = C - Cr  # (V,3)
    axis = vm_ref[:, 2, :3]  # camera forward (world) = 3rd row of R_ref
    axis = axis / axis.norm(dim=-1, keepdim=True)
    along = (d * axis).sum(-1)  # signed optical-axis component
    perp = (d - along.unsqueeze(-1) * axis).norm(dim=-1)
    return (
        d.norm(dim=-1).detach().cpu().numpy(),
        perp.detach().cpu().numpy(),
        along.detach().cpu().numpy(),
    )


def reproj_rmse(vm: torch.Tensor, Ks: torch.Tensor, kp_list):
    """Keypoint reprojection RMSE (px) over frames, deeperfly's native metric."""
    R = vm[:, :3, :3]
    t = vm[:, :3, 3]
    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]
    errs = []
    for p3d, p2d in kp_list:  # p3d (P,3), p2d (V,P,2)
        cam = torch.einsum("vij,pj->vpi", R, p3d) + t[:, None, :]
        z = cam[..., 2].clamp_min(1e-6)
        u = fx[:, None] * cam[..., 0] / z + cx[:, None]
        v = fy[:, None] * cam[..., 1] / z + cy[:, None]
        uv = torch.stack([u, v], -1)
        ok = torch.isfinite(p2d).all(-1)
        errs.append(((uv - p2d)[ok] ** 2).sum(-1))
    e = torch.cat(errs)
    return float(torch.sqrt(e.mean()).item())


def cauchy(residual_abs: torch.Tensor, f: float) -> torch.Tensor:
    """Cauchy robust penalty on |residual| (same family as deeperfly's BA loss)."""
    return (f * f / 2.0) * torch.log1p((residual_abs / f) ** 2)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frames", required=True, help="comma-separated frame dataset dirs"
    )
    ap.add_argument(
        "--anchor", type=int, default=3, help="index of fully-fixed camera (f=3)"
    )
    ap.add_argument(
        "--perturb", choices=["rot_trans", "tz", "none"], default="rot_trans"
    )
    ap.add_argument(
        "--rot-deg",
        type=float,
        default=0.1,
        help="rotation perturb (deg); note fx=22388 => 0.1deg ~ 39px image shift",
    )
    ap.add_argument(
        "--trans", type=float, default=0.1, help="in-plane perturb (world units)"
    )
    ap.add_argument(
        "--tz", type=float, default=4.0, help="optical-axis perturb (world units)"
    )
    ap.add_argument(
        "--free-tz", action="store_true", help="let refinement move along optical axis"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="frozen gaussians .pt: iNeRF-style camera-only "
        "refinement (isolates observability from geometry absorption). Single frame.",
    )
    ap.add_argument(
        "--init-count", type=int, default=45000, help="fixed gaussians per frame"
    )
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--refine", type=int, default=2400)
    ap.add_argument("--sh-degree", type=int, default=1)
    ap.add_argument("--cauchy-f", type=float, default=0.1)
    ap.add_argument(
        "--rot-lr", type=float, default=3e-4, help="Adam lr for so(3) delta (rad)"
    )
    ap.add_argument(
        "--trans-lr",
        type=float,
        default=1e-3,
        help="Adam lr for in-plane delta (x scene)",
    )
    ap.add_argument(
        "--blur0",
        type=float,
        default=5.0,
        help="initial coarse-to-fine blur sigma (px)",
    )
    ap.add_argument(
        "--blur-floor",
        type=float,
        default=0.0,
        help="min blur sigma (stabilises joint runs)",
    )
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    frame_dirs = [Path(p) for p in args.frames.split(",")]
    frames = [load_data(fd, device) for fd in frame_dirs]
    Ks = frames[0]["Ks"]
    W, H = frames[0]["W"], frames[0]["H"]
    scene_scale = float(np.mean([f["radius"] for f in frames]))
    vm_ref = frames[0]["viewmats"].clone()  # reference (deeperfly BA) cameras
    V = vm_ref.shape[0]

    # keypoints for the reprojection metric (needs no Gaussians)
    kp_list = []
    for fd in frame_dirs:
        kp = np.load(fd / "keypoints.npz")
        kp_list.append(
            (
                torch.tensor(kp["points3d"], dtype=torch.float32, device=device),
                torch.tensor(kp["pts2d"], dtype=torch.float32, device=device),
            )
        )

    # ---- known perturbation -> perturbed init cameras ----
    rng = np.random.default_rng(args.seed)
    omega0 = np.zeros((V, 3))
    u0 = np.zeros((V, 3))
    for i in range(V):
        if i == args.anchor:
            continue
        if args.perturb == "rot_trans":
            ax = rng.normal(size=3)
            omega0[i] = ax / np.linalg.norm(ax) * math.radians(args.rot_deg)
            d = rng.normal(size=2)
            u0[i, :2] = d / np.linalg.norm(d) * args.trans
        elif args.perturb == "tz":
            u0[i, 2] = args.tz * (1 if i % 2 else -1)
    with torch.no_grad():
        Rd = so3_exp(torch.tensor(omega0, dtype=torch.float32, device=device))
        R_pert = Rd @ vm_ref[:, :3, :3]
        t_pert = (Rd @ vm_ref[:, :3, 3].unsqueeze(-1)).squeeze(-1) + torch.tensor(
            u0, dtype=torch.float32, device=device
        )
        vm_pert = torch.eye(4, device=device).repeat(V, 1, 1)
        vm_pert[:, :3, :3] = R_pert
        vm_pert[:, :3, 3] = t_pert

    # ---- per-frame Gaussians (fixed count, no densification) ----
    per_frame = []
    for f in frames:
        # densify the init cloud to init_count with random fill around the fly
        pts, cols = f["points"], f["colors"]
        need = args.init_count - pts.shape[0]
        if need > 0:
            c = torch.tensor(f["center"], dtype=torch.float32, device=device)
            r = f["radius"]
            extra = c + torch.randn(need, 3, device=device) * (0.45 * r)
            pts = torch.cat([pts, extra])
            cols = torch.cat([cols, torch.full((need, 3), 0.5, device=device)])
        d = {
            "points": pts,
            "colors": cols,
            "center": f["center"],
            "radius": f["radius"],
        }
        params = build_params(d, args.sh_degree, 0.02 * scene_scale, 0.1, device)
        gopt = make_optimizers(params, scene_scale)
        per_frame.append((params, gopt, f))

    frozen_sh = args.sh_degree
    if args.model:
        # iNeRF-style: load a fitted model, freeze it, optimise cameras only.
        # No geometry to absorb the pose error -> clean observability + basin test.
        ck = torch.load(args.model, map_location=device)
        frozen = {
            k: ck[k].to(device).detach()
            for k in ["means", "scales", "quats", "opacities", "sh0", "shN"]
        }
        frozen_sh = int(
            round(math.sqrt(frozen["sh0"].shape[1] + frozen["shN"].shape[1]) - 1)
        )
        per_frame = [(frozen, None, frames[0])]  # gopt=None -> geometry frozen
        args.warmup = 0
        print(
            f"  [frozen-geometry mode] {frozen['means'].shape[0]} gaussians, sh_degree={frozen_sh}"
        )

    cam = CameraDeltas(vm_pert, anchor=args.anchor, lock_tz=not args.free_tz).to(device)
    cam_opt = torch.optim.Adam(
        [
            {"params": [cam.omega], "lr": args.rot_lr},
            {"params": [cam.u], "lr": args.trans_lr * scene_scale},
        ]
    )
    # capture base LRs so we can cosine-decay them as the blur sharpens
    cam_base = [g["lr"] for g in cam_opt.param_groups]
    geo_base = [
        None if gopt is None else {k: o.param_groups[0]["lr"] for k, o in gopt.items()}
        for _, gopt, _ in per_frame
    ]

    def set_cam_scale(scale):
        for g, b in zip(cam_opt.param_groups, cam_base):
            g["lr"] = b * scale

    def set_geo_scale(gopt, base, scale):
        for k, o in gopt.items():
            o.param_groups[0]["lr"] = base[k] * scale

    def photo_loss(out, alpha, gtd, sigma):
        bg_gt = gtd["_bg"]
        gt_comp = gtd["gt"] * gtd["mask"] + bg_gt[:, None, None, :] * (1 - gtd["mask"])
        out_b, gt_b = maybe_blur(out, sigma), maybe_blur(gt_comp, sigma)
        l1 = cauchy((out_b - gt_b).abs(), args.cauchy_f).mean()
        s = 1 - ssim(out_b, gt_b)
        a = (alpha - gtd["mask"]).abs().mean()
        return 0.8 * l1 + 0.2 * s + 0.5 * a

    def report(tag, vm):
        rot = geodesic_deg(vm[:, :3, :3], vm_ref[:, :3, :3])
        tot, perp, along = center_error_split(vm, vm_ref)
        rmse = reproj_rmse(vm, Ks, kp_list)
        free = [i for i in range(V) if i != args.anchor]
        return {
            "tag": tag,
            "reproj_rmse_px": round(rmse, 3),
            "rot_err_deg_mean": round(float(rot[free].mean()), 4),
            "rot_err_deg_max": round(float(rot[free].max()), 4),
            "center_err_perp_mean": round(float(perp[free].mean()), 4),
            "center_err_along_mean": round(float(np.abs(along[free]).mean()), 4),
            "per_cam_rot_deg": [round(float(x), 3) for x in rot],
            "per_cam_center_perp": [round(float(x), 3) for x in perp],
            "per_cam_center_along": [round(float(x), 3) for x in along],
        }

    total = args.warmup + args.refine
    print(
        f"[{args.tag}] {len(frames)} frame(s), {V} cams, anchor={args.anchor}, "
        f"lock_tz={not args.free_tz}, perturb={args.perturb}"
    )
    m_ref = report("reference(0-perturb)", vm_ref)
    m_pert = report("perturbed-init", vm_pert)
    print(
        f"  reproj RMSE  ref={m_ref['reproj_rmse_px']}  perturbed={m_pert['reproj_rmse_px']} px"
    )
    print(
        f"  perturb: rot {m_pert['rot_err_deg_mean']} deg (max {m_pert['rot_err_deg_max']}), "
        f"center perp {m_pert['center_err_perp_mean']}, along-axis {m_pert['center_err_along_mean']}"
    )

    best = {"rmse": float("inf"), "vm": None, "step": -1}
    for step in range(total):
        refining = step >= args.warmup
        # coarse-to-fine: sharp geometry during warmup, then blur->sharp while poses move,
        # with camera + geometry LRs cosine-decaying to ~0 as the blur vanishes so the
        # sharp (steep, huge-focal) landscape can't blow the poses up.
        if refining:
            p = (step - args.warmup) / max(1, args.refine)
            sigma = max(args.blur_floor, args.blur0 * max(0.0, 1 - p / 0.7))
            decay = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))  # 1 -> 0
            set_cam_scale(decay)
            for (_, gopt, _), gb in zip(per_frame, geo_base):
                if gopt is not None:
                    set_geo_scale(
                        gopt, gb, 0.3 * decay
                    )  # stiffen geometry so cameras carry the fix
        else:
            sigma = 0.0
        vm = cam.viewmats()
        infos = []
        loss = 0.0
        for params, gopt, f in per_frame:
            bg = torch.rand(V, 3, device=device)
            f["_bg"] = bg
            deg = frozen_sh if args.model else min(step // 500, args.sh_degree)
            out, alpha, info = render(params, vm, Ks, W, H, deg, bg, 1.0, 1000.0, True)
            loss = loss + photo_loss(out, alpha, f, sigma)
            infos.append((params, gopt))
        loss = loss / len(per_frame)
        cam_opt.zero_grad(set_to_none=True)
        for params, gopt in infos:
            if gopt is not None:
                for o in gopt.values():
                    o.zero_grad(set_to_none=True)
        loss.backward()
        for params, gopt in infos:
            if gopt is not None:
                for o in gopt.values():
                    o.step()
        if refining:
            cam_opt.step()
        if refining and step % 50 == 0:
            with torch.no_grad():
                r = reproj_rmse(cam.viewmats(), Ks, kp_list)
            if r < best["rmse"]:
                best = {"rmse": r, "vm": cam.viewmats().detach().clone(), "step": step}
        if step % 300 == 0 or step == total - 1:
            with torch.no_grad():
                rmse = reproj_rmse(cam.viewmats(), Ks, kp_list)
            phase = "refine" if refining else "warmup"
            print(
                f"  step {step:4d} [{phase}] loss {loss.item():.4f}  reproj {rmse:.3f}px"
            )

    vm_final = cam.viewmats().detach()
    m_final = report("refined", vm_final)
    m_best = report("refined-best", best["vm"] if best["vm"] is not None else vm_final)
    m_best["best_step"] = best["step"]

    print(f"\n[{args.tag}] RESULTS (mean over {len(frames)} frame(s), free cams only)")
    print(
        f"  {'metric':<26}{'reference':>11}{'perturbed':>11}{'refined':>11}{'best':>11}"
    )
    for key, lbl in [
        ("reproj_rmse_px", "reproj RMSE (px)"),
        ("rot_err_deg_mean", "rotation err (deg)"),
        ("center_err_perp_mean", "centre err in-plane"),
        ("center_err_along_mean", "centre err optical-axis"),
    ]:
        print(
            f"  {lbl:<26}{m_ref[key]:>11}{m_pert[key]:>11}{m_final[key]:>11}{m_best[key]:>11}"
        )
    print(f"  (best @ step {m_best['best_step']})")

    out_dir = frame_dirs[0].parent / "refine"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"metrics_{args.tag}.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "reference": m_ref,
                "perturbed": m_pert,
                "refined": m_final,
                "refined_best": m_best,
                "n_frames": len(frames),
                "scene_scale": scene_scale,
            },
            indent=2,
        )
    )
    print(f"  wrote {out_dir / f'metrics_{args.tag}.json'}")


if __name__ == "__main__":
    main()
