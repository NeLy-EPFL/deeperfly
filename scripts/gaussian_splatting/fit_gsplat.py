"""Fit a 3D Gaussian Splat to one frame from the 7 calibrated deeperfly cameras.

Runs in the isolated gsplat venv (``.venv-gsplat``); it does NOT import
deeperfly. Consumes the dataset written by ``prepare_data.py`` (cameras +
frame + fly masks + an init point cloud seeded from the triangulated
keypoints), optimises a set of 3D Gaussians against the 7 views with a
mask-aware photometric loss, and renders (a) the training views next to the
ground truth and (b) a novel-view orbit -- the visual sanity check.

Camera convention is OpenCV world->camera, matching both deeperfly and gsplat,
so ``viewmats``/``Ks`` from the dataset feed straight into ``rasterization``.

Usage (from repo root):
    .venv-gsplat/bin/python scripts/gaussian_splatting/fit_gsplat.py \
        --data <gs_data/frame_32> --iters 7000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

C0 = 0.28209479177387814  # SH band-0 constant


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simplified global SSIM on (C,H,W,3) tensors in [0,1]."""
    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 1, 2)
    mu_a = F.avg_pool2d(a, 11, 1, 5)
    mu_b = F.avg_pool2d(b, 11, 1, 5)
    sa = F.avg_pool2d(a * a, 11, 1, 5) - mu_a**2
    sb = F.avg_pool2d(b * b, 11, 1, 5) - mu_b**2
    sab = F.avg_pool2d(a * b, 11, 1, 5) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sa + sb + c2)
    )
    return s.mean()


def look_at(eye: np.ndarray, target: np.ndarray, up=(0, 0, 1)) -> np.ndarray:
    """OpenCV world->camera 4x4 for a camera at ``eye`` looking at ``target``."""
    up = np.asarray(up, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    vm = np.eye(4)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ eye
    return vm


def load_data(data: Path, device):
    cams = np.load(data / "cameras.npz", allow_pickle=True)
    names = [str(n) for n in cams["names"]]
    W, H = int(cams["width"]), int(cams["height"])
    viewmats = torch.tensor(cams["viewmats"], dtype=torch.float32, device=device)
    Ks = torch.tensor(cams["Ks"], dtype=torch.float32, device=device)
    imgs, masks = [], []
    for i, nm in enumerate(names):
        im = cv2.cvtColor(
            cv2.imread(str(data / "images" / f"{i}_{nm}.png")), cv2.COLOR_BGR2RGB
        )
        mk = cv2.imread(str(data / "masks" / f"{i}_{nm}.png"), cv2.IMREAD_GRAYSCALE)
        imgs.append(im)
        masks.append(mk)
    gt = (
        torch.tensor(np.stack(imgs), dtype=torch.float32, device=device) / 255.0
    )  # (V,H,W,3)
    mask = (
        torch.tensor(np.stack(masks), dtype=torch.float32, device=device)[..., None]
        / 255.0
    )
    init = np.load(data / "init.npz")
    return dict(
        names=names,
        W=W,
        H=H,
        viewmats=viewmats,
        Ks=Ks,
        gt=gt,
        mask=mask,
        points=torch.tensor(init["points"], dtype=torch.float32, device=device),
        colors=torch.tensor(init["colors"], dtype=torch.float32, device=device),
        center=np.asarray(init["center"], float),
        radius=float(init["radius"]),
    )


def build_params(d, sh_degree, init_scale, init_opacity, device):
    N = d["points"].shape[0]
    K = (sh_degree + 1) ** 2
    scales = torch.full((N, 3), math.log(init_scale), device=device)
    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0
    opac = torch.full((N,), math.log(init_opacity / (1 - init_opacity)), device=device)
    sh0 = rgb_to_sh(d["colors"]).unsqueeze(1)  # (N,1,3)
    shN = torch.zeros((N, K - 1, 3), device=device)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(d["points"].clone()),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opac),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }
    ).to(device)
    return params


def make_optimizers(params, scene_scale, lr_scale=1.0):
    lrs = {
        "means": 1.6e-4 * scene_scale * lr_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    return {
        k: torch.optim.Adam([{"params": params[k], "lr": lrs[k]}], eps=1e-15)
        for k in params
    }


def render(params, viewmats, Ks, W, H, sh_degree, bg, near, far, aa):
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N,K,3)
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        near_plane=near,
        far_plane=far,
        packed=False,
        absgrad=True,
        rasterize_mode="antialiased" if aa else "classic",
        backgrounds=bg,
    )
    return out, alpha, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="default: <data>/gs_out")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    ap.add_argument("--cap", type=int, default=150000, help="MCMC gaussian cap")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument(
        "--init-scale", type=float, default=None, help="default 0.02*radius"
    )
    ap.add_argument("--init-opacity", type=float, default=0.1)
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    ap.add_argument("--lambda-alpha", type=float, default=0.5)
    ap.add_argument(
        "--lambda-mask-rgb",
        action="store_true",
        help="supervise RGB only inside the mask (default: full frame w/ bg comp)",
    )
    ap.add_argument("--no-aa", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    data = Path(args.data)
    out = Path(args.out) if args.out else data / "gs_out"
    out.mkdir(parents=True, exist_ok=True)

    d = load_data(data, device)
    V, H, W = d["gt"].shape[0], d["H"], d["W"]
    scene_scale = d["radius"]
    init_scale = args.init_scale or 0.02 * d["radius"]
    near, far = 1.0, 1000.0
    aa = not args.no_aa
    print(
        f"loaded {V} views {W}x{H}; init {d['points'].shape[0]} gaussians; "
        f"radius {d['radius']:.3f}; strategy {args.strategy}"
    )

    params = build_params(d, args.sh_degree, init_scale, args.init_opacity, device)
    optimizers = make_optimizers(params, scene_scale)

    if args.strategy == "mcmc":
        strat = MCMCStrategy(
            cap_max=args.cap, verbose=False, refine_stop_iter=int(args.iters * 0.7)
        )
        strat_state = strat.initialize_state()
    else:
        strat = DefaultStrategy(
            verbose=False,
            refine_start_iter=500,
            refine_stop_iter=int(args.iters * 0.6),
            reset_every=3000,
            refine_every=100,
            absgrad=True,
        )
        strat_state = strat.initialize_state(scene_scale=scene_scale)
    strat.check_sanity(params, optimizers)

    gt, mask = d["gt"], d["mask"]
    for step in range(args.iters):
        sh_deg = min(step // 1000, args.sh_degree)
        bg = torch.rand(
            V, 3, device=device
        )  # random bg -> forces alpha to explain mask
        out_c, alpha, info = render(
            params, d["viewmats"], d["Ks"], W, H, sh_deg, bg, near, far, aa
        )
        info["means2d"].retain_grad()

        if args.strategy == "default":
            strat.step_pre_backward(params, optimizers, strat_state, step, info)

        # GT composited over the same random background using the fly mask
        gt_comp = gt * mask + bg[:, None, None, :] * (1 - mask)
        if args.lambda_mask_rgb:
            l1 = ((out_c - gt) * mask).abs().sum() / (mask.sum() * 3 + 1e-8)
        else:
            l1 = (out_c - gt_comp).abs().mean()
        loss = (1 - args.lambda_ssim) * l1 + args.lambda_ssim * (
            1 - ssim(out_c, gt_comp)
        )
        loss = loss + args.lambda_alpha * (alpha - mask).abs().mean()
        if args.strategy == "mcmc":
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).abs().mean()
            loss = loss + 0.01 * torch.exp(params["scales"]).abs().mean()

        loss.backward()

        if args.strategy == "mcmc":
            strat.step_post_backward(
                params,
                optimizers,
                strat_state,
                step,
                info,
                lr=optimizers["means"].param_groups[0]["lr"],
            )
        else:
            strat.step_post_backward(
                params, optimizers, strat_state, step, info, packed=False
            )

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 500 == 0 or step == args.iters - 1:
            with torch.no_grad():
                psnr = -10 * math.log10(((out_c - gt_comp) ** 2).mean().item() + 1e-12)
            print(
                f"  step {step:5d}  loss {loss.item():.4f}  psnr {psnr:5.2f}  "
                f"N={params['means'].shape[0]}"
            )

    save_outputs(params, d, args, out, near, far, aa, device)


@torch.no_grad()
def save_outputs(params, d, args, out, near, far, aa, device):
    V, H, W = d["gt"].shape[0], d["H"], d["W"]
    # ---- training-view comparison montage --------------------------------
    black = torch.zeros(V, 3, device=device)
    rgb, alpha, _ = render(
        params, d["viewmats"], d["Ks"], W, H, args.sh_degree, black, near, far, aa
    )
    rgb = rgb.clamp(0, 1)
    psnrs = []
    tiles = []
    for i in range(V):
        gt_i = (d["gt"][i] * d["mask"][i]).cpu().numpy()
        pr_i = rgb[i].cpu().numpy()
        al_i = alpha[i].repeat(1, 1, 3).cpu().numpy()
        mse = float(((rgb[i] - d["gt"][i] * d["mask"][i]) ** 2).mean())
        psnrs.append(-10 * math.log10(mse + 1e-12))
        row = np.concatenate([gt_i, pr_i, al_i], axis=1)
        tiles.append((row * 255).astype(np.uint8))
        cv2.putText(
            tiles[-1],
            f"{d['names'][i]}  psnr {psnrs[-1]:.1f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    montage = np.concatenate(tiles, axis=0)
    imageio.imwrite(out / "train_views_gt_render_alpha.png", montage)
    print(
        f"\nper-view PSNR (masked): {[round(p, 1) for p in psnrs]}  mean {np.mean(psnrs):.2f}"
    )

    # ---- novel-view orbit (within the covered azimuth arc) ---------------
    center = d["center"]
    cam_c = np.stack(
        [
            (-d["viewmats"][i, :3, :3].T @ d["viewmats"][i, :3, 3]).cpu().numpy()
            for i in range(V)
        ]
    )
    R_orbit = float(np.linalg.norm(cam_c - center, axis=1).mean())
    az = np.arctan2(cam_c[:, 1] - center[1], cam_c[:, 0] - center[0])
    a_lo, a_hi = np.degrees(az).min(), np.degrees(az).max()
    el = np.degrees(np.arcsin((cam_c[:, 2] - center[2]) / R_orbit)).mean()
    K0 = d["Ks"][0:1]
    frames = []
    seq = np.concatenate([np.linspace(a_lo, a_hi, 60), np.linspace(a_hi, a_lo, 60)])
    for a in seq:
        ar, er = math.radians(a), math.radians(el)
        eye = center + R_orbit * np.array(
            [math.cos(er) * math.cos(ar), math.cos(er) * math.sin(ar), math.sin(er)]
        )
        vm = torch.tensor(
            look_at(eye, center)[None], dtype=torch.float32, device=device
        )
        img, _, _ = render(
            params,
            vm,
            K0,
            W,
            H,
            args.sh_degree,
            torch.zeros(1, 3, device=device),
            near,
            far,
            aa,
        )
        frames.append((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimsave(out / "orbit.mp4", frames, fps=30, quality=8)
    print(f"wrote {out / 'train_views_gt_render_alpha.png'} and {out / 'orbit.mp4'}")

    # ---- save the gaussian model ----------------------------------------
    ckpt = {k: v.detach().cpu() for k, v in params.items()}
    ckpt["center"] = torch.tensor(center)
    ckpt["radius"] = torch.tensor(d["radius"])
    torch.save(ckpt, out / "gaussians.pt")


if __name__ == "__main__":
    main()
