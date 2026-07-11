"""Render novel viewpoints from a saved splat, to probe 3D consistency.

Renders views interpolated *between* the training cameras (within the covered
arc -> should look right) and a couple *outside* the arc (behind / top-down ->
expected to degrade), as an honest sanity check of what the 7-view splat
actually constrains.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from fit_gsplat import look_at  # same OpenCV look-at used in training
from gsplat import rasterization


@torch.no_grad()
def render_view(g, vm, K, W, H, sh_degree, device):
    colors = torch.cat([g["sh0"], g["shN"]], dim=1)
    out, alpha, _ = rasterization(
        means=g["means"],
        quats=g["quats"],
        scales=torch.exp(g["scales"]),
        opacities=torch.sigmoid(g["opacities"]),
        colors=colors,
        viewmats=vm,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree,
        near_plane=1.0,
        far_plane=1000.0,
        packed=False,
        rasterize_mode="antialiased",
        backgrounds=torch.zeros(1, 3, device=device),
    )
    return out[0].clamp(0, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None, help="default <data>/gs_out/gaussians.pt")
    ap.add_argument("--sh-degree", type=int, default=3)
    args = ap.parse_args()
    device = "cuda"
    data = Path(args.data)
    ckpt = Path(args.ckpt) if args.ckpt else data / "gs_out" / "gaussians.pt"
    g = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in torch.load(ckpt).items()
    }

    cams = np.load(data / "cameras.npz", allow_pickle=True)
    W, H = int(cams["width"]), int(cams["height"])
    K = torch.tensor(cams["Ks"][0:1], dtype=torch.float32, device=device)
    vms = cams["viewmats"]
    center = np.load(data / "init.npz")["center"].astype(float)
    cam_c = np.stack([-vms[i, :3, :3].T @ vms[i, :3, 3] for i in range(len(vms))])
    R = float(np.linalg.norm(cam_c - center, axis=1).mean())
    az = np.degrees(np.arctan2(cam_c[:, 1] - center[1], cam_c[:, 0] - center[0]))
    a_lo, a_hi = az.min(), az.max()

    def eye(a_deg, e_deg):
        a, e = math.radians(a_deg), math.radians(e_deg)
        return center + R * np.array(
            [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
        )

    # in-arc novel views (interpolated between training cams) + out-of-arc probes
    views = [
        (
            f"in-arc {a_lo + 0.17 * (a_hi - a_lo):.0f}deg",
            a_lo + 0.17 * (a_hi - a_lo),
            0,
        ),
        (f"in-arc {a_lo + 0.5 * (a_hi - a_lo):.0f}deg", a_lo + 0.5 * (a_hi - a_lo), 0),
        (
            f"in-arc {a_lo + 0.83 * (a_hi - a_lo):.0f}deg",
            a_lo + 0.83 * (a_hi - a_lo),
            0,
        ),
        ("in-arc +30 elev", a_lo + 0.5 * (a_hi - a_lo), 30),
        ("OUT top-down 80deg", a_lo + 0.5 * (a_hi - a_lo), 80),
        ("OUT behind 180deg", 180.0, 0),
    ]
    tiles = []
    for label, a, e in views:
        vm = torch.tensor(
            look_at(eye(a, e), center)[None], dtype=torch.float32, device=device
        )
        img = (render_view(g, vm, K, W, H, args.sh_degree, device) * 255).astype(
            np.uint8
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(
            img,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 220, 60) if label.startswith("in") else (60, 60, 240),
            2,
        )
        tiles.append(img)
    top = np.hstack(tiles[:3])
    bot = np.hstack(tiles[3:])
    montage = np.vstack([top, bot])
    outp = ckpt.parent / "novel_views.png"
    cv2.imwrite(str(outp), montage)
    print(
        "wrote",
        outp,
        montage.shape,
        "| covered arc az [%.0f, %.0f], R=%.1f" % (a_lo, a_hi, R),
    )


if __name__ == "__main__":
    main()
