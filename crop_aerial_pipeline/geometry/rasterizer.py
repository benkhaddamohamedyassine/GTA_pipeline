"""Fully NumPy-vectorized z-buffer point-splat renderer.

Deliberately does NOT use Open3D's ``OffscreenRenderer`` (unreliable/unsupported
in a headless Colab GL context -- no EGL/OpenGL is initialized anywhere in this
module) and does NOT loop over points in Python. Each point is splatted to a
``(2*splat_radius+1)^2`` pixel kernel to close point-cloud subsampling gaps;
this is a small, constant number of fully-vectorized scatter passes (one per
kernel offset), each processing all N points at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BACKGROUND_COLOR = 128  # neutral gray -- anything still this color after rendering is unfilled


@dataclass
class RenderResult:
    color_image: np.ndarray  # (H, W, 3) uint8
    valid_mask: np.ndarray  # (H, W) bool -- pixel received at least one splatted point
    synthesized_mask: np.ndarray  # (H, W) bool -- True where the winning (nearest) point was synthesized
    depth_buffer: np.ndarray  # (H, W) float32, np.inf where invalid


def render_zbuffer(
    points: np.ndarray,
    colors: np.ndarray,
    is_original: np.ndarray,
    extrinsic: np.ndarray,
    K: np.ndarray,
    out_w: int,
    out_h: int,
    splat_radius: int = 2,
) -> RenderResult:
    ones = np.ones((points.shape[0], 1))
    cam_pts = (extrinsic @ np.concatenate([points, ones], axis=1).T).T[:, :3]
    z = cam_pts[:, 2]
    in_front = z > 0.05
    cam_pts, colors_f, z, orig_f = cam_pts[in_front], colors[in_front], z[in_front], is_original[in_front]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u0 = np.round(cam_pts[:, 0] * fx / z + cx).astype(np.int64)
    v0 = np.round(cam_pts[:, 1] * fy / z + cy).astype(np.int64)
    colors_u8 = (colors_f * 255).astype(np.uint8)

    color_flat = np.full((out_h * out_w, 3), BACKGROUND_COLOR, dtype=np.uint8)
    depth_flat = np.full(out_h * out_w, np.inf, dtype=np.float32)
    synth_flat = np.zeros(out_h * out_w, dtype=bool)

    offsets = [
        (dy, dx)
        for dy in range(-splat_radius, splat_radius + 1)
        for dx in range(-splat_radius, splat_radius + 1)
    ]
    for dy, dx in offsets:
        uu, vv = u0 + dx, v0 + dy
        in_bounds = (uu >= 0) & (uu < out_w) & (vv >= 0) & (vv < out_h)
        if not np.any(in_bounds):
            continue
        idx = vv[in_bounds] * out_w + uu[in_bounds]
        z_b = z[in_bounds]
        c_b = colors_u8[in_bounds]
        o_b = orig_f[in_bounds]

        order = np.argsort(z_b)  # ascending -- nearest point first
        idx_sorted, z_sorted, c_sorted, o_sorted = idx[order], z_b[order], c_b[order], o_b[order]
        # first occurrence (in ascending-z order) of each unique pixel = the NEAREST
        # point mapping there -- a fully vectorized nearest-wins merge, no per-point loop.
        uniq_idx, first_pos = np.unique(idx_sorted, return_index=True)
        better = z_sorted[first_pos] < depth_flat[uniq_idx]
        sel = uniq_idx[better]
        depth_flat[sel] = z_sorted[first_pos][better]
        color_flat[sel] = c_sorted[first_pos][better]
        synth_flat[sel] = ~o_sorted[first_pos][better]

    color_img = color_flat.reshape(out_h, out_w, 3)
    valid_mask = np.isfinite(depth_flat).reshape(out_h, out_w)
    synthesized_mask = synth_flat.reshape(out_h, out_w) & valid_mask
    depth_buffer = depth_flat.reshape(out_h, out_w)

    return RenderResult(
        color_image=color_img,
        valid_mask=valid_mask,
        synthesized_mask=synthesized_mask,
        depth_buffer=depth_buffer,
    )


def depth_buffer_preview(depth_buffer: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Colorized depth-buffer preview for the Stage 8 debug output."""
    import matplotlib.cm as cm

    finite = depth_buffer[valid_mask]
    if finite.size == 0:
        return np.zeros((*depth_buffer.shape, 3), dtype=np.uint8)
    d_min, d_max = float(finite.min()), float(finite.max())
    norm = np.clip((depth_buffer - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    colored = (cm.get_cmap("inferno")(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid_mask] = 0
    return colored
