"""
Extract a triangle mesh from a 3DGS PLY file via TSDF fusion.

Pipeline:
  1. Load Gaussians from PLY (plyfile)
  2. Load camera poses from COLMAP sparse reconstruction
  3. Render depth + color from each view (pure PyTorch, no CUDA compilation)
  4. Fuse into TSDF volume (Open3D) and extract mesh

Usage:
  python ply_to_mesh.py \
    --ply_path path/to/point_cloud.ply \
    --colmap_path path/to/sparse/0 \
    --output_path output_mesh.ply \
    --voxel_length 0.004 \
    --sdf_trunc 0.02 \
    --depth_trunc 3.0 \
    --opacity_threshold 0.5

Dependencies:
  pip install plyfile open3d torch numpy
"""

import argparse
import os
import math
import time

import numpy as np
import torch
import open3d as o3d
from plyfile import PlyData

# Import colmap_loader directly (not via scene package) to avoid
# scene/__init__.py pulling in heavy deps like simple_knn
import importlib.util as _ilu
_repo = os.path.dirname(os.path.abspath(__file__))

def _load_mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cl = _load_mod("colmap_loader", os.path.join(_repo, "scene", "colmap_loader.py"))
read_extrinsics_binary = _cl.read_extrinsics_binary
read_extrinsics_text = _cl.read_extrinsics_text
read_intrinsics_binary = _cl.read_intrinsics_binary
read_intrinsics_text = _cl.read_intrinsics_text
qvec2rotmat = _cl.qvec2rotmat

# SH constants (from utils/sh_utils.py)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]


def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics at unit directions. sh: (..., C, K), dirs: (..., 3)"""
    result = SH_C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - SH_C1 * y * sh[..., 1] + SH_C1 * z * sh[..., 2] - SH_C1 * x * sh[..., 3]
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result + SH_C2[0] * xy * sh[..., 4] + SH_C2[1] * yz * sh[..., 5] +
                      SH_C2[2] * (2 * zz - xx - yy) * sh[..., 6] + SH_C2[3] * xz * sh[..., 7] +
                      SH_C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result + SH_C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          SH_C3[1] * xy * z * sh[..., 10] + SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          SH_C3[5] * z * (xx - yy) * sh[..., 14] +
                          SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15])
    return result


def quat_to_rotmat(q):
    """Convert quaternions (N, 4) wxyz to rotation matrices (N, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def load_gaussians(ply_path, opacity_threshold=0.5):
    """Load Gaussian parameters from a 3DGS PLY file, filtering by opacity."""
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacities_raw = v["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities_raw))

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    sh_rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if sh_rest_names:
        f_rest = np.stack([v[n].astype(np.float32) for n in sh_rest_names], axis=1)
    else:
        f_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    n_sh_total = 1 + f_rest.shape[1] // 3
    sh_degree = int(math.sqrt(n_sh_total)) - 1

    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.exp(np.stack([v[n].astype(np.float32) for n in scale_names], axis=1))

    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.stack([v[n].astype(np.float32) for n in rot_names], axis=1)

    mask = opacities > opacity_threshold
    print(f"Loaded {xyz.shape[0]} Gaussians, {mask.sum()} pass opacity > {opacity_threshold}")

    return {
        "xyz": xyz[mask],
        "opacities": opacities[mask],
        "f_dc": f_dc[mask],
        "f_rest": f_rest[mask] if f_rest.shape[1] > 0 else f_rest[:mask.sum()],
        "scales": scales[mask],
        "rotations": rots[mask],
        "sh_degree": sh_degree,
    }


def load_colmap_cameras(colmap_path):
    """Load camera intrinsics and extrinsics from COLMAP sparse folder."""
    if os.path.exists(os.path.join(colmap_path, "cameras.bin")):
        cameras = read_intrinsics_binary(os.path.join(colmap_path, "cameras.bin"))
    else:
        cameras = read_intrinsics_text(os.path.join(colmap_path, "cameras.txt"))

    if os.path.exists(os.path.join(colmap_path, "images.bin")):
        images = read_extrinsics_binary(os.path.join(colmap_path, "images.bin"))
    else:
        images = read_extrinsics_text(os.path.join(colmap_path, "images.txt"))

    cam_list = []
    for img_id, img in images.items():
        cam = cameras[img.camera_id]
        R = qvec2rotmat(img.qvec)
        t = img.tvec

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model == "SIMPLE_PINHOLE":
            fx = fy = cam.params[0]
            cx, cy = cam.params[1], cam.params[2]
        elif cam.model in ("SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx, cy = cam.params[1], cam.params[2]
        elif cam.model == "OPENCV":
            fx, fy = cam.params[0], cam.params[1]
            cx, cy = cam.params[2], cam.params[3]
        else:
            if len(cam.params) >= 4:
                fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            else:
                fx = fy = cam.params[0]
                cx, cy = cam.width / 2, cam.height / 2

        cam_list.append({
            "w2c": w2c,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": cam.width, "height": cam.height,
            "name": img.name,
        })

    print(f"Loaded {len(cam_list)} cameras from COLMAP")
    return cam_list


def render_view_pytorch(means3d, cov3d, opacities, sh_coeffs, sh_degree,
                        R_w2c, t_w2c, fx, fy, cx, cy, W, H, device):
    """
    Render one view using pure PyTorch EWA splatting.

    Args:
        means3d: (N, 3) world-space Gaussian centers
        cov3d: (N, 3, 3) world-space 3D covariance matrices
        opacities: (N,) opacity values
        sh_coeffs: (N, 3, K) SH coefficients
        sh_degree: int
        R_w2c, t_w2c: (3,3) and (3,) world-to-camera transform
        fx, fy, cx, cy: intrinsics
        W, H: image dimensions

    Returns:
        color: (H, W, 3) float32
        depth: (H, W) float32
    """
    N = means3d.shape[0]

    # Transform to camera space
    means_cam = means3d @ R_w2c.T + t_w2c  # (N, 3)

    # Filter: keep only in front of camera
    valid = means_cam[:, 2] > 0.2
    if valid.sum() == 0:
        return np.zeros((H, W, 3), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    means_cam = means_cam[valid]
    cov3d_v = cov3d[valid]
    opac_v = opacities[valid]
    sh_v = sh_coeffs[valid]

    z = means_cam[:, 2]
    x_cam = means_cam[:, 0]
    y_cam = means_cam[:, 1]

    # Project to 2D pixel coordinates
    px = fx * x_cam / z + cx
    py = fy * y_cam / z + cy

    # Filter: keep only those projecting into image (with margin)
    margin = 100
    in_image = (px > -margin) & (px < W + margin) & (py > -margin) & (py < H + margin)
    if in_image.sum() == 0:
        return np.zeros((H, W, 3), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    means_cam = means_cam[in_image]
    cov3d_v = cov3d_v[in_image]
    opac_v = opac_v[in_image]
    sh_v = sh_v[in_image]
    px = px[in_image]
    py = py[in_image]
    z = z[in_image]
    x_cam = x_cam[in_image]
    y_cam = y_cam[in_image]

    Nv = means_cam.shape[0]

    # Compute 2D covariance via EWA splatting
    # J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    J = torch.zeros(Nv, 2, 3, device=device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x_cam / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y_cam / (z * z)

    # Transform 3D covariance to camera space
    # cov3d_cam = R @ cov3d_world @ R^T
    cov3d_cam = R_w2c @ cov3d_v @ R_w2c.T  # (N, 3, 3) broadcast

    # Project to 2D: cov2d = J @ cov3d_cam @ J^T
    cov2d = J @ cov3d_cam @ J.transpose(-1, -2)  # (N, 2, 2)

    # Low-pass filter (anti-aliasing)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # Compute inverse and determinant of 2x2 covariance
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = a * c - b * b
    det = torch.clamp(det, min=1e-6)

    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det

    # Compute radius (3 sigma from max eigenvalue)
    trace = a + c
    sqrt_disc = torch.sqrt(torch.clamp((trace * 0.5) ** 2 - det, min=0))
    max_eig = trace * 0.5 + sqrt_disc
    radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(max_eig, min=0.1))).int()
    radius = torch.clamp(radius, max=256)

    # Compute SH colors for this view
    # View direction: from camera center to Gaussian in world space
    # Camera center in world: -R^T @ t
    cam_center_world = -R_w2c.T @ t_w2c  # (3,)
    # We need world-space means for the valid Gaussians
    # means_world = (means_cam - t_w2c) @ R_w2c  (inverse transform)
    means_world = (means_cam - t_w2c) @ R_w2c  # (Nv, 3)
    view_dirs = means_world - cam_center_world  # (Nv, 3)
    view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

    # Evaluate SH: sh_v is (Nv, 3, K)
    colors_sh = eval_sh(sh_degree, sh_v, view_dirs)  # (Nv, 3)
    colors_sh = torch.clamp(colors_sh + 0.5, 0.0, 1.0)

    # Sort by depth (front to back)
    sort_idx = z.argsort()

    # Initialize output buffers
    color_buf = torch.zeros(H, W, 3, device=device)
    depth_buf = torch.zeros(H, W, device=device)
    T_buf = torch.ones(H, W, device=device)  # transmittance

    # Alpha-composite front to back
    for k in range(Nv):
        idx = sort_idx[k].item()
        r = radius[idx].item()
        cx_g = px[idx].item()
        cy_g = py[idx].item()

        x_min = max(0, int(cx_g) - r)
        x_max = min(W, int(cx_g) + r + 1)
        y_min = max(0, int(cy_g) - r)
        y_max = min(H, int(cy_g) + r + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Pixel grid for this patch
        pix_y = torch.arange(y_min, y_max, device=device, dtype=torch.float32)
        pix_x = torch.arange(x_min, x_max, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(pix_y, pix_x, indexing='ij')

        dx = gx - px[idx]
        dy = gy - py[idx]

        # Evaluate 2D Gaussian: exp(-0.5 * [dx dy] @ cov_inv @ [dx dy]^T)
        power = -0.5 * (inv_a[idx] * dx * dx + 2 * inv_b[idx] * dx * dy + inv_c[idx] * dy * dy)
        # Clamp for numerical stability
        power = torch.clamp(power, max=0.0, min=-8.0)
        gauss = torch.exp(power)

        alpha = opac_v[idx] * gauss
        alpha = torch.clamp(alpha, max=0.99)

        T_patch = T_buf[y_min:y_max, x_min:x_max]

        # Skip if transmittance is too low everywhere in this patch
        if T_patch.max() < 1e-4:
            continue

        weight = alpha * T_patch

        color_buf[y_min:y_max, x_min:x_max] += weight.unsqueeze(-1) * colors_sh[idx]
        depth_buf[y_min:y_max, x_min:x_max] += weight * z[idx]

        T_buf[y_min:y_max, x_min:x_max] = T_patch * (1 - alpha)

    color_np = color_buf.cpu().numpy()
    depth_np = depth_buf.cpu().numpy()
    return color_np, depth_np


def render_depth_and_color(gaussians, cameras, device="cuda"):
    """Render depth and color maps from all cameras using pure PyTorch splatting."""
    means = torch.tensor(gaussians["xyz"], dtype=torch.float32, device=device)
    quats = torch.tensor(gaussians["rotations"], dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.tensor(gaussians["scales"], dtype=torch.float32, device=device)
    opacities = torch.tensor(gaussians["opacities"], dtype=torch.float32, device=device)

    sh_degree = gaussians["sh_degree"]
    n_coeffs = (sh_degree + 1) ** 2
    f_dc = gaussians["f_dc"]
    f_rest = gaussians["f_rest"]
    # Build SH coefficients: (N, 3, K) — 3 color channels, K SH basis functions
    N = means.shape[0]
    sh_all = np.zeros((N, 3, n_coeffs), dtype=np.float32)
    sh_all[:, :, 0] = f_dc  # DC term, (N, 3)
    if f_rest.shape[1] > 0:
        n_rest = n_coeffs - 1
        # f_rest layout in 3DGS PLY: (N, 3*(K-1)), grouped as [ch0_coeff1..ch0_coeffK-1, ch1_..., ch2_...]
        f_rest_r = f_rest.reshape(N, 3, n_rest)
        sh_all[:, :, 1:] = f_rest_r
    sh_coeffs = torch.tensor(sh_all, dtype=torch.float32, device=device)

    # Build 3D covariance: cov = R @ diag(s^2) @ R^T
    R_q = quat_to_rotmat(quats)  # (N, 3, 3)
    S2 = torch.diag_embed(scales ** 2)  # (N, 3, 3)
    cov3d = R_q @ S2 @ R_q.transpose(-1, -2)  # (N, 3, 3)

    renders = []
    t0 = time.time()
    for i, cam in enumerate(cameras):
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

        w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=device)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]

        with torch.no_grad():
            color, depth = render_view_pytorch(
                means, cov3d, opacities, sh_coeffs, sh_degree,
                R_w2c, t_w2c, fx, fy, cx, cy, int(W), int(H), device,
            )

        renders.append({"color": color, "depth": depth, "cam": cam})

        if (i + 1) % 5 == 0 or i == len(cameras) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(cameras) - i - 1)
            print(f"  Rendered {i + 1}/{len(cameras)} views ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    return renders


def tsdf_fusion(renders, voxel_length=0.004, sdf_trunc=0.02, depth_trunc=3.0):
    """Fuse rendered depth+color into a TSDF volume and extract mesh."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, r in enumerate(renders):
        cam = r["cam"]
        depth = r["depth"].copy()
        color = r["color"]

        if depth.max() <= 0:
            continue

        depth[depth > depth_trunc] = 0.0

        H, W = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(W), height=int(H),
            fx=cam["fx"], fy=cam["fy"],
            cx=cam["cx"], cy=cam["cy"],
        )

        color_u8 = (np.clip(color, 0, 1) * 255).astype(np.uint8)

        color_o3d = o3d.geometry.Image(color_u8)
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        extrinsic = cam["w2c"]
        volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))

        if (i + 1) % 10 == 0 or i == len(renders) - 1:
            print(f"  Integrated {i + 1}/{len(renders)} views")

    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(description="Extract mesh from 3DGS PLY via TSDF fusion")
    parser.add_argument("--ply_path", required=True, help="Path to 3DGS point_cloud.ply")
    parser.add_argument("--colmap_path", required=True, help="Path to COLMAP sparse/0 folder")
    parser.add_argument("--output_path", default="output_mesh.ply", help="Output mesh path")
    parser.add_argument("--voxel_length", type=float, default=0.004, help="TSDF voxel size")
    parser.add_argument("--sdf_trunc", type=float, default=0.02, help="TSDF truncation distance")
    parser.add_argument("--depth_trunc", type=float, default=3.0, help="Max depth to integrate")
    parser.add_argument("--opacity_threshold", type=float, default=0.5, help="Min opacity to keep")
    parser.add_argument("--sh_degree", type=int, default=None, help="SH degree override (auto-detected)")
    parser.add_argument("--device", default="cuda", help="Torch device")
    args = parser.parse_args()

    print("Step 1: Loading Gaussians...")
    gaussians = load_gaussians(args.ply_path, args.opacity_threshold)
    if args.sh_degree is not None:
        gaussians["sh_degree"] = args.sh_degree

    print("Step 2: Loading COLMAP cameras...")
    cameras = load_colmap_cameras(args.colmap_path)

    print("Step 3: Rendering depth + color maps (pure PyTorch)...")
    renders = render_depth_and_color(gaussians, cameras, device=args.device)

    print("Step 4: TSDF fusion...")
    mesh = tsdf_fusion(renders, args.voxel_length, args.sdf_trunc, args.depth_trunc)

    print(f"Saving mesh to {args.output_path}")
    print(f"  Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
    o3d.io.write_triangle_mesh(args.output_path, mesh)
    print("Done!")


if __name__ == "__main__":
    main()
