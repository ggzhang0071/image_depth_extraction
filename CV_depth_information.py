import argparse
import os
from typing import List, Optional

import cv2
import numpy as np
import zipfile


def edge_enhance(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    maxv = float(grad.max())
    if maxv <= 0:
        grad_u8 = np.zeros_like(gray, dtype=np.uint8)
    else:
        grad_u8 = (grad / maxv * 255).astype(np.uint8)
    return cv2.merge([gray, grad_u8, grad_u8])


def compute_disparity_sgbm(left: np.ndarray, right: np.ndarray, num_disp: int, block_size: int) -> np.ndarray:
    if num_disp % 16 != 0:
        raise ValueError(f"num_disp must be multiple of 16, got {num_disp}")
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError(f"block_size must be odd and >= 3, got {block_size}")

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo.compute(left, right).astype(np.float32) / 16.0


def depth_from_disparity(disparity: np.ndarray, focal_length_px: float, baseline_m: float) -> np.ndarray:
    return (focal_length_px * baseline_m) / (disparity + 1e-6)


def normalize_to_u8(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.uint8)
    v = img[finite]
    vmin, vmax = np.percentile(v, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(img.shape, dtype=np.uint8)
    out = (img - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)



def resize_keep_aspect_bgr(img_bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == target_w:
        return img_bgr
    target_h = max(1, int(round(h * (target_w / float(w)))))
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_overview_image(tiles: List[tuple[str, np.ndarray]], cols: int = 3, target_w: int = 640, pad: int = 12) -> np.ndarray:
    resized: List[tuple[str, np.ndarray]] = []
    max_h = 1
    for title, img in tiles:
        img_r = resize_keep_aspect_bgr(img, target_w=target_w)
        max_h = max(max_h, img_r.shape[0])
        resized.append((title, img_r))

    rows = (len(resized) + cols - 1) // cols
    tile_h = max_h
    tile_w = target_w

    canvas_h = pad + rows * (tile_h + pad)
    canvas_w = pad + cols * (tile_w + pad)
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)

    for idx, (title, img) in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
        cv2.putText(canvas, title, (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def decode_png_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    data = zf.read(member)
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode: {member}")
    return img


def list_kitti_stereo_pairs(zip_path: str) -> List[tuple[str, str, str]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    left = {}
    right = {}
    for n in names:
        if n.endswith("/") or not n.endswith(".png"):
            continue
        if "/image_02/data/" in n:
            fid = os.path.splitext(os.path.basename(n))[0]
            left[fid] = n
        elif "/image_03/data/" in n:
            fid = os.path.splitext(os.path.basename(n))[0]
            right[fid] = n

    common = sorted(set(left.keys()) & set(right.keys()))
    return [(fid, left[fid], right[fid]) for fid in common]


def process_stereo_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    out_dir: str,
    num_disp: int,
    block_size: int,
    focal_length_px: float,
    baseline_m: float,
    write_overview: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if left_bgr is None or right_bgr is None:
        raise ValueError("left/right image is None")
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        raise ValueError(f"left/right size mismatch: {left_bgr.shape} vs {right_bgr.shape}")

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    left_enh = edge_enhance(left_rgb)
    right_enh = edge_enhance(right_rgb)

    disp_map = compute_disparity_sgbm(left_enh, right_enh, num_disp=num_disp, block_size=block_size)
    depth_map = depth_from_disparity(disp_map, focal_length_px=focal_length_px, baseline_m=baseline_m)

    os.makedirs(out_dir, exist_ok=True)

    left_enh_bgr = cv2.cvtColor(left_enh, cv2.COLOR_RGB2BGR)
    right_enh_bgr = cv2.cvtColor(right_enh, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "left_enh.png"), left_enh_bgr)
    cv2.imwrite(os.path.join(out_dir, "right_enh.png"), right_enh_bgr)

    disp_u8 = normalize_to_u8(disp_map)
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(out_dir, "disparity.png"), disp_color)

    depth_u8 = normalize_to_u8(depth_map)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_color)

    if write_overview:
        overview = make_overview_image(
            [
                ("left", left_bgr),
                ("right", right_bgr),
                ("left_enh", left_enh_bgr),
                ("right_enh", right_enh_bgr),
                ("disparity", disp_color),
                ("depth", depth_color),
            ],
            cols=3,
            target_w=min(640, int(left_bgr.shape[1])),
            pad=12,
        )
        cv2.imwrite(os.path.join(out_dir, "overview.png"), overview)

    return disp_map, depth_map, disp_color, depth_color


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="left.png")
    parser.add_argument("--right", default="right.png")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--num_disp", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--focal_length_px", type=float, default=0.8)
    parser.add_argument("--baseline_m", type=float, default=0.1)
    parser.add_argument("--kitti_zip", default=None)
    parser.add_argument("--kitti_n", type=int, default=0)
    parser.add_argument("--kitti_start", type=int, default=0)
    parser.add_argument("--kitti_stride", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    if args.kitti_zip is None:
        left_bgr = cv2.imread(args.left, cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(args.right, cv2.IMREAD_COLOR)
        if left_bgr is None:
            raise FileNotFoundError(f"Failed to read left image: {args.left}")
        if right_bgr is None:
            raise FileNotFoundError(f"Failed to read right image: {args.right}")

        disp_map, depth_map, _, _ = process_stereo_pair(
            left_bgr=left_bgr,
            right_bgr=right_bgr,
            out_dir=args.out_dir,
            num_disp=args.num_disp,
            block_size=args.block_size,
            focal_length_px=args.focal_length_px,
            baseline_m=args.baseline_m,
            write_overview=True,
        )

        disp_finite = np.isfinite(disp_map)
        depth_finite = np.isfinite(depth_map)
        print(
            "disparity:",
            "min",
            float(np.min(disp_map[disp_finite])) if disp_finite.any() else None,
            "max",
            float(np.max(disp_map[disp_finite])) if disp_finite.any() else None,
            "mean",
            float(np.mean(disp_map[disp_finite])) if disp_finite.any() else None,
        )
        print(
            "depth:",
            "min",
            float(np.min(depth_map[depth_finite])) if depth_finite.any() else None,
            "max",
            float(np.max(depth_map[depth_finite])) if depth_finite.any() else None,
            "mean",
            float(np.mean(depth_map[depth_finite])) if depth_finite.any() else None,
        )
        print("saved:", os.path.abspath(args.out_dir))


        return 0

    pairs = list_kitti_stereo_pairs(args.kitti_zip)
    if not pairs:
        raise ValueError(f"No KITTI stereo pairs found in: {args.kitti_zip}")

    if args.kitti_n <= 0:
        kitti_n = min(12, max(1, (len(pairs) - args.kitti_start + args.kitti_stride - 1) // args.kitti_stride))
    else:
        kitti_n = args.kitti_n

    if args.focal_length_px == 0.8 and args.baseline_m == 0.1:
        focal_length_px = 721.5377
        baseline_m = 0.5327254279298227
    else:
        focal_length_px = args.focal_length_px
        baseline_m = args.baseline_m

    selected = []
    idx = args.kitti_start
    while idx < len(pairs) and len(selected) < kitti_n:
        selected.append(pairs[idx])
        idx += max(1, args.kitti_stride)

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    depth_tiles: List[tuple[str, np.ndarray]] = []
    with zipfile.ZipFile(args.kitti_zip, "r") as zf:
        for fid, left_member, right_member in selected:
            left_bgr = decode_png_from_zip(zf, left_member)
            right_bgr = decode_png_from_zip(zf, right_member)
            frame_out = os.path.join(out_root, f"kitti_{fid}")
            _, _, _, depth_color = process_stereo_pair(
                left_bgr=left_bgr,
                right_bgr=right_bgr,
                out_dir=frame_out,
                num_disp=args.num_disp,
                block_size=args.block_size,
                focal_length_px=focal_length_px,
                baseline_m=baseline_m,
                write_overview=True,
            )
            depth_tiles.append((fid, depth_color))

    summary = make_overview_image(depth_tiles, cols=4, target_w=480, pad=12)
    cv2.imwrite(os.path.join(out_root, "kitti_depth_summary.png"), summary)
    print("saved:", os.path.abspath(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 
