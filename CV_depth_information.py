import argparse
import os
from typing import List, Optional

import cv2
import numpy as np


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


def run_classifier(view_imgs_rgb: List[np.ndarray], num_classes: int) -> Optional[int]:
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T
    except Exception as e:
        print(f"Skip classifier (missing torch/torchvision): {e}")
        return None

    class MultiViewClassifier(nn.Module):
        def __init__(self, num_views: int, num_classes: int):
            super().__init__()
            self.num_views = num_views
            self.backbone = models.resnet18(weights=None)
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(512, num_classes)

        def forward(self, imgs: List[torch.Tensor]) -> torch.Tensor:
            feats = [self.backbone(img) for img in imgs]
            fused = torch.mean(torch.stack(feats), dim=0)
            return self.fc(fused)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    view_tensors = [transform(img).unsqueeze(0) for img in view_imgs_rgb]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewClassifier(num_views=len(view_tensors), num_classes=num_classes).to(device)
    model.eval()

    with torch.no_grad():
        view_tensors = [t.to(device) for t in view_tensors]
        logits = model(view_tensors)
        pred = int(torch.argmax(logits, dim=1).item())
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="left.png")
    parser.add_argument("--right", default="right.png")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--num_disp", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--focal_length_px", type=float, default=0.8)
    parser.add_argument("--baseline_m", type=float, default=0.1)
    parser.add_argument("--run_classifier", action="store_true")
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    left_bgr = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left_bgr is None:
        raise FileNotFoundError(f"Failed to read left image: {args.left}")
    if right_bgr is None:
        raise FileNotFoundError(f"Failed to read right image: {args.right}")
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        raise ValueError(f"left/right size mismatch: {left_bgr.shape} vs {right_bgr.shape}")

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    left_enh = edge_enhance(left_rgb)
    right_enh = edge_enhance(right_rgb)

    disp_map = compute_disparity_sgbm(left_enh, right_enh, num_disp=args.num_disp, block_size=args.block_size)
    depth_map = depth_from_disparity(disp_map, focal_length_px=args.focal_length_px, baseline_m=args.baseline_m)

    os.makedirs(args.out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(args.out_dir, "left_enh.png"), cv2.cvtColor(left_enh, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out_dir, "right_enh.png"), cv2.cvtColor(right_enh, cv2.COLOR_RGB2BGR))

    disp_u8 = normalize_to_u8(disp_map)
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(args.out_dir, "disparity.png"), disp_color)

    depth_u8 = normalize_to_u8(depth_map)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(args.out_dir, "depth.png"), depth_color)

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

    if args.run_classifier:
        pred = run_classifier([left_rgb, right_rgb], num_classes=args.num_classes)
        if pred is not None:
            print("Predicted object class:", pred)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
