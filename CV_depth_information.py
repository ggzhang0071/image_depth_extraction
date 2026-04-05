import argparse
import os
import cv2
import numpy as np


# =========================
# 1. 多曝光 → 灰度 → 加权融合
# =========================
def fuse_exposure_to_gray(img_list):
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) for img in img_list]
    weights = np.ones(len(grays)) / len(grays)
    fused = sum(w * g for w, g in zip(weights, grays))
    return fused.astype(np.uint8)


# =========================
# 2. 抽象 Agent（结构提取）
# =========================
def abstract_structure(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)

    # 结构保留（强梯度）
    _, structure = cv2.threshold(mag, 30, 255, cv2.THRESH_BINARY)
    return structure.astype(np.uint8)


# =========================
# 3. 回写结构 → 原图
# =========================
def writeback_structure(original, structure, alpha=0.8, beta=0.2):
    structure_3c = cv2.merge([structure, structure, structure])
    out = cv2.addWeighted(original, alpha, structure_3c, beta, 0)
    return out


# =========================
# 4. 边缘增强 Agent（轻量）
# =========================
def edge_enhance(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


# =========================
# 5. 视差计算
# =========================
def compute_disparity(left, right):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    return disp


# =========================
# 6. 深度计算
# =========================
def depth_from_disp(disp, f=721.5377, B=0.5327254279298227):
    return (f * B) / (disp + 1e-6)


# =========================
# 7. 拟合 Agent（重）
# =========================
def fit_depth(depth):
    # 简单平滑 + 拟合（占位实现）
    depth_blur = cv2.bilateralFilter(depth.astype(np.float32), 9, 75, 75)
    return depth_blur


# =========================
# 主流程（严格按你定义顺序）
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", nargs=5, help="5张曝光图")
    parser.add_argument("--left", default="left.png")
    parser.add_argument("--right", default="right.png")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.imgs is not None:
        imgs = [cv2.imread(p) for p in args.imgs]
        if any(img is None for img in imgs):
            missing = [p for p, img in zip(args.imgs, imgs) if img is None]
            raise FileNotFoundError(f"Failed to read: {missing}")
        left_raw = None
        right_raw = None
    else:
        left_raw = cv2.imread(args.left, cv2.IMREAD_COLOR)
        right_raw = cv2.imread(args.right, cv2.IMREAD_COLOR)
        if left_raw is None:
            raise FileNotFoundError(f"Failed to read left image: {args.left}")
        if right_raw is None:
            raise FileNotFoundError(f"Failed to read right image: {args.right}")
        if left_raw.shape[:2] != right_raw.shape[:2]:
            raise ValueError(f"left/right size mismatch: {left_raw.shape} vs {right_raw.shape}")
        imgs = [left_raw, left_raw, left_raw, left_raw, left_raw]

    # 1张原图 + 4张做灰度融合
    original = imgs[0]
    fused_gray = fuse_exposure_to_gray(imgs[1:])

    # 抽象 Agent
    structure = abstract_structure(fused_gray)

    # 回写结构
    enhanced = writeback_structure(original, structure)

    # 边缘增强 Agent
    edge_img = edge_enhance(enhanced)

    if right_raw is None:
        left = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        right = cv2.GaussianBlur(left, (5, 5), 0)
        right_edge_img = None
    else:
        enhanced_right = writeback_structure(right_raw, structure)
        right_edge_img = edge_enhance(enhanced_right)
        left = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right_edge_img, cv2.COLOR_BGR2GRAY)

    # 视差
    disp = compute_disparity(left, right)

    # 深度
    depth = depth_from_disp(disp)

    # 拟合 Agent
    depth_fit = fit_depth(depth)

    # 保存
    cv2.imwrite(os.path.join(args.out_dir, "fused_gray.png"), fused_gray)
    cv2.imwrite(os.path.join(args.out_dir, "structure.png"), structure)
    cv2.imwrite(os.path.join(args.out_dir, "enhanced.png"), enhanced)
    cv2.imwrite(os.path.join(args.out_dir, "edge.png"), edge_img)
    if right_edge_img is not None:
        cv2.imwrite(os.path.join(args.out_dir, "right_edge.png"), right_edge_img)

    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.normalize(depth_fit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(os.path.join(args.out_dir, "disparity.png"), disp_vis)
    cv2.imwrite(os.path.join(args.out_dir, "depth.png"), depth_vis)

    print("Pipeline done. Output in:", args.out_dir)


if __name__ == "__main__":
    main()
