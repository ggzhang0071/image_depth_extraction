import argparse
import cv2
import numpy as np
import os

# ==========================================
# 步骤 1: 读取图像并进行亚像素级对齐
# ==========================================
def align_images(images):
    """
    使用特征点匹配对齐图像序列
    """
    # 使用第一张作为参考图
    ref_img = images[0]
    aligned_imgs = [ref_img]
    
    # 创建特征检测器 (ORB 是一种快速且免费的特征点算法)
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    
    if des1 is None:
        print("警告: 未在第一张图中检测到特征点，跳过对齐。")
        return images

    # 对后续图像进行对齐
    for i in range(1, len(images)):
        kp2, des2 = orb.detectAndCompute(images[i], None)
        if des2 is None:
            continue
            
        # 特征点匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:
            print(f"警告: 第 {i+1} 张图匹配点太少，跳过对齐。")
            aligned_imgs.append(images[i])
            continue

        # 提取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 计算变换矩阵
        M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        # 透视变换对齐
        h, w, _ = ref_img.shape
        aligned = cv2.warpPerspective(images[i], M, (w, h))
        aligned_imgs.append(aligned)
        
    print(f"已完成 {len(aligned_imgs)} 张图像的对齐。")
    return aligned_imgs

# ==========================================
# 步骤 2: 您的核心算法 (亮度平均)
# ==========================================
def apply_average_luminance(original_rgb, other_images):
    """
    将多张图像的亮度平均，并应用到原始图像上
    """
    luminance_maps = []
    
    # 将后几张图转换为亮度图 (使用标准公式 0.299R + 0.587G + 0.114B)
    for img in other_images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        luminance_maps.append(gray.astype(np.float32))
    
    # 计算平均亮度图
    avg_luminance = np.mean(luminance_maps, axis=0)
    
    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB)
    
    # 归一化并替换 L 通道 (亮度)
    avg_luminance_norm = cv2.normalize(avg_luminance, None, 0, 255, cv2.NORM_MINMAX)
    lab[:, :, 0] = avg_luminance_norm.astype(np.uint8)
    
    # 转回 RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result

# ==========================================
# 步骤 3: 边缘增强 (Unsharp Mask)
# ==========================================
def unsharp_mask(image, radius=1.5, amount=1.2):
    """
    执行反锐化掩模以增强边缘
    :param radius: 模糊半径 (建议 0.8 - 2.5)
    :param amount: 锐化强度 (建议 0.5 - 1.5)
    """
    # 1. 创建模糊版本
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    
    # 2. 计算差值 (原图 - 模糊图) 并加权
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharp

# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", nargs="+", default=None)
    parser.add_argument("--image_folder", default=".")
    parser.add_argument("--file_prefix", default="sample_image_")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.2)
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--out", default="final_sharpened_moon.jpg")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--left_label", default="input")
    parser.add_argument("--right_label", default="output")
    args = parser.parse_args()

    print("--- 开始处理月亮图像 ---")
    
    images = []
    if args.imgs is not None:
        for p in args.imgs:
            img = cv2.imread(p)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        for i in range(1, args.num_images + 1):
            path = os.path.join(args.image_folder, f"{args.file_prefix}{i}.jpg")
            img = cv2.imread(path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if len(images) < 2:
        print("错误: 需要至少2张图片才能处理。")
    else:
        aligned_images = align_images(images)
        
        print("正在应用亮度平均算法...")
        processed_img = apply_average_luminance(aligned_images[0], aligned_images[1:])
        
        print("正在进行边缘增强...")
        final_img = unsharp_mask(processed_img, radius=args.radius, amount=args.amount)
        
        final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

        first_bgr = cv2.cvtColor(aligned_images[0], cv2.COLOR_RGB2BGR)
        if first_bgr.shape[:2] != final_bgr.shape[:2]:
            first_bgr = cv2.resize(first_bgr, (final_bgr.shape[1], final_bgr.shape[0]), interpolation=cv2.INTER_AREA)

        gap = 20
        canvas = np.full((final_bgr.shape[0], final_bgr.shape[1] * 2 + gap, 3), 18, dtype=np.uint8)
        canvas[:, 0 : final_bgr.shape[1]] = first_bgr
        canvas[:, final_bgr.shape[1] + gap :] = final_bgr

        cv2.putText(canvas, args.left_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            args.right_label,
            (final_bgr.shape[1] + gap + 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        root, ext = os.path.splitext(args.out)
        compare_path = f"{root}_compare{ext or '.jpg'}"

        if args.compare:
            cv2.imwrite(args.out, canvas)
            print(f"处理完成！对比图已保存为: {args.out}")
        else:
            cv2.imwrite(args.out, final_bgr)
            cv2.imwrite(compare_path, canvas)
            print(f"处理完成！最终图片已保存为: {args.out}")
            print(f"对比图已保存为: {compare_path}")
        print("建议查看效果，如果觉得太锐或不够锐，可以微调 radius 和 amount 参数。")
