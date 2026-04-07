import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 第一部分：N+1 图像融合算法 (类人视觉增强)
# ==========================================
def enhance_single_eye(image_folder, base_image_name):
    """
    模拟您的 N+1 方案：
    1. 读取文件夹内的多张短曝光图 (N张)
    2. 选取一张原图 (1张)
    3. 通过亮度加权，将N张的细节写回原图
    """
    print(f"正在处理目录: {image_folder}")
    
    images = []
    filenames = sorted(os.listdir(image_folder))
    
    if not filenames:
        raise ValueError("文件夹为空，请放入图片。")

    # 读取所有图片
    for filename in filenames:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    if len(images) < 2:
        raise ValueError("至少需要2张图片（1张原图 + 1张以上短曝光图）。")

    # 设定原图（假设第一张为原图，其余为短曝光图）
    base_img = images[0]
    exposure_imgs = images[1:]
    
    # 转换为浮点数以便计算
    base_float = base_img.astype(np.float32)
    merged_img = base_float.copy()

    # 核心逻辑：亮度加权融合
    # 原理：对于原图中过暗或过亮的区域，用短曝光图中的对应区域替换
    # 这里简化为：对短曝光图求平均，然后与原图按亮度比例混合
    exposure_stack = np.stack([img.astype(np.float32) for img in exposure_imgs])
    
    # 计算N张短曝光图的平均图（获取细节）
    avg_exposure = np.mean(exposure_stack, axis=0)
    
    # 创建掩码：原图中太暗的地方（< 80）用短曝光图补，太亮的地方（> 200）用短曝光图压
    dark_mask = base_float < 80
    bright_mask = base_float > 200
    
    # 融合
    # 暗部增强
    merged_img[dark_mask] = avg_exposure[dark_mask] * 0.8 + base_float[dark_mask] * 0.2
    # 亮部抑制（防止逆光过曝）
    merged_img[bright_mask] = avg_exposure[bright_mask] * 0.7 + base_float[bright_mask] * 0.3
    
    # 边缘强化 (Agent 强化)
    gray = cv2.cvtColor(merged_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 将边缘叠加回图像
    final_merged = cv2.addWeighted(merged_img, 1, edges_colored.astype(np.float32), 0.3, 0)
    
    print("N+1 融合与边缘强化完成。")
    return final_merged.astype(np.uint8)

# ==========================================
# 第二部分：双目视差计算 (空间结构重建)
# ==========================================
def calculate_disparity(left_img, right_img):
    """
    计算双目视差图。
    这是您提到的“挪5-6厘米”后的核心计算。
    """
    print("正在进行双目视差计算...")
    
    # 转为灰度图
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 初始化StereoSGBM（半全局块匹配）计算器
    # 参数需要根据实际拍摄的分辨率微调
    window_size = 5
    min_disp = 0
    num_disp = 16 * 10  # 必须是16的倍数
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 计算视差
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # 归一化以便显示
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_norm = disparity_norm.astype(np.uint8)
    
    print("视差图生成完成。")
    return disparity_norm

# ==========================================
# 第三部分：主程序与演示
# ==========================================
if __name__ == "__main__":
    # --- 配置路径 ---
    # 请将您的图片放入以下文件夹结构：
    # ./dataset/
    #   ├── left_eye/  (存放左眼组的6张图)
    #   └── right_eye/ (存放右眼组的6张图)
    
    LEFT_PATH = "./cup/left"
    RIGHT_PATH = "./cup/right"

    # 1. 单眼增强
    try:
        left_enhanced = enhance_single_eye(LEFT_PATH, "base.jpg")
        right_enhanced = enhance_single_eye(RIGHT_PATH, "base.jpg")
    except Exception as e:
        print(f"图像处理失败: {e}")
        print("请检查路径和图片数量。")
        exit()

    # 2. 计算视差
    depth_map = calculate_disparity(left_enhanced, right_enhanced)

    # 3. 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Left Eye Enhanced (N+1 Processed)")
    plt.imshow(cv2.cvtColor(left_enhanced, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Right Eye Enhanced (N+1 Processed)")
    plt.imshow(cv2.cvtColor(right_enhanced, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Depth Map (Spatial Structure)")
    plt.imshow(depth_map, cmap='plasma') # 使用热图显示深度
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 4. 保存结果
    cv2.imwrite("./output_left_enhanced.jpg", left_enhanced)
    cv2.imwrite("./output_depth_map.png", depth_map)
    print("结果已保存到当前目录。")