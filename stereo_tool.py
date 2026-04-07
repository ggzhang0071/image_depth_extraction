import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import sys

# ==========================================
# 核心配置区 (你可以随时调整这里)
# ==========================================
CONFIG = {
    "N_EXPOSURES": 5,           # 模拟连拍5张（微曝光模拟）
    "DISPARITY_STRENGTH": 15,   # 视差强度（出屏感大小，建议 10-25）
    "AGENT_CLIP_LIMIT": 2.0,    # 边缘Agent强度（建议 1.5-3.0）
    "OUTPUT_SIZE": (960, 1080)  # 输出SBS视频的单眼宽度 (宽, 高)
}

# ==========================================
# 核心算法模块 (严格遵循我们的“道”)
# ==========================================

def simulate_short_exposures(img, n=5):
    """第一步：模拟短曝光序列（模拟连拍N张，引入随机噪声）"""
    h, w = img.shape[:2]
    exposures = []
    for _ in range(n):
        # 模拟曝光变化（亮度扰动）
        exposure = img.astype(np.float32) * np.random.uniform(0.95, 1.05)
        # 模拟传感器噪声
        noise = np.random.normal(0, 5, (h, w, 3))
        exp = np.clip(exposure + noise, 0, 255).astype(np.uint8)
        exposures.append(exp)
    return exposures

def weighted_balance_and_rewrite(original_img, exposures):
    """第二步：加权平衡去噪，并将数据回写原图"""
    base = original_img.astype(np.float32)
    for exp in exposures:
        base += exp.astype(np.float32)
    # 平均化
    balanced = base / (len(exposures) + 1)
    return np.clip(balanced, 0, 255).astype(np.uint8)

def edge_enhancement_agent(img):
    """第三步：边缘增强Agent（舒展空间结构）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 使用 CLAHE 进行自适应对比度增强
    clahe = cv2.createCLAHE(
        clipLimit=CONFIG["AGENT_CLIP_LIMIT"], 
        tileGridSize=(8, 8)
    )
    l_enhanced = clahe.apply(l)
    
    # 合并回 LAB 并转回 BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def generate_stereo_pair(left_img):
    """第四步：生成左右眼图（科学视差投射）"""
    h, w = left_img.shape[:2]
    
    # 1. 提取深度（基于亮度，越亮越近）
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    depth = gray.astype(np.float32) / 255.0
    
    # 2. 反向映射（三角反推）
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            # 核心公式：视差 = 深度 * 强度
            disparity = (depth[y, x] - 0.5) * CONFIG["DISPARITY_STRENGTH"]
            map_x[y, x] = x + disparity
            map_y[y, x] = y
            
    # 3. 生成右眼图
    right_img = cv2.remap(left_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return left_img, right_img

def create_sbs_video(left_path, right_path, output_path, audio_path=None, bgm_path=None):
    """第五步：合成SBS视频并混音"""
    try:
        from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip, clips_array
    except Exception as e:
        raise ModuleNotFoundError(
            "缺少依赖 moviepy。请先安装：conda run -n base python -m pip install moviepy"
        ) from e

    left_clip = VideoFileClip(left_path)
    right_clip = VideoFileClip(right_path)
    
    # 拼接 SBS
    sbs_clip = clips_array([[left_clip, right_clip]])
    
    # 音频处理
    audio_clips = []
    if audio_path and os.path.exists(audio_path):
        audio_clips.append(AudioFileClip(audio_path))
    if bgm_path and os.path.exists(bgm_path):
        bgm = AudioFileClip(bgm_path).subclip(0, sbs_clip.duration)
        bgm = bgm.volumex(0.3) # BGM 音量
        audio_clips.append(bgm)
    
    if audio_clips:
        final_audio = CompositeAudioClip(audio_clips)
        sbs_clip = sbs_clip.set_audio(final_audio)
    
    # 写入文件
    sbs_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        fps=30
    )

# ==========================================
# 一键验证接口
# ==========================================

def process_image(input_path, output_dir):
    """处理单张图片"""
    print(f"🖼️ 处理图片: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print("❌ 无法读取图片")
        return

    # 执行全流程
    exposures = simulate_short_exposures(img, CONFIG["N_EXPOSURES"])
    denoised = weighted_balance_and_rewrite(img, exposures)
    enhanced = edge_enhancement_agent(denoised)
    left, right = generate_stereo_pair(enhanced)
    
    # 保存结果
    base_name = os.path.basename(input_path)
    cv2.imwrite(os.path.join(output_dir, f"L_{base_name}"), left)
    cv2.imwrite(os.path.join(output_dir, f"R_{base_name}"), right)
    
    # 生成红蓝测试图
    red_blue = np.zeros_like(left)
    red_blue[:, :, 0] = left[:, :, 0]  # Blue channel from Left
    red_blue[:, :, 2] = right[:, :, 2] # Red channel from Right
    cv2.imwrite(os.path.join(output_dir, f"TEST_{base_name}"), red_blue)
    print("✅ 图片处理完成！请查看 Output 文件夹。")

def process_video(input_path, output_dir):
    """处理视频（逐帧处理）"""
    print(f"🎬 处理视频: {input_path}")
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_left = os.path.join(output_dir, "temp_left.mp4")
    temp_right = os.path.join(output_dir, "temp_right.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_left = cv2.VideoWriter(temp_left, fourcc, fps, CONFIG["OUTPUT_SIZE"])
    out_right = cv2.VideoWriter(temp_right, fourcc, fps, CONFIG["OUTPUT_SIZE"])

    for _ in tqdm(range(frame_count), desc="视频处理进度"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 缩小尺寸以加快处理速度
        frame = cv2.resize(frame, CONFIG["OUTPUT_SIZE"])
        
        exposures = simulate_short_exposures(frame, CONFIG["N_EXPOSURES"])
        denoised = weighted_balance_and_rewrite(frame, exposures)
        enhanced = edge_enhancement_agent(denoised)
        left, right = generate_stereo_pair(enhanced)
        
        out_left.write(left)
        out_right.write(right)
        
    cap.release()
    out_left.release()
    out_right.release()
    
    # 合成最终视频
    print("🎵 正在合成SBS视频...")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    final_output = os.path.join(output_dir, f"SBS_{base_name}.mp4")
    create_sbs_video(temp_left, temp_right, final_output, input_path)
    
    # 清理临时文件
    os.remove(temp_left)
    os.remove(temp_right)
    print(f"✅ 视频处理完成！输出文件: {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="Input")
    parser.add_argument("--output_dir", default="Output")
    parser.add_argument("--no_prompt", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.input_dir == "Input" and not os.path.exists(args.input_dir):
        os.mkdir(args.input_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("=" * 40)
    print("🚀 裸眼3D 一键验证工具 v1.0")
    print("=" * 40)

    if not args.no_prompt and args.input_dir == "Input":
        print("请将图片或视频放入 'Input' 文件夹，然后按回车运行。")
        input("按回车键开始处理...")

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")

    files = glob.glob(os.path.join(args.input_dir, "*"))
    files.sort()
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print(f"❌ 输入目录为空: {args.input_dir}")
    else:
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                process_image(file, args.output_dir)
            elif file.lower().endswith((".mp4", ".mov", ".avi")):
                process_video(file, args.output_dir)

    print("\n🎉 所有任务完成！请在输出文件夹中查看结果。")
