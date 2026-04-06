import cv2
import numpy as np
import os
from tqdm import tqdm

class StereoscopicGenerator:
    def __init__(self, mode='standard'):
        # 模式配置（对应界面上的三个选项）
        self.config = {
            'soft': {'N': 4, 'noise': 5, 'shift': 1, 'disparity': 1.5, 'pop': False},
            'standard': {'N': 6, 'noise': 8, 'shift': 2, 'disparity': 3.0, 'pop': False},
            'strong': {'N': 8, 'noise': 12, 'shift': 3, 'disparity': -5.0, 'pop': True} # 负值=出屏
        }[mode]

    def _image_fitting(self, left_img, right_img):
        """智能体图像拟合模块（防晕核心）"""
        # 1. 颜色匹配
        matched_right = cv2.xphoto.createSimpleWB().balanceWhite(right_img)
        
        # 2. 边缘平滑与空洞修复
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(matched_right, cv2.COLOR_BGR2GRAY)
        
        # 计算视差图用于引导滤波
        stereo = cv2.StereoSGBM_create(minDisparity=-16, numDisparities=32, blockSize=5)
        disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # 引导滤波，平滑视差，保持边缘
        if disp.max() > disp.min():
            disp_norm = (disp - disp.min()) / (disp.max() - disp.min())
            disp_smooth = cv2.ximgproc.guidedFilter(left_img, disp_norm, 10, 0.01)
            # 重新映射右眼图
            h, w = left_img.shape[:2]
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    map_x[y, x] = x + (disp_smooth[y, x] - 0.5) * self.config['disparity']
                    map_y[y, x] = y
            fitted_right = cv2.remap(matched_right, map_x, map_y, cv2.INTER_LINEAR)
        else:
            fitted_right = matched_right
            
        return left_img, fitted_right

    def process_single_image(self, img_path):
        """处理单张图片的完整流程"""
        img = cv2.imread(img_path)
        if img is None:
            return None, None

        # Step 1 & 2: 1+N 扰动与加权融合
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fused = np.zeros_like(gray, dtype=np.float32)
        for _ in range(self.config['N']):
            g = gray.copy()
            g += np.random.normal(0, self.config['noise'], g.shape)
            dx = np.random.randint(-self.config['shift'], self.config['shift'] + 1)
            dy = np.random.randint(-self.config['shift'], self.config['shift'] + 1)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            g = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]))
            fused += g
        fused /= self.config['N']
        
        # Step 3: 增强原图结构
        enhanced = cv2.addWeighted(img, 0.7, cv2.cvtColor(fused.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Step 4: 生成左右眼图（视差计算）
        h, w = enhanced.shape[:2]
        left_eye = enhanced
        disp_val = self.config['disparity']
        right_eye = cv2.warpAffine(enhanced, np.float32([[1, 0, disp_val], [0, 1, 0]]), (w, h))
        
        # Step 5: 智能体图像拟合
        left_final, right_final = self._image_fitting(left_eye, right_eye)
        
        return left_final, right_final

    def batch_run(self, input_folder, output_folder):
        """批量处理文件夹"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg'))]
        
        for file in tqdm(files, desc="生成进度"):
            img_path = os.path.join(input_folder, file)
            left, right = self.process_single_image(img_path)
            if left is not None:
                cv2.imwrite(os.path.join(output_folder, f'L_{file}'), left)
                cv2.imwrite(os.path.join(output_folder, f'R_{file}'), right)

# ===== 主程序入口 =====
if __name__ == '__main__':
    # 配置路径
    INPUT_DIR = r'image_datasets/timmer'   # 改成你的输入文件夹
    OUTPUT_DIR = r'image_datasets/outputs_standard'    # 改成你的输出文件夹
    
    # 选择模式: 'soft', 'standard', 'strong'
    generator = StereoscopicGenerator(mode='standard')
    
    # 开始批量生成
    generator.batch_run(INPUT_DIR, OUTPUT_DIR)
    print("✅ 批量生成完成！")