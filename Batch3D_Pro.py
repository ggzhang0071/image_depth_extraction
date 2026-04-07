import cv2
import numpy as np
import os
from tqdm import tqdm

class StereoscopicGenerator:
    def __init__(self, mode='standard'):
        self.config = {
            'soft': {'N': 4, 'noise': 5, 'shift': 1, 'disparity': 1.5, 'pop': False},
            'standard': {'N': 6, 'noise': 8, 'shift': 2, 'disparity': 3.0, 'pop': False},
            'strong': {'N': 8, 'noise': 12, 'shift': 3, 'disparity': -5.0, 'pop': True}
        }[mode]

    def _image_fitting(self, left_img, right_img):
        """智能体图像拟合模块"""
        if left_img is None or right_img is None:
            return None, None
        matched_right = right_img
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createSimpleWB"):
            matched_right = cv2.xphoto.createSimpleWB().balanceWhite(right_img)
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(matched_right, cv2.COLOR_BGR2GRAY)
        
        stereo = cv2.StereoSGBM_create(minDisparity=-16, numDisparities=32, blockSize=5)
        disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        if disp.max() > disp.min():
            disp_norm = (disp - disp.min()) / (disp.max() - disp.min())
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
                disp_smooth = cv2.ximgproc.guidedFilter(left_img, disp_norm, 10, 0.01)
            else:
                disp_smooth = cv2.GaussianBlur(disp_norm, (0, 0), 1.5)
            h, w = left_img.shape[:2]
            xs = np.arange(w, dtype=np.float32)[None, :]
            map_x = xs + (disp_smooth.astype(np.float32) - 0.5) * float(self.config["disparity"])
            map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
            fitted_right = cv2.remap(matched_right, map_x, map_y, cv2.INTER_LINEAR)
        else:
            fitted_right = matched_right
            
        return left_img, fitted_right

    def _create_red_blue_anaglyph(self, left_img, right_img):
        """核心：合成红蓝3D测试图"""
        # 确保尺寸一致
        h, w = left_img.shape[:2]
        right_resized = cv2.resize(right_img, (w, h))
        
        # 分离通道
        left_b = left_img[:, :, 0]   # 左图取蓝通道
        _, _, right_r = cv2.split(right_resized) # 右图取红通道
        
        # 合成 (BGR格式)
        anaglyph = np.zeros_like(left_img)
        anaglyph[:, :, 0] = left_b      # Blue channel from Left
        anaglyph[:, :, 2] = right_r     # Red channel from Right
        # Green channel is left empty (or you can use left_g for better brightness)
        
        return anaglyph

    def _create_sbs(self, left_img, right_img):
        if left_img is None or right_img is None:
            return None
        h = min(left_img.shape[0], right_img.shape[0])
        left = left_img[:h, :]
        right = right_img[:h, :]
        return np.hstack((left, right))

    def process_single_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None

        # Step 1 & 2: 1+N 扰动与加权融合
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fused = np.zeros_like(gray, dtype=np.float32)
        for _ in range(self.config['N']):
            g = gray.astype(np.float32)
            g += np.random.normal(0, self.config['noise'], g.shape).astype(np.float32)
            g = np.clip(g, 0.0, 255.0)
            dx = np.random.randint(-self.config['shift'], self.config['shift'] + 1)
            dy = np.random.randint(-self.config['shift'], self.config['shift'] + 1)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            g = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]))
            fused += g
        fused /= self.config['N']
        
        # Step 3: 增强原图结构
        enhanced = cv2.addWeighted(img, 0.7, cv2.cvtColor(fused.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Step 4: 生成左右眼图
        h, w = enhanced.shape[:2]
        left_eye = enhanced
        disp_val = self.config['disparity']
        right_eye = cv2.warpAffine(enhanced, np.float32([[1, 0, disp_val], [0, 1, 0]]), (w, h))
        
        # Step 5: 智能体图像拟合
        left_final, right_final = self._image_fitting(left_eye, right_eye)
        if left_final is None or right_final is None:
            return None, None, None
        
        # Step 6: 生成红蓝测试图
        red_blue_test = self._create_red_blue_anaglyph(left_final, right_final)
        
        return left_final, right_final, red_blue_test

    def batch_run(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in tqdm(files, desc="生成进度"):
            img_path = os.path.join(input_folder, file)
            left, right, redblue = self.process_single_image(img_path)
            
            if left is not None:
                base_name = os.path.splitext(file)[0]
                left_path = os.path.join(output_folder, f"L_{file}")
                right_path = os.path.join(output_folder, f"R_{file}")
                cv2.imwrite(left_path, left)
                cv2.imwrite(right_path, right)
                cv2.imwrite(os.path.join(output_folder, f"TEST_RedBlue_{base_name}.jpg"), redblue)

                sbs = self._create_sbs(left, right)
                if sbs is not None:
                    cv2.imwrite(os.path.join(output_folder, f"ReadyForPhone_SBS_{base_name}.jpg"), sbs)

# ===== 主程序入口 =====
if __name__ == '__main__':
    INPUT_DIR = r'image_datasets/timmer'   # 改成你的输入文件夹
    OUTPUT_DIR = r'image_datasets/outputs_standard'    # 改成你的输出文件夹
    
    generator = StereoscopicGenerator(mode='standard')
    generator.batch_run(INPUT_DIR, OUTPUT_DIR)
    print("✅ 批量生成完成！包含左右眼图和红蓝测试图。")
