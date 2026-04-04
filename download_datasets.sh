#!/usr/bin/env bash
set -euo pipefail

# 运行前确保已经安装 gdown：pip install gdown
# 如果你的环境中没有网络或被墙，可先配置代理。

BASE_DIR="$(pwd)/image_datasets"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# 1) DTU dataset (MVS):
# 官方页面: http://roboimagedata.compute.dtu.dk/
# 可能需要手动下载，以下示例是历史可用链接（逐个下载大文件）
# 对于所有视角、设定，数据量非常大（几十GB）

echo "Download DTU dataset (示例：扫描样本，约 10GB)"
# wget -c "http://roboimagedata.compute.dtu.dk/data/MVS/Rectified/scan1.zip"
# wget -c "http://roboimagedata.compute.dtu.dk/data/MVS/Rectified/scan2.zip"

# 2) PASMVS dataset
# 官方 GitHub: https://github.com/JiayuZhuo/PASMVS
# 使用 gdown 下载 Google Drive
# 示例（请替换为当前可用 ID，若过期请从 GitHub README 拷贝最新链接）
# gdown --folder "https://drive.google.com/drive/folders/1YourPASMVSFolderID"

# 3) BlendedMVS dataset
# 官方 GitHub: https://github.com/YoYo000/BlendedMVS
# 目前提供 Google Drive、百度网盘，使用 gdown 下载：
# gdown --folder "https://drive.google.com/drive/folders/1YourBlendedMVSFolderID"

# 解压并整理（示例）
# for f in *.zip; do unzip -o "$f" -d "${f%.zip}"; done

echo "下载命令已写入。请根据实际可用URL替换示例ID，运行完成后检查 $BASE_DIR" 
