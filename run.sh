set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$root_dir"

read_calib() {
  conda run -n base python -c $'import re, sys\np = sys.argv[1]\nlines = open(p, \"r\", encoding=\"utf-8\", errors=\"ignore\").read().splitlines()\nfx = None\nbaseline = None\nfor line in lines:\n    line = line.strip()\n    if line.startswith(\"cam0=\"):\n        m = re.search(r\"cam0=\\[(.*?)\\]\", line)\n        if m:\n            row0 = m.group(1).split(\";\")[0].strip()\n            fx = float(row0.split()[0])\n    if line.startswith(\"baseline=\"):\n        baseline = float(line.split(\"=\", 1)[1].strip())\nbaseline_m = (baseline / 1000.0) if (baseline is not None and baseline > 10.0) else baseline\nprint(fx, baseline_m)\n' "$1"
}

run_pair() {
  name="$1"
  left="$2"
  right="$3"
  calib="$4"
  out="$5"

  read -r fx baseline_m <<<"$(read_calib "$calib")"

  conda run -n base python CV_depth_information.py \
    --left "$left" \
    --right "$right" \
    --focal_length_px "$fx" \
    --baseline_m "$baseline_m" \
    --out_dir "$out"

  echo "OK: $name -> $out"
}

# 1) 默认测试（当前 left.png / right.png）
conda run -n base python CV_depth_information.py \
  --left left.png --right right.png --out_dir outputs_conda_base_test

# 2) Middlebury 2014（带 calib.txt）
run_pair "middlebury2014_adirondack" \
  "image_datasets/middlebury2014/Adirondack/Adirondack-perfect/im0.png" \
  "image_datasets/middlebury2014/Adirondack/Adirondack-perfect/im1.png" \
  "image_datasets/middlebury2014/Adirondack/Adirondack-perfect/calib.txt" \
  "outputs_middlebury2014_adirondack"

run_pair "middlebury2014_motorcycle" \
  "image_datasets/middlebury2014/Motorcycle/Motorcycle-perfect/im0.png" \
  "image_datasets/middlebury2014/Motorcycle/Motorcycle-perfect/im1.png" \
  "image_datasets/middlebury2014/Motorcycle/Motorcycle-perfect/calib.txt" \
  "outputs_middlebury2014_motorcycle"

# 3) ETH3D two-view（带 calib.txt）
run_pair "eth3d_delivery_area_1l" \
  "image_datasets/eth3d/ETH3D/delivery_area_1l/im0.png" \
  "image_datasets/eth3d/ETH3D/delivery_area_1l/im1.png" \
  "image_datasets/eth3d/ETH3D/delivery_area_1l/calib.txt" \
  "outputs_eth3d_delivery_area_1l"

run_pair "eth3d_terrains_1l" \
  "image_datasets/eth3d/ETH3D/terrains_1l/im0.png" \
  "image_datasets/eth3d/ETH3D/terrains_1l/im1.png" \
  "image_datasets/eth3d/ETH3D/terrains_1l/calib.txt" \
  "outputs_eth3d_terrains_1l"
