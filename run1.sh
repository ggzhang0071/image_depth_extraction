set -euo pipefail

root_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$root_dir"

input_dir="${1:-cup}"
num_images="${2:-5}"
out_dir="${3:-outputs_image_enhancement}"

mkdir -p "$out_dir"

if [ "$(basename "$input_dir")" = "left" ] || [ "$(basename "$input_dir")" = "right" ]; then
  base_dir="$(dirname "$input_dir")"
else
  base_dir="$input_dir"
fi

left_dir="${base_dir}/left"
right_dir="${base_dir}/right"

mapfile -t left_images < <(
  find "$left_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    | sort \
    | head -n "$num_images"
)

mapfile -t right_images < <(
  find "$right_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    | sort \
    | head -n "$num_images"
)

images=("${left_images[@]}" "${right_images[@]}")

if [ "${#images[@]}" -lt 2 ]; then
  echo "错误: ${left_dir} + ${right_dir} 下需要至少 2 张图片，当前找到 ${#images[@]} 张"
  exit 2
fi

echo "使用以下图片进行处理："
echo "Left:"
printf ' - %s\n' "${left_images[@]}"
echo "Right:"
printf ' - %s\n' "${right_images[@]}"

left_compare="${out_dir}/left_compare.jpg"
right_compare="${out_dir}/right_compare.jpg"

conda run -n base python CV_image_enhancement.py \
  --imgs "${left_images[@]}" \
  --compare \
  --left_label "left 原图" \
  --right_label "left 锐化" \
  --out "$left_compare"

conda run -n base python CV_image_enhancement.py \
  --imgs "${right_images[@]}" \
  --compare \
  --left_label "right 原图" \
  --right_label "right 锐化" \
  --out "$right_compare"

conda run -n base python -c $'import os, sys\nimport cv2\nimport numpy as np\n\nleft_path = sys.argv[1]\nright_path = sys.argv[2]\nout_path = sys.argv[3]\n\nleft = cv2.imread(left_path, cv2.IMREAD_COLOR)\nright = cv2.imread(right_path, cv2.IMREAD_COLOR)\nif left is None:\n    raise RuntimeError(f\"Failed to read: {left_path}\")\nif right is None:\n    raise RuntimeError(f\"Failed to read: {right_path}\")\n\nw = max(left.shape[1], right.shape[1])\nif left.shape[1] != w:\n    left = cv2.copyMakeBorder(left, 0, 0, 0, w - left.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18))\nif right.shape[1] != w:\n    right = cv2.copyMakeBorder(right, 0, 0, 0, w - right.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18))\n\ngap = 24\nsep = np.full((gap, w, 3), 18, dtype=np.uint8)\nout = np.vstack([left, sep, right])\n\nos.makedirs(os.path.dirname(out_path) or \".\", exist_ok=True)\nok = cv2.imwrite(out_path, out)\nif not ok:\n    raise RuntimeError(f\"Failed to write: {out_path}\")\n' "$left_compare" "$right_compare" "${out_dir}/final_sharpened.jpg"

echo "输出: ${out_dir}/final_sharpened.jpg"
echo "对比: ${left_compare}"
echo "对比: ${right_compare}"
