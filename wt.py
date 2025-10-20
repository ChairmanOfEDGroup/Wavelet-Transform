import os
import pydicom
import numpy as np
import pywt
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import sys

# ==================== 配置区 ====================

INPUT_ROOT_DIR = r'D:\MATH663_Project\manifest-1616439774456\compressed_datasets\original'
OUTPUT_ROOT_DIR = r'D:\MATH663_Project\manifest-1616439774456\compressed_datasets'

if len(sys.argv) > 1:
    try:
        ratio = float(sys.argv[1])
        COEFFICIENT_PERCENTAGES_TO_KEEP = [ratio]
        print(f"📌 从命令行读取压缩比例: {ratio}")
    except:
        print("❌ 参数无效，请输入 0~1 的小数，比如 python wt.py 0.05")
        sys.exit(1)
else:
    COEFFICIENT_PERCENTAGES_TO_KEEP = [0.1]  # 默认值

WAVELET_TYPE = 'db4'
JPEG_QUALITY = 95
NUM_WORKERS = os.cpu_count()
WAVELET_LEVEL = 3  # 限制分解层数，防止极端压缩导致伪清晰

# ==================================================

def compress_dcm_and_save(dcm_path, output_path, keep_percentage):
    """
    读取单个 DICOM 文件，执行小波压缩并保存为 JPEG。
    """
    try:
        # 读取 DICOM 文件
        ds = pydicom.dcmread(dcm_path)
        pixels = ds.pixel_array.astype(np.float32)

        # 归一化到 [0, 255]
        if np.ptp(pixels) > 0:
            pixels = (pixels - np.min(pixels)) / np.ptp(pixels) * 255.0

        # 执行二维小波分解
        coeffs = pywt.wavedec2(pixels, WAVELET_TYPE, level=WAVELET_LEVEL)
        coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        # --- 改进阈值逻辑：直接保留前 N% 最大系数 ---
        num_to_keep = int(coeffs_arr.size * keep_percentage)
        if num_to_keep < 1:
            num_to_keep = 1  # 避免空数组
        abs_vals = np.abs(coeffs_arr).flatten()
        threshold = np.partition(abs_vals, -num_to_keep)[-num_to_keep]
        coeffs_arr_compressed = coeffs_arr * (np.abs(coeffs_arr) >= threshold)

        # 反变换重建图像
        coeffs_compressed = pywt.array_to_coeffs(coeffs_arr_compressed, coeff_slices, output_format='wavedec2')
        reconstructed = pywt.waverec2(coeffs_compressed, WAVELET_TYPE)
        reconstructed = np.clip(reconstructed, 0, 255)

        # --- 改进归一化：防止对比度伪增强 ---
        reconstructed = (reconstructed - np.min(reconstructed)) / (np.ptp(reconstructed) + 1e-8) * 255
        reconstructed = reconstructed.astype(np.uint8)

        # 保存 JPEG
        img = Image.fromarray(reconstructed)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'JPEG', quality=JPEG_QUALITY)

        return True

    except Exception as e:
        print(f"[❌ 错误] 文件 '{dcm_path}' 处理失败: {e}")
        return False


def process_task(task_args):
    """线程入口函数"""
    return compress_dcm_and_save(**task_args)


def main():
    print("🚀 开始生成多版本小波压缩样本集（多线程优化版）")

    if not os.path.isdir(INPUT_ROOT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_ROOT_DIR}")
        return

    tasks = []
    print("🔍 正在扫描 DICOM 文件...")
    for root, _, files in os.walk(INPUT_ROOT_DIR):
        for file in files:
            if file.lower().endswith('.dcm'):
                dcm_path = os.path.join(root, file)
                for keep_percent in COEFFICIENT_PERCENTAGES_TO_KEEP:
                    relative_path = os.path.relpath(dcm_path, INPUT_ROOT_DIR)
                    rel_no_ext = os.path.splitext(relative_path)[0]
                    level_folder = f"compressed_{str(keep_percent * 100).rstrip('0').rstrip('.')}_percent"
                    output_path = os.path.join(OUTPUT_ROOT_DIR, level_folder, f"{rel_no_ext}.jpeg")

                    tasks.append({
                        'dcm_path': dcm_path,
                        'output_path': output_path,
                        'keep_percentage': keep_percent
                    })

    if not tasks:
        print("⚠️ 没有找到任何 DCM 文件。")
        return

    total = len(tasks)
    print(f"✅ 共找到 {total // len(COEFFICIENT_PERCENTAGES_TO_KEEP)} 个 DICOM 文件，生成 {total} 个压缩任务。")

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_task, tasks, chunksize=10), total=total, desc="📦 压缩进度"))

    print("-" * 60)
    print(f"🎉 所有文件处理完毕！压缩结果保存在：{OUTPUT_ROOT_DIR}")


if __name__ == "__main__":
    main()
