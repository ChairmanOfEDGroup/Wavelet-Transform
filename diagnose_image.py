import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import os

def diagnose_image(image_path):
    """
    加载一张图片，模拟训练脚本中的处理流程，并可视化模型真正“看到”的数据。
    """
    print("="*50)
    print(f"🔍 正在诊断文件: {image_path}")
    print("="*50)

    if not os.path.exists(image_path):
        print(f"❌ 错误: 文件不存在！请检查路径。")
        return

    try:
        # --- 步骤 1: 加载原始像素数据 ---
        if image_path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(image_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            print("✅ 成功以 DICOM 格式加载。")
        else:
            # 对于 JPEG，我们直接用 Pillow 加载
            # 注意：JPEG已经是8位有损格式，信息量远低于DCM
            with Image.open(image_path) as img:
                pixel_array = np.array(img).astype(np.float32)
            print("✅ 成功以 JPEG/PNG 格式加载。")

        # --- 步骤 2: 打印原始数据的统计信息 (揭示真相的关键！) ---
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        mean_val = pixel_array.mean()
        std_val = pixel_array.std()

        print("\n--- 原始数据统计 (加载后，归一化前) ---")
        print(f"数据类型 (dtype): {pixel_array.dtype}")
        print(f"图像形状 (Shape): {pixel_array.shape}")
        print(f"最小像素值 (Min): {min_val}")
        print(f"最大像素值 (Max): {max_val}")
        print(f"平均像素值 (Mean): {mean_val:.2f}")
        print(f"像素标准差 (Std): {std_val:.2f}")
        
        if max_val == 0:
            print("\n⚠️ 警告: 图像所有像素值均为0，这是一张纯黑图像。")
        else:
            print("\n💡 发现: 最大像素值远大于0。这意味着图像中存在非黑色信息！")


        # --- 步骤 3: 执行与训练脚本完全相同的归一化 ---
        # 这是将高动态范围数据映射到 [0, 1] 区间的关键步骤
        if max_val > min_val:
            normalized_array = (pixel_array - min_val) / (max_val - min_val)
            print("\n✅ 已执行归一化: (pixel - min) / (max - min)")
        else:
            normalized_array = pixel_array # 避免除以零
            print("\n⚠️ 图像所有像素值相同，无法进行归一化。")

        # --- 步骤 4: 保存模型真正“看到”的图像 ---
        output_filename = f"diagnostic_output_of_{os.path.basename(image_path)}.png"
        
        # 使用 matplotlib 保存，因为它可以正确处理 [0, 1] 范围的浮点数
        # 我们使用灰度图 (cmap='gray') 来忠实地呈现单通道信息
        plt.imsave(output_filename, normalized_array, cmap='gray')
        print(f"\n✅ [重要] 已将模型归一化后看到的可视化结果保存为: '{output_filename}'")
        print("--- 请打开这张图片查看！---")


    except Exception as e:
        print(f"❌ 处理失败: {e}")

# --- 主程序入口 ---
if __name__ == '__main__':
    # ======================================================================
    # --- !! 在这里输入那张“全黑”图片的完整路径 !! ---
    #
    # 示例:
    # IMAGE_PATH_TO_CHECK = r"D:\...\compressed_datasets\compressed_0.001_percent\Malignant\D1-0132_1-2.jpeg"
    #
    IMAGE_PATH_TO_CHECK = r"D:\MATH663_Project\manifest-1616439774456\compressed_datasets\compressed_0.0000001_percent\Benign\D1-0001_1-1.jpeg"
    # ======================================================================
    
    diagnose_image(IMAGE_PATH_TO_CHECK)