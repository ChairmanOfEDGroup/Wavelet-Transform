import os
import shutil
import random
import concurrent.futures

# --- 可配置参数 ---

# 1. 源数据文件夹 (根据您的图片)
SOURCE_DIR = r'D:\MATH663_Project\manifest-1616439774456\compressed_datasets\compressed_0.00001_percent'

# 2. 划分后数据的输出文件夹
OUTPUT_DIR = 'compressed_0.00001_percent' 

# 3. 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO 将是剩余的部分 (1.0 - 0.8 - 0.1 = 0.1)

# 4. 随机种子，确保每次划分结果都一样，便于复现
RANDOM_SEED = 42

# 5. 使用的最大线程数
MAX_WORKERS = os.cpu_count()

# -------------------

random.seed(RANDOM_SEED)

def copy_file(src_path, dst_path):
    """
    单个文件复制函数，用于多线程调用。
    """
    try:
        # 确保目标文件夹存在 (虽然主线程已经创建，但这里再检查一次更安全)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path) # copy2 会保留元数据
        return None # 成功时返回 None
    except Exception as e:
        return f"Error copying {src_path} to {dst_path}: {e}"

def main():
    print(f"[*] 开始处理数据集...")
    print(f"    源文件夹: {SOURCE_DIR}")
    print(f"    输出文件夹: {OUTPUT_DIR}")

    # 1. 检查源文件夹
    if not os.path.isdir(SOURCE_DIR):
        print(f"[!] 错误: 源文件夹 '{SOURCE_DIR}' 不存在。")
        return

    # 2. 自动查找所有类别
    try:
        classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
        if not classes:
            print(f"[!] 错误: 在 '{SOURCE_DIR}' 中没有找到类别子文件夹 (如 'Benign', 'Malignant')。")
            return
        print(f"[*] 找到 {len(classes)} 个类别: {classes}")
    except Exception as e:
        print(f"[!] 错误: 无法读取源文件夹: {e}")
        return

    # 3. (关键步骤) 预先创建所有目标文件夹
    #    这可以避免多线程同时创建同一个文件夹时发生竞争条件 (race condition)
    print(f"[*] 正在创建输出目录结构...")
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            path = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(path, exist_ok=True)

    # 4. 准备所有复制任务列表（stratified split）
    all_copy_tasks = [] # 列表，每个元素是 (src_path, dst_path)
    
    for class_name in classes:
        print(f"[*] 正在处理类别: {class_name}")
        class_src_dir = os.path.join(SOURCE_DIR, class_name)
        
        try:
            # 过滤掉非文件项 (比如 .DS_Store 等)
            files = [f for f in os.listdir(class_src_dir) if os.path.isfile(os.path.join(class_src_dir, f))]
        except Exception as e:
            print(f"    [!] 无法读取 {class_src_dir}: {e}。跳过此类。")
            continue
            
        random.shuffle(files) # 打乱文件列表
        
        n = len(files)
        if n == 0:
            print(f"    [!] 类别 {class_name} 为空，已跳过。")
            continue

        # 计算分割点
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        # 分配文件
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:] # 剩余的都给 test

        print(f"    总计: {n} | 训练: {len(train_files)} | 验证: {len(val_files)} | 测试: {len(test_files)}")

        # 创建复制任务 (src, dst)
        for f in train_files:
            src = os.path.join(class_src_dir, f)
            dst = os.path.join(OUTPUT_DIR, 'train', class_name, f)
            all_copy_tasks.append((src, dst))
            
        for f in val_files:
            src = os.path.join(class_src_dir, f)
            dst = os.path.join(OUTPUT_DIR, 'val', class_name, f)
            all_copy_tasks.append((src, dst))

        for f in test_files:
            src = os.path.join(class_src_dir, f)
            dst = os.path.join(OUTPUT_DIR, 'test', class_name, f)
            all_copy_tasks.append((src, dst))

    print(f"\n[*] 准备就绪，总共需要复制 {len(all_copy_tasks)} 个文件。")

    # 5. (核心) 使用多线程执行复制任务
    print(f"[*] 开始使用 {MAX_WORKERS} 个线程并发复制文件...")
    errors = []
    
    # 使用线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        # executor.submit(fn, arg1, arg2)
        future_to_task = {executor.submit(copy_file, src, dst): (src, dst) for src, dst in all_copy_tasks}
        
        processed_count = 0
        
        # as_completed 会在任务完成时立即返回结果
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result() # 获取 copy_file 的返回值
            processed_count += 1
            
            if result: # 如果返回值不是 None, 说明发生了错误
                errors.append(result)
                print(f"[!] {result}") # 立即打印错误

            # 简单的进度显示
            if processed_count % (len(all_copy_tasks) // 20 + 1) == 0:
                 print(f"    ...已处理 {processed_count} / {len(all_copy_tasks)} ({(processed_count/len(all_copy_tasks)*100):.1f}%)")

    # 6. 最终报告
    print("\n" + "="*30)
    print("[*] 数据集划分完成！")
    print(f"    成功复制: {len(all_copy_tasks) - len(errors)} 个文件")
    print(f"    发生错误: {len(errors)} 个文件")
    if errors:
        print("[!] 详情请查看上面的错误日志。")
    print(f"[*] 数据已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("="*30)


if __name__ == "__main__":
    main()