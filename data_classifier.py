import pandas as pd
import os
import shutil
import concurrent.futures
from tqdm import tqdm
import threading

# --- 1. 用户配置 ---
METADATA_FILE = 'metadata.csv'
CLINICAL_DATA_EXCEL_FILE = 'CMMD_clinicaldata_revision.xlsx'
# 使用一个新的文件夹名，以防和旧数据混淆
NEW_DATASET_ROOT = 'organized_dcm_dataset_unique' 
MAX_WORKERS = os.cpu_count() or 4 

# --- 2. 线程安全的计数器 ---
class ThreadSafeCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_files_copied = 0
        self.records_processed = 0
        self.invalid_records = 0

    def increment_files(self, count=1):
        with self._lock:
            self.total_files_copied += count
    def increment_records(self):
        with self._lock:
            self.records_processed += 1
    def increment_invalid(self):
        with self._lock:
            self.invalid_records += 1

# --- 3. 单个任务处理函数 (包含新的命名逻辑) ---
def process_record(row_data, counter):
    """
    处理单条记录：找到文件夹中的.dcm文件并以唯一名称复制它们。
    """
    index, row = row_data
    source_folder_path = row['File Location']
    classification = str(row.get('classification', '')).capitalize()
    subject_id = row['Subject ID'] # 获取患者ID用于命名

    # 数据校验
    if not os.path.isdir(source_folder_path):
        counter.increment_invalid()
        return
    if classification not in ['Benign', 'Malignant']:
        counter.increment_invalid()
        return

    destination_folder = os.path.join(NEW_DATASET_ROOT, classification)
    os.makedirs(destination_folder, exist_ok=True)

    try:
        files_copied_in_task = 0
        for filename in os.listdir(source_folder_path):
            if filename.lower().endswith('.dcm'):
                source_file = os.path.join(source_folder_path, filename)
                
                # --- !! 核心改动：创建唯一的文件名 !! ---
                new_filename = f"{subject_id}_{filename}"
                destination_file = os.path.join(destination_folder, new_filename)
                
                shutil.copy(source_file, destination_file)
                files_copied_in_task += 1
        
        if files_copied_in_task > 0:
            counter.increment_files(files_copied_in_task)
            counter.increment_records()
    except Exception as e:
        print(f"\n[错误] 处理文件夹 '{source_folder_path}' 时出错: {e}")
        counter.increment_invalid()

# --- 4. 主函数 ---
def organize_files_multithreaded():
    """
    主函数，加载数据、创建线程池并分发任务。
    """
    try:
        print("步骤 1: 正在加载 'metadata.csv' 和 'CMMD_clinicaldata_revision.xlsx'...")
        df_meta = pd.read_csv(METADATA_FILE)
        df_labels = pd.read_excel(CLINICAL_DATA_EXCEL_FILE, engine='openpyxl')
        
        print("步骤 2: 正在将文件路径与分类标签进行链接 (基于 Subject ID 和 ID1)...")
        df_merged = pd.merge(df_meta, df_labels, left_on='Subject ID', right_on='ID1', how='left')
        tasks = list(df_merged.iterrows())
        print(f"数据准备完毕，总共 {len(tasks)} 条记录需要处理。")
    except Exception as e:
        print(f"错误：无法加载或处理数据文件。请检查文件是否存在且格式正确。\n错误信息: {e}")
        return

    counter = ThreadSafeCounter()

    print(f"\n步骤 3: 使用 {MAX_WORKERS} 个线程开始扫描路径并以唯一名称复制文件...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, task, counter) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="处理记录"):
            pass

    print("\n--- 整理完成！ ---\n")
    print(f"成功处理了 {counter.records_processed} 个有效的患者记录。")
    print(f"总共复制了 {counter.total_files_copied} 个 .dcm 文件 (已重命名以防覆盖)。")
    print(f"跳过了 {counter.invalid_records} 个无效或错误的记录。")
    print(f"数据已分类存放在新文件夹: '{NEW_DATASET_ROOT}'")

if __name__ == '__main__':
    organize_files_multithreaded()