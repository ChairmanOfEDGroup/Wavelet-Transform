import os
import sys
import time
import csv
import random
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# --- ROI裁剪函数 (无变化) ---
def crop_to_roi(image):
    """
    将一个PIL图像裁剪到其非黑色内容的边界框。
    """
    img_array = np.array(image)
    if len(img_array.shape) > 2:
        non_empty_pixels = np.where(np.any(img_array > 0, axis=-1))
    else:
        non_empty_pixels = np.where(img_array > 0)
    if non_empty_pixels[0].size == 0 or non_empty_pixels[1].size == 0:
        return image
    min_y, max_y = np.min(non_empty_pixels[0]), np.max(non_empty_pixels[0])
    min_x, max_x = np.min(non_empty_pixels[1]), np.max(non_empty_pixels[1])
    return image.crop((min_x, min_y, max_x + 1, max_y + 1))

# --- 1. 数据集加载器 (无变化) ---
class FlexibleImageDataset(Dataset):
    """
    数据集加载器，接收一个文件路径列表，并应用指定的转换。
    新增 augment_rotation 标志来控制是否应用4倍旋转增强。
    """
    def __init__(self, file_paths, transform=None, augment_rotation=False):
        self.file_paths = file_paths
        self.transform = transform
        self.augment_rotation = augment_rotation
        
    def __len__(self):
        if self.augment_rotation:
            return len(self.file_paths) * 4
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.augment_rotation:
            original_img_idx = idx // 4
            rotation_angle = (idx % 4) * 90
        else:
            original_img_idx = idx
            rotation_angle = 0 # 验证/测试集不进行旋转

        img_path, label = self.file_paths[original_img_idx]
        
        try:
            if img_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(img_path)
                pixel_array = ds.pixel_array
                window_center, window_width = -600, 1500
                img_min, img_max = window_center - window_width // 2, window_center + window_width // 2
                pixel_array = np.clip(pixel_array, img_min, img_max)
                pixel_array = ((pixel_array - img_min) / (img_max - img_min)) * 255.0
                image = Image.fromarray(pixel_array.astype(np.uint8)).convert("RGB")
            else:
                image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\n警告：加载文件 '{img_path}' 失败. 错误: {e}. 将返回黑色图像。")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        image = crop_to_roi(image)
        image = image.rotate(rotation_angle)

        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. Early Stopping 类 (无变化) ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience, self.verbose, self.counter, self.best_score = patience, verbose, 0, None
        self.early_stop, self.val_loss_min, self.delta, self.path = False, np.inf, delta, path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score, self.val_loss_min = score, val_loss
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score, self.val_loss_min, self.counter = score, val_loss, 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'验证损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型至 {self.path} ...')
        torch.save(model.state_dict(), self.path)

# --- 辅助函数 (无变化) ---
def load_paths_from_split(split_dir, class_to_idx):
    """
    从一个已分割的目录 (如 'train', 'val', 'test') 加载文件路径和标签。
    """
    file_paths = []
    supported_formats = ('.jpg','.jpeg','.png','.dcm')
    
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"警告: 在 {split_dir} 中未找到类别目录: {class_name}")
            continue
        
        for f in os.listdir(class_dir):
            if f.lower().endswith(supported_formats):
                file_paths.append((os.path.join(class_dir, f), idx))
    return file_paths

# --- 3. 核心训练与测试函数 (!! 模型已修改 !!) ---
def run_training_and_testing(data_dir): 
    # --- 配置 (无变化) ---
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE = 20, 16, 0.0001, 3
    NUM_WORKERS = min(os.cpu_count(), 8)
    MODEL_SAVE_PATH = f"best_model_{os.path.basename(data_dir)}.pt"

    # --- 转换流程 (无变化) ---
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # --- 数据加载 (无变化) ---
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    if not all(os.path.isdir(d) for d in [train_dir, val_dir, test_dir]):
        print(f"错误: '{data_dir}' 必须包含 'train', 'val', 和 'test' 子目录。跳过此数据集。")
        return None, None
    
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not classes:
        print(f"错误: 在 '{train_dir}' 中未找到类别子文件夹 (如 'Benign', 'Malignant')。跳过此数据集。")
        return None, None
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    train_paths = load_paths_from_split(train_dir, class_to_idx)
    val_paths = load_paths_from_split(val_dir, class_to_idx)
    test_paths = load_paths_from_split(test_dir, class_to_idx)

    if not train_paths or not val_paths or not test_paths:
        print(f"错误: '{data_dir}' 的 train/val/test 目录中至少有一个为空。请检查数据。跳过此数据集。")
        return None, None

    train_dataset = FlexibleImageDataset(file_paths=train_paths, transform=train_transforms, augment_rotation=True)
    val_dataset = FlexibleImageDataset(file_paths=val_paths, transform=val_test_transforms, augment_rotation=False)
    test_dataset = FlexibleImageDataset(file_paths=test_paths, transform=val_test_transforms, augment_rotation=False)

    print("\n" + "="*50)
    print(f"开始处理: {os.path.basename(data_dir)}")
    print(f" - 自动检测类别: {classes}")
    print(f" - (已加载) 训练集图像: {len(train_paths)}")
    print(f" - (已加载) 验证集图像: {len(val_paths)}")
    print(f" - (已加载) 测试集图像: {len(test_paths)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- (!! 已修改 !!) 模型、损失函数、优化器 ---
    train_labels = [label for _, label in train_paths]
    class_weights = torch.FloatTensor([len(train_labels)/Counter(train_labels)[i] for i in range(len(classes))]).to(device)
    
    # --- (!! 已修改 !!) ---
    # 从 resnet34 改为 resnet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) 
    # --- (!! 修改结束 !!) ---

    # 自动获取 `in_features`，因此无需更改
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=MODEL_SAVE_PATH)

    # --- 训练与验证循环 (无变化) ---
    epoch_times = []
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        for inputs, labels in tqdm(train_loader, desc="训练中"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="验证中"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += nn.CrossEntropyLoss()(outputs, labels).item() * inputs.size(0)
        
        val_epoch_loss = val_loss / len(val_dataset)
        print(f"验证 Loss: {val_epoch_loss:.4f}")
        scheduler.step(val_epoch_loss)
        epoch_times.append(time.time() - epoch_start_time)
        
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("触发 Early Stopping")
            break

    # --- 最终测试 (无变化) ---
    print("\n--- 在测试集上进行最终评估 ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试中"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()) # 假设是二分类，取类别1的概率

    test_acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    
    avg_method = 'binary' if len(classes) == 2 else 'weighted'
    test_precision = precision_score(all_labels, all_preds, average=avg_method, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average=avg_method, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average=avg_method, zero_division=0)
    
    test_auc = 0.0
    if len(classes) == 2:
        try:
            test_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            test_auc = 0.0
    else:
        print("AUC仅在二分类时计算。")
        
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    
    final_results = {
        "Dataset": os.path.basename(data_dir),
        "Test_Accuracy": f"{test_acc:.4f}",
        "Test_F1_Score": f"{test_f1:.4f}",
        "Test_AUC": f"{test_auc:.4f}",
        "Test_Precision": f"{test_precision:.4f}",
        "Test_Recall": f"{test_recall:.4f}",
        "Avg_Epoch_Time_sec": f"{avg_epoch_time:.2f}"
    }
    return final_results, MODEL_SAVE_PATH

# --- 4. 主程序入口 (无变化) ---
if __name__ == '__main__':
    
    # --- (!! 已修改 !!) ---
    # 自动获取脚本 (auto.py) 所在的目录
    # 例如: D:\MATH663_Project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据您的描述, 数据集的基础目录是脚本所在目录下的 "data_split" 文件夹
    # 例如: D:\MATH663_Project\data_split
    DATASETS_BASE_DIR = os.path.join(script_dir, "data_split")
    # --- (!! 修改结束 !!) ---

    
    # 要训练的数据集列表保持不变 (这些是 data_split 内部的子文件夹)
    datasets_to_train = [
        "compressed_0.00001_percent", 
        "compressed_0.0001_percent", 
        "compressed_0.001_percent",
        "compressed_0.01_percent", 
        "compressed_0.1_percent", 
        "compressed_1_percent",
        "compressed_10_percent",
        "compressed_100_percent"
    ]
    
    # 日志文件将保存在脚本所在的目录 (D:\MATH663_Project\training_test_results_resnet50.csv)
    log_file = os.path.join(script_dir, 'training_test_results_resnet50.csv')
    fieldnames = ["Dataset", "Test_Accuracy", "Test_F1_Score", "Test_AUC", 
                  "Test_Precision", "Test_Recall", "Avg_Epoch_Time_sec"]
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    overall_start_time = time.time()
    
    # 循环逻辑保持不变
    for dataset_name in datasets_to_train:
        # data_dir 现在会自动指向: D:\MATH663_Project\data_split\[dataset_name]
        data_dir = os.path.join(DATASETS_BASE_DIR, dataset_name) 
        
        if not os.path.isdir(data_dir):
            print(f"警告: 目录 '{data_dir}' 不存在，已跳过。")
            continue
        
        # run_training_and_testing 会去 data_dir 内部寻找 train/val/test
        results, best_model_path = run_training_and_testing(data_dir) 
        
        if results:
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(results)
            print(f"数据集 '{dataset_name}' (ResNet50) 的测试结果已追加到 {log_file}")
            print(f"此数据集的最佳模型已保存至: {best_model_path}")
    
    overall_time_min = (time.time() - overall_start_time) / 60
    print("\n" + "="*50)
    print("所有数据集处理完毕！")
    print(f"总耗时: {overall_time_min:.2f} 分钟")
    print(f"详细日志请查看: {log_file}")
    print("="*50)