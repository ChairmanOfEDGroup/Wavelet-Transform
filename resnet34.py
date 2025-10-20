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
from torchvision import models, transforms
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# --- 1. 数据集加载器 (可处理 JPG, PNG, DCM) ---
class FlexibleImageDataset(Dataset):
    """一个可以加载标准图像格式和DICOM文件的PyTorch数据集类。"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.file_paths = self._get_file_paths()
        
        if len(self.classes) != 2:
            print(f"警告：检测到 {len(self.classes)} 个类别。此脚本专为二分类设计，AUC等指标将按二分类计算。")

    def _get_file_paths(self):
        paths = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.dcm')
        print("正在扫描图像文件...")
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(valid_exts):
                    paths.append((os.path.join(cls_dir, file_name), self.class_to_idx[cls_name]))
        return paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx]
        
        try:
            if img_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(img_path)
                pixel_array = ds.pixel_array.astype(np.float32)
                min_val, max_val = pixel_array.min(), pixel_array.max()
                if max_val > min_val:
                    pixel_array = (pixel_array - min_val) / (max_val - min_val)
                image_rgb = np.stack([pixel_array] * 3, axis=-1)
                image = Image.fromarray((image_rgb * 255).astype(np.uint8))
            else:
                image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\n警告：加载文件失败 '{img_path}'. 错误: {e}. 将返回一个黑色图像。")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_labels(self):
        """获取数据集中所有样本的标签。"""
        return [label for _, label in self.file_paths]

# --- 2. Early Stopping 类 ---
class EarlyStopping:
    """在验证损失不再改善时提前停止训练。"""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience; self.verbose = verbose; self.counter = 0; self.best_score = None
        self.early_stop = False; self.val_loss_min = np.inf; self.delta = delta; self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 3. 核心训练函数 ---
def run_training(data_dir):
    """在指定的数据集上执行完整的训练流程。"""
    # --- 配置 ---
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    PATIENCE = 3
    NUM_WORKERS = min(os.cpu_count(), 8)
    MODEL_SAVE_PATH = f"best_model_{os.path.basename(data_dir)}.pt"

    # --- 数据增强与转换 ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 数据加载 ---
    full_dataset = FlexibleImageDataset(root_dir=data_dir, transform=data_transforms)
    
    if not full_dataset.file_paths:
        print(f"错误: 在目录 '{data_dir}' 中未找到任何有效的图像文件。")
        return None

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"数据集信息:")
    print(f" - 总图像数: {len(full_dataset)}")
    print(f" - 训练集大小: {len(train_dataset)}")
    print(f" - 验证集大小: {len(val_dataset)}")
    print(f" - 类别: {full_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"模型将在设备 '{device}' 上运行。")

    # --- 处理类别不平衡 ---
    train_labels = [full_dataset.file_paths[i][1] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    num_samples = len(train_labels)
    
    print("训练集类别分布:")
    for i, class_name in enumerate(full_dataset.classes):
        print(f" - {class_name}: {class_counts[i]} 个样本")
        
    class_weights = [num_samples / class_counts[i] for i in range(len(full_dataset.classes))]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"计算出的类别权重: {class_weights_tensor.cpu().numpy()}")

    # --- 模型、损失函数、优化器 ---
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # 在验证阶段，我们使用不带权重的损失函数来公平地评估
    criterion_val = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=MODEL_SAVE_PATH)

    # --- 训练循环 ---
    start_time = time.time()
    best_val_f1 = 0.0
    best_metrics = {}
    epoch_times = []

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # 训练阶段
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc="训练中"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # 验证阶段
        model.eval()
        val_loss, val_corrects = 0.0, 0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="验证中"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                probs = torch.softmax(outputs, dim=1)[:, 1] # 获取正类的概率
                _, preds = torch.max(outputs, 1)
                
                loss = criterion_val(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # --- 计算所有评估指标 ---
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        # 使用'binary'模式计算二分类指标, zero_division=0 避免分母为0时报错
        val_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0 # 当验证集中只有一个类别时无法计算AUC
            print("警告: 验证数据中仅存在一个类别，AUC无法计算。")

        print(f"验证 Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} F1: {val_f1:.4f} Precision: {val_precision:.4f} Recall: {val_recall:.4f} AUC: {val_auc:.4f}")

        # 根据 F1 分数来记录最佳指标
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                "Best Epoch": epoch + 1,
                "Best Val Acc": f"{val_epoch_acc.item():.4f}",
                "Best Val F1": f"{val_f1:.4f}",
                "Best Val Precision": f"{val_precision:.4f}",
                "Best Val Recall": f"{val_recall:.4f}",
                "Best Val AUC": f"{val_auc:.4f}"
            }

        epoch_times.append(time.time() - epoch_start_time)

        # Early stopping 依然基于验证集损失
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("触发 Early Stopping")
            break

    # --- 训练结束，计算并返回结果 ---
    total_time_min = (time.time() - start_time) / 60
    avg_epoch_time_sec = np.mean(epoch_times) if epoch_times else 0

    results = {
        "Dataset": os.path.basename(data_dir),
        "Total Images": len(full_dataset),
        "Total Time (min)": f"{total_time_min:.2f}",
        "Avg Epoch Time (sec)": f"{avg_epoch_time_sec:.2f}",
        "Model Path": MODEL_SAVE_PATH
    }
    results.update(best_metrics) # 将最佳指标合并到结果中
    
    return results

# --- 4. 主程序入口 ---
if __name__ == '__main__':
    
    # ======================================================================
    # --- !! 在这里设置你的数据集路径 !! ---
    DATASET_TO_TRAIN = r"D:\MATH663_Project\manifest-1616439774456\compressed_datasets\compressed_0.001_percent" 
    # ======================================================================
        
    training_results = run_training(DATASET_TO_TRAIN)

    if training_results:
        print("\n" + "="*50)
        print("                  训练结果总结")
        print("="*50)
        
        # 重新排序以获得更好的可读性
        field_order = [
            "Dataset", "Total Images", "Best Epoch", "Best Val F1", "Best Val Acc", 
            "Best Val Precision", "Best Val Recall", "Best Val AUC", 
            "Total Time (min)", "Avg Epoch Time (sec)", "Model Path"
        ]
        
        # 确保所有可能的键都在里面
        all_keys = list(training_results.keys())
        final_order = [f for f in field_order if f in all_keys] + [f for f in all_keys if f not in field_order]

        for key in final_order:
            if key in training_results:
                print(f"{key+':':<25} {training_results[key]}")
        print("="*50)

        summary_file = 'training_results.csv'
        file_exists = os.path.isfile(summary_file)
        
        # 读取现有header，以支持添加新列
        try:
            with open(summary_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
        except (FileNotFoundError, StopIteration):
            header = []
            
        # 合并新旧header
        final_header = header + [h for h in final_order if h not in header]

        # 写入文件
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_header)
            if not file_exists or not header: # 如果文件不存在或为空
                writer.writeheader()
            writer.writerow(training_results)
        
        print(f"\n结果已追加到文件: {summary_file}")

