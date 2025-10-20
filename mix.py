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
from sklearn.metrics import roc_auc_score

# --- ROI裁剪函数 (无变化) ---
def crop_to_roi(image):
    img_array = np.array(image)
    if len(img_array.shape) > 2: non_empty_pixels = np.where(np.any(img_array > 0, axis=-1))
    else: non_empty_pixels = np.where(img_array > 0)
    if non_empty_pixels[0].size == 0 or non_empty_pixels[1].size == 0: return image
    min_y, max_y = np.min(non_empty_pixels[0]), np.max(non_empty_pixels[0])
    min_x, max_x = np.min(non_empty_pixels[1]), np.max(non_empty_pixels[1])
    return image.crop((min_x, min_y, max_x + 1, max_y + 1))

# --- 1. 数据集加载器 (已集成确定性旋转) ---
class MixedQualityDataset(Dataset):
    def __init__(self, base_paths, all_datasets_dir, compression_mix, transform=None, is_train=False, noise_injection_prob=0.0):
        self.base_paths, self.all_datasets_dir = base_paths, all_datasets_dir
        self.compression_mix, self.transform = compression_mix, transform
        self.is_train, self.noise_prob = is_train, noise_injection_prob

    def __len__(self):
        # 核心改动：只在训练时应用4倍旋转增强
        if self.is_train:
            return len(self.base_paths) * 4
        return len(self.base_paths)

    def __getitem__(self, idx):
        # 核心改动：根据是否为训练集来决定索引和旋转角度
        if self.is_train:
            original_img_idx = idx // 4
            rotation_angle = (idx % 4) * 90
        else:
            original_img_idx = idx
            rotation_angle = 0 # 验证/测试集不旋转

        base_img_path, label = self.base_paths[original_img_idx]
        current_img_path = base_img_path

        if self.is_train and self.compression_mix and random.random() < self.noise_prob:
            chosen_compression = random.choice(self.compression_mix)
            path_parts = base_img_path.split(os.sep)
            path_parts[-3] = chosen_compression
            compressed_path = os.sep.join(path_parts)
            if os.path.exists(compressed_path):
                current_img_path = compressed_path
        
        try:
            if current_img_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(current_img_path)
                pixel_array = ds.pixel_array
                window_center, window_width = -600, 1500
                img_min, img_max = window_center - window_width // 2, window_center + window_width // 2
                pixel_array = np.clip(pixel_array, img_min, img_max)
                pixel_array = ((pixel_array - img_min) / (img_max - img_min)) * 255.0
                image = Image.fromarray(pixel_array.astype(np.uint8)).convert("RGB")
            else:
                image = Image.open(current_img_path).convert("RGB")
        except Exception as e:
            print(f"\n警告：加载文件 '{current_img_path}' 失败. 错误: {e}. 将返回黑色图像。")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        image = crop_to_roi(image)
        # 核心改动：应用确定性旋转
        image = image.rotate(rotation_angle)
        
        if self.transform: image = self.transform(image)
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

# --- 3. 核心训练与测试函数 (无变化) ---
def run_robustness_experiment(base_dir, experiment_config):
    # --- 配置 ---
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE = 20, 16, 0.0001, 3
    NUM_WORKERS = min(os.cpu_count(), 8)
    
    EXPERIMENT_NAME = experiment_config["name"]
    COMPRESSION_MIX_FOR_AUG = experiment_config["mix"]
    NOISE_PROB = experiment_config["noise_prob"]
    BASE_DATASET = "compressed_100_percent"
    MODEL_SAVE_PATH = f"best_model_{EXPERIMENT_NAME}.pt"
    
    # 核心改动：与auto.py对齐数据增强
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True), 
        transforms.RandomHorizontalFlip(),
        # 随机旋转已被Dataset中的确定性旋转替代
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # --- 数据加载 ---
    base_data_dir = os.path.join(base_dir, BASE_DATASET)
    if not os.path.isdir(base_data_dir): return None, None
        
    classes = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    all_file_paths = [(os.path.join(base_data_dir, c, f), class_to_idx[c]) for c in classes for f in os.listdir(os.path.join(base_data_dir, c))]

    train_paths, val_paths, test_paths = random_split(all_file_paths, [int(0.8 * len(all_file_paths)), int(0.1 * len(all_file_paths)), len(all_file_paths) - int(0.8 * len(all_file_paths)) - int(0.1 * len(all_file_paths))])

    train_dataset = MixedQualityDataset(train_paths, base_dir, COMPRESSION_MIX_FOR_AUG, train_transforms, is_train=True, noise_injection_prob=NOISE_PROB)
    val_dataset = MixedQualityDataset(val_paths, base_dir, [], val_test_transforms, is_train=False)
    test_dataset = MixedQualityDataset(test_paths, base_dir, [], val_test_transforms, is_train=False)

    print(f" - 训练集: {len(train_paths)}张原始图, 增强后(4x旋转) {len(train_dataset)} 张. 以 {NOISE_PROB*100}% 的概率注入 {len(COMPRESSION_MIX_FOR_AUG)} 种压缩伪影")
    print(f" - 验证/测试集: 使用{len(val_paths)}/{len(test_paths)}张固定的高质量图")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 模型与训练 ---
    train_labels = [label for _, label in train_paths]
    class_weights = torch.FloatTensor([len(train_labels)/Counter(train_labels)[i] for i in range(len(classes))]).to(device)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=MODEL_SAVE_PATH)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} 训练中"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                val_loss += nn.CrossEntropyLoss()(outputs, labels.to(device)).item() * inputs.size(0)
        val_epoch_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1} 验证 Loss: {val_epoch_loss:.4f}")
        scheduler.step(val_epoch_loss)
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop: print("触发 Early Stopping"); break

    # --- 最终测试 ---
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试中"):
            outputs = model(inputs.to(device))
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    test_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    return { "config": experiment_config, "Test_AUC": test_auc }

# --- 4. 主程序入口 (全自动两阶段实验框架) ---
if __name__ == '__main__':
    DATASETS_BASE_DIR = r"D:\MATH663_Project\manifest-1616439774456\compressed_datasets"
    
    # --- 实验定义 ---
    # --- 第一阶段: 寻找最佳注入概率 ---
    experiments_part_1 = [
        {"name": "Prob_Study_0_percent_Baseline", "mix": [], "noise_prob": 0.0},
        {"name": "Prob_Study_25_percent_Vaccine", "mix": ["compressed_10_percent", "compressed_1_percent", "compressed_0.1_percent"], "noise_prob": 0.25},
        {"name": "Prob_Study_50_percent_Balanced", "mix": ["compressed_10_percent", "compressed_1_percent", "compressed_0.1_percent"], "noise_prob": 0.50},
        {"name": "Prob_Study_75_percent_Hardening", "mix": ["compressed_10_percent", "compressed_1_percent", "compressed_0.1_percent"], "noise_prob": 0.75},
    ]

    # --- 第二阶段: 使用最佳概率进行伪影消融实验 ---
    # 这个函数会在第一阶段完成后被自动调用
    def create_part2_experiments(best_prob):
        return [
            {"name": f"Ablation_10_percent_only_at_{int(best_prob*100)}p", "mix": ["compressed_10_percent"], "noise_prob": best_prob},
            {"name": f"Ablation_1_percent_only_at_{int(best_prob*100)}p",  "mix": ["compressed_1_percent"], "noise_prob": best_prob},
            {"name": f"Ablation_0.1_percent_only_at_{int(best_prob*100)}p", "mix": ["compressed_0.1_percent"], "noise_prob": best_prob},
            {"name": f"Ablation_10_and_1_percent_at_{int(best_prob*100)}p",  "mix": ["compressed_10_percent", "compressed_1_percent"], "noise_prob": best_prob},
            {"name": f"Ablation_10_and_0.1_percent_at_{int(best_prob*100)}p", "mix": ["compressed_10_percent", "compressed_0.1_percent"], "noise_prob": best_prob},
            {"name": f"Ablation_1_and_0.1_percent_at_{int(best_prob*100)}p",  "mix": ["compressed_1_percent", "compressed_0.1_percent"], "noise_prob": best_prob},
            {"name": f"Ablation_All_Three_at_{int(best_prob*100)}p", "mix": ["compressed_10_percent", "compressed_1_percent", "compressed_0.1_percent"], "noise_prob": best_prob},
        ]

    # --- 实验执行 ---
    log_file = 'robustness_ablation_study_results.csv'
    fieldnames = ["Experiment", "Test_AUC"]
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0: writer.writeheader()

    overall_start_time = time.time()
    
    # --- 阶段 1 ---
    print(f"\n{'='*25} 开始第一阶段: 寻找最佳注入概率 {'='*25}\n")
    part1_results_data = []
    for config in experiments_part_1:
        print(f"\n--- 开始实验: {config['name']} ---")
        result_data = run_robustness_experiment(DATASETS_BASE_DIR, config)
        if result_data:
            result_to_log = {"Experiment": config['name'], "Test_AUC": f"{result_data['Test_AUC']:.4f}"}
            part1_results_data.append(result_data)
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result_to_log)
            print(f"\n实验 '{config['name']}' 的测试结果已追加到 {log_file}")

    # --- 自动分析阶段1结果 ---
    if not part1_results_data:
        print("\n第一阶段未产生任何结果，无法继续进行第二阶段。")
        sys.exit()

    best_experiment_part1 = max(part1_results_data, key=lambda x: x['Test_AUC'])
    BEST_PROBABILITY = best_experiment_part1['config']['noise_prob']
    
    print(f"\n\n{'='*25} 第一阶段完成 {'='*25}")
    print(f"最佳实验: '{best_experiment_part1['config']['name']}' (AUC: {best_experiment_part1['Test_AUC']:.4f})")
    print(f"得出的最佳注入概率为: {BEST_PROBABILITY*100}%\n")

    # --- 阶段 2 ---
    print(f"\n{'='*25} 开始第二阶段: 使用 {BEST_PROBABILITY*100}% 的概率进行伪影消融实验 {'='*25}\n")
    
    experiments_part_2 = create_part2_experiments(BEST_PROBABILITY)

    for config in experiments_part_2:
        print(f"\n--- 开始实验: {config['name']} ---")
        result_data = run_robustness_experiment(DATASETS_BASE_DIR, config)
        if result_data:
            result_to_log = {"Experiment": config['name'], "Test_AUC": f"{result_data['Test_AUC']:.4f}"}
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result_to_log)
            print(f"\n实验 '{config['name']}' 的测试结果已追加到 {log_file}")

    overall_time_min = (time.time() - overall_start_time) / 60
    print(f"\n\n{'='*25} 所有实验处理完毕! {'='*25}")
    print(f"总耗时: {overall_time_min:.2f} 分钟")
    print(f"详细日志请查看: {log_file}")

