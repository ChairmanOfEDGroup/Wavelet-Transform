import os
import sys
import argparse
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 辅助函数与类 (从训练脚本中提取，并简化) ---

def crop_to_roi(image):
    """将一个PIL图像裁剪到其非黑色内容的边界框。"""
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

class EvaluationDataset(Dataset):
    """一个专门用于评估的简化版数据集加载器。"""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.file_paths = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.dcm')
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(valid_exts):
                    self.file_paths.append((os.path.join(cls_dir, file_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx]
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
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model(model_path, test_dir):
    """主评估函数"""
    # --- 配置 ---
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 '{model_path}'")
        return
    if not os.path.exists(test_dir):
        print(f"错误: 测试集目录不存在 '{test_dir}'")
        return

    BATCH_SIZE = 16
    NUM_WORKERS = min(os.cpu_count(), 8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"将在设备 '{device}' 上运行评估...")

    # --- 数据加载 ---
    # 使用与训练时完全相同的验证/测试集转换流程
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = EvaluationDataset(root_dir=test_dir, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    if not test_dataset.file_paths:
        print(f"错误: 在目录 '{test_dir}' 中没有找到任何有效的图像文件。")
        return

    # --- 加载模型 ---
    num_classes = len(test_dataset.classes)
    print(f"检测到 {num_classes} 个类别: {test_dataset.classes}")
    
    model = models.resnet50() # 假设模型架构是ResNet34
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- 评估循环 ---
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="正在评估测试集"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            # 确保概率是针对正类 (class 1)
            if outputs.shape[1] > 1:
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            else: # 处理单输出模型
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())

    # --- 计算指标 ---
    if len(np.unique(all_labels)) < 2:
        print("\n警告: 测试集中只包含一个类别，无法计算AUC等指标。")
        auc = f1 = precision = recall = float('nan')
    else:
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

    accuracy = accuracy_score(all_labels, all_preds)
    
    # --- 打印结果 ---
    print("\n" + "="*50)
    print("           模型性能评估结果")
    print("="*50)
    print(f"{'模型文件:':<20} {os.path.basename(model_path)}")
    print(f"{'测试数据集:':<20} {os.path.basename(test_dir)}")
    print(f"{'总测试样本数:':<20} {len(test_dataset)}")
    print("-"*50)
    print(f"{'AUC (Area Under Curve):':<25} {auc:.4f}")
    print(f"{'F1 分数 (F1 Score):':<25} {f1:.4f}")
    print(f"{'准确率 (Accuracy):':<25} {accuracy:.4f}")
    print(f"{'精确率 (Precision):':<25} {precision:.4f}")
    print(f"{'召回率 (Recall):':<25} {recall:.4f}")
    print("="*50)

if __name__ == '__main__':
    # --- !! 在这里设置你要评估的模型和数据集路径 !! ---
    # 使用 r"..." 格式可以避免路径中的反斜杠问题
    MODEL_PATH_TO_EVALUATE = r"D:\MATH663_Project\best_model_Ablation_All_Three_at_75p.pt"
    TEST_DIR_TO_EVALUATE = r"D:\MATH663_Project\data_split\compressed_100_percent\test"
    # --- !! ------------------------------------ !! ---
    
    evaluate_model(MODEL_PATH_TO_EVALUATE, TEST_DIR_TO_EVALUATE)

