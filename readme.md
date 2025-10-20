# 🧠 Improving Medical Image Classification Robustness using Wavelet Compression Artifacts

> **A Study on Improving Medical Image Classification Model Robustness using Wavelet Compression Artifacts as a Data Augmentation Strategy**

---

## 📘 Overview

In medical image analysis, deep learning models often rely heavily on high-quality data. However, medical images (e.g., DICOM) are usually extremely large, making storage, transmission, and processing challenging.  
This project explores a **novel hypothesis** — that **wavelet compression artifacts**, instead of being harmful, can be leveraged as a **task-relevant form of noise** to enhance model robustness through **artifact-based data augmentation**.

Our goal is to determine whether intentionally injecting compression artifacts during training can lead to models that are **more stable**, **more generalizable**, and **perform better on clean, high-quality images**.

---

## 🎯 Motivation

Traditional wisdom suggests that cleaner data yields better models.  
We challenge this idea:  

> Could compression artifacts — often viewed as imperfections — actually help models learn more essential, generalizable features?

To test this, we propose using **wavelet compression artifacts as a structured data augmentation strategy** to simulate real-world variations in medical imaging.

Additionally, this project addresses the common **class imbalance problem** in medical datasets, using weighted loss functions and robust evaluation metrics.

---

## ⚙️ Methodology

### 1. Baseline and Problem Identification

- Base architecture: **ResNet50**
- Loss: **Weighted Cross-Entropy**
- Core metric: **AUC (Area Under ROC Curve)**, due to dataset imbalance
- Evaluated model performance across multiple compression ratios (100% → 0.00001%)

### 2. Finding the Compression Threshold

We conducted systematic experiments to identify when compression begins to severely degrade model performance.

| Coefficient % (kept) | AUC    | Dataset Size |
| -------------------- | ------ | ------------ |
| 100%                 | 0.6579 | 2.59 GB      |
| 10%                  | 0.6443 | 2.17 GB      |
| 1%                   | 0.6677 | 1.19 GB      |
| 0.1%                 | 0.6093 | 498 MB       |
| 0.01%                | 0.5223 | 323 MB       |

**Finding:**  
Once the coefficient percentage drops below **1%**, critical diagnostic features begin to vanish.

---

## 🧩 Data Preprocessing & Augmentation

- **DICOM Windowing:** Enhanced contrast for tissues, solving normalization-based information loss.  
- **ROI Cropping:** Focused on meaningful regions for more efficient learning.  
- **Standard Augmentations:** Deterministic 90/180/270° rotations and random horizontal flips.  
- **Artifact Injection:** The novel step — injecting compression artifacts as a “robustness vaccine.”

---

## 🧪 Two-Stage Robustness Framework

### **Stage 1: Injection Probability Study**

- Tested artifact injection rates of **0%, 25%, 50%, 75%**
- Used diverse artifact sources: 10%, 1%, 0.1% compressed datasets  
- **Result:** 75% injection probability achieved the best AUC improvement

### **Stage 2: Artifact Type Ablation Study**

- Tested all single and combined artifact types  
- Best performance with **combined artifacts at 75% probability**  
- Improved test AUC by **+0.05** over baseline

---

## 📊 Experimental Setup

- **Data Split:**  
  - 80% Train / 10% Validation / 10% Test  
  - Validation and Test sets always use clean (100%) data
- **Early Stopping** and **Adaptive Learning Rate**
- **Independent evaluation** on unseen test data

---

## 📈 Results

| Model                       | Artifact Injection    | AUC (↑)    |
| --------------------------- | --------------------- | ---------- |
| Baseline (clean only)       | 0%                    | 0.6579     |
| Robust Model (artifact 75%) | Mixed (10%, 1%, 0.1%) | **0.7079** |

> Injecting structured noise improves robustness and generalization.

---

## 🧬 Expected Outcomes

We expect that models trained with **wavelet artifact augmentation** will:

- Exhibit higher robustness to unseen image noise.
- Maintain or improve diagnostic accuracy on clean data.
- Provide a scalable path to model generalization for medical imaging.

---

## 🧰 Tech Stack

- **Framework:** PyTorch / TensorFlow (depending on setup)
- **Model:** ResNet50
- **Data Format:** DICOM → preprocessed to NumPy / PNG
- **Evaluation Metrics:** AUC, ROC Curve
- **Environment:** Python 3.10+, CUDA compatible GPU.

---

## 📂 Repository Structure

```
📦 MedicalImageRobustness
├── data/
│   ├── compressed_100_percent/
│   ├── compressed_10_percent/
│   ├── compressed_1_percent/
│   └── ...
├── src/
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
├── results/
│   ├── auc_plots/
│   ├── compression_visuals/
│   └── logs/
├── requirements.txt
└── README.md
```

---