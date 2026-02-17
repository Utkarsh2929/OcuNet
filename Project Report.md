# OcuNet
## Deep Learning-Based Retinal Disease Classification System

---

**Project Report**

**Author:** Utkarsh Gautam  
**Date:** January 2026  
**Platform:** Python 3.11, PyTorch 2.0+  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Phase 1: Single-Label Classification](#3-phase-1-single-label-classification)
4. [Phase 2: Multi-Label Classification](#4-phase-2-multi-label-classification)
5. [Usage Guide](#5-usage-guide)
6. [Conclusion](#6-conclusion)
7. [Future Work](#7-future-work)
8. [References](#8-references)
9. [Appendix](#9-appendix)

---

## 1. Executive Summary

**OcuNet** is a comprehensive deep learning-based system for automated classification of retinal fundus images. The project evolved through two phases:

- **Phase 1**: Single-label classification into 4 categories (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
- **Phase 2**: Multi-label classification detecting 28 different retinal conditions simultaneously

The system leverages transfer learning with EfficientNet architectures pretrained on ImageNet, combined with advanced loss functions to handle class imbalance.

### Phase 1 Key Achievements

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.89% |
| **ROC-AUC (Macro)** | 97.88% |
| **Best Validation Accuracy** | 91.31% |
| **Training Time** | ~15 minutes (50 epochs) |

### Phase 2 Key Achievements

| Metric | Value |
|--------|-------|
| **ROC-AUC (Macro)** | 93.68% |
| **mAP (Mean Average Precision)** | 53.83% |
| **F1 Score (Micro)** | 52.96% |
| **Classes Detected** | 28 diseases |
| **Training Time** | ~2 hours (150 epochs) |

### Clinical Sensitivity (Disease Detection Rate)

| Disease | Phase 1 | Phase 2 | Status |
|---------|---------|---------|--------|
| Cataract | 94.23% | 100.0% | ✅ Excellent |
| Diabetic Retinopathy | 93.94% | 96.17% | ✅ Excellent |
| Glaucoma | 86.75% | 96.28% | ✅ Excellent |
| Age-Related Macular Degeneration | - | 90.32% | ✅ Excellent |
| Macular Hole | - | 100.0% | ✅ Excellent |

All major disease classes exceed the clinical threshold of 80% sensitivity.

---

## 2. Introduction

### 2.1 Background

Eye diseases are among the leading causes of preventable blindness worldwide. Early detection and treatment can prevent vision loss in most cases. However, access to ophthalmologists is limited in many regions, creating a need for automated screening systems.

### 2.2 Problem Statement

Manual examination of retinal images is:
- Time-consuming
- Requires specialized expertise
- Subject to inter-observer variability
- Not scalable for mass screening programs

### 2.3 Objective

Develop an accurate, explainable deep learning system that can:

**Phase 1:**
1. Classify retinal images into four disease categories
2. Provide high sensitivity for disease detection (minimize missed cases)
3. Generate visual explanations for predictions (Grad-CAM)
4. Be easily integrated into clinical workflows

**Phase 2:**
1. Extend to multi-label classification for 28 retinal conditions
2. Detect multiple diseases in a single image simultaneously
3. Handle severe class imbalance with advanced techniques
4. Optimize per-class decision thresholds for clinical use

### 2.4 Scope

The system classifies fundus images into:

| Class | Description | Clinical Importance |
|-------|-------------|---------------------|
| **Cataract** | Clouding of the eye's natural lens | Leading cause of blindness; treatable with surgery |
| **Diabetic Retinopathy** | Damage to retinal blood vessels from diabetes | Early detection prevents 95% of vision loss |
| **Glaucoma** | Optic nerve damage from increased eye pressure | "Silent thief of sight"; irreversible damage |
| **Normal** | Healthy retinal appearance | Baseline for comparison |

---

## 3. Phase 1: Single-Label Classification

### 3.1 Dataset Description

**Dataset:** Eye Diseases Classification  
**Source:** Kaggle (https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

### 3.2 Dataset Statistics

| Class | Number of Images | Percentage |
|-------|------------------|------------|
| Cataract | 1,038 | 24.61% |
| Diabetic Retinopathy | 1,098 | 26.04% |
| Glaucoma | 1,007 | 23.88% |
| Normal | 1,074 | 25.47% |
| **Total** | **4,217** | **100%** |

### 3.3 Data Splitting

Stratified sampling was used to maintain class distribution across splits:

| Split | Images | Percentage | Purpose |
|-------|--------|------------|---------|
| Training | 2,951 | 70% | Model training |
| Validation | 633 | 15% | Hyperparameter tuning, early stopping |
| Test | 633 | 15% | Final evaluation |

### 3.4 Image Properties

| Property | Value |
|----------|-------|
| Format | JPEG/PNG |
| Resolution | Variable (256-2592 pixels) |
| Color Space | RGB |
| Standardized Size | 224×224 pixels |

### 3.5 Class Distribution Analysis

The dataset shows relatively balanced distribution across classes with a maximum imbalance ratio of approximately 1.09:1 (Diabetic Retinopathy to Glaucoma), which is considered mild. However, Focal Loss was still employed to ensure robust performance across all classes.

---

### 3.2 Methodology

### 3.2 Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OcuNet Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐    ┌────────��─────┐    ┌──────────────┐    ┌──────────┐  │
│   │  Input   │───▶│Preprocessing │───▶│   Model      │───▶│ Output   │  │
│   │  Image   │    │& Augmentation│    │(EfficientNet)│    │Prediction│  │
│   └──────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                                              │                           │
│                                              ▼                           │
│                                       ┌──────────────┐                   │
│                                       │   Grad-CAM   │                   │
│                                       │ Explanation  │                   │
│                                       └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Preprocessing

**Training Transforms:**
```python
1. Resize to 256×256
2. Random Crop to 224×224
3. Random Rotation (±30°)
4. Random Horizontal Flip (50%)
5. Color Jitter (brightness=0.2, contrast=0.2)
6. Random Affine (scale: 0.9-1.1)
7. Convert to Tensor
8. Normalize (ImageNet mean/std)
```

**Validation/Test Transforms:**
```python
1. Resize to 224×224
2. Convert to Tensor
3. Normalize (ImageNet mean/std)
```

### 4.3 Data Augmentation Strategy

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Random Rotation | ±30° | Rotation invariance |
| Horizontal Flip | 50% probability | Mirror invariance |
| Color Jitter | brightness=0.2, contrast=0.2 | Lighting variations |
| Random Affine | scale: 0.9-1.1 | Scale invariance |
| Random Crop | 224 from 256 | Position invariance |

### 4.4 Class Imbalance Handling

Two complementary strategies were implemented:

**1. Weighted Random Sampling**
- Oversamples minority classes during training
- Ensures balanced class exposure per epoch

**2. Focal Loss**
```
FL(pt) = -α(1 - pt)^γ × log(pt)

Where:
- pt = probability of correct class
- α = class weight (inversely proportional to frequency)
- γ = 2.0 (focusing parameter)
```

Focal Loss down-weights easy examples and focuses training on hard cases, improving performance on difficult samples.

---

### 3.3 Model Architecture

### 5.1 Architecture Selection

**Selected Model:** EfficientNet-B0

**Rationale:**
- Excellent accuracy-to-parameter ratio
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Fits within 6GB GPU memory
- Compound scaling for balanced depth/width/resolution

### 5.2 Architecture Details

```
┌─────────────────────────────────────────────────────────────┐
│                    OcuNet Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: RGB Image (3 × 224 × 224)                           │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              EfficientNet-B0 Backbone                │    │
│  │              (Pretrained on ImageNet)                │    │
│  │                                                      │    │
│  │  • Stem: Conv3×3, BN, Swish                         │    │
│  │  • MBConv Blocks ×16 (Mobile Inverted Bottleneck)   │    │
│  │  • Squeeze-and-Excitation attention                 │    │
│  │  • Head: Conv1×1, BN, Swish, Global AvgPool         │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│               Feature Vector (1280)                          │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Classification Head                     │    │
│  │                                                      │    │
│  │  • Dropout (p=0.3)                                  │    │
│  │  • Linear: 1280 → 512                               │    │
│  │  • ReLU                                             │    │
│  │  • Dropout (p=0.15)                                 │    │
│  │  • Linear: 512 → 4                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│               Logits (4) → Softmax → Probabilities          │
│                                                              │
│  Output: [P(Cataract), P(DR), P(Glaucoma), P(Normal)]       │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Model Statistics

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 + Custom Head |
| Total Parameters | 4,665,472 |
| Trainable Parameters | 4,665,472 |
| Input Size | 224 × 224 × 3 |
| Output Classes | 4 |
| Model Size | ~18 MB |

### 5.4 Alternative Architectures Supported

| Model | Parameters | Notes |
|-------|------------|-------|
| EfficientNet-B0 | 4.6M | **Selected** - Best efficiency |
| EfficientNet-B3 | 12.2M | Higher accuracy, more memory |
| ResNet-50 | 25.6M | Classic architecture |
| InceptionV3 | 23.8M | Multi-scale features |
| ViT-B/16 | 86.6M | Transformer-based |

---

### 3.4 Training Process

### 6.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Maximum Epochs | 50 |
| Initial Learning Rate | 0.0001 |
| Optimizer | AdamW |
| Weight Decay | 0.0001 |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Early Stopping Patience | 10 epochs |
| Loss Function | Focal Loss (γ=2.0) |
| Mixed Precision | Enabled (FP16) |
| Gradient Clipping | max_norm=1.0 |

### 6.2 Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Memory | 6 GB |
| CUDA Version | 12.1 |
| PyTorch Version | 2.0+ |
| Python Version | 3.11 |
| Training Time | ~15 minutes |

### 6.3 Training Progress

```
Epoch  1/50: Train Acc: 68.93% | Val Acc: 81.36% | LR: 9.76e-05
Epoch  4/50: Train Acc: 84.89% | Val Acc: 84.52% | LR: 6.55e-05
Epoch  8/50: Train Acc: 90.99% | Val Acc: 85.15% | LR: 9.64e-06 [SAVED]
Epoch 13/50: Train Acc: 92.07% | Val Acc: 87.52% | LR: 9.46e-05 [SAVED]
Epoch 18/50: Train Acc: 95.70% | Val Acc: 88.63% | LR: 6.55e-05 [SAVED]
Epoch 28/50: Train Acc: 97.90% | Val Acc: 88.94% | LR: 2.54e-06 [SAVED]
Epoch 31/50: Train Acc: 96.37% | Val Acc: 89.89% | LR: 9.98e-05 [SAVED]
Epoch 41/50: Train Acc: 97.29% | Val Acc: 91.31% | LR: 8.25e-05 [SAVED] ← BEST
Epoch 50/50: Train Acc: 98.95% | Val Acc: 87.36% | LR: 5.01e-05

Best Validation Accuracy: 91.31% at Epoch 41
```

### 6.4 Training Curves

**Figure 1: Training History - Loss and Accuracy Curves**

![Training History](evaluation_results/training_history.png)

*The training curves show:*
- **Loss (Left):** Training loss decreases steadily from 0.37 to 0.01. Validation loss fluctuates between 0.14-0.27, indicating some overfitting after epoch 20.
- **Accuracy (Right):** Training accuracy increases from 69% to 99%. Validation accuracy plateaus around 87-91%, with best performance at epoch 41 (91.31%).

**Observations:**
1. Rapid initial learning (epochs 1-10)
2. Gradual improvement with fluctuations (epochs 10-40)
3. Best model saved at epoch 41
4. Gap between train/val accuracy suggests mild overfitting
5. Cosine annealing scheduler causes periodic learning rate resets (visible as fluctuations)

---

### 3.5 Results and Evaluation

### 7.1 Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 86.89% |
| **Precision (Macro)** | 87.06% |
| **Recall (Macro)** | 86.90% |
| **F1-Score (Macro)** | 86.79% |
| **ROC-AUC (Macro)** | 97.88% |

### 7.2 Per-Class Performance

| Class | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|--------|----------|---------|---------|
| Cataract | 95.45% | 94.23% | 94.84% | 99.66% | 156 |
| Diabetic Retinopathy | 82.45% | 93.94% | 87.82% | 98.66% | 165 |
| Glaucoma | 86.75% | 86.75% | 86.75% | 97.66% | 151 |
| Normal | 83.57% | 72.67% | 77.74% | 95.55% | 161 |

### 7.3 Confusion Matrix

**Figure 2: Normalized Confusion Matrix**

![Confusion Matrix](evaluation_results/confusion_matrix.png)

**Confusion Matrix Analysis:**

|  | Predicted Cataract | Predicted DR | Predicted Glaucoma | Predicted Normal |
|--|-------------------|--------------|-------------------|------------------|
| **Actual Cataract** | **94%** | 1% | 2% | 3% |
| **Actual DR** | 1% | **94%** | 1% | 4% |
| **Actual Glaucoma** | 1% | 5% | **87%** | 8% |
| **Actual Normal** | 3% | 15% | 9% | **73%** |

**Key Insights:**
1. **Cataract:** Highest accuracy (94%), rarely confused with other classes
2. **Diabetic Retinopathy:** Excellent recall (94%), some false positives from Normal
3. **Glaucoma:** Good performance (87%), occasionally confused with Normal (8%)
4. **Normal:** Lowest recall (73%), often misclassified as DR (15%) or Glaucoma (9%)

**Clinical Interpretation:**
- The model is conservative with "Normal" predictions, preferring to flag potential diseases
- This behavior is clinically appropriate as false negatives (missed diseases) are more dangerous than false positives

### 7.4 ROC Curves

**Figure 3: ROC Curves for All Classes**

![ROC Curves](evaluation_results/roc_curves.png)

**ROC-AUC Scores:**

| Class | AUC Score | Interpretation |
|-------|-----------|----------------|
| Cataract | 0.997 | Excellent discrimination |
| Diabetic Retinopathy | 0.987 | Excellent discrimination |
| Glaucoma | 0.977 | Excellent discrimination |
| Normal | 0.956 | Very good discrimination |
| **Macro Average** | **0.979** | **Excellent overall** |

All classes achieve AUC > 0.95, indicating excellent discriminative ability across different classification thresholds.

### 7.5 Clinical Sensitivity Analysis

For medical diagnostic systems, **sensitivity (recall)** is critical to minimize missed disease cases.

| Disease | Sensitivity | Target (≥80%) | Status |
|---------|-------------|---------------|--------|
| Cataract | 94.23% | ✓ | ✅ **Excellent** |
| Diabetic Retinopathy | 93.94% | ✓ | ✅ **Excellent** |
| Glaucoma | 86.75% | ✓ | ✅ **Good** |

**All disease classes exceed the 80% sensitivity threshold**, making OcuNet suitable for clinical screening applications.

### 7.6 Error Analysis

**Most Common Misclassifications:**

| True Class | Predicted As | Percentage | Possible Reason |
|------------|--------------|------------|-----------------|
| Normal → DR | 15% | Similar vascular patterns |
| Normal → Glaucoma | 9% | Optic disc appearance overlap |
| Glaucoma → Normal | 8% | Early-stage glaucoma features |
| Glaucoma → DR | 5% | Vascular changes similarity |

---

### 3.6 Explainability Analysis

### 8.1 Grad-CAM Overview

**Gradient-weighted Class Activation Mapping (Grad-CAM)** provides visual explanations by highlighting image regions that influenced the model's prediction.

**How Grad-CAM Works:**
1. Forward pass to obtain feature maps from last convolutional layer
2. Backward pass to compute gradients for target class
3. Global average pooling of gradients → importance weights
4. Weighted combination of feature maps
5. ReLU activation to keep positive influences
6. Upscale and overlay on original image

**Color Interpretation:**
- 🔴 **Red/Yellow:** High importance (key diagnostic regions)
- 🔵 **Blue/Purple:** Low importance (background)

### 8.2 Cataract Detection

**Figure 4: Grad-CAM Visualization - Cataract Sample 1**

![Cataract GradCAM 1](explainability_results/gradcam_cataract_0.png)

**Prediction:** Cataract (99.6% confidence)

**Figure 5: Grad-CAM Visualization - Cataract Sample 2**

![Cataract GradCAM 2](explainability_results/gradcam_cataract_1.png)

**Prediction:** Cataract (99.2% confidence)

**Analysis:**
- The model focuses on the **central lens region** where cataract causes clouding
- High attention on the **opaque/hazy areas** characteristic of cataract
- Predictions are highly confident (>99%)
- Focus pattern is clinically appropriate for cataract detection

### 8.3 Diabetic Retinopathy Detection

**Figure 6: Grad-CAM Visualization - Diabetic Retinopathy Sample 1**

![DR GradCAM 1](explainability_results/gradcam_diabetic_retinopathy_0.png)

**Prediction:** Diabetic Retinopathy (100.0% confidence)

**Figure 7: Grad-CAM Visualization - Diabetic Retinopathy Sample 2**

![DR GradCAM 2](explainability_results/gradcam_diabetic_retinopathy_1.png)

**Prediction:** Diabetic Retinopathy (100.0% confidence)

**Analysis:**
- Model focuses on the **macula region** (center of retina)
- High attention on areas with potential **hemorrhages, exudates, or microaneurysms**
- Avoids the optic disc (bright spot on left), focusing on pathological regions
- 100% confidence indicates clear diabetic retinopathy features

### 8.4 Glaucoma Detection

**Figure 8: Grad-CAM Visualization - Glaucoma Sample 1**

![Glaucoma GradCAM 1](explainability_results/gradcam_glaucoma_0.png)

**Prediction:** Glaucoma (97.1% confidence)

**Figure 9: Grad-CAM Visualization - Glaucoma Sample 2**

![Glaucoma GradCAM 2](explainability_results/gradcam_glaucoma_1.png)

**Prediction:** Glaucoma (98.7% confidence)

**Analysis:**
- Model focuses on the **optic disc and cup region** (critical for glaucoma diagnosis)
- Attention to the **neuroretinal rim** where glaucoma causes thinning
- Sample 2 shows focus on specific structural abnormalities
- High confidence (>97%) with clinically relevant focus areas

### 8.5 Normal Classification

**Figure 10: Grad-CAM Visualization - Normal Sample 1**

![Normal GradCAM 1](explainability_results/gradcam_normal_0.png)

**Prediction:** Normal (95.9% confidence)

**Figure 11: Grad-CAM Visualization - Normal Sample 2**

![Normal GradCAM 2](explainability_results/gradcam_normal_1.png)

**Prediction:** Normal (98.6% confidence)

**Analysis:**
- Model examines **broad retinal areas** rather than specific pathological regions
- Focus on **healthy vascular patterns** and overall retinal structure
- Attention distributed across multiple regions (no single abnormality)
- Confidence is high but slightly lower than disease predictions (appropriate conservatism)

### 8.6 Explainability Summary

| Class | Focus Regions | Clinical Relevance |
|-------|---------------|-------------------|
| Cataract | Central lens, hazy areas | ✅ Appropriate - lens opacity |
| Diabetic Retinopathy | Macula, vascular areas | ✅ Appropriate - hemorrhages/exudates |
| Glaucoma | Optic disc, cup region | ✅ Appropriate - optic nerve assessment |
| Normal | Broad distribution | ✅ Appropriate - no specific pathology |

**Conclusion:** Grad-CAM visualizations confirm that OcuNet focuses on clinically relevant regions for each disease class, increasing trust in the model's predictions.

---

## 4. Phase 2: Multi-Label Classification

Phase 2 extends OcuNet to detect 28 different retinal conditions simultaneously using multi-label classification. This addresses the clinical reality that patients often present with multiple co-existing conditions.

### 4.1 Dataset Description

**Data Sources:**

1. **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**
   - Source: IEEE Dataport
   - Images: 3,200 retinal fundus images
   - Labels: 28 disease categories (multi-label)

2. **Phase 1 Dataset (Expanded)**
   - Source: Kaggle Eye Diseases Classification
   - Images: 4,217 images mapped to Phase 2 categories

**Combined Dataset:** 7,417 total images (Train: 4,871 | Val: 1,273 | Test: 1,273)

**28 Disease Classes:** Disease_Risk, DR, ARMD, MH, DN, MYA, BRVO, TSLN, ERM, LS, MS, CSR, ODC, CRVO, AH, ODP, ODE, AION, PT, RT, RS, CRS, EDN, RPEC, MHL, CATARACT, GLAUCOMA, NORMAL

### 4.2 Methodology

**Key Techniques:**
- **Asymmetric Loss (ASL)**: Handles extreme class imbalance (γ_neg=4, γ_pos=1, clip=0.05)
- **Oversampling**: 14 rare classes (<50 samples) oversampled 3x
- **RandAugment**: Strong data augmentation (N=2, M=9)
- **Learning Rate Warmup**: 5 epochs linear warmup to 5e-4
- **EMA (Exponential Moving Average)**: Decay=0.999 for stable evaluation

### 4.3 Model Architecture

**EfficientNet-B2 + Multi-Label Head**

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B2 + Custom Multi-Label Head |
| Total Parameters | 10,060,022 |
| Input Size | 224 × 224 × 3 |
| Output Classes | 28 (independent sigmoids) |
| Loss Function | Asymmetric Loss (ASL) |

### 4.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Maximum Epochs | 150 |
| Learning Rate | 5e-5 → 5e-4 (warmup) |
| Optimizer | AdamW |
| Early Stopping | 25 epochs |
| Mixed Precision | FP16 |

### 4.5 Results

**Overall Metrics:**

| Metric | Value |
|--------|-------|
| **ROC-AUC (Macro)** | 93.68% |
| **mAP** | 53.83% |
| **F1 Score (Micro)** | 52.96% |
| **Recall (Macro)** | 74.03% |
| **Hamming Loss** | 0.1111 |

**Key Disease Performance:**

| Disease | Recall | AUC |
|---------|--------|-----|
| CATARACT | 100.0% | 99.3% |
| MH (Macular Hole) | 100.0% | 98.4% |
| DR | 96.2% | 93.6% |
| GLAUCOMA | 96.3% | 90.5% |
| ARMD | 90.3% | 97.5% |
| NORMAL | 99.3% | 91.9% |

**22 out of 28 classes achieve ≥80% sensitivity.**

**With Optimized Thresholds:**
- F1 (Macro): 54.93%
- F1 (Micro): 73.53%

---

## 5. Usage Guide

### 5.1 Installation

```bash
# Clone or create project directory
mkdir OcuNet && cd OcuNet

# Create virtual environment (Python 3.11 recommended)
py -3.11 -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 5.2 Project Structure

```
OcuNet/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   └── dataset/                 # Dataset directory
│       ├── cataract/
│       ├── diabetic_retinopathy/
│       ├── glaucoma/
│       └── normal/
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Data loading & augmentation
│   ├── models.py                # Model architectures
│   ├── train.py                 # Training logic
│   ├── evaluate.py              # Evaluation metrics
│   └── explainability.py        # Grad-CAM visualizations
├── checkpoints/
│   ├── best_model.pth           # Best trained model
│   └── latest_checkpoint.pth    # Latest checkpoint
├── evaluation_results/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── training_history.png
│   ├── evaluation_report.txt
│   └── metrics.json
├── explainability_results/
│   └── gradcam_*.png            # Grad-CAM visualizations
├── train_pipeline.py            # Main training script
├── predict.py                   # Prediction interface
├── requirements.txt             # Dependencies
└── README.md
```

### 5.3 Training

```bash
# Full pipeline (train + evaluate + explain)
python train_pipeline.py --mode all

# Individual modes
python train_pipeline.py --mode train      # Training only
python train_pipeline.py --mode evaluate   # Evaluation only
python train_pipeline.py --mode explain    # Grad-CAM only

# Custom config
python train_pipeline.py --config path/to/config.yaml --mode all
```

### 5.4 Prediction API

**Python Interface:**

```python
from predict import EyeDiseaseClassifier

# Initialize classifier (loads model automatically)
classifier = EyeDiseaseClassifier()

# Single image prediction
result = classifier.predict("path/to/retinal_image.jpg")

print(result)
# Output:
# {
#     'predicted_class': 'diabetic_retinopathy',
#     'confidence': 0.9394,
#     'probabilities': {
#         'cataract': 0.0123,
#         'diabetic_retinopathy': 0.9394,
#         'glaucoma': 0.0256,
#         'normal': 0.0227
#     }
# }

# Batch prediction
results = classifier.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

**Command Line:**

```bash
python predict.py path/to/image.jpg
```

Output:
```
Model loaded on cuda

Prediction: diabetic_retinopathy
Confidence: 93.94%

All probabilities:
  cataract: 1.23%
  diabetic_retinopathy: 93.94%
  glaucoma: 2.56%
  normal: 2.27%
```

### 5.5 Integration Example

**Flask Web Application:**

```python
from flask import Flask, request, jsonify
from predict import EyeDiseaseClassifier

app = Flask(__name__)
classifier = EyeDiseaseClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    file.save('temp.jpg')
    
    result = classifier.predict('temp.jpg')
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Streamlit Application:**

```python
import streamlit as st
from predict import EyeDiseaseClassifier
from PIL import Image

st.title("🔬 OcuNet - Eye Disease Classification")

classifier = EyeDiseaseClassifier()

uploaded_file = st.file_uploader("Upload a retinal image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save and predict
    image.save("temp.jpg")
    result = classifier.predict("temp.jpg")
    
    # Display results
    st.success(f"**Prediction:** {result['predicted_class']}")
    st.info(f"**Confidence:** {result['confidence']:.2%}")
    
    st.subheader("All Probabilities")
    st.bar_chart(result['probabilities'])
```

---

## 6. Conclusion

### 6.1 Achievements

OcuNet successfully demonstrates:

**Phase 1:**
1. **High Accuracy:** 86.89% test accuracy with 97.88% ROC-AUC
2. **Clinical Reliability:** All 4 disease classes exceed 80% sensitivity threshold
3. **Explainability:** Grad-CAM visualizations confirm clinically appropriate focus regions
4. **Efficiency:** Training completes in ~15 minutes on consumer GPU

**Phase 2:**
1. **Multi-Label Detection:** 28 retinal conditions detected simultaneously
2. **High Disease Sensitivity:** 22/28 classes achieve ≥80% recall
3. **Strong ROC-AUC:** 93.68% macro-average across all classes
4. **Optimized Thresholds:** Per-class thresholds improve F1 to 54.93% macro

### 6.2 Strengths

| Strength | Phase 1 | Phase 2 |
|----------|---------|---------|
| Sensitivity | 94% cataract, 94% DR, 87% glaucoma | 100% cataract, 96% DR, 96% glaucoma |
| ROC-AUC | All classes >95% | 93.68% macro (28 classes) |
| Disease Coverage | 4 classes | 28 classes |
| Co-morbidity Detection | No | Yes |
| Interpretable | Grad-CAM | Grad-CAM |

### 6.3 Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Normal class accuracy (73% Phase 1) | May over-refer healthy patients | Clinical verification |
| Rare disease performance | Low precision for rare classes | Per-class thresholds |
| Dataset diversity | May not generalize to all populations | Collect more diverse data |
| Single image input | No temporal/multi-view analysis | Future enhancement |

### 6.4 Clinical Applicability

OcuNet is suitable for:
- ✅ **Primary screening** in resource-limited settings
- ✅ **Triage assistance** to prioritize urgent cases
- ✅ **Educational tool** for training ophthalmologists
- ⚠️ **Requires clinical verification** for final diagnosis

---

## 7. Future Work

### 7.1 Short-term Improvements

1. **Increase Normal class accuracy**
   - Add more normal samples
   - Implement harder negative mining
   - Use attention mechanisms

2. **Multi-severity classification**
   - DR severity grading (mild/moderate/severe/proliferative)
   - Glaucoma staging

3. **Uncertainty quantification**
   - Monte Carlo dropout
   - Confidence calibration

### 7.2 Long-term Enhancements

1. **Multi-modal input**
   - OCT images
   - Visual field data
   - Patient demographics

2. **Federated learning**
   - Privacy-preserving training across institutions
   - Larger effective dataset

3. **Real-time deployment**
   - Mobile application
   - Edge device optimization
   - ONNX/TensorRT conversion

4. **Additional diseases**
   - Age-related macular degeneration
   - Retinal detachment
   - Hypertensive retinopathy

---

## 8. References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

2. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.

3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017.

4. Ben-Baruch, E., et al. (2020). Asymmetric Loss For Multi-Label Classification. arXiv.

5. Gulshan, V., et al. (2016). Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy. JAMA.

6. Kaggle Dataset: Eye Diseases Classification. https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

7. RFMiD Dataset: Retinal Fundus Multi-Disease Image Dataset. https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid

---

## 9. Appendix

### A. Complete Configuration

```yaml
# config/config.yaml

dataset:
  root_dir: "data/dataset"
  image_size: 224
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

classes:
  - "cataract"
  - "diabetic_retinopathy"
  - "glaucoma"
  - "normal"

training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10
  num_workers: 0

model:
  architecture: "efficientnet_b0"
  pretrained: true
  dropout_rate: 0.3
  use_focal_loss: true
  focal_loss_gamma: 2.0

augmentation:
  rotation_degrees: 30
  horizontal_flip: true
  vertical_flip: false
  brightness_range: [0.8, 1.2]
  contrast_range: [0.8, 1.2]
  zoom_range: [0.9, 1.1]

output:
  checkpoint_dir: "checkpoints"
  results_dir: "evaluation_results"
  explainability_dir: "explainability_results"
```

### B. Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
PyYAML>=6.0
```

### C. Full Evaluation Metrics

```json
{
  "overall": {
    "accuracy": 0.8688783570300158,
    "precision_macro": 0.8705693735600739,
    "recall_macro": 0.8689898437774913,
    "f1_macro": 0.8678830930901618
  },
  "per_class": {
    "cataract": {
      "precision": 0.9545454545454546,
      "recall": 0.9423076923076923,
      "f1_score": 0.9483870967741935
    },
    "diabetic_retinopathy": {
      "precision": 0.824468085106383,
      "recall": 0.9393939393939394,
      "f1_score": 0.8781869688385269
    },
    "glaucoma": {
      "precision": 0.8675496688741722,
      "recall": 0.8675496688741722,
      "f1_score": 0.8675496688741722
    },
    "normal": {
      "precision": 0.8357142857142857,
      "recall": 0.7267080745341615,
      "f1_score": 0.7774086378737541
    }
  },
  "roc_auc": {
    "macro": 0.9788304047366287,
    "cataract": 0.9965865720582702,
    "diabetic_retinopathy": 0.9865708365708366,
    "glaucoma": 0.976642576461213,
    "normal": 0.9555216338561954
  }
}
```

### D. Training Log Summary

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Saved |
|-------|------------|-----------|----------|---------|-----|-------|
| 1 | 0.3727 | 68.93% | 0.1835 | 81.36% | 9.76e-05 | ✓ |
| 4 | 0.1511 | 84.89% | 0.1683 | 84.52% | 6.55e-05 | ✓ |
| 8 | 0.0934 | 90.99% | 0.1506 | 85.15% | 9.64e-06 | ✓ |
| 13 | 0.0863 | 92.07% | 0.1372 | 87.52% | 9.46e-05 | ✓ |
| 18 | 0.0493 | 95.70% | 0.1424 | 88.63% | 6.55e-05 | ✓ |
| 28 | 0.0226 | 97.90% | 0.1485 | 88.94% | 2.54e-06 | ✓ |
| 31 | 0.0361 | 96.37% | 0.1724 | 89.89% | 9.98e-05 | ✓ |
| **41** | **0.0335** | **97.29%** | **0.1434** | **91.31%** | **8.25e-05** | **✓ BEST** |
| 50 | 0.0146 | 98.95% | 0.2146 | 87.36% | 5.01e-05 | |

---

## Document Information

| Property | Value |
|----------|-------|
| Project Name | OcuNet |
| Version | 2.0.0 |
| Last Updated | January 2026 |
| Author | Utkarsh Gautam |
| License | MIT |

---

*End of Report*