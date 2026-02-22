# OcuNet
## Deep Learning-Based Retinal Disease Classification System

---

**Project Report**

**Author:** Utkarsh Gautam  
**Date:** February 2026  
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

The system leverages transfer learning with EfficientNet architectures pretrained on ImageNet, combined with advanced loss functions, probability calibration, and intelligent preprocessing to handle class imbalance and maximize precision.

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
| **F1 Score (Macro, Optimized)** | 54.93% |
| **F1 Score (Micro, Optimized)** | 73.53% |
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

#### Dataset Statistics

| Class | Number of Images | Percentage |
|-------|------------------|------------|
| Cataract | 1,038 | 24.61% |
| Diabetic Retinopathy | 1,098 | 26.04% |
| Glaucoma | 1,007 | 23.88% |
| Normal | 1,074 | 25.47% |
| **Total** | **4,217** | **100%** |

#### Data Splitting

Stratified sampling was used to maintain class distribution across splits:

| Split | Images | Percentage | Purpose |
|-------|--------|------------|---------|
| Training | 2,951 | 70% | Model training |
| Validation | 633 | 15% | Hyperparameter tuning, early stopping |
| Test | 633 | 15% | Final evaluation |

#### Image Properties

| Property | Value |
|----------|-------|
| Format | JPEG/PNG |
| Resolution | Variable (256-2592 pixels) |
| Color Space | RGB |
| Standardized Size | 224×224 pixels |

#### Class Distribution Analysis

The dataset shows relatively balanced distribution across classes with a maximum imbalance ratio of approximately 1.09:1 (Diabetic Retinopathy to Glaucoma), which is considered mild. However, Focal Loss was still employed to ensure robust performance across all classes.

---

### 3.2 Methodology

#### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OcuNet Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
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

#### Data Preprocessing

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

#### Data Augmentation Strategy

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Random Rotation | ±30° | Rotation invariance |
| Horizontal Flip | 50% probability | Mirror invariance |
| Color Jitter | brightness=0.2, contrast=0.2 | Lighting variations |
| Random Affine | scale: 0.9-1.1 | Scale invariance |
| Random Crop | 224 from 256 | Position invariance |

#### Class Imbalance Handling

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

#### Architecture Selection

**Selected Model:** EfficientNet-B0

**Rationale:**
- Excellent accuracy-to-parameter ratio
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Fits within 6GB GPU memory
- Compound scaling for balanced depth/width/resolution

#### Architecture Details

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

#### Model Statistics

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 + Custom Head |
| Total Parameters | 4,665,472 |
| Trainable Parameters | 4,665,472 |
| Input Size | 224 × 224 × 3 |
| Output Classes | 4 |
| Model Size | ~18 MB |

---

### 3.4 Training Process

#### Training Configuration

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

#### Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Memory | 6 GB |
| CUDA Version | 12.1 |
| PyTorch Version | 2.0+ |
| Python Version | 3.11 |
| Training Time | ~15 minutes |

#### Training Progress

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

**Observations:**
1. Rapid initial learning (epochs 1-10)
2. Gradual improvement with fluctuations (epochs 10-40)
3. Best model saved at epoch 41
4. Gap between train/val accuracy suggests mild overfitting
5. Cosine annealing scheduler causes periodic learning rate resets (visible as fluctuations)

---

### 3.5 Results and Evaluation

#### Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 86.89% |
| **Precision (Macro)** | 87.06% |
| **Recall (Macro)** | 86.90% |
| **F1-Score (Macro)** | 86.79% |
| **ROC-AUC (Macro)** | 97.88% |

#### Per-Class Performance

| Class | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|--------|----------|---------|---------|
| Cataract | 95.45% | 94.23% | 94.84% | 99.66% | 156 |
| Diabetic Retinopathy | 82.45% | 93.94% | 87.82% | 98.66% | 165 |
| Glaucoma | 86.75% | 86.75% | 86.75% | 97.66% | 151 |
| Normal | 83.57% | 72.67% | 77.74% | 95.55% | 161 |

#### Confusion Matrix Analysis

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

#### ROC-AUC Scores

| Class | AUC Score | Interpretation |
|-------|-----------|----------------|
| Cataract | 0.997 | Excellent discrimination |
| Diabetic Retinopathy | 0.987 | Excellent discrimination |
| Glaucoma | 0.977 | Excellent discrimination |
| Normal | 0.956 | Very good discrimination |
| **Macro Average** | **0.979** | **Excellent overall** |

All classes achieve AUC > 0.95, indicating excellent discriminative ability across different classification thresholds.

#### Clinical Sensitivity Analysis

For medical diagnostic systems, **sensitivity (recall)** is critical to minimize missed disease cases.

| Disease | Sensitivity | Target (≥80%) | Status |
|---------|-------------|---------------|--------|
| Cataract | 94.23% | ✓ | ✅ **Excellent** |
| Diabetic Retinopathy | 93.94% | ✓ | ✅ **Excellent** |
| Glaucoma | 86.75% | ✓ | ✅ **Good** |

**All disease classes exceed the 80% sensitivity threshold**, making OcuNet suitable for clinical screening applications.

#### Error Analysis

**Most Common Misclassifications:**

| True Class | Predicted As | Percentage | Possible Reason |
|------------|--------------|------------|-----------------|
| Normal → DR | 15% | Similar vascular patterns |
| Normal → Glaucoma | 9% | Optic disc appearance overlap |
| Glaucoma → Normal | 8% | Early-stage glaucoma features |
| Glaucoma → DR | 5% | Vascular changes similarity |

---

### 3.6 Explainability Analysis

#### Grad-CAM Overview

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

#### Cataract Detection

**Prediction:** Cataract (99.6% confidence)

**Analysis:**
- The model focuses on the **central lens region** where cataract causes clouding
- High attention on the **opaque/hazy areas** characteristic of cataract
- Predictions are highly confident (>99%)
- Focus pattern is clinically appropriate for cataract detection

#### Diabetic Retinopathy Detection

**Prediction:** Diabetic Retinopathy (100.0% confidence)

**Analysis:**
- Model focuses on the **macula region** (center of retina)
- High attention on areas with potential **hemorrhages, exudates, or microaneurysms**
- Avoids the optic disc (bright spot on left), focusing on pathological regions
- 100% confidence indicates clear diabetic retinopathy features

#### Glaucoma Detection

**Prediction:** Glaucoma (97.1% confidence)

**Analysis:**
- Model focuses on the **optic disc and cup region** (critical for glaucoma diagnosis)
- Attention to the **neuroretinal rim** where glaucoma causes thinning
- High confidence (>97%) with clinically relevant focus areas

#### Normal Classification

**Prediction:** Normal (95.9% confidence)

**Analysis:**
- Model examines **broad retinal areas** rather than specific pathological regions
- Focus on **healthy vascular patterns** and overall retinal structure
- Attention distributed across multiple regions (no single abnormality)
- Confidence is high but slightly lower than disease predictions (appropriate conservatism)

#### Explainability Summary

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

3. **Phase 3 Dataset (Augmented Dataset)**
   - Source: Custom Augmented Dataset for underrepresented classes
   - Purpose: Enhance performance on rare conditions via expanded training distribution

**Combined Dataset:** Over 7,417 total images (Train: 70% | Val: 15% | Test: 15%)

**28 Disease Classes:**

| # | Code | Disease | Test Support |
|---|------|---------|-------------|
| 0 | Disease_Risk | General disease risk indicator | 978 |
| 1 | DR | Diabetic Retinopathy | 287 |
| 2 | ARMD | Age-Related Macular Degeneration | 31 |
| 3 | MH | Macular Hole | 104 |
| 4 | DN | Diabetic Nephropathy | 46 |
| 5 | MYA | Myopia | 32 |
| 6 | BRVO | Branch Retinal Vein Occlusion | 23 |
| 7 | TSLN | Tessellation | 53 |
| 8 | ERM | Epiretinal Membrane | 5 |
| 9 | LS | Laser Scars | 15 |
| 10 | MS | Maculopathy | 7 |
| 11 | CSR | Central Serous Retinopathy | 13 |
| 12 | ODC | Optic Disc Cupping | 91 |
| 13 | CRVO | Central Retinal Vein Occlusion | 9 |
| 14 | AH | Asteroid Hyalosis | 5 |
| 15 | ODP | Optic Disc Pallor | 24 |
| 16 | ODE | Optic Disc Edema | 17 |
| 17 | AION | Anterior Ischemic Optic Neuropathy | 4 |
| 18 | PT | Pigment Changes | 4 |
| 19 | RT | Retinitis | 5 |
| 20 | RS | Retinal Scars | 14 |
| 21 | CRS | Chorioretinal Scars | 11 |
| 22 | EDN | Macular Edema | 4 |
| 23 | RPEC | RPE Changes | 4 |
| 24 | MHL | Macular Holes (Large) | 3 |
| 25 | CATARACT | Cataract | 158 |
| 26 | GLAUCOMA | Glaucoma | 242 |
| 27 | NORMAL | Normal/Healthy | 295 |

### 4.2 Methodology

**Key Techniques:**
- **Asymmetric Loss (ASL)**: Handles extreme class imbalance (γ_neg=6, γ_pos=1, clip=0.1)
- **Class-Balanced Effective Weights**: Replaces naive oversampling with effective sample-based reweighting (β=0.9999)
- **Fundus ROI Preprocessing**: Automatic detection and cropping of circular fundus region to remove black borders
- **CLAHE Illumination Normalization**: Contrast-Limited Adaptive Histogram Equalization on green channel
- **Temperature Scaling Calibration**: Post-training probability calibration for better-calibrated confidence scores
- **Top-K Prediction Constraint**: Limits maximum predictions (default 5) with minimum confidence floor (0.3)
- **RandAugment**: Data augmentation (N=2, M=7) — toned down from M=9 for medical imaging
- **Learning Rate Warmup**: 5 epochs linear warmup to 5e-4
- **EMA (Exponential Moving Average)**: Decay=0.999 for stable evaluation
- **Gradient Accumulation**: Supports larger effective batch sizes for high-resolution (384px) training
- **Random Erasing**: 30% probability for occlusion robustness
- **Augmentation**: ±45° rotation, horizontal flip, shear=15°, color jitter (toned down for medical images)
- **PyTorch Compilation**: Enabled `torch.compile` for faster GPU execution during training

**Backbone Options (v4.2.0):** EfficientNet-B3 (default), EfficientNet-B2, EfficientNet-V2-S, Swin Transformer (Tiny), ConvNeXt Tiny

### 4.3 Model Architecture

**EfficientNet-B3 + Improved Multi-Label Head**

```
┌─────────────────────────────────────────────────────────────┐
│              OcuNet Phase 2 Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: RGB Image (3 × 384 × 384)                           │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              EfficientNet-B3 Backbone                │    │
│  │              (Pretrained on ImageNet)                │    │
│  │                                                      │    │
│  │  • Compound scaled (depth, width, resolution)       │    │
│  │  • Mobile Inverted Bottleneck (MBConv) blocks       │    │
│  │  • Squeeze-and-Excitation attention                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│               Feature Vector (1536)                          │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Improved Classification Head                 │    │
│  │                                                      │    │
│  │  • Squeeze-and-Excitation Block (SE)                │    │
│  │  • Dropout (p=0.4)                                  │    │
│  │  • Linear: 1536 → 512 + BatchNorm + GELU            │    │
│  │  • Dropout (p=0.4)                                 │    │
│  │  • Linear: 512 → 256 + BatchNorm + GELU             │    │
│  │  • Dropout (p=0.4)                                  │    │
│  │  • Linear: 256 → 28                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│        Logits (28) → Sigmoid → Independent Probabilities     │
│                                                              │
│  Output: P(Disease_i) for each of 28 classes                 │
└─────────────────────────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 + Improved Classification Head |
| Total Parameters | ~12M |
| Trainable Parameters | ~12M |
| Input Size | 384 × 384 × 3 |
| Output Classes | 28 (independent sigmoids) |
| Loss Function | Asymmetric Loss (ASL) |
| Classification Head | SE Block + 3-layer MLP with BatchNorm + GELU |

### 4.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Maximum Epochs | 150 |
| Learning Rate | 5e-5 → 5e-4 (warmup) |
| Peak Learning Rate | 5e-4 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Early Stopping | 25 epochs patience |
| Mixed Precision | FP16 |
| LR Schedule | Cosine Annealing after warmup |
| EMA Decay | 0.999 |
| Loss Function | ASL (configurable: ASL/Focal/BCE) |
| Class Weighting | Effective sample-based (β=0.9999) |
| Preprocessing | Fundus ROI Crop + CLAHE |
| Gradient Accumulation | Configurable (default 1) |

### 4.5 Training Progress

```
Epoch   1/150: Val F1: 0.0916 | mAP: 0.0737 | ROC-AUC: 0.4741 [SAVED]
Epoch   5/150: Val F1: 0.1070 | mAP: 0.1042 | ROC-AUC: 0.6444 (warmup end)
Epoch  20/150: Val F1: 0.1406 | mAP: 0.4406 | ROC-AUC: 0.9097 [SAVED]
Epoch  42/150: Val F1: 0.2047 | mAP: 0.5422 | ROC-AUC: 0.9319 [SAVED]
Epoch  66/150: Val F1: 0.2856 | mAP: 0.5627 | ROC-AUC: 0.9371 [SAVED]
Epoch  83/150: Val F1: 0.3181 | mAP: 0.5827 | ROC-AUC: 0.9424 [SAVED]
Epoch  94/150: Val F1: 0.3379 | mAP: 0.5815 | ROC-AUC: 0.9384 [SAVED]
Epoch 120/150: Val F1: 0.3588 | mAP: 0.5646 | ROC-AUC: 0.9364 [SAVED]
Epoch 135/150: Val F1: 0.3699 | mAP: 0.5604 | ROC-AUC: 0.9355 [SAVED] ← BEST
Epoch 150/150: Val F1: 0.3533 | mAP: 0.5592 | ROC-AUC: 0.9359

Best Validation F1: 0.3699 at Epoch 135
Best Validation mAP: 0.5604
```

**Observations:**
1. ROC-AUC rapidly improved to >0.90 by epoch 20 and stabilized around 0.93-0.94
2. F1 score steadily improved from 0.09 to 0.37 over 135 epochs
3. mAP peaked around 0.58 and stabilized
4. Model continued improving slowly even in late epochs (epoch 135 best)
5. Train mAP reached 0.95, indicating the model learned the training distribution well

### 4.6 Results

#### Overall Metrics

| Metric | Default Threshold (0.5) | Optimized Thresholds |
|--------|------------------------|---------------------|
| **F1 Score (Macro)** | 33.79% | **54.93%** |
| **F1 Score (Micro)** | 52.96% | **73.53%** |
| **F1 Score (Samples)** | 55.67% | - |
| **Precision (Macro)** | 23.65% | - |
| **Recall (Macro)** | 74.03% | - |
| **mAP** | 53.83% | - |
| **ROC-AUC (Macro)** | **93.68%** | - |
| **Hamming Loss** | 0.1111 | - |
| **Exact Match Ratio** | 9.98% | - |

#### Per-Class Performance

| Class | Precision | Recall | F1 | AP | AUC | Support |
|-------|-----------|--------|----|----|-----|---------|
| Disease_Risk | 0.955 | 0.827 | 0.887 | 0.973 | 0.912 | 978 |
| NORMAL | 0.388 | 0.993 | 0.558 | 0.742 | 0.919 | 295 |
| DR | 0.416 | 0.962 | 0.580 | 0.845 | 0.936 | 287 |
| GLAUCOMA | 0.308 | 0.963 | 0.467 | 0.720 | 0.905 | 242 |
| CATARACT | 0.347 | 1.000 | 0.515 | 0.958 | 0.993 | 158 |
| MH | 0.272 | 1.000 | 0.427 | 0.865 | 0.984 | 104 |
| ODC | 0.178 | 0.923 | 0.299 | 0.546 | 0.909 | 91 |
| TSLN | 0.202 | 1.000 | 0.337 | 0.653 | 0.984 | 53 |
| DN | 0.102 | 0.957 | 0.184 | 0.312 | 0.910 | 46 |
| MYA | 0.244 | 0.938 | 0.387 | 0.805 | 0.975 | 32 |
| ARMD | 0.243 | 0.903 | 0.384 | 0.729 | 0.975 | 31 |
| ODP | 0.133 | 0.750 | 0.226 | 0.344 | 0.952 | 24 |
| BRVO | 0.333 | 0.826 | 0.475 | 0.845 | 0.980 | 23 |
| ODE | 0.255 | 0.824 | 0.389 | 0.795 | 0.957 | 17 |
| LS | 0.113 | 0.800 | 0.198 | 0.371 | 0.953 | 15 |
| RS | 0.361 | 0.929 | 0.520 | 0.731 | 0.995 | 14 |
| CSR | 0.256 | 0.769 | 0.385 | 0.530 | 0.989 | 13 |
| CRS | 0.095 | 0.364 | 0.151 | 0.216 | 0.946 | 11 |
| CRVO | 0.195 | 0.889 | 0.320 | 0.737 | 0.990 | 9 |
| MS | 0.167 | 0.429 | 0.240 | 0.360 | 0.918 | 7 |
| ERM | 0.167 | 0.200 | 0.182 | 0.066 | 0.679 | 5 |
| AH | 0.185 | 1.000 | 0.312 | 0.821 | 0.999 | 5 |
| RT | 0.250 | 0.400 | 0.308 | 0.315 | 0.983 | 5 |
| AION | 0.182 | 0.500 | 0.267 | 0.348 | 0.837 | 4 |
| PT | 0.053 | 0.250 | 0.087 | 0.077 | 0.856 | 4 |
| EDN | 0.091 | 0.500 | 0.154 | 0.193 | 0.927 | 4 |
| RPEC | 0.062 | 0.500 | 0.111 | 0.067 | 0.886 | 4 |
| MHL | 0.067 | 0.333 | 0.111 | 0.110 | 0.984 | 3 |

#### Clinical Sensitivity Analysis

| Disease | Recall | Rating |
|---------|--------|--------|
| CATARACT | 1.000 | ✅ EXCELLENT |
| MH | 1.000 | ✅ EXCELLENT |
| TSLN | 1.000 | ✅ EXCELLENT |
| AH | 1.000 | ✅ EXCELLENT |
| NORMAL | 0.993 | ✅ EXCELLENT |
| GLAUCOMA | 0.963 | ✅ EXCELLENT |
| DR | 0.962 | ✅ EXCELLENT |
| DN | 0.957 | ✅ EXCELLENT |
| MYA | 0.938 | ✅ EXCELLENT |
| RS | 0.929 | ✅ EXCELLENT |
| ODC | 0.923 | ✅ EXCELLENT |
| ARMD | 0.903 | ✅ EXCELLENT |
| CRVO | 0.889 | ✅ GOOD |
| Disease_Risk | 0.827 | ✅ GOOD |
| BRVO | 0.826 | ✅ GOOD |
| ODE | 0.824 | ✅ GOOD |
| LS | 0.800 | ✅ GOOD |
| CSR | 0.769 | ⚠️ MODERATE |
| ODP | 0.750 | ⚠️ MODERATE |
| AION | 0.500 | ❌ NEEDS IMPROVEMENT |
| EDN | 0.500 | ❌ NEEDS IMPROVEMENT |
| RPEC | 0.500 | ❌ NEEDS IMPROVEMENT |
| MS | 0.429 | ❌ NEEDS IMPROVEMENT |
| RT | 0.400 | ❌ NEEDS IMPROVEMENT |
| CRS | 0.364 | ❌ NEEDS IMPROVEMENT |
| MHL | 0.333 | ❌ NEEDS IMPROVEMENT |
| PT | 0.250 | ❌ NEEDS IMPROVEMENT |
| ERM | 0.200 | ❌ NEEDS IMPROVEMENT |

**Summary:** 17 out of 28 classes achieve ≥80% sensitivity. Classes with low sensitivity have extremely small test sets (3-11 samples), making reliable evaluation challenging.

### 4.7 Threshold Optimization

Per-class threshold optimization significantly improved F1 scores:

| Class | Optimized Threshold | Optimized F1 |
|-------|-------------------|-------------|
| Disease_Risk | 0.40 | 0.8967 |
| DR | 0.70 | 0.7896 |
| ARMD | 0.80 | 0.6742 |
| MH | 0.85 | 0.7664 |
| CATARACT | 0.85 | 0.8820 |
| GLAUCOMA | 0.70 | 0.6183 |
| NORMAL | 0.75 | 0.6382 |
| BRVO | 0.75 | 0.7111 |
| AION | 0.55 | 0.7692 |
| AH | 0.65 | 0.7500 |
| RS | 0.75 | 0.7568 |
| ODE | 0.85 | 0.6957 |
| MYA | 0.85 | 0.6452 |
| TSLN | 0.85 | 0.6710 |
| PT | 0.85 | 0.6667 |
| CRVO | 0.85 | 0.6000 |
| CSR | 0.85 | 0.5882 |
| RT | 0.80 | 0.5333 |

**With Optimized Thresholds:**
- F1 (Macro): **54.93%** (vs 33.79% default) — **+62.6% improvement**
- F1 (Micro): **73.53%** (vs 52.96% default) — **+38.8% improvement**

---

## 5. Usage Guide

### 5.1 Installation

```bash
# Clone the repository
git clone https://github.com/Utkarsh2929/OcuNet.git
cd OcuNet

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
│   └── config.yaml              # Training & model configuration
├── data/                         # Dataset directory (not in git)
│   ├── dataset/                 # Phase 1 dataset (4 classes)
│   └── rfmid/                   # Phase 2 RFMiD dataset (28 classes)
├── src/
│   ├── __init__.py              # Package exports
│   ├── dataset.py               # Data loading, augmentation, class-balanced weights
│   ├── models.py                # Multi-backbone support, ASL/Focal/BCE loss, SE blocks
│   ├── train.py                 # Training loop with EMA, warmup, gradient accumulation
│   ├── evaluate.py              # Evaluation metrics, plots, reports
│   ├── calibrate.py             # Temperature scaling calibration & threshold optimization
│   ├── preprocessing.py         # Fundus ROI crop, CLAHE illumination normalization
│   └── utils.py                 # Utility functions
├── tests/
│   └── test_improvements.py     # 22 automated tests for all modules
├── checkpoints/                 # Saved model weights (not in git)
│   ├── best_model.pth           # Best trained model (~116 MB)
│   └── latest_checkpoint.pth    # Latest checkpoint
├── evaluation_results/          # Metrics, plots, reports (not in git)
│   └── calibration.yaml         # Temperature scaling + calibrated thresholds
├── train_pipeline.py            # Main training entry point
├── predict.py                   # Multi-label prediction with calibration & top-K
├── setup_datasets.py            # Dataset setup & verification
├── requirements.txt             # Dependencies
├── Project Report.md            # This document
├── Training Log *.txt           # Training run logs
└── README.md                    # Quick-start documentation
```

### 5.3 Dataset Setup

**Phase 1 Dataset:** Download from [Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) and extract to `data/dataset/`:

```
data/dataset/
├── cataract/
├── diabetic_retinopathy/
├── glaucoma/
└── normal/
```

**Phase 2 Dataset (RFMiD):** Download from [IEEE Dataport](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid) and extract to `data/rfmid/`:

```
data/rfmid/
├── Training_Set/
├── Evaluation_Set/
├── Test_Set/
└── RFMiD_*.csv
```

Use `setup_datasets.py` to verify dataset setup:
```bash
python setup_datasets.py verify
```

### 5.4 Training

```bash
# Full pipeline (train + evaluate + optimize + calibrate)
python train_pipeline.py --mode all

# Individual modes
python train_pipeline.py --mode train      # Training only
python train_pipeline.py --mode evaluate   # Evaluation only
python train_pipeline.py --mode optimize   # Threshold optimization only
python train_pipeline.py --mode calibrate  # Temperature scaling calibration only

# Custom config
python train_pipeline.py --config path/to/config.yaml --mode all
```

### 5.5 Prediction API

**Python Interface:**

```python
from predict import ImprovedMultiLabelClassifier

# Initialize classifier with calibration and top-K constraints
classifier = ImprovedMultiLabelClassifier(
    max_predictions=5,    # Limit false positives
    min_confidence=0.3    # Minimum confidence floor
)

# Single image prediction (with temperature scaling + calibrated thresholds)
result = classifier.predict("path/to/retinal_image.jpg")

print(result['detected_diseases'])  # ['DR', 'ARMD']
print(result['probabilities'])      # {'DR': 0.85, 'ARMD': 0.72, ...}
print(result['temperature'])        # 1.2345 (calibration temperature)

# Generate a detailed medical report
classifier.generate_report(result, output_path="report.txt")
```

**Command Line:**

```bash
python predict.py path/to/image.jpg
```

---

## 6. Conclusion

### 6.1 Achievements

**Phase 1:**
1. **High Accuracy:** 86.89% test accuracy with 97.88% ROC-AUC
2. **Clinical Reliability:** All 4 disease classes exceed 80% sensitivity threshold
3. **Explainability:** Grad-CAM visualizations confirm clinically appropriate focus regions
4. **Efficiency:** Training completes in ~15 minutes on consumer GPU

**Phase 2:**
1. **Multi-Label Detection:** 28 retinal conditions detected simultaneously
2. **High ROC-AUC:** 93.68% macro-average across all 28 classes
3. **Strong Sensitivity:** 17/28 classes achieve ≥80% recall
4. **Optimized Thresholds:** Per-class thresholds improve F1 (Micro) to 73.53%
5. **Co-morbidity Detection:** Multiple diseases detected in single images

### 6.2 Strengths

| Strength | Phase 1 | Phase 2 |
|----------|---------|---------| 
| Sensitivity | 94% cataract, 94% DR, 87% glaucoma | 100% cataract, 96% DR, 96% glaucoma |
| ROC-AUC | All classes >95% | 93.68% macro (28 classes) |
| Disease Coverage | 4 classes | 28 classes |
| Co-morbidity Detection | No | Yes |
| Threshold Optimization | N/A | Per-class optimized |
| Interpretable | Grad-CAM | Grad-CAM |

### 6.3 Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Normal class accuracy (73% Phase 1) | May over-refer healthy patients | Clinical verification |
| Rare disease performance | Low precision for rare classes (<10 samples) | Per-class thresholds, collect more data |
| Dataset diversity | May not generalize to all populations | Collect more diverse data |
| Single image input | No temporal/multi-view analysis | Future enhancement |

### 6.4 Clinical Applicability

OcuNet is suitable for:
- ✅ **Primary screening** in resource-limited settings
- ✅ **Triage assistance** to prioritize urgent cases
- ✅ **Educational tool** for training ophthalmologists
- ⚠️ **Requires clinical verification** for final diagnosis

---


### 7.0 Recently Completed (v2.2.0)

The following items from previous Future Work have been implemented:

- ✅ **Confidence calibration**: Temperature scaling for probability calibration
- ✅ **Hard negative mining**: Configuration and hooks for upweighting difficult samples
- ✅ **Class-balanced loss**: Effective sample-based reweighting (replaces naive oversampling)
- ✅ **Fundus preprocessing**: ROI crop + CLAHE illumination normalization
- ✅ **Stronger backbones**: EfficientNetV2-S, Swin-T options alongside EfficientNet-B2
- ✅ **Top-K prediction constraint**: Limits false positives with min confidence floor

### 7.1 Short-term Improvements

1. **Increase rare class performance**
   - Collect additional samples for classes with <10 test samples
   - Enable hard negative mining in training (config hooks ready)
   - Explore synthetic data generation (e.g., diffusion models)

2. **Multi-severity classification**
   - DR severity grading (mild/moderate/severe/proliferative)
   - Glaucoma staging

3. **Advanced calibration**
   - Evaluate precision-at-recall threshold strategy for clinical deployment
   - Ensemble of calibrated models

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

### A. Complete Configuration (Phase 2 — v2.2.0)

```yaml
# config/config.yaml

# Dataset Configuration
dataset:
  phase1_root: "data/dataset_4classes"
  use_phase1: true
  phase2_root: "data/rfmid"
  use_phase2: true
  phase3_root: "data/Augmented Dataset"
  use_phase3: true
  image_size: 384
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42
  min_samples_per_class: 10
  oversample_rare_classes: false

# Preprocessing
preprocessing:
  fundus_roi_crop: true
  roi_padding: 0.05
  clahe: true
  clahe_clip_limit: 2.0
  clahe_channel: "green"

# Training Configuration
training:
  batch_size: 16
  num_epochs: 200
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 30
  num_workers: 8
  threshold: 0.5
  use_class_specific_thresholds: true
  warmup_epochs: 5
  use_effective_weights: true
  effective_weights_beta: 0.9999
  gradient_accumulation_steps: 1
  hard_negative_mining: false

# Model Configuration
model:
  architecture: "efficientnet_b3"
  pretrained: true
  dropout_rate: 0.4
  loss_function: "asl"
  asl_gamma_neg: 6
  asl_gamma_pos: 1
  asl_clip: 0.1
  multi_label: true

# Augmentation Configuration
augmentation:
  rotation_degrees: 45
  horizontal_flip: true
  vertical_flip: false
  zoom_range: [0.8, 1.2]
  shear_range: 15
  brightness_range: [0.7, 1.3]
  contrast_range: [0.7, 1.3]
  saturation_range: [0.8, 1.2]
  hue_range: 0.05
  random_erasing: true
  random_erasing_prob: 0.3
  random_erasing_scale: [0.02, 0.2]
  use_randaugment: true
  randaugment_n: 2
  randaugment_m: 7

# Calibration
calibration:
  strategy: "f1"
  min_recall: 0.8

# Output
output:
  checkpoint_dir: "checkpoints"
  results_dir: "evaluation_results"
  explainability_dir: "explainability_results"

# Experiment
experiment:
  name: "ocunet_v4"
  version: "4.2.0"
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

### C. Phase 1 Full Evaluation Metrics

```json
{
  "overall": {
    "accuracy": 0.8689,
    "precision_macro": 0.8706,
    "recall_macro": 0.8690,
    "f1_macro": 0.8679
  },
  "per_class": {
    "cataract": { "precision": 0.9545, "recall": 0.9423, "f1_score": 0.9484 },
    "diabetic_retinopathy": { "precision": 0.8245, "recall": 0.9394, "f1_score": 0.8782 },
    "glaucoma": { "precision": 0.8675, "recall": 0.8675, "f1_score": 0.8675 },
    "normal": { "precision": 0.8357, "recall": 0.7267, "f1_score": 0.7774 }
  },
  "roc_auc": {
    "macro": 0.9788,
    "cataract": 0.9966,
    "diabetic_retinopathy": 0.9866,
    "glaucoma": 0.9766,
    "normal": 0.9555
  }
}
```

### D. Phase 1 Training Log Summary

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
| 50 | 0.0146 | 98.95% | 0.2146 | 87.36% | |

---

## Document Information

| Property | Value |
|----------|-------|
| Project Name | OcuNet |
| Version | 2.2.0 |
| Last Updated | February 2026 |
| Author | Utkarsh Gautam |
| License | MIT |

---

*End of Report*