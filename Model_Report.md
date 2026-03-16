# OcuNet — Detailed Model Report

**Deep Learning-Based Multi-Label Retinal Disease Classification System**

---

**Author:** Utkarsh Gautam
**Date:** March 2026
**Version:** 4.2.0 (OcuNet V4)
**Framework:** PyTorch 2.0+ / Python 3.11
**License:** MIT

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Disease Classes](#3-disease-classes)
4. [Datasets](#4-datasets)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Data Augmentation](#6-data-augmentation)
7. [Loss Functions](#7-loss-functions)
8. [Optimizer and Learning Rate Schedule](#8-optimizer-and-learning-rate-schedule)
9. [Class Imbalance Handling](#9-class-imbalance-handling)
10. [Training Configuration](#10-training-configuration)
11. [Regularization Techniques](#11-regularization-techniques)
12. [Calibration and Threshold Optimization](#12-calibration-and-threshold-optimization)
13. [Evaluation Metrics and Results](#13-evaluation-metrics-and-results)
14. [Per-Class Performance](#14-per-class-performance)
15. [Inference Pipeline](#15-inference-pipeline)
16. [Hardware Requirements](#16-hardware-requirements)
17. [Software Dependencies](#17-software-dependencies)
18. [Project File Structure](#18-project-file-structure)
19. [Limitations and Future Work](#19-limitations-and-future-work)
20. [References](#20-references)

---

## 1. Project Overview

### 1.1 Purpose

OcuNet is a deep learning system designed for **automated multi-label classification of retinal fundus images**. It detects **28 eye disease classes** simultaneously from a single retinal photograph. The system targets **clinical triage and screening** in resource-limited settings, where access to ophthalmologists is scarce.

### 1.2 Clinical Disclaimer

> **⚠️ This is a research prototype for triage support only.** It is NOT cleared for standalone clinical diagnosis. Prospective clinical validation is required before any deployment.

### 1.3 Development Phases

The project was developed through two phases:

| Phase | Task | Classes | Dataset | Key Result |
|-------|------|---------|---------|------------|
| **Phase 1** | Single-label classification | 4 (Cataract, DR, Glaucoma, Normal) | Kaggle Eye Diseases | 86.89% accuracy, 97.88% ROC-AUC |
| **Phase 2** | Multi-label classification | 28 diseases | RFMiD + Kaggle + Augmented | 93.68% macro ROC-AUC, 73.53% optimized micro F1 |

Phase 2 is the primary scientific contribution and the focus of this report.

### 1.4 Key Performance Summary

| Metric | Value |
|--------|-------|
| **ROC-AUC (Macro)** | 93.68% |
| **mAP (Mean Average Precision)** | 53.83% |
| **F1 Score (Micro, default threshold)** | 52.96% |
| **F1 Score (Micro, optimized thresholds)** | 73.53% |
| **F1 Score (Macro, optimized thresholds)** | 54.93% |
| **Hamming Loss** | 0.1111 |
| **Diseases with ≥80% recall** | 17 / 28 |

---

## 2. Model Architecture

### 2.1 High-Level Design

OcuNet uses a **transfer learning** approach: a pre-trained **EfficientNet-B3** backbone extracts visual features from retinal images, and a custom **Improved Classification Head** with Squeeze-and-Excitation attention maps those features to 28 disease outputs.

```
Input Image (384 × 384 × 3, RGB)
        │
        ▼
┌─────────────────────────┐
│   EfficientNet-B3       │  ← Pre-trained on ImageNet (1536-dim output)
│   (Backbone)            │
└────────────┬────────────┘
             │ 1536-dim feature vector
             ▼
┌─────────────────────────┐
│  Squeeze-and-Excitation │  ← Channel attention (reduction=16)
│  (SE Block)             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dropout(0.40)          │
│  Linear(1536 → 1024)   │
│  BatchNorm1d(1024)      │
│  GELU                   │
│  Dropout(0.32)          │
│  Linear(1024 → 512)    │
│  BatchNorm1d(512)       │
│  GELU                   │
│  Dropout(0.24)          │
│  Linear(512 → 256)     │
│  BatchNorm1d(256)       │
│  GELU                   │
│  Dropout(0.16)          │
│  Linear(256 → 28)      │  ← 28 sigmoid logits (one per disease)
└─────────────────────────┘
```

### 2.2 Backbone — EfficientNet-B3

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 |
| Pre-trained Weights | ImageNet-1K (IMAGENET1K_V1) |
| Input Resolution | 384 × 384 |
| Feature Dimension | 1536 |
| Total Parameters | ~12 million (trainable) |
| Source | `torchvision.models.efficientnet_b3` |

EfficientNet-B3 was chosen for its balance between accuracy and computational efficiency. It uses compound scaling (depth, width, resolution) to achieve strong performance on image classification tasks while remaining feasible for a 6 GB GPU.

### 2.3 Squeeze-and-Excitation (SE) Block

The SE block applies **channel attention** to the 1536-dimensional feature vector before the classification layers. It learns to re-weight feature channels based on their importance.

```
Input (B, C)
    │
    ▼
Linear(C → C/16) + ReLU        ← Squeeze: compress to C/16 dimensions
    │
    ▼
Linear(C/16 → C) + Sigmoid     ← Excitation: produce channel weights
    │
    ▼
Element-wise multiply with Input  ← Scale original features
```

| Parameter | Value |
|-----------|-------|
| Input Channels (C) | 1536 |
| Reduction Ratio | 16 |
| Bottleneck Size | 96 (= 1536 / 16) |

### 2.4 Classification Head

The `ImprovedClassificationHead` consists of three hidden layers with decreasing dropout, batch normalization, and GELU activations:

| Layer | Input → Output | Dropout | Activation |
|-------|---------------|---------|------------|
| SE Block | 1536 → 1536 | — | Sigmoid (internal) |
| Dropout | — | 0.40 | — |
| Linear + BN + GELU | 1536 → 1024 | 0.32 | GELU |
| Linear + BN + GELU | 1024 → 512 | 0.24 | GELU |
| Linear + BN + GELU | 512 → 256 | 0.16 | GELU |
| Output Linear | 256 → 28 | — | — (raw logits) |

**Weight Initialization:**
- Linear layers: truncated normal with std=0.02
- BatchNorm layers: weight=1, bias=0

### 2.5 Supported Backbone Alternatives

The architecture is configurable. The following backbones are supported via the `model.architecture` config field:

| Backbone | Feature Dim | Notes |
|----------|-------------|-------|
| `efficientnet_b0` | 1280 | Lightest EfficientNet |
| `efficientnet_b1` | 1280 | |
| `efficientnet_b2` | 1408 | |
| **`efficientnet_b3`** | **1536** | **Default / Used** |
| `efficientnet_b4` | 1792 | |
| `efficientnet_v2_s` | 1280 | V2 architecture |
| `convnext_tiny` | 768 | ConvNeXt family |
| `convnext_small` | 768 | |
| `swin_t` | 768 | Swin Transformer |

---

## 3. Disease Classes

OcuNet classifies images into **28 disease categories** simultaneously (multi-label). A single image can have multiple positive labels.

| Index | Code | Full Name | Description |
|-------|------|-----------|-------------|
| 0 | Disease_Risk | General Disease Risk | Overall screening flag indicating presence of any pathology |
| 1 | DR | Diabetic Retinopathy | Retinal damage from diabetes; leading cause of blindness |
| 2 | ARMD | Age-Related Macular Degeneration | Degeneration of the macula affecting central vision |
| 3 | MH | Macular Hole | Full-thickness defect in the fovea |
| 4 | DN | Diabetic Nephropathy (retinal signs) | Diabetes-related nerve/retinal damage |
| 5 | MYA | Myopia (pathological) | High myopia with retinal changes |
| 6 | BRVO | Branch Retinal Vein Occlusion | Blocked branch vessel in the retina |
| 7 | TSLN | Tessellation | Visible choroidal vessels through thin retina |
| 8 | ERM | Epiretinal Membrane | Scar tissue formation on the retinal surface |
| 9 | LS | Laser Scars | Marks from previous laser treatment |
| 10 | MS | Maculopathy | White patches / macular pathology |
| 11 | CSR | Central Serous Retinopathy | Fluid accumulation under the retina |
| 12 | ODC | Optic Disc Cupping | Enlarged cup-to-disc ratio; glaucoma indicator |
| 13 | CRVO | Central Retinal Vein Occlusion | Blocked central retinal vessel |
| 14 | AH | Asteroid Hyalosis | Calcium-lipid deposits in the vitreous |
| 15 | ODP | Optic Disc Pallor | Pale optic disc indicating nerve damage |
| 16 | ODE | Optic Disc Edema | Swelling of the optic disc |
| 17 | AION | Anterior Ischemic Optic Neuropathy | Loss of blood supply to the optic nerve |
| 18 | PT | Pigment Changes | Retinal pigment changes |
| 19 | RT | Retinitis | Inflammation of the retina |
| 20 | RS | Retinal Scars | Scarred retinal tissue |
| 21 | CRS | Chorioretinitis Scars | Deep choroidal scarring |
| 22 | EDN | Macular Edema | Fluid buildup in the macula |
| 23 | RPEC | RPE Changes | Retinal pigment epithelium changes |
| 24 | MHL | Macular Holes (Large) | Large full-thickness macular defects |
| 25 | CATARACT | Cataract | Lens opacity obscuring fundus view |
| 26 | GLAUCOMA | Glaucoma | Optic nerve damage from elevated pressure |
| 27 | NORMAL | Normal / Healthy | No pathology detected |

---

## 4. Datasets

### 4.1 Overview

OcuNet uses a **three-phase combined dataset** strategy to maximize training data and address class imbalance:

| Phase | Dataset | Type | Total Images | Classes |
|-------|---------|------|-------------|---------|
| Phase 1 | Kaggle Eye Diseases | Single-label | 4,217 | 4 |
| Phase 2 | RFMiD (Retinal Fundus Multi-Disease) | Multi-label | ~3,200 | 28 |
| Phase 3 | Augmented Rare-Class | Multi-label | ~16,242 | 10 |

### 4.2 Phase 1 — Kaggle Eye Diseases Dataset

- **Source:** [Kaggle Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Classes:** 4 (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
- **Total Images:** 4,217
- **Format:** JPEG/PNG, RGB
- **Label Type:** Single-label (folder-based)

| Class | Count | Percentage |
|-------|-------|-----------|
| Cataract | 1,038 | 24.61% |
| Diabetic Retinopathy | 1,098 | 26.04% |
| Glaucoma | 1,007 | 23.88% |
| Normal | 1,074 | 25.47% |

**Splits:** 70% train (2,951) / 15% val (633) / 15% test (633)

### 4.3 Phase 2 — RFMiD Dataset

- **Source:** [IEEE Dataport — RFMiD](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)
- **Classes:** 28 diseases (multi-label, binary labels per disease)
- **Total Images:** ~3,200
- **Format:** JPEG/PNG, RGB
- **Label Type:** CSV files with binary columns for each disease

**Splits:** ~1,920 train / ~640 val / ~640 test

### 4.4 Phase 3 — Augmented Rare-Class Dataset

- **Purpose:** Supplement underrepresented disease classes from RFMiD
- **Classes Augmented:** 10 (CSR, DR, ODE, Glaucoma, Healthy, Macular Scar, Myopia, Pterygium, RD, RP)
- **Total Images:** ~11,369 train / ~2,436 val / ~2,437 test
- **Note:** These are synthetic augmented variants of clinical images, not independent clinical samples

### 4.5 Combined Dataset Summary

| Split | Samples |
|-------|---------|
| **Training** | 16,240 |
| **Validation** | 3,709 |
| **Test** | 3,710 |
| **Total** | 23,659 |

### 4.6 Training Class Distribution (Combined)

| Class | Training Samples | Relative Frequency |
|-------|-----------------|-------------------|
| Disease_Risk | 13,246 | Most common |
| DR | 3,571 | Common |
| GLAUCOMA | 2,720 | Common |
| NORMAL | 2,593 | Common |
| MYA | 1,694 | Moderate |
| MS | 1,381 | Moderate |
| CATARACT | 711 | Moderate |
| RP | 596 | Low |
| ODE | 590 | Low |
| RD | 534 | Low |
| CSR | 446 | Low |
| MH | 317 | Low |
| ODC | 282 | Low |
| TSLN | 186 | Rare |
| DN | 138 | Rare |
| ARMD | 100 | Rare |
| PT | 88 | Rare |
| BRVO | 73 | Rare |
| ODP | 65 | Rare |
| LS | 47 | Very rare |
| RS | 43 | Very rare |
| CRS | 32 | Very rare |
| CRVO | 28 | Very rare |
| RPEC | 22 | Very rare |
| AION | 17 | Very rare |
| AH | 16 | Very rare |
| EDN | 15 | Very rare |
| ERM | 14 | Very rare |
| RT | 14 | Very rare |
| MHL | 11 | Very rare |

The extreme class imbalance (Disease_Risk: 13,246 vs MHL: 11) is one of the primary challenges addressed by the model's loss function and reweighting strategies.

---

## 5. Data Preprocessing

Two preprocessing transforms are applied to all images **before** augmentation and model input. These transforms are implemented as torchvision-compatible transforms in `src/preprocessing.py`.

### 5.1 Fundus ROI Crop (`FundusROICrop`)

**Purpose:** Detect and crop the circular fundus (retinal) region from the image, removing black borders that are commonly present in fundus photographs.

**Why it matters:** Without ROI cropping, the model can learn shortcuts by exploiting the shape of the black border rather than learning from retinal content.

**Method:**
1. Convert image to grayscale
2. Apply Otsu's thresholding to separate bright fundus from dark background
3. Fall back to fixed threshold (intensity > 15) if Otsu gives poor results (foreground ratio < 10% or > 95%)
4. Clean binary mask with morphological operations (close: 3 iterations, open: 2 iterations, 5×5 elliptical kernel)
5. Find the largest contour (the fundus circle)
6. Compute bounding rectangle and add padding
7. Crop and return; return original image if detection fails

| Parameter | Value | Description |
|-----------|-------|-------------|
| `padding_ratio` | 0.05 | Extra padding around ROI as fraction of diameter |
| `min_radius_ratio` | 0.2 | Minimum valid ROI radius as fraction of image size |
| Morphological Kernel | 5×5 elliptical | For close and open operations |

### 5.2 CLAHE Normalization (`CLAHETransform`)

**Purpose:** Normalize illumination variations across fundus images captured by different cameras and under different imaging conditions.

**Why it matters:** The combined dataset (RFMiD + Kaggle + Augmented) contains images from diverse sources with varying brightness, contrast, and color profiles. CLAHE stabilizes these features.

**Method:**
- Apply Contrast Limited Adaptive Histogram Equalization to the **green channel** of the RGB image
- The green channel provides the best contrast for retinal structures (blood vessels, optic disc, lesions)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `clip_limit` | 2.0 | Contrast limiting threshold |
| `tile_grid_size` | 8×8 | Grid size for local histogram equalization |
| `channel` | `"green"` | Which channel to apply CLAHE to |

**Alternative mode:** LAB color space (apply CLAHE to L channel) — configurable but green channel is the default.

### 5.3 Image Normalization

All images are resized and normalized using ImageNet statistics:

| Parameter | Value |
|-----------|-------|
| Resize | 384 × 384 pixels |
| Mean (RGB) | [0.485, 0.456, 0.406] |
| Std (RGB) | [0.229, 0.224, 0.225] |

---

## 6. Data Augmentation

Data augmentation is applied **only during training** to improve generalization and reduce overfitting. The augmentation strategy is conservative compared to natural image tasks, as medical images require preserving diagnostic features.

### 6.1 RandAugment

RandAugment applies N randomly selected transformations from a pool of 14 operations:

| Parameter | Value |
|-----------|-------|
| N (operations per image) | 2 |
| M (magnitude level, 0–30) | 7 |

**Available operations:**
1. Identity (no change)
2. AutoContrast
3. Equalize
4. Rotation
5. Solarize
6. Color jitter
7. Posterize
8. Contrast adjustment
9. Brightness adjustment
10. Sharpness adjustment
11. Shear X
12. Shear Y
13. Translate X
14. Translate Y

### 6.2 Geometric Augmentations

| Augmentation | Value | Notes |
|-------------|-------|-------|
| Rotation | ±45° | Random continuous rotation |
| Horizontal Flip | 50% probability | Realistic for fundus (left/right eye) |
| Vertical Flip | Disabled | Unrealistic for fundus images |
| Zoom Range | 0.8×–1.2× | Random scale |
| Shear Range | ±15° | Random shear |

### 6.3 Color Augmentations

Color augmentations are intentionally conservative for medical imaging:

| Augmentation | Range | Notes |
|-------------|-------|-------|
| Brightness | 0.7–1.3× | Reduced from 0.6–1.4 |
| Contrast | 0.7–1.3× | Reduced from 0.6–1.4 |
| Saturation | 0.8–1.2× | Moderate |
| Hue | ±0.05 | Reduced from ±0.1 (preserve color integrity) |

### 6.4 Advanced Augmentations

| Augmentation | Setting | Notes |
|-------------|---------|-------|
| Random Erasing | Enabled, p=0.3, scale=[0.02, 0.2] | Simulates occlusions |
| Mixup | Disabled (alpha=0.0) | Not suitable for multi-label |
| CutMix | Disabled (alpha=0.0) | Not suitable for multi-label |

---

## 7. Loss Functions

### 7.1 Primary Loss — Asymmetric Loss (ASL)

The primary loss function is **Asymmetric Loss**, which is specifically designed for multi-label classification with class imbalance.

**Formula:**

```
L_ASL = -y · log(p) · (1 - p)^γ_pos − (1 - y) · log(clip(1 - p)) · p^γ_neg
```

Where:
- `p = sigmoid(logit)` — predicted probability
- `y` — ground truth label (0 or 1)
- `γ_pos` — focusing parameter for positive examples
- `γ_neg` — focusing parameter for negative examples
- `clip(x) = max(x + clip_value, 1)` — asymmetric clipping

**Configuration:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `gamma_neg` | 6 | Strong down-weighting of easy negatives |
| `gamma_pos` | 1 | Modest focus on hard positives |
| `clip` | 0.1 | Asymmetric clipping to handle extreme negatives |
| `pos_weight` | Per-class (computed) | Class-balanced effective weights |

**Why ASL:** In multi-label classification, most labels per sample are negative. Standard BCE treats all negatives equally, leading to the model predicting "no disease" for rare classes. ASL addresses this by focusing on hard negatives (γ_neg=6) while applying minimal focusing to positives (γ_pos=1), combined with asymmetric clipping.

**Reference:** Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", arXiv:2009.14119

### 7.2 Alternative Loss Functions

Two alternative loss functions are available via the `model.loss_function` config field:

**Focal Loss:**
```
L_focal = -α_t · (1 - p_t)^γ · log(p_t)
```
- `gamma`: 2.0
- Supports per-class `pos_weight`

**Binary Cross-Entropy (BCE):**
- Standard `BCEWithLogitsLoss` with per-class `pos_weight`

---

## 8. Optimizer and Learning Rate Schedule

### 8.1 Optimizer — AdamW

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ (0.0003) |
| Weight Decay | 1×10⁻⁴ (0.0001) |
| Betas | (0.9, 0.999) — PyTorch defaults |

### 8.2 Learning Rate Schedule — Warmup + Cosine Decay

The learning rate follows a two-phase schedule implemented in `WarmupCosineScheduler`:

**Phase 1 — Linear Warmup (Epochs 0–4):**
```
lr(step) = base_lr × (step + 1) / warmup_steps
```

**Phase 2 — Cosine Annealing (Epochs 5–200):**
```
lr(step) = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × progress))
```

| Parameter | Value |
|-----------|-------|
| Warmup Epochs | 5 |
| Total Epochs | 200 (max) |
| Minimum LR | 1×10⁻⁷ |
| Base LR | 3×10⁻⁴ |

---

## 9. Class Imbalance Handling

The dataset exhibits severe class imbalance (Disease_Risk has ~1,200× more samples than MHL). OcuNet uses a multi-pronged strategy:

### 9.1 Effective Sample Reweighting (Primary)

Based on the **Effective Number of Samples** framework (Cui et al., CVPR 2019):

```
effective_num(n) = (1 - β^n) / (1 - β)
weight(n) = 1 / effective_num(n) = (1 - β) / (1 - β^n)
normalized_weight = weight / mean(all_weights)
```

| Parameter | Value |
|-----------|-------|
| β (beta) | 0.9999 |
| Normalization | Mean weight = 1.0 |

With β=0.9999, the weights provide strong up-weighting for rare classes while being numerically stable. The computed per-class weights are used as `pos_weight` in the loss function.

### 9.2 Weighted Random Sampling

Per-sample weights are computed as the inverse frequency of the rarest class present in that sample. This is implemented via PyTorch's `WeightedRandomSampler` to ensure rare disease samples appear more frequently during training.

### 9.3 Class Oversampling (Optional, Disabled)

An optional oversampling strategy duplicates samples from classes with fewer than a threshold number of instances:

| Parameter | Value |
|-----------|-------|
| Enabled | `false` (disabled in current config) |
| Threshold | 50 samples |
| Factor | 3× duplication |

This is disabled in favor of effective sample reweighting, which is more principled.

---

## 10. Training Configuration

### 10.1 Core Training Parameters

| Parameter | Value | Config Key |
|-----------|-------|------------|
| Batch Size | 16 | `training.batch_size` |
| Gradient Accumulation Steps | 1 | `training.gradient_accumulation_steps` |
| Effective Batch Size | 16 (= 16 × 1) | — |
| Max Epochs | 200 | `training.num_epochs` |
| Early Stopping Patience | 30 epochs | `training.early_stopping_patience` |
| Data Loading Workers | 8 | `training.num_workers` |
| Default Threshold | 0.5 | `training.threshold` |
| Per-Class Thresholds | Enabled | `training.use_class_specific_thresholds` |
| Random Seed | 42 | `dataset.random_seed` |

### 10.2 Mixed Precision Training (AMP)

| Parameter | Value |
|-----------|-------|
| Enabled | Yes (on CUDA devices) |
| Precision | FP16 for forward/backward, FP32 for loss |
| Benefit | ~20% faster, ~50% less GPU memory |
| Implementation | `torch.cuda.amp.autocast` + `GradScaler` |

### 10.3 Model Compilation

| Parameter | Value |
|-----------|-------|
| `torch.compile` | Enabled (when available) |
| Benefit | ~15–20% speedup on compatible GPUs |

### 10.4 Training Pipeline Modes

The training pipeline (`train_pipeline.py`) supports multiple execution modes:

```bash
python train_pipeline.py --mode all       # Full pipeline: train → evaluate → optimize
python train_pipeline.py --mode train     # Training only
python train_pipeline.py --mode evaluate  # Evaluation from checkpoint
python train_pipeline.py --mode optimize  # Threshold optimization only
```

---

## 11. Regularization Techniques

| Technique | Configuration | Purpose |
|-----------|--------------|---------|
| **Dropout** | Cascading: 0.40 → 0.32 → 0.24 → 0.16 | Prevents co-adaptation of features |
| **Batch Normalization** | 3 layers (1024, 512, 256) | Stabilizes internal activations |
| **Weight Decay** | 1×10⁻⁴ (L2 regularization) | Penalizes large weights |
| **Warmup** | 5 epochs linear ramp | Prevents instability in early training |
| **Gradient Clipping** | max_norm=1.0 | Prevents exploding gradients |
| **Early Stopping** | Patience=30 epochs on val F1 | Stops training when val metric plateaus |
| **Data Augmentation** | RandAugment + geometric + color | Increases effective training set diversity |
| **EMA (Exponential Moving Average)** | Decay=0.999 | Smooths weights for better generalization |

### 11.1 Exponential Moving Average (EMA)

EMA maintains a running average of model weights throughout training:

```
ema_weight = decay × ema_weight + (1 - decay) × model_weight
```

| Parameter | Value |
|-----------|-------|
| Decay | 0.999 |
| Application | During validation and inference |
| Benefit | More stable predictions, better generalization |

---

## 12. Calibration and Threshold Optimization

### 12.1 Temperature Scaling

**Purpose:** Learn a single scalar temperature parameter T that rescales logits before the sigmoid function, so that predicted probabilities better reflect true likelihoods (i.e., if the model says 80% confidence, the condition is actually present ~80% of the time).

**Formula:**
```
calibrated_probability = sigmoid(logit / T)
```

| Parameter | Value |
|-----------|-------|
| Learned Temperature (T) | 1.1926 |
| Optimization Method | L-BFGS |
| Learning Rate | 0.01 |
| Max Iterations | 200 |
| Loss | Binary cross-entropy with logits |

**Interpretation:**
- T > 1: Probabilities are pushed toward 0.5 (less confident, softer predictions)
- T < 1: Probabilities are pushed toward 0 or 1 (more confident, sharper predictions)
- T = 1.1926 indicates the raw model is slightly overconfident, and calibration softens predictions modestly

**Reference:** Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

### 12.2 Per-Class Threshold Optimization

**Purpose:** Different diseases have different optimal decision boundaries. Rather than using a uniform 0.5 threshold, each class gets an independently optimized threshold.

**Strategy:** F1-Maximization (default)
- For each class, search over thresholds from 0.05 to 0.95 in steps of 0.025
- Select the threshold that maximizes F1 score for that class

**Alternative Strategy:** Precision-at-Recall
- Find the best precision while maintaining recall ≥ 0.8 (configurable)

**Impact of threshold optimization:**

| Metric | Uniform 0.5 | Optimized Thresholds | Improvement |
|--------|-------------|---------------------|-------------|
| F1 (Macro) | 33.79% | 54.93% | +62.6% relative |
| F1 (Micro) | 52.96% | 73.53% | +38.8% relative |

**Per-class calibrated thresholds (from Training Log 5):**

| Class | Optimized Threshold | Calibrated F1 |
|-------|-------------------|---------------|
| Disease_Risk | 0.300 | 0.9043 |
| DR | 0.575 | 0.5538 |
| ARMD | 0.900 | 0.4894 |
| MH | 0.850 | 0.5789 |
| DN | 0.825 | 0.2558 |
| MYA | 0.750 | 0.5866 |
| BRVO | 0.925 | 0.3902 |
| TSLN | 0.925 | 0.5256 |
| ERM | 0.625 | 0.2500 |
| LS | 0.750 | 0.2326 |
| MS | 0.800 | 0.5495 |
| CSR | 0.925 | 0.5082 |
| ODC | 0.900 | 0.2920 |
| CRVO | 0.925 | 0.5833 |
| AH | 0.925 | 0.2105 |
| ODP | 0.800 | 0.0963 |
| ODE | 0.875 | 0.7794 |
| AION | 0.925 | 0.4000 |
| PT | 0.925 | 0.9231 |
| RT | 0.875 | 0.5000 |
| RS | 0.875 | 0.7179 |
| CRS | 0.900 | 0.2051 |
| EDN | 0.925 | 0.2174 |
| RPEC | 0.925 | 0.1429 |
| MHL | 0.900 | 0.3333 |
| CATARACT | 0.925 | 0.6848 |
| GLAUCOMA | 0.725 | 0.4209 |
| NORMAL | 0.650 | 0.5158 |

Output files:
- `evaluation_results/optimal_thresholds.yaml`
- `evaluation_results/calibration.yaml`

---

## 13. Evaluation Metrics and Results

### 13.1 Metrics Used

| Metric | Formula / Description | Relevance |
|--------|-----------------------|-----------|
| **ROC-AUC (Macro)** | Average AUC of ROC curves across all classes | Rank-based metric, threshold-independent |
| **mAP** | Mean Average Precision across all classes | Threshold-independent precision-recall summary |
| **F1 (Micro)** | 2TP / (2TP + FP + FN) globally aggregated | Overall positive prediction quality |
| **F1 (Macro)** | Unweighted mean of per-class F1 | Equal importance to all classes |
| **Hamming Loss** | Fraction of incorrect label predictions | Lower is better; accounts for all labels |
| **Exact Match Ratio** | Fraction of samples with all labels correct | Very strict for multi-label |
| **Precision** | TP / (TP + FP) | How many predictions are correct |
| **Recall (Sensitivity)** | TP / (TP + FN) | How many true positives are found |

### 13.2 Aggregate Results (Phase 2, Test Set)

| Metric | Value (Default 0.5 Threshold) | Value (Optimized Thresholds) |
|--------|------------------------------|------------------------------|
| **ROC-AUC (Macro)** | 93.68% | 93.68% (threshold-independent) |
| **mAP** | 53.83% | 53.83% (threshold-independent) |
| **F1 (Micro)** | 52.96% | 73.53% |
| **F1 (Macro)** | 33.79% | 54.93% |
| **Hamming Loss** | 0.1111 | — |
| **Precision (Macro)** | — | 39.93% (calibrated) |

### 13.3 Phase 1 Results (4-Class, for reference)

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 91.31% (Epoch 41) |
| Training Accuracy | 97.29% |
| ROC-AUC | 97.88% |
| Training Loss | 0.0335 |
| Validation Loss | 0.1434 |

---

## 14. Per-Class Performance

### 14.1 Clinical Sensitivity (Recall) for Major Diseases

The system prioritizes **recall (sensitivity)** to minimize missed diagnoses. 17 out of 28 classes exceed the 80% recall threshold commonly used in clinical screening (e.g., NHS screening programs).

| Disease | Recall | Status |
|---------|--------|--------|
| Cataract | 100.0% | ✅ Excellent — Perfect detection |
| Macular Hole (MH) | 100.0% | ✅ Excellent — Perfect detection |
| Diabetic Retinopathy (DR) | 96.2% | ✅ Excellent |
| Glaucoma | 96.3% | ✅ Excellent |
| Myopia (MYA) | 93.8% | ✅ Excellent |
| Age-Related Macular Degeneration (ARMD) | 90.3% | ✅ Excellent |

### 14.2 Calibration Impact (Uncalibrated vs. Calibrated)

From the latest training run (Training Log 5), temperature scaling at T=1.1926 showed measurable improvements for many classes:

| Metric | Uncalibrated | Calibrated | Change |
|--------|-------------|-----------|--------|
| F1 (Macro) | 0.4557 | 0.4741 | +0.0185 |
| F1 (Micro) | 0.6642 | 0.6703 | +0.0061 |
| Precision (Macro) | 0.3755 | 0.3993 | +0.0238 |

Notable per-class improvements after calibration:
- **CSR:** F1 improved from 0.4033 → 0.5082 (+0.1049)
- **MHL:** F1 improved from 0.2500 → 0.3333 (+0.0833)
- **AION:** F1 improved from 0.3333 → 0.4000 (+0.0667)
- **PT:** F1 improved from 0.8667 → 0.9231 (+0.0564)

---

## 15. Inference Pipeline

### 15.1 Prediction Interface

The `ImprovedMultiLabelClassifier` class in `predict.py` provides a complete inference pipeline:

```python
from predict import ImprovedMultiLabelClassifier

classifier = ImprovedMultiLabelClassifier(
    checkpoint_path="checkpoints/ocunetv4.pth",
    config_path="config/config.yaml",
    thresholds_path="evaluation_results/optimal_thresholds.yaml",
    calibration_path="evaluation_results/calibration.yaml",
    max_predictions=5,
    min_confidence=0.3
)

# Single image prediction
result = classifier.predict("path/to/retinal_image.jpg")

# Access results
print(result['diseases'])          # List of detected diseases
print(result['probabilities'])     # Dict of all disease probabilities
print(result['confidence'])        # Max confidence score

# Generate medical report
classifier.generate_report(result, output_path="report.txt")
```

### 15.2 Inference Steps

1. Load and preprocess image (ROI crop → CLAHE → resize to 384×384 → normalize)
2. Run forward pass through EfficientNet-B3 + classification head
3. Apply temperature scaling to logits (T = 1.1926)
4. Apply sigmoid to get calibrated probabilities
5. Apply per-class optimized thresholds
6. Return detected diseases with confidence scores

### 15.3 Command Line Interface

```bash
python predict.py path/to/retinal_image.jpg
```

### 15.4 Performance Characteristics

| Metric | Value |
|--------|-------|
| Inference Time | ~50–100ms per image (RTX 4050) |
| Batch Processing | Configurable batch size |
| Memory Usage | ~2 GB GPU RAM during inference |

---

## 16. Hardware Requirements

### 16.1 Tested Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM) |
| CUDA | 12.1+ |
| System RAM | 16 GB |
| Storage | ~20 GB (datasets + checkpoints) |

### 16.2 Minimum Requirements

| Component | Minimum |
|-----------|---------|
| GPU VRAM | 6 GB |
| System RAM | 16 GB |
| CUDA | 12.1+ compatible GPU |
| Disk Space | ~100 GB (all datasets, checkpoints, results) |

### 16.3 Training Performance

| Metric | Value |
|--------|-------|
| Training Time (Phase 2) | ~2 hours (150 epochs) |
| Samples per Second | ~160 (batch_size=16 @ 384×384) |
| Best Model Epoch | ~120 |

---

## 17. Software Dependencies

### 17.1 Core Requirements

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `torchvision` | ≥0.15.0 | Backbone models, transforms |
| `numpy` | ≥1.24.0 | Numerical computation |
| `pandas` | ≥2.0.0 | Data loading, CSV handling |
| `scikit-learn` | ≥1.3.0 | Metrics (F1, ROC-AUC, etc.) |
| `Pillow` | ≥10.0.0 | Image loading/processing |
| `opencv-python` | ≥4.8.0 | CLAHE, image preprocessing |
| `matplotlib` | ≥3.7.0 | Plotting |
| `seaborn` | ≥0.12.0 | Statistical visualization |
| `tqdm` | ≥4.65.0 | Progress bars |
| `PyYAML` | ≥6.0 | Configuration file parsing |

### 17.2 Optional Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `albumentations` | ≥1.3.0 | Advanced augmentation (optional) |
| `timm` | ≥0.9.0 | Additional backbone architectures |

### 17.3 Python Version

- **Required:** Python 3.10–3.12
- **Tested:** Python 3.11

---

## 18. Project File Structure

```
OcuNet/
├── config/
│   └── config.yaml                    # Complete training & model configuration
├── src/
│   ├── __init__.py                    # Package initialization and exports
│   ├── models.py                      # Model architecture definitions
│   │                                   #   - ImprovedMultiLabelModel (main model)
│   │                                   #   - ImprovedClassificationHead
│   │                                   #   - SEBlock (Squeeze-and-Excitation)
│   │                                   #   - AsymmetricLossOptimized
│   │                                   #   - FocalLossMultiLabel
│   │                                   #   - compute_effective_weights()
│   │                                   #   - create_improved_model()
│   │                                   #   - create_improved_loss()
│   │                                   #   - count_parameters()
│   ├── dataset.py                     # Data loading & augmentation
│   │                                   #   - ImprovedMultiLabelDataset
│   │                                   #   - ImprovedDataManager
│   │                                   #   - RandAugment (14 operations)
│   ├── preprocessing.py               # Fundus image preprocessing
│   │                                   #   - FundusROICrop
│   │                                   #   - CLAHETransform
│   ├── train.py                       # Training loop & utilities
│   │                                   #   - ImprovedTrainer
│   │                                   #   - WarmupCosineScheduler
│   │                                   #   - EMA (Exponential Moving Average)
│   │                                   #   - EarlyStopping
│   ├── evaluate.py                    # Evaluation metrics & visualization
│   │                                   #   - MultiLabelEvaluator
│   │                                   #   - MultiLabelMetrics
│   │                                   #   - Plotting functions
│   ├── calibrate.py                   # Probability calibration
│   │                                   #   - TemperatureScaling
│   │                                   #   - optimize_thresholds_calibrated()
│   │                                   #   - calibrate_model()
│   └── utils.py                       # Shared utilities
│                                       #   - set_seed()
│                                       #   - get_device()
│                                       #   - count_parameters()
│                                       #   - AverageMeter
├── tests/
│   └── test_improvements.py           # Unit tests
├── train_pipeline.py                  # Main training entry point
├── predict.py                         # Inference interface
├── setup_datasets.py                  # Dataset organization helper
├── requirements.txt                   # Python dependencies
├── README.md                          # Quick start guide
├── Project Report.md                  # Detailed project documentation
├── Model_Report.md                    # This report
├── Training Log 1.txt                 # Phase 1 training log
├── Training Log 2.txt                 # Phase 1 continued
├── Training Log 3.txt                 # Phase 2 initial
├── Training Log 4.txt                 # Phase 2 continued
├── Training Log 5.txt                 # Phase 2 final (latest)
├── checkpoints/                       # Saved model weights (.pth)
├── evaluation_results/                # Metrics, plots, thresholds
│   ├── optimal_thresholds.yaml        # Per-class decision thresholds
│   ├── calibration.yaml               # Temperature scaling parameters
│   └── *.png                          # Evaluation plots
└── explainability_results/            # Grad-CAM visualizations
    └── *.png
```

---

## 19. Limitations and Future Work

### 19.1 Current Limitations

1. **Augmented data is synthetic:** Phase 3 augmented images are variants of existing clinical data, not independent samples. This limits the effective diversity of training data for rare classes.

2. **Severe class imbalance persists:** Despite reweighting and ASL, extremely rare classes (MHL: 11 samples, RT: 14 samples) remain challenging. Their F1 scores are well below 50%.

3. **No prospective clinical validation:** All evaluation is retrospective on held-out test sets. Clinical deployment requires prospective validation on real-world patient populations.

4. **Single-center data:** The RFMiD dataset and Kaggle datasets may not represent the full diversity of retinal cameras, patient populations, and imaging conditions encountered in practice.

5. **Explainability needs clinician validation:** Grad-CAM visualizations are available but have not been formally validated by ophthalmologists.

6. **Precision-recall trade-off:** High recall for major diseases comes at the cost of lower precision (more false positives), which is acceptable for screening but would need adjustment for diagnosis.

### 19.2 Future Work

1. **Multi-center prospective validation** across different retinal cameras and clinical settings
2. **Active learning** to efficiently annotate rare disease samples
3. **Model distillation** for edge/mobile deployment
4. **Ensemble methods** combining multiple backbones
5. **Temporal analysis** for longitudinal disease progression monitoring
6. **Integration with electronic health records (EHR)** for clinical decision support
7. **Federated learning** to train across institutions without sharing patient data

---

## 20. References

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.

2. Ben-Baruch, E., et al. (2020). *Asymmetric Loss For Multi-Label Classification*. arXiv:2009.14119.

3. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). *Class-Balanced Loss Based on Effective Number of Samples*. CVPR 2019.

4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.

5. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.

6. Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.

7. Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). *Randaugment: Practical Automated Data Augmentation with a Reduced Search Space*. NeurIPS 2020.

8. **Kaggle Eye Diseases Dataset:** https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

9. **RFMiD Dataset:** https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid

---

*Report generated for OcuNet v4.2.0 — March 2026*
