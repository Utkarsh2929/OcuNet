# OcuNet

A comprehensive, deep learning-based system for automated multi-label classification of retinal fundus images. OcuNet detects **28 eye disease classes** simultaneously using a highly optimized EfficientNet-B3 architecture with transfer learning, designed specificially for clinical reliability, interpretability, and robust performance.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Clinical Significance](#clinical-significance)
3. [Key Achievements](#key-achievements)
4. [Model Architecture](#model-architecture)
5. [Clinical Sensitivity Analysis](#clinical-sensitivity-analysis)
6. [Explainability (Grad-CAM)](#explainability-grad-cam)
7. [Quick Start & Installation](#quick-start--installation)
8. [Usage Guide](#usage-guide)
9. [Supported Diseases](#supported-diseases-28-classes)
10. [Documentation & Research](#documentation--research)

---

## Executive Summary

OcuNet represents a two-phase evolution in retinal pathology classification:
- **Phase 1:** Single-label classification into 4 fundamental categories (Cataract, Diabetic Retinopathy, Glaucoma, Normal) reaching **97.88% ROC-AUC**.
- **Phase 2:** Massive scale-up to **multi-label classification predicting 28 distinct conditions** simultaneously, handling complex co-morbidities natively with **93.68% ROC-AUC**.

The system utilizes Asymmetric Loss (ASL), calibrated probabilities, and intelligent preprocessing to solve extreme class imbalances common in medical datasets. 

---

## Clinical Significance

Devastating ocular pathologies consistently rank as one of the leading global causes contributing to irreversible blindness. Manual examination of retinal images requires specialized medical expertise, is time-consuming, and completely unscalable for mass screening programs (specifically in rural demographics). 

OcuNet aims to bridge this gap by providing an accurate, unbiased, and natively interpretable tool capable of mapping exact physiological regions (Grad-CAM) for disease detection, acting as a crucial triage system.

---

## Key Achievements

With the incorporation of Per-Class Threshold Optimization, OcuNet achieves the following metrics on the **Phase 2 (28-class)** dataset:

| Metric | Value |
|--------|-------|
| **ROC-AUC (Macro)** | 93.68% |
| **mAP (Mean Average Precision)** | 53.83% |
| **F1 Score (Micro, Optimized)** | 73.53% |
| **F1 Score (Macro, Optimized)** | 54.93% |
| **Hamming Loss** | 0.1111 |

---

## Model Architecture

The core of OcuNet relies on a dynamically scalable **EfficientNet-B3** backbone (~12M parameters, 384x384 resolution) enhanced with an improved Multi-Label Classification Head.

### Key Architectural Improvements:
- **Squeeze-and-Excitation (SE) Interception:** We route the flattened 1536-dimensional feature vector across SE layers to dynamically multiply absolute channel weighting.
- **Heavy Dimensional MLP:** A cascading configuration (1536 → 512 → 256 → 28). Each tier strictly utilizes Batch Normalization + Gaussian Error Linear Unit (GELU) to stabilize extreme derivations.
- **Independent Sigmoids:** Softmax is completely removed in favor of 28 isolated Sigmoid activation functions, mapping distinct $0 \rightarrow 1$ limit thresholds seamlessly per disease class.

### Training Optimizations:
- **Asymmetric Loss (ASL)** with $\gamma_{neg}=6, \gamma_{pos}=1, clip=0.1$ to aggressively down-weight easy background negatives.
- **Learning Rate Warmup & Cosine Annealing**
- **Exponential Moving Average (EMA)** (decay=0.999) for stable parameter evaluations.

---

## Clinical Sensitivity Analysis

For medical diagnostic systems, **sensitivity (recall)** is critical to minimize missed positive cases.

| Disease | Recall | Clinical Target (≥80%) |
|---------|--------|------------------------|
| **Cataract** | 100.0% | Excellent |
| **Macular Hole** | 100.0% | Excellent |
| **Glaucoma** | 96.3% | Excellent |
| **Diabetic Retinopathy** | 96.2% | Excellent |
| **Diabetic Nephropathy** | 95.7% | Excellent |
| **Age-Related Macular Degeneration** | 90.3% | Excellent |

_Summary: 17 out of 28 critical classes achieve the ≥80% clinical sensitivity threshold, making OcuNet viable for preliminary screenings._

---

## Explainability (Grad-CAM)

OcuNet ships with **Gradient-weighted Class Activation Mapping (Grad-CAM)** specifically mapping the exact structural pixel derivations the model views. 

- **Cataract:** Highlights the central lens region and opaque/hazy areas perfectly.
- **Diabetic Retinopathy:** Focuses specifically on the macula and distinct potential hemorrhages/exudates.
- **Glaucoma:** Actively isolates the optic disc and cup region precisely diagnosing neuroretinal rim thinning.
- **Normal:** Model diffuses focus across broad healthy vascular patterns, validating comprehensive health confidently.

---

## Quick Start & Installation

### 1. Environment Setup

*GPU with 6GB+ VRAM (e.g. RTX 4050 or higher) and 16GB RAM is highly recommended.*

```bash
# Create virtual environment 
python -m venv venv
# Activate environment
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

# Install PyTorch with CUDA (adjust to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. Downloading the Models (Git LFS)
OcuNet uses **Git Large File Storage (LFS)** for model weight files (`.pth`). Make sure it is installed.

```bash
# Install Git LFS
git lfs install

# Pull the latest LFS files (including models/ocunetv4.pth)
git lfs pull
```

### 3. Setting Up the Dataset

To replicate the training experiments, download the datasets and place them in the correct directories:

**Phase 1 Dataset:** Download from [Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) to `data/dataset/`
**Phase 2 Dataset (RFMiD):** Download from [IEEE Dataport](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid) to `data/rfmid/`
**Phase 3 Dataset:** Augmented images mapped to `data/Augmented Dataset/`

Verify correct directory creation:
```bash
python setup_datasets.py verify
```

---

## Usage Guide

### Prediction / Inference

You can run predictions on new, unseen retinal images right from the command line:

```bash
python predict.py path/to/image.jpg
```

Or consume the API programmatically:
```python
from predict import ImprovedMultiLabelClassifier

# Loads the LFS ocunetv4.pth weights cleanly
classifier = ImprovedMultiLabelClassifier()
result = classifier.predict("path/to/retinal_image.jpg")

print("Detected Diseases:", result['diseases'])          
print("Probabilities:", result['probabilities'])     

# Generate a detailed clinical text report automatically
classifier.generate_report(result, output_path="report.txt")
```

### Training Pipeline Automation

The training framework is fully configurable via `config/config.yaml`.

```bash
# Execute the full pipeline: Train -> Evaluate -> Optimize Thresholds -> Calibrate Letencies
python train_pipeline.py --mode all

# Run specific functional blocks:
python train_pipeline.py --mode train      # Training only
python train_pipeline.py --mode evaluate   # Validation evaluation only
python train_pipeline.py --mode optimize   # Dynamic probability threshold optimizer
```

---

## Supported Diseases (28 Classes)

| # | Code | Full Disease Name | # | Code | Full Disease Name |
|---|------|-------------------|---|------|-------------------|
| 0 | `Disease_Risk` | General Disease Risk | 14 | `AH` | Asteroid Hyalosis |
| 1 | `DR` | Diabetic Retinopathy | 15 | `ODP` | Optic Disc Pallor |
| 2 | `ARMD` | Age-Related Macular Degeneration | 16 | `ODE` | Optic Disc Edema |
| 3 | `MH` | Macular Hole | 17 | `AION` | Anterior Ischemic Optic Neuropathy |
| 4 | `DN` | Diabetic Nephropathy | 18 | `PT` | Pigment Changes |
| 5 | `MYA` | Myopia | 19 | `RT` | Retinitis |
| 6 | `BRVO` | Branch Retinal Vein Occlusion | 20 | `RS` | Retinal Scars |
| 7 | `TSLN` | Tessellation | 21 | `CRS` | Chorioretinal Scars |
| 8 | `ERM` | Epiretinal Membrane | 22 | `EDN` | Macular Edema |
| 9 | `LS` | Laser Scars | 23 | `RPEC` | RPE Changes |
| 10 | `MS` | Maculopathy | 24 | `MHL` | Macular Holes (Large) |
| 11 | `CSR` | Central Serous Retinopathy | 25 | `CATARACT` | Cataract |
| 12 | `ODC` | Optic Disc Cupping | 26 | `GLAUCOMA` | Glaucoma |
| 13 | `CRVO` | Central Retinal Vein Occlusion | 27 | `NORMAL` | Normal/Healthy |

---

## Documentation & Research

For an expansive deep dive into validation methodologies, ablation studies, false positive error diagnostics, or layer-by-layer architectures, please refer to the fully drafted **[Project Report.md](Project%20Report.md)** natively housed in the repo.

---

## License & Author

**Author:** Utkarsh Gautam  
**Updated:** 2026

This open-source medical diagnostics project is licensed under the MIT License. Clinical deployment demands institutional board oversight and medical confirmation checks unconditionally.

---

### *References*
1. *Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.*
2. *Ben-Baruch, E., et al. (2020). Asymmetric Loss for Multi-Label Classification. arXiv.*
3. *[RFMiD Dataset: Retinal Fundus Multi-Disease Image Dataset](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)*

