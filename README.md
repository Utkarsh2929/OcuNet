# OcuNet 🔬

A deep learning system for automated multi-label classification of retinal fundus images, detecting **28 eye disease classes** simultaneously using EfficientNet-B3 with transfer learning.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **ROC-AUC (Macro)** | 93.68% |
| **mAP (Mean Average Precision)** | 53.83% |
| **F1 Score (Micro)** | 52.96% |
| **Hamming Loss** | 0.1111 |

#### Key Disease Detection (Recall)

| Disease | Sensitivity | Status |
|---------|-------------|--------|
| Cataract | 100.0% | ✅ Excellent |
| Diabetic Retinopathy | 96.2% | ✅ Excellent |
| Glaucoma | 96.3% | ✅ Excellent |
| Age-Related Macular Degeneration | 90.3% | ✅ Excellent |
| Macular Hole | 100.0% | ✅ Excellent |
| Myopia | 93.8% | ✅ Excellent |

---

## 🚀 Quick Start

### 1. Setup

```bash
# Create virtual environment (Python 3.10-3.12)
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

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

**Phase 3 Dataset (Augmented Dataset):** Place the supplementary augmented dataset for underrepresented diseases in `data/Augmented Dataset/`.

Use `setup_datasets.py` to verify dataset setup:
```bash
python setup_datasets.py verify
```

### 3. Training

```bash
# Full pipeline: train + evaluate + threshold optimization
python train_pipeline.py --mode all

# Train only
python train_pipeline.py --mode train

# Evaluate only (from checkpoint)
python train_pipeline.py --mode evaluate

# Optimize per-class thresholds
python train_pipeline.py --mode optimize
```

### 4. Prediction

```python
from predict import ImprovedMultiLabelClassifier

classifier = ImprovedMultiLabelClassifier()
result = classifier.predict("path/to/retinal_image.jpg")

print(result['diseases'])          # ['DR', 'ARMD']
print(result['probabilities'])     # {'DR': 0.85, 'ARMD': 0.72, ...}

# Generate a detailed medical report
classifier.generate_report(result, output_path="report.txt")
```

**Command Line:**
```bash
python predict.py path/to/image.jpg
```

---

## 📁 Project Structure

```
OcuNet/
├── config/
│   └── config.yaml              # Training & model configuration
├── data/
│   ├── dataset/                 # Phase 1 dataset (4 classes)
│   ├── rfmid/                   # Phase 2 RFMiD dataset (28 classes)
│   └── Augmented Dataset/       # Phase 3 Augmented dataset
├── src/
│   ├── __init__.py              # Package exports
│   ├── dataset.py               # Data loading, augmentation, oversampling
│   ├── models.py                # EfficientNet-B3 model architecture
│   ├── train.py                 # Training loop with EMA, warmup, cosine LR
│   ├── evaluate.py              # Evaluation metrics, plots, reports
│   └── utils.py                 # Utility functions
├── checkpoints/                 # Saved model weights
├── evaluation_results/          # Metrics, plots, reports
├── train_pipeline.py            # Main training entry point
├── predict.py                   # Prediction interface
├── setup_datasets.py            # Dataset setup & verification
├── requirements.txt             # Dependencies
├── Project Report.md            # Detailed project documentation
└── Training Log *.txt           # Training run logs
```

---

## 🏗️ Model Architecture

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 + Improved Classification Head |
| Parameters | ~12M trainable |
| Input Size | 384 × 384 × 3 |
| Output Classes | 28 |
| Loss Function | Asymmetric Loss (ASL) |
| Features | EMA, Warmup, RandAugment, Torch Compile |

---

## 🏷️ Supported Disease Classes (28)

| # | Code | Disease |
|---|------|---------|
| 0 | Disease_Risk | General disease risk indicator |
| 1 | DR | Diabetic Retinopathy |
| 2 | ARMD | Age-Related Macular Degeneration |
| 3 | MH | Macular Hole |
| 4 | DN | Diabetic Nephropathy |
| 5 | MYA | Myopia |
| 6 | BRVO | Branch Retinal Vein Occlusion |
| 7 | TSLN | Tessellation |
| 8 | ERM | Epiretinal Membrane |
| 9 | LS | Laser Scars |
| 10 | MS | Maculopathy |
| 11 | CSR | Central Serous Retinopathy |
| 12 | ODC | Optic Disc Cupping |
| 13 | CRVO | Central Retinal Vein Occlusion |
| 14 | AH | Asteroid Hyalosis |
| 15 | ODP | Optic Disc Pallor |
| 16 | ODE | Optic Disc Edema |
| 17 | AION | Anterior Ischemic Optic Neuropathy |
| 18 | PT | Pigment Changes |
| 19 | RT | Retinitis |
| 20 | RS | Retinal Scars |
| 21 | CRS | Chorioretinitis Scars |
| 22 | EDN | Macular Edema |
| 23 | RPEC | RPE Changes |
| 24 | MHL | Macular Holes (Large) |
| 25 | CATARACT | Cataract |
| 26 | GLAUCOMA | Glaucoma |
| 27 | NORMAL | Normal/Healthy |

---

## 📈 Training Details

### Hardware Requirements
- GPU with 6GB+ VRAM (tested on NVIDIA RTX 4050)
- 16GB+ RAM recommended
- CUDA 12.1+

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 (Gradient Accumulation = 1) |
| Max Epochs | 200 |
| Learning Rate | 3e-4 with warmup |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Early Stopping | 30 epochs |
| Mixed Precision | ✅ FP16 |
| Compilation | ✅ `torch.compile` |

---

## 🛠️ Key Features

- **Asymmetric Loss (ASL)**: Better handling of class imbalance in multi-label setting
- **Exponential Moving Average (EMA)**: Improved model stability
- **Learning Rate Warmup**: Gradual learning rate increase for better convergence
- **RandAugment**: Strong data augmentation for regularization
- **Class Oversampling**: Automatic oversampling for rare disease classes
- **Per-Class Threshold Optimization**: Optimized decision thresholds for each disease
- **Squeeze-and-Excitation**: Channel attention in the classification head

---

## 📋 Requirements

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

---

## 📚 Documentation

For detailed technical documentation, see [Project Report.md](Project%20Report.md).

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Utkarsh Gautam**  
January 2026

---

## 📖 References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
2. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
3. Ben-Baruch, E., et al. (2020). Asymmetric Loss for Multi-Label Classification. arXiv.
4. [Kaggle Eye Diseases Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
5. [RFMiD Dataset](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)
