"""
Utility functions for the eye disease classification pipeline.
"""

import os
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> torch.device:
    """Get the best available device with detailed GPU info."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Using GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")

        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
        print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU (GPU not available)")
        print("   Tip: Install CUDA toolkit for GPU acceleration")

        # Check if CUDA is supposed to be available
        if torch.cuda.device_count() == 0:
            print("   No CUDA devices found. Check your NVIDIA drivers.")

    return device


def check_gpu_status():
    """Print detailed GPU status information."""
    print("=" * 50)
    print("GPU STATUS CHECK")
    print("=" * 50)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
    else:
        print("\nNo CUDA GPUs available.")
        print("Possible reasons:")
        print("  1. NVIDIA GPU not installed")
        print("  2. NVIDIA drivers not installed")
        print("  3. CUDA toolkit not installed")
        print("  4. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    print("=" * 50)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """Print model summary."""
    params = count_parameters(model)
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total Parameters:     {params['total']:,}")
    print(f"Trainable Parameters: {params['trainable']:,}")
    print(f"Frozen Parameters:    {params['frozen']:,}")
    print("=" * 50)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def load_model(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint