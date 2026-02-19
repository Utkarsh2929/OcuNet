#!/usr/bin/env python3
"""
Temperature Scaling and Calibrated Threshold Optimization for OcuNet Phase 2.

Temperature scaling learns a single scalar T that divides logits before sigmoid,
calibrating predicted probabilities so they better reflect true likelihoods.
This makes per-class threshold optimization far more effective.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import yaml
import warnings

# Suppress undefined metric warnings in sklearn for rare classes
warnings.filterwarnings("ignore", "No positive class found in y_true, recall is set to one for all thresholds.")


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.

    Learns a single temperature parameter T > 0 such that
    calibrated_prob = sigmoid(logit / T).

    When T > 1, probabilities are pushed toward 0.5 (less confident).
    When T < 1, probabilities are pushed toward 0 or 1 (more confident).
    """

    def __init__(self):
        super().__init__()
        # Initialize temperature to 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities."""
        return torch.sigmoid(self.forward(logits))


def collect_logits(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw logits and targets from data loader."""
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Collecting logits"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_targets)


def fit_temperature(
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200
) -> TemperatureScaling:
    """
    Fit temperature parameter using NLL loss on validation set.

    Args:
        logits: Raw model logits (N, C)
        targets: Ground truth labels (N, C)
        lr: Learning rate for optimization
        max_iter: Maximum iterations

    Returns:
        Fitted TemperatureScaling module
    """
    temp_model = TemperatureScaling()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_model(logits)
        # BCE with logits loss for multi-label
        loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Ensure temperature is positive
    with torch.no_grad():
        temp_model.temperature.clamp_(min=0.01)

    return temp_model


def optimize_thresholds_calibrated(
        probs: np.ndarray,
        targets: np.ndarray,
        class_names: List[str],
        strategy: str = "f1",
        min_recall: float = 0.8
) -> Dict[str, Dict]:
    """
    Optimize per-class thresholds on calibrated probabilities.

    Args:
        probs: Calibrated probabilities (N, C)
        targets: Ground truth labels (N, C)
        class_names: List of class names
        strategy: "f1" for max F1, or "precision_at_recall" for
                  best precision while maintaining recall >= min_recall
        min_recall: Minimum recall constraint (for precision_at_recall strategy)

    Returns:
        Dict mapping class_name -> {threshold, f1, precision, recall}
    """
    thresholds = {}

    for i, name in enumerate(class_names):
        best_threshold = 0.5
        best_score = -1.0
        best_metrics = {}

        for threshold in np.arange(0.05, 0.95, 0.025):
            preds = (probs[:, i] >= threshold).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            prec = precision_score(targets[:, i], preds, zero_division=0)
            rec = recall_score(targets[:, i], preds, zero_division=0)

            if strategy == "f1":
                score = f1
            elif strategy == "precision_at_recall":
                # Maximize precision subject to recall >= min_recall
                score = prec if rec >= min_recall else -1.0
            else:
                score = f1

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': float(round(threshold, 3)),
                    'f1': float(round(f1, 4)),
                    'precision': float(round(prec, 4)),
                    'recall': float(round(rec, 4)),
                }

        # Fallback: if no threshold meets recall constraint, use max-F1
        if best_score < 0:
            best_metrics = _fallback_f1_threshold(probs[:, i], targets[:, i])

        thresholds[name] = best_metrics

    return thresholds


def _fallback_f1_threshold(probs_col, targets_col):
    """Fallback: find threshold maximizing F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.025):
        preds = (probs_col >= t).astype(int)
        f1 = f1_score(targets_col, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    preds = (probs_col >= best_t).astype(int)
    prec = precision_score(targets_col, preds, zero_division=0)
    rec = recall_score(targets_col, preds, zero_division=0)
    return {
        'threshold': float(round(best_t, 3)),
        'f1': float(round(best_f1, 4)),
        'precision': float(round(prec, 4)),
        'recall': float(round(rec, 4)),
    }


def calibrate_model(
        model: nn.Module,
        val_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        output_dir: str = "evaluation_results",
        strategy: str = "f1",
        min_recall: float = 0.8
) -> Tuple[float, Dict]:
    """
    Full calibration pipeline:
    1. Collect logits from validation set
    2. Fit temperature parameter
    3. Re-optimize thresholds on calibrated probabilities
    4. Save temperature and thresholds

    Args:
        model: Trained model
        val_loader: Validation data loader
        class_names: List of class names
        device: torch device
        output_dir: Where to save results
        strategy: Threshold optimization strategy
        min_recall: Minimum recall for precision_at_recall strategy

    Returns:
        (temperature_value, optimized_thresholds)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect logits
    print("Collecting logits for calibration...")
    logits, targets = collect_logits(model, val_loader, device)

    # Step 2: Fit temperature
    print("Fitting temperature parameter...")
    temp_model = fit_temperature(logits, targets)
    temperature = temp_model.temperature.item()
    print(f"  Learned temperature: {temperature:.4f}")

    # Step 3: Get calibrated probabilities
    with torch.no_grad():
        calibrated_probs = temp_model.calibrate(logits).numpy()
    uncalibrated_probs = torch.sigmoid(logits).numpy()
    targets_np = targets.numpy()

    # Step 4: Optimize thresholds on calibrated probabilities
    print(f"\nOptimizing thresholds (strategy={strategy})...")
    calibrated_thresholds = optimize_thresholds_calibrated(
        calibrated_probs, targets_np, class_names, strategy, min_recall
    )

    # Also get uncalibrated thresholds for comparison
    uncalibrated_thresholds = optimize_thresholds_calibrated(
        uncalibrated_probs, targets_np, class_names, "f1"
    )

    # Print comparison
    print(f"\n{'Class':<25} {'Uncalib F1':>12} {'Calib F1':>12} {'Δ F1':>8} {'Threshold':>10}")
    print("-" * 70)
    for name in class_names:
        uf1 = uncalibrated_thresholds[name]['f1']
        cf1 = calibrated_thresholds[name]['f1']
        delta = cf1 - uf1
        t = calibrated_thresholds[name]['threshold']
        marker = " ✓" if delta > 0 else ""
        print(f"  {name:<23} {uf1:>12.4f} {cf1:>12.4f} {delta:>+8.4f} {t:>10.3f}{marker}")

    # Compute overall metrics
    print("\nOverall metrics with calibrated thresholds:")
    cal_preds = np.zeros_like(calibrated_probs)
    uncal_preds = np.zeros_like(uncalibrated_probs)
    for i, name in enumerate(class_names):
        cal_preds[:, i] = (calibrated_probs[:, i] >= calibrated_thresholds[name]['threshold']).astype(int)
        uncal_preds[:, i] = (uncalibrated_probs[:, i] >= uncalibrated_thresholds[name]['threshold']).astype(int)

    cal_f1_macro = f1_score(targets_np, cal_preds, average='macro', zero_division=0)
    cal_f1_micro = f1_score(targets_np, cal_preds, average='micro', zero_division=0)
    cal_prec = precision_score(targets_np, cal_preds, average='macro', zero_division=0)

    uncal_f1_macro = f1_score(targets_np, uncal_preds, average='macro', zero_division=0)
    uncal_f1_micro = f1_score(targets_np, uncal_preds, average='micro', zero_division=0)
    uncal_prec = precision_score(targets_np, uncal_preds, average='macro', zero_division=0)

    print(f"  {'Metric':<25} {'Uncalibrated':>14} {'Calibrated':>14} {'Δ':>8}")
    print(f"  {'-'*65}")
    print(f"  {'F1 (Macro)':<25} {uncal_f1_macro:>14.4f} {cal_f1_macro:>14.4f} {cal_f1_macro - uncal_f1_macro:>+8.4f}")
    print(f"  {'F1 (Micro)':<25} {uncal_f1_micro:>14.4f} {cal_f1_micro:>14.4f} {cal_f1_micro - uncal_f1_micro:>+8.4f}")
    print(f"  {'Precision (Macro)':<25} {uncal_prec:>14.4f} {cal_prec:>14.4f} {cal_prec - uncal_prec:>+8.4f}")

    # Step 5: Save results
    calibration_data = {
        'temperature': float(temperature),
        'strategy': strategy,
        'min_recall': float(min_recall),
        'thresholds': calibrated_thresholds,
        'metrics': {
            'calibrated_f1_macro': float(cal_f1_macro),
            'calibrated_f1_micro': float(cal_f1_micro),
            'calibrated_precision_macro': float(cal_prec),
            'uncalibrated_f1_macro': float(uncal_f1_macro),
            'uncalibrated_f1_micro': float(uncal_f1_micro),
            'uncalibrated_precision_macro': float(uncal_prec),
        }
    }

    cal_path = output_path / 'calibration.yaml'
    with open(cal_path, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    print(f"\nSaved: {cal_path}")

    return temperature, calibrated_thresholds
