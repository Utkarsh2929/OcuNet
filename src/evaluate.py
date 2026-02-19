#!/usr/bin/env python3
"""
Multi-Label Evaluation Module for OcuNet Phase 2.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, multilabel_confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve, auc,
    f1_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Suppress undefined metric warnings in sklearn for rare classes
warnings.filterwarnings("ignore", "No positive class found in y_true, recall is set to one for all thresholds.")


class MultiLabelMetrics:
    """Compute metrics for multi-label classification."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []

    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        """Update with batch predictions."""
        self.all_probs.append(probs.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
        self.all_preds.append((probs >= self.threshold).float().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        probs = np.vstack(self.all_probs)
        targets = np.vstack(self.all_targets)
        preds = np.vstack(self.all_preds)

        metrics = {}

        # Sample-wise metrics
        metrics['exact_match'] = (preds == targets).all(axis=1).mean()

        # Macro F1
        try:
            metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
            metrics['f1_micro'] = f1_score(targets, preds, average='micro', zero_division=0)
            metrics['f1_samples'] = f1_score(targets, preds, average='samples', zero_division=0)
        except:
            metrics['f1_macro'] = 0.0
            metrics['f1_micro'] = 0.0
            metrics['f1_samples'] = 0.0

        # Precision and Recall
        try:
            metrics['precision_macro'] = precision_score(targets, preds, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(targets, preds, average='macro', zero_division=0)
        except:
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0

        # mAP (mean Average Precision)
        try:
            metrics['mAP'] = average_precision_score(targets, probs, average='macro')
        except:
            metrics['mAP'] = 0.0

        # ROC-AUC
        try:
            valid_classes = []
            for i in range(targets.shape[1]):
                if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                    valid_classes.append(i)

            if valid_classes:
                metrics['roc_auc_macro'] = roc_auc_score(
                    targets[:, valid_classes],
                    probs[:, valid_classes],
                    average='macro'
                )
            else:
                metrics['roc_auc_macro'] = 0.0
        except:
            metrics['roc_auc_macro'] = 0.0

        # Hamming Loss (lower is better)
        metrics['hamming_loss'] = (preds != targets).mean()

        return metrics

    def compute_per_class(self, class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics."""
        probs = np.vstack(self.all_probs)
        targets = np.vstack(self.all_targets)
        preds = np.vstack(self.all_preds)

        per_class = {}

        for i, name in enumerate(class_names):
            class_metrics = {}

            try:
                class_metrics['precision'] = precision_score(targets[:, i], preds[:, i], zero_division=0)
                class_metrics['recall'] = recall_score(targets[:, i], preds[:, i], zero_division=0)
                class_metrics['f1'] = f1_score(targets[:, i], preds[:, i], zero_division=0)
            except:
                class_metrics['precision'] = 0.0
                class_metrics['recall'] = 0.0
                class_metrics['f1'] = 0.0

            try:
                if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                    class_metrics['roc_auc'] = roc_auc_score(targets[:, i], probs[:, i])
                    class_metrics['ap'] = average_precision_score(targets[:, i], probs[:, i])
                else:
                    class_metrics['roc_auc'] = 0.0
                    class_metrics['ap'] = 0.0
            except:
                class_metrics['roc_auc'] = 0.0
                class_metrics['ap'] = 0.0

            class_metrics['support'] = int(targets[:, i].sum())

            per_class[name] = class_metrics

        return per_class


class MultiLabelEvaluator:
    """Comprehensive evaluator for multi-label classification."""

    def __init__(
            self,
            model: nn.Module,
            test_loader: DataLoader,
            class_names: List[str],
            device: torch.device,
            output_dir: str = "evaluation_results",
            threshold: float = 0.5
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

        self.num_classes = len(class_names)

        # Storage
        self.all_probs = None
        self.all_preds = None
        self.all_targets = None

    @torch.no_grad()
    def collect_predictions(self):
        """Collect all predictions from test set."""
        print("Collecting predictions...")

        all_probs = []
        all_targets = []

        for images, labels in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)

            outputs = self.model(images)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        self.all_probs = np.vstack(all_probs)
        self.all_targets = np.vstack(all_targets)
        self.all_preds = (self.all_probs >= self.threshold).astype(np.float32)

        print(f"Collected {len(self.all_probs)} samples")

    def compute_metrics(self) -> Dict:
        """Compute comprehensive metrics."""
        if self.all_probs is None:
            self.collect_predictions()

        metrics = MultiLabelMetrics(self.threshold)
        metrics.all_probs = [self.all_probs]
        metrics.all_targets = [self.all_targets]
        metrics.all_preds = [self.all_preds]

        overall = metrics.compute()
        per_class = metrics.compute_per_class(self.class_names)

        return {'overall': overall, 'per_class': per_class}

    def plot_multilabel_confusion_matrices(self):
        """Plot confusion matrix for each class."""
        if self.all_preds is None:
            self.collect_predictions()

        # Compute confusion matrices
        mcm = multilabel_confusion_matrix(self.all_targets, self.all_preds)

        # Determine grid size
        n_classes = len(self.class_names)
        n_cols = 5
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, (cm, name) in enumerate(zip(mcm, self.class_names)):
            ax = axes[i]

            # Normalize
            cm_norm = cm.astype('float') / cm.sum()

            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                        xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
            ax.set_title(f'{name}', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide empty subplots
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Per-Class Confusion Matrices', fontsize=14)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'confusion_matrices_multilabel.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: confusion_matrices_multilabel.png")

    def plot_precision_recall_curves(self):
        """Plot precision-recall curves for each class."""
        if self.all_probs is None:
            self.collect_predictions()

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab20(np.linspace(0, 1, len(self.class_names)))

        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            if self.all_targets[:, i].sum() == 0:
                continue

            precision, recall, _ = precision_recall_curve(self.all_targets[:, i], self.all_probs[:, i])
            ap = average_precision_score(self.all_targets[:, i], self.all_probs[:, i])

            ax.plot(recall, precision, color=color, lw=2, label=f'{name} (AP={ap:.3f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / 'precision_recall_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: precision_recall_curves.png")

    def plot_roc_curves(self):
        """Plot ROC curves for each class."""
        if self.all_probs is None:
            self.collect_predictions()

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab20(np.linspace(0, 1, len(self.class_names)))

        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            if self.all_targets[:, i].sum() == 0 or self.all_targets[:, i].sum() == len(self.all_targets):
                continue

            fpr, tpr, _ = roc_curve(self.all_targets[:, i], self.all_probs[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / 'roc_curves_multilabel.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: roc_curves_multilabel.png")

    def plot_class_distribution(self):
        """Plot class distribution in predictions vs ground truth."""
        if self.all_preds is None:
            self.collect_predictions()

        gt_counts = self.all_targets.sum(axis=0)
        pred_counts = self.all_preds.sum(axis=0)

        x = np.arange(len(self.class_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))

        bars1 = ax.bar(x - width / 2, gt_counts, width, label='Ground Truth', color='steelblue')
        bars2 = ax.bar(x + width / 2, pred_counts, width, label='Predicted', color='coral')

        ax.set_xlabel('Disease Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution: Ground Truth vs Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(self.output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: class_distribution.png")

    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        metrics = self.compute_metrics()

        report_lines = [
            "=" * 70,
            "MULTI-LABEL CLASSIFICATION - EVALUATION REPORT",
            "=" * 70,
            "",
            "[OVERALL METRICS]",
            "-" * 50,
            f"F1 Score (Macro):     {metrics['overall']['f1_macro']:.4f}",
            f"F1 Score (Micro):     {metrics['overall']['f1_micro']:.4f}",
            f"F1 Score (Samples):   {metrics['overall']['f1_samples']:.4f}",
            f"Precision (Macro):    {metrics['overall']['precision_macro']:.4f}",
            f"Recall (Macro):       {metrics['overall']['recall_macro']:.4f}",
            f"mAP:                  {metrics['overall']['mAP']:.4f}",
            f"ROC-AUC (Macro):      {metrics['overall']['roc_auc_macro']:.4f}",
            f"Hamming Loss:         {metrics['overall']['hamming_loss']:.4f}",
            f"Exact Match Ratio:    {metrics['overall']['exact_match']:.4f}",
            "",
            "[PER-CLASS METRICS]",
            "-" * 50,
            f"{'Class':<25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP':>8} {'AUC':>8} {'Support':>8}",
            "-" * 80,
        ]

        # Sort by support (descending)
        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )

        for name, m in sorted_classes:
            report_lines.append(
                f"{name:<25} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
                f"{m['ap']:>8.3f} {m['roc_auc']:>8.3f} {m['support']:>8}"
            )

        report_lines.extend([
            "",
            "[CLINICAL SENSITIVITY ANALYSIS]",
            "-" * 50,
            "(Higher recall = fewer missed disease cases)",
            ""
        ])

        for name, m in sorted_classes:
            if m['support'] > 0:
                if m['recall'] >= 0.9:
                    status = "[EXCELLENT]"
                elif m['recall'] >= 0.8:
                    status = "[GOOD]"
                elif m['recall'] >= 0.7:
                    status = "[MODERATE]"
                else:
                    status = "[NEEDS IMPROVEMENT]"
                report_lines.append(f"  {name}: {m['recall']:.4f} {status}")

        report_lines.extend(["", "=" * 70])

        report = "\n".join(report_lines)

        # Save report
        with open(self.output_dir / 'evaluation_report_multilabel.txt', 'w') as f:
            f.write(report)

        # Save metrics as JSON
        # Convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        metrics_serializable = convert_to_serializable(metrics)
        with open(self.output_dir / 'metrics_multilabel.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        print(f"Saved: evaluation_report_multilabel.txt")
        print(f"Saved: metrics_multilabel.json")

        return report

    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation pipeline."""
        print("\n" + "=" * 70)
        print("RUNNING MULTI-LABEL EVALUATION")
        print("=" * 70)

        # Collect predictions
        self.collect_predictions()

        # Compute metrics
        metrics = self.compute_metrics()

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_multilabel_confusion_matrices()
        self.plot_precision_recall_curves()
        self.plot_roc_curves()
        self.plot_class_distribution()

        # Generate report
        print("\nGenerating report...")
        report = self.generate_report()
        print("\n" + report)

        print(f"\nAll results saved to: {self.output_dir}")

        return metrics


def plot_multilabel_training_history(history: Dict, output_dir: str = "evaluation_results"):
    """Plot training history for multi-label classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation', color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[0, 1].plot(history['train_f1'], label='Train', color='blue')
    axes[0, 1].plot(history['val_f1'], label='Validation', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score (Macro)')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # mAP
    axes[1, 0].plot(history['train_mAP'], label='Train', color='blue')
    axes[1, 0].plot(history['val_mAP'], label='Validation', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('Mean Average Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(history['learning_rate'], color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.suptitle('Multi-Label Training History', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'training_history_multilabel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: training_history_multilabel.png")