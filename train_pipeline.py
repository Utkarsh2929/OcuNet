#!/usr/bin/env python3
"""
OcuNet Phase 2 - Improved Multi-Label Training Pipeline.
"""

import argparse
from pathlib import Path
import yaml
import random
import numpy as np
import torch

from src.dataset import ImprovedDataManager
from src.models import create_improved_model, create_improved_loss, count_parameters
from src.train import ImprovedTrainer
from src.evaluate import MultiLabelEvaluator, plot_multilabel_training_history
from src.calibrate import calibrate_model as run_calibration


def set_seed(seed: int = 42):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """Load config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Get device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train(config: dict):
    """Train improved model."""
    print("\n" + "=" * 70)
    print("OCUNET PHASE 2 - IMPROVED TRAINING")
    print("=" * 70)

    set_seed(config['dataset']['random_seed'])
    device = get_device()

    # Load data
    print("\nLoading data with oversampling...")
    data_manager = ImprovedDataManager(config)
    train_loader, val_loader, test_loader, pos_weights, class_names = data_manager.create_data_loaders()

    num_classes = len(class_names)
    print(f"\nClasses: {num_classes}")

    # Create model
    print("\nCreating improved model...")
    model = create_improved_model(config, num_classes)
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Create loss
    criterion = create_improved_loss(config, pos_weights, device)
    print(f"Loss: {criterion.__class__.__name__}")

    # Train
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
        class_names=class_names
    )

    history = trainer.train()

    # Plot history
    plot_multilabel_training_history(history, config['output']['results_dir'])

    return model, test_loader, class_names


def evaluate(config: dict, model=None, test_loader=None, class_names=None, checkpoint_path=None):
    """Evaluate model."""
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    device = get_device()

    # Load model if not provided
    if model is None:
        ckpt_path = checkpoint_path or Path(config['output']['checkpoint_dir']) / 'best_model.pth'
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        class_names = checkpoint.get('class_names', [])
        num_classes = len(class_names)

        model = create_improved_model(config, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded: {ckpt_path}")
        print(f"  Best F1: {checkpoint.get('best_val_f1', 'N/A')}")
        print(f"  Classes: {num_classes}")

    model = model.to(device)

    # Load test data if not provided
    if test_loader is None or class_names is None:
        data_manager = ImprovedDataManager(config)
        _, _, test_loader, _, class_names = data_manager.create_data_loaders()

    threshold = config['training'].get('threshold', 0.5)

    # Run evaluation
    evaluator = MultiLabelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=config['output']['results_dir'],
        threshold=threshold
    )

    metrics = evaluator.run_full_evaluation()

    return metrics


def optimize_thresholds(config: dict, checkpoint_path: str = None):
    """Find optimal per-class thresholds."""
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)

    device = get_device()

    # Load model
    ckpt_path = checkpoint_path or Path(config['output']['checkpoint_dir']) / 'best_model.pth'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_names = checkpoint.get('class_names', [])
    num_classes = len(class_names)

    model = create_improved_model(config, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load validation data
    data_manager = ImprovedDataManager(config)
    _, val_loader, _, _, _ = data_manager.create_data_loaders()

    # Collect predictions
    print("Collecting predictions...")
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # Find optimal thresholds per class
    print("\nOptimizing thresholds...")
    optimal_thresholds = {}

    from sklearn.metrics import f1_score

    for i, name in enumerate(class_names):
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs[:, i] >= threshold).astype(int)
            f1 = f1_score(all_targets[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds[name] = {
            'threshold': float(best_threshold),
            'f1': float(best_f1)
        }

        print(f"  {name}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")

    # Save thresholds
    output_path = Path(config['output']['results_dir']) / 'optimal_thresholds.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(optimal_thresholds, f)

    print(f"\nSaved: {output_path}")

    # Evaluate with optimal thresholds
    print("\nEvaluating with optimized thresholds...")

    preds_optimized = np.zeros_like(all_probs)
    for i, name in enumerate(class_names):
        threshold = optimal_thresholds[name]['threshold']
        preds_optimized[:, i] = (all_probs[:, i] >= threshold).astype(int)

    # Calculate improved metrics
    f1_macro = f1_score(all_targets, preds_optimized, average='macro', zero_division=0)
    f1_micro = f1_score(all_targets, preds_optimized, average='micro', zero_division=0)

    print(f"\nWith optimized thresholds:")
    print(f"  F1 (Macro): {f1_macro:.4f}")
    print(f"  F1 (Micro): {f1_micro:.4f}")

    return optimal_thresholds


def calibrate(config: dict, checkpoint_path: str = None):
    """Calibrate model with temperature scaling and re-optimize thresholds."""
    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 70)

    device = get_device()

    # Load model
    ckpt_path = checkpoint_path or Path(config['output']['checkpoint_dir']) / 'best_model.pth'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_names = checkpoint.get('class_names', [])
    num_classes = len(class_names)

    model = create_improved_model(config, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded: {ckpt_path}")
    print(f"  Classes: {num_classes}")

    # Load validation data
    data_manager = ImprovedDataManager(config)
    _, val_loader, _, _, _ = data_manager.create_data_loaders()

    # Run calibration
    strategy = config.get('calibration', {}).get('strategy', 'f1')
    min_recall = config.get('calibration', {}).get('min_recall', 0.8)

    temperature, thresholds = run_calibration(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        output_dir=config['output']['results_dir'],
        strategy=strategy,
        min_recall=min_recall
    )

    print(f"\nTemperature: {temperature:.4f}")
    return temperature, thresholds


def compare_models(config: dict):
    """Compare original vs improved model."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    checkpoint_dir = Path(config['output']['checkpoint_dir'])

    original_path = checkpoint_dir / 'best_model_original.pth'
    improved_path = checkpoint_dir / 'best_model.pth'

    if not original_path.exists():
        print("Original model not found. Skipping comparison.")
        return

    original = torch.load(original_path, map_location='cpu', weights_only=False)
    improved = torch.load(improved_path, map_location='cpu', weights_only=False)

    print("\n" + "-" * 50)
    print(f"{'Metric':<25} {'Original':>12} {'Improved':>12} {'Change':>12}")
    print("-" * 50)

    metrics_to_compare = ['f1_macro', 'mAP', 'roc_auc', 'hamming_loss']

    for metric in metrics_to_compare:
        orig_val = original.get('metrics', {}).get(metric, 0)
        impr_val = improved.get('metrics', {}).get(metric, 0)

        if metric == 'hamming_loss':
            change = orig_val - impr_val  # Lower is better
            change_str = f"{change:+.4f}" if change != 0 else "0.0000"
        else:
            change = impr_val - orig_val  # Higher is better
            change_str = f"{change:+.4f}" if change != 0 else "0.0000"

        print(f"{metric:<25} {orig_val:>12.4f} {impr_val:>12.4f} {change_str:>12}")

    print("-" * 50)

    # Best epochs
    print(f"\n{'Best Epoch (Original):':<30} {original.get('epoch', 'N/A')}")
    print(f"{'Best Epoch (Improved):':<30} {improved.get('epoch', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='OcuNet Phase 2 - Improved Training')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'optimize', 'calibrate', 'compare', 'all'])
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 70)
    print("OCUNET PHASE 2 - IMPROVED MULTI-LABEL CLASSIFICATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Model: {config['model']['architecture']}")
    print(f"Oversampling: {config['dataset'].get('oversample_rare_classes', False)}")
    loss_type = config['model'].get('loss_function', 'asl' if config['model'].get('use_asymmetric_loss') else 'focal')
    print(f"Loss: {loss_type.upper()}")
    print("=" * 70)

    model, test_loader, class_names = None, None, None

    if args.mode in ['train', 'all']:
        model, test_loader, class_names = train(config)

    if args.mode in ['evaluate', 'all']:
        evaluate(config, model, test_loader, class_names, args.checkpoint)

    if args.mode in ['optimize', 'all']:
        optimize_thresholds(config, args.checkpoint)

    if args.mode in ['calibrate', 'all']:
        calibrate(config, args.checkpoint)

    if args.mode in ['compare']:
        compare_models(config)

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()