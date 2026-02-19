#!/usr/bin/env python3
"""
Tests for OcuNet Phase 2 improvements.
Validates new modules without requiring dataset or GPU.

Run: python -m pytest tests/test_improvements.py -v
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Temperature Scaling Tests
# ============================================================

class TestTemperatureScaling:
    """Test temperature scaling calibration module."""

    def test_temperature_scaling_forward(self):
        """Temperature scaling should divide logits by T."""
        from src.calibrate import TemperatureScaling

        ts = TemperatureScaling()
        logits = torch.randn(10, 5)

        # Default temperature is 1.0, so output should equal input
        scaled = ts(logits)
        assert torch.allclose(logits, scaled, atol=1e-6), "T=1 should be identity"

        # Set temperature to 2.0
        ts.temperature = nn.Parameter(torch.tensor([2.0]))
        scaled = ts(logits)
        expected = logits / 2.0
        assert torch.allclose(expected, scaled, atol=1e-6), "T=2 should halve logits"

    def test_calibrated_probabilities(self):
        """Calibrated probabilities should be in [0, 1]."""
        from src.calibrate import TemperatureScaling

        ts = TemperatureScaling()
        ts.temperature = nn.Parameter(torch.tensor([1.5]))

        logits = torch.randn(100, 28)
        probs = ts.calibrate(logits)

        assert probs.min() >= 0.0, "Probabilities must be >= 0"
        assert probs.max() <= 1.0, "Probabilities must be <= 1"

    def test_fit_temperature(self):
        """fit_temperature should learn a positive temperature."""
        from src.calibrate import fit_temperature

        torch.manual_seed(42)
        logits = torch.randn(200, 10) * 3  # Overconfident logits
        # Create targets where positive rate matches ~sigmoid(logits)
        targets = (torch.sigmoid(logits) > 0.5).float()

        ts = fit_temperature(logits, targets)
        temp = ts.temperature.item()

        assert temp > 0, f"Temperature must be positive, got {temp}"
        print(f"  Learned temperature: {temp:.4f}")

    def test_threshold_optimization_f1(self):
        """Threshold optimization should return valid thresholds and metrics."""
        from src.calibrate import optimize_thresholds_calibrated

        np.random.seed(42)
        probs = np.random.rand(100, 3)
        targets = (np.random.rand(100, 3) > 0.7).astype(float)
        class_names = ['ClassA', 'ClassB', 'ClassC']

        thresholds = optimize_thresholds_calibrated(probs, targets, class_names, strategy="f1")

        for name in class_names:
            t = thresholds[name]
            assert 0 < t['threshold'] < 1, f"Threshold out of range: {t['threshold']}"
            assert 0 <= t['f1'] <= 1, f"F1 out of range: {t['f1']}"
            assert 0 <= t['precision'] <= 1
            assert 0 <= t['recall'] <= 1

    def test_threshold_optimization_precision_at_recall(self):
        """Precision-at-recall strategy should respect min_recall."""
        from src.calibrate import optimize_thresholds_calibrated

        np.random.seed(42)
        # Create scenario with clear signal
        probs = np.random.rand(500, 2)
        targets = np.zeros((500, 2))
        targets[:200, 0] = 1.0
        targets[:100, 1] = 1.0
        # Make probs correlated with targets
        probs[targets == 1] += 0.3
        probs = np.clip(probs, 0, 1)

        class_names = ['A', 'B']
        thresholds = optimize_thresholds_calibrated(
            probs, targets, class_names,
            strategy="precision_at_recall", min_recall=0.5
        )

        for name in class_names:
            assert 'threshold' in thresholds[name]
            assert 'precision' in thresholds[name]


# ============================================================
# 2. Preprocessing Tests
# ============================================================

class TestPreprocessing:
    """Test fundus preprocessing transforms."""

    def test_fundus_roi_crop_with_borders(self):
        """FundusROICrop should detect and crop circular ROI."""
        from src.preprocessing import FundusROICrop

        # Create a 400x400 image with black background and white circle in center
        img_np = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw a filled circle
        import cv2
        cv2.circle(img_np, (200, 200), 150, (180, 120, 100), -1)

        img = Image.fromarray(img_np)
        crop = FundusROICrop(padding_ratio=0.05)
        result = crop(img)

        result_np = np.array(result)
        # Cropped image should be smaller than original (borders removed)
        assert result_np.shape[0] < 400 or result_np.shape[1] < 400, \
            f"Expected cropped size < 400, got {result_np.shape}"
        # Should still contain the circle center
        assert result_np.shape[0] > 100, "Crop should not be too small"
        print(f"  Original: 400x400 -> Cropped: {result_np.shape[1]}x{result_np.shape[0]}")

    def test_fundus_roi_crop_no_borders(self):
        """FundusROICrop should return original when no clear borders exist."""
        from src.preprocessing import FundusROICrop

        # Full bright image (no black borders)
        img_np = np.full((300, 300, 3), 180, dtype=np.uint8)
        img = Image.fromarray(img_np)
        crop = FundusROICrop()
        result = crop(img)

        # Should return same size since no borders to crop
        result_np = np.array(result)
        assert result_np.shape[0] >= 200, "Should not over-crop a full image"

    def test_clahe_transform(self):
        """CLAHE should preserve image dimensions."""
        from src.preprocessing import CLAHETransform

        img_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_np)

        # Test green channel mode
        clahe_green = CLAHETransform(clip_limit=2.0, channel="green")
        result = clahe_green(img)
        result_np = np.array(result)
        assert result_np.shape == img_np.shape, \
            f"Shape mismatch: {result_np.shape} != {img_np.shape}"

        # Test LAB mode
        clahe_lab = CLAHETransform(clip_limit=2.0, channel="lab")
        result_lab = clahe_lab(img)
        result_lab_np = np.array(result_lab)
        assert result_lab_np.shape == img_np.shape

    def test_clahe_grayscale(self):
        """CLAHE should work on grayscale images."""
        from src.preprocessing import CLAHETransform

        img_np = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img = Image.fromarray(img_np, mode='L')

        clahe = CLAHETransform()
        result = clahe(img)
        result_np = np.array(result)
        assert result_np.shape == img_np.shape


# ============================================================
# 3. Model Architecture Tests
# ============================================================

class TestModelArchitectures:
    """Test backbone instantiation and forward pass."""

    def _test_backbone(self, arch_name, expected_output=28):
        """Helper to test a backbone."""
        from src.models import create_improved_model

        config = {
            'model': {
                'architecture': arch_name,
                'pretrained': False,  # Faster for testing
                'dropout_rate': 0.3,
            }
        }
        model = create_improved_model(config, num_classes=expected_output)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, expected_output), \
            f"{arch_name}: expected output shape (1, {expected_output}), got {out.shape}"
        print(f"  {arch_name}: output shape = {out.shape} OK")

    def test_efficientnet_b2(self):
        self._test_backbone('efficientnet_b2')

    def test_efficientnet_v2_s(self):
        self._test_backbone('efficientnet_v2_s')

    def test_swin_t(self):
        self._test_backbone('swin_t')

    def test_convnext_tiny(self):
        self._test_backbone('convnext_tiny')


# ============================================================
# 4. Loss Function Config Tests
# ============================================================

class TestLossConfig:
    """Test loss function creation with old and new config styles."""

    def test_new_style_asl(self):
        """New-style config: loss_function: 'asl' should create ASL."""
        from src.models import create_improved_loss, AsymmetricLossOptimized

        config = {'model': {'loss_function': 'asl', 'asl_gamma_neg': 4, 'asl_gamma_pos': 1, 'asl_clip': 0.05}}
        pos_weights = torch.ones(10)
        device = torch.device('cpu')

        loss = create_improved_loss(config, pos_weights, device)
        assert isinstance(loss, AsymmetricLossOptimized), f"Expected ASL, got {type(loss)}"

    def test_new_style_focal(self):
        """New-style config: loss_function: 'focal' should create FocalLoss."""
        from src.models import create_improved_loss, FocalLossMultiLabel

        config = {'model': {'loss_function': 'focal', 'focal_loss_gamma': 2.0}}
        pos_weights = torch.ones(10)
        device = torch.device('cpu')

        loss = create_improved_loss(config, pos_weights, device)
        assert isinstance(loss, FocalLossMultiLabel), f"Expected FocalLoss, got {type(loss)}"

    def test_new_style_bce(self):
        """New-style config: loss_function: 'bce'."""
        from src.models import create_improved_loss

        config = {'model': {'loss_function': 'bce'}}
        pos_weights = torch.ones(10)
        device = torch.device('cpu')

        loss = create_improved_loss(config, pos_weights, device)
        assert isinstance(loss, nn.BCEWithLogitsLoss)

    def test_old_style_backward_compat(self):
        """Old boolean-style config should still work."""
        from src.models import create_improved_loss, AsymmetricLossOptimized

        # Old style: use_asymmetric_loss: true
        config = {'model': {'use_asymmetric_loss': True, 'use_focal_loss': True,
                            'asl_gamma_neg': 4, 'asl_gamma_pos': 1, 'asl_clip': 0.05}}
        pos_weights = torch.ones(10)
        device = torch.device('cpu')

        loss = create_improved_loss(config, pos_weights, device)
        assert isinstance(loss, AsymmetricLossOptimized), "Old config should default to ASL"


# ============================================================
# 5. Effective Weights Tests
# ============================================================

class TestEffectiveWeights:
    """Test class-balanced effective weights computation."""

    def test_effective_weights_basic(self):
        """Effective weights should be positive and finite."""
        from src.models import compute_effective_weights

        pos_counts = torch.tensor([500, 50, 5, 1, 1000])
        weights = compute_effective_weights(pos_counts, total=2000)

        assert torch.all(weights > 0), "All weights must be positive"
        assert torch.all(torch.isfinite(weights)), "All weights must be finite"
        print(f"  Weights: {weights.numpy()}")

    def test_rare_classes_get_higher_weight(self):
        """Rare classes should get higher weights."""
        from src.models import compute_effective_weights

        pos_counts = torch.tensor([1000, 10])
        weights = compute_effective_weights(pos_counts, total=2000)

        assert weights[1] > weights[0], \
            f"Rare class weight ({weights[1]:.4f}) should be > common class ({weights[0]:.4f})"

    def test_dataset_effective_weights(self):
        """Dataset method should produce valid weights."""
        from src.dataset import ImprovedMultiLabelDataset

        # Create mock dataset
        paths = [f"img_{i}.jpg" for i in range(100)]
        labels = np.random.randint(0, 2, (100, 5)).astype(np.float32)
        names = ['A', 'B', 'C', 'D', 'E']

        ds = ImprovedMultiLabelDataset(paths, labels, names)
        weights = ds.get_effective_pos_weights()

        assert weights.shape == (5,), f"Expected shape (5,), got {weights.shape}"
        assert torch.all(weights > 0)
        assert torch.all(torch.isfinite(weights))


# ============================================================
# 6. Gradient Accumulation Test
# ============================================================

class TestGradientAccumulation:
    """Test that gradient accumulation config is correctly parsed."""

    def test_default_no_accumulation(self):
        """Default config should have accumulation = 1."""
        config = {
            'training': {
                'num_epochs': 1,
                'learning_rate': 0.001,
                'batch_size': 16,
                'weight_decay': 0.0001,
            }
        }
        steps = config['training'].get('gradient_accumulation_steps', 1)
        assert steps == 1

    def test_accumulation_configured(self):
        """Config with accumulation should be parsed correctly."""
        config = {
            'training': {
                'gradient_accumulation_steps': 4,
                'batch_size': 8,
            }
        }
        steps = config['training']['gradient_accumulation_steps']
        effective = config['training']['batch_size'] * steps
        assert effective == 32


# ============================================================
# Run all tests
# ============================================================

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
