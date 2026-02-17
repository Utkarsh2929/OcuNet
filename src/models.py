#!/usr/bin/env python3
"""
Improved Multi-Label Models for OcuNet Phase 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional, List
import math


class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    Optimized version with better handling of class imbalance.

    Paper: https://arxiv.org/abs/2009.14119
    """

    def __init__(
            self,
            gamma_neg: float = 4,
            gamma_pos: float = 1,
            clip: float = 0.05,
            eps: float = 1e-8,
            pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Sigmoid
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Apply pos_weight if provided
        if self.pos_weight is not None:
            los_pos = los_pos * self.pos_weight.to(x.device)

        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


class FocalLossMultiLabel(nn.Module):
    """Focal Loss for Multi-Label with class weights."""

    def __init__(
            self,
            gamma: float = 2.0,
            pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none',
            pos_weight=self.pos_weight.to(inputs.device) if self.pos_weight is not None else None
        )

        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        return (focal_weight * bce).mean()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape
        y = F.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class ImprovedClassificationHead(nn.Module):
    """Improved classification head with attention and deeper layers."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            dropout_rate: float = 0.5
    ):
        super().__init__()

        self.attention = SEBlock(in_features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return self.classifier(x)


class ImprovedMultiLabelModel(nn.Module):
    """
    Improved multi-label classifier with stronger backbone and better head.
    """

    def __init__(
            self,
            num_classes: int,
            backbone: str = 'efficientnet_b2',
            pretrained: bool = True,
            dropout_rate: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Create backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)

        # Classification head
        self.classifier = ImprovedClassificationHead(
            self.feature_dim, num_classes, dropout_rate
        )

    def _create_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create backbone network."""

        if name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        elif name == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b1(weights=weights)
            feature_dim = 1280
        elif name == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b2(weights=weights)
            feature_dim = 1408
        elif name == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            feature_dim = 1536
        elif name == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            feature_dim = 1792
        elif name == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_tiny(weights=weights)
            feature_dim = 768
            model.classifier = nn.Identity()
            return model, feature_dim
        elif name == 'convnext_small':
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_small(weights=weights)
            feature_dim = 768
            model.classifier = nn.Identity()
            return model, feature_dim
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        # Remove classifier for EfficientNet
        model.classifier = nn.Identity()

        return model, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.flatten(1)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


def create_improved_model(config: dict, num_classes: int) -> nn.Module:
    """Create improved model."""

    architecture = config['model'].get('architecture', 'efficientnet_b2')
    pretrained = config['model'].get('pretrained', True)
    dropout_rate = config['model'].get('dropout_rate', 0.5)

    return ImprovedMultiLabelModel(
        num_classes=num_classes,
        backbone=architecture,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


def create_improved_loss(config: dict, pos_weights: torch.Tensor, device: torch.device) -> nn.Module:
    """Create improved loss function."""

    pos_weights = pos_weights.to(device)

    if config['model'].get('use_asymmetric_loss', True):
        return AsymmetricLossOptimized(
            gamma_neg=config['model'].get('asl_gamma_neg', 4),
            gamma_pos=config['model'].get('asl_gamma_pos', 1),
            clip=config['model'].get('asl_clip', 0.05),
            pos_weight=pos_weights
        )
    elif config['model'].get('use_focal_loss', True):
        return FocalLossMultiLabel(
            gamma=config['model'].get('focal_loss_gamma', 2.0),
            pos_weight=pos_weights
        )
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable