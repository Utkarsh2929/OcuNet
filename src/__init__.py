"""OcuNet - Multi-Label Eye Disease Classification Package"""

from .dataset import (
    ImprovedMultiLabelDataset,
    ImprovedDataManager,
    RandAugment
)
from .models import (
    ImprovedMultiLabelModel,
    ImprovedClassificationHead,
    AsymmetricLossOptimized,
    FocalLossMultiLabel,
    SEBlock,
    create_improved_model,
    create_improved_loss,
    count_parameters
)
from .train import (
    ImprovedTrainer,
    MultiLabelMetrics,
    EarlyStopping,
    WarmupCosineScheduler,
    EMA
)
from .evaluate import (
    MultiLabelEvaluator,
    plot_multilabel_training_history
)

__version__ = "2.1.0"