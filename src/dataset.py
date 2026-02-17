#!/usr/bin/env python3
"""
Improved Multi-Label Dataset Implementation for OcuNet Phase 2.
Features:
- RFMiD + Phase 1 dataset combination
- Advanced augmentation with RandAugment
- Oversampling for rare classes
- Weighted sampling
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


class RandAugment:
    """
    RandAugment implementation for medical images.
    Applies N random transformations from a set of augmentation operations.
    """

    def __init__(self, n: int = 2, m: int = 9):
        """
        Args:
            n: Number of transformations to apply
            m: Magnitude of transformations (0-30)
        """
        self.n = n
        self.m = m

        # Define augmentation operations suitable for medical images
        self.augment_list = [
            self._identity,
            self._autocontrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = np.random.choice(self.augment_list, self.n, replace=False)
        for op in ops:
            img = op(img, self.m)
        return img

    def _identity(self, img, _):
        return img

    def _autocontrast(self, img, _):
        from PIL import ImageOps
        return ImageOps.autocontrast(img)

    def _equalize(self, img, _):
        from PIL import ImageOps
        return ImageOps.equalize(img)

    def _rotate(self, img, m):
        degrees = (m / 30) * 30  # Max 30 degrees
        return img.rotate(degrees * np.random.choice([-1, 1]))

    def _solarize(self, img, m):
        from PIL import ImageOps
        threshold = int((1 - m / 30) * 256)
        return ImageOps.solarize(img, threshold)

    def _color(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * np.random.choice([-1, 1])
        return ImageEnhance.Color(img).enhance(factor)

    def _posterize(self, img, m):
        from PIL import ImageOps
        bits = max(1, int(8 - (m / 30) * 4))
        return ImageOps.posterize(img, bits)

    def _contrast(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * np.random.choice([-1, 1])
        return ImageEnhance.Contrast(img).enhance(factor)

    def _brightness(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * np.random.choice([-1, 1])
        return ImageEnhance.Brightness(img).enhance(factor)

    def _sharpness(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * np.random.choice([-1, 1])
        return ImageEnhance.Sharpness(img).enhance(factor)

    def _shear_x(self, img, m):
        shear = (m / 30) * 0.3 * np.random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))

    def _shear_y(self, img, m):
        shear = (m / 30) * 0.3 * np.random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))

    def _translate_x(self, img, m):
        pixels = int((m / 30) * img.size[0] * 0.3) * np.random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))

    def _translate_y(self, img, m):
        pixels = int((m / 30) * img.size[1] * 0.3) * np.random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))


class ImprovedMultiLabelDataset(Dataset):
    """
    Improved multi-label dataset with oversampling support.
    """

    def __init__(
            self,
            image_paths: List[str],
            labels: np.ndarray,
            disease_names: List[str],
            transform: Optional[transforms.Compose] = None,
            oversample: bool = False,
            oversample_threshold: int = 50,
            oversample_factor: int = 3
    ):
        self.original_paths = image_paths
        self.original_labels = labels
        self.disease_names = disease_names
        self.transform = transform
        self.num_classes = len(disease_names)

        # Apply oversampling if requested
        if oversample:
            self.image_paths, self.labels = self._oversample(
                image_paths, labels, oversample_threshold, oversample_factor
            )
        else:
            self.image_paths = image_paths
            self.labels = labels

    def _oversample(
            self,
            paths: List[str],
            labels: np.ndarray,
            threshold: int,
            factor: int
    ) -> Tuple[List[str], np.ndarray]:
        """Oversample images containing rare classes."""

        # Count samples per class
        class_counts = labels.sum(axis=0)
        rare_classes = np.where(class_counts < threshold)[0]

        print(f"  Oversampling {len(rare_classes)} rare classes (threshold={threshold})")

        new_paths = list(paths)
        new_labels = list(labels)

        for class_idx in rare_classes:
            # Find images with this class
            class_mask = labels[:, class_idx] == 1
            class_indices = np.where(class_mask)[0]

            # Duplicate these images
            for _ in range(factor - 1):
                for idx in class_indices:
                    new_paths.append(paths[idx])
                    new_labels.append(labels[idx])

            class_name = self.disease_names[class_idx]
            original_count = len(class_indices)
            new_count = original_count * factor
            print(f"    {class_name}: {original_count} -> {new_count}")

        return new_paths, np.array(new_labels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

    def get_pos_weights(self) -> torch.Tensor:
        """Calculate positive class weights."""
        pos_counts = self.labels.sum(axis=0)
        neg_counts = len(self.labels) - pos_counts
        pos_counts = np.maximum(pos_counts, 1)
        pos_weights = neg_counts / pos_counts
        return torch.FloatTensor(pos_weights)

    def get_sample_weights(self) -> np.ndarray:
        """Calculate per-sample weights for weighted sampling."""
        # Weight based on inverse frequency of rarest class in each sample
        class_counts = self.labels.sum(axis=0)
        class_weights = 1.0 / np.maximum(class_counts, 1)

        sample_weights = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            if label.sum() > 0:
                # Max weight of present classes
                sample_weights[i] = np.max(class_weights[label == 1])
            else:
                sample_weights[i] = 1.0

        return sample_weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        counts = self.labels.sum(axis=0)
        return {name: int(count) for name, count in zip(self.disease_names, counts)}


class ImprovedDataManager:
    """
    Improved data manager with oversampling and advanced augmentation.
    """

    # Disease columns to use
    DISEASE_COLUMNS = [
        'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
        'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE',
        'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP',
        'CWS', 'CB', 'ODPM', 'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME',
        'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL'
    ]

    def __init__(self, config: dict):
        self.config = config
        self.phase1_root = Path(config['dataset'].get('phase1_root', 'data/dataset'))
        self.phase2_root = Path(config['dataset'].get('phase2_root', 'data/rfmid'))
        self.use_phase1 = config['dataset'].get('use_phase1', True)
        self.use_phase2 = config['dataset'].get('use_phase2', True)
        self.image_size = config['dataset']['image_size']
        self.min_samples = config['dataset'].get('min_samples_per_class', 10)

        # Oversampling settings
        self.oversample = config['dataset'].get('oversample_rare_classes', True)
        self.oversample_threshold = config['dataset'].get('oversample_threshold', 50)
        self.oversample_factor = config['dataset'].get('oversample_factor', 3)

        # Load data and determine classes
        self.disease_names = None
        self._load_all_data()

    def _find_csv_and_images(self, split: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Find CSV and image directory for a split."""
        # Try different directory structures
        possible_csv_paths = [
            self.phase2_root / f"{split}.csv",
            self.phase2_root / f"RFMiD_{split}.csv",
            self.phase2_root / split / f"{split}.csv",
        ]

        possible_img_dirs = [
            self.phase2_root / f"{split}_images",
            self.phase2_root / f"RFMiD_{split}_images",
            self.phase2_root / split / "images",
            self.phase2_root / split,
        ]

        csv_path = None
        for path in possible_csv_paths:
            if path.exists():
                csv_path = path
                break

        img_dir = None
        for path in possible_img_dirs:
            if path.exists() and path.is_dir():
                img_dir = path
                break

        return csv_path, img_dir

    def _load_rfmid_split(self, split: str) -> Tuple[List[str], np.ndarray]:
        """Load a split of the RFMiD dataset."""
        csv_path, img_dir = self._find_csv_and_images(split)

        if csv_path is None or img_dir is None:
            print(f"  Warning: Could not find {split} data")
            return [], np.array([])

        df = pd.read_csv(csv_path)
        print(f"  Found {len(df)} samples in {split}")

        paths = []
        labels = []

        for _, row in df.iterrows():
            # Find image file
            img_id = row['ID']
            possible_names = [f"{img_id}.png", f"{img_id}.jpg", f"{img_id}.jpeg"]
            img_path = None

            for name in possible_names:
                full_path = img_dir / name
                if full_path.exists():
                    img_path = full_path
                    break

            if img_path is None:
                continue

            # Create label vector
            label = np.zeros(len(self.disease_names), dtype=np.float32)
            for i, disease in enumerate(self.disease_names):
                if disease in df.columns and row[disease] == 1:
                    label[i] = 1

            paths.append(str(img_path))
            labels.append(label)

        return paths, np.array(labels) if labels else np.array([])

    def _load_phase1(self) -> Tuple[List[str], np.ndarray]:
        """Load Phase 1 dataset."""

        if not self.phase1_root.exists():
            print(f"  Phase 1 root not found: {self.phase1_root}")
            return [], np.array([])

        # Phase 1 class mapping
        phase1_classes = {
            'cataract': 'CATARACT',
            'diabetic_retinopathy': 'DR',
            'glaucoma': 'GLAUCOMA',
            'normal': 'NORMAL'
        }

        paths = []
        labels = []

        for folder_name, disease_name in phase1_classes.items():
            folder_path = self.phase1_root / folder_name
            if not folder_path.exists():
                continue

            for img_file in folder_path.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Create label
                    label = np.zeros(len(self.disease_names), dtype=np.float32)

                    if disease_name in self.disease_names:
                        idx = self.disease_names.index(disease_name)
                        label[idx] = 1

                    # Set Disease_Risk for non-normal
                    if disease_name != 'NORMAL' and 'Disease_Risk' in self.disease_names:
                        risk_idx = self.disease_names.index('Disease_Risk')
                        label[risk_idx] = 1

                    paths.append(str(img_file))
                    labels.append(label)

        return paths, np.array(labels) if labels else np.array([])

    def _load_all_data(self):
        """Load all data and determine disease classes."""

        # First pass: determine disease classes from RFMiD
        csv_path, _ = self._find_csv_and_images('train')
        if csv_path:
            df = pd.read_csv(csv_path)

            # Find disease columns with enough samples
            all_diseases = []
            for col in self.DISEASE_COLUMNS:
                if col in df.columns:
                    count = df[col].sum()
                    if count >= self.min_samples:
                        all_diseases.append(col)

            # Add Phase 1 specific classes
            if self.use_phase1:
                for disease in ['CATARACT', 'GLAUCOMA', 'NORMAL']:
                    if disease not in all_diseases:
                        all_diseases.append(disease)

            self.disease_names = all_diseases
        else:
            # Default diseases
            self.disease_names = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'ODC',
                                  'CATARACT', 'GLAUCOMA', 'NORMAL']

        print(f"\nDisease classes ({len(self.disease_names)}):")
        for i, d in enumerate(self.disease_names):
            print(f"  {i}: {d}")

    def get_transforms(self, mode: str = "train") -> transforms.Compose:
        """Get improved transforms."""

        if mode == "train":
            aug_config = self.config.get('augmentation', {})

            transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
            ]

            # RandAugment
            if aug_config.get('use_randaugment', True):
                n = aug_config.get('randaugment_n', 2)
                m = aug_config.get('randaugment_m', 9)
                transform_list.append(RandAugment(n=n, m=m))

            # Basic augmentations
            rotation = aug_config.get('rotation_degrees', 30)
            transform_list.append(transforms.RandomRotation(rotation))

            if aug_config.get('horizontal_flip', True):
                transform_list.append(transforms.RandomHorizontalFlip())

            if aug_config.get('vertical_flip', True):
                transform_list.append(transforms.RandomVerticalFlip())

            # Color augmentations
            brightness = aug_config.get('brightness_range', [0.8, 1.2])
            contrast = aug_config.get('contrast_range', [0.8, 1.2])
            saturation = aug_config.get('saturation_range', [0.8, 1.2])
            hue = aug_config.get('hue_range', 0.1)

            transform_list.append(transforms.ColorJitter(
                brightness=(brightness[0] - 1, brightness[1] - 1) if isinstance(brightness, list) else brightness,
                contrast=(contrast[0] - 1, contrast[1] - 1) if isinstance(contrast, list) else contrast,
                saturation=(saturation[0] - 1, saturation[1] - 1) if isinstance(saturation, list) else saturation,
                hue=hue
            ))

            # Zoom (random resized crop)
            zoom = aug_config.get('zoom_range', [0.8, 1.0])
            transform_list.append(transforms.RandomResizedCrop(
                self.image_size,
                scale=(zoom[0], zoom[1]) if isinstance(zoom, list) else (zoom, 1.0)
            ))

            # To tensor and normalize
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Random erasing
            if aug_config.get('random_erasing', True):
                prob = aug_config.get('random_erasing_prob', 0.25)
                scale = aug_config.get('random_erasing_scale', [0.02, 0.2])
                transform_list.append(transforms.RandomErasing(
                    p=prob,
                    scale=tuple(scale) if isinstance(scale, list) else (0.02, scale)
                ))

            return transforms.Compose(transform_list)

        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, List[str]]:
        """Create data loaders with improvements."""

        # Load RFMiD data
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        test_paths, test_labels = [], []

        if self.use_phase2:
            print("\nLoading RFMiD dataset...")
            p2_train_paths, p2_train_labels = self._load_rfmid_split('train')
            p2_val_paths, p2_val_labels = self._load_rfmid_split('val')
            p2_test_paths, p2_test_labels = self._load_rfmid_split('test')

            if len(p2_train_paths) > 0:
                train_paths.extend(p2_train_paths)
                train_labels.extend(p2_train_labels)
            if len(p2_val_paths) > 0:
                val_paths.extend(p2_val_paths)
                val_labels.extend(p2_val_labels)
            if len(p2_test_paths) > 0:
                test_paths.extend(p2_test_paths)
                test_labels.extend(p2_test_labels)

            print(f"  RFMiD - Train: {len(p2_train_paths)}, Val: {len(p2_val_paths)}, Test: {len(p2_test_paths)}")

        # Load Phase 1 data
        if self.use_phase1:
            print("\nLoading Phase 1 dataset...")
            p1_paths, p1_labels = self._load_phase1()

            if len(p1_paths) > 0:
                # Split Phase 1
                p1_train_paths, p1_temp_paths, p1_train_labels, p1_temp_labels = train_test_split(
                    p1_paths, p1_labels, test_size=0.3, random_state=42
                )
                p1_val_paths, p1_test_paths, p1_val_labels, p1_test_labels = train_test_split(
                    p1_temp_paths, p1_temp_labels, test_size=0.5, random_state=42
                )

                train_paths.extend(p1_train_paths)
                train_labels.extend(p1_train_labels)
                val_paths.extend(p1_val_paths)
                val_labels.extend(p1_val_labels)
                test_paths.extend(p1_test_paths)
                test_labels.extend(p1_test_labels)

                print(f"  Phase 1 - Train: {len(p1_train_paths)}, Val: {len(p1_val_paths)}, Test: {len(p1_test_paths)}")

        # Convert to arrays
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)

        print(f"\nCombined - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        print(f"Classes: {len(self.disease_names)}")

        # Create datasets
        train_transform = self.get_transforms("train")
        val_transform = self.get_transforms("val")

        train_dataset = ImprovedMultiLabelDataset(
            train_paths, train_labels, self.disease_names, train_transform,
            oversample=self.oversample,
            oversample_threshold=self.oversample_threshold,
            oversample_factor=self.oversample_factor
        )

        val_dataset = ImprovedMultiLabelDataset(
            val_paths, val_labels, self.disease_names, val_transform,
            oversample=False
        )

        test_dataset = ImprovedMultiLabelDataset(
            test_paths, test_labels, self.disease_names, val_transform,
            oversample=False
        )

        # Get weights
        pos_weights = train_dataset.get_pos_weights()

        # Print distribution
        print(f"\nClass distribution (after oversampling):")
        dist = train_dataset.get_class_distribution()
        for name, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count}")

        # Create loaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']

        # Use weighted sampling for training
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader, pos_weights, self.disease_names