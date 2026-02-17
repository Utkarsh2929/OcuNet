#!/usr/bin/env python3
"""
Setup script to organize datasets for OcuNet Phase 2.
Run this after downloading both datasets.
"""

import shutil
from pathlib import Path
import pandas as pd


def setup_phase1_dataset(source: str, target: str = "data/dataset"):
    """Setup Phase 1 folder-based dataset."""
    source = Path(source)
    target = Path(target)

    if target.exists():
        print(f"Phase 1 dataset already exists at {target}")
        return

    print(f"Setting up Phase 1 dataset from {source} to {target}")

    expected_classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

    for class_name in expected_classes:
        class_source = source / class_name
        class_target = target / class_name

        if class_source.exists():
            shutil.copytree(class_source, class_target)
            count = len(list(class_target.glob("*")))
            print(f"  {class_name}: {count} images")
        else:
            print(f"  Warning: {class_source} not found")

    print("Phase 1 dataset setup complete!")


def setup_phase2_dataset(source: str, target: str = "data/rfmid"):
    """Setup Phase 2 RFMiD dataset."""
    source = Path(source)
    target = Path(target)

    if target.exists():
        print(f"Phase 2 dataset already exists at {target}")
        return

    print(f"Setting up Phase 2 dataset from {source} to {target}")
    target.mkdir(parents=True, exist_ok=True)

    # Expected RFMiD structure
    expected_items = [
        "Training_Set",
        "Evaluation_Set",
        "Test_Set",
        "RFMiD_Training_Labels.csv",
        "RFMiD_Validation_Labels.csv",
        "RFMiD_Testing_Labels.csv"
    ]

    for item in expected_items:
        item_source = source / item
        item_target = target / item

        if item_source.exists():
            if item_source.is_dir():
                shutil.copytree(item_source, item_target)
                # Count images
                images = list(item_target.rglob("*.png")) + list(item_target.rglob("*.jpg"))
                print(f"  {item}: {len(images)} images")
            else:
                shutil.copy(item_source, item_target)
                df = pd.read_csv(item_target)
                print(f"  {item}: {len(df)} rows")
        else:
            # Try alternative names
            alternatives = list(source.glob(f"*{item.lower().replace('_', '')}*"))
            if alternatives:
                alt = alternatives[0]
                if alt.is_dir():
                    shutil.copytree(alt, item_target)
                else:
                    shutil.copy(alt, item_target)
                print(f"  {item}: Found as {alt.name}")
            else:
                print(f"  Warning: {item} not found")

    print("Phase 2 dataset setup complete!")


def verify_datasets():
    """Verify both datasets are properly setup."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    # Phase 1
    phase1_path = Path("data/dataset")
    if phase1_path.exists():
        print("\nPhase 1 Dataset:")
        classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        total = 0
        for cls in classes:
            cls_path = phase1_path / cls
            if cls_path.exists():
                count = len(list(cls_path.glob("*")))
                total += count
                print(f"  {cls}: {count}")
            else:
                print(f"  {cls}: NOT FOUND")
        print(f"  Total: {total}")
    else:
        print("\nPhase 1 Dataset: NOT FOUND")

    # Phase 2
    phase2_path = Path("data/rfmid")
    if phase2_path.exists():
        print("\nPhase 2 Dataset (RFMiD):")

        # Check CSVs
        for csv_name in ["RFMiD_Training_Labels.csv", "RFMiD_Validation_Labels.csv", "RFMiD_Testing_Labels.csv"]:
            csv_path = phase2_path / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"  {csv_name}: {len(df)} rows, {len(df.columns)} columns")
            else:
                # Try finding it
                found = list(phase2_path.rglob(f"*{csv_name.split('_')[1].lower()}*.csv"))
                if found:
                    df = pd.read_csv(found[0])
                    print(f"  {found[0].name}: {len(df)} rows")
                else:
                    print(f"  {csv_name}: NOT FOUND")

        # Check image folders
        for folder in ["Training_Set", "Evaluation_Set", "Test_Set"]:
            folder_path = phase2_path / folder
            if folder_path.exists():
                images = list(folder_path.rglob("*.png")) + list(folder_path.rglob("*.jpg"))
                print(f"  {folder}: {len(images)} images")
            else:
                print(f"  {folder}: NOT FOUND")
    else:
        print("\nPhase 2 Dataset: NOT FOUND")

    print("\n" + "=" * 60)


def print_dataset_info():
    """Print detailed dataset information."""
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    # Phase 2 disease mapping info
    print("\nRFMiD to OcuNet Class Mapping:")
    print("-" * 40)
    mapping = {
        'cataract': ['Cataract', 'CATARACT'],
        'diabetic_retinopathy': ['DR', 'Disease_Risk', 'MH', 'DN', 'ARMD'],
        'glaucoma': ['GLAUCOMA', 'Glaucoma', 'ODC'],
        'normal': ['Images with no positive disease labels']
    }

    for ocunet_class, rfmid_labels in mapping.items():
        print(f"\n{ocunet_class.upper()}:")
        for label in rfmid_labels:
            print(f"  - {label}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python setup_datasets.py verify")
        print("  python setup_datasets.py phase1 <source_path>")
        print("  python setup_datasets.py phase2 <source_path>")
        print("  python setup_datasets.py info")
        sys.exit(1)

    command = sys.argv[1]

    if command == "verify":
        verify_datasets()
    elif command == "phase1" and len(sys.argv) >= 3:
        setup_phase1_dataset(sys.argv[2])
        verify_datasets()
    elif command == "phase2" and len(sys.argv) >= 3:
        setup_phase2_dataset(sys.argv[2])
        verify_datasets()
    elif command == "info":
        print_dataset_info()
    else:
        print(f"Unknown command: {command}")