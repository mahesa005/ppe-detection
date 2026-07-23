#!/usr/bin/env python3
"""
Split a flat YOLO dataset (images/ + labels/) into train/val/test structure.
"""

import argparse
import shutil
import sys
from pathlib import Path
import random
import yaml


def main():
    parser = argparse.ArgumentParser(description="Split flat YOLO dataset into train/val/test")
    parser.add_argument("--src", type=str, default="data/cleaned_fragment_1/persons_supplemented",
                        help="Source flat dataset (images/ + labels/)")
    parser.add_argument("--dst", type=str, default="data/cleaned_fragment_1/persons_supplemented_split",
                        help="Output directory with train/val/test structure")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dst if exists")

    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    images_dir = src_path / "images"
    labels_dir = src_path / "labels"

    # Validation
    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Missing images/ or labels/ in {src_path}")
        sys.exit(1)

    if dst_path.exists() and not args.overwrite:
        print(f"ERROR: {dst_path} already exists. Use --overwrite to replace.")
        sys.exit(1)

    # Get image files
    image_files = sorted([
        f for f in images_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")
    print(f"Split: train={args.train} val={args.val} test={args.test}")

    # Verify split ratios
    if abs(args.train + args.val + args.test - 1.0) > 0.001:
        print("ERROR: train + val + test must equal 1.0")
        sys.exit(1)

    # Create output dirs
    if dst_path.exists():
        shutil.rmtree(dst_path)

    splits = ["train", "val", "test"]
    for split in splits:
        (dst_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(image_files)

    n_train = int(len(image_files) * args.train)
    n_val = int(len(image_files) * args.val)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    split_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # Copy files
    print("\nCopying files...")
    for split, files in split_dict.items():
        for img_path in files:
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name

            dst_img = dst_path / split / "images" / img_path.name
            dst_lbl = dst_path / split / "labels" / label_name

            shutil.copy2(img_path, dst_img)
            if label_path.exists():
                shutil.copy2(label_path, dst_lbl)

        print(f"  {split:5s}: {len(files):5d} images")

    # Copy and update data.yaml
    src_yaml = src_path / "data.yaml"
    if src_yaml.exists():
        with open(src_yaml, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = {"nc": 5, "names": []}

    # Update paths
    data["train"] = "train/images"
    data["val"] = "val/images"
    data["test"] = "test/images"

    with open(dst_path / "data.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nOK: Split dataset saved to {dst_path.resolve()}")
    print(f"    data.yaml created with proper train/val/test paths")


if __name__ == "__main__":
    main()
