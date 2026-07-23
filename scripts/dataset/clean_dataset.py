#!/usr/bin/env python3
"""
Clean PPE detection dataset: remove unwanted classes and remap class IDs
Keeps only: Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import yaml


# Class mapping: original ID → new ID
KEEP_IDS = {3, 8, 10, 11, 13}
REMAP = {
    3: 0,   # Hardhat
    8: 1,   # NO-Hardhat
    10: 2,  # NO-Safety Vest
    11: 3,  # Person
    13: 4,  # Safety Vest
}
NEW_NAMES = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Person", "Safety Vest"]
SPLITS = ["train", "valid", "test"]


def process_label_file(label_path):
    """
    Read label file, filter to kept classes, remap IDs.
    Returns: (kept_lines, original_count, kept_count)
    """
    kept_lines = []
    original_count = 0
    kept_count = 0

    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip blank lines
                    continue

                original_count += 1
                parts = line.split()

                if len(parts) < 5:
                    print(f"    ⚠ Malformed line in {label_path.name}:{line_num} (< 5 tokens, skipped)")
                    continue

                try:
                    # Handle float class IDs (e.g., "3.0")
                    class_id = int(float(parts[0]))
                except (ValueError, IndexError):
                    print(f"    ⚠ Invalid class ID in {label_path.name}:{line_num} (skipped)")
                    continue

                # Keep only desired classes and remap
                if class_id in KEEP_IDS:
                    parts[0] = str(REMAP[class_id])
                    kept_lines.append(' '.join(parts) + '\n')
                    kept_count += 1

    except Exception as e:
        print(f"    ✗ Error reading {label_path}: {e}")
        return ([], 0, 0)

    return (kept_lines, original_count, kept_count)


def process_split(src_split_dir, dst_split_dir, split_name, stats, dry_run=False):
    """
    Process all images/labels in one split.
    """
    src_labels_dir = src_split_dir / "labels"
    src_images_dir = src_split_dir / "images"

    if not src_labels_dir.exists():
        return

    if not dry_run:
        dst_labels_dir = dst_split_dir / "labels"
        dst_images_dir = dst_split_dir / "images"

    label_files = sorted(src_labels_dir.glob("*.txt"))
    print(f"\n  Processing split '{split_name}': {len(label_files)} label files")

    for label_file in label_files:
        kept_lines, orig_count, kept_count = process_label_file(label_file)

        # Find image file (try .jpg, .jpeg, .png, .bmp, .webp)
        image_path = None
        stem = label_file.stem
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = src_images_dir / (stem + ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None and kept_lines:
            print(f"    ⚠ Image file not found for: {stem} (label file kept anyway)")

        # Update stats
        stats[split_name]["images_before"] += 1

        if kept_lines:
            # Keep this image+label pair
            stats[split_name]["images_kept"] += 1
            stats[split_name]["annotations_kept"] += kept_count

            if not dry_run:
                # Copy image if it exists
                if image_path:
                    dst_image = dst_images_dir / image_path.name
                    shutil.copy2(image_path, dst_image)

                # Write label file
                dst_label = dst_labels_dir / label_file.name
                with open(dst_label, 'w') as f:
                    f.writelines(kept_lines)
        else:
            # Delete this pair
            stats[split_name]["images_deleted"] += 1
            stats[split_name]["annotations_removed"] += orig_count


def main():
    parser = argparse.ArgumentParser(
        description="Clean PPE dataset: remove unwanted classes and remap class IDs"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/PPE_Detection_YOLOv8",
        help="Source dataset directory (default: data/PPE_Detection_YOLOv8)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/PPE_Detection_YOLOv8_cleaned",
        help="Output directory (default: data/PPE_Detection_YOLOv8_cleaned)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, no file writes"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate dst if it already exists"
    )

    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    # Validation
    if not src_path.exists():
        print(f"✗ Source dataset not found: {src_path}")
        sys.exit(1)

    if not (src_path / "data.yaml").exists():
        print(f"✗ data.yaml not found in {src_path}")
        sys.exit(1)

    if dst_path.exists() and not args.overwrite:
        print(f"✗ Destination already exists: {dst_path}")
        print("   Use --overwrite to replace it")
        sys.exit(1)

    # Print plan
    print("\n" + "=" * 70)
    print("DATASET CLEANING PLAN")
    print("=" * 70)
    print(f"Source:      {src_path.resolve()}")
    print(f"Destination: {dst_path.resolve()}")
    print(f"Dry run:     {args.dry_run}")
    print(f"Classes:     Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest")
    print("=" * 70)

    if not args.dry_run:
        response = input("\nProceed with cleaning? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Cancelled.")
            sys.exit(0)

    # Create output directory structure (skip in dry-run)
    if not args.dry_run:
        if dst_path.exists():
            shutil.rmtree(dst_path)
        dst_path.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            (dst_path / split / "images").mkdir(parents=True, exist_ok=True)
            (dst_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Initialize stats
    stats = {split: {
        "images_before": 0,
        "images_kept": 0,
        "images_deleted": 0,
        "annotations_kept": 0,
        "annotations_removed": 0,
    } for split in SPLITS}

    # Process each split
    print("\n" + "-" * 70)
    for split in SPLITS:
        src_split = src_path / split
        dst_split = dst_path / split
        if src_split.exists():
            process_split(src_split, dst_split, split, stats, dry_run=args.dry_run)

    # Rewrite data.yaml (skip in dry-run)
    if not args.dry_run:
        data_yaml = {
            "train": "../train/images",
            "val": "../valid/images",
            "test": "../test/images",
            "nc": 5,
            "names": NEW_NAMES,
        }
        with open(dst_path / "data.yaml", 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Print summary
    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"\n{'Split':<10} {'Before':<10} {'Kept':<10} {'Deleted':<10} {'Annots Removed':<15}")
    print("-" * 70)

    total_before = 0
    total_kept = 0
    total_deleted = 0
    total_annots_removed = 0

    for split in SPLITS:
        s = stats[split]
        total_before += s["images_before"]
        total_kept += s["images_kept"]
        total_deleted += s["images_deleted"]
        total_annots_removed += s["annotations_removed"]

        print(
            f"{split:<10} {s['images_before']:<10} {s['images_kept']:<10} "
            f"{s['images_deleted']:<10} {s['annotations_removed']:<15}"
        )

    print("-" * 70)
    print(
        f"{'TOTAL':<10} {total_before:<10} {total_kept:<10} "
        f"{total_deleted:<10} {total_annots_removed:<15}"
    )
    print("=" * 70)

    if not args.dry_run:
        print(f"\n✓ Cleaned dataset saved to: {dst_path.resolve()}")
        print(f"✓ New data.yaml created with nc=5")
    else:
        print("\n⚠ DRY RUN: No files were modified")

    print()


if __name__ == "__main__":
    main()
