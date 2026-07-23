#!/usr/bin/env python
"""
Export curated PPE dataset by filename list.

Reads a txt file containing image paths and copies matching images + annotations
to a new YOLO format dataset.

Usage:
    python export_by_filelist.py --filelist data/cleaned_fragment_1/PPE_Detection_YOLOv8_cleaned_fragment.txt
    python export_by_filelist.py --filelist data/cleaned_fragment_1/PPE_Detection_YOLOv8_cleaned_fragment.txt --output-dir data/ppe_curated_export/
"""

import argparse
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Export PPE dataset by filename list'
    )
    parser.add_argument(
        '--filelist',
        required=True,
        help='Path to txt file with image paths (one per line)'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/ppe_curated_export/',
        help='Output directory for exported dataset'
    )
    parser.add_argument(
        '--original-dataset',
        default='./data/PPE_Detection_YOLOv8_cleaned/',
        help='Path to original PPE dataset'
    )

    args = parser.parse_args()

    filelist = Path(args.filelist).resolve()
    output_dir = Path(args.output_dir).resolve()
    original_dataset = Path(args.original_dataset).resolve()

    # Validate inputs
    if not filelist.exists():
        logger.error(f"Filelist not found: {filelist}")
        return

    if not original_dataset.exists():
        logger.error(f"Original dataset not found: {original_dataset}")
        return

    # Check if output directory exists
    if output_dir.exists():
        logger.warning(f"Output directory already exists: {output_dir}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Export cancelled.")
            return
        shutil.rmtree(output_dir)

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    logger.info(f"Reading filelist: {filelist}")

    # Read image paths
    with open(filelist, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"Found {len(image_paths)} images in filelist")

    # Copy images and labels
    copied = 0
    missing_labels = 0
    not_found = 0

    for image_path_str in image_paths:
        image_path = Path(image_path_str)

        # Check if image exists
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            not_found += 1
            continue

        # Get filename (without extension)
        image_name = image_path.name
        stem = image_path.stem

        # Find corresponding label file
        # Labels are in the same split folder but in "labels" subfolder
        # e.g., train/images/file.jpg -> train/labels/file.txt
        label_path = image_path.parent.parent / "labels" / f"{stem}.txt"

        # Copy image
        dest_image = images_dir / image_name
        shutil.copy2(image_path, dest_image)

        # Copy label if exists
        if label_path.exists():
            dest_label = labels_dir / f"{stem}.txt"
            shutil.copy2(label_path, dest_label)
        else:
            logger.warning(f"Label not found: {label_path}")
            missing_labels += 1

        copied += 1

        if copied % 1000 == 0:
            logger.info(f"Progress: {copied}/{len(image_paths)} images copied")

    # Copy data.yaml
    data_yaml_src = original_dataset / "data.yaml"
    if data_yaml_src.exists():
        shutil.copy2(data_yaml_src, output_dir / "data.yaml")
        logger.info("Copied data.yaml")
    else:
        logger.warning("data.yaml not found in original dataset")

    # Print summary
    logger.info("=" * 70)
    logger.info("EXPORT COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Images copied: {copied}")
    logger.info(f"Missing labels: {missing_labels}")
    logger.info(f"Images not found: {not_found}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    logger.info(f"\nExported structure:")
    logger.info(f"  {images_dir}  ({len(list(images_dir.glob('*')))} files)")
    logger.info(f"  {labels_dir}  ({len(list(labels_dir.glob('*')))} files)")
    logger.info(f"  {output_dir / 'data.yaml'}")


if __name__ == '__main__':
    main()
