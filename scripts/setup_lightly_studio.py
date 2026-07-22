#!/usr/bin/env python
"""
Setup script to load PPE dataset into LightlyStudio.

This script loads the PPE_Detection_YOLOv8_cleaned YOLO dataset into a local
LightlyStudio database and launches the web UI for interactive curation.

Usage:
    python setup_lightly_studio.py \
        --data-yaml ./data/PPE_Detection_YOLOv8_cleaned/data.yaml \
        --db-file ./data/lightly_studio.db

Then open http://localhost:8001 in your browser.
"""

import argparse
import logging
from pathlib import Path

import lightly_studio as ls

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Load PPE dataset into LightlyStudio'
    )
    parser.add_argument(
        '--data-yaml',
        default='./data/PPE_Detection_YOLOv8_cleaned/data.yaml',
        help='Path to YOLO data.yaml file'
    )
    parser.add_argument(
        '--dataset-name',
        default='ppe_detection',
        help='Name of dataset in LightlyStudio'
    )
    parser.add_argument(
        '--db-file',
        default='./data/lightly_studio.db',
        help='Path to LightlyStudio database file'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Port for LightlyStudio web UI'
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host for LightlyStudio web UI'
    )

    args = parser.parse_args()

    # Convert paths to absolute
    data_yaml = Path(args.data_yaml).resolve()
    db_file = Path(args.db_file).resolve()

    if not data_yaml.exists():
        logger.error(f"data.yaml not found at {data_yaml}")
        return

    db_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Database file: {db_file}")

    # ========================================================================
    # Step 1: Connect to database
    # ========================================================================
    logger.info("Connecting to LightlyStudio database...")
    ls.db_manager.connect(db_file=str(db_file))

    # ========================================================================
    # Step 2: Load or create dataset
    # ========================================================================
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = ls.ImageDataset.load_or_create(name=args.dataset_name)

    # Check if dataset already loaded
    existing_samples = len(dataset.query().to_list())
    if existing_samples > 0:
        logger.info(f"Dataset already loaded with {existing_samples} samples.")
        logger.info("Skipping image loading. Starting GUI...")
    else:
        logger.info(f"Loading images from {data_yaml}...")
        dataset.add_samples_from_yolo(data_yaml=str(data_yaml))
        total_samples = len(dataset.query().to_list())
        logger.info(f"✓ Loaded {total_samples} images")

    # ========================================================================
    # Step 3: Summary
    # ========================================================================
    total = len(dataset.query().to_list())
    logger.info("\n" + "="*70)
    logger.info("PPE DATASET LOADED INTO LIGHTLYSTUDIO")
    logger.info("="*70)
    logger.info(f"Total images: {total}")
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info(f"Database: {db_file}")
    logger.info("="*70)
    logger.info("\n🚀 Starting LightlyStudio GUI...\n")
    logger.info(f"   Open your browser: http://{args.host}:{args.port}")
    logger.info("\n   Available in the UI:")
    logger.info("   • Grid View: Browse images, filter by tags")
    logger.info("   • Embedding Plot: Visualize data in 2D space")
    logger.info("   • Sampling: Deduplicate, select diverse subset, balance classes")
    logger.info("   • Outlier Detection: Find anomalies via typicality scoring")
    logger.info("   • Annotations: Add/edit bounding boxes")
    logger.info("   • Export: Save curated subset to YOLO format")
    logger.info("\n   Press Ctrl+C to stop the server.\n")

    # ========================================================================
    # Step 4: Start GUI
    # ========================================================================
    try:
        ls.start_gui(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("\n\nServer stopped.")


if __name__ == '__main__':
    main()
