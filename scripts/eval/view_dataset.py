#!/usr/bin/env python3
"""
Load a flat YOLO dataset and view it in FiftyOne browser UI.

Usage:
  python scripts/view_dataset.py --src data/cleaned_fragment_1/persons_supplemented
  python scripts/view_dataset.py --src data/cleaned_fragment_1/ppe_7class_export --name "7-class shoes"

The browser will open at http://localhost:5151
"""

import argparse
import sys
from pathlib import Path
import yaml
import fiftyone as fo


def load_class_names(data_yaml_path):
    """Load class names from data.yaml."""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return names


def build_dataset(src_path, dataset_name="PPE Dataset"):
    """
    Manually build FiftyOne dataset from flat YOLO structure.

    Expected structure:
      src_path/
        images/  (*.jpg, *.png, etc.)
        labels/  (*.txt in YOLO format)
        data.yaml
    """
    src_path = Path(src_path)
    images_dir = src_path / "images"
    labels_dir = src_path / "labels"
    yaml_path = src_path / "data.yaml"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Missing images/ or labels/ in {src_path}")
        sys.exit(1)

    # Load class names
    class_names = []
    if yaml_path.exists():
        class_names = load_class_names(yaml_path)
    print(f"Classes: {class_names}")

    # Get image files
    image_files = sorted([
        f for f in images_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])
    print(f"Found {len(image_files)} images")

    # Create FiftyOne dataset
    dataset = fo.Dataset(name=dataset_name)

    # Build samples
    print(f"\nLoading images and labels...")
    for i, img_path in enumerate(image_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(image_files)}")

        # Read image (copy path, don't actually load)
        image_path = str(img_path.resolve())

        # Read labels
        label_path = labels_dir / (img_path.stem + ".txt")
        detections = []

        if label_path.exists():
            # Get image dimensions for coordinate conversion
            from PIL import Image
            img = Image.open(img_path)
            img_width, img_height = img.size

            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except ValueError:
                        continue

                    # Convert YOLO (cx cy w h normalized) to FiftyOne format
                    # FiftyOne expects [left, top, width, height] in absolute pixel coords
                    left = (cx - bw / 2) * img_width
                    top = (cy - bh / 2) * img_height
                    width = bw * img_width
                    height = bh * img_height

                    class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

                    detection = fo.Detection(
                        label=class_name,
                        bounding_box=[left, top, width, height]
                    )
                    detections.append(detection)

        # Create sample
        sample = fo.Sample(filepath=image_path)
        sample["ground_truth"] = fo.Detections(detections=detections)
        dataset.add_sample(sample)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="View YOLO dataset in FiftyOne browser UI"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/cleaned_fragment_1/persons_supplemented",
        help="Dataset directory (must contain images/, labels/, data.yaml)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name for FiftyOne (default: auto from src)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for FiftyOne app (default: 5151)"
    )

    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print(f"ERROR: Dataset directory not found: {src_path}")
        sys.exit(1)

    # Auto-generate dataset name from source path
    dataset_name = args.name or src_path.name
    print(f"\n{'='*70}")
    print(f"FiftyOne Dataset Viewer")
    print(f"{'='*70}")
    print(f"Source:  {src_path.resolve()}")
    print(f"Dataset: {dataset_name}")
    print(f"Port:    {args.port}")
    print(f"{'='*70}\n")

    # Build dataset
    dataset = build_dataset(src_path, dataset_name=dataset_name)
    print(f"\nDataset has {len(dataset)} samples")

    # Launch app
    print(f"\nLaunching FiftyOne app at http://localhost:{args.port}")
    print("Controls:")
    print("  - Click images to view details")
    print("  - Filter by class, confidence, or other properties")
    print("  - Export data if needed")
    print("  - Close browser or press Ctrl+C to exit\n")

    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
