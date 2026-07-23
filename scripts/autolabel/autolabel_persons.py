#!/usr/bin/env python3
"""
Supplement incomplete Person (class 3) annotations using Grounding DINO.

This script detects persons in images and merges them with existing Person
annotations, only adding new detections where IoU < threshold.
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import yaml
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_device():
    """Auto-detect CUDA device, fallback to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gdino_model(device):
    """Load Grounding DINO model from HuggingFace."""
    print("\n📦 Loading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model


def detect_persons(image_path, processor, model, device, box_threshold=0.35, text_threshold=0.25):
    """
    Detect persons in image using Grounding DINO.

    Returns: list of xyxy boxes (absolute pixel coordinates)
    """
    image = Image.open(image_path)
    width, height = image.size

    # Run GDINO
    inputs = processor(images=image, text="person .", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract detections
    target_sizes = torch.Tensor([[height, width]])
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=target_sizes
    )

    boxes = results[0]["boxes"].cpu().numpy()  # xyxy format, absolute coords
    return boxes


def read_yolo_labels(label_path):
    """
    Read YOLO label file. Returns list of lines (strings) and
    list of (class_id, box) tuples where box is (cx, cy, w, h) normalized.
    """
    lines = []
    person_boxes = []

    if not label_path.exists():
        return lines, person_boxes

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines.append(line)
            parts = line.split()

            try:
                class_id = int(float(parts[0]))
                if class_id == 3:  # Person class
                    cx, cy, w, h = map(float, parts[1:5])
                    person_boxes.append((class_id, (cx, cy, w, h)))
            except (ValueError, IndexError):
                pass

    return lines, person_boxes


def xyxy_to_yolo(box_xyxy, img_width, img_height):
    """Convert xyxy (absolute) to YOLO (cx cy w h normalized)."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return (cx, cy, w, h)


def yolo_to_xyxy(box_yolo, img_width, img_height):
    """Convert YOLO (cx cy w h normalized) to xyxy (absolute)."""
    cx, cy, w, h = box_yolo
    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (x1, y1, x2, y2)


def compute_iou(box1_xyxy, box2_xyxy):
    """Compute IoU between two boxes in xyxy format."""
    x1_min, y1_min, x1_max, y1_max = box1_xyxy
    x2_min, y2_min, x2_max, y2_max = box2_xyxy

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def process_image(src_image_path, src_label_path, dst_image_path, dst_label_path,
                  processor, model, device, box_threshold, iou_threshold):
    """
    Process one image: detect persons, merge with existing labels.
    Returns: (num_persons_added, total_persons_in_label)
    """
    # Load image and labels
    image = Image.open(src_image_path)
    img_width, img_height = image.size
    lines, person_boxes_yolo = read_yolo_labels(src_label_path)

    # Detect persons
    person_boxes_xyxy = detect_persons(src_image_path, processor, model, device,
                                       box_threshold=box_threshold)

    # Convert existing person boxes to xyxy for comparison
    existing_xyxy = [yolo_to_xyxy(box, img_width, img_height)
                     for _, box in person_boxes_yolo]

    # For each detected person, check if it overlaps with existing
    persons_added = 0
    for det_box_xyxy in person_boxes_xyxy:
        max_iou = max([compute_iou(det_box_xyxy, existing_box)
                       for existing_box in existing_xyxy], default=0.0)

        if max_iou < iou_threshold:
            # New person detected
            det_box_yolo = xyxy_to_yolo(det_box_xyxy, img_width, img_height)
            line = f"3 {det_box_yolo[0]:.6f} {det_box_yolo[1]:.6f} {det_box_yolo[2]:.6f} {det_box_yolo[3]:.6f}"
            lines.append(line)
            persons_added += 1

    # Copy image
    shutil.copy2(src_image_path, dst_image_path)

    # Write merged label
    with open(dst_label_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    return persons_added, len(person_boxes_yolo)


def main():
    parser = argparse.ArgumentParser(
        description="Supplement incomplete Person annotations using Grounding DINO"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/cleaned_fragment_1/ppe_curated_export",
        help="Source dataset directory"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/cleaned_fragment_1/persons_supplemented",
        help="Output directory"
    )
    parser.add_argument(
        "--box-thresh",
        type=float,
        default=0.35,
        help="GDINO box confidence threshold"
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold to consider existing person box"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N images (for testing)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists"
    )

    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    # Validation
    if not src_path.exists():
        print(f"✗ Source dataset not found: {src_path}")
        sys.exit(1)

    src_images_dir = src_path / "images"
    src_labels_dir = src_path / "labels"

    if not src_images_dir.exists() or not src_labels_dir.exists():
        print(f"✗ Missing images/ or labels/ in {src_path}")
        sys.exit(1)

    if dst_path.exists() and not args.overwrite:
        print(f"✗ Output already exists: {dst_path}")
        print("   Use --overwrite to replace")
        sys.exit(1)

    # Create output dirs
    if dst_path.exists():
        shutil.rmtree(dst_path)

    dst_images_dir = dst_path / "images"
    dst_labels_dir = dst_path / "labels"
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    # Load GDINO model
    device = get_device()
    processor, model = load_gdino_model(device)

    print(f"\n📍 Processing images from {src_images_dir}")
    print(f"📤 Output to {dst_images_dir}")

    # Process all images
    image_files = sorted(src_images_dir.glob("*"))
    if args.limit:
        image_files = image_files[:args.limit]

    total_persons_added = 0
    total_persons_existing = 0

    with tqdm(total=len(image_files), desc="Processing") as pbar:
        for img_path in image_files:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                pbar.update(1)
                continue

            label_name = img_path.stem + ".txt"
            label_path = src_labels_dir / label_name

            dst_img = dst_images_dir / img_path.name
            dst_lbl = dst_labels_dir / label_name

            try:
                added, existing = process_image(
                    img_path, label_path, dst_img, dst_lbl,
                    processor, model, device, args.box_thresh, args.iou_thresh
                )
                total_persons_added += added
                total_persons_existing += existing
                pbar.set_postfix({
                    "persons_added": total_persons_added,
                    "existing": total_persons_existing
                })
            except Exception as e:
                print(f"\n    ⚠ Error processing {img_path.name}: {e}")

            pbar.update(1)

    # Copy data.yaml (unchanged)
    src_yaml = src_path / "data.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, dst_path / "data.yaml")

    # Print summary
    print("\n" + "=" * 70)
    print("PERSON ANNOTATION SUMMARY")
    print("=" * 70)
    print(f"Images processed:     {len(image_files)}")
    print(f"Persons added:        {total_persons_added}")
    print(f"Existing persons:     {total_persons_existing}")
    print(f"Output directory:     {dst_path.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
