#!/usr/bin/env python3
"""
Add Safety Boots (class 5) and NO-Safety Boots (class 6) annotations using Grounding DINO.

This script:
1. Detects safety boots/shoes in images and labels them as class 5
2. For each Person box, infers a foot region and labels it NO-Safety Boots (class 6)
   if no safety boots are found in that region
3. Outputs 7-class labels (adds classes to existing 5-class annotations)
"""

import argparse
import shutil
import sys
from pathlib import Path
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


def detect_shoes(image_path, processor, model, device, box_threshold=0.30, text_threshold=0.25):
    """
    Detect safety boots/shoes in image using Grounding DINO.

    Returns: list of xyxy boxes (absolute pixel coordinates)
    """
    image = Image.open(image_path)
    width, height = image.size

    # Run GDINO with multiple shoe prompts
    prompt = "safety boots . safety shoes . work boots . steel toe boots ."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

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


def apply_nms(boxes_xyxy, nms_thresh=0.45):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.

    Args:
        boxes_xyxy: list of (x1, y1, x2, y2) in absolute coords
        nms_thresh: IoU threshold for suppression

    Returns: list of indices to keep
    """
    if len(boxes_xyxy) == 0:
        return []

    boxes = np.array(boxes_xyxy)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        inters = np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
        inters = inters * np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))

        union = areas[i] + areas[order[1:]] - inters
        ious = inters / union

        order = order[np.where(ious <= nms_thresh)[0] + 1]

    return keep


def read_yolo_labels(label_path):
    """
    Read YOLO label file. Returns list of lines and list of parsed annotations.
    Each annotation is (class_id, cx, cy, w, h).
    """
    lines = []
    annotations = []

    if not label_path.exists():
        return lines, annotations

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines.append(line)
            parts = line.split()

            try:
                class_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                annotations.append((class_id, cx, cy, w, h))
            except (ValueError, IndexError):
                pass

    return lines, annotations


def yolo_to_xyxy(cx, cy, w, h, img_width, img_height):
    """Convert YOLO (cx cy w h normalized) to xyxy (absolute)."""
    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (x1, y1, x2, y2)


def xyxy_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """Convert xyxy (absolute) to YOLO (cx cy w h normalized)."""
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return (cx, cy, w, h)


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
                  processor, model, device, box_threshold, nms_thresh, foot_ratio):
    """
    Process one image: detect shoes, infer NO-Safety Boots, merge with existing labels.
    Returns: (num_safety_boots, num_no_safety_boots)
    """
    # Load image and labels
    image = Image.open(src_image_path)
    img_width, img_height = image.size
    lines, annotations = read_yolo_labels(src_label_path)

    # Detect shoes
    shoe_boxes_xyxy = detect_shoes(src_image_path, processor, model, device,
                                   box_threshold=box_threshold)

    # Apply NMS to deduplicate shoe detections
    if len(shoe_boxes_xyxy) > 0:
        keep_indices = apply_nms(shoe_boxes_xyxy, nms_thresh=nms_thresh)
        shoe_boxes_xyxy = [shoe_boxes_xyxy[i] for i in keep_indices]

    # Add Safety Boots (class 5)
    safety_boots_added = 0
    for shoe_box_xyxy in shoe_boxes_xyxy:
        cx, cy, w, h = xyxy_to_yolo(shoe_box_xyxy[0], shoe_box_xyxy[1],
                                     shoe_box_xyxy[2], shoe_box_xyxy[3],
                                     img_width, img_height)
        line = f"5 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        lines.append(line)
        safety_boots_added += 1

    # Add NO-Safety Boots (class 6) - infer from Person boxes
    no_safety_boots_added = 0
    for class_id, cx, cy, w, h in annotations:
        if class_id != 3:  # Skip if not Person
            continue

        # Compute foot region: lower 25% of person box
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_width, img_height)
        foot_height = (y2 - y1) * foot_ratio
        foot_y1 = y2 - foot_height
        foot_x1 = x1
        foot_x2 = x2
        foot_y2 = y2

        foot_box_xyxy = (foot_x1, foot_y1, foot_x2, foot_y2)

        # Check if any safety boots overlap this foot region
        boots_in_foot = False
        for shoe_box_xyxy in shoe_boxes_xyxy:
            iou = compute_iou(foot_box_xyxy, shoe_box_xyxy)
            # Also check containment or high overlap (IoU > 0.1)
            if iou > 0.1:
                boots_in_foot = True
                break

        if not boots_in_foot:
            # No boots detected in foot region → label NO-Safety Boots
            foot_cx, foot_cy, foot_w, foot_h = xyxy_to_yolo(
                foot_x1, foot_y1, foot_x2, foot_y2, img_width, img_height
            )
            line = f"6 {foot_cx:.6f} {foot_cy:.6f} {foot_w:.6f} {foot_h:.6f}"
            lines.append(line)
            no_safety_boots_added += 1

    # Copy image
    shutil.copy2(src_image_path, dst_image_path)

    # Write merged 7-class label
    with open(dst_label_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    return safety_boots_added, no_safety_boots_added


def main():
    parser = argparse.ArgumentParser(
        description="Add Safety Boots and NO-Safety Boots annotations using Grounding DINO"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/cleaned_fragment_1/persons_supplemented",
        help="Source dataset directory (output of autolabel_persons.py)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/cleaned_fragment_1/ppe_7class_export",
        help="Output directory (7-class format)"
    )
    parser.add_argument(
        "--box-thresh",
        type=float,
        default=0.30,
        help="GDINO box confidence threshold for shoes"
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,
        help="NMS IoU threshold for suppressing overlapping shoe detections"
    )
    parser.add_argument(
        "--foot-ratio",
        type=float,
        default=0.25,
        help="Fraction of person box height used as foot region (0-1)"
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

    total_boots = 0
    total_no_boots = 0

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
                boots, no_boots = process_image(
                    img_path, label_path, dst_img, dst_lbl,
                    processor, model, device, args.box_thresh,
                    args.nms_thresh, args.foot_ratio
                )
                total_boots += boots
                total_no_boots += no_boots
                pbar.set_postfix({
                    "safety_boots": total_boots,
                    "no_safety_boots": total_no_boots
                })
            except Exception as e:
                print(f"\n    ⚠ Error processing {img_path.name}: {e}")

            pbar.update(1)

    # Write 7-class data.yaml
    data_yaml = {
        "nc": 7,
        "names": [
            "Hardhat",
            "NO-Hardhat",
            "NO-Safety Vest",
            "Person",
            "Safety Vest",
            "Safety Boots",
            "NO-Safety Boots"
        ]
    }
    with open(dst_path / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SAFETY BOOTS ANNOTATION SUMMARY")
    print("=" * 70)
    print(f"Images processed:     {len(image_files)}")
    print(f"Safety Boots added:   {total_boots}")
    print(f"NO-Safety Boots:      {total_no_boots}")
    print(f"Output directory:     {dst_path.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
