#!/usr/bin/env python3
"""
Visualize YOLO labels overlaid on images.
Opens a window showing each image with bounding boxes and class names.
Press any key to go to next image, 'q' to quit.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


# Class colors (BGR) — one per class, up to 10
COLORS = [
    (0, 255, 0),      # 0: Hardhat — green
    (0, 0, 255),      # 1: NO-Hardhat — red
    (0, 165, 255),    # 2: NO-Safety Vest — orange
    (255, 255, 0),    # 3: Person — cyan
    (255, 0, 255),    # 4: Safety Vest — magenta
    (0, 255, 255),    # 5: Safety Boots — yellow
    (128, 0, 255),    # 6: NO-Safety Boots — purple
    (255, 128, 0),    # 7: spare
    (0, 128, 255),    # 8: spare
    (128, 255, 0),    # 9: spare
]


def load_class_names(data_yaml_path):
    """Load class names from data.yaml."""
    import yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return names


def draw_labels(image, label_path, class_names):
    """Draw YOLO bounding boxes on image."""
    h, w = image.shape[:2]
    count = 0

    if not label_path.exists():
        return image, 0

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

            # Convert YOLO normalized → pixel coords
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = COLORS[class_id % len(COLORS)]
            name = class_names[class_id] if class_id < len(class_names) else str(class_id)

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            label_text = f"{name} ({class_id})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            count += 1

    return image, count


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels overlaid on images"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/cleaned_fragment_1/persons_supplemented",
        help="Dataset directory (must contain images/ and labels/)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Path to data.yaml (auto-detected from --src if not provided)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to show"
    )
    parser.add_argument(
        "--filter-class",
        type=int,
        default=None,
        help="Only show images that contain this class ID"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save visualizations to this directory instead of showing interactively"
    )

    args = parser.parse_args()

    src_path = Path(args.src)
    images_dir = src_path / "images"
    labels_dir = src_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Missing images/ or labels/ in {src_path}")
        sys.exit(1)

    # Load class names
    yaml_path = Path(args.classes) if args.classes else src_path / "data.yaml"
    class_names = []
    if yaml_path.exists():
        class_names = load_class_names(yaml_path)
        print(f"Classes: {class_names}")
    else:
        print("No data.yaml found, using class IDs only")

    # Get image list
    image_files = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    # Filter to images containing a specific class
    if args.filter_class is not None:
        filtered = []
        for img_path in image_files:
            lbl = labels_dir / (img_path.stem + ".txt")
            if lbl.exists():
                content = lbl.read_text()
                if any(line.split() and int(float(line.split()[0])) == args.filter_class
                       for line in content.splitlines() if line.strip()):
                    filtered.append(img_path)
        image_files = filtered
        print(f"Found {len(image_files)} images with class {args.filter_class}")

    if args.limit:
        image_files = image_files[:args.limit]

    if args.save:
        save_path = Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nShowing {len(image_files)} images from {images_dir}")
    if not args.save:
        print("Controls: any key = next image | 'q' = quit\n")

    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        image, bbox_count = draw_labels(image, label_path, class_names)

        # Add image info overlay
        info = f"[{i+1}/{len(image_files)}] {img_path.name} | {bbox_count} boxes"
        cv2.putText(image, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(image, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1)

        if args.save:
            out_path = Path(args.save) / img_path.name
            cv2.imwrite(str(out_path), image)
            print(f"Saved: {out_path.name} ({bbox_count} boxes)")
        else:
            cv2.imshow("PPE Label Viewer", image)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    if not args.save:
        cv2.destroyAllWindows()

    print(f"\nDone. Showed {len(image_files)} images.")


if __name__ == "__main__":
    main()
