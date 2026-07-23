#!/usr/bin/env python3
"""
Per-Class Metrics Extractor
Runs validation on trained models and extracts per-class precision/recall/mAP
"""

from pathlib import Path
import sys
import subprocess


def run_validation(model_path: str, data_path: str, imgsz: int = 640) -> str:
    """Run YOLO validation and capture output"""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        results = model.val(data=data_path, imgsz=imgsz, verbose=True)

        return results
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error running validation: {e}")
        return None


def extract_class_metrics(results) -> dict:
    """Extract per-class metrics from YOLO validation results"""
    if not results:
        return {}

    metrics = {}

    # YOLO stores per-class results in results.box
    if hasattr(results, 'box'):
        print("\n" + "="*80)
        print("PER-CLASS METRICS".center(80))
        print("="*80 + "\n")

        # Get class names
        class_names = results.names

        # Get per-class metrics
        if hasattr(results.box, 'per_class_results'):
            per_class = results.box.per_class_results
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
            print("-" * 80)

            for class_id, class_name in class_names.items():
                if class_id < len(per_class):
                    p = per_class[class_id][0]  # precision
                    r = per_class[class_id][1]  # recall
                    m50 = per_class[class_id][2]  # mAP50
                    m95 = per_class[class_id][3]  # mAP50-95

                    print(f"{class_name:<20} {p:<12.3f} {r:<12.3f} {m50:<12.3f} {m95:<12.3f}")
                    metrics[class_name] = {
                        'precision': p,
                        'recall': r,
                        'mAP50': m50,
                        'mAP50-95': m95,
                    }

    return metrics


def main():
    project_root = Path(__file__).parent.parent

    models_to_check = [
        ("ppe_v2-4", "YOLO11n"),
        ("ppe_v2-5", "YOLO11s"),
        ("ppe_v2-10", "YOLO12n"),
        ("ppe_v2-11", "YOLO12s"),
        ("ppe_v2-12", "YOLO26n"),
        ("ppe_v2-13", "YOLO26s"),
    ]

    data_yaml = str(project_root / "data" / "PPR_EXT.v2-raw.yolov11" / "data.yaml")

    for run_dir, model_name in models_to_check:
        model_path = str(project_root / "runs" / "detect" / run_dir / "weights" / "best.pt")

        if not Path(model_path).exists():
            print(f"⊘ {model_name} ({run_dir}): Model not found")
            continue

        print(f"\n{'█'*80}")
        print(f"Validating {model_name} ({run_dir})".center(80))
        print(f"{'█'*80}")

        results = run_validation(model_path, data_yaml)

        if results:
            metrics = extract_class_metrics(results)
            print(f"\n✓ Validation complete for {model_name}")
        else:
            print(f"✗ Could not validate {model_name}")


if __name__ == "__main__":
    main()
