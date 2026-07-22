#!/usr/bin/env python3
"""
PPE Detection Testing Pipeline
Validates all trained models on test dataset and generates comparison report
Saves detailed results (per-class metrics) to JSON and CSV
"""

import torch
import time
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# SEQUENTIAL TESTING CONFIGURATION
# ============================================================
# List of trained models to test sequentially
# Format: (run_folder, model_name)

MODELS = [
    ("runs/detect/ppe_v2-4", "YOLO11n"),
    ("runs/detect/ppe_v2-5", "YOLO11s"),
    ("runs/detect/ppe_v2-10", "YOLO12n"),
    ("runs/detect/ppe_v2-11", "YOLO12s"),
    ("runs/detect/ppe_v2-12", "YOLO26n"),
    ("runs/detect/ppe_v2-13", "YOLO26s"),
]

# Dataset config
DATA_YAML = "./data/PPR_EXT.v2-raw.yolov11/data.yaml"

# Test on: "val" (validation) or "test" (test split)
TEST_SPLIT = "val"

# Results output directory
RESULTS_DIR = Path("runs/test_results")

# Set to True to skip failed models and continue testing remaining ones
CONTINUE_ON_ERROR = True


def _format_duration(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def _print_header(message):
    """Print centered section header"""
    width = 80
    print("\n" + "=" * width)
    print(message.center(width))
    print("=" * width + "\n")


def _print_class_table(per_class_data):
    """Print per-class metrics as table"""
    if not per_class_data:
        print("   (No per-class data available)")
        return

    print("\n   PER-CLASS METRICS:")
    print("   " + "-" * 90)
    print("   {:.<25} {:^12} {:^12} {:^12} {:^12}".format("Class", "Precision", "Recall", "mAP50", "mAP50-95"))
    print("   " + "-" * 90)

    for class_name, metrics in per_class_data.items():
        print(
            "   {:.<25} {:<12.3f} {:<12.3f} {:<12.3f} {:<12.3f}".format(
                class_name,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("mAP50", 0),
                metrics.get("mAP50-95", 0),
            )
        )

    print("   " + "-" * 90 + "\n")


def test_single_model(model_path, model_name, data_yaml, imgsz=640):
    """Test a single model and return overall + per-class metrics"""
    try:
        # Validate model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Print model info
        print(f"\n   Model       : {model_name}")
        print(f"   Path        : {model_path}")
        print(f"   Dataset     : {data_yaml}")
        print(f"   Image size  : {imgsz}x{imgsz}\n")

        # Test
        start_time = time.time()
        model = YOLO(model_path)
        results = model.val(
            data=data_yaml,
            imgsz=imgsz,
            verbose=False,
            plots=False,
        )
        duration = time.time() - start_time

        # ========== EXTRACT OVERALL METRICS ==========
        overall_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50-95": 0.0,
        }

        try:
            overall_metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0))
            overall_metrics["recall"] = float(results.results_dict.get("metrics/recall(B)", 0))
            overall_metrics["mAP50"] = float(results.results_dict.get("metrics/mAP50(B)", 0))
            overall_metrics["mAP50-95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0))
        except Exception as e:
            print(f"   Warning: Could not extract overall metrics: {e}")

        # ========== EXTRACT PER-CLASS METRICS ==========
        per_class_metrics = {}

        try:
            if hasattr(results, 'names') and hasattr(results, 'box'):
                class_names = results.names  # {0: 'hardhat', 1: 'no_hardhat', ...}

                # Get per-class metrics arrays
                if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
                    class_indices = results.box.ap_class_index
                else:
                    # Fallback: use all classes
                    class_indices = range(len(class_names))

                for i, class_id in enumerate(class_indices):
                    class_id_int = int(class_id)
                    class_name = class_names.get(class_id_int, f"Class_{class_id_int}")

                    # Extract metrics with bounds checking
                    try:
                        p = float(results.box.p[i]) if hasattr(results.box, 'p') and i < len(results.box.p) else 0.0
                        r = float(results.box.r[i]) if hasattr(results.box, 'r') and i < len(results.box.r) else 0.0
                        ap50 = (
                            float(results.box.ap50[i])
                            if hasattr(results.box, 'ap50') and i < len(results.box.ap50)
                            else 0.0
                        )
                        ap = float(results.box.ap[i]) if hasattr(results.box, 'ap') and i < len(results.box.ap) else 0.0

                        per_class_metrics[class_name] = {
                            "precision": p,
                            "recall": r,
                            "mAP50": ap50,
                            "mAP50-95": ap,
                        }
                    except (IndexError, TypeError, ValueError):
                        per_class_metrics[class_name] = {
                            "precision": 0.0,
                            "recall": 0.0,
                            "mAP50": 0.0,
                            "mAP50-95": 0.0,
                        }
        except Exception as e:
            print(f"   Warning: Could not extract per-class metrics: {e}")

        # Print overall results
        print(f"   Duration    : {_format_duration(duration)}")
        print(f"   Precision   : {overall_metrics['precision']:.3f}")
        print(f"   Recall      : {overall_metrics['recall']:.3f}")
        print(f"   mAP50       : {overall_metrics['mAP50']:.3f}")
        print(f"   mAP50-95    : {overall_metrics['mAP50-95']:.3f}")

        # Print per-class metrics
        _print_class_table(per_class_metrics)

        return {
            "success": True,
            "model_name": model_name,
            "model_path": model_path,
            "duration": duration,
            "overall": overall_metrics,
            "per_class": per_class_metrics,
            "error": None,
        }

    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        print(f"\n   ✗ ERROR: {str(e)}")
        return {
            "success": False,
            "model_name": model_name,
            "model_path": model_path,
            "duration": duration,
            "overall": {},
            "per_class": {},
            "error": str(e),
        }


def save_json_results(results, timestamp):
    """Save detailed results to JSON file"""
    output_file = RESULTS_DIR / f"results_{timestamp}.json"

    json_data = {
        "timestamp": timestamp,
        "num_models": len(results),
        "models": [
            {
                "model_name": r["model_name"],
                "model_path": r["model_path"],
                "duration": r["duration"],
                "success": r["success"],
                "overall": r["overall"],
                "per_class": r["per_class"],
            }
            for r in results
        ],
    }

    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✓ Saved detailed results to: {output_file}")


def save_csv_summary(results, timestamp):
    """Save summary results to CSV file (easy to open in Excel)"""
    output_file = RESULTS_DIR / f"results_{timestamp}_summary.csv"

    rows = []
    for r in results:
        if r["success"]:
            rows.append(
                {
                    "Model": r["model_name"],
                    "Precision": f"{r['overall'].get('precision', 0):.3f}",
                    "Recall": f"{r['overall'].get('recall', 0):.3f}",
                    "mAP50": f"{r['overall'].get('mAP50', 0):.3f}",
                    "mAP50-95": f"{r['overall'].get('mAP50-95', 0):.3f}",
                    "Duration": _format_duration(r["duration"]),
                    "Status": "PASS",
                }
            )
        else:
            rows.append(
                {
                    "Model": r["model_name"],
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "mAP50": "N/A",
                    "mAP50-95": "N/A",
                    "Duration": _format_duration(r["duration"]),
                    "Status": "FAIL",
                }
            )

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Precision", "Recall", "mAP50", "mAP50-95", "Duration", "Status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Saved CSV summary to: {output_file}")


def test_all_models():
    """Test all models in MODELS array sequentially"""
    if not MODELS:
        print("No models configured. Add models to MODELS array in test.py")
        return

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _print_header(f"STARTING SEQUENTIAL TESTING - {len(MODELS)} models")

    results = []
    total_start = time.time()

    for idx, (model_dir, model_name) in enumerate(MODELS, 1):
        model_path = f"{model_dir}/weights/best.pt"
        print(f"[{idx}/{len(MODELS)}] Testing: {model_name}")

        result = test_single_model(model_path, model_name, DATA_YAML)
        results.append(result)

        if not result["success"] and not CONTINUE_ON_ERROR:
            print(f"\n✗ Testing failed. Set CONTINUE_ON_ERROR=True to skip and test remaining models.")
            raise Exception(f"Testing failed for {model_name}: {result['error']}")

    total_duration = time.time() - total_start

    # Print summary
    _print_header("TESTING COMPLETE")
    successful = sum(1 for r in results if r["success"])
    print(f"Models tested: {successful}/{len(MODELS)}")
    print(f"Total time: {_format_duration(total_duration)}\n")

    # Print results table
    print("OVERALL RESULTS TABLE:")
    print("-" * 100)
    print(f"{'Model':<15} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12} {'Duration':<12} {'Status':<10}")
    print("-" * 100)

    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        if r["success"]:
            print(
                f"{r['model_name']:<15} "
                f"{r['overall'].get('mAP50', 0):<12.3f} "
                f"{r['overall'].get('mAP50-95', 0):<12.3f} "
                f"{r['overall'].get('precision', 0):<12.3f} "
                f"{r['overall'].get('recall', 0):<12.3f} "
                f"{_format_duration(r['duration']):<12} "
                f"{status:<10}"
            )
        else:
            print(f"{r['model_name']:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {_format_duration(r['duration']):<12} {status:<10}")

    print("-" * 100)

    # Print recommendations
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        best = max(successful_results, key=lambda x: x['overall'].get('mAP50', 0))
        print(f"\n✓ Best performing model: {best['model_name']}")
        print(f"  - mAP50: {best['overall'].get('mAP50', 0):.3f}")
        print(f"  - mAP50-95: {best['overall'].get('mAP50-95', 0):.3f}")
        print(f"  - Path: {best['model_path']}")

    # Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json_results(results, timestamp)
    save_csv_summary(results, timestamp)

    print()


def main():
    # GPU check
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected — testing on CPU will be slow")

    # Test all configured models sequentially
    test_all_models()


if __name__ == "__main__":
    main()
