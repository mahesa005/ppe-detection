import yaml
import torch
import time
import os
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# SEQUENTIAL TRAINING CONFIGURATION
# ============================================================
# List of config files to train sequentially
# Add or remove config paths here

MODELS = [
    "config/config_yolov11n.yaml",
    # "config_v2.yaml",
    # "config_v3.yaml",
]

# Set to True to skip failed models and continue training remaining ones
CONTINUE_ON_ERROR = False


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


def load_config(path="../config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_train_args(cfg):
    # Parameters that always apply regardless of augmentation toggle
    base_args = {
        "data":             cfg["data"],
        "epochs":           cfg["epochs"],
        "imgsz":            cfg["imgsz"],
        "batch":            cfg["batch"],
        "device":           cfg["device"],
        "workers":          cfg["workers"],
        "patience":         cfg["patience"],
        "optimizer":        cfg["optimizer"],
        "lr0":              cfg["lr0"],
        "lrf":              cfg["lrf"],
        "momentum":         cfg["momentum"],
        "weight_decay":     cfg["weight_decay"],
        "warmup_epochs":    cfg["warmup_epochs"],
        "warmup_momentum":  cfg["warmup_momentum"],
        "warmup_bias_lr":   cfg["warmup_bias_lr"],
        "box":              cfg["box"],
        "cls":              cfg["cls"],
        "dfl":              cfg["dfl"],
        "project":          cfg["project"],
        "name":             cfg["name"],
        "save":             cfg["save"],
        "save_period":      cfg["save_period"],
        "exist_ok":         cfg["exist_ok"],
        "verbose":          cfg["verbose"],
    }

    # Augmentation toggle
    if cfg.get("use_augmentation", True):
        print("Augmentation ON — YOLO will handle augmentation during training")
        aug_args = {
            "hsv_h":      cfg["hsv_h"],
            "hsv_s":      cfg["hsv_s"],
            "hsv_v":      cfg["hsv_v"],
            "fliplr":     cfg["fliplr"],
            "flipud":     cfg["flipud"],
            "degrees":    cfg["degrees"],
            "translate":  cfg["translate"],
            "scale":      cfg["scale"],
            "mosaic":     cfg["mosaic"],
            "mixup":      cfg["mixup"],
            "copy_paste": cfg["copy_paste"],
        }
    else:
        print("Augmentation OFF — Roboflow already augmented the dataset")
        aug_args = {
            "hsv_h":      0.0,
            "hsv_s":      0.0,
            "hsv_v":      0.0,
            "fliplr":     0.0,
            "flipud":     0.0,
            "degrees":    0.0,
            "translate":  0.0,
            "scale":      0.0,
            "mosaic":     0.0,
            "mixup":      0.0,
            "copy_paste": 0.0,
        }

    return {**base_args, **aug_args}


def train_single_model(config_path):
    """Train a single model with given config file"""
    try:
        # Validate config exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config
        cfg = load_config(config_path)
        train_args = build_train_args(cfg)

        # Print model info
        print(f"\n   Model     : {cfg['model']}")
        print(f"   Dataset   : {cfg['data']}")
        print(f"   Epochs    : {cfg['epochs']}")
        print(f"   Batch     : {cfg['batch']}")
        print(f"   Image size: {cfg['imgsz']}")
        print(f"   Output    : {cfg['project']}/{cfg['name']}\n")

        # Train
        start_time = time.time()
        model = YOLO(cfg["model"])
        results = model.train(**train_args)
        duration = time.time() - start_time

        # Print results
        print(f"\n   Duration  : {_format_duration(duration)}")
        print(f"   Best model: {results.save_dir}/weights/best.pt")
        print(f"   Last model: {results.save_dir}/weights/last.pt")

        return {
            "success": True,
            "config": config_path,
            "duration": duration,
            "results_dir": results.save_dir,
            "error": None
        }

    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        print(f"\n   ✗ ERROR: {str(e)}")
        return {
            "success": False,
            "config": config_path,
            "duration": duration,
            "results_dir": None,
            "error": str(e)
        }


def train_all_models():
    """Train all models in MODELS array sequentially"""
    if not MODELS:
        print("No models configured. Add models to MODELS array in train.py")
        return

    _print_header(f"STARTING SEQUENTIAL TRAINING - {len(MODELS)} models")

    results = []
    total_start = time.time()

    for idx, config_path in enumerate(MODELS, 1):
        print(f"[{idx}/{len(MODELS)}] Training: {config_path}")

        result = train_single_model(config_path)
        results.append(result)

        if not result["success"] and not CONTINUE_ON_ERROR:
            print(f"\n✗ Training failed. Set CONTINUE_ON_ERROR=True to skip and train remaining models.")
            raise Exception(f"Training failed for {config_path}: {result['error']}")

    total_duration = time.time() - total_start

    # Print summary
    _print_header("TRAINING COMPLETE")
    successful = sum(1 for r in results if r["success"])
    print(f"Models trained: {successful}/{len(MODELS)}")
    print(f"Total time: {_format_duration(total_duration)}\n")

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['config']} ({_format_duration(r['duration'])})")

    print()


def main():
    # GPU check
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected — training on CPU will be very slow")

    # Train all configured models sequentially
    train_all_models()


if __name__ == "__main__":
    main()  