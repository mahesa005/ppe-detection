import yaml
import torch
from ultralytics import YOLO


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


def main():
    # GPU check
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected — training on CPU will be very slow")

    # Load config
    cfg = load_config("config.yaml")
    train_args = build_train_args(cfg)

    print(f"\nTraining config:")
    print(f"   Model     : {cfg['model']}")
    print(f"   Dataset   : {cfg['data']}")
    print(f"   Epochs    : {cfg['epochs']}")
    print(f"   Batch     : {cfg['batch']}")
    print(f"   Image size: {cfg['imgsz']}")
    print(f"   Output    : {cfg['project']}/{cfg['name']}\n")

    # Load model and train
    model = YOLO(cfg["model"])
    results = model.train(**train_args)

    print(f"\n Training complete")
    print(f"   Best model : {results.save_dir}/weights/best.pt")
    print(f"   Last model : {results.save_dir}/weights/last.pt")
    print(f"\n To evaluate:")
    print(f"   yolo val model={results.save_dir}/weights/best.pt data={cfg['data']}")


if __name__ == "__main__":
    main()  