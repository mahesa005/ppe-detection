# PPE Dataset Curation with LightlyStudio

Complete workflow for curating the PPE_Detection_YOLOv8_cleaned dataset following the official LightlyStudio tutorial.

## Prerequisites

- Python 3.10 or newer
- ~200MB disk space for LightlyStudio, plugins, and database
- Windows, Linux, or macOS
- GPU not required

## Step 0: Install LightlyStudio and YOLO Plugin

### Option A: Install from PyPI (Recommended)
```bash
pip install lightly-studio
pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/yolo_object_detection/"
```

### Option B: Install from Local Clone
If you have the lightly-studio repo cloned locally:
```bash
cd C:\Users\Mahesa\OneDrive\ITB\Coding\Personal\Tools\lightly-studio
pip install -e .
cd <back to ppe-detection>
pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/yolo_object_detection/"
```

> **Important:** Stop any running LightlyStudio server before installing the plugin.

## Step 1: Load the Dataset

Run the setup script to load the PPE dataset and launch LightlyStudio:

```bash
cd C:\Users\Mahesa\OneDrive\ITB\Coding\Work\spt\project-2\ppe-detection
python scripts/setup_lightly_studio.py \
  --data-yaml ./data/PPE_Detection_YOLOv8_cleaned/data.yaml \
  --db-file ./data/lightly_studio.db
```

Expected output:
```
INFO: Found 24299 images
INFO: Open the LightlyStudio GUI under: http://localhost:8001
INFO: Using MobileCLIP embedding generator for images.
```

The script will:
- Load all 24,299 images from the PPE dataset
- Compute MobileCLIP embeddings for semantic search
- Launch the web UI at http://localhost:8001
- Keep running until you press Ctrl+C

**Open http://localhost:8001 in your browser.**

## Step 2: Explore the Dataset

### View Embedding Plot
1. Click the **Embed** button in the upper right of the GUI
2. The embedding plot shows all 24,299 images grouped by visual similarity
3. Explore clusters to understand dataset structure:
   - Dense clusters = similar images (e.g., same scene from multiple angles)
   - Isolated points = outliers (unusual angles, poor quality, irrelevant images)

### Identify Outliers
While exploring, look for:
- Blurry images
- Images without PPE (no persons or safety equipment)
- Extreme angles or crops
- Images with wrong labels
- Low-quality or corrupted images

## Step 3: Tag Outliers

### Use the Lasso Tool
1. At the bottom of the embedding plot, select the **Lasso tool**
2. Draw a lasso around outlier points (isolated samples on the plot)
3. Right-click and create a new tag, e.g., `outlier` or `bad-quality`

### Tag Good Samples
1. Select promising clusters in the embedding plot (dense regions = good data)
2. Draw a lasso around them
3. Create a tag like `quality-samples` or `good-candidates`

**Tip:** The bottom-left shows the current count of visible images. Track this as you filter.

## Step 4: Filter Out Outliers and Deduplicate

### Filter Using Query
1. Open the **Query Filter** on the right side
2. Enter a query to exclude outliers (replace tag name as needed):
```
NOT "outlier" IN tags
```
3. Click **Apply** to filter the dataset

### Deduplicate
1. Open **Menu** → **Sampling**
2. Select **Deduplication Sampling**
3. Configure to reduce the filtered set to a smaller, diverse subset:
   - Example: 15,000 images from ~20,000 filtered
4. Select the resulting samples and tag as `deduplicated`

This removes near-duplicate images while keeping diverse examples.

## Step 5: Run YOLO Auto-Labeling (Optional but Recommended)

### Install YOLO Plugin
If not already done:
```bash
pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/yolo_object_detection/"
```

### Run Plugin
1. Select the deduplicated samples (filter by tag `deduplicated`)
2. Click **Menu** → **Plugins**
3. Click **YOLO Object Detection**
4. Pick a model (e.g., `yolov8m.pt` for balance), set confidence threshold
5. Name the annotation source, e.g., `yolov8-auto`
6. Click **Execute**

Wait ~20-60 seconds for predictions to run. Predictions will appear in the GUI.

### Review Annotations
Two approaches:

**Approach A: Tag Bad Predictions**
- Grid View: Scroll through images, tag those with faulty predictions as `review-needed`
- Recommended if a separate team will correct annotations

**Approach B: Edit Directly**
- Switch to **Annotation View** (shows objects as cropped images)
- Use class filters to review one class at a time
- Edit bounding boxes and class labels directly
- Recommended for quick QA yourself

## Step 6: Split into Training and Test Sets

### Create Training Set (80%)
1. Filter by `deduplicated` tag using Query Filter:
```
"deduplicated" IN tags
```
2. Open **Menu** → **Sampling** → **Diversity Sampling**
3. Configure to select 80% of deduplicated samples
4. Tag the selected samples as `train`

### Create Test Set (20%)
1. Enter a new Query Filter to find remaining deduplicated samples:
```
"deduplicated" IN tags AND NOT "train" IN tags
```
2. Click **Apply**, then **Select all**
3. Tag these samples as `test`

**Result:**
- `train`: 80% of deduplicated samples for model training
- `test`: 20% for evaluation

## Step 7: Export in YOLO Format

Create a Python script `scripts/export_curated_dataset.py`:

```python
#!/usr/bin/env python
"""
Export curated PPE dataset splits in YOLO format.
"""

import lightly_studio as ls
from lightly_studio.core.dataset_query.image_sample_field import ImageSampleField
from pathlib import Path

# Load the dataset
dataset = ls.ImageDataset.load(name="ppe_detection")

# Tags to export
tags_to_export = ["train", "test"]

# Create export directory
export_dir = Path("./data/ppe_curated")
export_dir.mkdir(exist_ok=True)

# Export each split
for tag in tags_to_export:
    query = dataset.query().match(ImageSampleField.tags.contains(tag))
    output_path = str(export_dir / f"{tag}_yolo")
    dataset.export(query).to_yolo_object_detections(output_path)
    print(f"✓ Exported samples with tag '{tag}' to {output_path}")

    # Print split info
    split_query = dataset.query().match(ImageSampleField.tags.contains(tag))
    count = len(split_query.to_list())
    print(f"  → {count} images in {tag} split")
```

Run it:
```bash
python scripts/export_curated_dataset.py
```

Expected output:
```
✓ Exported samples with tag 'train' to ./data/ppe_curated/train_yolo
  → 19440 images in train split
✓ Exported samples with tag 'test' to ./data/ppe_curated/test_yolo
  → 4860 images in test split
```

**Exported structure:**
```
data/ppe_curated/
├── train_yolo/
│   ├── images/       # 19,440 images
│   ├── labels/       # 19,440 .txt files with bboxes
│   └── data.yaml     # Train config
└── test_yolo/
    ├── images/       # 4,860 images
    ├── labels/       # 4,860 .txt files with bboxes
    └── data.yaml     # Test config
```

## Step 8: Train YOLO Model (Optional)

Create `scripts/train_ppe_yolo.py`:

```python
#!/usr/bin/env python
"""
Train YOLO model on curated PPE dataset.
"""

import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

# Paths
train_yolo_dir = Path("data/ppe_curated/train_yolo")
test_yolo_dir = Path("data/ppe_curated/test_yolo")

def find_image(stem, original_images_dir):
    """Find image file by stem name."""
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = original_images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def copy_images_for_split(split_dir: Path, original_images_dir: Path):
    """Copy images into split directories (if needed)."""
    labels_dir = split_dir / "labels"
    images_out = split_dir / "images"

    if images_out.exists() and list(images_out.glob("*")):
        print(f"{split_dir}: images already present, skipping copy")
        return

    images_out.mkdir(exist_ok=True)
    count = 0
    for label_file in labels_dir.glob("*.txt"):
        image_file = find_image(label_file.stem, original_images_dir)
        if image_file is None:
            continue
        shutil.copy2(image_file, images_out / image_file.name)
        count += 1

    print(f"{split_dir}: copied {count} images")

# Original dataset directory
original_images_dir = Path("data/PPE_Detection_YOLOv8_cleaned/images")

# Copy images if not already present
print("Ensuring images are in split directories...")
# Note: export already includes images, so this may not be necessary

# Create combined data.yaml for training
with open(train_yolo_dir / "data.yaml") as f:
    config = yaml.safe_load(f)

# Get class names from original data.yaml
with open("data/PPE_Detection_YOLOv8_cleaned/data.yaml") as f:
    original_config = yaml.safe_load(f)

class_names = original_config["names"]
num_classes = original_config["nc"]

combined_config = {
    "train": str(train_yolo_dir / "images"),
    "val": str(test_yolo_dir / "images"),
    "nc": num_classes,
    "names": class_names,
}

data_yaml_path = Path("data/ppe_train.yaml")
with open(data_yaml_path, "w") as f:
    yaml.dump(combined_config, f)

print(f"✓ Created {data_yaml_path}")
print(f"  Classes: {num_classes} — {class_names}")

# Train YOLO
print("\n🚀 Starting YOLO training...")
model = YOLO("yolov8m.pt")  # medium model; use 'yolov8n.pt' for faster training

model.train(
    data=str(data_yaml_path),
    epochs=50,
    imgsz=640,
    device=0,  # GPU 0; set to 'cpu' if no GPU
    patience=10,  # early stopping
)

# Evaluate
print("\n📊 Evaluating model...")
metrics = model.val(data=str(data_yaml_path))
print(f"mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
```

Install Ultralytics first:
```bash
pip install ultralytics
```

Run training:
```bash
python scripts/train_ppe_yolo.py
```

Training will:
- Run for 50 epochs (adjust as needed)
- Save best model to `runs/detect/train/weights/best.pt`
- Evaluate on test set after training
- Display mAP scores

## Workflow Summary

```
1. Load dataset (24,299 images)
   ↓
2. Explore embeddings, identify outliers
   ↓
3. Tag outliers and good samples
   ↓
4. Filter + deduplicate (~15,000 images)
   ↓
5. Run YOLO auto-labeling (optional)
   ↓
6. Review/correct annotations (optional)
   ↓
7. Split: 80% train, 20% test
   ↓
8. Export to YOLO format
   ↓
9. Train YOLO model (optional)
```

## Useful Query Filter Examples

- Exclude outliers: `NOT "outlier" IN tags`
- Get only deduplicated: `"deduplicated" IN tags`
- Get training split: `"train" IN tags`
- Get test split: `"test" IN tags`
- Multiple tags: `"train" IN tags AND "quality-samples" IN tags`
- Exclude tag: `NOT "review-needed" IN tags`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Plugin not found | Stop server, install plugin, restart |
| Embeddings not computed | Check network, may take time for 24K images |
| No images visible | Check Query Filter and tags |
| Export fails | Verify data.yaml exists and paths are correct |
| YOLO training slow | Use smaller model (`yolov8n.pt`) or reduce epochs |

## Database Management

The LightlyStudio database is stored at:
```
./data/lightly_studio.db
```

To reset and start fresh:
```bash
rm ./data/lightly_studio.db
python scripts/setup_lightly_studio.py
```

This will reload all 24,299 images and recompute embeddings.

## Next Steps

After training:
1. Evaluate metrics on test set
2. If performance is poor:
   - Review more annotations for quality
   - Adjust train/test split ratio
   - Try different YOLO model size
   - Collect more diverse samples
3. Deploy best model for inference

---

**Tutorial Reference:** [LightlyStudio YOLO Curation Tutorial](https://docs.lightly.ai/tutorials/yolo-curation/)
