# Grounding DINO Auto-Labeler for PPE Dataset

This guide explains how to use the two auto-labeling scripts to add Person and Safety Boots annotations to your PPE detection dataset.

## Overview

The dataset `cleaned_fragment_1/ppe_curated_export/` is missing:
1. **Complete Person annotations** — the existing labels have gaps
2. **Safety Boots / NO-Safety Boots classes** — these don't exist yet

The solution uses **Grounding DINO** (zero-shot object detection) in two steps:

### Step 1: Supplement Person Annotations
**Script:** `scripts/autolabel_persons.py`
- Detects all persons in images using GDINO
- Merges detections with existing Person labels (IoU-based deduplication)
- Only adds new persons where IoU < 0.5 with existing annotations
- **Input:** `data/cleaned_fragment_1/ppe_curated_export/` (5-class)
- **Output:** `data/cleaned_fragment_1/persons_supplemented/` (5-class, enhanced)

### Step 2: Add Safety Boots Annotations
**Script:** `scripts/autolabel_shoes.py`
- Detects safety boots/shoes in images using GDINO
- Applies NMS to deduplicate overlapping detections
- Adds **Safety Boots (class 5)** labels
- Infers **NO-Safety Boots (class 6)** for foot regions with no boots detected
- **Input:** `data/cleaned_fragment_1/persons_supplemented/` (5-class)
- **Output:** `data/cleaned_fragment_1/ppe_7class_export/` (7-class)

---

## Quick Start

### 1. Test with a Small Sample

Test the person supplementation on 20 images:
```bash
python scripts/autolabel_persons.py --limit 20
```

Check the output:
```bash
ls -la data/cleaned_fragment_1/persons_supplemented/labels | head -10
```

Test the shoe labeling on those same images:
```bash
python scripts/autolabel_shoes.py --limit 20
```

### 2. Run Full Pipeline

Once satisfied with the test, run on all 12,319 images:

**Step 1: Supplement persons**
```bash
python scripts/autolabel_persons.py --overwrite
```
Expected output: `data/cleaned_fragment_1/persons_supplemented/` with ~12,319 images/labels

**Step 2: Add shoes**
```bash
python scripts/autolabel_shoes.py --overwrite
```
Expected output: `data/cleaned_fragment_1/ppe_7class_export/` with 7-class labels

---

## Configuration

### autolabel_persons.py

```bash
python scripts/autolabel_persons.py \
  --src data/cleaned_fragment_1/ppe_curated_export \
  --dst data/cleaned_fragment_1/persons_supplemented \
  --box-thresh 0.35 \
  --iou-thresh 0.5 \
  --limit 100 \
  --overwrite
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--src` | `data/cleaned_fragment_1/ppe_curated_export` | Source dataset path |
| `--dst` | `data/cleaned_fragment_1/persons_supplemented` | Output path |
| `--box-thresh` | `0.35` | GDINO box confidence threshold (0-1, lower = more detections) |
| `--iou-thresh` | `0.5` | IoU threshold to consider box as "already labeled" |
| `--limit` | None | Process only first N images (omit for all) |
| `--overwrite` | False | Delete and recreate `--dst` if it exists |

### autolabel_shoes.py

```bash
python scripts/autolabel_shoes.py \
  --src data/cleaned_fragment_1/persons_supplemented \
  --dst data/cleaned_fragment_1/ppe_7class_export \
  --box-thresh 0.30 \
  --nms-thresh 0.45 \
  --foot-ratio 0.25 \
  --limit 100 \
  --overwrite
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--src` | `data/cleaned_fragment_1/persons_supplemented` | Source dataset (output of step 1) |
| `--dst` | `data/cleaned_fragment_1/ppe_7class_export` | Output path (7-class format) |
| `--box-thresh` | `0.30` | GDINO box confidence for boots |
| `--nms-thresh` | `0.45` | NMS IoU threshold (higher = more detections kept) |
| `--foot-ratio` | `0.25` | Fraction of person height used as "foot region" for NO-boots inference |
| `--limit` | None | Process only first N images |
| `--overwrite` | False | Delete and recreate `--dst` if it exists |

---

## Output Format

### persons_supplemented/
```
persons_supplemented/
├── images/          (12,319 JPGs, copied from source)
├── labels/          (5-class YOLO format)
└── data.yaml        (5 classes: Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest)
```

### ppe_7class_export/
```
ppe_7class_export/
├── images/          (12,319 JPGs, copied from persons_supplemented)
├── labels/          (7-class YOLO format)
└── data.yaml        (7 classes: + Safety Boots, NO-Safety Boots)
```

**Class IDs in ppe_7class_export:**
```
0: Hardhat
1: NO-Hardhat
2: NO-Safety Vest
3: Person
4: Safety Vest
5: Safety Boots       ← new
6: NO-Safety Boots   ← new
```

---

## Next Steps

1. **Inspect Results**
   - Open a few label files and check for quality
   - Visualize bboxes using a tool like [LabelImg](https://github.com/heartexcel/labelImg)

2. **Train**
   - Update a config file in `config/` to point to `ppe_7class_export/data.yaml`
   - Run training: `python src/train.py`

3. **Evaluate**
   - Use `scripts/benchmark_report.py` or `scripts/class_metrics.py` to compare 5-class vs 7-class models

---

## Troubleshooting

### "Module not found: transformers"
Install the required packages:
```bash
.venv/Scripts/pip install transformers supervision
```

### Script runs slowly
- Grounding DINO runs on GPU by default. If you see low GPU utilization, ensure PyTorch CUDA is properly installed
- Use `--limit N` to test on a small subset first

### High false positives (too many detections)
- Lower `--box-thresh` (e.g., 0.25 instead of 0.35) to include more detections
- Lower `--nms-thresh` to filter overlapping detections more aggressively

### Too few detections
- Raise `--box-thresh` (e.g., 0.45) to only include confident detections
- Try different prompts in the scripts (e.g., add "work shoes" to the shoe detection prompt)

---

## Model Details

**Grounding DINO Model:** `IDEA-Research/grounding-dino-base`
- Zero-shot object detection (no fine-tuning needed)
- Works with text prompts
- ~100M parameters, runs on CUDA or CPU
- Downloaded automatically on first run (~350 MB)

**Prompts Used:**
- **Persons:** `"person ."`
- **Shoes:** `"safety boots . safety shoes . work boots . steel toe boots ."`

---

## Advanced: Custom Prompts

To modify the detection prompts, edit the scripts:

**For persons** (line ~67 in `autolabel_persons.py`):
```python
prompt = "person ."  # Add more options: "person . worker . person working ."
```

**For shoes** (line ~77 in `autolabel_shoes.py`):
```python
prompt = "safety boots . safety shoes . work boots . steel toe boots . safety footwear ."
```

The `.` token tells GDINO to detect multiple objects matching those phrases.

---

## License

Uses HuggingFace Transformers and Grounding DINO (IDEA-Research/grounding-dino-base)
