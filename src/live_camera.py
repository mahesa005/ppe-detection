#!/usr/bin/env python3
"""
PPE Detection - Live Camera Inference
Run trained YOLO models on live webcam feed with real-time detection
"""

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import time


def get_available_models():
    """List all available trained models"""
    models = []
    detect_dir = Path("runs/detect")

    if not detect_dir.exists():
        return models

    for run_dir in sorted(detect_dir.iterdir()):
        model_path = run_dir / "weights" / "best.pt"
        if model_path.exists():
            models.append({
                "name": run_dir.name,
                "path": str(model_path),
            })

    return models


def run_live_detection(model_path, conf_threshold=0.5, imgsz=640):
    """Run live detection on webcam feed"""

    # Validate model exists
    if not Path(model_path).exists():
        print(f"✗ Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get class names
    class_names = model.names
    print(f"Classes: {class_names}")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # FPS tracking
    fps_deque = deque(maxlen=30)

    print("\n" + "="*60)
    print("LIVE DETECTION STARTED")
    print("="*60)
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Image size: {imgsz}x{imgsz}")
    print("\nControls:")
    print("  SPACE   - Pause/Resume")
    print("  C       - Capture screenshot")
    print("  ESC/Q   - Quit")
    print("="*60 + "\n")

    paused = False
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            if not paused:
                # Run inference
                results = model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)

                # Draw results
                annotated_frame = results[0].plot()

                # Get detections info
                detections = results[0].boxes
                det_info = f"Detections: {len(detections)}"

                # Add info text
                cv2.putText(annotated_frame, det_info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                frame_time = time.time() - start_time
                fps_deque.append(1.0 / frame_time if frame_time > 0 else 0)
            else:
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "PAUSED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display FPS
            if fps_deque:
                avg_fps = sum(fps_deque) / len(fps_deque)
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}",
                           (annotated_frame.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("PPE Detection - Live Camera", annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord(' '):  # SPACE - Pause/Resume
                paused = not paused
                status = "PAUSED" if paused else "RUNNING"
                print(f"Detection {status}")
            elif key == ord('c'):  # C - Capture
                filename = f"capture_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✓ Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("LIVE DETECTION STOPPED")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run live PPE detection on webcam feed"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model file (e.g., runs/detect/ppe_v2-4/weights/best.pt)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available trained models"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference (default: 640)"
    )

    args = parser.parse_args()

    # List available models
    if args.list_models:
        models = get_available_models()
        if not models:
            print("✗ No trained models found in runs/detect/")
            return

        print("\n" + "="*60)
        print("AVAILABLE TRAINED MODELS")
        print("="*60)
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Path: {model['path']}")
        print("="*60 + "\n")
        return

    # Get model path
    if args.model:
        model_path = args.model
    else:
        # Use latest model if not specified
        models = get_available_models()
        if not models:
            print("✗ No trained models found. Run train.py first or use --list-models to see available models")
            return
        model_path = models[-1]["path"]  # Use latest
        print(f"Using model: {models[-1]['name']}")

    # GPU check
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU detected — inference will be slower")

    # Run detection
    run_live_detection(model_path, conf_threshold=args.conf, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
