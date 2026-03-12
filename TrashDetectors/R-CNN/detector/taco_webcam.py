#!/usr/bin/env python3
"""
Real-time webcam inference with TACO Mask R-CNN weights.

Usage (from inside the detector/ directory):

    python3 taco_webcam.py \
        --weights ../models/logs/mask_rcnn_taco_10.h5 \
        --class_map ./taco_config/map_10.csv \
        --video_source 0 \
        --min_confidence 0.5

Press 'q' to quit.
"""

import os
import sys
import time
import csv
import argparse

import numpy as np
import cv2

# Make sure we can import the TACO / Matterport Mask R-CNN modules
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from config import Config
from model import MaskRCNN   # this is the Matterport Mask R-CNN (slightly modified)


def load_class_names(class_map_path):
    """
    Read the TACO class map CSV and build a class_names list.

    The map_*.csv files are of the form:
        original_name,target_name

    We only need the target names; order here only affects the label text,
    not the underlying detections.
    """
    class_names = ["BG"]  # background
    seen = set()

    with open(class_map_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            target = row[1].strip()
            if target and target not in seen:
                seen.add(target)
                class_names.append(target)

    return class_names


def random_colors(N, seed=42):
    np.random.seed(seed)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(N)]


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply a colored mask on a BGR image (OpenCV format).
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            image[:, :, c],
        )
    return image


def run_webcam(weights_path, class_map_path, video_source, min_confidence):
    # Read class names from CSV
    class_names = load_class_names(class_map_path)
    num_classes = len(class_names)

    # Define a config for inference
    class TacoWebcamConfig(Config):
        NAME = "taco_webcam"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = num_classes  # background + target classes
        DETECTION_MIN_CONFIDENCE = min_confidence
        USE_MINI_MASK = False

    config = TacoWebcamConfig()
    config.display()

    # Create model in inference mode
    model = MaskRCNN(mode="inference", config=config,
                     model_dir=os.path.join(ROOT_DIR, "logs"))

    # Load trained weights
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path, weights_path, by_name=True)

    # Colors for each class
    colors = random_colors(num_classes)

    # Open webcam / video
    try:
        # allow integer index (0, 1, ...) or file path
        src = int(video_source)
    except ValueError:
        src = video_source

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{video_source}'")
        return

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break

            # Convert BGR (OpenCV) → RGB (Mask R-CNN)
            rgb = frame[:, :, ::-1]

            t0 = time.time()
            results = model.detect([rgb], verbose=0)[0]
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0

            rois = results["rois"]
            masks = results["masks"]
            class_ids = results["class_ids"]
            scores = results["scores"]

            N = rois.shape[0]
            for i in range(N):
                y1, x1, y2, x2 = rois[i]
                class_id = int(class_ids[i])
                score = float(scores[i])
                mask = masks[:, :, i]

                # Color for this instance
                color = colors[class_id % len(colors)]

                # Apply mask and draw bounding box / label
                frame = apply_mask(frame, mask, color, alpha=0.4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"id_{class_id}"

                label = f"{class_name} {score:.2f}"
                y = max(y1 - 10, 15)
                cv2.putText(
                    frame,
                    label,
                    (x1, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Put FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("TACO Mask R-CNN (webcam)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Mask R-CNN inference on webcam using TACO weights."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to weights .h5 file "
             "(e.g. ../models/logs/mask_rcnn_taco_10.h5 from the 1.0 release).",
    )
    parser.add_argument(
        "--class_map",
        default=os.path.join(ROOT_DIR, "taco_config", "map_10.csv"),
        help="Path to TACO class map CSV (default: taco_config/map_10.csv).",
    )
    parser.add_argument(
        "--video_source",
        default="0",
        help="Webcam index (0, 1, ...) or path to a video file. Default: 0.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Detection minimum confidence threshold. Default: 0.5",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(
        weights_path=args.weights,
        class_map_path=args.class_map,
        video_source=args.video_source,
        min_confidence=args.min_confidence,
    )

