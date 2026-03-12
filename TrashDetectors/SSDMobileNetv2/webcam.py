#!/usr/bin/env python3
"""
Real-time webcam inference for a TF1 Object Detection frozen graph
(e.g., ssd_mobilenet_v2_taco_2018_03_29.pb from your notebook).

Usage (example):
  python webcam.py \
      --model /path/to/ssd_mobilenet_v2_taco_2018_03_29.pb \
      --labels /path/to/labelmap.pbtxt \
      --score 0.5 \
      --camera 0

Dependencies:
  pip install tensorflow==2.* opencv-python numpy
  # (We use tf.compat.v1 for TF1 graphs; TF2 works with eager disabled.)
"""

import argparse
import time
import re
import cv2
import numpy as np
import tensorflow as tf

def load_labelmap(pbtxt_path):
    """
    Minimal parser for labelmap.pbtxt (item { id: X name: "Y" } ...)
    Returns: dict[int, str]
    """
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        text = f.read()
    items = re.findall(r'item\s*{[^}]*}', text, flags=re.S)
    id2name = {}
    for it in items:
        mid = re.search(r'id\s*:\s*(\d+)', it)
        mname = re.search(r'name\s*:\s*"?([^\n"]+)"?', it)
        if mid and mname:
            id2name[int(mid.group(1))] = mname.group(1).strip()
    return id2name

def color_for_class(cid):
    # Stable pseudo-random BGR color for a class id
    np.random.seed(cid * 97 + 13)
    return tuple(int(x) for x in np.random.randint(64, 256, size=3))

def draw_detections(frame_bgr, boxes, classes, scores, id2name, score_thresh=0.5):
    """
    boxes: [N,4] in normalized y1,x1,y2,x2
    classes: [N] (float -> int)
    scores: [N]
    """
    h, w = frame_bgr.shape[:2]
    for y1, x1, y2, x2, c, s in zip(boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], classes, scores):
        if s < score_thresh:
            continue
        cls = int(c)
        name = id2name.get(cls, str(cls))
        x1p, y1p, x2p, y2p = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        color = color_for_class(cls)
        thick = max(2, int(round(0.002 * (h + w) / 2)))
        cv2.rectangle(frame_bgr, (x1p, y1p), (x2p, y2p), color, thick)

        label = f"{name}: {s:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_txt = max(th + 6, y1p)
        cv2.rectangle(frame_bgr, (x1p, y1_txt - th - 6), (x1p + tw + 4, y1_txt + 4), color, -1)
        cv2.putText(frame_bgr, label, (x1p + 2, y1_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def build_graph(pb_path):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, "rb") as fid:
            serialized = fid.read()
            od_graph_def.ParseFromString(serialized)
            tf.import_graph_def(od_graph_def, name="")
    return graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to frozen inference graph (.pb)")
    ap.add_argument("--labels", required=True, help="Path to labelmap.pbtxt")
    ap.add_argument("--score", type=float, default=0.5, help="Score threshold")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--width", type=int, default=0, help="Resize capture width (keep aspect). 0 = native")
    ap.add_argument("--show_fps", action="store_true", help="Overlay FPS counter")
    args = ap.parse_args()

    id2name = load_labelmap(args.labels)
    print(f"Loaded {len(id2name)} classes from {args.labels}")

    detection_graph = build_graph(args.model)

    # Tensor names typical for TF1 Object Detection API frozen graphs
    input_name = "image_tensor:0"
    boxes_name = "detection_boxes:0"
    scores_name = "detection_scores:0"
    classes_name = "detection_classes:0"
    num_name = "num_detections:0"

    gpu_opts = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_opts)
    sess = tf.compat.v1.Session(graph=detection_graph, config=config)

    # Get tensors
    image_tensor = detection_graph.get_tensor_by_name(input_name)
    boxes_tensor = detection_graph.get_tensor_by_name(boxes_name)
    scores_tensor = detection_graph.get_tensor_by_name(scores_name)
    classes_tensor = detection_graph.get_tensor_by_name(classes_name)
    num_tensor = detection_graph.get_tensor_by_name(num_name)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)  # CAP_DSHOW helps on Windows; harmless elsewhere
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    # Optional resize for speed
    target_w = args.width if args.width and args.width > 160 else 0

    print("Press 'q' to quit.")
    t0 = time.time()
    frame_count = 0
    fps = 0.0
    EMA = 0.9  # smoothing for fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if target_w:
                h, w = frame.shape[:2]
                scale = target_w / float(w)
                frame = cv2.resize(frame, (target_w, int(h * scale)))

            # Inference expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = np.expand_dims(rgb, axis=0)

            t_infer0 = time.time()
            boxes, scores, classes, num = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
                feed_dict={image_tensor: batch}
            )
            t_infer1 = time.time()

            boxes = boxes[0]
            scores = scores[0]
            classes = classes[0]

            draw_detections(frame, boxes, classes, scores, id2name, score_thresh=args.score)

            # FPS
            dt = t_infer1 - t_infer0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps = EMA * fps + (1 - EMA) * inst_fps
            frame_count += 1

            if args.show_fps:
                txt = f"FPS: {fps:5.1f}"
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Webcam Detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sess.close()
        t1 = time.time()
        if frame_count:
            print(f"Avg FPS over {frame_count} frames: {frame_count / (t1 - t0):.2f}")

if __name__ == "__main__":
    main()

