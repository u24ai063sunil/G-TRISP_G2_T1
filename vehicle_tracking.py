"""
Vehicle Detection, Tracking, and Trajectory Extraction (Enhanced)
===============================================================
Model      : YOLOv8x (larger, more accurate, pretrained on COCO)
Tracker    : ByteTrack (Supervision)
Output     : outputs/annotated_video.mp4 + outputs/trajectory.csv
Enhancements:
- More vehicle types (car, motorcycle, bus, truck, bicycle, person, train)
- Improved tracking with larger YOLOv8x model
- Output folder: outputs
Author     : Enhanced by Copilot
"""

import cv2
import csv
import os
import argparse
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VEHICLE_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"
}

# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection & Tracking (Enhanced)")
    parser.add_argument("--video",       type=str, default="traffic_video.mp4", help="Path to input video")
    parser.add_argument("--start_frame", type=int, default=0,    help="Start frame (inclusive)")
    parser.add_argument("--end_frame",   type=int, default=-1,   help="End frame (-1 = till end)")
    parser.add_argument("--conf",        type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--output_dir",  type=str, default="outputs", help="Output directory")
    return parser.parse_args()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_video = os.path.join(args.output_dir, "annotated_video.mp4")
    out_csv   = os.path.join(args.output_dir, "trajectory.csv")

    # ---------- Load model ----------
    print("[INFO] Loading YOLOv8x model …")
    model = YOLO("yolov8x.pt")  # larger, more accurate model

    # ---------- Open video ----------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = max(0, args.start_frame)
    end   = total_frames if args.end_frame == -1 else min(args.end_frame, total_frames)
    print(f"[INFO] Processing frames {start} → {end}  |  FPS={fps:.1f}  |  {width}×{height}")

    # ---------- Video writer ----------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    # ---------- Tracker & Annotators ----------
    tracker    = sv.ByteTrack()
    box_ann    = sv.BoxAnnotator(thickness=2)
    label_ann  = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann  = sv.TraceAnnotator(thickness=2, trace_length=40)

    # ---------- CSV writer ----------
    csv_file = open(out_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "vehicle_class",
                         "x1", "y1", "x2", "y2",
                         "centroid_x", "centroid_y"])

    # ---------- Seek to start frame ----------
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start

    print("[INFO] Running detection + tracking …")
    while frame_idx < end:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO inference ────────────────────────────────────────────────
        results = model(frame, conf=args.conf, verbose=False)[0]

        # Filter only vehicle/person classes
        mask = np.isin(results.boxes.cls.cpu().numpy(),
                       list(VEHICLE_CLASSES.keys()))

        boxes_xyxy  = results.boxes.xyxy.cpu().numpy()[mask]
        confs       = results.boxes.conf.cpu().numpy()[mask]
        class_ids   = results.boxes.cls.cpu().numpy()[mask].astype(int)

        # ── Build Supervision Detections ─────────────────────────────────
        detections = sv.Detections(
            xyxy      = boxes_xyxy,
            confidence= confs,
            class_id  = class_ids,
        )

        # ── ByteTrack update ─────────────────────────────────────────────
        detections = tracker.update_with_detections(detections)

        # ── Build labels ────────────────────────────────────────────────
        labels = [
            f"ID:{tid} {VEHICLE_CLASSES.get(cid, 'object')} {conf:.2f}"
            for tid, cid, conf in zip(
                detections.tracker_id,
                detections.class_id,
                detections.confidence,
            )
        ]

        # ── Annotate frame ──────────────────────────────────────────────
        annotated = frame.copy()
        annotated = trace_ann.annotate(annotated, detections)
        annotated = box_ann.annotate(annotated, detections)
        annotated = label_ann.annotate(annotated, detections, labels=labels)

        # Frame counter overlay
        cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(annotated)

        # ── Write CSV rows ──────────────────────────────────────────────
        for xyxy, tid, cid in zip(detections.xyxy,
                                  detections.tracker_id,
                                  detections.class_id):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            csv_writer.writerow([frame_idx, tid,
                                 VEHICLE_CLASSES.get(cid, "object"),
                                 x1, y1, x2, y2, cx, cy])

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = (frame_idx - start) / (end - start) * 100
            print(f"  … {frame_idx}/{end}  ({pct:.1f}%)")

    # ---------- Cleanup ----------
    cap.release()
    writer.release()
    csv_file.close()

    print(f"\n[DONE] Annotated video  → {out_video}")
    print(f"[DONE] Trajectory CSV   → {out_csv}")

    # Quick summary
    df = pd.read_csv(out_csv)
    print(f"\n── Trajectory Summary ──────────────────────────────")
    print(f"  Total detections  : {len(df)}")
    print(f"  Unique track IDs  : {df['track_id'].nunique()}")
    print(f"  Vehicle classes   : {df['vehicle_class'].value_counts().to_dict()}")
    print(f"────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
