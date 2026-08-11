#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from marker_detection.detector import AprilTagDetector, TemporalTagTracker
from marker_detection.io_utils import (
    is_video,
    iter_image_folder,
    iter_video,
    make_video_writer,
)
from marker_detection.tag_layout import TAG_CONFIG
from marker_detection.visualization import draw_tags


def serialize_tag(tag):
    return {
        "tag_id": int(tag["id"]),
        "screen": tag.get("screen"),
        "source": tag.get("source", "detected"),
        "center_px": np.asarray(tag["center"]).astype(float).tolist(),
        "corners_px": np.asarray(tag["corners"]).astype(float).tolist(),
        "decision_margin": float(tag.get("decision_margin", -1.0)),
        "scale": tag.get("scale"),
        "variant": tag.get("variant"),
        "detector_id": tag.get("detector_id"),
    }


def write_csv_rows(writer, frame_info, tags):
    for tag in tags:
        corners = np.asarray(tag["corners"], dtype=np.float32)
        row = {
            "frame_index": frame_info["frame_index"],
            "frame_name": frame_info["frame_name"],
            "timestamp_sec": frame_info["timestamp_sec"],
            "tag_id": int(tag["id"]),
            "screen": tag.get("screen"),
            "source": tag.get("source", "detected"),
            "center_x": float(tag["center"][0]),
            "center_y": float(tag["center"][1]),
            "decision_margin": float(tag.get("decision_margin", -1.0)),
        }
        for idx, corner in enumerate(corners):
            row[f"corner{idx}_x"] = float(corner[0])
            row[f"corner{idx}_y"] = float(corner[1])
        writer.writerow(row)


def process_stream(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated_frames"
    if args.save_annotated and not is_video(args.input):
        annotated_dir.mkdir(parents=True, exist_ok=True)

    detector = AprilTagDetector(mode=args.mode, tag_family=args.tag_family)
    tracker = TemporalTagTracker(max_missing=args.max_temporal_missing)
    input_is_video = is_video(args.input)

    if input_is_video:
        capture = cv2.VideoCapture(str(args.input))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.input}")
        fps = capture.get(cv2.CAP_PROP_FPS) or args.output_fps
        capture.release()
        frame_iter = iter_video(args.input, args.frame_step, args.max_frames)
    else:
        fps = args.output_fps
        frame_iter = iter_image_folder(args.input, args.frame_step, args.max_frames)

    jsonl_path = args.output_dir / "detections.jsonl"
    csv_path = args.output_dir / "detections.csv"
    writer_video = None
    summary = {
        "mode": args.mode,
        "input": str(args.input),
        "tag_family": args.tag_family,
        "temporal": bool(args.temporal),
        "frames_processed": 0,
        "real_detections": 0,
        "temporal_detections": 0,
        "final_detections": 0,
        "tag_config": TAG_CONFIG,
    }

    fieldnames = [
        "frame_index",
        "frame_name",
        "timestamp_sec",
        "tag_id",
        "screen",
        "source",
        "center_x",
        "center_y",
        "decision_margin",
        "corner0_x",
        "corner0_y",
        "corner1_x",
        "corner1_y",
        "corner2_x",
        "corner2_y",
        "corner3_x",
        "corner3_y",
    ]

    with jsonl_path.open("w") as jsonl_file, csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for item in frame_iter:
            frame = item["frame"]
            real_tags = detector.detect(frame)
            if args.temporal:
                real_tags, temporal_tags, final_tags = tracker.merge(frame, real_tags)
            else:
                temporal_tags = []
                final_tags = real_tags

            frame_record = {
                "frame_index": item["frame_index"],
                "frame_name": item["frame_name"],
                "timestamp_sec": item["timestamp_sec"],
                "real": [serialize_tag(tag) for tag in real_tags],
                "temporal": [serialize_tag(tag) for tag in temporal_tags],
                "final": [serialize_tag(tag) for tag in final_tags],
                "counts_by_screen": count_by_screen(final_tags),
            }
            jsonl_file.write(json.dumps(frame_record) + "\n")
            write_csv_rows(csv_writer, item, final_tags)

            summary["frames_processed"] += 1
            summary["real_detections"] += len(real_tags)
            summary["temporal_detections"] += len(temporal_tags)
            summary["final_detections"] += len(final_tags)

            if args.save_annotated:
                annotated = draw_tags(frame, final_tags, style=args.mode)
                if input_is_video:
                    if writer_video is None:
                        writer_video = make_video_writer(
                            args.output_dir / "annotated.mp4",
                            fps / max(1, args.frame_step),
                            annotated.shape,
                        )
                    writer_video.write(annotated)
                else:
                    output_path = annotated_dir / item["frame_name"]
                    cv2.imwrite(str(output_path), annotated)

            if (
                summary["frames_processed"] == 1
                or summary["frames_processed"] % 50 == 0
            ):
                print(
                    f"[INFO] {summary['frames_processed']} frames | "
                    f"last final tags={len(final_tags)} "
                    f"screens={frame_record['counts_by_screen']}"
                )

    if writer_video is not None:
        writer_video.release()

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] Frames: {summary['frames_processed']}")
    print(f"[DONE] Final detections: {summary['final_detections']}")
    print(f"[DONE] Output: {args.output_dir}")


def count_by_screen(tags):
    counts = {"left": 0, "center": 0, "right": 0, "unknown": 0}
    for tag in tags:
        screen = tag.get("screen") or "unknown"
        counts[screen] = counts.get(screen, 0) + 1
    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect fixed AprilTags in screen recordings or headcam footage."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("screen", "headcam"), required=True)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--max-temporal-missing", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--output-fps", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input does not exist: {args.input}")
    process_stream(args)


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        raise
