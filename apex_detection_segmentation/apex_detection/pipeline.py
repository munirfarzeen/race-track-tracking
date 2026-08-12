import json
import shutil

import cv2
import numpy as np

from .apex import estimate_turn, find_apex
from .boundaries import extract_left_right_boundaries, fit_poly
from .io_utils import list_frames, load_points
from .segmentation import (
    build_predictor,
    enable_tf32_if_supported,
    prepare_sam2_frame_dir,
    resolve_device,
    segment_video,
)
from .visualization import visualize


def process_frame(frame, mask, poly_degree):
    left_pts, right_pts = extract_left_right_boundaries(mask)
    left_coeff = fit_poly(left_pts, degree=poly_degree)
    right_coeff = fit_poly(right_pts, degree=poly_degree)

    turn = "unknown"
    apex_point, apex_side = None, None

    if left_coeff is not None and right_coeff is not None:
        turn = estimate_turn(left_coeff, right_coeff, frame.shape[0])
        apex_point, apex_side = find_apex(left_coeff, right_coeff, turn, frame.shape[0])

    return left_coeff, right_coeff, turn, apex_point, apex_side


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list_frames(args.input)
    positive_points, negative_points = load_points(args.points_file)

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Frames found: {len(frame_paths)}")
    print(
        f"[INFO] Points file: {args.points_file} "
        f"({len(positive_points)} positive / {len(negative_points)} negative)"
    )

    sam_frame_dir = prepare_sam2_frame_dir(frame_paths, args.output_dir / "sam2_frames")

    device = resolve_device(args.device)
    enable_tf32_if_supported(device)
    print(f"[INFO] Using device: {device}")

    predictor = build_predictor(args.model_cfg, args.checkpoint, device)

    video_segments = segment_video(
        predictor,
        sam_frame_dir,
        positive_points,
        negative_points,
        object_id=args.object_id,
        annotation_frame_idx=args.annotation_frame,
        device=device,
    )

    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "points_file": str(args.points_file),
        "device": device,
        "frames_total": len(frame_paths),
        "frames_segmented": 0,
        "turn_counts": {"left": 0, "right": 0, "straight": 0, "unknown": 0},
    }

    results_path = args.output_dir / "apex_results.jsonl"
    with results_path.open("w") as results_file:
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"[WARN] Could not read frame: {frame_path}")
                continue

            mask = video_segments.get(idx, {}).get(args.object_id)
            if mask is None:
                print(f"[WARN] No SAM2 mask for frame {idx}: {frame_path.name}")
                continue

            mask = mask.astype(np.uint8)
            left_coeff, right_coeff, turn, apex_point, apex_side = process_frame(
                frame, mask, args.poly_degree
            )

            if args.save_masks:
                cv2.imwrite(str(masks_dir / f"{frame_path.stem}_mask.png"), mask * 255)
            if args.save_overlays:
                overlay = visualize(frame, mask, left_coeff, right_coeff, apex_point, apex_side, turn)
                cv2.imwrite(str(overlays_dir / f"{frame_path.stem}_overlay.png"), overlay)

            record = {
                "frame_index": idx,
                "frame_name": frame_path.name,
                "turn": turn,
                "apex_side": apex_side,
                "apex_x": float(apex_point[0]) if apex_point is not None else None,
                "apex_y": float(apex_point[1]) if apex_point is not None else None,
            }
            results_file.write(json.dumps(record) + "\n")

            summary["frames_segmented"] += 1
            summary["turn_counts"][turn] += 1

            if (idx + 1) == 1 or (idx + 1) % 25 == 0 or (idx + 1) == len(frame_paths):
                print(f"[INFO] {idx + 1}/{len(frame_paths)} {frame_path.name} | turn={turn} | side={apex_side}")

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if not args.keep_sam2_frame_cache:
        shutil.rmtree(sam_frame_dir, ignore_errors=True)

    print(f"[DONE] Frames segmented: {summary['frames_segmented']}/{summary['frames_total']}")
    print(f"[DONE] Turn counts: {summary['turn_counts']}")
    print(f"[DONE] Output: {args.output_dir}")
