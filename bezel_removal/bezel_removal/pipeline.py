import json
from pathlib import Path

import cv2
import numpy as np

from .inpainting import Inpainter, feathered_composite
from .io_utils import (
    is_video,
    iter_image_folder,
    iter_video,
    make_video_writer,
    video_fps,
)
from .mask_model import FolderMaskProvider, ModelMaskProvider
from .mask_refinement import RefinementConfig, make_overlay, refine_mask


def build_mask_provider(args):
    if args.mask_source == "folder":
        if not args.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask folder does not exist: {args.mask_dir}")
        return FolderMaskProvider(args.mask_dir)

    return ModelMaskProvider(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.model_threshold,
        morph_size=args.model_dilate,
        max_components=args.model_max_components,
        min_component_area=args.model_min_component_area,
        min_component_height_ratio=args.model_min_component_height_ratio,
        min_component_aspect=args.model_min_component_aspect,
        max_component_width_ratio=args.model_max_component_width_ratio,
        no_geometry_filter=args.no_model_geometry_filter,
    )


def build_refinement_config(args):
    return RefinementConfig(
        mask_threshold=args.mask_threshold,
        extra_dilate=args.extra_dilate,
        close_size=args.close_size,
        line_width=args.line_width,
        line_y_padding=args.line_y_padding,
        keep_predicted_width=args.keep_predicted_width,
        max_components=args.max_components,
        min_component_area=args.min_component_area,
        min_component_height_ratio=args.min_component_height_ratio,
        min_component_aspect=args.min_component_aspect,
        max_component_width_ratio=args.max_component_width_ratio,
    )


def process_frame(frame, frame_name, mask_provider, inpainter, refine_config, args):
    predicted_mask = mask_provider.mask_for(frame, frame_name)
    if predicted_mask is None:
        final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    else:
        final_mask = refine_mask(predicted_mask, frame.shape, refine_config)

    if np.any(final_mask):
        raw_result = inpainter.inpaint(frame, final_mask, args.inpaint_radius)
        result = feathered_composite(frame, raw_result, final_mask, args.feather)
    else:
        result = frame.copy()

    return result, final_mask


def save_debug_outputs(output_dir, frame_name, original, mask, result, args):
    if args.save_masks:
        mask_path = output_dir / "masks" / f"{Path(frame_name).stem}.png"
        cv2.imwrite(str(mask_path), mask)

    if args.save_overlays:
        overlay_path = output_dir / "overlays" / frame_name
        cv2.imwrite(str(overlay_path), make_overlay(original, mask))

    if args.save_comparison:
        comparison = np.concatenate([original, result], axis=1)
        comparison_path = output_dir / "comparisons" / frame_name
        cv2.imwrite(str(comparison_path), comparison)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        (args.output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if args.save_comparison:
        (args.output_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    mask_provider = build_mask_provider(args)
    refine_config = build_refinement_config(args)
    inpainter = Inpainter(args.backend)

    summary = {
        "input": str(args.input),
        "mask_source": args.mask_source,
        "backend": inpainter.backend,
        "frames_processed": 0,
        "frames_with_mask": 0,
        "output_dir": str(args.output_dir),
    }

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Mask source: {args.mask_source}")
    print(f"[INFO] Backend: {inpainter.backend}")
    print(f"[INFO] Output: {args.output_dir}")

    if is_video(args.input):
        run_video(args, mask_provider, inpainter, refine_config, summary)
    else:
        run_image_folder(args, mask_provider, inpainter, refine_config, summary)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] Frames: {summary['frames_processed']}")
    print(f"[DONE] Frames with mask: {summary['frames_with_mask']}")
    print(f"[DONE] Output: {args.output_dir}")


def run_image_folder(args, mask_provider, inpainter, refine_config, summary):
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {args.input}")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_iter = iter_image_folder(
        args.input,
        frame_step=args.frame_step,
        max_frames=args.limit,
        only=args.only,
    )

    for item in frame_iter:
        result, mask = process_frame(
            item["frame"],
            item["frame_name"],
            mask_provider,
            inpainter,
            refine_config,
            args,
        )
        cv2.imwrite(str(frames_dir / item["frame_name"]), result)
        save_debug_outputs(
            args.output_dir,
            item["frame_name"],
            item["frame"],
            mask,
            result,
            args,
        )
        update_summary(summary, mask)

        if summary["frames_processed"] == 1 or summary["frames_processed"] % 25 == 0:
            print(f"[INFO] Processed {summary['frames_processed']} frames")


def run_video(args, mask_provider, inpainter, refine_config, summary):
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.input}")
    if args.mask_source == "folder":
        print("[WARN] Folder masks are matched by generated frame names for video input.")

    input_fps = video_fps(args.input, 30.0)
    output_fps = args.output_fps or input_fps / max(1, args.frame_step)
    writer = None

    try:
        for item in iter_video(args.input, args.frame_step, args.limit):
            result, mask = process_frame(
                item["frame"],
                item["frame_name"],
                mask_provider,
                inpainter,
                refine_config,
                args,
            )

            if writer is None:
                writer = make_video_writer(
                    args.output_dir / "bezel_removed.mp4",
                    output_fps,
                    result.shape,
                )
            writer.write(result)
            save_debug_outputs(
                args.output_dir,
                item["frame_name"],
                item["frame"],
                mask,
                result,
                args,
            )
            update_summary(summary, mask)

            if summary["frames_processed"] == 1 or summary["frames_processed"] % 25 == 0:
                print(f"[INFO] Processed {summary['frames_processed']} frames")
    finally:
        if writer is not None:
            writer.release()


def update_summary(summary, mask):
    summary["frames_processed"] += 1
    if np.any(mask):
        summary["frames_with_mask"] += 1
