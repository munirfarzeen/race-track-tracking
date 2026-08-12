import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def resolve_device(preferred="auto"):
    if preferred != "auto":
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_tf32_if_supported(device):
    """Ampere+ GPUs get a free speedup from TF32 matmuls, as recommended by SAM2."""
    if device != "cuda" or not torch.cuda.is_available():
        return
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_predictor(model_cfg, checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    return build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)


def prepare_sam2_frame_dir(frame_paths, sam_frame_dir, jpeg_quality=95):
    """
    SAM2's video predictor expects a directory of zero-padded JPEG frames
    (000000.jpg, 000001.jpg, ...). JPEG inputs are copied as-is; other formats
    (e.g. PNG) are re-encoded.
    """
    sam_frame_dir = Path(sam_frame_dir)
    sam_frame_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(frame_paths):
        out_path = sam_frame_dir / f"{i:06d}.jpg"
        if path.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(path, out_path)
            continue

        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read frame: {path}")
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    return sam_frame_dir


def segment_video(
    predictor,
    sam_frame_dir,
    positive_points,
    negative_points,
    object_id,
    annotation_frame_idx,
    device,
):
    """
    Prompts SAM2 with the seed clicks on `annotation_frame_idx` and propagates
    the resulting mask across the whole frame directory.

    Returns {frame_idx: {object_id: bool_mask}}.
    """
    labels = np.concatenate([
        np.ones(len(positive_points), dtype=np.int32),
        np.zeros(len(negative_points), dtype=np.int32),
    ])
    points = np.concatenate([positive_points, negative_points], axis=0)

    video_segments = {}

    with torch.inference_mode():
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_ctx:
            state = predictor.init_state(video_path=str(sam_frame_dir))

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=annotation_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks = {}
                for i, obj_id in enumerate(obj_ids):
                    mask = (mask_logits[i] > 0.0).cpu().numpy()
                    masks[obj_id] = np.squeeze(mask)
                video_segments[frame_idx] = masks

    return video_segments
