import json
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_frames(frame_dir):
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame folder does not exist: {frame_dir}")

    paths = sorted(
        p for p in frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    return paths


def load_points(points_file):
    """
    Loads the SAM2 seed clicks for the annotation frame from a JSON file with
    "positive_points" / "negative_points" arrays of [x, y] pixel coordinates.
    """
    points_file = Path(points_file)
    if not points_file.is_file():
        raise FileNotFoundError(f"Points file does not exist: {points_file}")

    data = json.loads(points_file.read_text())
    positive_points = np.asarray(data.get("positive_points", []), dtype=np.float32)
    negative_points = np.asarray(data.get("negative_points", []), dtype=np.float32)

    if len(positive_points) == 0:
        raise ValueError(f"{points_file} has no positive_points")

    return positive_points, negative_points
