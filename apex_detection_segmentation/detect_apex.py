#!/usr/bin/env python3
import argparse
from pathlib import Path

from apex_detection.pipeline import run


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path("/home/shoaib/work/research/Otto_work/data_zandvoort")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "track_apex_zandvoort"
DEFAULT_CHECKPOINT = Path("checkpoints/sam2.1_hiera_tiny.pt")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_POINTS_FILE = REPO_ROOT / "default_points.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment the track surface with SAM2 and estimate the turn direction/apex per frame."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="Folder of frames (jpg/jpeg/png).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="SAM2 checkpoint file.")
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG, help="SAM2 model config (Hydra config name/path).")
    parser.add_argument(
        "--points-file",
        type=Path,
        default=DEFAULT_POINTS_FILE,
        help="JSON file with positive_points/negative_points seed clicks for the annotation frame.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--annotation-frame", type=int, default=0, help="Frame index the seed points are clicked on.")
    parser.add_argument("--poly-degree", type=int, default=3, help="Degree of the polynomial fit to each boundary.")
    parser.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the binary SAM2 mask per frame to <output-dir>/masks/.",
    )
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the annotated overlay per frame to <output-dir>/overlays/.",
    )
    parser.add_argument(
        "--keep-sam2-frame-cache",
        action="store_true",
        help="Keep the intermediate <output-dir>/sam2_frames/ directory instead of deleting it after the run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
