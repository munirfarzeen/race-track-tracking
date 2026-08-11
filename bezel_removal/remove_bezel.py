#!/usr/bin/env python3
import argparse
from pathlib import Path

from bezel_removal.pipeline import run


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "2026_04_14_Zandvoo" / "frames_test_sample" / "headcam_frames"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "lol" / "bezel_mask_option_a" / "checkpoints" / "bezel_unet_best.pt"
)
DEFAULT_MASK_DIR = REPO_ROOT / "lol" / "bezel_mask_option_a" / "predicted_masks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bezel_removal" / "runs" / "headcam_bezel_removed"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove physical monitor bezels from headcam frames or video."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mask-source",
        choices=("model", "folder"),
        default="model",
        help="Use the trained U-Net checkpoint or existing predicted PNG masks.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument(
        "--backend",
        choices=("auto", "lama", "opencv"),
        default="auto",
        help="LaMa gives better generative fills; OpenCV is faster for quick tests.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--output-fps", type=float, default=None)

    parser.add_argument("--model-threshold", type=float, default=0.45)
    parser.add_argument("--model-dilate", type=int, default=13)
    parser.add_argument("--model-max-components", type=int, default=2)
    parser.add_argument("--model-min-component-area", type=int, default=900)
    parser.add_argument("--model-min-component-height-ratio", type=float, default=0.25)
    parser.add_argument("--model-min-component-aspect", type=float, default=1.6)
    parser.add_argument("--model-max-component-width-ratio", type=float, default=0.18)
    parser.add_argument("--no-model-geometry-filter", action="store_true")

    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--extra-dilate", type=int, default=1)
    parser.add_argument("--close-size", type=int, default=5)
    parser.add_argument("--line-width", type=int, default=54)
    parser.add_argument("--line-y-padding", type=int, default=8)
    parser.add_argument(
        "--keep-predicted-width",
        action="store_true",
        help="Use the full predicted blob instead of a fitted narrow centerline.",
    )
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=900)
    parser.add_argument("--min-component-height-ratio", type=float, default=0.18)
    parser.add_argument("--min-component-aspect", type=float, default=1.35)
    parser.add_argument("--max-component-width-ratio", type=float, default=0.22)

    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--feather", type=int, default=11)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--save-comparison", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
