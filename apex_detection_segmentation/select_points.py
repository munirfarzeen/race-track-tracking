#!/usr/bin/env python3
"""Click positive/negative seed points on a frame and save them as a points-file
JSON usable with `detect_apex.py --points-file`."""
import argparse
import json
from pathlib import Path

import cv2


WINDOW_NAME = "click points"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Frame to click points on.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the points JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    positive_points = []
    negative_points = []
    mode = {"value": "pos"}

    def redraw():
        display = img.copy()
        for x, y in positive_points:
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
        for x, y in negative_points:
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow(WINDOW_NAME, display)

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if mode["value"] == "pos":
            positive_points.append([x, y])
            print("POS:", [x, y])
        else:
            negative_points.append([x, y])
            print("NEG:", [x, y])
        redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Controls:")
    print("  p   = switch to positive/asphalt mode (default)")
    print("  n   = switch to negative/non-track mode")
    print("  u   = undo last point in the current mode")
    print("  q/Esc = quit and save")

    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("p"):
            mode["value"] = "pos"
            print("Mode: POSITIVE")
        elif key == ord("n"):
            mode["value"] = "neg"
            print("Mode: NEGATIVE")
        elif key == ord("u"):
            points = positive_points if mode["value"] == "pos" else negative_points
            if points:
                removed = points.pop()
                print("Undid:", removed)
            redraw()
        elif key in (ord("q"), 27):
            break
        elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    if not positive_points:
        raise RuntimeError("No positive points were clicked; at least one is required.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "_notes": f"Seed points clicked on {args.image.name}.",
                "positive_points": positive_points,
                "negative_points": negative_points,
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"\nSaved {len(positive_points)} positive / {len(negative_points)} "
        f"negative points to {args.output}"
    )


if __name__ == "__main__":
    main()
