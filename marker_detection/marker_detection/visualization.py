import cv2
import numpy as np


SCREEN_COLORS = {
    "left": (0, 255, 0),
    "center": (255, 0, 0),
    "right": (0, 0, 255),
    None: (255, 255, 255),
}

SCREEN_LABELS = {
    "left": "L",
    "center": "C",
    "right": "R",
}

ANNOTATION_STYLES = {
    "screen": {
        "line_divisor": 120,
        "min_line": 4,
        "halo_extra": 4,
        "center_divisor": 90,
        "min_center": 5,
        "font_divisor": 420,
        "min_font": 0.9,
        "text_divisor": 210,
        "min_text": 2,
        "padding_divisor": 120,
        "min_padding": 5,
    },
    "headcam": {
        "line_divisor": 260,
        "min_line": 2,
        "halo_extra": 2,
        "center_divisor": 190,
        "min_center": 3,
        "font_divisor": 1050,
        "min_font": 0.55,
        "text_divisor": 520,
        "min_text": 1,
        "padding_divisor": 240,
        "min_padding": 3,
    },
}


def draw_tags(frame, tags, label_screens=True, style="screen"):
    out = frame.copy()
    height, width = out.shape[:2]
    metrics = annotation_metrics(min(height, width), style)

    for tag in tags:
        corners = np.asarray(tag["corners"], dtype=np.int32)
        screen = tag.get("screen")
        color = SCREEN_COLORS.get(screen, (255, 255, 255))
        cv2.polylines(
            out,
            [corners],
            True,
            (0, 0, 0),
            metrics["halo_thickness"],
            cv2.LINE_AA,
        )
        cv2.polylines(
            out,
            [corners],
            True,
            color,
            metrics["line_thickness"],
            cv2.LINE_AA,
        )

        center = np.asarray(tag["center"], dtype=np.int32)
        cv2.circle(
            out,
            tuple(center),
            metrics["center_radius"] + 2,
            (0, 0, 0),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            out,
            tuple(center),
            metrics["center_radius"],
            color,
            -1,
            cv2.LINE_AA,
        )

        label = str(int(tag["id"]))
        if label_screens and screen is not None:
            label = f"{SCREEN_LABELS.get(screen, screen[:1].upper())}:{label}"
        if tag.get("source") == "temporal":
            label += "*"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            metrics["font_scale"],
            metrics["text_thickness"],
        )
        x, y = label_origin(
            corners,
            text_width,
            text_height,
            baseline,
            metrics["padding"],
            width,
            height,
        )
        box_top_left = (
            x - metrics["padding"],
            y - text_height - metrics["padding"],
        )
        box_bottom_right = (
            x + text_width + metrics["padding"],
            y + baseline + metrics["padding"],
        )
        cv2.rectangle(out, box_top_left, box_bottom_right, (0, 0, 0), -1)
        cv2.rectangle(
            out,
            box_top_left,
            box_bottom_right,
            color,
            metrics["line_thickness"],
        )
        cv2.putText(
            out,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            metrics["font_scale"],
            (255, 255, 255),
            metrics["text_thickness"],
            cv2.LINE_AA,
        )

    return out


def annotation_metrics(short_side, style):
    style_config = ANNOTATION_STYLES.get(style, ANNOTATION_STYLES["screen"])
    line_thickness = max(
        style_config["min_line"],
        int(round(short_side / style_config["line_divisor"])),
    )
    return {
        "line_thickness": line_thickness,
        "halo_thickness": line_thickness + style_config["halo_extra"],
        "center_radius": max(
            style_config["min_center"],
            int(round(short_side / style_config["center_divisor"])),
        ),
        "font_scale": max(
            style_config["min_font"],
            short_side / style_config["font_divisor"],
        ),
        "text_thickness": max(
            style_config["min_text"],
            int(round(short_side / style_config["text_divisor"])),
        ),
        "padding": max(
            style_config["min_padding"],
            int(round(short_side / style_config["padding_divisor"])),
        ),
    }


def label_origin(corners, text_width, text_height, baseline, padding, width, height):
    x_min = int(np.min(corners[:, 0]))
    x_max = int(np.max(corners[:, 0]))
    y_min = int(np.min(corners[:, 1]))
    y_max = int(np.max(corners[:, 1]))

    x = x_max + padding * 2
    y = y_min + text_height

    if x + text_width + padding >= width:
        x = x_min - text_width - padding * 2
    if x < padding:
        x = x_min

    box_top = y - text_height - padding
    if box_top < 0:
        y = y_max + text_height + padding * 2
    if y + baseline + padding >= height:
        y = y_min - padding * 2
    if y - text_height - padding < 0:
        y = text_height + padding

    x = max(padding, min(x, width - text_width - padding))
    y = max(text_height + padding, min(y, height - baseline - padding))
    return int(x), int(y)
