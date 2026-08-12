import cv2
import numpy as np

from .boundaries import sample_poly


def draw_polyline(img, coeff, color):
    if coeff is None:
        return
    pts = sample_poly(coeff, img.shape[0])
    pts = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, 3, cv2.LINE_AA)


def visualize(frame, mask, left_coeff, right_coeff, apex, side, turn):
    out = frame.copy()

    green = np.zeros_like(out)
    green[:, :, 1] = mask.astype(np.uint8) * 255
    out = cv2.addWeighted(out, 0.75, green, 0.35, 0)

    draw_polyline(out, left_coeff, (255, 0, 0))
    draw_polyline(out, right_coeff, (0, 0, 255))

    if left_coeff is not None and right_coeff is not None:
        height = out.shape[0]
        ys = np.linspace(int(0.35 * height), int(0.92 * height), 250)
        center_x = 0.5 * (np.polyval(left_coeff, ys) + np.polyval(right_coeff, ys))

        center = np.stack([center_x, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [center], False, (255, 255, 255), 2, cv2.LINE_AA)

    if apex is not None:
        x, y = int(apex[0]), int(apex[1])
        cv2.circle(out, (x, y), 16, (0, 255, 255), -1)
        cv2.circle(out, (x, y), 22, (0, 0, 0), 3)
        cv2.putText(
            out,
            f"APEX candidate: {side}",
            (max(10, x - 80), max(35, y - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"turn: {turn}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out
