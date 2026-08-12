import numpy as np


def curvature(coeff, ys):
    poly = np.poly1d(coeff)
    first_deriv = np.polyder(poly, 1)
    second_deriv = np.polyder(poly, 2)

    xp = first_deriv(ys)
    xpp = second_deriv(ys)

    return np.abs(xpp) / np.power(1.0 + xp * xp, 1.5)


def estimate_turn(left_coeff, right_coeff, height, y_start_ratio=0.45, y_end_ratio=0.85, straight_threshold=1e-4):
    """Fits a parabola to the track centerline and reads its curvature sign."""
    ys = np.linspace(int(y_start_ratio * height), int(y_end_ratio * height), 120)
    center_x = 0.5 * (np.polyval(left_coeff, ys) + np.polyval(right_coeff, ys))

    centerline_coeff = np.polyfit(ys, center_x, 2)
    curvature_sign = centerline_coeff[0]

    if abs(curvature_sign) < straight_threshold:
        return "straight"
    return "right" if curvature_sign > 0 else "left"


def find_apex(left_coeff, right_coeff, turn, height, y_start_ratio=0.45, y_end_ratio=0.85, n=250):
    """
    Apex candidate is the point of maximum curvature on the inside boundary
    of the turn (left edge for a left turn, right edge for a right turn).
    """
    if turn == "left":
        coeff, side = left_coeff, "left"
    elif turn == "right":
        coeff, side = right_coeff, "right"
    else:
        return None, None

    ys = np.linspace(int(y_start_ratio * height), int(y_end_ratio * height), n)
    k = curvature(coeff, ys)

    best_idx = int(np.argmax(k))
    apex_y = ys[best_idx]
    apex_x = np.polyval(coeff, apex_y)

    return np.array([apex_x, apex_y], dtype=np.float32), side
