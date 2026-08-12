import numpy as np


def extract_left_right_boundaries(mask, y_start_ratio=0.35, y_end_ratio=0.95, min_row_pixels=50):
    """
    Walks the SAM2 mask row by row over the lower part of the frame and takes
    the outermost lit pixel on each side as the left/right track edge.
    """
    height = mask.shape[0]
    left_pts = []
    right_pts = []

    for y in range(int(y_start_ratio * height), int(y_end_ratio * height)):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < min_row_pixels:
            continue
        left_pts.append([xs[0], y])
        right_pts.append([xs[-1], y])

    return np.asarray(left_pts), np.asarray(right_pts)


def fit_poly(points, degree=3, min_points=30):
    if points is None or len(points) < min_points:
        return None
    return np.polyfit(points[:, 1], points[:, 0], degree)


def sample_poly(coeff, height, n=250, y_start_ratio=0.35, y_end_ratio=0.92):
    ys = np.linspace(int(y_start_ratio * height), int(y_end_ratio * height), n)
    xs = np.polyval(coeff, ys)
    return np.stack([xs, ys], axis=1)
