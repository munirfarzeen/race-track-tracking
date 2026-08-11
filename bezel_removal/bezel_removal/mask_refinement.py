from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RefinementConfig:
    mask_threshold: int = 127
    extra_dilate: int = 1
    close_size: int = 5
    line_width: int = 54
    line_y_padding: int = 8
    keep_predicted_width: bool = False
    max_components: int = 2
    min_component_area: int = 900
    min_component_height_ratio: float = 0.18
    min_component_aspect: float = 1.35
    max_component_width_ratio: float = 0.22


def make_overlay(image, mask):
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image, 0.45, red, 0.55, 0)[mask_bool]
    return overlay


def keep_vertical_components(mask, config):
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []

    min_height = int(height * config.min_component_height_ratio)
    max_width = int(width * config.max_component_width_ratio)

    for label_id in range(1, num_labels):
        _, _, w, h, area = stats[label_id]
        aspect = h / max(1, w)

        if area < config.min_component_area:
            continue
        if h < min_height:
            continue
        if w > max_width:
            continue
        if aspect < config.min_component_aspect:
            continue

        score = float(area) * min(aspect, 5.0)
        candidates.append((score, label_id))

    if not candidates:
        return np.zeros_like(mask)

    keep = {
        label_id
        for _, label_id in sorted(candidates, reverse=True)[: config.max_components]
    }
    return np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)


def rebuild_as_centerlines(mask, config):
    if config.keep_predicted_width:
        return mask

    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    centerline_mask = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area <= 0:
            continue

        ys, xs = np.where(labels == label_id)
        if len(ys) < 2:
            continue

        row_ys = []
        row_xs = []
        for row in np.unique(ys):
            row_values = xs[ys == row]
            if len(row_values) == 0:
                continue
            row_ys.append(float(row))
            row_xs.append(float(np.median(row_values)))

        if len(row_ys) < 8:
            continue

        row_ys = np.array(row_ys, dtype=np.float32)
        row_xs = np.array(row_xs, dtype=np.float32)
        slope, intercept = np.polyfit(row_ys, row_xs, 1)

        residuals = np.abs(row_xs - (slope * row_ys + intercept))
        inliers = residuals <= max(10.0, config.line_width * 0.45)
        if inliers.sum() >= max(8, int(len(row_ys) * 0.55)):
            slope, intercept = np.polyfit(row_ys[inliers], row_xs[inliers], 1)

        top_y = int(max(0, y - config.line_y_padding))
        bottom_y = int(min(height - 1, y + h - 1 + config.line_y_padding))
        top_x = int(np.clip(slope * top_y + intercept, 0, width - 1))
        bottom_x = int(np.clip(slope * bottom_y + intercept, 0, width - 1))
        cv2.line(
            centerline_mask,
            (top_x, top_y),
            (bottom_x, bottom_y),
            255,
            config.line_width,
        )

    return centerline_mask


def refine_mask(mask, image_shape, config):
    height, width = image_shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(mask, config.mask_threshold, 255, cv2.THRESH_BINARY)
    mask = keep_vertical_components(mask, config)
    if not np.any(mask):
        return mask

    mask = rebuild_as_centerlines(mask, config)
    if not np.any(mask):
        return mask

    if config.close_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_size, config.close_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    if config.extra_dilate > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.extra_dilate, config.extra_dilate)
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return keep_vertical_components(mask, config)

