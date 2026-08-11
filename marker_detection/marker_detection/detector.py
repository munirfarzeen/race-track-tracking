import cv2
import numpy as np
from pupil_apriltags import Detector

from .tag_layout import VALID_TAG_IDS, screen_for_tag


SCREEN_SETTINGS = [
    {"quad_decimate": 1.0, "quad_sigma": 0.0, "decode_sharpening": 0.25},
]

HEADCAM_SETTINGS = [
    {"quad_decimate": 1.0, "quad_sigma": 0.0, "decode_sharpening": 0.25},
    {"quad_decimate": 0.8, "quad_sigma": 0.0, "decode_sharpening": 0.5},
    {"quad_decimate": 1.0, "quad_sigma": 0.4, "decode_sharpening": 0.8},
    {"quad_decimate": 0.6, "quad_sigma": 0.0, "decode_sharpening": 1.0},
]

SCALES_BY_MODE = {
    "screen": [1.0],
    "headcam": [1.0, 1.5, 2.0, 2.5],
}


class AprilTagDetector:
    def __init__(self, mode="headcam", tag_family="tag36h11", valid_ids=None):
        self.mode = mode
        self.tag_family = tag_family
        self.valid_ids = set(VALID_TAG_IDS if valid_ids is None else valid_ids)
        settings = SCREEN_SETTINGS if mode == "screen" else HEADCAM_SETTINGS
        self.detectors = [
            Detector(
                families=tag_family,
                nthreads=4,
                refine_edges=True,
                debug=False,
                **setting,
            )
            for setting in settings
        ]

    def detect(self, frame):
        tags = []
        scales = SCALES_BY_MODE.get(self.mode, [1.0])

        for scale in scales:
            scaled = resize_for_scale(frame, scale)
            for variant_name, gray in preprocess_variants(scaled, self.mode):
                for detector_id, detector in enumerate(self.detectors):
                    detections = detector.detect(gray)
                    for det in detections:
                        tag = tag_from_detection(
                            det,
                            scale=scale,
                            variant=variant_name,
                            detector_id=detector_id,
                        )
                        if int(tag["id"]) not in self.valid_ids:
                            continue
                        add_or_update_duplicate(tags, tag)

        return sorted(tags, key=lambda item: int(item["id"]))


def resize_for_scale(frame, scale):
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess_variants(frame, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "screen":
        return [("gray", gray)]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray_clahe, 1.8, blur, -0.8, 0)
    bright = cv2.convertScaleAbs(gray, alpha=1.35, beta=20)
    bright_clahe = clahe.apply(bright)
    blur2 = cv2.GaussianBlur(bright_clahe, (0, 0), 0.8)
    sharp2 = cv2.addWeighted(bright_clahe, 1.6, blur2, -0.6, 0)

    return [
        ("gray", gray),
        ("clahe", gray_clahe),
        ("clahe_sharp", sharp),
        ("bright", bright),
        ("bright_clahe_sharp", sharp2),
        ("gamma_075", apply_gamma(gray, gamma=0.75)),
    ]


def apply_gamma(gray, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(gray, table)


def tag_from_detection(det, scale, variant, detector_id):
    center = np.array(det.center, dtype=np.float32) / scale
    corners = np.array(det.corners, dtype=np.float32) / scale
    tag_id = int(det.tag_id)
    return {
        "id": tag_id,
        "screen": screen_for_tag(tag_id),
        "center": center,
        "corners": corners,
        "decision_margin": float(det.decision_margin),
        "scale": float(scale),
        "variant": variant,
        "detector_id": int(detector_id),
        "source": "detected",
    }


def add_or_update_duplicate(tags, new_tag, duplicate_center_thresh=18):
    for old_tag in tags:
        if old_tag["id"] != new_tag["id"]:
            continue
        if np.linalg.norm(old_tag["center"] - new_tag["center"]) >= duplicate_center_thresh:
            continue
        if new_tag["decision_margin"] > old_tag["decision_margin"]:
            old_tag.update(new_tag)
        return
    tags.append(new_tag)


class TemporalTagTracker:
    def __init__(self, max_missing=5):
        self.max_missing = max_missing
        self.prev_gray = None
        self.tracks = {}

    def merge(self, frame, detected_tags):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recovered = []
        if self.prev_gray is not None:
            recovered = self.recover(curr_gray, detected_tags)

        final_tags = sorted(detected_tags + recovered, key=lambda item: int(item["id"]))
        self.update_tracks(final_tags)
        self.prev_gray = curr_gray.copy()
        return detected_tags, recovered, final_tags

    def recover(self, curr_gray, detected_tags):
        detected_ids = {int(tag["id"]) for tag in detected_tags}
        recovered = []

        for tag_id, track in list(self.tracks.items()):
            if tag_id in detected_ids or track["missing"] >= self.max_missing:
                continue

            prev_pts = track["corners"].astype(np.float32).reshape(-1, 1, 2)
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                curr_gray,
                prev_pts,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    40,
                    0.01,
                ),
            )
            if curr_pts is None or status is None:
                continue

            status = status.reshape(-1)
            if int(status.sum()) < 4:
                continue

            corners = curr_pts.reshape(-1, 2).astype(np.float32)
            if not valid_quad(corners, curr_gray.shape):
                continue

            recovered.append(
                {
                    "id": int(tag_id),
                    "screen": screen_for_tag(tag_id),
                    "center": corners.mean(axis=0),
                    "corners": corners,
                    "decision_margin": -1.0,
                    "scale": None,
                    "variant": "temporal",
                    "detector_id": None,
                    "source": "temporal",
                }
            )

        return recovered

    def update_tracks(self, tags):
        current_ids = {int(tag["id"]) for tag in tags}
        for tag in tags:
            tag_id = int(tag["id"])
            self.tracks[tag_id] = {
                "corners": np.array(tag["corners"], dtype=np.float32),
                "missing": 0,
            }

        for tag_id in list(self.tracks):
            if tag_id not in current_ids:
                self.tracks[tag_id]["missing"] += 1
                if self.tracks[tag_id]["missing"] > self.max_missing:
                    del self.tracks[tag_id]


def valid_quad(corners, image_shape):
    height, width = image_shape[:2]
    if corners.shape != (4, 2):
        return False
    if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
        return False
    if np.any(corners[:, 0] < -10) or np.any(corners[:, 0] > width + 10):
        return False
    if np.any(corners[:, 1] < -10) or np.any(corners[:, 1] > height + 10):
        return False
    if not cv2.isContourConvex(corners.astype(np.int32)):
        return False

    side_lengths = np.array(
        [
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[1] - corners[2]),
            np.linalg.norm(corners[2] - corners[3]),
            np.linalg.norm(corners[3] - corners[0]),
        ]
    )
    if np.any(side_lengths < 6):
        return False
    if np.max(side_lengths) / max(np.min(side_lengths), 1e-6) > 2.2:
        return False
    return cv2.contourArea(corners.astype(np.float32)) >= 30
