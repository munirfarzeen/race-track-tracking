PANORAMA_WIDTH = 7680
PANORAMA_HEIGHT = 1440

SCREEN_RANGES = {
    "left": (0, 2559),
    "center": (2560, 5119),
    "right": (5120, 7679),
}

TAG_BBOXES = {
    3: [486, 14, 103, 103],
    7: [1430, 14, 103, 103],
    11: [2374, 14, 103, 103],
    13: [2846, 14, 103, 103],
    17: [3790, 14, 103, 103],
    21: [4734, 14, 103, 103],
    23: [5206, 14, 103, 103],
    27: [6150, 14, 103, 103],
    31: [7094, 14, 103, 103],
    47: [3082, 1162, 103, 103],
    53: [4498, 1162, 103, 103],
    69: [486, 1326, 103, 103],
    73: [1430, 1326, 103, 103],
    77: [2374, 1326, 103, 103],
    79: [2846, 1326, 103, 103],
    87: [4734, 1326, 103, 103],
    89: [5206, 1326, 103, 103],
    93: [6150, 1326, 103, 103],
    97: [7094, 1326, 103, 103],
}

VALID_TAG_IDS = set(TAG_BBOXES)


def bbox_to_corners(bbox):
    x, y, w, h = bbox
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
    ]


def screen_from_panorama_x(x):
    for screen, (x0, x1) in SCREEN_RANGES.items():
        if x0 <= x <= x1:
            return screen
    return None


def build_tag_config():
    config = {}
    for tag_id, bbox in TAG_BBOXES.items():
        x, y, w, h = bbox
        center_x = x + w / 2.0
        screen = screen_from_panorama_x(center_x)
        config[int(tag_id)] = {
            "tag_id": int(tag_id),
            "screen": screen,
            "bbox_panorama_px": [float(v) for v in bbox],
            "corners_panorama_px": bbox_to_corners(bbox),
        }
    return config


TAG_CONFIG = build_tag_config()


def screen_for_tag(tag_id):
    item = TAG_CONFIG.get(int(tag_id))
    if item is None:
        return None
    return item["screen"]
