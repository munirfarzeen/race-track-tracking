from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def list_images(folder):
    folder = Path(folder)
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def iter_image_folder(folder, frame_step=1, max_frames=None, only=None):
    images = list_images(folder)
    if only:
        requested = set(only)
        images = [
            path for path in images if path.name in requested or path.stem in requested
        ]

    written = 0
    for source_index, path in enumerate(images):
        if source_index % frame_step != 0:
            continue
        if max_frames is not None and written >= max_frames:
            break

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        yield {
            "frame": frame,
            "frame_index": source_index,
            "frame_name": path.name,
            "timestamp_sec": None,
        }
        written += 1


def iter_video(path, frame_step=1, max_frames=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    source_index = 0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_index % frame_step != 0:
                source_index += 1
                continue
            if max_frames is not None and written >= max_frames:
                break

            timestamp_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield {
                "frame": frame,
                "frame_index": source_index,
                "frame_name": f"frame_{source_index:06d}.jpg",
                "timestamp_sec": timestamp_sec,
            }
            written += 1
            source_index += 1
    finally:
        capture.release()


def video_fps(path, fallback=30.0):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return fallback
    fps = capture.get(cv2.CAP_PROP_FPS) or fallback
    capture.release()
    return fps


def make_video_writer(path, fps, frame_shape):
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer

