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


def iter_image_folder(input_dir, frame_step=1, max_frames=None):
    paths = list_images(input_dir)
    yielded = 0
    for frame_index, path in enumerate(paths):
        if frame_index % frame_step != 0:
            continue
        if max_frames is not None and yielded >= max_frames:
            break

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        yield {
            "frame": frame,
            "frame_index": frame_index,
            "frame_name": path.name,
            "timestamp_sec": None,
        }
        yielded += 1


def iter_video(video_path, frame_step=1, max_frames=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    yielded = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            timestamp = frame_index / fps if fps > 0 else None
            yield {
                "frame": frame,
                "frame_index": frame_index,
                "frame_name": f"frame_{frame_index:06d}",
                "timestamp_sec": timestamp,
            }
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break

        frame_index += 1

    capture.release()


def make_video_writer(output_path, fps, frame_shape):
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
