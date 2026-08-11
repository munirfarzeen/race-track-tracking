# Marker Detection

Marker-detection code for the simulator screen setup.

 detect and assign the fixed 19 AprilTags in either screen
recordings or headcam video/frames. 

## Layout

- `detect_markers.py` - command-line entry point.
- `marker_detection/detector.py` - AprilTag detection, multipass headcam mode,
  and optional temporal recovery.
- `marker_detection/tag_layout.py` - fixed 19-tag layout and screen assignment.
- `marker_detection/io_utils.py` - input/output helpers for image folders/videos.
- `requirements.txt` - Python dependencies.

## Inputs

The CLI accepts either:

- an image folder containing frames such as `frame_002539.jpg`
- a video file such as `.mp4`, `.avi`, `.mov`, `.mkv`

Use `--mode screen` for direct screen recordings. Use `--mode headcam` for
camera footage of the three physical screens.

## Commands

Screen recording frames:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python marker_detection/detect_markers.py \
  --mode screen \
  --input /home/farzeen/work/otto_col/2026_04_14_Zandvoo/frames_test_sample/screen_frames \
  --output-dir marker_detection/runs/screen_frames \
  --save-annotated
```

Headcam frames:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python marker_detection/detect_markers.py \
  --mode headcam \
  --input /home/farzeen/work/otto_col/2026_04_14_Zandvoo/frames_test_sample/headcam_frames \
  --output-dir marker_detection/runs/headcam_frames \
  --save-annotated
```

Headcam video with temporal recovery:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python marker_detection/detect_markers.py \
  --mode headcam \
  --input /path/to/headcam_video.mp4 \
  --output-dir marker_detection/runs/headcam_video \
  --temporal \
  --save-annotated
```

## Outputs

Each run writes:

- `detections.csv` - one row per detected/recovered tag.
- `detections.jsonl` - one JSON object per processed frame.
- `summary.json` - run-level counts and settings.
- `annotated_frames/` for image-folder input when `--save-annotated` is used.
- `annotated.mp4` for video input when `--save-annotated` is used.

Known tag IDs are assigned to `left`, `center`, or `right` screens using the
fixed layout from the latest `lol/Step1_3.py` work.
