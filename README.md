# Race Track Tracking

This repository is organized around two cleaned processing stages for the
simulator/headcam workflow.

## Folder Layout


```text
race_track tracking/
  README.md
  marker_detection/
  bezel_removal/
```

## 1. Marker Detection

Folder: `marker_detection/`

Purpose:

- Detect the fixed AprilTags/markers in screen recordings.
- Detect the same marker layout in headcam frames or videos.
- Assign detected markers to the `left`, `center`, or `right` screen.
- Save detection outputs for later alignment and tracking work.

Main command:

```bash
 marker_detection/detect_markers.py \
  --mode headcam \
  --input /path/to/headcam_frames_or_video \
  --output-dir marker_detection/runs/headcam_markers \
  --temporal \
  --save-annotated
```

Use `--mode screen` for direct screen recordings and `--mode headcam` for camera
video of the physical screens.

Main outputs:

- `detections.csv`
- `detections.jsonl`
- `summary.json`
- annotated frames or annotated video when requested

## 2. Bezel Removal

Folder: `bezel_removal/`

Purpose:

- Remove the physical monitor bezels from headcam frames or video.
- Use the trained bezel-mask model or existing predicted masks.
- Refine masks to at most two vertical screen separators.
- Inpaint the removed regions so downstream road detection sees a cleaner image.

Main command:

```bash
 bezel_removal/remove_bezel.py \
  --input /path/to/headcam_frames_or_video \
  --output-dir bezel_removal/runs/headcam_bezel_removed \
  --mask-source model \
  --checkpoint /path/to/bezel_unet_best.pt \
  --backend auto \
  --save-masks \
  --save-overlays
```

Use `--backend auto` or `--backend lama` for final results. Use
`--backend opencv` for quick debugging.

Main outputs:

- cleaned frames or `bezel_removed.mp4`
- final masks when `--save-masks` is used
- mask overlays when `--save-overlays` is used
- `summary.json`

## Pipeline Order

1. Run marker detection to verify the screen/tag geometry.
2. Run bezel removal on the headcam frames/video.
3. Use the cleaned headcam output for the next downstream task: road detection
   and curve/tangent-point estimation.

## Notes

