# Race Track Tracking

This repository is organized around cleaned processing stages for the
simulator/headcam workflow: marker detection, bezel removal, and track
apex/turn detection.

## Folder Layout

```text
race_track tracking/
  README.md
  marker_detection/
  bezel_removal/
  apex_detection_segmentation/
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

## 3. Apex Detection

Folder: `apex_detection_segmentation/`

Purpose:

- Segment the track (asphalt) surface in headcam frames with SAM2's video
  predictor, seeded from a handful of positive/negative click points on one
  annotation frame.
- Fit the left/right track boundaries per frame from the SAM2 mask.
- Estimate the turn direction (`left`/`right`/`straight`) and the apex
  candidate (point of maximum curvature on the inside edge of the turn).

Main command:

```bash
apex_detection_segmentation/detect_apex.py \
  --input /path/to/frames \
  --output-dir apex_detection_segmentation/runs/zandvoort \
  --checkpoint /path/to/checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml
```

The bundled `default_points.json` is calibrated for the Zandvoort headcam
framing only. For any other camera setup, click new seed points with
`select_points.py` and pass the result via `--points-file` — see
`apex_detection_segmentation/README.md` for the full workflow.

Main outputs:

- `apex_results.jsonl` - per-frame turn direction and apex coordinates
- `summary.json`
- `masks/` and `overlays/` when `--save-masks`/`--save-overlays` are used
  (on by default)

## Pipeline Order

1. Run marker detection to verify the screen/tag geometry.
2. Run bezel removal on the headcam frames/video.
3. Run apex detection on the cleaned headcam output to get per-frame turn
   direction and apex candidates.

## Notes

