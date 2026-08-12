# Apex Detection

Segments the track (asphalt) surface in headcam frames with SAM2's video
predictor, fits the left/right track boundaries per frame, and estimates the
turn direction and apex candidate (the point of maximum curvature on the
inside edge of the turn).

SAM2 is prompted once, with a handful of positive/negative point clicks on a
single annotation frame, then the resulting mask is propagated across the
rest of the frame sequence.

## Layout

- `detect_apex.py` - command-line entry point.
- `select_points.py` - interactive tool to click seed points on a frame and save them as a points-file JSON.
- `apex_detection/segmentation.py` - SAM2 predictor setup and video mask propagation.
- `apex_detection/boundaries.py` - row-wise left/right boundary extraction and polynomial fitting.
- `apex_detection/apex.py` - turn direction and apex-candidate estimation from boundary curvature.
- `apex_detection/visualization.py` - mask/boundary/apex overlay drawing.
- `apex_detection/pipeline.py` - per-run orchestration (SAM2 propagation, per-frame outputs, summary).
- `apex_detection/io_utils.py` - frame listing and seed-point loading.
- `default_points.json` - bundled seed clicks, calibrated for the Zandvoort headcam framing.
- `requirements.txt` - Python dependencies.

## Seed points

SAM2 needs a few clicks on the annotation frame (frame 0 by default) to know
what "track" means:

- **positive points** land on asphalt/track.
- **negative points** land on grass, curb, barrier, HUD, or sky.

`default_points.json` has the points calibrated for the Zandvoort headcam
framing. **These are specific to one camera position/crop** — for any other
video, pick new pixel coordinates on that video's annotation frame and pass
your own file with `--points-file`:

```json
{
  "positive_points": [[x, y], ...],
  "negative_points": [[x, y], ...]
}
```

### Generating your own seed points

`select_points.py` opens the annotation frame in an OpenCV window so you can
click the points by hand and save them straight to a points-file JSON — no
manual copy-pasting or format conversion needed:

```bash
python select_points.py \
  --image /path/to/frames/frame_00000.png \
  --output my_points.json
```

Controls: `p` = positive/asphalt mode (default), `n` = negative/non-track
mode, `u` = undo the last point in the current mode, `q` or `Esc` = save and
quit. At least one positive point is required. Use the frame at whatever
index you pass to `detect_apex.py --annotation-frame` (frame 0 by default)
so the points line up with what SAM2 is prompted on. Requires a display
(local machine or X-forwarded session) since it opens a GUI window.

## Commands

Run with the bundled Zandvoort points against a frame folder:

```bash
python apex_detection_segmentation/detect_apex.py \
  --input /path/to/frames \
  --output-dir apex_detection_segmentation/runs/zandvoort \
  --checkpoint /path/to/checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml
```

Run against a new video/camera setup with its own seed points:

```bash
python apex_detection_segmentation/detect_apex.py \
  --input /path/to/frames \
  --output-dir apex_detection_segmentation/runs/new_track \
  --points-file /path/to/my_points.json \
  --checkpoint /path/to/checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml
```

Skip writing per-frame masks/overlays and only keep the numeric results:

```bash
python apex_detection_segmentation/detect_apex.py \
  --input /path/to/frames \
  --output-dir apex_detection_segmentation/runs/fast \
  --no-save-masks \
  --no-save-overlays
```

`--checkpoint` and `--model-cfg` must point at a SAM2 checkpoint/config pair
from the [SAM2 repo](https://github.com/facebookresearch/sam2) (or your own
fine-tuned checkpoint using the matching config).

## Outputs

- `apex_results.jsonl` - one JSON object per successfully segmented frame: `frame_index`, `frame_name`, `turn` (`left`/`right`/`straight`/`unknown`), `apex_side`, `apex_x`, `apex_y`.
- `summary.json` - run-level counts (frames processed, turn-direction breakdown, config used).
- `masks/` - binary track masks per frame, when `--save-masks` is used (on by default).
- `overlays/` - mask + boundary + apex visualization per frame, when `--save-overlays` is used (on by default).
- `sam2_frames/` - intermediate zero-padded JPEG copy of the input frames that SAM2's video predictor requires; deleted after the run unless `--keep-sam2-frame-cache` is passed.

## Notes

- Turn direction is read from the curvature sign of a parabola fit to the
  track centerline; if the fit is nearly flat, the frame is classified as
  `straight`.
- The apex candidate is the point of maximum curvature on the inside
  boundary of the turn (left edge for a left turn, right edge for a right
  turn) and is only reported when the frame isn't classified as `straight`
  or `unknown`.
- Boundaries are only fit where a mask row has at least 50 lit pixels, so
  frames with a very thin or broken track mask fall back to `turn=unknown`.
