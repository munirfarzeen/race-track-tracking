# Bezel Removal

Clean task-2 code for removing the physical monitor bezels from headcam
frames/videos before downstream road detection.

This folder is inference-focused. It uses the latest trained mask model from
`lol/bezel_mask_option_a/checkpoints/bezel_unet_best.pt`, then refines the mask
to a narrow bezel centerline before inpainting. The old dataset/training
experiments are left in `lol/bezel_mask_option_a/` for reference.

## Layout

- `remove_bezel.py` - command-line entry point.
- `bezel_removal/mask_model.py` - small U-Net architecture and mask prediction.
- `bezel_removal/mask_refinement.py` - geometry filtering, narrowing, cleanup.
- `bezel_removal/inpainting.py` - LaMa/OpenCV inpainting backends.
- `bezel_removal/pipeline.py` - frame-folder and video processing.
- `requirements.txt` - Python dependencies.

## Recommended Commands

Run the trained mask model and remove bezels from the sampled headcam frames:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python bezel_removal/remove_bezel.py \
  --input /home/farzeen/work/otto_col/2026_04_14_Zandvoo/frames_test_sample/headcam_frames \
  --output-dir bezel_removal/runs/headcam_frames \
  --mask-source model \
  --checkpoint /home/farzeen/work/otto_col/lol/bezel_mask_option_a/checkpoints/bezel_unet_best.pt \
  --backend auto \
  --save-masks \
  --save-overlays
```

Use already predicted masks instead of running the model:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python bezel_removal/remove_bezel.py \
  --input /home/farzeen/work/otto_col/2026_04_14_Zandvoo/frames_test_sample/headcam_frames \
  --output-dir bezel_removal/runs/headcam_frames_from_masks \
  --mask-source folder \
  --mask-dir /home/farzeen/work/otto_col/lol/bezel_mask_option_a/predicted_masks \
  --backend auto \
  --save-masks \
  --save-overlays
```

Process only a few frames for QA:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python bezel_removal/remove_bezel.py \
  --mask-source model \
  --backend opencv \
  --only frame_002578.jpg frame_002760.jpg frame_004910.jpg \
  --output-dir bezel_removal/runs/qa_frames \
  --save-masks \
  --save-overlays \
  --save-comparison
```

Process a headcam video:

```bash
/home/farzeen/miniconda3/envs/lol/bin/python bezel_removal/remove_bezel.py \
  --input /path/to/headcam_video.mp4 \
  --output-dir bezel_removal/runs/headcam_video \
  --mask-source model \
  --backend auto \
  --save-masks
```

## Outputs

For image-folder input:

- `frames/` - bezel-removed output frames.
- `masks/` - final refined masks when `--save-masks` is used.
- `overlays/` - red mask overlays when `--save-overlays` is used.
- `comparisons/` - original/result side-by-side QA images when requested.
- `summary.json` - run-level counts.

For video input:

- `bezel_removed.mp4` - output video.
- optional `masks/`, `overlays/`, `comparisons/`, and `summary.json`.

## Notes

`--backend auto` uses LaMa if `simple_lama_inpainting` is installed, otherwise it
falls back to OpenCV Telea inpainting. For final results, LaMa is preferred. For
quick tests, `--backend opencv` starts faster.

The mask is capped at two vertical components per frame because there are three
physical screens and therefore at most two screen separators. By default the
predicted blob is fitted to a narrower centerline before inpainting, which helps
preserve road and guardrail structure for later road detection.
