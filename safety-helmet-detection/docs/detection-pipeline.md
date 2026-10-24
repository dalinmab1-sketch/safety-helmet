# Detection pipeline

## Per-frame processing

1. `CameraSource` reads the configured source. Webcam frames are mirrored for
   a familiar preview; video files and network streams are kept in their
   original orientation.
2. `DetectionEngine.detect` runs the person model and filters person boxes by
   the configured confidence and minimum frame-area ratio.
3. `DetectionEngine.detect_status_detections` runs the helmet model on the
   complete frame. The model's loaded class names determine `HELMET` or
   `NO HELMET`; class IDs are not hard-coded.
4. `associate_people_with_helmets` assigns each status box to at most one
   person based on overlap and containment. Unmatched or conflicting evidence
   remains `UNKNOWN`.
5. `PersonStatusTracker` associates detections across frames and confirms a
   status using its short rolling history. Unknown observations clear
   classification evidence instead of preserving a stale color.
6. The dashboard computes live counts, updates violation alerts and CSV logs,
   then renders each person with a status-colored box.

The full-frame status inference is important for these bundled weights. On
sampled project clips the model produced worker-sized status boxes, while the
previous head-only crop path produced no status predictions. A missing status
box is not proof of missing protective equipment.

## Current defaults

Defaults are in `src/config.py`: person confidence `0.12`, minimum person area
ratio `0.005`, helmet confidence `0.90`, no-helmet confidence `0.35`, three
status-history frames with two votes required, and three alert frames with a
five-second cooldown. They are starting settings, not validated workplace
thresholds. In particular, the conservative helmet threshold can leave
otherwise plausible results as `UNKNOWN`.

## Banner and box status priority

- Any confirmed `NO HELMET` result: red violation banner.
- Otherwise, any unknown or missing per-person result: yellow unverified
  banner.
- All detected people confirmed `HELMET`: green banner.
- No detected people: neutral banner.
- A missing model or unavailable person detector: red error banner.

A person box is green or red only after status evidence passes configured
confidence thresholds and temporal confirmation. The system does not infer
`NO HELMET` from an absent helmet detection.
