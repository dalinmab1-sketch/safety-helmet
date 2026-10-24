# System architecture

## Runtime flow

`main.py` builds `DetectionConfig` and starts the Flask dashboard. The dashboard
owns a camera-processing worker thread and serves its latest JPEG frame through
the local `/video_feed` endpoint. The web page polls `/api/status` for live
counts, errors, model readiness, and recording state.

For each frame, `CameraSource` reads from a configured webcam, video path, or
stream URL. `DetectionEngine` runs the person detector and helmet-status model.
The status detections are associated with people, then `PersonStatusTracker`
maintains track IDs and short-term classification evidence. The dashboard
updates statistics and violation alerts, calls `render_frame`, and publishes
the annotated frame. Camera loss is retried; end-of-file and invalid video
sources are reported separately.

`VideoRecorder` writes video only after the dashboard recording control is
enabled. `EventLogger` appends confirmed alert events to the configured CSV
file. No face recognition or worker identity is performed.

## Model files

- `models/models.helmet_model.pt`: Ultralytics YOLO detection model. Its loaded
  class names are `With Helmet` and `Without Helmet`.
- `yolov8n.pt`: COCO person detector. The person class ID is resolved from the
  loaded names rather than assumed by the status classifier.

The class names and IDs are read at load time. A model without any recognized
helmet-status class is rejected instead of producing success-shaped,
unverified results.

## Current boundaries

- Webcam, local video-file, and stream URL sources are configured through
  `DetectionConfig.input_source`; there is no in-dashboard source picker or
  uploaded-image workflow.
- The dashboard recording control starts and stops recordings. Camera
  pause/resume and event-triggered snapshots are not currently implemented.
- A local browser dashboard does not imply the model is accurate enough for
  workplace enforcement. Use human review and site-specific validation.
