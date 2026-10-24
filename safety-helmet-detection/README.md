# Safety Helmet Detection

This project implements a browser-based YOLO safety helmet dashboard for webcam or video input. It detects people, identifies helmet/no-helmet conditions when trained weights are available, handles alerts, and tracks live statistics.

## Features

- Real-time webcam or video processing
- Local browser dashboard with live camera stream and safety-system logo
- YOLO-based person and PPE detection
- Helmet/no-helmet/unknown classification
- Stable no-helmet alerting with cooldown logic
- JavaScript hand-gesture controls (one-finger swipes to scroll; thumbs up to start and open palm to stop recording)
- Live statistics overlay
- CSV logging for violation events
- Start/stop real-time video recording and a saved-recordings download list
- Modular project structure for further extension

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Connect a webcam and allow desktop camera access in Windows settings.
3. Helmet weights are installed at `models/models.helmet_model.pt` and use the
   MIT-licensed YOLOv8n model from
   [iam-tsr/yolov8n-helmet-detection](https://huggingface.co/iam-tsr/yolov8n-helmet-detection).
   Its classes are `With Helmet` and `Without Helmet`. Person detection uses the
   bundled COCO weights at `yolov8n.pt`, independently of the helmet model.
   Replace the weights only with another compatible model if you need to change
   either detector.
   The helmet model is run on the complete camera frame because its output
   boxes cover workers; running it only on an assumed head crop can discard
   valid classifications. Its supported status classes are read from the
   loaded weights (`With Helmet` and `Without Helmet`) rather than inferred
   from fixed class IDs. Full-frame helmet detections are associated with the
   independently detected person boxes. The dashboard reports an unverified
   status rather than claiming a helmet when model confidence is weak.
   Helmet detections cannot create person detections;
   if the person detector is unavailable, status is reported as unverified.
   Conflicting classifications remain unknown, and statuses are confirmed
   across consecutive frames before they affect live counts or alerts. The
   violation rate excludes unknown classifications and displays N/A when no
   statuses are confirmed.
   For reliable site-specific classification, retrain and validate the helmet
   model with a representative, accurately labeled dataset of helmeted and
   unhelmeted workers. The included weights can still confuse hair with a
   helmet; this conservative pipeline is not a substitute for measuring model
   accuracy against such a dataset.
4. Browser hand tracking uses the included MediaPipe task file,
   `models/gesture_recognizer.task`. The MediaPipe Tasks Vision JavaScript and
   WebAssembly runtime are loaded from jsDelivr, so gesture controls need an
   internet connection; the model itself is served by the local dashboard.
5. Run the application:

   ```bash
   python main.py
   ```

   The app starts a local dashboard at `http://127.0.0.1:5000` and opens Google
   Chrome when available, otherwise the default browser. Keep the terminal
   running while using the dashboard.

## Controls

- Use **Stop recording** / **Start recording** in the dashboard to control MP4 recording.
- Keep your hand clearly visible in the camera feed. Point with your index finger to move the on-screen cursor and pinch your thumb and index finger to click a control. Swipe up or down with one finger to smoothly scroll the page; hold a **thumbs up** for about a second to start recording or an **open palm** to stop. The gesture status shows when a hand is detected. Use the dashboard toggle to turn gesture controls on or off.
- Camera capture requests up to 60 FPS and gesture tracking checks for hand movement up to 30 times per second. The actual frame rate depends on the camera, model-processing speed, and computer; the cursor is eased between tracking updates.
- Recordings are saved in the `recordings` folder and listed in the dashboard for download.
- Press `Ctrl+C` in the terminal to stop the dashboard, release the camera, and finish saving.

## Camera troubleshooting

The app tries Windows DirectShow and Media Foundation capture backends, then
OpenCV's automatic backend. It tries the configured camera index first, then
checks indexes `0` through `4`, and retries automatically if the camera is
temporarily unavailable. The default camera is device index `0`; to use
another webcam, change `input_source` in `src/config.py` (for example, to `1`).
To process a recorded video, set `input_source` to its local path, for example
`r"C:\Videos\site.mp4"`. RTSP URLs are treated as streams. Video files are not
mirrored, and reaching the end of a file is shown as `VIDEO COMPLETE` rather
than being reported as a disconnected camera.
If the camera cannot open, allow camera access for desktop apps in Windows
Privacy & security settings and close other programs currently using it.
The default capture target is 60 FPS (`camera_fps` in `src/config.py`); the camera
may report and deliver a lower rate if its driver or current resolution does not
support that speed.

Set `diagnostic_mode=True` in `src/config.py` to log model paths and classes,
person detections, full-frame helmet predictions, final track
statuses, and inference time/FPS.

The project includes recorded clips, but they are not labeled evaluation data.
Detection on those clips can verify that inference runs and returns boxes; it
cannot establish precision, recall, mAP, or a safe operating threshold. Those
metrics require a separate, representative labeled test set.

## Notes

This is a demonstration and decision-support system. It should be validated in the target workplace environment before operational use.

For implementation details and evaluation limits, see
[the architecture overview](docs/architecture.md),
[the detection pipeline](docs/detection-pipeline.md), and
[the model evaluation note](docs/model-evaluation.md).
