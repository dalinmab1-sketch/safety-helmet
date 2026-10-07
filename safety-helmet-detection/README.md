# Safety Helmet Detection

This project implements a browser-based YOLO safety helmet dashboard for webcam or video input. It detects people, identifies helmet/no-helmet conditions when trained weights are available, handles alerts, and tracks live statistics.

## Features

- Real-time webcam or video processing
- Local browser dashboard with live camera stream and safety-system logo
- YOLO-based person and PPE detection
- Helmet/no-helmet/unknown classification
- Stable no-helmet alerting with cooldown logic
- Hand-gesture recording controls (thumbs up to start, open palm to stop)
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
   Its classes are `With Helmet` and `Without Helmet`. Replace the weights only
   with another compatible model if you need to change the classifier.
4. Hand gesture recognition uses the included MediaPipe task file,
   `models/gesture_recognizer.task` (MediaPipe model:
   https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer).
5. Run the application:

   ```bash
   python main.py
   ```

   The app starts a local dashboard at `http://127.0.0.1:5000` and opens Google
   Chrome when available, otherwise the default browser. Keep the terminal
   running while using the dashboard.

## Controls

- Use **Stop recording** / **Start recording** in the dashboard to control MP4 recording.
- Hold a **thumbs up** in view for one second to start recording; hold an **open palm** for one second to stop. Turn gesture controls on or off in the dashboard.
- Recordings are saved in the `recordings` folder and listed in the dashboard for download.
- Press `Ctrl+C` in the terminal to stop the dashboard, release the camera, and finish saving.

## Camera troubleshooting

The app tries Windows DirectShow and Media Foundation capture backends, then
OpenCV's automatic backend. The default camera is device index `0`; to use
another webcam, change `input_source` in `src/config.py` (for example, to `1`).
If the camera cannot open, allow camera access for desktop apps in Windows
Privacy & security settings and close other programs currently using it.

## Notes

This is a demonstration and decision-support system. It should be validated in the target workplace environment before operational use.
