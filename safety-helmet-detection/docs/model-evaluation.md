# Model evaluation status

## Available evidence

The bundled weights load as Ultralytics YOLO detection models. The helmet model
exposes `With Helmet` and `Without Helmet`; `yolov8n.pt` exposes the COCO
`person` class. A smoke test ran the actual models on a frame from a bundled
recording and exercised person detection, full-frame helmet inference,
association, and tracking. That frame yielded eight person boxes, two
associated `NO HELMET` results, and six `UNKNOWN` results at the current
thresholds.

This smoke test verifies that the inference path executes. It is not a labeled
accuracy evaluation, and its classifications are not ground truth.

## Metrics not reported

`datasets/data.yaml` does not define a usable labeled test set. Therefore this
project cannot currently report precision, recall, mAP, false-positive rate,
false-negative rate, helmet-status accuracy, or person-to-status association
error. No confidence threshold has been calibrated against held-out labels.
The sample recordings are not treated as ground truth.

## Evaluation needed before operational use

Prepare a representative, independently labeled test set covering helmeted
and unhelmeted workers, distances, lighting, occlusion, partial visibility,
and multiple people. Keep identities out of the dataset. Evaluate person
detection and helmet status separately; report confidence intervals and
latency alongside precision, recall, and mAP. Tune thresholds only on a
validation split, then report final results on held-out footage.

Until this work is completed, treat the output as decision support and review
uncertain detections manually. Do not use a single prediction as the basis for
punitive action.
