"""
object_detector.py
-------------------
Thin wrapper around a pretrained YOLOv8n (COCO-80) model for general road
scene object detection: vehicles, pedestrians, animals, and the handful of
COCO classes relevant to signage ('stop sign', 'traffic light').

COCO does not have a fine-grained IRC sign taxonomy (mandatory/cautionary/
informatory per IRC:67 Section 3 classification), so a dedicated sign shape/
colour classifier (`sign_shape_detector.py`) is layered on top for that.

This module is a clean seam: swap `YOLO(YOLO_WEIGHTS)` for a custom
fine-tuned checkpoint (e.g. trained on an Indian traffic-sign dataset) without
touching any downstream code.
"""

from ultralytics import YOLO
from config import YOLO_WEIGHTS, YOLO_CONF_THRESH

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(YOLO_WEIGHTS)
    return _model


# COCO classes we care about for road-safety / IRC checks
RELEVANT_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "dog", "cow", "horse", "sheep",
}

# animals that indicate uncontrolled livestock / stray-animal hazard on carriageway
ANIMAL_CLASSES = {"dog", "cow", "horse", "sheep", "bird", "cat"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck", "train"}


def detect_objects(frame_bgr, conf=YOLO_CONF_THRESH):
    model = get_model()
    result = model.predict(frame_bgr, conf=conf, verbose=False)[0]
    names = result.names
    detections = []
    for b in result.boxes:
        cls_name = names[int(b.cls[0])]
        confidence = float(b.conf[0])
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        detections.append({
            "class": cls_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "is_animal": cls_name in ANIMAL_CLASSES,
            "is_vehicle": cls_name in VEHICLE_CLASSES,
        })
    return detections


if __name__ == "__main__":
    import sys
    import cv2
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else
                      "F:\CVIT\IRASTE\knowledgeGraph\irc_compliance_pipeline\frames\f000015.jpg")
    for d in detect_objects(img):
        print(d)