"""
sign_shape_detector.py
-----------------------
Classical CV shape+colour classifier for traffic signs in the UPPER portion of
the frame, mapped onto the IRC:67 Section-3 sign classification:

    circle          -> Mandatory / Regulatory sign   (IRC:67 Sec 14)
    triangle (up)   -> Cautionary / Warning sign      (IRC:67 Sec 15)
    rectangle/square-> Informatory sign               (IRC:67 Sec 16/17)

Also flags a coarse "condition" (ok / faded / low_contrast) using saturation
and local contrast as a proxy for retro-reflectivity/maintenance issues
(relevant to IRC:67 Sec 8 colour & Sec 13 maintenance).

This is a triage-level heuristic, not a certified sign-recognition model.
A YOLO model fine-tuned on a labelled Indian traffic-sign dataset is the
natural production replacement — see NOTE at bottom of file.
"""

import cv2
import numpy as np


def _upper_roi(frame):
    h, w = frame.shape[:2]
    y1 = int(h * 0.55)
    return frame[0:y1, :], 0


def _red_blue_yellow_masks(hsv):
    red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    blue = cv2.inRange(hsv, (100, 80, 60), (130, 255, 255))
    yellow = cv2.inRange(hsv, (15, 60, 120), (35, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    return {"red": red, "blue": blue, "yellow": yellow, "white": white}


def _classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    v = len(approx)
    area = cv2.contourArea(contour)
    if area <= 0:
        return None
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    circularity = area / circle_area if circle_area > 0 else 0

    if circularity > 0.75:
        return "circle"
    if v == 3:
        return "triangle"
    if v == 4:
        return "rectangle"
    return None


SHAPE_TO_IRC67_CATEGORY = {
    "circle": "Mandatory/Regulatory sign (IRC:67 Sec 14)",
    "triangle": "Cautionary/Warning sign (IRC:67 Sec 15)",
    "rectangle": "Informatory sign (IRC:67 Sec 16/17)",
}


def detect_signs(frame_bgr, min_area=250):
    roi, y_off = _upper_roi(frame_bgr)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = _red_blue_yellow_masks(hsv)

    signs = []
    for colour_name, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            shape = _classify_shape(c)
            if shape is None:
                continue
            x, y, w, h = cv2.boundingRect(c)
            patch = roi[y:y + h, x:x + w]
            sat = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1].mean() if patch.size else 0
            contrast = patch.std() if patch.size else 0

            condition = "ok"
            if sat < 40 or contrast < 20:
                condition = "faded_or_low_contrast"

            signs.append({
                "shape": shape,
                "colour": colour_name,
                "irc67_category": SHAPE_TO_IRC67_CATEGORY.get(shape, "Unclassified"),
                "condition": condition,
                "bbox": [int(x), int(y + y_off), int(x + w), int(y + h + y_off)],
                "area": int(area),
            })
    return signs


# NOTE (upgrade path):
#   from ultralytics import YOLO
#   sign_model = YOLO("path/to/finetuned_indian_traffic_signs.pt")
#   -> replace this module's body with sign_model.predict(frame_bgr) and map
#      the fine-tuned class names directly to IRC:67 Annexure I/II sign codes.


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else
                      "F:\CVIT\IRASTE\knowledgeGraph\irc_compliance_pipeline\frames\f000015.jpg")
    for s in detect_signs(img):
        print(s)