"""
marking_detector.py
--------------------
Classical computer-vision detector for road MARKINGS (relevant to IRC:35).
No internet / pretrained weights needed — pure OpenCV so it always runs.

Detects, per frame:
    - presence / absence of longitudinal lane/centre markings in the lower ROI
    - marking colour (white / yellow / none)
    - continuity (solid vs broken vs absent)
    - rough estimate of degradation (faded marking -> low pixel coverage)

These are heuristics suitable for a first-pass triage pipeline. They are
intentionally conservative and documented so a trained segmentation model
(e.g. a fine-tuned lane/marking segmentation network) can be dropped in later
as a drop-in replacement for `detect_markings()`.
"""

import cv2
import numpy as np


def _lower_roi(frame):
    h, w = frame.shape[:2]
    y0 = int(h * 0.55)
    return frame[y0:h, 0:w], y0


def _colour_masks(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # white marking: low saturation, high value
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # yellow marking
    yellow_lower = np.array([15, 60, 120])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    return white_mask, yellow_mask


def _keep_elongated_components(mask, min_area=15, min_elong=2.2):
    """Keep only connected components that look like paint strokes (elongated,
    thin) and drop blobby regions (buildings, hoardings, vehicles) that happen
    to share the target hue."""
    out = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        long_side = max(w, h)
        short_side = max(min(w, h), 1)
        elong = long_side / short_side
        if elong >= min_elong and area < 0.02 * mask.shape[0] * mask.shape[1]:
            out[labels == lbl] = 255
    return out


def detect_markings(frame_bgr):
    """
    Returns a dict describing marking state in this frame:
    {
        'marking_present': bool,
        'colour': 'white'|'yellow'|'none',
        'continuity': 'solid'|'broken'|'absent',
        'coverage_ratio': float,   # fraction of ROI road-surface pixels that are marking
        'num_segments': int,       # number of distinct line segments found (Hough)
        'bbox_roi': (x0,y0,x1,y1)  # ROI in full-frame coords, for drawing
    }
    """
    roi, y_off = _lower_roi(frame_bgr)
    h, w = roi.shape[:2]

    white_mask, yellow_mask = _colour_masks(roi)

    # restrict search to a road-surface band (exclude far left/right pavement/shops)
    road_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(w*0.12), h], [int(w*0.42), int(h*0.05)],
                     [int(w*0.58), int(h*0.05)], [int(w*0.92), h]], dtype=np.int32)
    cv2.fillPoly(road_mask, [pts], 255)

    white_mask = cv2.bitwise_and(white_mask, road_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, road_mask)

    white_mask = _keep_elongated_components(white_mask)
    yellow_mask = _keep_elongated_components(yellow_mask)

    white_px = int(cv2.countNonZero(white_mask))
    yellow_px = int(cv2.countNonZero(yellow_mask))
    road_px = max(int(cv2.countNonZero(road_mask)), 1)

    colour = "none"
    mask_used = None
    if yellow_px > white_px and yellow_px > 0:
        colour = "yellow"
        mask_used = yellow_mask
    elif white_px > 0:
        colour = "white"
        mask_used = white_mask

    coverage_ratio = (max(white_px, yellow_px) / road_px)

    num_segments = 0
    continuity = "absent"
    if mask_used is not None and coverage_ratio > 0.0005:
        edges = cv2.Canny(mask_used, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                                 minLineLength=int(h * 0.05), maxLineGap=int(h * 0.03))
        if lines is not None:
            num_segments = len(lines)

        marking_present = coverage_ratio > 0.0015
        if not marking_present:
            continuity = "absent"
        elif num_segments >= 6:
            continuity = "broken"
        elif num_segments >= 1:
            continuity = "solid"
        else:
            continuity = "absent"
    else:
        marking_present = False

    marking_present = coverage_ratio > 0.0015

    return {
        "marking_present": bool(marking_present),
        "colour": colour if marking_present else "none",
        "continuity": continuity,
        "coverage_ratio": round(float(coverage_ratio), 5),
        "num_segments": int(num_segments),
        "bbox_roi": (0, y_off, frame_bgr.shape[1], frame_bgr.shape[0]),
    }


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "F:\CVIT\IRASTE\knowledgeGraph\irc_compliance_pipeline\frames\f000015.jpg")
    print(detect_markings(img))