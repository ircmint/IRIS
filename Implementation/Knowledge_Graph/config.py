"""
config.py — central configuration for the IRC Compliance Pipeline.

Only the sections explicitly requested are extracted into the knowledge base
for each IRC code, and each code's rules are kept in a SEPARATE knowledge base
(as required: "each pdf should treat separately").
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Input documents -------------------------------------------------
IRC_DOCS = {
    "IRC35_2015": {
        "pdf_path": "F:\CVIT\IRASTE\knowledgeGraph\IRC35_2015.pdf",
        "title": "IRC:35-2015 Code of Practice for Road Markings",
        "domain": "road_markings",
        # top-level section number -> (start_page, end_page_exclusive), 0-indexed pdfplumber pages
        # derived by scanning the actual PDF's own section headers
        "section_pages": {
            2:  (13, 17),
            3:  (17, 20),
            4:  (20, 33),
            5:  (33, 41),
            6:  (41, 45),
            7:  (45, 54),
            8:  (54, 60),
            9:  (60, 77),
            10: (77, 80),
            11: (80, 88),
            12: (88, 92),
            13: (92, 99),
            14: (99, 102),
            15: (102, 106),
            16: (106, 120),
        },
        # sections the user asked us to actually load (image_1 instructions)
        "requested_sections": ["3", "4", "6.1", "6.2", "7", "8", "11"],
    },
    "IRC67_2022": {
        "pdf_path": "F:\CVIT\IRASTE\knowledgeGraph\IRC67_2022.pdf",
        "title": "IRC:67-2022 Code of Practice for Road Signs",
        "domain": "road_signs",
        "section_pages": {
            1:  (11, 12),
            2:  (12, 14),
            3:  (14, 15),
            4:  (15, 18),
            5:  (18, 19),
            6:  (19, 29),
            7:  (29, 30),
            8:  (30, 32),
            9:  (32, 32),
            10: (32, 32),
            11: (32, 33),
            12: (33, 35),
            13: (35, 36),
            14: (36, 48),
            15: (48, 58),
            16: (58, 64),
            17: (64, 69),
            18: (69, 71),
            19: (71, 71),
            20: (71, 72),
            21: (72, 74),
            22: (74, 77),
            23: (77, 79),
            24: (79, 81),
            25: (81, 82),
            26: (82, 82),
            27: (82, 145),
        },
        "requested_sections": ["3", "11", "13", "14", "15", "16", "17", "24", "25", "26"],
    },
}

KB_DIR = os.path.join(BASE_DIR, "kb")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIS_DIR = os.path.join(BASE_DIR, "vis")

YOLO_WEIGHTS = "yolov8n.pt"
YOLO_CONF_THRESH = 0.25

# Frame sampling rate when reading a video (1 = every Nth second per FPS group)
FRAME_SAMPLE_EVERY_N_FRAMES = 15  # adjust to taste / video fps

os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)