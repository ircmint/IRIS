"""
config.py -- shared paths for the metrics pipeline.

This file was missing from the provided Metrics folder (compute_chr.py,
compute_rhr.py, etc. all import OUTPUT_DIR from here) -- added so the
existing scripts run as-is.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
