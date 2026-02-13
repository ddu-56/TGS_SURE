"""Generate ArUco markers for table corner detection.

Generates 4 marker images (IDs 0-3) from DICT_4X4_50 as individual PNGs
and a combined printable sheet.

Usage:
    python scripts/generate_markers.py [output_dir]
"""

import os
import sys

import cv2
import numpy as np


def generate_markers(output_dir: str = "markers") -> None:
    os.makedirs(output_dir, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 200  # pixels per marker
    labels = {0: "Top-Left", 1: "Top-Right", 2: "Bottom-Right", 3: "Bottom-Left"}

    markers = {}
    for marker_id, label in labels.items():
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        # Add white border for easier cutting
        bordered = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        # Add label text
        labeled = cv2.copyMakeBorder(bordered, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(
            labeled,
            f"ID {marker_id}: {label}",
            (10, labeled.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            0,
            2,
        )
        markers[marker_id] = labeled
        path = os.path.join(output_dir, f"marker_{marker_id}.png")
        cv2.imwrite(path, labeled)
        print(f"Saved {path} ({label})")

    # Create combined sheet (2x2 grid)
    h, w = list(markers.values())[0].shape[:2]
    gap = 40
    sheet = np.ones((h * 2 + gap * 3, w * 2 + gap * 3), dtype=np.uint8) * 255
    positions = [(gap, gap), (w + gap * 2, gap), (w + gap * 2, h + gap * 2), (gap, h + gap * 2)]
    for marker_id, (x, y) in zip(sorted(markers.keys()), positions):
        sheet[y:y + h, x:x + w] = markers[marker_id]

    sheet_path = os.path.join(output_dir, "marker_sheet.png")
    cv2.imwrite(sheet_path, sheet)
    print(f"\nSaved combined sheet: {sheet_path}")
    print("Print this sheet, cut out markers, and place at table corners.")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "markers"
    generate_markers(output_dir)
