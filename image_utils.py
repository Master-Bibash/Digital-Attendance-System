"""
image_utils.py – simple white background image saver (NO MediaPipe Tasks)
"""

import os
import cv2
import numpy as np
from typing import List
from werkzeug.datastructures import FileStorage


# -------------------------------------------------
# Simple white background (no face detection here)
# -------------------------------------------------
def make_background_white_simple(img: np.ndarray) -> np.ndarray:
    """
    Convert light background to white using grayscale threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    mask_3c = cv2.merge([mask, mask, mask])

    fg = cv2.bitwise_and(img, cv2.bitwise_not(mask_3c))
    bg = cv2.bitwise_and(white_bg, mask_3c)

    return cv2.add(fg, bg)


# -------------------------------------------------
# Save uploaded images with white background
# -------------------------------------------------
def save_images_with_white_bg(files: List[FileStorage], folder: str) -> int:
    os.makedirs(folder, exist_ok=True)
    saved_count = 0

    for f in files:
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_white = make_background_white_simple(img)

        filename = os.path.join(folder, f"{saved_count + 1}.jpg")
        cv2.imwrite(filename, img_white)
        saved_count += 1

    return saved_count
