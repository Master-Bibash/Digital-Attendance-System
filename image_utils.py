
import os
import cv2
import numpy as np
import uuid
from typing import List
from werkzeug.datastructures import FileStorage

# ------------------------------------------------------------------
# single public helper  (used by both training and Flask routes)
# ------------------------------------------------------------------
def make_background_white_with_face(img: np.ndarray) -> np.ndarray:
    """
    HSV skin/hair mask → GrabCut → largest-contour → white background
    Thread-safe: creates its own MediaPipe detector every call.
    """
    h, w = img.shape[:2]
    white_bg = np.ones_like(img, dtype=np.uint8) * 255

    # ---- 1.  skin + hair mask ------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_skin = cv2.inRange(hsv, np.array([0, 30, 60]),   np.array([20, 180, 255]))
    mask_hair = cv2.inRange(hsv, np.array([0, 0, 0]),     np.array([180, 100, 120]))
    mask = cv2.bitwise_or(mask_skin, mask_hair)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,  iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # ---- 2.  GrabCut refine --------------------------------------
    gc_mask = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    mask_final = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # ---- 3.  keep largest contour -------------------------------
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_clean, [largest], -1, 255, thickness=cv2.FILLED)
        mask_final = cv2.GaussianBlur(mask_clean, (7, 7), 0)

    # ---- 4.  composite ------------------------------------------
    mask_bgr = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
    fg   = cv2.bitwise_and(img, mask_bgr)
    bg   = cv2.bitwise_and(white_bg, cv2.bitwise_not(mask_bgr))
    return cv2.add(fg, bg)


def save_images_with_white_bg(files: List[FileStorage], folder: str) -> int:
    os.makedirs(folder, exist_ok=True)

    # 1.  preprocess & write new images to *temporary* names
    tmp_paths = []
    for f in files:
        buf = f.read()
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"warning: corrupted image {f.filename}")   # or app_log
            continue
        img_white = make_background_white_with_face(img)
        tmp = os.path.join(folder, f"{uuid.uuid4().hex}.jpg")
        if cv2.imwrite(tmp, img_white):
            tmp_paths.append(tmp)

    # 2.  only now that we *know* the new images are on disk, delete olds
    for old in os.listdir(folder):
        if old.endswith(".jpg"):
            os.remove(os.path.join(folder, old))

    # 3.  optional: rename temps to final names (keeps uuid anyway)
    saved = len(tmp_paths)
    return saved