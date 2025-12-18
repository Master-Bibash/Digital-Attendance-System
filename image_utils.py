# image_utils.py  –  crisp face + white background
import os
import cv2
import numpy as np
from typing import List
from werkzeug.datastructures import FileStorage
import mediapipe as mp

# ---------- helper ----------
def _white_bg_around_face(bgr: np.ndarray,
                          x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Return image where everything OUTSIDE (x1,y1)-(x2,y2) is white.
    Face region is copied 1:1 – no resize, no re-compression.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    result = bgr.copy()

    # expand rectangle a bit so we don’t paint over chin/forehead
    margin = int(0.25 * max(y2 - y1, x2 - x1))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    # flood-fill from the four corners
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(result, mask, seed, (255, 255, 255),
                      loDiff=(3, 3, 3, 3), upDiff=(3, 3, 3, 3), flags=8)

    # restore original crisp face
    result[y1:y2, x1:x2] = bgr[y1:y2, x1:x2]
    return result


# ---------- main entry ----------
def save_images_with_white_bg(files: List[FileStorage], folder: str) -> int:
    os.makedirs(folder, exist_ok=True)
    saved_count = 0

    base_options = mp.tasks.BaseOptions(
        model_asset_path="model/blaze_face_short_range.tflite")
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5)
    detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    for f in files:
        try:
            data = f.read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detections = detector.detect(mp_image).detections
            if not detections:
                continue

            bbox = detections[0].bounding_box
            x1 = int(max(0, bbox.origin_x))
            y1 = int(max(0, bbox.origin_y))
            x2 = int(min(img.shape[1], bbox.origin_x + bbox.width))
            y2 = int(min(img.shape[0], bbox.origin_y + bbox.height))

            white_bg = _white_bg_around_face(img, x1, y1, x2, y2)

            # optional light enhancements (only outside the face)
            white_bg = cv2.convertScaleAbs(white_bg, alpha=1.1, beta=10)
            white_bg = cv2.fastNlMeansDenoisingColored(white_bg, None,
                                                       10, 10, 7, 21)

            out_path = os.path.join(folder, f"{saved_count + 1}.jpg")
            cv2.imwrite(out_path, white_bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        except Exception as e:
            print(f"white-bg error: {e}")
            continue

    return saved_count