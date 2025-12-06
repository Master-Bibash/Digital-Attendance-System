import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List
from werkzeug.datastructures import FileStorage  # for Flask uploaded files
import os
import cv2
import numpy as np
from typing import List
from werkzeug.datastructures import FileStorage

mp_face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

def make_background_white_with_face(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    white_bg = np.ones_like(img, dtype=np.uint8) * 255

    results = mp_face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return img  # fallback if no face detected

    mask = np.zeros((h, w), dtype=np.uint8)
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * w))
        y1 = int(max(0, bbox.ymin * h))
        x2 = int(min(w, (bbox.xmin + bbox.width) * w))
        y2 = int(min(h, (bbox.ymin + bbox.height) * h))
        mask[y1:y2, x1:x2] = 255

    mask_3c = cv2.merge([mask, mask, mask])
    fg = cv2.bitwise_and(img, mask_3c)
    bg = cv2.bitwise_and(white_bg, cv2.bitwise_not(mask_3c))
    return cv2.add(fg, bg)
def save_images_with_white_bg(files: List[FileStorage], folder: str) -> int:
    """
    Save uploaded images with white background.
    Hard-coded algorithm capturing face + hair using HSV masks + GrabCut fallback.
    """
    os.makedirs(folder, exist_ok=True)
    saved_count = 0

    for f in files:
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        white_bg = np.ones_like(img, dtype=np.uint8) * 255

        # ---------------- Hard-coded logic ----------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Skin mask
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 180, 255])
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

        # Hair mask (broader range)
        lower_hair = np.array([0, 0, 0])
        upper_hair = np.array([180, 100, 120])  # increased coverage to capture dark hair
        mask_hair = cv2.inRange(hsv, lower_hair, upper_hair)

        # Combine masks
        mask = cv2.bitwise_or(mask_skin, mask_hair)

        # Smooth & fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        # GrabCut fallback
        grabcut_mask = np.where(mask>0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(img, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask_grabcut = np.where((grabcut_mask==cv2.GC_FGD) | (grabcut_mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        except:
            mask_grabcut = mask

        # Keep largest contour
        contours, _ = cv2.findContours(mask_grabcut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_final = np.zeros((h, w), dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            mask_final = mask_grabcut

        mask_final = cv2.GaussianBlur(mask_final, (7,7), 0)

        # Apply mask
        mask_3c = cv2.merge([mask_final]*3)
        fg = cv2.bitwise_and(img, mask_3c)
        bg = cv2.bitwise_and(white_bg, cv2.bitwise_not(mask_3c))
        img_white_bg = cv2.add(fg, bg)

        # Save
        filename = os.path.join(folder, f"{saved_count+1}.jpg")
        cv2.imwrite(filename, img_white_bg)
        saved_count += 1

    return saved_count
