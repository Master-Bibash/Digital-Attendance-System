import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Dict, Tuple, Callable
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


MODEL_PATH = "model.pkl"


# ------------------------------------
# Crop face & create embedding
# ------------------------------------
def crop_face_and_embed(bgr_image: np.ndarray, detection) -> Optional[np.ndarray]:
    h, w = bgr_image.shape[:2]
    
    # Fixed: MediaPipe Tasks API uses bounding_box instead of location_data.relative_bounding_box
    bbox = detection.bounding_box
    
    # bbox has origin_x, origin_y, width, height (in pixels, not relative)
    x1 = int(max(0, bbox.origin_x))
    y1 = int(max(0, bbox.origin_y))
    x2 = int(min(w, bbox.origin_x + bbox.width))
    y2 = int(min(h, bbox.origin_y + bbox.height))

    if x2 <= x1 or y2 <= y1:
        return None

    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)

    emb = face.flatten().astype(np.float32) / 255.0
    return emb


# ------------------------------------
# Extract embeddings from image
# ------------------------------------
def extract_embedding_for_image(stream_or_bytes, first_only: bool = True) -> List[np.ndarray]:
    base_options = mp_tasks.BaseOptions(
        model_asset_path="model/blaze_face_short_range.tflite"
    )

    detector_options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    face_detector = vision.FaceDetector.create_from_options(detector_options)

    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fixed: Use correct import path for Image
    from mediapipe import Image as mp_Image
    from mediapipe import ImageFormat
    
    mp_image = mp_Image(
        image_format=ImageFormat.SRGB,
        data=rgb_image
    )

    results = face_detector.detect(mp_image)
    if not results.detections:
        return []

    embeddings = []
    for det in results.detections:
        emb = crop_face_and_embed(img, det)
        if emb is not None:
            embeddings.append(emb)
        if first_only:
            break

    return embeddings


# ------------------------------------
# Model load / save
# ------------------------------------
def load_model_if_exists() -> Optional[RandomForestClassifier]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def save_model(clf: RandomForestClassifier):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)


# ------------------------------------
# Predict single embedding
# ------------------------------------
def predict_with_model(clf: RandomForestClassifier, emb: np.ndarray) -> Tuple[int, float]:
    proba = clf.predict_proba([emb])[0]
    idx = int(np.argmax(proba))
    label = int(clf.classes_[idx])
    conf = float(proba[idx])
    return label, conf


# ------------------------------------
# Predict multiple faces
# ------------------------------------
def predict_multiple_faces(
    clf: RandomForestClassifier,
    embeddings: List[np.ndarray],
    threshold: float = 0.5
) -> Dict:

    predictions = []
    for emb in embeddings:
        label, conf = predict_with_model(clf, emb)
        predictions.append({"label": label, "confidence": conf})

    filtered = [p for p in predictions if p["confidence"] >= threshold]
    if not filtered:
        return {"recognized": False}

    winner = max(filtered, key=lambda x: x["confidence"])

    return {
        "recognized": True,
        "label": winner["label"],
        "confidence": winner["confidence"]
    }


# ------------------------------------
# Train model (BACKGROUND THREAD)
# ------------------------------------
def train_model_background(
    dataset_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
):
    try:
        base_options = mp_tasks.BaseOptions(
            model_asset_path="model/blaze_face_short_range.tflite"
        )

        detector_options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )

        mp_face = vision.FaceDetector.create_from_options(detector_options)

        # Fixed: Import Image class correctly
        from mediapipe import Image as mp_Image
        from mediapipe import ImageFormat

        X, y = [], []

        student_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]

        if not student_dirs:
            if progress_callback:
                progress_callback(0, "No student folders found")
            return

        total = len(student_dirs)

        for idx, sid in enumerate(student_dirs, start=1):
            folder = os.path.join(dataset_dir, sid)

            for file in os.listdir(folder):
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                path = os.path.join(folder, file)
                img = cv2.imread(path)
                if img is None:
                    continue

                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Fixed: Use correct Image class
                mp_image = mp_Image(
                    image_format=ImageFormat.SRGB,
                    data=rgb_image
                )

                results = mp_face.detect(mp_image)
                if not results.detections:
                    continue

                emb = crop_face_and_embed(img, results.detections[0])
                if emb is None:
                    continue

                X.append(emb)
                y.append(int(sid))

            if progress_callback:
                progress = int((idx / total) * 100)
                progress_callback(progress, f"Processed {idx}/{total} students")

        if not X:
            if progress_callback:
                progress_callback(0, "No faces detected in any images")
            return

        if progress_callback:
            progress_callback(95, "Training classifier...")

        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(np.array(X), np.array(y))

        save_model(clf)

        if progress_callback:
            progress_callback(100, "Training completed successfully")
            
    except Exception as e:
        if progress_callback:
            progress_callback(0, f"Training error: {str(e)}")
        raise