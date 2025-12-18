"""
model.py  –  embedding extraction + Random-Forest training / inference
"""
import os
import cv2
import numpy as np
import pickle
from typing import List, Optional, Dict, Tuple, Callable
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
MODEL_PATH = "model.pkl"
from model import CONFIDENCE_THRESHOLD
from image_utils import make_background_white_with_face   # authoritative version



def crop_face_and_embed(bgr_image: np.ndarray, detection) -> Optional[np.ndarray]:
    """Crop face, convert to 32×32 grey, flatten, L2-normalise."""
    h, w = bgr_image.shape[:2]
    bbox = detection.location_data.relative_bounding_box
    x1 = int(max(0, bbox.xmin * w))
    y1 = int(max(0, bbox.ymin * h))
    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
    y2 = int(min(h, (bbox.ymin + bbox.height) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    face = cv2.cvtColor(bgr_image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
    return (face.flatten().astype(np.float32) / 255.0)


def extract_embedding_for_image(stream_or_bytes, first_only: bool = True) -> List[np.ndarray]:
    """Return list of face embeddings (length 0 … N) from file-like object."""
    data = stream_or_bytes.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []

    # thread-safe: create a NEW detector every call (cost ≈ 2 ms)
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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


# ------------------------------------------------------------------
# 2.  model I/O
# ------------------------------------------------------------------
def load_model_if_exists() -> Optional[RandomForestClassifier]:
    return pickle.load(open(MODEL_PATH, "rb")) if os.path.exists(MODEL_PATH) else None


def save_model(clf: RandomForestClassifier) -> None:
    tmp = MODEL_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(clf, f)
    os.replace(tmp, MODEL_PATH)          # atomic on POSIX and Win


# ------------------------------------------------------------------
# 3.  prediction
# ------------------------------------------------------------------
def predict_with_model(clf: RandomForestClassifier, emb: np.ndarray) -> Tuple[int, float]:
    """Single embedding → (student_id, confidence)."""
    proba = clf.predict_proba([emb])[0]
    if np.allclose(proba, 0.0):
        return -1, 0.0
    idx = int(np.argmax(proba))
    return int(clf.classes_[idx]), float(proba[idx])


def predict_multiple_faces(clf: RandomForestClassifier,
                           embeddings: List[np.ndarray],
                           threshold: float = CONFIDENCE_THRESHOLD) -> Dict:
    """Vote over many faces (same image)."""
    preds = [predict_with_model(clf, e) for e in embeddings]
    filtered = [(lbl, conf) for lbl, conf in preds if conf >= threshold]
    if not filtered:
        return {"recognized": False, "winner_label": None}

    # average confidence per label
    conf_per_label: Dict[int, List[float]] = {}
    for lbl, conf in filtered:
        conf_per_label.setdefault(lbl, []).append(conf)
    avg_conf = {lbl: sum(c) / len(c) for lbl, c in conf_per_label.items()}
    winner = max(avg_conf, key=avg_conf.get)
    return {"recognized": True,
            "winner_label": winner,
            "average_confidence": avg_conf[winner]}


# ------------------------------------------------------------------
# 4.  training
# ------------------------------------------------------------------
def train_model_background(dataset_dir: str,
                           progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
    """Train Random-Forest on white-bg images inside dataset_dir."""
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    X, y = [], []
    student_dirs = [d for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d))]
    total = max(1, len(student_dirs))

    for idx, sid in enumerate(student_dirs, 1):
        folder = os.path.join(dataset_dir, sid)
        for fn in os.listdir(folder):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                continue

            img = make_background_white_with_face(img)          # authoritative
            results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.detections:
                continue

            emb = crop_face_and_embed(img, results.detections[0])
            if emb is not None:
                X.append(emb)
                y.append(int(sid))

        if progress_callback:
            pct = int(idx / total * 100)
            progress_callback(pct, f"Processed {idx}/{total} students")

    if not X:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(np.stack(X), np.array(y))
    save_model(clf)
    if progress_callback:
        progress_callback(100, "Training complete")


