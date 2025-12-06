import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Dict, Tuple, Callable
from image_utils import make_background_white_with_face


MODEL_PATH = "model.pkl"


# ------------------------------
# Utility: Make background white
# ------------------------------
import cv2
import numpy as np

def make_background_white_in_memory(img: np.ndarray, threshold: int = 190) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold: background pixels -> white
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Smooth mask to remove small holes
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert mask to 3 channels
    mask_3c = cv2.merge([mask, mask, mask])
    
    # Create white background
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    
    # Apply mask
    fg = cv2.bitwise_and(img, cv2.bitwise_not(mask_3c))
    bg = cv2.bitwise_and(white_bg, mask_3c)
    
    return cv2.add(fg, bg)



def make_background_white(image_path: str, output_path: str, threshold: int = 240):
    """
    Apply white-background preprocessing to an image file and save result.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    result = make_background_white_with_face(img)

    cv2.imwrite(output_path, result)
    print(f"Saved white-background image: {output_path}")


# ------------------------------
# Utility: Crop face & create embedding
# ------------------------------
def crop_face_and_embed(bgr_image: np.ndarray, detection) -> Optional[np.ndarray]:
    """
    Crop a detected face and convert it to a normalized 32x32 grayscale embedding.
    """
    h, w = bgr_image.shape[:2]
    bbox = detection.location_data.relative_bounding_box
    x1 = int(max(0, bbox.xmin * w))
    y1 = int(max(0, bbox.ymin * h))
    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
    y2 = int(min(h, (bbox.ymin + bbox.height) * h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32) / 255.0
    return emb


# ------------------------------
# Embedding extraction
# ------------------------------
def extract_embedding_for_image(stream_or_bytes, first_only: bool = True) -> List[np.ndarray]:
    """
    Extract embeddings from bytes stream or file object.
    Returns list of embeddings (or single if first_only=True).
    """
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    
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


# ------------------------------
# Model load/save
# ------------------------------
def load_model_if_exists() -> Optional[RandomForestClassifier]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def save_model(clf: RandomForestClassifier):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)


# ------------------------------
# Single embedding prediction
# ------------------------------
def predict_with_model(clf: RandomForestClassifier, emb: np.ndarray) -> Tuple[int, float]:
    """
    Predict a single embedding.
    Returns (label, confidence)
    """
    proba = clf.predict_proba([emb])[0]
    idx = int(np.argmax(proba))
    label = int(clf.classes_[idx])
    conf = float(proba[idx])
    return label, conf


# ------------------------------
# Multi-face prediction
# ------------------------------
def predict_multiple_faces(
    clf: RandomForestClassifier,
    embeddings: List[np.ndarray],
    threshold: float = 0.5
) -> Dict:
    """
    Predict multiple faces with simple voting aggregation.
    """
    predictions = []
    for emb in embeddings:
        label, conf = predict_with_model(clf, emb)
        predictions.append({"label": label, "confidence": conf})
    
    # Filter by confidence threshold
    filtered = [p for p in predictions if p["confidence"] >= threshold]
    if not filtered:
        return {"recognized": False, "predictions": predictions, "winner_label": None}
    
    # Average confidence per label
    label_conf = {}
    for p in filtered:
        lbl = p["label"]
        label_conf.setdefault(lbl, []).append(p["confidence"])
    avg_conf = {lbl: sum(confs)/len(confs) for lbl, confs in label_conf.items()}
    
    winner_label = max(avg_conf, key=lambda k: avg_conf[k])
    
    return {
        "recognized": True,
        "predictions": predictions,
        "winner_label": winner_label,
        "average_confidence": avg_conf[winner_label]
    }


# ------------------------------
# Training function
# ------------------------------
def train_model_background(
    dataset_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    threshold: int = 240
):
    """
    Train RandomForest using dataset directory.
    dataset_dir/
        student_id/
            img1.jpg
            img2.jpg
    """
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    X, y = [], []
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_students = max(1, len(student_dirs))
    
    for idx, sid in enumerate(student_dirs, start=1):
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            
            # Make background white
            # Make background white using face-aware method
            img = make_background_white_with_face(img)

            
            results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.detections:
                continue
            
            emb = crop_face_and_embed(img, results.detections[0])
            if emb is None:
                continue
            
            X.append(emb)
            y.append(int(sid))
        
        if progress_callback:
            pct = int((idx / total_students) * 100)
            progress_callback(pct, f"Processed {idx}/{total_students} students")
    
    if not X:
        if progress_callback:
            progress_callback(0, "No training data found")
        return
    
    X = np.stack(X)
    y = np.array(y)
    
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    
    save_model(clf)
    if progress_callback:
        progress_callback(100, "Training complete")


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Single image preprocessing
    make_background_white("input.jpg", "output_white_bg.jpg")
    
    # Optionally train model
    # train_model_background("dataset")
