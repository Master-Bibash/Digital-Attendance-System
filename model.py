import os
import cv2
import numpy as np
import pickle
from typing import List, Optional, Dict, Tuple, Callable
from deepface import DeepFace

MODEL_PATH = "model.pkl"

# ------------------------------------
# Extract embeddings from image
# ------------------------------------
def extract_embedding_for_image(stream_or_bytes) -> Optional[np.ndarray]:
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None
        print("[ERROR] Could not read image")
        return None

    try:
        embedding = DeepFace.represent(img, model_name='ArcFace', enforce_detection=False)[0]['embedding']
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] DeepFace embedding failed: {e}")
        return None

# ------------------------------------
# Load / Save model
# ------------------------------------
def load_model_if_exists() -> Optional[dict]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def save_model(canonical: dict):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(canonical, f)

# ------------------------------------
# Predict single embedding
# ------------------------------------
def predict_with_model(canonical: dict, emb: np.ndarray) -> Tuple[int, float]:
    best_id, best_sim = None, 0
    for sid, template in canonical.items():
        sim = np.dot(emb, template) / (np.linalg.norm(emb) * np.linalg.norm(template))
        if sim > best_sim:
            best_sim, best_id = sim, sid
    return best_id, float(best_sim)

# ------------------------------------
# Train model (BACKGROUND THREAD)
# ------------------------------------
def train_model_background(dataset_dir: str, progress_callback: Optional[Callable[[int, str], None]] = None):
    try:
        student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if not student_dirs:
            if progress_callback:
                progress_callback(0, "No student folders found")
            return

        total = len(student_dirs)
        canonical = {}

        for idx, sid in enumerate(student_dirs, start=1):
            folder = os.path.join(dataset_dir, sid)
            embs = []

            for file in os.listdir(folder):
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                path = os.path.join(folder, file)
                img = cv2.imread(path)
                if img is None:
                    continue

                try:
                    emb = DeepFace.represent(img, model_name='ArcFace', enforce_detection=False)[0]['embedding']
                    embs.append(np.array(emb, dtype=np.float32))
                except Exception as e:
                    print(f"[ERROR] DeepFace embedding failed: {e}")

            if embs:
                canonical[int(sid)] = np.mean(embs, axis=0)

            if progress_callback:
                progress = int((idx / total) * 100)
                progress_callback(progress, f"Processed {idx}/{total} students")

        if not canonical:
            if progress_callback:
                progress_callback(0, "No faces detected in dataset")
            return

        save_model(canonical)
        if progress_callback:
            progress_callback(100, "Training completed successfully")

    except Exception as e:
        if progress_callback:
            progress_callback(0, f"Training error: {str(e)}")
        raise