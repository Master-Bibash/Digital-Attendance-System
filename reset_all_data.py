import os
import sqlite3
import shutil
import json

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")
TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

print("=" * 60)
print("🗑️  COMPLETE DATA RESET TOOL")
print("=" * 60)

# Step 1: Delete database
print("\n[1/4] Clearing database...")
try:
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Delete all rows
        cur.execute("DELETE FROM students")
        cur.execute("DELETE FROM attendance")
        
        # Reset auto-increment counters
        cur.execute("DELETE FROM sqlite_sequence")
        
        conn.commit()
        conn.close()
        print("✅ Database cleared!")
    else:
        print("⚠️  Database file not found (already deleted)")
except Exception as e:
    print(f"❌ Error clearing database: {e}")

# Step 2: Delete dataset folder (all student face images)
print("\n[2/4] Deleting student dataset folder...")
try:
    if os.path.isdir(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        os.makedirs(DATASET_DIR, exist_ok=True)
        print("✅ Dataset folder cleared!")
    else:
        print("⚠️  Dataset folder not found (already deleted)")
except Exception as e:
    print(f"❌ Error deleting dataset: {e}")

# Step 3: Delete trained model
print("\n[3/4] Deleting trained model...")
try:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("✅ Model deleted!")
    else:
        print("⚠️  Model file not found (already deleted)")
except Exception as e:
    print(f"❌ Error deleting model: {e}")

# Step 4: Reset training status
print("\n[4/4] Resetting training status...")
try:
    status = {"running": False, "progress": 0, "message": "No training yet."}
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status, f)
    print("✅ Training status reset!")
except Exception as e:
    print(f"❌ Error resetting status: {e}")

print("\n" + "=" * 60)
print("✨ RESET COMPLETE! Ready to start fresh.")
print("=" * 60)
print("\nNext steps:")
print("1. Refresh your browser (F5 or Ctrl+R)")
print("2. Add new students")
print("3. Upload face images")
print("4. Train the model")


import os
import pickle

MODEL_PATH = 'model.pkl'

def remove_student_from_model(student_id):
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if student_id in model:
        del model[student_id]

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Student with ID {student_id} has been removed from the model.")

if __name__ == "__main__":
    student_id = 2
    remove_student_from_model(student_id)