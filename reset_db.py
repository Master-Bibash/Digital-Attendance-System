import os
import sqlite3
import shutil

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")

# ------------------- Reset database -------------------
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

# ------------------- Delete dataset folders -------------------
if os.path.isdir(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)
print("✅ Dataset folder cleared!")
