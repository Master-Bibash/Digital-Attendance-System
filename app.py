import os
import io
import threading
import sqlite3
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file

from functions import apply_smoothing, histogram_equalization, rgb_to_grayscale
from model import (
    load_model_if_exists,
    extract_embedding_for_image,
    train_model_background,
    predict_with_model
)





MODEL_CACHE = {"clf": None}

def get_model():
    if MODEL_CACHE["clf"] is None:
        MODEL_CACHE["clf"] = load_model_if_exists()
    return MODEL_CACHE["clf"]

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DB helpers ----------
def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT UNIQUE,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/next_roll")
def next_roll():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT MAX(CAST(roll AS INTEGER)) FROM students WHERE roll != ''")
    max_roll = c.fetchone()[0] or 0
    conn.close()
    return jsonify(next_roll=max_roll + 1)

@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = get_db()
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    if df.empty:
        from datetime import date, timedelta
        days = [(date.today() - timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
    counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
    dates = [d.strftime("%d-%b") for d in last_30]
    return jsonify({"dates": dates, "counts": counts})

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    data = request.form
    name = data.get("name", "").strip()
    roll = data.get("roll", "").strip()
    cls = data.get("class", "").strip()
    sec = data.get("sec", "").strip()
    reg_no = data.get("reg_no", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    conn = get_db()
    c = conn.cursor()
    if roll:
        c.execute("SELECT id FROM students WHERE roll = ?", (roll,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": f"Roll number '{roll}' already exists"}), 400
    now = datetime.datetime.now(datetime.UTC).isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, reg_no, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, roll, cls, sec, reg_no, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})

@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id required"}), 400
    files = request.files.getlist("images[]")
    folder = os.path.join(DATASET_DIR, student_id)
    saved_count = 0
    for f in files:
        f.save(os.path.join(folder, f.filename))
        saved_count += 1
    return jsonify({"saved": saved_count})

@app.route("/train_model", methods=["GET"])
def train_model_route():
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status": "already_running"}), 202
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    def training_callback(p, m):
        write_train_status({"running": p < 100, "progress": p, "message": m})
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, training_callback))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    import time
    return render_template("mark_attendance.html", now=time.time())

@app.route("/student_portal")
def student_portal():
    return render_template("student_portal.html")


@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error": "no image"}), 400
    img_file = request.files["image"]
    try:
        image_bytes = img_file.read()
        
        # Manually decode the image bytes to RGB format
        from PIL import Image
        from io import BytesIO
        
        image = Image.open(BytesIO(image_bytes))
        rgb_image = list(image.getdata())
        width, height = image.size
        
        # Convert the flat list of RGB tuples to a 2D list
        rgb_image_2d = [rgb_image[i * width:(i + 1) * width] for i in range(height)]
        
        # Convert the RGB image to grayscale
        grayscale_image = rgb_to_grayscale(rgb_image_2d)
        
        # Apply smoothing to the grayscale image
        smoothed_image = apply_smoothing(grayscale_image)
        
        # Apply histogram equalization to enhance contrast
        equalized_image = histogram_equalization(smoothed_image)
        
        # Convert the equalized image back to a format suitable for processing
        equalized_bytes = bytearray()
        for row in equalized_image:
            for gray in row:
                equalized_bytes.append(gray)
        
        # Convert the equalized bytes back to an image
        equalized_image = Image.frombytes('L', (width, height), bytes(equalized_bytes))
        
        # Save the equalized image to a BytesIO object
        equalized_io = BytesIO()
        equalized_image.save(equalized_io, format='JPEG')
        equalized_io.seek(0)
        
        # Extract embeddings from the equalized image
        emb = extract_embedding_for_image(equalized_io)
        if emb is None:
            return jsonify({"recognized": False, "error": "no face detected"}), 200

        clf = get_model()
        if clf is None:
            return jsonify({"recognized": False, "error": "model not trained"}), 200

        pred_label, conf = predict_with_model(clf, emb)
        if conf < 0.7:
            return jsonify({"recognized": False, "reason": "low_confidence", "confidence": float(conf)})

        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (int(pred_label),))
        row = c.fetchone()
        name = row[0] if row else "Unknown"

        today = datetime.date.today().isoformat()
        c.execute("""SELECT id FROM attendance WHERE student_id = ? AND date(timestamp) = ?""", (int(pred_label), today))
        if c.fetchone():
            return jsonify({"recognized": False, "student_id": int(pred_label), "name": name, "error": "Attendance already recorded today"}), 200

        ts = datetime.datetime.now(datetime.UTC).isoformat()
        c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", (int(pred_label), name, ts))
        conn.commit()
        conn.close()
        print(f"Recognized: {name}, Confidence: {conf}")

        return jsonify({"recognized": True, "student_id": int(pred_label), "name": name, "confidence": float(conf)}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")
    conn = get_db()
    c = conn.cursor()
    q = "SELECT id, student_id, name, timestamp FROM attendance"
    params = ()
    if period == "daily":
        today = datetime.date.today().isoformat()
        q += " WHERE date(timestamp) = ?"
        params = (today,)
    elif period == "weekly":
        start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        q += " WHERE date(timestamp) >= ?"
        params = (start,)
    elif period == "monthly":
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        q += " WHERE date(timestamp) >= ?"
        params = (start,)
    q += " ORDER BY timestamp DESC LIMIT 5000"
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return render_template("attendance_record.html", records=rows, period=period)

@app.route("/student_count")
def student_count():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    total = c.fetchone()[0]
    conn.close()
    return jsonify({"total_students": total})

@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, student_id, name, timestamp FROM attendance ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    output = io.StringIO()
    output.write("id,student_id,name,timestamp\n")
    for r in rows:
        output.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

@app.route("/students", methods=["GET"])
def students_list():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [{"id": r[0], "name": r[1], "roll": r[2], "class": r[3], "section": r[4], "reg_no": r[5], "created_at": r[6]} for r in rows]
    return jsonify({"students": data})

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})

if __name__ == "__main__":
    app.run(debug=True)