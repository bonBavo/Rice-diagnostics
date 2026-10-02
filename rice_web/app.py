"""
Rice Leaf Doctor - Flask web app for the rice disease classifier.

Run:   python app.py        then open http://127.0.0.1:5000
Needs: a trained model folder (default: ./model_out) produced by rice_cnn.py,
       and rice_cnn.py in the same folder as this file.

Environment variables (all optional):
    MODEL_DIR             folder with rice_disease_model.keras + class_names.json
    CONFIDENCE_THRESHOLD  below this the result is flagged "not sure" (default 0.70)
    SECRET_KEY            keeps browser sessions valid across restarts
    HOST, PORT            default 127.0.0.1 and 5000
"""
import contextlib
import json
import os
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import (Flask, abort, g, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image, ImageOps, UnidentifiedImageError

import tensorflow as tf

import rice_cnn

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE / "model_out"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE / "uploads"))
DB_PATH = Path(os.environ.get("DB_PATH", BASE / "history.db"))
MAX_UPLOAD_MB = 8
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
EAT = timezone(timedelta(hours=3), "EAT")          # Kenya time (no daylight saving)
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}           # MPO = some phone JPEGs

Image.MAX_IMAGE_PIXELS = 50_000_000                # refuse absurdly large images

app = Flask(__name__)
if not os.environ.get("SECRET_KEY"):
    print("Note: SECRET_KEY is not set, so history links reset when the server restarts.")
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
    SESSION_COOKIE_SAMESITE="Lax",                 # also blocks cross-site form posts
    PERMANENT_SESSION_LIFETIME=timedelta(days=365),
)

ADVICE = json.loads((BASE / "advice.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Model (loaded once at start-up)
# --------------------------------------------------------------------------
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_DIR / rice_cnn.MODEL_FILE)
        meta = json.loads((MODEL_DIR / rice_cnn.META_FILE).read_text())
        return model, meta, None
    except Exception as exc:  # missing or incompatible model
        print(f"Could not load model from {MODEL_DIR}: {exc}")
        return None, None, (f"The diagnosis model was not found in '{MODEL_DIR}'. "
                            "Train it first with rice_cnn.py, then restart the app.")


MODEL, META, MODEL_ERROR = load_model()
MODEL_LOCK = threading.Lock()                      # one prediction at a time


# --------------------------------------------------------------------------
# Database (SQLite for the prototype; swap for MySQL/PostgreSQL when scaling)
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS diagnoses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    image_file  TEXT NOT NULL,
    prediction  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    status      TEXT NOT NULL,
    top_k       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_client ON diagnoses (client_id, id);
"""


def init_db():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(DB_PATH)) as db:
        db.executescript(SCHEMA)
        db.commit()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


init_db()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def client_id():
    """Anonymous per-browser id: each browser only sees its own history."""
    if "cid" not in session:
        session["cid"] = uuid.uuid4().hex
        session.permanent = True
    return session["cid"]


def display_name(label):
    if label == "unsure":
        return "Not sure"
    entry = ADVICE.get(label)
    return entry["name"] if entry else label.replace("_", " ").capitalize()


def save_upload(file_storage):
    """Validate the upload as a real image and store a clean JPEG copy.

    The original file name is never used, so nothing a user sends can
    become a path on the server.
    """
    try:
        img = Image.open(file_storage.stream)
        if img.format not in ALLOWED_FORMATS:
            return None
        img = ImageOps.exif_transpose(img).convert("RGB")   # upright, fully decoded
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None
    img.thumbnail((1600, 1600))
    name = f"{uuid.uuid4().hex}.jpg"
    img.save(UPLOAD_DIR / name, "JPEG", quality=90)
    return name


@app.template_filter("local_time")
def local_time(iso):
    return datetime.fromisoformat(iso).astimezone(EAT).strftime("%d %b %Y, %H:%M")


@app.template_filter("pct")
def pct(value):
    return f"{value * 100:.1f}%"


app.jinja_env.globals["display_name"] = display_name


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", model_error=MODEL_ERROR)


@app.post("/diagnose", endpoint="submit")
def submit():
    if MODEL is None:
        return render_template("index.html", model_error=MODEL_ERROR), 503
    file = request.files.get("photo")
    if file is None or not file.filename:
        return render_template("index.html", error="Please choose a photo first."), 400
    saved = save_upload(file)
    if saved is None:
        return render_template(
            "index.html",
            error="We could not read that file. Please upload a JPEG or PNG photo of a rice leaf."), 400

    with MODEL_LOCK:
        result = rice_cnn.diagnose(UPLOAD_DIR / saved, MODEL, META, top_k=3, threshold=THRESHOLD)

    db = get_db()
    cur = db.execute(
        "INSERT INTO diagnoses (client_id, created_at, image_file, prediction, confidence, status, top_k) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (client_id(), datetime.now(timezone.utc).isoformat(timespec="seconds"), saved,
         result["prediction"], result["confidence"], result["status"], json.dumps(result["top_k"])))
    db.commit()
    return redirect(url_for("result", diagnosis_id=cur.lastrowid))   # refresh-safe


def own_record(diagnosis_id):
    row = get_db().execute("SELECT * FROM diagnoses WHERE id = ? AND client_id = ?",
                           (diagnosis_id, client_id())).fetchone()
    if row is None:
        abort(404)
    return row


@app.get("/result/<int:diagnosis_id>")
def result(diagnosis_id):
    row = own_record(diagnosis_id)
    advice = ADVICE.get(row["prediction"]) if row["status"] == "confident" else None
    return render_template("result.html", row=row, top_k=json.loads(row["top_k"]), advice=advice,
                           threshold=THRESHOLD)


@app.get("/photo/<filename>")
def photo(filename):
    row = get_db().execute("SELECT id FROM diagnoses WHERE image_file = ? AND client_id = ?",
                           (filename, client_id())).fetchone()
    if row is None:
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/history")
def history():
    db, cid = get_db(), client_id()
    rows = db.execute("SELECT * FROM diagnoses WHERE client_id = ? ORDER BY id DESC LIMIT 200",
                      (cid,)).fetchall()
    counts = db.execute(
        "SELECT CASE WHEN status = 'confident' THEN prediction ELSE 'unsure' END AS label, "
        "COUNT(*) AS n FROM diagnoses WHERE client_id = ? GROUP BY label ORDER BY n DESC",
        (cid,)).fetchall()
    return render_template("history.html", rows=rows, counts=counts)


@app.post("/delete/<int:diagnosis_id>")
def delete(diagnosis_id):
    row = own_record(diagnosis_id)
    with contextlib.suppress(OSError):
        (UPLOAD_DIR / row["image_file"]).unlink()
    db = get_db()
    db.execute("DELETE FROM diagnoses WHERE id = ?", (diagnosis_id,))
    db.commit()
    return redirect(url_for("history"))


@app.errorhandler(413)
def too_large(_e):
    return render_template("index.html",
                           error=f"That photo is too large. Please use one under {MAX_UPLOAD_MB} MB."), 413


@app.errorhandler(404)
def not_found(_e):
    return render_template("message.html", title="Not found",
                           text="That page or record does not exist."), 404


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))
