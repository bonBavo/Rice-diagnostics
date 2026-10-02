"""
Flask routes for the Rice Leaf Doctor web application.

Provides:
    GET  /                 – upload page (frontend)
    GET  /health           – application health check
    POST /diagnose         – web form submission (redirects to result page)
    POST /api/v1/predict   – JSON API for image upload
    GET  /result/<id>      – diagnosis result page
    GET  /history          – diagnosis history page
    GET  /photo/<filename> – serve uploaded photos
    POST /delete/<id>      – delete a diagnosis record
"""
import contextlib
import json
import uuid
from datetime import datetime, timezone

from flask import (Blueprint, abort, current_app, g, redirect,
                   render_template, request, send_from_directory,
                   session, url_for, jsonify)
from PIL import Image, ImageOps, UnidentifiedImageError

bp = Blueprint("main", __name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}
Image.MAX_IMAGE_PIXELS = 50_000_000


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    import sqlite3
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def client_id():
    """Anonymous per-browser id so each browser only sees its own history."""
    if "cid" not in session:
        session["cid"] = uuid.uuid4().hex
        session.permanent = True
    return session["cid"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def display_name(label):
    if label == "unsure":
        return "Not sure"
    advice = current_app.config.get("ADVICE", {})
    entry = advice.get(label)
    return entry["name"] if entry else label.replace("_", " ").capitalize()


def save_upload(file_storage):
    """Validate the upload as a real image and store a clean JPEG copy."""
    try:
        img = Image.open(file_storage.stream)
        if img.format not in ALLOWED_FORMATS:
            return None
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None
    img.thumbnail((1600, 1600))
    name = f"{uuid.uuid4().hex}.jpg"
    upload_dir = current_app.config["UPLOAD_DIR"]
    img.save(upload_dir / name, "JPEG", quality=90)
    return name


# --------------------------------------------------------------------------
# Template filters and globals
# --------------------------------------------------------------------------
@bp.app_template_filter("local_time")
def local_time(iso):
    from datetime import timedelta
    EAT = timezone(timedelta(hours=3), "EAT")
    return datetime.fromisoformat(iso).astimezone(EAT).strftime("%d %b %Y, %H:%M")


@bp.app_template_filter("pct")
def pct(value):
    return f"{value * 100:.1f}%"


@bp.app_template_global()
def display_name_global(label):
    return display_name(label)


# Make display_name available as display_name() in templates
@bp.before_app_request
def _inject_display_name():
    current_app.jinja_env.globals["display_name"] = display_name


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@bp.get("/")
def index():
    model_error = current_app.config.get("MODEL_ERROR")
    return render_template("index.html", model_error=model_error)


@bp.get("/health")
def health():
    predictor = current_app.config.get("PREDICTOR")
    status = "ok" if predictor else "model_not_loaded"
    return jsonify({"status": status, "classes": predictor.class_names if predictor else []})


@bp.post("/diagnose", endpoint="submit")
def submit():
    predictor = current_app.config.get("PREDICTOR")
    if predictor is None:
        return render_template("index.html",
                               model_error=current_app.config.get("MODEL_ERROR")), 503

    file = request.files.get("photo")
    if file is None or not file.filename:
        return render_template("index.html", error="Please choose a photo first."), 400

    saved = save_upload(file)
    if saved is None:
        return render_template(
            "index.html",
            error="We could not read that file. Please upload a JPEG or PNG photo of a rice leaf."
        ), 400

    upload_dir = current_app.config["UPLOAD_DIR"]
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
    result = predictor.predict(upload_dir / saved, top_k=3, threshold=threshold)

    db = get_db()
    stored_prediction = (
        result["prediction"] if result["status"] == "confident" else "unsure"
    )
    cur = db.execute(
        "INSERT INTO diagnoses (client_id, created_at, image_file, prediction, "
        "confidence, status, top_k) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (client_id(), datetime.now(timezone.utc).isoformat(timespec="seconds"),
         saved, stored_prediction, result["confidence"],
         result["status"], json.dumps(result["top_k"])))
    db.commit()
    return redirect(url_for("main.result", diagnosis_id=cur.lastrowid))


@bp.post("/api/v1/predict")
def api_predict():
    """JSON API endpoint for image classification."""
    predictor = current_app.config.get("PREDICTOR")
    if predictor is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    file = request.files.get("image") or request.files.get("photo")
    if file is None or not file.filename:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    saved = save_upload(file)
    if saved is None:
        return jsonify({"success": False,
                        "error": "Unsupported or corrupt image file"}), 400

    upload_dir = current_app.config["UPLOAD_DIR"]
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
    try:
        result = predictor.predict(upload_dir / saved, top_k=3, threshold=threshold)
    except Exception:
        return jsonify({"success": False, "error": "Prediction failed"}), 500

    return jsonify({
        "success": True,
        "prediction": (
            result["prediction"] if result["status"] == "confident" else "unsure"
        ),
        "confidence": result["confidence"],
        "status": result["status"],
        "message": (
            None if result["status"] == "confident"
            else "This may not be a rice plant or leaf, or the image may be unclear. "
                 "The system could not confidently identify a supported rice-leaf class. "
                 "Upload a clear photo of one rice leaf or consult an agricultural extension officer."
        ),
    })


def _own_record(diagnosis_id):
    row = get_db().execute(
        "SELECT * FROM diagnoses WHERE id = ? AND client_id = ?",
        (diagnosis_id, client_id())).fetchone()
    if row is None:
        abort(404)
    return row


@bp.get("/result/<int:diagnosis_id>")
def result(diagnosis_id):
    row = _own_record(diagnosis_id)
    advice_data = current_app.config.get("ADVICE", {})
    advice = advice_data.get(row["prediction"]) if row["status"] == "confident" else None
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
    return render_template("result.html", row=row,
                           top_k=json.loads(row["top_k"]), advice=advice,
                           threshold=threshold)


@bp.get("/photo/<filename>")
def photo(filename):
    row = get_db().execute(
        "SELECT id FROM diagnoses WHERE image_file = ? AND client_id = ?",
        (filename, client_id())).fetchone()
    if row is None:
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_DIR"], filename)


@bp.get("/history")
def history():
    db, cid = get_db(), client_id()
    rows = db.execute(
        "SELECT * FROM diagnoses WHERE client_id = ? ORDER BY id DESC LIMIT 200",
        (cid,)).fetchall()
    counts = db.execute(
        "SELECT CASE WHEN status = 'confident' THEN prediction ELSE 'unsure' END AS label, "
        "COUNT(*) AS n FROM diagnoses WHERE client_id = ? GROUP BY label ORDER BY n DESC",
        (cid,)).fetchall()
    return render_template("history.html", rows=rows, counts=counts)


@bp.post("/delete/<int:diagnosis_id>")
def delete(diagnosis_id):
    row = _own_record(diagnosis_id)
    with contextlib.suppress(OSError):
        (current_app.config["UPLOAD_DIR"] / row["image_file"]).unlink()
    db = get_db()
    db.execute("DELETE FROM diagnoses WHERE id = ?", (diagnosis_id,))
    db.commit()
    return redirect(url_for("main.history"))


@bp.errorhandler(413)
def too_large(_e):
    max_mb = current_app.config.get("MAX_UPLOAD_MB", 8)
    return render_template(
        "index.html",
        error=f"That photo is too large. Please use one under {max_mb} MB."), 413


@bp.errorhandler(404)
def not_found(_e):
    return render_template("message.html", title="Not found",
                           text="That page or record does not exist."), 404
