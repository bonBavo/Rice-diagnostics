"""
Rice Leaf Doctor — Flask application factory.

Usage:
    from app import create_app
    app = create_app()
    app.run()
"""
import contextlib
import json
import os
import secrets
import sqlite3
from datetime import timedelta
from pathlib import Path

from flask import Flask, g

BASE_DIR = Path(__file__).resolve().parent.parent  # project root


def create_app(test_config: dict | None = None) -> Flask:
    """Application factory (see Flask docs on the factory pattern)."""

    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

    # ---- configuration ------------------------------------------------
    model_dir = Path(os.environ.get("MODEL_DIR", BASE_DIR / "model"))
    upload_dir = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
    db_path = str(Path(os.environ.get("DB_PATH", BASE_DIR / "history.db")))
    max_upload_mb = 8

    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=max_upload_mb * 1024 * 1024,
        MAX_UPLOAD_MB=max_upload_mb,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=365),
        MODEL_DIR=model_dir,
        UPLOAD_DIR=upload_dir,
        DB_PATH=db_path,
        CONFIDENCE_THRESHOLD=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70")),
    )

    if test_config:
        app.config.update(test_config)
    model_dir = Path(app.config["MODEL_DIR"])
    upload_dir = Path(app.config["UPLOAD_DIR"])
    db_path = str(Path(app.config["DB_PATH"]))

    # ---- advice data --------------------------------------------------
    advice_path = BASE_DIR / "app" / "advice.json"
    if not advice_path.exists():
        advice_path = BASE_DIR / "rice_web" / "advice.json"
    if advice_path.exists():
        app.config["ADVICE"] = json.loads(advice_path.read_text(encoding="utf-8"))
    else:
        app.config["ADVICE"] = {}

    # ---- model --------------------------------------------------------
    try:
        from app.services.predictor import Predictor
        predictor = Predictor(model_dir)
        app.config["PREDICTOR"] = predictor
        app.config["MODEL_ERROR"] = None
    except Exception as exc:
        print(f"Could not load model from {model_dir}: {exc}")
        app.config["PREDICTOR"] = None
        app.config["MODEL_ERROR"] = (
            f"The diagnosis model was not found in '{model_dir}'. "
            "Train it first with rice_cnn.py, then restart the app."
        )

    # ---- database -----------------------------------------------------
    upload_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(db_path)) as db:
        db.executescript("""
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
        """)
        db.commit()

    # ---- teardown -----------------------------------------------------
    @app.teardown_appcontext
    def close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ---- blueprints ---------------------------------------------------
    from app.routes import bp
    app.register_blueprint(bp)

    return app
