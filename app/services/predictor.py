"""
Prediction service — loads the trained CNN once and runs inference.

This is the single source of truth for class names and image preprocessing
during inference. The training script (rice_cnn.py) writes class_names.json
into the model folder; this module reads it back so the two always agree.
"""
import json
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Default model directory (relative to project root)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"

# File names produced by rice_cnn.py
MODEL_FILE = "rice_disease_model.keras"
META_FILE = "class_names.json"


class Predictor:
    """Wraps a trained Keras model for thread-safe single-image inference."""

    def __init__(self, model_dir: Path | str | None = None):
        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        meta = json.loads((model_dir / META_FILE).read_text())
        self.class_names: list[str] = meta["class_names"]
        self.img_size: int = meta["img_size"]
        self.model = tf.keras.models.load_model(model_dir / MODEL_FILE)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, image_path: str | Path, top_k: int = 3,
                threshold: float = 0.70) -> dict:
        """Classify a single leaf image.

        Returns a dict with keys:
            prediction  – predicted class name
            confidence  – float 0-1
            status      – 'confident' or 'uncertain'
            top_k       – list of {label, confidence} dicts
        """
        img = self._preprocess(str(image_path))
        with self._lock:
            probs = self.model.predict(tf.expand_dims(img, 0), verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [
            {"label": self.class_names[i], "confidence": float(probs[i])}
            for i in top_indices
        ]
        return {
            "prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "status": "confident" if results[0]["confidence"] >= threshold else "uncertain",
            "top_k": results,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _preprocess(self, path: str) -> tf.Tensor:
        """Load and resize an image exactly as the training pipeline does.

        The model contains a built-in Rescaling layer, so we feed raw
        0-255 uint8 pixels here.
        """
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.img_size, self.img_size])
        return img
