"""Prediction functions used by the notebook and the API.

`Predictor` keeps the model in memory and hot-reloads it whenever the model
file changes on disk (e.g. after a retraining run promoted a new version),
so every API worker / container picks up the new model without a restart.
"""
import threading
import time

import numpy as np

from src.config import CLASS_DISPLAY, CLASSES, METADATA_PATH, MODEL_PATH
from src.model import load_metadata, load_model
from src.preprocessing import preprocess_bytes, preprocess_path


class Predictor:
    def __init__(self, model_path=MODEL_PATH, meta_path=METADATA_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self._lock = threading.Lock()
        self._mtime = None
        self.model = None
        self.metadata = {}
        self._last_check = 0.0
        self.reload_if_changed(force=True)

    def reload_if_changed(self, force=False):
        now = time.time()
        if not force and now - self._last_check < 2:       # stat() at most every 2 s
            return False
        self._last_check = now
        try:
            mtime = self.model_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if force or mtime != self._mtime:
            with self._lock:
                model = load_model(self.model_path)
                model.predict(np.zeros((1, *model.input_shape[1:]), np.uint8), verbose=0)  # warm-up
                self.model, self._mtime = model, mtime
                self.metadata = load_metadata(self.meta_path)
            return True
        return False

    @property
    def version(self):
        return self.metadata.get("version")

    def predict_array(self, arr: np.ndarray) -> dict:
        self.reload_if_changed()
        x = arr[None, ...] if arr.ndim == 3 else arr
        proba = self.model(x, training=False).numpy()[0]
        idx = int(proba.argmax())
        return {
            "label": CLASSES[idx],
            "display_label": CLASS_DISPLAY[CLASSES[idx]],
            "confidence": float(proba[idx]),
            "probabilities": {CLASS_DISPLAY[c]: float(p) for c, p in zip(CLASSES, proba)},
            "model_version": self.version,
        }

    def predict_bytes(self, data: bytes) -> dict:
        return self.predict_array(preprocess_bytes(data))

    def predict_path(self, path) -> dict:
        return self.predict_array(preprocess_path(path))


def predict_image(path) -> dict:
    """Convenience one-shot helper (used in the notebook)."""
    return Predictor().predict_path(path)
