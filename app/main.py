"""Step 6 - API (FastAPI) + web UI.

Run locally:   uvicorn app.main:app --port 8000
Swagger docs:  http://localhost:8000/docs
"""
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import database as db
from src import retrain as rt
from src.config import AUTO_RETRAIN, CLASS_DISPLAY, CLASSES, INSIGHTS_PATH, MODEL_PATH, RETRAIN_THRESHOLD, TEST_DIR
from src.model import evaluate
from src.prediction import Predictor
from src.preprocessing import IMAGE_EXTS, InvalidImage, extract_zip, load_directory, normalise_label, preprocess_bytes, sha256

STATIC = Path(__file__).parent / "static"
STARTED = time.time()
STATS = {"requests": 0, "predictions": 0, "errors": 0, "latency_sum_ms": 0.0}

app = FastAPI(title="Potato Leaf Disease Classifier", version="1.0",
              description="Predict potato leaf disease from a photo, upload new labelled data and retrain the model.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")

def _seed_model_dir():
    """If MODELS_DIR points at an empty persistent disk, copy the bundled model there on first boot."""
    bundled = Path(__file__).resolve().parent.parent / "models"
    if not MODEL_PATH.exists() and (bundled / MODEL_PATH.name).exists() and bundled != MODEL_PATH.parent:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        for f in ("potato_cnn.h5", "metadata.json", "insights.json"):
            shutil.copy2(bundled / f, MODEL_PATH.parent / f)


_seed_model_dir()
db.init_db()
db.fail_stale_runs()
predictor = Predictor()


@app.middleware("http")
async def count_requests(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        STATS["errors"] += 1
        raise
    STATS["requests"] += 1
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    response.headers["X-Worker-PID"] = str(os.getpid())
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# ---------------------------------------------------------------- monitoring
@app.get("/health", tags=["monitoring"])
def health():
    """Model up-time and serving status."""
    predictor.reload_if_changed()
    meta = predictor.metadata
    up = time.time() - STARTED
    n = STATS["predictions"]
    return {
        "status": "ok" if predictor.model is not None else "no-model",
        "uptime_seconds": round(up, 1),
        "started_at": datetime.fromtimestamp(STARTED, timezone.utc).isoformat(timespec="seconds"),
        "model_version": meta.get("version"),
        "model_saved_at": meta.get("saved_at"),
        "model_test_accuracy": meta.get("test_metrics", {}).get("accuracy"),
        "worker_pid": os.getpid(),
        "requests_served": STATS["requests"],
        "predictions_served": n,
        "avg_prediction_latency_ms": round(STATS["latency_sum_ms"] / n, 2) if n else None,
        "retraining": db.running_run() is not None,
    }


@app.get("/model", tags=["monitoring"])
def model_info():
    """Metadata of the model in production: version, test metrics, version history."""
    predictor.reload_if_changed()
    return predictor.metadata


@app.get("/monitoring", tags=["monitoring"])
def monitoring():
    """Live production statistics from logged predictions (confidence, latency, class mix)."""
    return db.prediction_stats()


@app.post("/evaluate", tags=["monitoring"])
async def evaluate_in_production(file: Optional[UploadFile] = File(None, description="optional zip of <class>/<image> to evaluate on")):
    """Evaluate the model that is currently serving.

    No file -> held-out test set bundled with the deployment.
    Zip file -> labelled images you provide (folders named after the classes)."""
    if file is not None and file.filename:
        X, y = [], []
        for _, data, label in extract_zip(await file.read()):
            if label:
                try:
                    X.append(preprocess_bytes(data)); y.append(CLASSES.index(label))
                except InvalidImage:
                    pass
        if not X:
            raise HTTPException(400, "zip contained no labelled images (use folders named early_blight / healthy / late_blight)")
        X, y, source = np.stack(X), np.array(y), f"uploaded zip ({len(y)} images)"
    else:
        X, y, _ = load_directory(TEST_DIR)
        source = f"held-out test set ({len(y)} images)"
    t0 = time.perf_counter()
    m = evaluate(predictor.model, X, y)
    return {"source": source, "model_version": predictor.version, "seconds": round(time.perf_counter() - t0, 2),
            "classes": [CLASS_DISPLAY[c] for c in CLASSES],
            **{k: m[k] for k in ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                                 "roc_auc_ovr", "confusion_matrix", "n_samples")}}


@app.get("/insights", tags=["data"])
def insights():
    """Pre-computed dataset feature statistics for the visualisation page."""
    if not INSIGHTS_PATH.exists():
        raise HTTPException(404, "insights not generated - run the notebook")
    return JSONResponse(json.loads(INSIGHTS_PATH.read_text()))


# ---------------------------------------------------------------- prediction
@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(..., description="photo of a potato leaf (jpg/png)")):
    """Predict the class of ONE leaf image."""
    data = await file.read()
    t0 = time.perf_counter()
    try:
        arr = preprocess_bytes(data)
    except InvalidImage as e:
        raise HTTPException(400, str(e))
    result = predictor.predict_array(arr)
    ms = (time.perf_counter() - t0) * 1000
    STATS["predictions"] += 1
    STATS["latency_sum_ms"] += ms
    try:
        db.log_prediction(result["label"], result["confidence"], ms, result["model_version"])
    except Exception:
        pass   # monitoring must never break serving
    return {**result, "filename": file.filename, "latency_ms": round(ms, 2)}


# ---------------------------------------------------------------- data upload + retraining
@app.post("/upload", tags=["retraining"])
async def upload(files: List[UploadFile] = File(..., description="images and/or zip files"),
                 label: Optional[str] = Form(None, description="class for loose images: early_blight | healthy | late_blight")):
    """Bulk-upload labelled images for retraining. They are validated, then saved into the database.

    * Loose images need the `label` field.
    * Zip files are labelled by their folder names (e.g. `late_blight/img1.jpg`)."""
    form_label = normalise_label(label) if label else None
    if label and not form_label:
        raise HTTPException(400, f"unknown label '{label}'. Use one of {CLASSES}")
    accepted, duplicates, rejected = 0, 0, []

    def store(name, data, lab):
        nonlocal accepted, duplicates
        if not lab:
            rejected.append({"file": name, "reason": "no label (choose a class or use class-named folders in the zip)"}); return
        try:
            preprocess_bytes(data)            # validation = same preprocessing as training
        except InvalidImage as e:
            rejected.append({"file": name, "reason": str(e)}); return
        if db.add_upload(name, lab, sha256(data), data):
            accepted += 1
        else:
            duplicates += 1

    for f in files:
        data = await f.read()
        if f.filename.lower().endswith(".zip"):
            try:
                for name, img, zlabel in extract_zip(data):
                    store(name, img, zlabel or form_label)
            except InvalidImage as e:
                rejected.append({"file": f.filename, "reason": str(e)})
        elif Path(f.filename).suffix.lower() in IMAGE_EXTS:
            store(f.filename, data, form_label)
        else:
            rejected.append({"file": f.filename, "reason": "unsupported file type"})

    triggered = rt.maybe_auto_trigger() if accepted else False
    counts = db.upload_counts()
    return {"accepted": accepted, "duplicates": duplicates, "rejected": rejected[:50], "n_rejected": len(rejected),
            "pending_total": counts["pending_total"], "pending_by_class": counts["pending"],
            "auto_retrain_threshold": RETRAIN_THRESHOLD if AUTO_RETRAIN else None,
            "auto_retrain_triggered": triggered}


@app.get("/uploads", tags=["retraining"])
def uploads():
    """How much uploaded data is waiting for / has been used in retraining."""
    return {**db.upload_counts(), "auto_retrain_threshold": RETRAIN_THRESHOLD if AUTO_RETRAIN else None}


@app.post("/retrain", tags=["retraining"])
def retrain():
    """Trigger retraining on the uploaded data (runs in the background)."""
    if db.upload_counts()["pending_total"] == 0:
        raise HTTPException(400, "no new uploaded data - upload images first")
    if db.running_run() is not None or not rt.retrain_async("manual"):
        raise HTTPException(409, "a retraining run is already in progress")
    return {"started": True, "run_id": rt.STATUS.get("run_id")}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status():
    runs = db.list_runs(10)
    return {"running": db.running_run() is not None, "live": rt.STATUS, "runs": runs}
