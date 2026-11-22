"""Step 5 - Model retraining.

Flow (each step is visible in the UI and recorded in the `training_runs` table):
  1. Read new labelled images that users uploaded -> stored as BLOBs in the database
  2. Preprocess them with the SAME preprocessing as training (src.preprocessing)
  3. Mix in a replay sample of the original training data (prevents catastrophic forgetting)
  4. Load the CURRENT production model as the pre-trained starting point and fine-tune it
     with a low learning rate, early stopping and class weights
  5. Evaluate old vs new model on the untouched test set
  6. Promote the new model only if it is not worse (tolerance configurable); the old one is archived

Triggers:
  * manual  - "Retrain" button in the UI  (POST /retrain)
  * auto    - fires when pending uploads >= RETRAIN_THRESHOLD (checked after every upload)
  * cli     - python -m src.retrain [--load-dir data/incoming]   (e.g. from a cron job)
"""
import argparse
import shutil
import threading
import time
import traceback

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src import database as db
from src.config import (ARCHIVE_DIR, AUTO_RETRAIN, CLASSES, MODEL_PATH, PROMOTION_TOLERANCE, REPLAY_PER_CLASS,
                        RETRAIN_EPOCHS, RETRAIN_LR, RETRAIN_THRESHOLD, SEED, TEST_DIR, TRAIN_DIR)
from src.model import class_weights, compile_model, evaluate, load_metadata, load_model, make_dataset, save_model
from src.preprocessing import InvalidImage, list_images, load_directory, preprocess_bytes, sha256

_lock = threading.Lock()
STATUS = {"state": "idle", "run_id": None, "stage": None, "epoch": None, "epochs": None, "log": []}


def _set(stage, **kw):
    STATUS.update(stage=stage, **kw)
    STATUS["log"] = (STATUS["log"] + [f"{time.strftime('%H:%M:%S')}  {stage}"])[-30:]


class _Progress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        STATUS["epoch"] = epoch + 1
        _set(f"epoch {epoch + 1}: loss={logs.get('loss', 0):.3f} acc={logs.get('accuracy', 0):.3f} "
             f"val_loss={logs.get('val_loss', 0):.3f} val_acc={logs.get('val_accuracy', 0):.3f}")


def retrain(trigger="manual") -> dict:
    if not _lock.acquire(blocking=False):
        raise RuntimeError("a retraining run is already in progress")
    run_id = None
    try:
        STATUS.update(state="running", log=[], epoch=0, epochs=RETRAIN_EPOCHS)
        run_id = db.start_run(trigger)
        STATUS["run_id"] = run_id

        # 1-2. new data from the database -> preprocess
        _set("loading uploaded images from database")
        rows = db.pending_uploads()
        if not rows:
            raise ValueError("no new uploaded data to retrain on - upload images first")
        X_new, y_new, ids = [], [], []
        for r in rows:
            try:
                X_new.append(preprocess_bytes(bytes(r["image"])))
                y_new.append(CLASSES.index(r["label"]))
                ids.append(r["id"])
            except (InvalidImage, ValueError):
                continue
        X_new, y_new = np.stack(X_new), np.array(y_new)
        _set(f"preprocessed {len(X_new)} new images (resize {X_new.shape[1]}px, RGB)")

        # 3. replay buffer from original training data
        X_old, y_old, _ = load_directory(TRAIN_DIR, limit_per_class=REPLAY_PER_CLASS, seed=int(time.time()) % 10_000)
        X = np.concatenate([X_new, X_old])
        y = np.concatenate([y_new, y_old])
        _set(f"combined with {len(X_old)} replay images -> {len(X)} total")
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)

        # 4. fine-tune the current production model (custom pre-trained model)
        meta = load_metadata()
        old_version = meta.get("version", 1)
        _set(f"loading current model v{old_version} as pre-trained starting point")
        old_model = load_model()
        model = load_model()
        compile_model(model, lr=RETRAIN_LR)
        hist = model.fit(
            make_dataset(X_tr, y_tr, 32), validation_data=make_dataset(X_val, y_val, 32, shuffle=False),
            epochs=RETRAIN_EPOCHS, class_weight=class_weights(y_tr), verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                       _Progress()],
        )

        # 5. evaluate old vs new on the untouched test set
        _set("evaluating old vs new model on held-out test set")
        X_test, y_test, _ = load_directory(TEST_DIR)
        old_m, new_m = evaluate(old_model, X_test, y_test), evaluate(model, X_test, y_test)

        # 6. promote?
        promoted = new_m["accuracy"] >= old_m["accuracy"] - PROMOTION_TOLERANCE
        new_version = old_version
        if promoted:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MODEL_PATH, ARCHIVE_DIR / f"potato_cnn_v{old_version}.h5")
            new_version = old_version + 1
            history = meta.get("history", []) + [{"version": new_version, "run_id": run_id,
                                                  "accuracy": new_m["accuracy"], "f1_macro": new_m["f1_macro"],
                                                  "trained_on_new": len(X_new)}]
            save_model(model, new_m, new_version, extra={"history": history, "trigger": trigger})
            db.mark_used(ids, run_id)
            msg = (f"promoted v{new_version}: accuracy {old_m['accuracy']:.3f} -> {new_m['accuracy']:.3f}, "
                   f"F1 {old_m['f1_macro']:.3f} -> {new_m['f1_macro']:.3f}")
        else:
            msg = (f"kept v{old_version}: new model accuracy {new_m['accuracy']:.3f} worse than "
                   f"{old_m['accuracy']:.3f} - uploads stay pending")
        _set(msg)
        slim = lambda m: {k: m[k] for k in ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro",
                                            "roc_auc_ovr", "confusion_matrix")}
        db.finish_run(run_id, "completed", n_new=len(X_new), n_replay=len(X_old),
                      epochs_run=len(hist.history["loss"]), old_metrics=slim(old_m), new_metrics=slim(new_m),
                      promoted=int(promoted), model_version=new_version, message=msg)
        STATUS.update(state="idle")
        return {"run_id": run_id, "promoted": promoted, "model_version": new_version, "message": msg,
                "old_metrics": slim(old_m), "new_metrics": slim(new_m)}
    except Exception as e:
        _set(f"failed: {e}")
        STATUS.update(state="failed")
        if run_id:
            db.finish_run(run_id, "failed", message=str(e))
        traceback.print_exc()
        raise
    finally:
        _lock.release()


def retrain_async(trigger="manual") -> bool:
    """Start retraining in a background thread. Returns False if one is already running."""
    if _lock.locked():
        return False

    def _run():
        try:
            retrain(trigger)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="retrain").start()
    time.sleep(0.2)
    return True


def maybe_auto_trigger() -> bool:
    """The automatic trigger: enough new labelled data has accumulated."""
    if not AUTO_RETRAIN:
        return False
    if db.upload_counts()["pending_total"] >= RETRAIN_THRESHOLD and not _lock.locked():
        return retrain_async("auto-threshold")
    return False


def load_dir_into_db(root) -> int:
    n = 0
    for p, label in list_images(root):
        data = p.read_bytes()
        n += db.add_upload(p.name, label, sha256(data), data)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrain the potato leaf model on uploaded data")
    ap.add_argument("--load-dir", help="first insert images from DIR/<class>/ into the database (e.g. data/incoming)")
    args = ap.parse_args()
    db.init_db()
    if args.load_dir:
        print(f"inserted {load_dir_into_db(args.load_dir)} images into the database")
    print(retrain("cli")["message"])
