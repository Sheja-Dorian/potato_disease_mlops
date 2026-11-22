"""Step 3 & 4 - Model creation and model testing.

`build_model()` is the optimised CNN used in production:
  * data augmentation layers (flip / rotate / zoom / contrast) - regularisation
  * BatchNormalization (momentum 0.9) after each convolution - stable, faster training
  * L2 weight decay on conv + dense kernels - regularisation
  * Dropout before the classifier - regularisation
  * GlobalAveragePooling instead of Flatten - far fewer parameters, less overfitting
  * Adam optimiser, EarlyStopping(restore_best_weights) and ReduceLROnPlateau
  * class weights to counter the small 'healthy' class
"""
import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.config import CLASSES, IMG_SIZE, METADATA_PATH, MODEL_PATH, SEED


def _augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def build_model(img_size: int = IMG_SIZE, n_classes: int = len(CLASSES), l2: float = 1e-4,
                dropout: float = 0.4, lr: float = 1e-3) -> keras.Model:
    reg = regularizers.l2(l2)
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255, name="rescale")(x)
    for i, filters in enumerate([32, 64, 128, 256]):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg, name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"bn{i+1}")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(name=f"pool{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="dense")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs, outputs, name="potato_leaf_cnn")
    compile_model(model, lr)
    return model


def build_baseline(img_size: int = IMG_SIZE, n_classes: int = len(CLASSES)) -> keras.Model:
    """'Vanilla' CNN for comparison: no augmentation, no regularisation, plain SGD."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def compile_model(model: keras.Model, lr: float = 1e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def class_weights(y: np.ndarray) -> dict:
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    return {int(c): float(v) for c, v in zip(present, w)}


def make_dataset(X, y, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def callbacks(patience=8, log_dir=None):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    if log_dir:
        cbs.append(keras.callbacks.CSVLogger(str(log_dir)))
    return cbs


def train(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=32, patience=8, use_class_weights=True, verbose=1):
    t0 = time.time()
    history = model.fit(
        make_dataset(X_train, y_train, batch_size),
        validation_data=make_dataset(X_val, y_val, batch_size, shuffle=False),
        epochs=epochs,
        class_weight=class_weights(y_train) if use_class_weights else None,
        callbacks=callbacks(patience),
        verbose=verbose,
    )
    return history, time.time() - t0


def predict_proba(model, X, batch_size=64) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=0)


def evaluate(model, X, y) -> dict:
    """All the metrics the rubric asks for, computed on a held-out set."""
    proba = predict_proba(model, X)
    pred = proba.argmax(1)
    labels = list(range(len(CLASSES)))
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "loss": float(log_loss(y, np.clip(proba, 1e-7, 1), labels=labels)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": classification_report(y, pred, labels=labels, target_names=CLASSES,
                                           output_dict=True, zero_division=0),
    }
    try:
        metrics["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
    except ValueError:
        metrics["roc_auc_ovr"] = None
    return metrics


def save_model(model, metrics: dict, version: int, extra: dict | None = None, path=MODEL_PATH, meta_path=METADATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.h5")
    # Keras 3 cannot re-save a model that was itself loaded from .h5 ("cannot pickle 'module'"),
    # so we always copy the weights into a freshly built network and save that one.
    fresh = build_model()
    fresh.set_weights(model.get_weights())
    fresh.save(tmp)                      # HDF5 (.h5) as required by the brief
    os.replace(tmp, path)                # atomic swap so a serving process never reads half a file
    meta = {
        "version": version,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "file": path.name,
        "test_metrics": metrics,
    }
    if extra:
        meta.update(extra)
    tmp_meta = meta_path.with_suffix(".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, meta_path)
    return meta


def load_model(path=MODEL_PATH) -> keras.Model:
    return keras.models.load_model(path, compile=False)


def load_metadata(meta_path=METADATA_PATH) -> dict:
    try:
        return json.loads(meta_path.read_text())
    except FileNotFoundError:
        return {}
