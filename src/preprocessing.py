"""Step 2 - Data processing.

The same functions are used by the notebook, the training script, the API's
/predict endpoint and the retraining job, so training and serving can never
drift apart (no train/serve skew).

Pipeline for one image:
    raw bytes -> decode (PIL) -> fix EXIF rotation -> force RGB (drops alpha / grayscale)
              -> resize to IMG_SIZE x IMG_SIZE (bilinear) -> uint8 array (H, W, 3)
Pixel scaling to [0, 1] and data augmentation live INSIDE the Keras model
(Rescaling + Random* layers), so every caller just passes uint8 arrays.
"""
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import CLASSES, IMG_SIZE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InvalidImage(ValueError):
    pass


def preprocess_bytes(data: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """Decode raw image bytes into a (img_size, img_size, 3) uint8 array."""
    if not data:
        raise InvalidImage("empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage("file larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImage(f"not a readable image: {e}") from e
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess_path(path, img_size: int = IMG_SIZE) -> np.ndarray:
    return preprocess_bytes(Path(path).read_bytes(), img_size)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_images(root: Path):
    """Yield (path, label) for every image under root/<class>/."""
    for label in CLASSES:
        d = Path(root) / label
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                yield p, label


def load_directory(root: Path, img_size: int = IMG_SIZE, limit_per_class: int | None = None, seed: int = 42):
    """Load root/<class>/*.jpg into X (N,H,W,3 uint8), y (N,) int labels, plus file paths."""
    rng = np.random.default_rng(seed)
    X, y, paths = [], [], []
    for label in CLASSES:
        items = [p for p, lab in list_images(root) if lab == label]
        if limit_per_class is not None and len(items) > limit_per_class:
            items = list(rng.choice(items, size=limit_per_class, replace=False))
        for p in items:
            try:
                X.append(preprocess_path(p, img_size))
            except InvalidImage:
                continue
            y.append(CLASSES.index(label))
            paths.append(str(p))
    if not X:
        return np.zeros((0, img_size, img_size, 3), np.uint8), np.zeros((0,), np.int64), []
    return np.stack(X), np.array(y, dtype=np.int64), paths


def normalise_label(name: str) -> str | None:
    """Map folder names like 'Late Blight', 'Potato___Late_blight', 'late-blight' to a class id."""
    key = name.strip().lower().replace("potato___", "").replace("-", "_").replace(" ", "_")
    return key if key in CLASSES else None


def extract_zip(data: bytes):
    """Yield (filename, bytes, label) from a zip laid out as <class>/<image>.

    The label is taken from the closest parent folder whose name is a known class,
    so 'batch/late_blight/img1.jpg' and 'Late Blight/img1.jpg' both work.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise InvalidImage("not a valid zip archive") from e
    for info in zf.infolist():
        if info.is_dir() or "__MACOSX" in info.filename or Path(info.filename).name.startswith("."):
            continue
        p = Path(info.filename)
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = None
        for part in reversed(p.parts[:-1]):
            label = normalise_label(part)
            if label:
                break
        yield p.name, zf.read(info), label
