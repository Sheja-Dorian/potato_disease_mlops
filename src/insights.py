"""Dataset insights shown in the UI and the notebook.

Hand-crafted, interpretable features computed per image (on the leaf pixels only):
  * green_ratio     - G / (R+G+B): how green the leaf tissue is
  * lesion_fraction - share of leaf pixels whose hue is yellow/brown (dead or diseased tissue)
  * texture         - mean Sobel edge strength: how 'busy' (spotted / ringed) the leaf surface is
  * brightness      - mean HSV value of leaf pixels
Leaf pixels are separated from the grey PlantVillage background by saturation.
"""
import json

import numpy as np
from PIL import Image
from scipy import ndimage

from src.config import CLASS_DISPLAY, CLASSES, INSIGHTS_PATH, TEST_DIR, TRAIN_DIR
from src.preprocessing import list_images

FEATURES = ["green_ratio", "lesion_fraction", "texture", "brightness"]


def image_features(arr: np.ndarray) -> dict:
    rgb = arr.astype(np.float32)
    hsv = np.asarray(Image.fromarray(arr).convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = (s > 50) & (v > 30)
    if leaf.sum() < 50:
        leaf = np.ones_like(leaf)
    R, G, B = rgb[..., 0][leaf], rgb[..., 1][leaf], rgb[..., 2][leaf]
    green_ratio = float(np.mean(G / (R + G + B + 1e-6)))
    hue = h[leaf]                                   # PIL hue 0-255; green ~ 60-110, yellow ~ 40, brown/orange < 35
    lesion_fraction = float(np.mean(hue < 45))
    gray = rgb.mean(-1)
    edges = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    texture = float(edges[leaf].mean() / 255.0)
    brightness = float(v[leaf].mean() / 255.0)
    return {"green_ratio": green_ratio, "lesion_fraction": lesion_fraction, "texture": texture,
            "brightness": brightness}


def feature_table(root=TRAIN_DIR, size=128):
    import pandas as pd
    rows = []
    for p, label in list_images(root):
        arr = np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.uint8)
        rows.append({"file": p.name, "label": label, **image_features(arr)})
    return pd.DataFrame(rows)


def build_insights(df=None) -> dict:
    if df is None:
        df = feature_table()
    out = {"classes": [CLASS_DISPLAY[c] for c in CLASSES], "class_counts": {}, "features": {}}
    for split, root in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        out["class_counts"][split] = {CLASS_DISPLAY[c]: sum(1 for _, l in list_images(root) if l == c) for c in CLASSES}
    for f in FEATURES:
        lo, hi = float(df[f].quantile(0.005)), float(df[f].quantile(0.995))
        bins = np.linspace(lo, hi, 26)
        out["features"][f] = {
            "bins": [round(float(b), 4) for b in bins],
            "hist": {CLASS_DISPLAY[c]: np.histogram(df[df.label == c][f].clip(lo, hi), bins=bins)[0].tolist()
                     for c in CLASSES},
            "mean": {CLASS_DISPLAY[c]: round(float(df[df.label == c][f].mean()), 4) for c in CLASSES},
            "median": {CLASS_DISPLAY[c]: round(float(df[df.label == c][f].median()), 4) for c in CLASSES},
        }
    return out


def save_insights(interpretations: dict | None = None, df=None):
    ins = build_insights(df)
    if interpretations:
        ins["interpretations"] = interpretations
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSIGHTS_PATH.write_text(json.dumps(ins, indent=2))
    return ins


if __name__ == "__main__":
    df = feature_table()
    print(df.groupby("label")[FEATURES].describe().T.round(3).to_string())
