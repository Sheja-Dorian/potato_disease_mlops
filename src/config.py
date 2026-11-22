"""Central configuration shared by the notebook, the API and the retraining job."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
INCOMING_DIR = DATA_DIR / "incoming"          # held-out "new field data" for the retraining demo

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "potato_cnn.h5"
METADATA_PATH = MODELS_DIR / "metadata.json"
INSIGHTS_PATH = MODELS_DIR / "insights.json"
ARCHIVE_DIR = MODELS_DIR / "archive"

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "app.db"))

# Class folder names (sorted -> label index) and human-friendly names
CLASSES = ["early_blight", "healthy", "late_blight"]
CLASS_DISPLAY = {
    "early_blight": "Early Blight",
    "healthy": "Healthy",
    "late_blight": "Late Blight",
}

IMG_SIZE = 128
SEED = 42

# Retraining
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "100"))   # pending uploads that auto-trigger retraining
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
REPLAY_PER_CLASS = int(os.getenv("REPLAY_PER_CLASS", "120"))     # old images mixed in to avoid forgetting
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "8"))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", "1e-4"))
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", "0.01"))  # new model may be at most 1pt worse
