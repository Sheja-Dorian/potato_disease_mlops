"""SQLite storage for uploaded training data, retraining runs and production predictions.

Uploaded images are stored IN the database (BLOB column) together with their label,
so the retraining job reads its new data straight from the database.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    label       TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    image       BLOB NOT NULL,
    uploaded_at TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | used
    run_id      INTEGER
);
CREATE TABLE IF NOT EXISTS training_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,                     -- running | completed | failed
    trigger       TEXT NOT NULL,                     -- manual | auto-threshold | cli
    n_new         INTEGER, n_replay INTEGER, epochs_run INTEGER,
    old_metrics   TEXT, new_metrics TEXT,
    promoted      INTEGER, model_version INTEGER, message TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    predicted     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    model_version INTEGER
);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path=None):
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path=None):
    with connect(path) as c:
        c.executescript(SCHEMA)


# ---------- uploads ----------
def add_upload(filename, label, sha, data) -> bool:
    """Returns False if the exact same image was already uploaded."""
    with connect() as c:
        try:
            c.execute("INSERT INTO uploads(filename,label,sha256,image,uploaded_at) VALUES (?,?,?,?,?)",
                      (filename, label, sha, sqlite3.Binary(data), now()))
            return True
        except sqlite3.IntegrityError:
            return False


def pending_uploads():
    with connect() as c:
        return c.execute("SELECT id, filename, label, image FROM uploads WHERE status='pending' ORDER BY id").fetchall()


def mark_used(ids, run_id):
    with connect() as c:
        c.executemany("UPDATE uploads SET status='used', run_id=? WHERE id=?", [(run_id, i) for i in ids])


def upload_counts():
    with connect() as c:
        rows = c.execute("SELECT status, label, COUNT(*) n FROM uploads GROUP BY status, label").fetchall()
    out = {"pending": {}, "used": {}}
    for r in rows:
        out.setdefault(r["status"], {})[r["label"]] = r["n"]
    out["pending_total"] = sum(out["pending"].values())
    out["used_total"] = sum(out["used"].values())
    return out


# ---------- training runs ----------
def start_run(trigger) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO training_runs(started_at,status,trigger) VALUES (?,?,?)", (now(), "running", trigger))
        return cur.lastrowid


def finish_run(run_id, status, **fields):
    fields = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in fields.items()}
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE training_runs SET status=?, finished_at=?{', ' + cols if cols else ''} WHERE id=?",
                  (status, now(), *fields.values(), run_id))


def list_runs(limit=20):
    with connect() as c:
        rows = c.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("old_metrics", "new_metrics"):
            if d.get(k):
                d[k] = json.loads(d[k])
        out.append(d)
    return out


def running_run():
    with connect() as c:
        r = c.execute("SELECT id FROM training_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
    return r["id"] if r else None


def fail_stale_runs():
    """A run left 'running' by a crashed/restarted process can never finish - mark it failed on startup."""
    with connect() as c:
        c.execute("UPDATE training_runs SET status='failed', finished_at=?, message='interrupted by restart' "
                  "WHERE status='running'", (now(),))


# ---------- production monitoring ----------
def log_prediction(predicted, confidence, latency_ms, model_version):
    with connect() as c:
        c.execute("INSERT INTO predictions(ts,predicted,confidence,latency_ms,model_version) VALUES (?,?,?,?,?)",
                  (now(), predicted, confidence, latency_ms, model_version))


def prediction_stats(limit=500):
    with connect() as c:
        rows = c.execute("SELECT predicted, confidence, latency_ms FROM predictions ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
        total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if not rows:
        return {"total": total, "recent": 0}
    conf = [r["confidence"] for r in rows]
    lat = sorted(r["latency_ms"] for r in rows)
    by_class = {}
    for r in rows:
        by_class[r["predicted"]] = by_class.get(r["predicted"], 0) + 1
    return {
        "total": total,
        "recent": len(rows),
        "mean_confidence": sum(conf) / len(conf),
        "low_confidence_share": sum(c < 0.7 for c in conf) / len(conf),
        "latency_p50_ms": lat[len(lat) // 2],
        "latency_p95_ms": lat[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0],
        "by_class": by_class,
    }
