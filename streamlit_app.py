"""LeafCheck - Streamlit front-end for Streamlit Community Cloud.

Uses exactly the same pipeline code as the FastAPI service (src/): same preprocessing,
same model file, same SQLite database for uploads, same retraining job.
Run locally:  streamlit run streamlit_app.py
"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from src import database as db
from src import retrain as rt
from src.config import CLASS_DISPLAY, CLASSES, INSIGHTS_PATH, TEST_DIR
from src.model import evaluate
from src.prediction import Predictor
from src.preprocessing import IMAGE_EXTS, InvalidImage, extract_zip, load_directory, preprocess_bytes, sha256

st.set_page_config(page_title="LeafCheck", page_icon="🥔", layout="wide")

COLORS = {"Early Blight": "#d08c2b", "Healthy": "#3a8f4b", "Late Blight": "#6b4f9e"}
ADVICE = {
    "Healthy": "No disease detected. Keep scouting weekly, especially after cool, wet weather.",
    "Early Blight": "Early blight (Alternaria solani). Remove infected lower leaves, avoid overhead irrigation, "
                    "rotate crops and consider a protectant fungicide. Ask your local extension officer.",
    "Late Blight": "Late blight (Phytophthora infestans) can destroy a field within days. Act quickly: remove infected "
                   "plants and apply a recommended fungicide. Ask your local agronomist.",
}


@st.cache_resource
def boot():
    """Runs once per server process: database, model, start time."""
    db.init_db()
    db.fail_stale_runs()
    return {"predictor": Predictor(), "started": time.time()}


@st.cache_data
def load_test_set():
    return load_directory(TEST_DIR)[:2]


def fmt_uptime(s):
    d, h, m = int(s // 86400), int(s % 86400 // 3600), int(s % 3600 // 60)
    return f"{d}d {h}h {m}m" if d else f"{h}h {m}m {int(s % 60)}s" if h else f"{m}m {int(s % 60)}s"


state = boot()
predictor = state["predictor"]
predictor.reload_if_changed()
meta = predictor.metadata
tm = meta.get("test_metrics", {})

st.markdown("## 🍃 LeafCheck: potato leaf disease classifier")
st.caption("Healthy · Early Blight · Late Blight. CNN trained on PlantVillage potato leaves.")

tab_dash, tab_pred, tab_ins, tab_rt = st.tabs(["Dashboard", "Predict", "Data insights", "Upload & retrain"])

# ------------------------------------------------------------------ dashboard
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uptime", fmt_uptime(time.time() - state["started"]),
              help="Time since this server process started")
    c2.metric("Model version", f"v{meta.get('version')}", help=f"saved {meta.get('saved_at')}")
    c3.metric("Test accuracy", f"{tm.get('accuracy', 0):.1%}")
    mon = db.prediction_stats()
    c4.metric("Predictions logged", mon.get("total", 0))
    st.caption(f"Online since {datetime.fromtimestamp(state['started'], timezone.utc):%Y-%m-%d %H:%M UTC}. "
               "Click any tab or refresh the page to update.")

    left, right = st.columns(2)
    with left:
        st.subheader("Model metrics on held-out test set")
        rows = [("Accuracy", tm.get("accuracy")), ("Loss (cross-entropy)", tm.get("loss")),
                ("Precision (macro)", tm.get("precision_macro")), ("Recall (macro)", tm.get("recall_macro")),
                ("F1 (macro)", tm.get("f1_macro")), ("F1 (weighted)", tm.get("f1_weighted")),
                ("ROC-AUC (one-vs-rest)", tm.get("roc_auc_ovr"))]
        st.dataframe(pd.DataFrame([(n, round(v, 4) if v is not None else None) for n, v in rows],
                                  columns=["Metric", f"v{meta.get('version')}"]), hide_index=True, width="stretch")
    with right:
        st.subheader("Version history")
        hist = pd.DataFrame(meta.get("history", []))
        if len(hist):
            st.line_chart(hist.set_index("version")[["accuracy", "f1_macro"]], height=260)

    st.subheader("Evaluate the model in production")
    st.caption("Runs the model that is serving right now on the bundled held-out test set.")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating on the test set…"):
            X, y = load_test_set()
            m = evaluate(predictor.model, X, y)
        names = [CLASS_DISPLAY[c] for c in CLASSES]
        a, b = st.columns([1, 1])
        a.dataframe(pd.DataFrame({"Metric": ["Accuracy", "Loss", "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                             "ROC-AUC"],
                                  "Value": [round(m[k], 4) for k in ("accuracy", "loss", "precision_macro",
                                                                      "recall_macro", "f1_macro", "roc_auc_ovr")]}),
                    hide_index=True, width="stretch")
        b.markdown("**Confusion matrix** (rows = true, columns = predicted)")
        b.dataframe(pd.DataFrame(m["confusion_matrix"], index=names, columns=names), width="stretch")
        st.success(f"Model v{predictor.version} on {m['n_samples']} test images: accuracy {m['accuracy']:.2%}, "
                   f"macro-F1 {m['f1_macro']:.3f}")

# ------------------------------------------------------------------ predict
with tab_pred:
    left, right = st.columns(2)
    with left:
        st.subheader("Upload one leaf photo")
        f = st.file_uploader("Image (jpg / png)", type=[e.strip(".") for e in IMAGE_EXTS], key="pred")
        if f:
            st.image(f, width=320)
    with right:
        st.subheader("Result")
        if f and st.button("Predict", type="primary"):
            t0 = time.perf_counter()
            try:
                res = predictor.predict_bytes(f.getvalue())
            except InvalidImage as e:
                st.error(str(e))
                st.stop()
            ms = (time.perf_counter() - t0) * 1000
            db.log_prediction(res["label"], res["confidence"], ms, res["model_version"])
            lab = res["display_label"]
            st.markdown(f"<h2 style='color:{COLORS[lab]};margin:0'>{lab}</h2>", unsafe_allow_html=True)
            st.caption(f"{res['confidence']:.1%} confidence · model v{res['model_version']} · {ms:.0f} ms")
            for k, v in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]):
                st.progress(v, text=f"{k}: {v:.1%}")
            st.info(ADVICE[lab])
        elif not f:
            st.caption("Upload an image, then press Predict.")

# ------------------------------------------------------------------ insights
with tab_ins:
    ins = json.loads(INSIGHTS_PATH.read_text())
    interp = ins.get("interpretations", {})
    st.caption("Four interpretable features computed on the leaf pixels of every training image.")

    def feature_chart(key):
        F = ins["features"][key]
        mids = [round((a + b) / 2, 3) for a, b in zip(F["bins"][:-1], F["bins"][1:])]
        df = pd.DataFrame({c: np.array(F["hist"][c]) / max(1, sum(F["hist"][c])) for c in ins["classes"]}, index=mids)
        st.area_chart(df, color=[COLORS[c] for c in ins["classes"]], height=230)
        st.caption("Medians: " + " · ".join(f"{c} {F['median'][c]:.3f}" for c in ins["classes"]))

    a, b = st.columns(2)
    with a:
        st.subheader("1 · Class distribution")
        st.bar_chart(pd.DataFrame(ins["class_counts"]), stack=False, height=230)
        st.info(interp.get("class_counts", ""))
    with b:
        st.subheader("2 · Green ratio G/(R+G+B)")
        feature_chart("green_ratio")
        st.info(interp.get("green_ratio", ""))
    a, b = st.columns(2)
    with a:
        st.subheader("3 · Lesion fraction (yellow/brown tissue)")
        feature_chart("lesion_fraction")
        st.info(interp.get("lesion_fraction", ""))
    with b:
        st.subheader("4 · Texture (edge strength)")
        feature_chart("texture")
        st.info(interp.get("texture", ""))
    st.success("**The story:** colour features split healthy from diseased leaves; texture separates the two blights. "
               "No single feature separates all three classes, which is why a CNN that learns colour and texture "
               "together works well.")

# ------------------------------------------------------------------ upload + retrain
with tab_rt:
    left, right = st.columns(2)
    with left:
        st.subheader("1 · Upload new labelled data")
        st.caption("Upload many images or a .zip. In a zip, folder names give the labels "
                   "(early_blight/, healthy/, late_blight/). Loose images use the class you pick. "
                   "Images are validated with the training preprocessing and saved to the database.")
        files = st.file_uploader("Images or zip files", accept_multiple_files=True,
                                 type=[e.strip(".") for e in IMAGE_EXTS] + ["zip"], key="up")
        label = st.selectbox("Label for loose images", ["(from zip folders)"] + CLASSES,
                             format_func=lambda c: CLASS_DISPLAY.get(c, c))
        if st.button("Upload", disabled=not files):
            accepted = dup = 0
            rejected = []
            form_label = None if label.startswith("(") else label
            for f in files:
                items = extract_zip(f.getvalue()) if f.name.lower().endswith(".zip") else [(f.name, f.getvalue(), None)]
                for name, data, zlabel in items:
                    lab = zlabel or form_label
                    if not lab:
                        rejected.append(name); continue
                    try:
                        preprocess_bytes(data)
                    except InvalidImage:
                        rejected.append(name); continue
                    if db.add_upload(name, lab, sha256(data), data):
                        accepted += 1
                    else:
                        dup += 1
            st.success(f"{accepted} images saved to the database"
                       + (f" · {dup} duplicates skipped" if dup else "")
                       + (f" · {len(rejected)} rejected" if rejected else ""))

    with right:
        st.subheader("2 · Trigger retraining")
        counts = db.upload_counts()
        st.markdown(f"**{counts['pending_total']} images waiting** · "
                    + " · ".join(f"{CLASS_DISPLAY[k]}: {v}" for k, v in counts["pending"].items()))
        st.caption("Fine-tunes the current production model (as the pre-trained starting point) on the new data "
                   "plus a replay sample of the original data, compares old vs new on the test set, and promotes "
                   "the new model only if it is not worse.")
        if st.button("Retrain model", type="primary", disabled=counts["pending_total"] == 0):
            with st.status("Retraining…", expanded=True) as status:
                log_box = st.empty()
                import threading
                result = {}

                def run():
                    try:
                        result["r"] = rt.retrain("manual")
                    except Exception as e:
                        result["e"] = e

                th = threading.Thread(target=run)
                th.start()
                while th.is_alive():
                    log_box.code("\n".join(rt.STATUS["log"]) or "starting…")
                    time.sleep(1)
                log_box.code("\n".join(rt.STATUS["log"]))
                if "e" in result:
                    status.update(label=f"Retraining failed: {result['e']}", state="error")
                else:
                    r = result["r"]
                    status.update(label=r["message"], state="complete")
                    predictor.reload_if_changed(force=True)

    st.subheader("Retraining runs")
    runs = db.list_runs(10)
    if runs:
        st.dataframe(pd.DataFrame([{
            "#": r["id"], "started": r["started_at"], "trigger": r["trigger"], "status": r["status"],
            "new / replay images": f"{r.get('n_new') or '-'} / {r.get('n_replay') or '-'}",
            "accuracy old → new": (f"{r['old_metrics']['accuracy']:.2%} → {r['new_metrics']['accuracy']:.2%}"
                                   if r.get("old_metrics") else "-"),
            "F1 old → new": (f"{r['old_metrics']['f1_macro']:.3f} → {r['new_metrics']['f1_macro']:.3f}"
                             if r.get("old_metrics") else "-"),
            "result": (f"promoted v{r['model_version']}" if r.get("promoted") else (r.get("message") or "")),
        } for r in runs]), hide_index=True, width="stretch")
    else:
        st.caption("No retraining runs yet.")

st.divider()
st.caption("Data: PlantVillage (CC BY-SA 3.0) · Code: github.com/Sheja-Dorian/potato_disease_mlops")
