# Video demo script (≈ 6–8 minutes, camera ON)

The rubric awards full marks for a clear demo of **prediction** and **retraining** with the camera on.
Keep your face visible in a corner of the screen recording (OBS, Loom, or Zoom "share screen + camera").

| # | Show | Say (roughly) | Time |
|---|---|---|---|
| 1 | Yourself + README on GitHub | Who you are, the problem: potato blight in Rwanda, 3 classes, PlantVillage data | 0:30 |
| 2 | Notebook, scrolled | Preprocessing steps → baseline vs optimised CNN (augmentation, BN, L2, dropout, Adam, early stopping) → metrics table (accuracy, loss, precision, recall, F1, ROC-AUC) → confusion matrix | 1:30 |
| 3 | Live Render URL → **Dashboard** | Uptime counter, model version, test metrics. Click **Run evaluation**: the model is evaluated *in production* | 0:45 |
| 4 | **Data insights** tab | The 3 features + the story (colour splits healthy vs diseased, texture splits the two blights) | 0:45 |
| 5 | **Predict** tab | Upload `demo/predict_samples/late_blight_1.jpg` → *Late Blight* with confidence; repeat with a healthy one. Say the true label out loud before predicting | 1:00 |
| 6 | **Upload & retrain** tab | Upload `demo/new_field_data.zip` → "N images saved to the database". Point out pending counts per class. Click **Retrain model** and narrate the live log: loads from DB → preprocesses → loads current model as pre-trained → fine-tunes → compares old vs new → promotes v2 | 1:30 |
| 7 | Dashboard again | Model version is now v2, version-history chart updated | 0:20 |
| 8 | Locust results (README table / charts) | Latency and RPS with 1 vs 2 vs 4 replicas, what the numbers mean | 0:45 |

Tips
* Open the Render URL and check the Dashboard says "online" before recording.
* With `RETRAIN_THRESHOLD=100`, uploading the 324-image zip will **auto-trigger** retraining. Mention this as the automatic trigger. If you want to press the button yourself on camera, set `RETRAIN_THRESHOLD` to 1000 in Render first, or upload a smaller zip.
* Retraining takes about 2 minutes on a 1-CPU instance. Cut or speed up the waiting part of the video.
