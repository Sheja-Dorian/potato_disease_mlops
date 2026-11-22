const COLORS = { "Early Blight": "#d08c2b", "Healthy": "#3a8f4b", "Late Blight": "#6b4f9e" };
const ADVICE = {
  "Healthy": "No disease detected. Keep scouting weekly, especially after cool, wet weather.",
  "Early Blight": "Early blight (Alternaria solani). Remove infected lower leaves, avoid overhead irrigation, rotate crops, and consider a protectant fungicide such as mancozeb. Ask your local extension officer.",
  "Late Blight": "Late blight (Phytophthora infestans) can destroy a field within days. Act quickly: remove and destroy infected plants and apply a recommended fungicide. Ask your local agronomist.",
};
const $ = (s) => document.querySelector(s);
const pct = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const charts = {};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab,.panel").forEach(e => e.classList.remove("active"));
  t.classList.add("active"); $("#" + t.dataset.tab).classList.add("active");
  if (t.dataset.tab === "insights") loadInsights();
  if (t.dataset.tab === "retrain") refreshRetrain();
});

/* ---------- dashboard / uptime ---------- */
function fmtUptime(s) {
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60), sec = Math.floor(s % 60);
  return d ? `${d}d ${h}h ${m}m` : h ? `${h}h ${m}m ${sec}s` : `${m}m ${sec}s`;
}
async function refreshHealth() {
  try {
    const h = await api("/health");
    $("#status-pill").className = "pill up";
    $("#status-text").textContent = h.retraining ? "online · retraining…" : `online · model v${h.model_version}`;
    $("#k-uptime").textContent = fmtUptime(h.uptime_seconds);
    $("#k-started").textContent = "since " + new Date(h.started_at).toLocaleString();
    $("#k-version").textContent = "v" + h.model_version;
    $("#k-saved").textContent = h.model_saved_at ? "saved " + new Date(h.model_saved_at).toLocaleString() : "";
    $("#k-acc").textContent = pct(h.model_test_accuracy);
    $("#k-preds").textContent = h.predictions_served.toLocaleString();
    $("#k-lat").textContent = h.avg_prediction_latency_ms ? `avg ${h.avg_prediction_latency_ms} ms (this worker)` : "this worker";
    if (window._ver !== h.model_version) { window._ver = h.model_version; loadModel(); }
  } catch (e) {
    $("#status-pill").className = "pill down"; $("#status-text").textContent = "offline";
  }
}
async function loadModel() {
  const m = await api("/model");
  const t = m.test_metrics || {};
  const rows = [["Accuracy", t.accuracy], ["Loss (cross-entropy)", t.loss, true], ["Precision (macro)", t.precision_macro],
    ["Recall (macro)", t.recall_macro], ["F1 (macro)", t.f1_macro], ["F1 (weighted)", t.f1_weighted], ["ROC-AUC (one-vs-rest)", t.roc_auc_ovr, true]];
  $("#metrics-table").innerHTML = `<tr><th>Metric</th><th style="text-align:right">v${m.version}</th></tr>` +
    rows.map(([n, v, raw]) => `<tr><td>${n}</td><td class="num">${v == null ? "–" : raw ? v.toFixed(4) : pct(v, 2)}</td></tr>`).join("");
  $("#k-f1").textContent = t.f1_macro ? "macro-F1 " + t.f1_macro.toFixed(3) : "";
  const hist = m.history || [];
  charts.hist?.destroy();
  charts.hist = new Chart($("#history-chart"), {
    type: "line",
    data: { labels: hist.map(h => "v" + h.version), datasets: [
      { label: "Accuracy", data: hist.map(h => h.accuracy), borderColor: "#3a8f4b", backgroundColor: "#3a8f4b", tension: .2 },
      { label: "Macro-F1", data: hist.map(h => h.f1_macro), borderColor: "#d08c2b", backgroundColor: "#d08c2b", tension: .2 }] },
    options: { scales: { y: { suggestedMin: .8, max: 1 } }, plugins: { tooltip: { callbacks: {
      afterBody: (it) => { const h = hist[it[0].dataIndex]; return h.trained_on_new ? `+${h.trained_on_new} new images` : "initial model"; } } } } }
  });
}
async function refreshMonitoring() {
  const s = await api("/monitoring");
  if (!s.recent) return;
  const mix = Object.entries(s.by_class).map(([k, v]) => `<span class="chip">${k.replace("_", " ")}: ${v}</span>`).join(" ");
  $("#mon-out").innerHTML = `<div class="kpis" style="margin:0 0 10px">
    <div class="kpi"><label>Logged predictions</label><b>${s.total.toLocaleString()}</b></div>
    <div class="kpi"><label>Mean confidence</label><b>${pct(s.mean_confidence)}</b><small>last ${s.recent}</small></div>
    <div class="kpi"><label>Low-confidence (&lt;70%)</label><b>${pct(s.low_confidence_share)}</b><small>possible drift signal</small></div>
    <div class="kpi"><label>Latency p50 / p95</label><b>${s.latency_p50_ms.toFixed(0)} / ${s.latency_p95_ms.toFixed(0)} ms</b></div></div>
    <div class="pending">${mix}</div>`;
}
function cmTable(cm, names) {
  return `<table class="cm"><tr><th>true ↓ / pred →</th>${names.map(n => `<th>${n}</th>`).join("")}</tr>` +
    cm.map((r, i) => `<tr><th>${names[i]}</th>${r.map((v, j) => `<td style="background:${i === j ? "#e3f1e5" : v ? "#fbe9e6" : "#fff"}">${v}</td>`).join("")}</tr>`).join("") + "</table>";
}
$("#eval-btn").onclick = async () => {
  const btn = $("#eval-btn"); btn.disabled = true; btn.textContent = "Evaluating…";
  try {
    const fd = new FormData(); const f = $("#eval-zip").files[0]; if (f) fd.append("file", f);
    const r = await api("/evaluate", { method: "POST", body: fd });
    $("#eval-out").innerHTML = `<div class="eval-grid"><table>
      <tr><th>Evaluated on</th><td>${r.source} · model v${r.model_version} · ${r.seconds}s</td></tr>
      <tr><th>Accuracy</th><td>${pct(r.accuracy, 2)}</td></tr><tr><th>Loss</th><td>${r.loss.toFixed(4)}</td></tr>
      <tr><th>Precision (macro)</th><td>${pct(r.precision_macro, 2)}</td></tr><tr><th>Recall (macro)</th><td>${pct(r.recall_macro, 2)}</td></tr>
      <tr><th>F1 (macro)</th><td>${pct(r.f1_macro, 2)}</td></tr><tr><th>ROC-AUC</th><td>${r.roc_auc_ovr ? r.roc_auc_ovr.toFixed(4) : "–"}</td></tr></table>
      <div>${cmTable(r.confusion_matrix, r.classes)}</div></div>`;
  } catch (e) { $("#eval-out").innerHTML = `<p class="bad">${e.message}</p>`; }
  btn.disabled = false; btn.textContent = "Run evaluation";
};

/* ---------- predict ---------- */
const drop = $("#drop");
function setFile(f) {
  if (!f) return; window._predFile = f;
  const img = $("#pred-preview"); img.src = URL.createObjectURL(f); img.hidden = false; $("#drop-hint").hidden = true;
  $("#pred-btn").disabled = false; $("#pred-out").innerHTML = '<span class="muted">Ready. Click Predict.</span>';
}
$("#pred-file").onchange = (e) => setFile(e.target.files[0]);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); setFile(e.dataTransfer.files[0]); };
$("#pred-btn").onclick = async () => {
  const btn = $("#pred-btn"); btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const fd = new FormData(); fd.append("file", window._predFile);
    const r = await api("/predict", { method: "POST", body: fd });
    const bars = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
      `<div class="bar"><span>${k}</span><div class="track"><div class="fill" style="width:${v * 100}%;background:${COLORS[k]}"></div></div><em>${pct(v)}</em></div>`).join("");
    $("#pred-out").innerHTML = `<div class="muted">${r.filename}</div>
      <div class="result-label" style="color:${COLORS[r.display_label]}">${r.display_label}</div>
      <div class="conf">${pct(r.confidence)} confidence · model v${r.model_version} · ${r.latency_ms} ms</div>${bars}
      <div class="advice">${ADVICE[r.display_label]}</div>`;
    refreshHealth();
  } catch (e) { $("#pred-out").innerHTML = `<p class="bad">${e.message}</p>`; }
  btn.disabled = false; btn.textContent = "Predict";
};

/* ---------- insights ---------- */
async function loadInsights() {
  if (window._insights) return;
  const ins = window._insights = await api("/insights");
  const interp = ins.interpretations || {};
  Object.entries(interp).forEach(([k, v]) => { const el = $("#i-" + k); if (el) el.textContent = v; });
  new Chart($("#c-counts"), { type: "bar", data: { labels: ins.classes, datasets: [
    { label: "train", data: ins.classes.map(c => ins.class_counts.train[c]), backgroundColor: ins.classes.map(c => COLORS[c]) },
    { label: "test", data: ins.classes.map(c => ins.class_counts.test[c]), backgroundColor: ins.classes.map(c => COLORS[c] + "77") }] },
    options: { plugins: { legend: { display: true } } } });
  ["green_ratio", "lesion_fraction", "texture"].forEach(f => {
    const F = ins.features[f]; const mids = F.bins.slice(0, -1).map((b, i) => ((b + F.bins[i + 1]) / 2).toFixed(f === "lesion_fraction" ? 2 : 3));
    new Chart($("#c-" + f), { type: "line", data: { labels: mids, datasets: ins.classes.map(c => {
      const tot = F.hist[c].reduce((a, b) => a + b, 0) || 1;
      return { label: `${c} (median ${F.median[c].toFixed(3)})`, data: F.hist[c].map(v => v / tot), borderColor: COLORS[c],
        backgroundColor: COLORS[c] + "33", fill: true, tension: .35, pointRadius: 0 }; }) },
      options: { scales: { y: { title: { display: true, text: "share of class" } }, x: { ticks: { maxTicksLimit: 8 } } } } });
  });
}

/* ---------- upload + retrain ---------- */
$("#up-btn").onclick = async () => {
  const files = $("#up-files").files; if (!files.length) return alert("Choose images or a zip first");
  const btn = $("#up-btn"); btn.disabled = true; btn.textContent = "Uploading…";
  try {
    const fd = new FormData(); [...files].forEach(f => fd.append("files", f));
    if ($("#up-label").value) fd.append("label", $("#up-label").value);
    const r = await api("/upload", { method: "POST", body: fd });
    $("#up-out").innerHTML = `<p><span class="ok">${r.accepted} images saved to the database</span>` +
      (r.duplicates ? ` · ${r.duplicates} duplicates skipped` : "") + (r.n_rejected ? ` · <span class="bad">${r.n_rejected} rejected</span>` : "") + "</p>" +
      (r.rejected.length ? `<ul class="muted">${r.rejected.slice(0, 5).map(x => `<li>${x.file}: ${x.reason}</li>`).join("")}</ul>` : "") +
      (r.auto_retrain_triggered ? `<p class="ok">Threshold of ${r.auto_retrain_threshold} reached: retraining started automatically.</p>` : "");
    refreshRetrain();
  } catch (e) { $("#up-out").innerHTML = `<p class="bad">${e.message}</p>`; }
  btn.disabled = false; btn.textContent = "Upload";
};
$("#rt-btn").onclick = async () => {
  try { await api("/retrain", { method: "POST" }); refreshRetrain(); } catch (e) { alert(e.message); }
};
async function refreshRetrain() {
  const [u, s] = await Promise.all([api("/uploads"), api("/retrain/status")]);
  const by = Object.entries(u.pending).map(([k, v]) => `<span class="chip">${k.replace("_", " ")}: ${v}</span>`).join("");
  $("#pending").innerHTML = `<span class="chip"><b>${u.pending_total}</b> images waiting</span>${by}` +
    (u.auto_retrain_threshold ? `<span class="chip">auto-retrain at ${u.auto_retrain_threshold}</span>` : "");
  $("#rt-btn").disabled = s.running || !u.pending_total;
  $("#rt-btn").textContent = s.running ? "Retraining…" : "Retrain model";
  const log = $("#rt-log");
  if (s.live.log && s.live.log.length) { log.hidden = false; log.textContent = s.live.log.join("\n"); log.scrollTop = 1e6; }
  $("#runs").innerHTML = `<tr><th>#</th><th>Started</th><th>Trigger</th><th>Status</th><th>New / replay imgs</th><th>Accuracy old → new</th><th>F1 old → new</th><th>Result</th></tr>` +
    s.runs.map(r => `<tr><td>${r.id}</td><td>${new Date(r.started_at).toLocaleString()}</td><td>${r.trigger}</td><td>${r.status}</td>
      <td>${r.n_new ?? "–"} / ${r.n_replay ?? "–"}</td>
      <td>${r.old_metrics ? pct(r.old_metrics.accuracy, 2) + " → " + pct(r.new_metrics.accuracy, 2) : "–"}</td>
      <td>${r.old_metrics ? r.old_metrics.f1_macro.toFixed(3) + " → " + r.new_metrics.f1_macro.toFixed(3) : "–"}</td>
      <td>${r.status === "completed" ? (r.promoted ? `<span class="ok">promoted v${r.model_version}</span>` : "kept old model") : (r.message || "")}</td></tr>`).join("");
  clearTimeout(window._rtTimer);
  if (s.running) window._rtTimer = setTimeout(refreshRetrain, 2000);
  else if (window._wasRunning) refreshHealth();
  window._wasRunning = s.running;
}

refreshHealth(); refreshMonitoring();
setInterval(refreshHealth, 5000); setInterval(refreshMonitoring, 10000);
