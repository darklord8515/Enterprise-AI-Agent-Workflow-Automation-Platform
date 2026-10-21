// Supplier Evidence Review: front-end for the FastAPI backend.
// Vanilla ES module, no build step. All server-provided text is inserted with textContent.

const $ = (sel, root = document) => root.querySelector(sel);

const STATUS = {
  met: { cls: "met", label: "Met", icon: "i-check" },
  gap_identified: { cls: "gap", label: "Gap identified", icon: "i-gap" },
  insufficient_evidence: { cls: "insufficient", label: "Insufficient evidence", icon: "i-unknown" },
};
const STEPS = [
  { key: "plan", label: "Plan", nodes: ["planner"] },
  { key: "retrieve", label: "Retrieve", nodes: ["retriever"] },
  { key: "analyse", label: "Analyse", nodes: ["analyst"] },
  { key: "verify", label: "Verify", nodes: ["evidence_reviewer"] },
  { key: "route", label: "Route", nodes: ["router", "insufficient_evidence"] },
  { key: "report", label: "Report", nodes: ["report_generator"] },
];
const NODE_TO_STEP = Object.fromEntries(STEPS.flatMap((s, i) => s.nodes.map((n) => [n, i])));
const SUPPLIER_RE = /^[A-Za-z0-9_\-]{1,64}$/;
const HISTORY_KEY = "ser-history";
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  health: null,
  apiKey: sessionGet("ser-api-key") || "",
  docs: [],
  transient: new Map(), // local rows for uploads in flight: id -> {filename, status, note}
  category: "supplier",
  reqMode: "policy",
  busy: false,
  currentTaskId: null,
  pollTimer: null,
  clockTimer: null,
};

/* ------------------------------------------------------------------ utilities */

function h(tag, attrs = {}, ...children) {
  const el = document.createElementNS(
    ["svg", "use"].includes(tag) ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml",
    tag,
  );
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") {
      for (const [prop, val] of Object.entries(v)) {
        if (prop.startsWith("--")) el.style.setProperty(prop, val); else el.style[prop] = val;
      }
    }
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const icon = (id, cls = "ic") => h("svg", { class: cls, "aria-hidden": "true" }, h("use", { href: `#${id}` }));

function sessionGet(k) { try { return sessionStorage.getItem(k); } catch { return null; } }
function sessionSet(k, v) { try { v ? sessionStorage.setItem(k, v) : sessionStorage.removeItem(k); } catch { /* storage unavailable */ } }
function localGet(k, fallback) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fallback; } catch { return fallback; } }
function localSet(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } }

function fmtMs(ms) {
  if (ms == null) return "–";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}
function fmtNum(n) { return new Intl.NumberFormat().format(n ?? 0); }
function fmtBytes(b) { return b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`; }
function relTime(iso) {
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function countUp(el, to, duration = 700) {
  if (reducedMotion || to === 0) { el.textContent = String(to); return; }
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    el.textContent = String(Math.round(to * (1 - Math.pow(1 - p, 3))));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ------------------------------------------------------------------ toasts */

function toast(kind, title, msg, ttl = 3600) {
  const ic = kind === "ok" ? "i-check" : kind === "error" ? "i-gap" : "i-shield";
  const el = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" },
    icon(ic),
    h("div", { class: "toast-body" }, h("div", { class: "toast-title", text: title }), msg ? h("div", { class: "toast-msg", text: msg }) : null),
    h("button", { class: "icon-btn", "aria-label": "Dismiss", style: { width: "24px", height: "24px" }, onclick: () => dismiss() }, icon("i-x")),
  );
  const dismiss = () => { el.classList.add("out"); setTimeout(() => el.remove(), 260); };
  $("#toasts").append(el);
  if (ttl) setTimeout(dismiss, ttl);
}

/* ------------------------------------------------------------------ API */

class ApiError extends Error {
  constructor(status, message, code) { super(message); this.status = status; this.code = code; }
}

async function api(path, { method = "GET", json, form } = {}) {
  const headers = {};
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form) body = form;
  let res;
  try {
    res = await fetch(path, { method, headers, body });
  } catch {
    throw new ApiError(0, "Cannot reach the server. Is it running?", "network");
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* non-JSON */ }
  if (!res.ok) {
    let message = `Request failed (HTTP ${res.status})`;
    let code;
    if (data?.error) { message = data.error.message; code = data.error.code; }
    else if (Array.isArray(data?.detail)) message = data.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    else if (typeof data?.detail === "string") message = data.detail;
    if (res.status === 401) openKeyDialog();
    throw new ApiError(res.status, message, code);
  }
  return data;
}

/* ------------------------------------------------------------------ theme, health, key */

function initTheme() {
  $("#theme-btn").addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("ser-theme", next); } catch { /* ignore */ }
  });
}

async function loadHealth() {
  const pill = $("#env-pill");
  try {
    const hres = await api("/health");
    state.health = hres;
    pill.dataset.state = hres.provider_mode === "azure" ? "live" : "mock";
    $("#env-text").textContent = `${hres.provider_mode} · ${hres.retrieval_mode}`;
    pill.title = hres.provider_mode === "azure"
      ? "Azure OpenAI + Azure AI Search"
      : "Mock mode: deterministic local providers, no Azure calls";
    $("#key-btn").hidden = !hres.auth_enabled;
    updateKeyButton();
    return true;
  } catch {
    pill.dataset.state = "down";
    $("#env-text").textContent = "offline";
    return false;
  }
}

function updateKeyButton() {
  $("#key-btn-text").textContent = state.apiKey ? "Key set" : "API key";
}

function openKeyDialog() {
  const dlg = $("#key-dialog");
  if (dlg.open) return;
  $("#key-input").value = state.apiKey;
  dlg.showModal();
  setTimeout(() => $("#key-input").focus(), 30);
}

function initKeyDialog() {
  $("#key-btn").addEventListener("click", openKeyDialog);
  // Enter in the field means "save" (implicit submission would pick the first button, Cancel).
  $("#key-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); $("#key-dialog").close("save"); }
  });
  $("#key-dialog").addEventListener("close", async () => {
    if ($("#key-dialog").returnValue !== "save") return;
    state.apiKey = $("#key-input").value.trim();
    sessionSet("ser-api-key", state.apiKey);
    updateKeyButton();
    await refreshDocs();
  });
}

/* ------------------------------------------------------------------ segmented controls */

function initSeg(el, onChange) {
  const buttons = [...el.querySelectorAll("button")];
  const select = (i, notify = true) => {
    buttons.forEach((b, j) => b.setAttribute("aria-checked", String(i === j)));
    el.dataset.index = String(i);
    if (notify) onChange(buttons[i].dataset.value);
  };
  buttons.forEach((b, i) => b.addEventListener("click", () => select(i)));
  el.addEventListener("keydown", (e) => {
    const i = Number(el.dataset.index || 0);
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const next = (i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
      select(next); buttons[next].focus(); e.preventDefault();
    }
  });
  select(0, false);
}

/* ------------------------------------------------------------------ documents */

async function refreshDocs() {
  try {
    state.docs = await api("/documents");
  } catch (e) {
    if (e.status !== 401) toast("error", "Could not load documents", e.message);
    state.docs = [];
  }
  renderDocs();
}

function renderDocs() {
  const list = $("#doc-list");
  const rows = [];
  const sorted = [...state.docs].sort((a, b) => (a.category === b.category ? a.filename.localeCompare(b.filename) : a.category === "policy" ? -1 : 1));
  for (const d of sorted) {
    const t = state.transient.get(d.document_id);
    const st = t?.status === "indexing"
      ? h("span", { class: "doc-state busy" }, h("span", { class: "spinner" }), "Indexing")
      : d.indexed
        ? h("span", { class: "doc-state indexed" }, `${d.chunk_count} chunk${d.chunk_count === 1 ? "" : "s"}`)
        : h("span", { class: "doc-state" }, "Not indexed");
    rows.push(h("li", { class: "doc" },
      icon("i-file"),
      h("span", {}, h("span", { class: "doc-name", title: d.filename, text: d.filename }),
        h("span", { class: "doc-meta", text: `${d.category === "policy" ? "Policy" : `Supplier · ${d.supplier_id}`} · ${fmtBytes(d.size_bytes)}` })),
      st,
    ));
  }
  for (const [id, t] of state.transient) {
    if (state.docs.some((d) => d.document_id === id)) continue;
    rows.push(h("li", { class: "doc" }, icon("i-file"),
      h("span", {}, h("span", { class: "doc-name", text: t.filename }), h("span", { class: "doc-meta", text: t.note || "" })),
      t.status === "error"
        ? h("span", { class: "doc-state error" }, "Failed")
        : h("span", { class: "doc-state busy" }, h("span", { class: "spinner" }), "Uploading"),
    ));
  }
  list.replaceChildren(...rows);
  $("#docs-empty").hidden = rows.length > 0;

  const pending = state.docs.filter((d) => !d.indexed);
  const idx = $("#index-btn");
  idx.disabled = pending.length === 0 || state.busy;
  idx.textContent = pending.length
    ? `Index ${pending.length} document${pending.length > 1 ? "s" : ""}`
    : state.docs.length ? "All documents indexed" : "Index documents";
  $("#dropzone").classList.toggle("compact", rows.length > 0);

  renderSupplierSelect();
}

function renderSupplierSelect() {
  const sel = $("#supplier-select");
  const previous = sel.value;
  const suppliers = new Map();
  for (const d of state.docs) {
    if (d.category !== "supplier") continue;
    const s = suppliers.get(d.supplier_id) || { total: 0, indexed: 0 };
    s.total += 1; if (d.indexed) s.indexed += 1;
    suppliers.set(d.supplier_id, s);
  }
  if (!suppliers.size) {
    sel.replaceChildren(h("option", { value: "", text: "Upload supplier documents first" }));
  } else {
    sel.replaceChildren(...[...suppliers].sort().map(([id, s]) =>
      h("option", { value: id, text: `${id}  (${s.indexed}/${s.total} indexed)` })));
    if (suppliers.has(previous)) sel.value = previous;
  }
  updateRunButton();
}

function updateRunButton() {
  const supplier = $("#supplier-select").value;
  const hasIndexed = state.docs.some((d) => d.category === "supplier" && d.supplier_id === supplier && d.indexed);
  $("#run-btn").disabled = state.busy || !supplier || !hasIndexed;
}

async function uploadFiles(files) {
  if (!files.length) return;
  const category = state.category;
  const supplierInput = $("#supplier-input");
  const supplierId = supplierInput.value.trim();
  if (category === "supplier" && !SUPPLIER_RE.test(supplierId)) {
    supplierInput.setAttribute("aria-invalid", "true");
    supplierInput.focus();
    toast("error", "Supplier ID needed", "Use letters, numbers, - or _ (max 64).");
    return;
  }
  supplierInput.removeAttribute("aria-invalid");
  let ok = 0;
  for (const file of files) {
    const tmp = `tmp-${crypto.randomUUID()}`;
    state.transient.set(tmp, { filename: file.name, status: "uploading", note: category === "policy" ? "Policy" : `Supplier · ${supplierId}` });
    renderDocs();
    try {
      await uploadOne(file, category, supplierId);
      state.transient.delete(tmp);
      ok += 1;
    } catch (e) {
      state.transient.set(tmp, { filename: file.name, status: "error", note: e.message });
      toast("error", `Upload failed: ${file.name}`, e.message);
      setTimeout(() => { state.transient.delete(tmp); renderDocs(); }, 6000);
    }
    await refreshDocs();
  }
  if (ok) toast("ok", `Uploaded ${ok} file${ok > 1 ? "s" : ""}`, "Index them to make them searchable.");
}

async function uploadOne(file, category, supplierId) {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("category", category);
  if (category === "supplier") form.append("supplier_id", supplierId);
  return api("/documents/upload", { method: "POST", form });
}

async function indexPending() {
  const pending = state.docs.filter((d) => !d.indexed).map((d) => d.document_id);
  if (!pending.length) return;
  setBusy(true);
  pending.forEach((id) => state.transient.set(id, { status: "indexing" }));
  renderDocs();
  const btn = $("#index-btn");
  btn.replaceChildren(h("span", { class: "spinner" }), `Indexing ${pending.length}…`);
  try {
    const res = await api("/documents/index", { method: "POST", json: { document_ids: pending } });
    const chunks = res.indexed.reduce((n, d) => n + d.chunk_count, 0);
    toast("ok", "Indexed", `${res.indexed.length} document${res.indexed.length > 1 ? "s" : ""} · ${chunks} chunks · ${res.provider_mode}`);
  } catch (e) {
    toast("error", "Indexing failed", e.message, 9000);
  } finally {
    pending.forEach((id) => state.transient.delete(id));
    setBusy(false);
    await refreshDocs();
  }
}

function initDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
  let depth = 0;
  dz.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; dz.classList.add("drag"); });
  dz.addEventListener("dragover", (e) => e.preventDefault());
  dz.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; dz.classList.remove("drag"); } });
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; dz.classList.remove("drag");
    uploadFiles([...(e.dataTransfer?.files || [])]);
  });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => e.preventDefault());
}

/* ------------------------------------------------------------------ sample dataset */

let sampleSpec = null;

async function initSamples() {
  try {
    const res = await fetch("/samples/cases.json");
    if (!res.ok) return;
    sampleSpec = await res.json();
    $("#sample-btn").hidden = false;
  } catch { /* samples not served */ }
}

async function loadSamples() {
  if (!sampleSpec || state.busy) return;
  const btn = $("#sample-btn");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    for (const doc of sampleSpec.documents) {
      const exists = state.docs.some((d) => d.filename === doc.file && d.category === doc.category && (d.supplier_id || null) === (doc.supplier_id || null));
      if (exists) continue;
      const tmp = `tmp-${crypto.randomUUID()}`;
      state.transient.set(tmp, { filename: doc.file, status: "uploading", note: doc.category === "policy" ? "Policy" : `Supplier · ${doc.supplier_id}` });
      renderDocs();
      const blob = await (await fetch(`/samples/documents/${encodeURIComponent(doc.file)}`)).blob();
      try {
        await uploadOne(new File([blob], doc.file), doc.category, doc.supplier_id);
      } finally {
        state.transient.delete(tmp);
      }
      await refreshDocs();
    }
    await indexPending();
    const first = sampleSpec.cases[0];
    $("#supplier-select").value = first.supplier_id;
    $("#task-input").value = first.task;
    $("#supplier-input").value = first.supplier_id;
    updateRunButton();
    toast("info", "Sample set ready", "Fictional policy and two suppliers. One supplier note contains a planted prompt injection.");
  } catch (e) {
    toast("error", "Could not load samples", e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Load sample set";
  }
}

/* ------------------------------------------------------------------ run */

function setBusy(b) {
  state.busy = b;
  $("#dropzone").classList.toggle("disabled", b);
  renderDocs();
}

async function runEvaluation() {
  const supplier = $("#supplier-select").value;
  const task = $("#task-input").value.trim();
  if (task.length < 5) { toast("error", "Describe the evaluation", "The request needs at least 5 characters."); return; }
  const body = { task, supplier_id: supplier };
  if (state.reqMode === "custom") {
    const reqs = $("#req-input").value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!reqs.length) { toast("error", "No requirements", "Add one requirement per line, or extract them from policy."); return; }
    body.requirements = reqs;
  }
  setBusy(true);
  const runBtn = $("#run-btn");
  runBtn.replaceChildren(h("span", { class: "spinner" }), h("span", { text: "Running…" }));
  try {
    const accepted = await api("/tasks/analyze", { method: "POST", json: body });
    state.currentTaskId = accepted.task_id;
    showRunning(supplier, task);
    await poll(accepted.task_id, supplier);
  } catch (e) {
    toast("error", "Could not start evaluation", e.message);
  } finally {
    setBusy(false);
    runBtn.replaceChildren(icon("i-play"), h("span", { text: "Run evaluation" }));
    updateRunButton();
  }
}

function pipelineEl({ running = false } = {}) {
  return h("ol", { class: `pipeline${running ? " running" : ""}`, "aria-label": "Workflow progress" },
    STEPS.map((s, i) => h("li", { "data-step": s.key },
      h("span", { class: "node" }, h("span", { class: "n", text: String(i + 1) }), icon("i-check")),
      h("span", { class: "node-label", text: s.label }),
      h("span", { class: "node-time" }),
    )));
}

function showRunning(supplier, task) {
  $("#intro").hidden = true;
  const run = $("#run");
  run.hidden = false;
  const elapsed = h("span", { text: "0.0 s" });
  const mode = state.health?.provider_mode === "azure" ? "Azure OpenAI · typically 20–40 s" : "mock providers · near-instant";
  run.replaceChildren(
    h("section", { class: "card run-status" },
      h("div", { class: "run-status-head" },
        h("div", { class: "run-status-title" }, h("span", { class: "spinner" }), h("span", { text: `Evaluating ${supplier}` })),
        h("div", { class: "run-status-meta" }, h("span", {}, icon("i-clock"), elapsed), h("span", { text: mode })),
      ),
      h("p", { class: "muted small", style: { marginTop: "6px" }, text: task }),
      pipelineEl({ running: true }),
    ),
    h("div", { class: "skeleton", "aria-hidden": "true" }, h("div", { class: "sk" }), h("div", { class: "sk" }), h("div", { class: "sk" })),
  );
  const t0 = performance.now();
  clearInterval(state.clockTimer);
  state.clockTimer = setInterval(() => { elapsed.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`; }, 100);
  $("#main").focus({ preventScroll: true });
  $("#main").scrollIntoView({ block: "start", behavior: reducedMotion ? "auto" : "smooth" });
}

async function poll(taskId, supplier) {
  for (;;) {
    await sleep(900);
    if (state.currentTaskId !== taskId) return;
    let rec;
    try {
      rec = await api(`/tasks/${taskId}`);
    } catch (e) {
      clearInterval(state.clockTimer);
      toast("error", "Lost track of the task", e.message);
      return;
    }
    if (rec.status === "completed" || rec.status === "failed") {
      clearInterval(state.clockTimer);
      saveHistory(rec, supplier);
      await renderTask(rec);
      return;
    }
  }
}

/* ------------------------------------------------------------------ history */

function saveHistory(rec, supplier) {
  const list = localGet(HISTORY_KEY, []).filter((x) => x.id !== rec.task_id);
  list.unshift({ id: rec.task_id, supplier: supplier || rec.request?.supplier_id, at: rec.created_at, status: rec.status, summary: rec.report?.summary || null });
  localSet(HISTORY_KEY, list.slice(0, 8));
  renderHistory();
}

function renderHistory() {
  const list = localGet(HISTORY_KEY, []);
  $("#history-panel").hidden = list.length === 0;
  $("#history").replaceChildren(...list.map((item) => {
    const s = item.summary;
    const total = s ? (s.met + s.gap_identified + s.insufficient_evidence) || 1 : 1;
    const bar = s
      ? h("span", { class: "mini-bar", "aria-hidden": "true" },
        h("span", { class: "fill-met", style: { width: `${(s.met / total) * 100}%` } }),
        h("span", { class: "fill-gap", style: { width: `${(s.gap_identified / total) * 100}%` } }),
        h("span", { class: "fill-insufficient", style: { width: `${(s.insufficient_evidence / total) * 100}%` } }))
      : h("span", { class: "doc-state error", style: { gridRow: "1 / span 2" } }, "Failed");
    return h("li", {}, h("button", {
      type: "button", "aria-current": String(item.id === state.currentTaskId),
      onclick: () => openHistory(item.id),
    }, h("span", { class: "h-name", text: item.supplier }), bar, h("span", { class: "h-time", text: relTime(item.at) })));
  }));
}

async function openHistory(id) {
  try {
    const rec = await api(`/tasks/${id}`);
    state.currentTaskId = id;
    $("#intro").hidden = true;
    $("#run").hidden = false;
    await renderTask(rec, { replay: false });
    renderHistory();
  } catch (e) {
    if (e.status === 404) {
      localSet(HISTORY_KEY, localGet(HISTORY_KEY, []).filter((x) => x.id !== id));
      renderHistory();
      toast("info", "Run no longer available", "Tasks are kept in server memory and are cleared on restart.");
    } else toast("error", "Could not open run", e.message);
  }
}

/* ------------------------------------------------------------------ report */

async function renderTask(rec, { replay = true } = {}) {
  const run = $("#run");
  const pipe = pipelineEl();
  const statusTitle = rec.status === "completed"
    ? h("div", { class: "run-status-title" }, h("span", { text: "Workflow trace" }))
    : h("div", { class: "run-status-title" }, h("span", { text: "Evaluation failed" }));
  const report = rec.report;
  const meta = report
    ? h("div", { class: "run-status-meta" },
      h("span", {}, icon("i-clock"), fmtMs(report.latency_ms)),
      h("span", { text: `${report.usage.llm_calls} LLM calls` }),
      h("span", { text: report.provider_mode }))
    : h("div", { class: "run-status-meta" }, h("span", { text: rec.error?.code || "error" }));
  const statusCard = h("section", { class: "card run-status" }, h("div", { class: "run-status-head" }, statusTitle, meta), pipe);
  const children = [statusCard];
  if (rec.status === "failed") {
    children.push(h("section", { class: "card error-card", role: "alert" }, icon("i-gap"),
      h("div", {}, h("h3", { text: rec.error?.message || "The workflow failed." }),
        h("p", {}, "Code ", h("code", { text: rec.error?.code || "unknown" }),
          rec.error?.failed_node ? [" · failed at ", h("code", { text: rec.error.failed_node })] : null,
          rec.error?.retryable ? " · retryable" : null))));
  } else {
    children.push(...reportEls(rec));
  }
  run.replaceChildren(...children);
  await animatePipeline(pipe, rec, replay);
}

async function animatePipeline(pipe, rec, replay) {
  const items = [...pipe.children];
  const trace = rec.report?.trace || [];
  const durations = new Array(STEPS.length).fill(0);
  const reached = new Set();
  for (const ev of trace) {
    const i = NODE_TO_STEP[ev.node];
    if (i === undefined) continue;
    durations[i] += ev.duration_ms;
    reached.add(i);
  }
  let failedAt = -1;
  if (rec.status === "failed") {
    failedAt = NODE_TO_STEP[rec.error?.failed_node] ?? 0;
    for (let i = 0; i < failedAt; i++) reached.add(i);
  }
  for (let i = 0; i < items.length; i++) {
    const li = items[i];
    if (i === failedAt) {
      li.classList.add("failed");
      $(".node-time", li).textContent = "failed";
      $(".node-time", li).style.opacity = "1";
      break;
    }
    if (!reached.has(i)) { li.classList.add("skipped"); continue; }
    if (replay && !reducedMotion) { li.classList.add("active"); await sleep(170); li.classList.remove("active"); }
    li.classList.add("done");
    $(".node-time", li).textContent = rec.status === "failed" ? "" : fmtMs(durations[i]);
  }
  const revisions = rec.report?.revisions || 0;
  if (revisions > 0) {
    items[1].append(h("span", { class: "revision-chip", title: "Bounded revision: unsupported findings were re-retrieved and re-analysed" },
      icon("i-loop"), `${revisions} revision${revisions > 1 ? "s" : ""}`));
  }
}

let tabsObserver = null;

function reportEls(rec) {
  const r = rec.report;
  const s = r.summary;
  const total = s.met + s.gap_identified + s.insufficient_evidence || 1;

  // Header + summary
  const bar = h("div", { class: "stack-bar", role: "img", "aria-label": `${s.met} met, ${s.gap_identified} gaps, ${s.insufficient_evidence} insufficient evidence` },
    h("span", { class: "fill-met" }), h("span", { class: "fill-gap" }), h("span", { class: "fill-insufficient" }));
  const nums = [h("span", { class: "stat-num", text: "0" }), h("span", { class: "stat-num", text: "0" }), h("span", { class: "stat-num", text: "0" })];
  const head = h("section", { class: "card report-head" },
    h("div", { class: "report-title-row" },
      h("div", {}, h("p", { class: "eyebrow", text: "Supplier assessment" }), h("h1", { text: r.supplier_id }), h("p", { class: "report-task", text: r.task })),
      h("div", { class: "report-actions" },
        h("button", { class: "btn ghost sm", type: "button", onclick: () => exportJson(rec) }, icon("i-download"), "JSON"),
        h("button", { class: "btn ghost sm", type: "button", onclick: () => { openAll(true); setTimeout(() => window.print(), 350); } }, icon("i-print"), "Print"))),
    h("div", { class: "summary" }, bar,
      h("div", { class: "stats" },
        h("div", { class: "stat met" }, nums[0], h("span", { class: "stat-label", text: "Met" })),
        h("div", { class: "stat gap" }, nums[1], h("span", { class: "stat-label", text: "Gaps identified" })),
        h("div", { class: "stat insufficient" }, nums[2], h("span", { class: "stat-label", text: "Insufficient evidence" })))),
  );
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const [a, b, c] = bar.children;
    a.style.width = `${(s.met / total) * 100}%`;
    b.style.width = `${(s.gap_identified / total) * 100}%`;
    c.style.width = `${(s.insufficient_evidence / total) * 100}%`;
    countUp(nums[0], s.met); countUp(nums[1], s.gap_identified); countUp(nums[2], s.insufficient_evidence);
  }));

  const els = [head];

  if (r.warnings.length) {
    els.push(h("section", { class: "callout", role: "note" }, icon("i-shield"),
      h("div", {}, h("strong", { text: `${r.warnings.length} warning${r.warnings.length > 1 ? "s" : ""} for human review` }),
        h("ul", {}, r.warnings.map((w) => h("li", { text: w }))))));
  }

  // Filters
  const counts = { all: r.findings.length, gap_identified: s.gap_identified, insufficient_evidence: s.insufficient_evidence, met: s.met };
  const tabDefs = [["all", "All"], ["gap_identified", "Gaps"], ["insufficient_evidence", "Insufficient"], ["met", "Met"]];
  const thumb = h("span", { class: "tabs-thumb", "aria-hidden": "true" });
  const tabs = h("div", { class: "tabs", role: "tablist", "aria-label": "Filter findings" }, thumb,
    tabDefs.map(([key, label], i) => h("button", {
      type: "button", role: "tab", "aria-selected": String(i === 0), "data-filter": key,
      onclick: (e) => applyFilter(key, e.currentTarget),
    }, label, h("span", { class: "count", text: String(counts[key]) }))));
  const expandBtn = h("button", { class: "btn link sm expand-all", type: "button", onclick: () => openAll(!allOpen()) }, "Expand all");
  const toolbar = h("div", { class: "toolbar" }, tabs, expandBtn);
  els.push(toolbar);

  const findings = h("div", { class: "findings" }, r.findings.map((f, i) => findingEl(f, i)));
  const empty = h("p", { class: "empty-filter", hidden: true, text: "No findings in this category." });
  els.push(findings, empty);

  function moveThumb(btn) {
    thumb.style.width = `${btn.offsetWidth}px`;
    thumb.style.transform = `translateX(${btn.offsetLeft}px)`;
  }
  requestAnimationFrame(() => moveThumb(tabs.querySelector('[aria-selected="true"]')));
  tabsObserver?.disconnect();
  tabsObserver = new ResizeObserver(() => moveThumb(tabs.querySelector('[aria-selected="true"]')));
  tabsObserver.observe(tabs);

  function applyFilter(key, btn) {
    tabs.querySelectorAll("[role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    moveThumb(btn);
    let shown = 0;
    [...findings.children].forEach((card) => {
      const match = key === "all" || card.dataset.status === key;
      card.hidden = !match;
      if (match) { card.style.setProperty("--i", String(shown++)); card.style.animation = "none"; void card.offsetWidth; card.style.animation = ""; }
    });
    empty.hidden = shown > 0;
  }
  function allOpen() { return [...findings.children].every((c) => c.dataset.open === "true"); }
  function openAll(open) {
    [...findings.children].forEach((c) => setOpen(c, open));
    expandBtn.textContent = open ? "Collapse all" : "Expand all";
  }

  // Run details
  const maxDur = Math.max(1, ...r.trace.map((t) => t.duration_ms));
  const traceRows = r.trace.map((t) => h("tr", {},
    h("td", { text: t.node }),
    h("td", {}, t.detail, h("span", { class: "trace-bar", "data-w": `${(t.duration_ms / maxDur) * 100}` })),
    h("td", { class: "num", text: fmtMs(t.duration_ms) })));
  const details = h("details", { class: "card details" },
    h("summary", {}, h("span", { text: "Run details" }), icon("i-chevron")),
    h("div", { class: "details-body" },
      h("dl", { class: "kv" },
        kv("Task ID", rec.task_id, true), kv("Providers", r.provider_mode, true), kv("Latency", fmtMs(r.latency_ms)),
        kv("Revisions", String(r.revisions)), kv("LLM calls", String(r.usage.llm_calls)),
        kv("Prompt tokens", fmtNum(r.usage.prompt_tokens)), kv("Completion tokens", fmtNum(r.usage.completion_tokens)),
        kv("Embedding tokens", fmtNum(r.usage.embedding_tokens))),
      h("table", { class: "trace-table" },
        h("thead", {}, h("tr", {}, h("th", { text: "Node" }), h("th", { text: "Detail" }), h("th", { text: "Time", style: { textAlign: "right" } }))),
        h("tbody", {}, traceRows))));
  details.addEventListener("toggle", () => {
    if (details.open) requestAnimationFrame(() => details.querySelectorAll(".trace-bar").forEach((b) => { b.style.width = `${b.dataset.w}%`; }));
  });
  els.push(details);

  els.push(h("p", { class: "disclaimer" }, icon("i-shield"), h("span", { text: r.disclaimer })));
  return els;
}

function kv(k, v, wide = false) { return h("div", { class: wide ? "wide" : null }, h("dt", { text: k }), h("dd", { text: v })); }

function setOpen(card, open) {
  card.dataset.open = String(open);
  card.querySelector(".finding-head").setAttribute("aria-expanded", String(open));
}

function findingEl(f, i) {
  const st = STATUS[f.status];
  const bodyId = `finding-${f.requirement_id}-${i}`;
  const evidence = f.supporting_evidence.length
    ? f.supporting_evidence.map((e) => h("figure", { class: "quote" },
      h("blockquote", { text: e.quote }),
      h("figcaption", {},
        h("span", { class: "src" }, icon("i-file"), e.source_file),
        e.section ? [h("span", { class: "sep" }), h("span", { text: e.section })] : null,
        e.page ? [h("span", { class: "sep" }), h("span", { text: `page ${e.page}` })] : null,
        h("span", { class: "sep" }), h("span", { class: "chunk", title: "Chunk ID", text: e.chunk_id }))))
    : [h("p", { class: "no-evidence", text: "No verified supplier evidence was found for this requirement." })];

  const lists = [];
  if (f.missing_information.length) lists.push(h("div", {}, h("p", { class: "section-label", text: "Missing information" }), h("ul", { class: "bullets" }, f.missing_information.map((m) => h("li", { text: m })))));
  if (f.recommended_follow_up.length) lists.push(h("div", {}, h("p", { class: "section-label", text: "Recommended follow-up" }), h("ul", { class: "bullets" }, f.recommended_follow_up.map((m) => h("li", { text: m })))));

  const reqId = h("span", { class: "req-id", text: f.requirement_id });
  const head = h("button", { class: "finding-head", type: "button", "aria-controls": bodyId },
    reqId,
    h("span", { class: "req-text", text: f.requirement }),
    h("span", { class: `badge ${st.cls}` }, icon(st.icon), st.label),
    icon("i-chevron", "ic chev"));

  const card = h("article", { class: `finding ${st.cls}`, "data-status": f.status, style: { "--i": String(i) } },
    head,
    h("div", { class: "finding-body", id: bodyId },
      h("div", { class: "finding-body-inner" },
        h("div", { class: "finding-content" },
          h("p", { class: "explanation", text: f.explanation }),
          h("div", {}, h("p", { class: "section-label", text: "Evidence" }), evidence),
          lists.length ? h("div", { class: "two-col" }, lists) : null,
          f.review_notes.length
            ? h("div", { class: "review-notes" }, h("strong", { text: "Evidence reviewer" }), h("ul", {}, f.review_notes.map((n) => h("li", { text: n }))))
            : null))));
  head.addEventListener("click", () => setOpen(card, card.dataset.open !== "true"));
  // Gaps and insufficient evidence need attention, so they start expanded.
  setOpen(card, f.status !== "met");
  requestAnimationFrame(() => card.style.setProperty("--id-w", `${reqId.offsetWidth}px`));
  return card;
}

function exportJson(rec) {
  const blob = new Blob([JSON.stringify(rec, null, 2)], { type: "application/json" });
  const a = h("a", { href: URL.createObjectURL(blob), download: `evaluation-${rec.report?.supplier_id || "task"}-${rec.task_id.slice(0, 8)}.json` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  initTheme();
  initKeyDialog();
  initDropzone();
  initSeg($("#cat-seg"), (v) => {
    state.category = v;
    $("#supplier-field").hidden = v !== "supplier";
  });
  initSeg($("#req-seg"), (v) => {
    state.reqMode = v;
    $("#req-wrap").dataset.open = String(v === "custom");
    if (v === "custom") setTimeout(() => $("#req-input").focus(), 200);
  });
  $("#supplier-input").addEventListener("input", (e) => e.target.removeAttribute("aria-invalid"));
  $("#supplier-select").addEventListener("change", updateRunButton);
  $("#index-btn").addEventListener("click", indexPending);
  $("#run-btn").addEventListener("click", runEvaluation);
  $("#sample-btn").addEventListener("click", loadSamples);
  renderHistory();

  const ok = await loadHealth();
  if (!ok) { toast("error", "Server unreachable", "Start the API with uvicorn and reload.", 0); return; }
  if (state.health.auth_enabled && !state.apiKey) openKeyDialog();
  await Promise.all([refreshDocs(), initSamples()]);
}

boot();
