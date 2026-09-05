"use strict";
const $ = (id) => document.getElementById(id);
let last = null;

/* ================= grid layout: snap, push-apart, persist ================= */
const COLS = 12, ROW_H = 84, GAP = 12;
const LAYOUT_KEY = "simpipe_grid_v4";
const UNLOCK_KEY = "simpipe_layout_unlocked";
// Fixed layout: config + partition/placement editor on row 1, summary as a
// full-width strip between them and the gantt, then per-rank stats + yaml.
const DEFAULT_GRID = {
  "panel-config":    { x: 0, y: 0, w: 5, h: 6 },
  "panel-partplace": { x: 5, y: 0, w: 7, h: 6 },
  "panel-summary":   { x: 0, y: 6, w: 12, h: 2 },
  "panel-gantt":     { x: 0, y: 8, w: 12, h: 5 },
  "panel-ranks":     { x: 0, y: 13, w: 7, h: 4 },
  "panel-yaml":      { x: 7, y: 13, w: 5, h: 4 },
};
const MIN_W = 2, MIN_H = 2;
let unlocked = localStorage.getItem(UNLOCK_KEY) === "1";
let grid = unlocked ? loadGrid() : JSON.parse(JSON.stringify(DEFAULT_GRID));

function loadGrid() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY));
    if (saved && Object.keys(DEFAULT_GRID).every(k => saved[k])) return saved;
  } catch {}
  return JSON.parse(JSON.stringify(DEFAULT_GRID));
}
const saveGrid = () => { if (unlocked) localStorage.setItem(LAYOUT_KEY, JSON.stringify(grid)); };
const colW = () => ($("board").clientWidth - GAP) / COLS;

function rectPx(item) {
  const cw = colW();
  return {
    left: item.x * cw + GAP, top: item.y * ROW_H + GAP,
    width: item.w * cw - GAP, height: item.h * ROW_H - GAP,
  };
}
function applyGrid() {
  for (const [id, item] of Object.entries(grid)) {
    const el = $(id);
    if (!el) continue;
    const r = rectPx(item);
    el.style.left = r.left + "px"; el.style.top = r.top + "px";
    el.style.width = r.width + "px"; el.style.height = r.height + "px";
  }
}
const overlaps = (a, b) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/* Push non-pinned panels below whatever they overlap, then compact. */
function resolveCollisions(pinnedId) {
  const ids = Object.keys(grid);
  let guard = 0;
  let changed = true;
  while (changed && guard++ < 200) {
    changed = false;
    for (const id of ids) {
      if (id === pinnedId) continue;
      for (const other of ids) {
        if (other === id) continue;
        if (overlaps(grid[id], grid[other])) {
          grid[id].y = grid[other].y + grid[other].h;
          changed = true;
        }
      }
    }
  }
  compact(pinnedId);
}
/* Gravity: move panels up while space is free (pinned stays put). */
function compact(pinnedId) {
  const ids = Object.keys(grid).sort((a, b) => grid[a].y - grid[b].y || grid[a].x - grid[b].x);
  for (const id of ids) {
    if (id === pinnedId) continue;
    const item = grid[id];
    while (item.y > 0) {
      const probe = { ...item, y: item.y - 1 };
      if (Object.keys(grid).some(o => o !== id && overlaps(probe, grid[o]))) break;
      item.y -= 1;
    }
  }
}

function startDrag(panel, ev) {
  if (!unlocked) return;
  ev.preventDefault();
  const id = panel.id;
  const startX = ev.clientX, startY = ev.clientY;
  const orig = { ...grid[id] };
  const ghost = $("ghost");
  panel.classList.add("moving");
  let moved = false;
  function move(e) {
    moved = true;
    const cw = colW();
    const dx = e.clientX - startX, dy = e.clientY - startY;
    // free-floating pixel position for the dragged panel
    const px = orig.x * cw + GAP + dx, py = orig.y * ROW_H + GAP + dy;
    panel.style.left = px + "px"; panel.style.top = py + "px";
    // snapped target cell
    const nx = Math.max(0, Math.min(COLS - orig.w, Math.round((px - GAP) / cw)));
    const ny = Math.max(0, Math.round((py - GAP) / ROW_H));
    if (grid[id].x !== nx || grid[id].y !== ny) {
      grid[id].x = nx; grid[id].y = ny;
      resolveCollisions(id);
      applyGridExcept(id);
    }
    const r = rectPx(grid[id]);
    Object.assign(ghost.style, { display: "block", left: r.left + "px", top: r.top + "px", width: r.width + "px", height: r.height + "px" });
  }
  function up() {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    panel.classList.remove("moving");
    ghost.style.display = "none";
    if (moved) { resolveCollisions(id); }
    applyGrid(); saveGrid();
  }
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
}
function applyGridExcept(skipId) {
  for (const [id, item] of Object.entries(grid)) {
    if (id === skipId) continue;
    const el = $(id), r = rectPx(item);
    el.style.left = r.left + "px"; el.style.top = r.top + "px";
    el.style.width = r.width + "px"; el.style.height = r.height + "px";
  }
}
function startResize(panel, ev) {
  if (!unlocked) return;
  ev.preventDefault(); ev.stopPropagation();
  const id = panel.id;
  const startX = ev.clientX, startY = ev.clientY;
  const orig = { ...grid[id] };
  panel.classList.add("moving");
  function move(e) {
    const cw = colW();
    const nw = Math.max(MIN_W, Math.min(COLS - orig.x, Math.round(orig.w + (e.clientX - startX) / cw)));
    const nh = Math.max(MIN_H, Math.round(orig.h + (e.clientY - startY) / ROW_H));
    if (grid[id].w !== nw || grid[id].h !== nh) {
      grid[id].w = nw; grid[id].h = nh;
      resolveCollisions(id);
      applyGrid();
    }
  }
  function up() {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    panel.classList.remove("moving");
    resolveCollisions(id); applyGrid(); saveGrid();
  }
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
}
for (const panel of document.querySelectorAll("[data-panel]")) {
  panel.querySelector(".panel-head").addEventListener("mousedown", (ev) => {
    if (ev.target.closest("button, input, select, label, .dual")) return;
    startDrag(panel, ev);
  });
  panel.querySelector("[data-resize]").addEventListener("mousedown", (ev) => startResize(panel, ev));
}
window.addEventListener("resize", applyGrid);
function applyLock() {
  document.body.classList.toggle("locked", !unlocked);
  $("layout-unlock").checked = unlocked;
}
$("layout-unlock").addEventListener("change", () => {
  unlocked = $("layout-unlock").checked;
  localStorage.setItem(UNLOCK_KEY, unlocked ? "1" : "0");
  // locked = the fixed default arrangement; unlocked = last custom layout
  grid = unlocked ? loadGrid() : JSON.parse(JSON.stringify(DEFAULT_GRID));
  applyLock();
  applyGrid();
});
applyLock();
applyGrid();
$("reset-layout").addEventListener("click", () => {
  localStorage.removeItem(LAYOUT_KEY);
  grid = JSON.parse(JSON.stringify(DEFAULT_GRID));
  applyGrid();
});

/* ================= config form view ================= */
/* Schema of adjustable fields.  desc doubles as the hover tooltip; def is the
   engine default (shown as placeholder); min/max/itemMin are hard limits
   enforced on input (out-of-range edits are rejected and marked red). */
/* engine schedule value -> display name */
const SCHED_LABELS = { "1f1b": "1F1B", zbh: "ZBH", interleaved: "Interleaved",
  octopipe: "OctoPipe", recycle: "ReCycle", bapar: "Mist", afab: "AFAB" };
const schedLabel = (s) => SCHED_LABELS[s] || s;

const CFG_SCHEMA = [
  { sec: "General", fields: [
    { path: "schedule", type: "select",
      options: ["1f1b", "zbh", "interleaved", "octopipe", "recycle", "bapar", "afab"],
      labels: SCHED_LABELS, def: "1f1b",
      desc: "Pipeline schedule. OctoPipe enables partition / placement / order tuning." },
    { path: "time_limit", type: "int", def: 1000000, min: 1, max: 1e12,
      desc: "Simulation tick budget in 0.01 ms units; the run reports STALLED when exceeded." },
  ]},
  { sec: "Model", fields: [
    { path: "model.name", type: "dselect", optsKey: "models", def: "mock_model", noEmpty: true,
      desc: "Profiled model (has profiles/<name>.json) or mock_model for synthetic timings. Custom names (used with profile_times_path) can be set in the YAML view." },
    { path: "model.num_layers", type: "int", def: 32, min: 1, max: 4096,
      desc: "Transformer body layer count (embedding/head excluded)." },
    { path: "model.pattern", type: "text", wide: true, ph: "ET*32L",
      desc: "Layer pattern: E embedding, L head, body types M mamba / * attn / - MLP / T transformer / # MoE; X*N repeats X N times. Editable for mock_model (num_layers follows the pattern; raising num_layers pads T). Profiled models show their pattern read-only." },
    { path: "model.layer_time", type: "float", min: 0.01, max: 1e9, ph: "e.g. 100 (= 1 ms)", mockOnly: true,
      desc: "Mock timing: uniform per-layer duration in 0.01 ms ticks, F = B = W. Embedding/head cost 0." },
    { path: "model.layer_f_time", type: "float", min: 0.01, max: 1e9, mockOnly: true,
      desc: "Mock timing: forward duration override (0.01 ms ticks)." },
    { path: "model.layer_b_time", type: "float", min: 0.01, max: 1e9, mockOnly: true,
      desc: "Mock timing: backward duration override; defaults to the forward time." },
    { path: "model.layer_w_time", type: "float", min: 0.01, max: 1e9, mockOnly: true,
      desc: "Mock timing: weight-update duration override; defaults to the forward time." },
  ]},
  { sec: "Parallel", fields: [
    { path: "parallel.pp_size", type: "int", def: 1, min: 1, max: 1024,
      desc: "Pipeline-parallel size = number of devices." },
    { path: "parallel.tp_size", type: "int", def: 1, min: 1, max: 64,
      desc: "Tensor-parallel size; scales analytic timing and per-rank model/activation memory." },
    { path: "parallel.ep_size", type: "int", def: 1, min: 1, max: 512,
      desc: "Expert-parallel size for MoE models; shards experts across ranks." },
    { path: "parallel.dp_size", type: "int", def: 1, min: 1, max: 4096,
      desc: "Data-parallel size; with ZeRO it shards optimizer/gradient state in the memory estimate." },
    { path: "parallel.zero_stage", type: "int", def: 1, min: 0, max: 3,
      desc: "ZeRO stage (0-3) used by the per-rank model-state memory estimate." },
    { path: "model.seq_len", type: "int", def: 4096, min: 1, max: 16777216,
      desc: "Reference sequence length of the profiled shape; varlen batch scales relative to it." },
    { path: "parallel.micro_batch_num", type: "int", def: 8, min: 1, max: 16384,
      desc: "Microbatches per iteration. Derived from batch.microbatches / batch.time_scales when those are set." },
    { path: "model.micro_batch_size", type: "int", def: 1, min: 1, max: 65536,
      desc: "Reference microbatch size of the profiled shape." },
    { path: "parallel.bwd_split", type: "bool", desc: "Split backward into B (grad-input) and W (grad-weight) workloads (zero-bubble style)." },
    { path: "model.recompute", type: "bool", desc: "Full activation recompute: each backward re-runs the forward first." },
    { path: "parallel.chunk_num", type: "int", min: 1, max: 256, ph: "auto",
      desc: "Virtual-pipeline chunks per device; empty = auto (interleaved: max, else 1)." },
  ]},
  { sec: "Batch: Variable-length microbatches", id: "batch", fields: [
    { path: "batch.mode", type: "select", options: ["", "pack", "pad"], emptyLabel: "Off",
      desc: "pack: concat sequences, varlen kernels (linear ~ sum(len), attention ~ sum(len^2)); pad: pad to the longest sequence (linear ~ n*max, attention ~ n*max^2). Requires exactly one of microbatches / time_scales below." },
    { path: "batch.microbatches", type: "lines", wide: true, itemMin: 1, itemMax: 16777216,
      ph: "4096\n2048, 2048\n...  (one microbatch per line; line count = micro_batch_num)",
      desc: "Sequence lengths per microbatch, one microbatch per line. The line count must equal parallel.micro_batch_num (or leave micro_batch_num empty to derive it)." },
    { path: "batch.time_scales", type: "floatlist", wide: true, itemMin: 1e-6, itemMax: 1e9,
      ph: "1, 1.2, 1.8, ...  (count = micro_batch_num)",
      desc: "Direct per-microbatch compute multipliers. Count must equal parallel.micro_batch_num. With time_ref the values are absolute times. Mutually exclusive with microbatches." },
    { path: "batch.time_ref", type: "float", def: 1.0, min: 1e-6, max: 1e12,
      desc: "Reference for absolute time_scales: scale = value / time_ref." },
  ]},
  { sec: "Tuning", fields: [
    { path: "tuning.auto_tune", type: "bool", def: false,
      desc: "Search partition / placement / chunking. Off unless enabled; locked on while schedule = octopipe." },
    { path: "tuning.batch_order_tune", type: "bool", def: false,
      desc: "Search the microbatch execution order for variable batches. Off unless enabled." },
    { path: "tuning.batch_order_max_sims", type: "int", def: 64, min: 1, max: 100000,
      desc: "Simulation budget for the order search." },
    { path: "tuning.max_inflight_layers", type: "int", min: 1, max: 1000000, ph: "unlimited",
      desc: "Activation cap: max in-flight layer*microbatch units per device before F admission blocks." },
  ]},
  { sec: "Hardware", fields: [
    { path: "hardware.gpu_peak_tflops", type: "float", def: 312.0, min: 0.1, max: 100000,
      desc: "Peak TFLOPs per GPU; drives analytic timing when profiled data is off." },
    { path: "hardware.gpu_hbm_gb", type: "float", def: 80.0, min: 1, max: 8192,
      desc: "HBM capacity per GPU used by the memory feasibility check." },
    { path: "hardware.intra_node_bw_gbps", type: "float", def: 600.0, min: 0.1, max: 100000,
      desc: "Intra-node bandwidth in GB/s for communication estimates." },
    { path: "hardware.workload_overhead_ms", type: "float", def: 0.0, min: 0, max: 1000,
      desc: "Constant launch overhead in ms added to every F/B/W workload (dispatch, bookkeeping, chunk switch)." },
    { path: "hardware.p2p_latency_ms", type: "float", def: 0.0, min: 0, max: 10000,
      desc: "Extra latency in ms on every cross-device dependency edge (P2P transfer)." },
    { path: "hardware.comp_power", type: "float", def: 1.0, min: 0.01, max: 1000,
      desc: "Relative compute speed multiplier applied to workload durations." },
  ]},
];

let cfgObj = {};
/* dropdown option lists fetched from /api/options (models, profile files) */
let DYN_OPTS = { models: [], profile_paths: [], model_meta: {} };
async function fetchOptions() {
  try { DYN_OPTS = await (await fetch("/api/options")).json(); } catch {}
}

/* Switching the model fills its known config values into model.*:
   preset metadata + num_layers from the profiled pattern.  Keys that
   describe the model are replaced; user intent (recompute, paths) stays. */
const MODEL_INTRINSIC_KEYS = ["hidden_size", "num_layers", "num_attention_heads",
  "seq_len", "vocab_size", "micro_batch_size", "intermediate_size",
  "use_moe", "num_experts", "top_k"];
const MOCK_TIME_KEYS = ["layer_time", "layer_f_time", "layer_b_time", "layer_w_time",
  "pattern", "forward_ms", "backward_ms", "weight_ms"];
function applyModelMeta(name) {
  if (!cfgObj.model || typeof cfgObj.model !== "object") cfgObj.model = {};
  const m = cfgObj.model;
  for (const k of MODEL_INTRINSIC_KEYS) delete m[k];
  Object.assign(m, (DYN_OPTS.model_meta || {})[name] || {});
  if (name === "mock_model") {
    if (MOCK_TIME_KEYS.every(k => m[k] === undefined)) {
      m.pattern = "ET*32L";
      m.forward_ms = { T: 1.0 };
      m.num_layers = 32;
    }
  } else {
    for (const k of MOCK_TIME_KEYS) delete m[k];
  }
  alignSource = ""; // model switched: the mock no longer mirrors a preset
  // partition/placement belong to the previous model's layer count:
  // drop manual arrays and the last-run preview so the editor re-splits.
  delete cfgObj.partition_layers;
  delete cfgObj.placement;
  ppLastRun = null;
  materializeDefaults(); // fill engine defaults for keys the model has no data for
  refreshFormValues();
  scheduleDump();
}

const getPath = (obj, path) => {
  let cur = obj;
  for (const k of path.split(".")) {
    if (cur === null || typeof cur !== "object") return undefined;
    cur = cur[k];
  }
  return cur;
};
function setPath(obj, path, val) {
  const keys = path.split(".");
  if (val === undefined) {
    const stack = [];
    let cur = obj;
    for (const k of keys.slice(0, -1)) {
      if (!cur[k] || typeof cur[k] !== "object") return;
      stack.push([cur, k]); cur = cur[k];
    }
    delete cur[keys[keys.length - 1]];
    for (let i = stack.length - 1; i >= 0; i--) {
      const [parent, k] = stack[i];
      if (Object.keys(parent[k]).length === 0) delete parent[k];
    }
  } else {
    let cur = obj;
    for (const k of keys.slice(0, -1)) {
      if (!cur[k] || typeof cur[k] !== "object") cur[k] = {};
      cur = cur[k];
    }
    cur[keys[keys.length - 1]] = val;
  }
}

/* text widget value -> config value; throws on malformed or out-of-range input */
function decodeField(f, el) {
  // bools with an explicit default stay in the config as true/false
  if (f.type === "bool") return el.checked ? true : (f.def !== undefined ? false : undefined);
  const raw = (el.value || "").trim();
  if (raw === "") return undefined;
  const wantInt = f.type === "int" || f.type === "intlist" || f.type === "lines";
  const num = (s, lo, hi) => {
    const v = Number(s);
    if (!Number.isFinite(v)) throw new Error(`bad number: ${s}`);
    if (wantInt && !Number.isInteger(v)) throw new Error(`not an integer: ${s}`);
    if (lo !== undefined && v < lo) throw new Error(`${v} < min ${lo}`);
    if (hi !== undefined && v > hi) throw new Error(`${v} > max ${hi}`);
    return v;
  };
  // list inputs tolerate YAML-style brackets and trailing commas:
  // "13, 13", "[13, 13]", "[[0], [1]]" and one-line-per-entry all parse.
  const items = (s) => s.replace(/[\[\]]/g, "").split(",")
    .map(x => x.trim()).filter(x => x !== "");
  switch (f.type) {
    case "int": case "float": return num(raw, f.min, f.max);
    case "select": case "dselect": case "text": return raw;
    case "intlist": case "floatlist": {
      const vals = items(raw).map(s => num(s, f.itemMin, f.itemMax));
      return vals.length ? vals : undefined;
    }
    case "lines": {
      const text = raw.includes("]") ? raw.replace(/\]\s*,?/g, "]\n") : raw;
      const rows = text.split("\n").map(l => l.trim()).filter(Boolean)
        .map(l => items(l).map(s => num(s, f.itemMin, f.itemMax)))
        .filter(row => row.length);
      return rows.length ? rows : undefined;
    }
  }
  return raw;
}
function encodeField(f, el, val) {
  if (f.type === "bool") { el.checked = val === true; return; }
  if (f.type === "select" || f.type === "dselect") {
    const v = val === undefined || val === null ? "" : String(val);
    // keep custom values (from YAML/imported configs) selectable
    if (v && !Array.from(el.options).some(o => o.value === v)) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = `${v} (Custom)`;
      el.appendChild(opt);
    }
    el.value = v;
    return;
  }
  if (val === undefined || val === null) { el.value = ""; return; }
  switch (f.type) {
    case "intlist": case "floatlist":
      el.value = Array.isArray(val) ? val.join(", ") : String(val); break;
    case "lines":
      el.value = Array.isArray(val) ? val.map(r => Array.isArray(r) ? r.join(", ") : String(r)).join("\n") : String(val); break;
    default: el.value = String(val);
  }
}

/* "pp_size" -> "PP size", "gpu_hbm_gb" -> "GPU HBM (GB)" */
const LABEL_ACRONYMS = { pp: "PP", tp: "TP", ep: "EP", dp: "DP", gpu: "GPU",
  hbm: "HBM", p2p: "P2P", bw: "BW", f: "F", b: "B", w: "W" };
/* trailing unit words render in parentheses */
const LABEL_UNITS = { gb: "(GB)", ms: "(ms)", gbps: "(GB/s)", tflops: "(TFLOPs)" };
function labelize(name) {
  const words = name.split("_");
  let unit = "";
  if (words.length > 1 && LABEL_UNITS[words[words.length - 1]])
    unit = " " + LABEL_UNITS[words.pop()];
  const out = words.map(w => LABEL_ACRONYMS[w] || w);
  const first = out[0];
  if (!(first in LABEL_ACRONYMS) && first)
    out[0] = first[0].toUpperCase() + first.slice(1);
  return out.join(" ") + unit;
}

function buildForm() {
  const root = $("config-form");
  for (const sec of CFG_SCHEMA) {
    const box = document.createElement("div");
    box.className = "form-sec";
    box.innerHTML = `<h4>${sec.sec}</h4>`;
    const fgrid = document.createElement("div");
    fgrid.className = "fgrid";
    for (const f of sec.fields) {
      const row = document.createElement("div");
      row.className = f.wide ? "frow wide" : "frow";
      if (f.mockOnly) row.dataset.mockonly = "1"; // hidden unless mock_model
      row.title = f.desc; // hover shows the comment
      const name = labelize(f.path.split(".").pop());
      const ph = f.ph !== undefined ? f.ph : (f.def !== undefined ? String(f.def) : "");
      const phAttr = ph ? ` placeholder="${String(ph).replace(/"/g, "&quot;")}"` : "";
      let ctl;
      if (f.type === "bool") ctl = `<input type="checkbox" data-path="${f.path}">`;
      else if (f.type === "select" || f.type === "dselect") {
        const opts = f.type === "dselect"
          ? (f.noEmpty ? [] : [""]).concat(DYN_OPTS[f.optsKey] || []) : f.options;
        ctl = `<select data-path="${f.path}">${opts.map(o =>
          `<option value="${o}">${(f.labels && f.labels[o]) || o || f.emptyLabel || "Default"}</option>`).join("")}</select>`;
      }
      else if (f.type === "lines")
        ctl = `<textarea rows="2" data-path="${f.path}" spellcheck="false"${phAttr}></textarea>`;
      else if (f.type === "int" || f.type === "float") {
        const step = f.type === "int" ? "1" : "any";
        const lim = (f.min !== undefined ? ` min="${f.min}"` : "") + (f.max !== undefined ? ` max="${f.max}"` : "");
        ctl = `<input type="number" step="${step}"${lim} data-path="${f.path}" spellcheck="false"${phAttr}>`;
      } else {
        ctl = `<input type="text" data-path="${f.path}" spellcheck="false"${phAttr}>`;
      }
      row.innerHTML = `<label title="${f.desc.replace(/"/g, "&quot;")}">${name}</label>${ctl}`;
      fgrid.appendChild(row);
    }
    box.appendChild(fgrid);
    if (sec.sec === "Model") {
      const mt = document.createElement("div");
      mt.id = "model-times"; // per-layer-type f/b/w table, filled dynamically
      box.appendChild(mt);
    }
    if (sec.id === "batch") {
      const warn = document.createElement("div");
      warn.className = "form-warn";
      warn.id = "batch-warn";
      box.appendChild(warn);
    }
    root.appendChild(box);
  }
  const fields = {};
  for (const sec of CFG_SCHEMA) for (const f of sec.fields) fields[f.path] = f;
  root.addEventListener("input", (ev) => onFormEdit(ev, fields));
  root.addEventListener("change", (ev) => onFormEdit(ev, fields));
}
function onFormEdit(ev, fields) {
  const el = ev.target;
  const f = fields[el.dataset && el.dataset.path];
  if (!f) return;
  // selects and checkboxes fire both "input" and "change" for one edit;
  // handle only "change" so side effects (applyModelMeta) run exactly once.
  if (ev.type === "input" && (el.tagName === "SELECT" || el.type === "checkbox")) return;
  try {
    const val = decodeField(f, el);
    if (f.path === "model.pattern" && val !== undefined && !PAT_CHARS.test(expandPat(val)))
      throw new Error("pattern may only use E M * - T # L and X*N repeats");
    setPath(cfgObj, f.path, val);
    el.classList.remove("invalid");
    el.title = f.desc;
    if (f.path === "model.name") applyModelMeta(val); // fill the model's values
    if (f.path === "schedule" && val === "octopipe") {
      // OctoPipe schedules B and W separately: switching to it turns bwd
      // split on (still user-editable afterwards).
      setPath(cfgObj, "parallel.bwd_split", true);
      const bs = document.querySelector('[data-path="parallel.bwd_split"]');
      if (bs) bs.checked = true;
    }
    if (f.path === "model.pattern") {
      // pattern is the source of truth for the layer count
      alignSource = ""; // structure changed: no longer mirrors a preset
      const n = val === undefined ? undefined : patBody(val).length;
      if (n) {
        setPath(cfgObj, "model.num_layers", n);
        const nl = document.querySelector('[data-path="model.num_layers"]');
        if (nl) nl.value = String(n);
      }
      ensurePatternTimes();
    }
    if (f.path === "model.num_layers" && cfgObj.model && cfgObj.model.pattern
        && val !== undefined) {
      // pad with T before the head / trim body symbols from the end
      let body = patBody(cfgObj.model.pattern);
      if (val > body.length) body = body.concat(Array(val - body.length).fill("T"));
      else if (val < body.length) body = body.slice(0, val);
      const exp = expandPat(cfgObj.model.pattern);
      const compact = compressPat(
        (exp.includes("E") ? "E" : "") + body.join("") + (exp.includes("L") ? "L" : ""));
      cfgObj.model.pattern = compact;
      const pe = document.querySelector('[data-path="model.pattern"]');
      if (pe) pe.value = compact;
      ensurePatternTimes();
    }
    if (f.path === "parallel.ep_size") {
      // EP shards the data-parallel group, so DP must be >= EP
      const ep = getPath(cfgObj, "parallel.ep_size") || 1;
      const dp = getPath(cfgObj, "parallel.dp_size") || 1;
      if (dp < ep) {
        setPath(cfgObj, "parallel.dp_size", ep);
        const dpEl = document.querySelector('[data-path="parallel.dp_size"]');
        if (dpEl) dpEl.value = String(ep);
      }
    }
    scheduleDump();
    scheduleAutoRun();
  } catch (err) {
    el.classList.add("invalid");
    el.title = `${err.message}\n\n${f.desc}`;
  }
  validateBatchLive();
  syncTuningLock();
  renderPartPlace();
  renderModelTimes();
}
function refreshFormValues() {
  for (const el of $("config-form").querySelectorAll("[data-path]")) {
    const path = el.dataset.path;
    const f = CFG_SCHEMA.flatMap(s => s.fields).find(x => x.path === path);
    encodeField(f, el, getPath(cfgObj, path));
    el.classList.remove("invalid");
    el.title = f.desc;
  }
  validateBatchLive();
  syncTuningLock();
  renderPartPlace();
  renderModelTimes();
}

/* ============ partition / placement visual editor ============
   Rank rows contain stage boxes; stage boxes contain layer tiles.
   Drag a layer tile onto another stage -> repartition.
   Drag a stage chip onto another rank row -> re-placement.
   Any edit writes partition_layers / placement into the config; without
   explicit arrays the panel previews the last run (or a uniform split). */
let ppDrag = null;     // {kind: "layer"|"stage", sid}
let ppLastRun = null;  // {part, place} from the most recent run

const ppSize = () => getPath(cfgObj, "parallel.pp_size") || 1;

function ppStagesFromCfg() {
  const part = Array.isArray(cfgObj.partition_layers) && cfgObj.partition_layers.length
    ? cfgObj.partition_layers : null;
  const placeRaw = Array.isArray(cfgObj.placement) && cfgObj.placement.length
    ? cfgObj.placement : null;
  if (!part && !placeRaw) return null;
  const rankOf = {};
  (placeRaw || []).forEach((sids, r) =>
    (Array.isArray(sids) ? sids : [sids]).forEach(x => { rankOf[x] = r; }));
  const sids = Object.keys(rankOf).map(Number);
  const nStages = part ? part.length : (sids.length ? Math.max(...sids) + 1 : ppSize());
  const L = getPath(cfgObj, "model.num_layers") || 32;
  return Array.from({ length: nStages }, (_, sid) => ({
    sid,
    layers: part ? Number(part[sid]) || 0
      : Math.floor(L / nStages) + (sid < L % nStages ? 1 : 0),
    rank: rankOf[sid] !== undefined ? rankOf[sid] : sid % ppSize(),
  }));
}
function ppStagesPreview() {
  // ignore a cached run whose layer sum no longer matches the model
  const L0 = getPath(cfgObj, "model.num_layers");
  if (ppLastRun && L0 !== undefined && Array.isArray(ppLastRun.part)
      && ppLastRun.part.reduce((a, b) => a + (Number(b) || 0), 0) !== L0)
    ppLastRun = null;
  if (ppLastRun && Array.isArray(ppLastRun.part) && ppLastRun.part.length) {
    const rankOf = {};
    (ppLastRun.place || []).forEach((sids, r) =>
      (Array.isArray(sids) ? sids : [sids]).forEach(x => { rankOf[x] = r; }));
    return ppLastRun.part.map((n, sid) => ({
      sid, layers: Number(n) || 0,
      rank: rankOf[sid] !== undefined ? rankOf[sid] : sid % ppSize(),
    }));
  }
  const pp = ppSize(), L = getPath(cfgObj, "model.num_layers") || 32;
  const base = Math.floor(L / pp);
  return Array.from({ length: pp }, (_, sid) => ({
    sid, layers: base + (sid < L % pp ? 1 : 0), rank: sid,
  }));
}

function ppCommit(stages) {
  stages.sort((a, b) => a.sid - b.sid);
  const nRanks = Math.max(ppSize(), ...stages.map(s => s.rank + 1));
  const place = Array.from({ length: nRanks }, () => []);
  for (const s of stages) place[Math.min(s.rank, nRanks - 1)].push(s.sid);
  setPath(cfgObj, "partition_layers", stages.map(s => s.layers));
  setPath(cfgObj, "placement", place);
  scheduleDump();
  scheduleAutoRun();
  renderPartPlace();
}

/* pattern symbol -> css class + readable name (matches simpipe.models.pattern) */
const SYM_INFO = { M: ["mamba", "mamba"], "*": ["attn", "attention"],
  "-": ["mlp", "MLP"], T: ["transformer", "transformer"], "#": ["moe", "MoE"],
  E: ["embed", "embedding"], L: ["head", "head"] };

/* run-length pattern syntax: "ET*32L" <-> "E" + "T"*32 + "L".
   '*' followed by digits repeats the previous char; a lone '*' is attention. */
const expandPat = (s) => s.replace(/(.)\*(\d+)/g, (m, c, n) => c.repeat(+n));
const compressPat = (s) => s.replace(/((.)\2{2,})/g, (m, run, c) => `${c}*${run.length}`);
const PAT_CHARS = /^[EML\-*T#]*$/;
const patBody = (s) => expandPat(s).split("").filter(c => c !== "E" && c !== "L");

/* Effective per-symbol ms tables of a mock pattern config (engine defaults:
   backward = forward, weight = backward, E/L = 0). */
function mockTables(m) {
  const f = { E: 0, L: 0, ...(m.forward_ms || {}) };
  const b = { ...f, ...(m.backward_ms || {}) };
  const w = { ...b, ...(m.weight_ms || {}) };
  return { f, b, w };
}

/* Every body type in the mock pattern needs a forward time or the engine
   rejects the config; give new types (e.g. padded T) a 1 ms default. */
function ensurePatternTimes() {
  const m = cfgObj.model || {};
  if (m.name !== "mock_model" || !m.pattern) return;
  const exp = expandPat(m.pattern);
  if (!PAT_CHARS.test(exp)) return;
  for (const sym of new Set(patBody(m.pattern))) {
    if ((m.forward_ms || {})[sym] === undefined) {
      if (!m.forward_ms) m.forward_ms = {};
      m.forward_ms[sym] = 1.0;
    }
  }
}

/* Layer detail of the selected model: body symbols (E/L stripped) and
   per-symbol f/b/w times in ms.  null when nothing is known. */
function ppLayerInfo() {
  const m = cfgObj.model || {};
  if (m.pattern && PAT_CHARS.test(expandPat(m.pattern))) {
    const t = mockTables(m);
    return { body: patBody(m.pattern), ...t };
  }
  const lm = (DYN_OPTS.model_layers || {})[getPath(cfgObj, "model.name")];
  if (lm && lm.pattern) {
    return { body: lm.pattern.split("").filter(c => c !== "E" && c !== "L"),
             f: lm.f || {}, b: lm.b || {}, w: lm.w || {} };
  }
  // legacy mock timing: uniform per-layer ticks (0.01 ms) from the config
  const ft = m.layer_f_time !== undefined ? m.layer_f_time : m.layer_time;
  if (ft !== undefined) {
    const bt = m.layer_b_time !== undefined ? m.layer_b_time : ft;
    const wt = m.layer_w_time !== undefined ? m.layer_w_time : ft;
    return { body: null, uniform: { f: ft / 100, b: bt / 100, w: wt / 100 } };
  }
  return null;
}
const fmtMs = (x) => x === undefined || x === null ? "?" : (+x).toFixed(2);
function symTimesText(info, sym) {
  if (info.uniform)
    return `F ${fmtMs(info.uniform.f)} · B ${fmtMs(info.uniform.b)} · W ${fmtMs(info.uniform.w)} ms`;
  return `F ${fmtMs(info.f[sym])} · B ${fmtMs(info.b[sym])} · W ${fmtMs(info.w[sym])} ms`;
}

/* Model section: per-layer-type F/B/W table for the selected model.
   mock_model: editable ms inputs per type plus an "align" preset loader;
   profiled models: read-only values; legacy uniform mock: a note. */
function typeCounts(pattern) {
  const counts = {}, order = [];
  for (const c of pattern) {
    if (!(c in counts)) order.push(c);
    counts[c] = (counts[c] || 0) + 1;
  }
  return { counts, order };
}

function renderModelTimes() {
  const el = document.getElementById("model-times");
  if (!el) return;
  const name = getPath(cfgObj, "model.name");
  const isMock = name === "mock_model";
  const m = cfgObj.model || {};
  const hasPattern = !!(m.pattern && PAT_CHARS.test(expandPat(m.pattern)));
  // legacy uniform mock fields only matter without a pattern
  for (const row of document.querySelectorAll('[data-mockonly]'))
    row.style.display = isMock && !hasPattern ? "" : "none";
  // real (profiled) models: every model property comes from the profile,
  // so all Model fields except the name selector are locked.
  const pe = document.querySelector('[data-path="model.pattern"]');
  if (pe) {
    const lmSel = (DYN_OPTS.model_layers || {})[name];
    if (!isMock) pe.value = lmSel && lmSel.pattern ? compressPat(lmSel.pattern) : "";
    setLocked(pe, !isMock, "Locked: the pattern comes from the model profile. " +
      "Select mock_model to edit it.", fieldDesc("model.pattern"));
  }
  const nlEl = document.querySelector('[data-path="model.num_layers"]');
  if (nlEl) setLocked(nlEl, !isMock, "Locked: the layer count comes from the " +
    "model profile. Select mock_model to edit it.", fieldDesc("model.num_layers"));

  const head = `<div class="mt-row mt-head"><span></span><span>Type</span>
      <span>Count</span><span>F <i class="unit">(ms)</i></span><span>B <i class="unit">(ms)</i></span><span>W <i class="unit">(ms)</i></span></div>`;
  // mock only: dropdown that copies a profiled model's config into the mock
  const alignRow = `<div class="mt-row mt-align"><span></span>
      <span title="Copy the selected model's pattern, per-type times and model config into this mock; every value stays editable.">Align model</span><span></span>
      <span class="mt-align-box"><select id="mt-align">
        <option value="">--</option>
        ${Object.keys(DYN_OPTS.model_layers || {}).map(n => `<option>${n}</option>`).join("")}
      </select></span></div>`;
  const wireAlign = () => {
    const sel = el.querySelector("#mt-align");
    if (!sel) return;
    sel.value = alignSource; // keep showing which model the mock mirrors
    sel.addEventListener("change", (ev) => {
      if (ev.target.value) alignMockToModel(ev.target.value);
    });
  };

  if (isMock && hasPattern) {
    const t = mockTables(m);
    const { counts, order } = typeCounts(expandPat(m.pattern));
    const num = (v) => v === undefined || v === null ? "" : String(v);
    let html = alignRow + head;
    for (const c of order) {
      const [cls, label] = SYM_INFO[c] || ["mlp", c];
      const fixed = c === "E" || c === "L";
      html += `<div class="mt-row${fixed ? " mt-fixed" : ""}">
        <span class="mt-dot sym-${cls}"></span>
        <span>${labelize(label)} (${c})</span>
        <span>${counts[c]}</span>
        <span><input type="number" step="any" min="0" data-sym="${c}" data-kind="forward_ms" value="${num(t.f[c])}"></span>
        <span><input type="number" step="any" min="0" data-sym="${c}" data-kind="backward_ms" value="${num(t.b[c])}"></span>
        <span><input type="number" step="any" min="0" data-sym="${c}" data-kind="weight_ms" value="${num(t.w[c])}"></span></div>`;
    }
    el.innerHTML = html;
    el.style.display = "";
    wireAlign();
    for (const inp of el.querySelectorAll("input[data-sym]")) {
      inp.addEventListener("change", () => {
        const v = inp.value.trim() === "" ? undefined : Number(inp.value);
        if (v !== undefined && (!Number.isFinite(v) || v < 0)) return;
        const kind = inp.dataset.kind, sym = inp.dataset.sym;
        if (!cfgObj.model[kind]) cfgObj.model[kind] = {};
        if (v === undefined) delete cfgObj.model[kind][sym];
        else cfgObj.model[kind][sym] = v;
        if (!Object.keys(cfgObj.model[kind]).length) delete cfgObj.model[kind];
        scheduleDump();
        renderPartPlace();
        scheduleAutoRun();
      });
    }
    return;
  }

  const lm = (DYN_OPTS.model_layers || {})[name];
  if (lm && lm.pattern) {
    const { counts, order } = typeCounts(lm.pattern);
    let html = head;
    for (const c of order) {
      const [cls, label] = SYM_INFO[c] || ["mlp", c];
      const fixed = c === "E" || c === "L";
      html += `<div class="mt-row${fixed ? " mt-fixed" : ""}">
        <span class="mt-dot sym-${cls}"></span>
        <span>${labelize(label)} (${c})</span>
        <span>${counts[c]}</span><span>${fmtMs((lm.f || {})[c])}</span>
        <span>${fmtMs((lm.b || {})[c])}</span><span>${fmtMs((lm.w || {})[c])}</span></div>`;
    }
    el.innerHTML = html;
    el.style.display = "";
  } else if (isMock) {
    // legacy uniform mock (no pattern): still offer align as the way in
    const info = ppLayerInfo();
    el.innerHTML = alignRow + (info && info.uniform
      ? `<div class="mt-note">All layers: F ${fmtMs(info.uniform.f)} · B ${fmtMs(info.uniform.b)}
         · W ${fmtMs(info.uniform.w)} ms — embedding/head cost 0</div>`
      : "");
    el.style.display = "";
    wireAlign();
  } else {
    el.innerHTML = "";
    el.style.display = "none";
  }
}

/* Copy a profiled model's pattern, per-type times and intrinsic metadata
   into the mock config (name stays mock_model, everything stays editable). */
let alignSource = ""; // which profiled model the mock currently mirrors
function alignMockToModel(src) {
  const lm = (DYN_OPTS.model_layers || {})[src];
  if (!lm || !lm.pattern) return;
  const m = cfgObj.model;
  Object.assign(m, (DYN_OPTS.model_meta || {})[src] || {});
  m.name = "mock_model";
  for (const k of ["layer_time", "layer_f_time", "layer_b_time", "layer_w_time"])
    delete m[k];
  m.pattern = compressPat(lm.pattern);
  m.forward_ms = { ...(lm.f || {}) };
  m.backward_ms = { ...(lm.b || {}) };
  m.weight_ms = { ...(lm.w || {}) };
  m.num_layers = patBody(m.pattern).length;
  alignSource = src;
  // the old partition belongs to the previous layer count
  delete cfgObj.partition_layers;
  delete cfgObj.placement;
  ppLastRun = null;
  materializeDefaults();
  refreshFormValues();
  scheduleDump();
  scheduleAutoRun();
}

function ppStageBox(s, stages) {
  const info = ppLayerInfo();
  const maxSid = Math.max(...stages.map(t => t.sid));
  const box = document.createElement("div");
  box.className = "pp-stage" + (s.layers === 0 ? " empty" : "");
  const head = document.createElement("div");
  head.className = "pp-stage-head";
  head.draggable = true;
  head.title = "Drag onto another rank row to move this stage";
  head.innerHTML = `<b>S${s.sid}</b><span>${s.layers} layer${s.layers === 1 ? "" : "s"}</span>`;
  head.addEventListener("dragstart", (ev) => {
    ppDrag = { kind: "stage", sid: s.sid };
    ev.dataTransfer.effectAllowed = "move";
  });
  head.addEventListener("dragend", () => { ppDrag = null; });
  box.appendChild(head);

  const tiles = document.createElement("div");
  tiles.className = "pp-layers";
  let off = 0; // global index of this stage's first layer (stages are sid-ordered)
  for (const t of stages) { if (t.sid === s.sid) break; off += t.layers; }

  const fixedTile = (sym) => {
    const t = document.createElement("div");
    t.className = "pp-layer fixed";
    t.textContent = sym;
    // mock models: embedding/head cost 0 by definition
    const times = !info ? ""
      : info.uniform ? " — F 0 · B 0 · W 0 ms" : ` — ${symTimesText(info, sym)}`;
    t.title = `${SYM_INFO[sym][1]}${times}\nFixed to this stage; not counted in layers`;
    return t;
  };
  if (s.sid === 0) tiles.appendChild(fixedTile("E")); // embedding on the first stage

  const MAX_TILES = 64;
  for (let i = 0; i < Math.min(s.layers, MAX_TILES); i++) {
    const g = off + i;
    const sym = info && info.body && g < info.body.length ? info.body[g] : null;
    const tile = document.createElement("div");
    tile.className = "pp-layer" + (sym ? ` sym-${(SYM_INFO[sym] || ["mlp"])[0]}` : "");
    tile.draggable = true;
    tile.textContent = sym || String(g);
    const kind = sym ? ` · ${(SYM_INFO[sym] || ["", sym])[1]}` : "";
    const times = info ? ` — ${symTimesText(info, sym)}` : "";
    tile.title = `Layer ${g}${kind}${times}\nDrag onto another stage to move one layer`;
    tile.addEventListener("dragstart", (ev) => {
      ev.stopPropagation();
      ppDrag = { kind: "layer", sid: s.sid };
      ev.dataTransfer.effectAllowed = "move";
    });
    tile.addEventListener("dragend", () => { ppDrag = null; });
    tiles.appendChild(tile);
  }
  if (s.layers > MAX_TILES) {
    const more = document.createElement("div");
    more.className = "pp-layer more";
    more.textContent = `+${s.layers - MAX_TILES}`;
    more.title = "Layer count shown on the stage chip; drag any tile to move layers";
    tiles.appendChild(more);
  }
  if (s.sid === maxSid) tiles.appendChild(fixedTile("L")); // head on the last stage
  if (s.layers === 0) {
    const hint = document.createElement("div");
    hint.className = "pp-empty-hint";
    hint.textContent = "Drop layers here";
    tiles.appendChild(hint);
  }
  box.appendChild(tiles);

  box.addEventListener("dragover", (ev) => {
    if (ppDrag && ppDrag.kind === "layer" && ppDrag.sid !== s.sid) {
      ev.preventDefault(); ev.stopPropagation();
      box.classList.add("drop-ok");
    }
  });
  box.addEventListener("dragleave", () => box.classList.remove("drop-ok"));
  box.addEventListener("drop", (ev) => {
    if (!(ppDrag && ppDrag.kind === "layer" && ppDrag.sid !== s.sid)) return;
    ev.preventDefault(); ev.stopPropagation();
    box.classList.remove("drop-ok");
    const src = stages.find(x => x.sid === ppDrag.sid);
    if (src && src.layers > 0) { src.layers -= 1; s.layers += 1; ppCommit(stages); }
  });
  return box;
}

function renderPartPlace() {
  const body = $("partplace-body");
  if (!body) return;
  const fromCfg = ppStagesFromCfg();
  const manual = !!fromCfg;
  const stages = (fromCfg || ppStagesPreview()).sort((a, b) => a.sid - b.sid);
  const badge = $("pp-badge");
  badge.textContent = manual ? "Manual — in config" : "Auto preview — drag to pin";
  badge.className = "pp-badge " + (manual ? "manual" : "auto");
  $("pp-auto").style.display = manual ? "" : "none";

  const nRanks = Math.max(ppSize(), ...stages.map(s => s.rank + 1));
  body.innerHTML = "";
  const ranksBox = document.createElement("div");
  ranksBox.className = "pp-ranks" + (ppLayout === "tiled" ? " tiled" : "");
  for (let r = 0; r < nRanks; r++) {
    const row = document.createElement("div");
    row.className = "pp-rank";
    row.innerHTML = `<div class="pp-rank-label">Rank ${r}</div><div class="pp-rank-stages"></div>`;
    const cont = row.querySelector(".pp-rank-stages");
    for (const s of stages) {
      if (Math.min(s.rank, nRanks - 1) === r) cont.appendChild(ppStageBox(s, stages));
    }
    row.addEventListener("dragover", (ev) => {
      if (ppDrag && ppDrag.kind === "stage") { ev.preventDefault(); row.classList.add("drop-ok"); }
    });
    row.addEventListener("dragleave", () => row.classList.remove("drop-ok"));
    row.addEventListener("drop", (ev) => {
      ev.preventDefault();
      row.classList.remove("drop-ok");
      if (!(ppDrag && ppDrag.kind === "stage")) return;
      const st = stages.find(s => s.sid === ppDrag.sid);
      if (st && st.rank !== r) { st.rank = r; ppCommit(stages); }
    });
    ranksBox.appendChild(row);
  }
  body.appendChild(ranksBox);

  const part = stages.map(s => s.layers);
  const place = Array.from({ length: nRanks }, () => []);
  for (const s of stages) place[Math.min(s.rank, nRanks - 1)].push(s.sid);
  const pre = document.createElement("pre");
  pre.className = "pp-arrays";
  pre.textContent = `partition_layers: [${part.join(", ")}]\n` +
    `placement: [${place.map(x => `[${x.join(", ")}]`).join(", ")}]`;
  body.appendChild(pre);

  const warns = [];
  const L = getPath(cfgObj, "model.num_layers");
  const sum = part.reduce((a, b) => a + b, 0);
  if (part.some(n => n === 0)) warns.push("Empty stage — drag layers in before running");
  if (L !== undefined && sum !== L) warns.push(`Layer sum ${sum} ≠ model.num_layers ${L}`);
  place.forEach((sids, r) => { if (!sids.length) warns.push(`Rank ${r} has no stage`); });
  const sched = cfgObj.schedule || "1f1b";
  if (["1f1b", "bapar", "zbh", "afab"].includes(sched) && place.some(sids => sids.length > 1))
    warns.push(`Schedule ${schedLabel(sched)} runs one stage per rank — use Interleaved or OctoPipe for multi-stage ranks`);
  if (manual && warns.length) {
    const w = document.createElement("div");
    w.className = "pp-warn";
    w.textContent = warns.join("  ·  ");
    body.appendChild(w);
  }
}

$("pp-add").addEventListener("click", () => {
  const stages = ppStagesFromCfg() || ppStagesPreview();
  const sid = stages.length ? Math.max(...stages.map(s => s.sid)) + 1 : 0;
  stages.push({ sid, layers: 0, rank: ppSize() - 1 });
  ppCommit(stages);
});
$("pp-del").addEventListener("click", () => {
  const stages = (ppStagesFromCfg() || ppStagesPreview()).sort((a, b) => a.sid - b.sid);
  if (stages.length <= 1) return;
  const gone = stages.pop();
  stages[stages.length - 1].layers += gone.layers;
  ppCommit(stages);
});
$("pp-auto").addEventListener("click", () => {
  setPath(cfgObj, "partition_layers", undefined);
  setPath(cfgObj, "placement", undefined);
  scheduleDump();
  renderPartPlace();
});

/* rank layout: "tiled" flows rank cards side by side, "rows" aligns one
   pp rank per row.  The button shows the current mode. */
let ppLayout = localStorage.getItem("simpipe-pp-layout") || "tiled";
function updatePpLayoutBtn() {
  $("pp-layout").textContent = ppLayout === "tiled" ? "Tiled" : "By rank";
}
$("pp-layout").addEventListener("click", () => {
  ppLayout = ppLayout === "tiled" ? "rows" : "tiled";
  localStorage.setItem("simpipe-pp-layout", ppLayout);
  updatePpLayoutBtn();
  renderPartPlace();
});
updatePpLayoutBtn();

/* octopipe always runs the partition/placement search: while schedule is
   octopipe the auto_tune checkbox is forced on and locked. */
/* Grey out a control and put the reason in the row tooltip (disabled
   elements do not fire hover events, so the title must sit on the row). */
function setLocked(el, locked, note, desc) {
  el.disabled = locked;
  const text = locked ? `${note}\n\n${desc}` : desc;
  el.title = text;
  const row = el.closest(".frow");
  if (row) {
    row.classList.toggle("locked", locked);
    row.title = text;
    const lab = row.querySelector("label");
    if (lab) lab.title = text;
  }
}

/* MoE detection: preset/YAML use_moe flag or a '#' in the profiled pattern */
function modelIsMoe() {
  if (getPath(cfgObj, "model.use_moe")) return true;
  const lm = (DYN_OPTS.model_layers || {})[getPath(cfgObj, "model.name")];
  return !!(lm && lm.pattern && lm.pattern.includes("#"));
}

const fieldDesc = (path) =>
  CFG_SCHEMA.flatMap(s => s.fields).find(f => f.path === path).desc;

function syncTuningLock() {
  const el = document.querySelector('[data-path="tuning.auto_tune"]');
  if (!el) return;
  // auto tune is an OctoPipe feature: forced on there, forced off (and
  // greyed out) for every other schedule.
  const oct = cfgObj.schedule === "octopipe";
  if (getPath(cfgObj, "tuning.auto_tune") !== oct) {
    setPath(cfgObj, "tuning.auto_tune", oct);
    el.checked = oct;
    scheduleDump();
  }
  setLocked(el, true, oct ? "Locked: OctoPipe always tunes."
    : "Only OctoPipe supports auto tune.", fieldDesc("tuning.auto_tune"));

  // chunk_num is only meaningful for multi-chunk schedules; the engine
  // fixes it to 1 everywhere else, so lock the field accordingly.
  const ck = document.querySelector('[data-path="parallel.chunk_num"]');
  if (ck) {
    const multi = cfgObj.schedule === "interleaved" || oct;
    if (!multi) {
      if (getPath(cfgObj, "parallel.chunk_num") !== undefined) {
        delete (cfgObj.parallel || {}).chunk_num;
        scheduleDump();
      }
      ck.value = "1";
    }
    setLocked(ck, !multi,
      "Only Interleaved and OctoPipe support multiple chunks.",
      fieldDesc("parallel.chunk_num"));
  }

  // expert parallelism only applies to MoE models
  const ep = document.querySelector('[data-path="parallel.ep_size"]');
  if (ep) {
    const moe = modelIsMoe();
    if (!moe) {
      if (getPath(cfgObj, "parallel.ep_size") !== undefined) {
        delete (cfgObj.parallel || {}).ep_size;
        scheduleDump();
      }
      ep.value = "1";
    }
    setLocked(ep, !moe,
      "Only MoE models use expert parallelism.", fieldDesc("parallel.ep_size"));
  }
}

/* Batch rule: any batch content requires exactly one of microbatches /
   time_scales, sized to parallel.micro_batch_num when that is set. */
function validateBatchLive() {
  const warn = $("batch-warn");
  if (!warn) return;
  const b = cfgObj.batch;
  let msg = "";
  if (b && Object.keys(b).length) {
    const mb = Array.isArray(b.microbatches) ? b.microbatches.length : 0;
    const ts = Array.isArray(b.time_scales) ? b.time_scales.length : 0;
    const n = getPath(cfgObj, "parallel.micro_batch_num");
    if (!mb && !ts)
      msg = "batch is enabled: provide microbatches (one line per microbatch) or time_scales.";
    else if (mb && ts)
      msg = "batch: provide only one of microbatches / time_scales, not both.";
    else if (n !== undefined && (mb || ts) !== n)
      msg = `batch: ${mb ? "microbatches" : "time_scales"} count ${mb || ts} must equal parallel.micro_batch_num = ${n} (or clear micro_batch_num to derive it).`;
  }
  warn.textContent = msg;
  warn.style.display = msg ? "block" : "none";
  for (const p of ["batch.microbatches", "batch.time_scales"]) {
    const el = document.querySelector(`[data-path="${p}"]`);
    if (el) el.classList.toggle("batch-invalid", !!msg);
  }
}

/* --- form <-> YAML sync (server does the YAML parse/dump) --- */
let dumpTimer = null;
let dumpInflight = Promise.resolve();
function scheduleDump() { clearTimeout(dumpTimer); dumpTimer = setTimeout(dumpNow, 250); }
function dumpNow() {
  clearTimeout(dumpTimer); dumpTimer = null;
  dumpInflight = (async () => {
    const resp = await fetch("/api/dump", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: cfgObj }),
    });
    const r = await resp.json();
    if (r.ok) $("config").value = r.text;
  })().catch(() => {});
  return dumpInflight;
}
async function flushDump() { if (dumpTimer) await dumpNow(); else await dumpInflight; }

/* Fill engine defaults into the config so every form field shows its actual
   value.  Optional sections (batch) are not created just to hold defaults. */
function materializeDefaults() {
  let changed = false;
  for (const sec of CFG_SCHEMA) {
    for (const f of sec.fields) {
      if (f.def === undefined) continue;
      if (f.path.startsWith("batch.") && !cfgObj.batch) continue;
      if (getPath(cfgObj, f.path) === undefined) {
        setPath(cfgObj, f.path, f.def);
        changed = true;
      }
    }
  }
  return changed;
}

async function parseYamlToForm(showErrors) {
  try {
    const resp = await fetch("/api/parse", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: $("config").value }),
    });
    const r = await resp.json();
    if (!r.ok) { if (showErrors) showError("YAML parse failed: " + r.error); return false; }
    cfgObj = r.data || {};
    if (materializeDefaults()) scheduleDump(); // keep the YAML view in sync
    refreshFormValues();
    return true;
  } catch (e) {
    if (showErrors) showError("Request failed: " + e);
    return false;
  }
}

async function setCfgMode(mode) {
  if (mode === "form") {
    if (!(await parseYamlToForm(true))) return; // stay on YAML if it does not parse
  } else {
    await flushDump();
  }
  $("panel-config").classList.toggle("mode-form", mode === "form");
  for (const b of $("cfg-seg").querySelectorAll("button"))
    b.classList.toggle("active", b.dataset.mode === mode);
}
for (const b of $("cfg-seg").querySelectorAll("button"))
  b.addEventListener("click", () => setCfgMode(b.dataset.mode));

const formActive = () => $("panel-config").classList.contains("mode-form");
function setConfigText(text) {
  $("config").value = text;
  if (formActive()) parseYamlToForm(true);
}

/* --- import a local YAML file --- */
$("import-btn").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    setConfigText(String(reader.result));
    $("example-select").value = "";
  };
  reader.readAsText(file);
  ev.target.value = "";
});

/* ================= examples ================= */
async function loadExamples() {
  const sel = $("example-select");
  sel.innerHTML = "<option value=''>-- example configs --</option>";
  try {
    const examples = await (await fetch("/api/examples")).json();
    for (const ex of examples) {
      const opt = document.createElement("option");
      opt.value = ex.name;
      opt.textContent = ex.name;
      opt.dataset.content = ex.content;
      sel.appendChild(opt);
    }
    const preferred = examples.find(e => e.name === "mock_model.yaml")
      || examples.find(e => e.name === "varlen.yaml") || examples[0];
    if (preferred && !$("config").value) {
      sel.value = preferred.name;
      $("config").value = preferred.content;
    }
  } catch {}
  // default to the form view once the initial config is in place
  await setCfgMode("form");
}
$("example-select").addEventListener("change", (ev) => {
  const opt = ev.target.selectedOptions[0];
  if (opt && opt.dataset.content !== undefined) setConfigText(opt.dataset.content);
});

/* ================= rendering ================= */
function renderSummary(r) {
  const mem = r.memory;
  const peak = mem ? (mem.peak_gb !== undefined ? mem.peak_gb : mem.peak_bytes / 1024 ** 3) : null;
  let html = [
    `<div class="metric"><span>Model</span><b>${r.model}</b></div>`,
    `<div class="metric"><span>Schedule</span><b>${schedLabel(r.schedule)}</b></div>`,
    `<div class="metric"><span>Makespan</span><b>${Math.round(r.makespan).toLocaleString()}</b></div>`,
    `<div class="metric"><span>Bubble</span><b>${(r.bubble_ratio * 100).toFixed(2)}%</b></div>`,
  ].join("");
  if (peak !== null) {
    const cls = mem.feasible ? "ok" : "bad", txt = mem.feasible ? "Fits" : "OOM";
    // badge sits inline right after the value, not on its own line
    html += `<div class="metric"><span>Peak mem</span><b>${peak.toFixed(2)} GB` +
      ` <span class="badge ${cls}">${txt}</span></b></div>`;
  }
  if (r.stalled)
    html += `<div class="metric"><span>Status</span><b><span class="badge bad">Stalled</span></b></div>`;
  const lines = r.tuning_lines || [];
  html += `<div id="tuning-lines"${lines.length ? ' style="display:block"' : ""}>${lines.join("\n")}</div>`;
  $("summary-body").innerHTML = html;
}
const fmtNum = (x) => x === null || x === undefined ? "-" : Math.round(x).toLocaleString();
const fmtGb = (x) => x === null || x === undefined ? "-" : x.toFixed(2);
function renderRanks(rows) {
  if (!rows || !rows.length) {
    $("ranks-body").innerHTML = "<div class='placeholder'>No per-rank data.</div>";
    return;
  }
  let html = `<div class="tbl-wrap"><table><thead><tr>
    <th>Rank</th><th>Stages</th><th>Layers</th><th>Comp</th><th>Bubble</th><th>Bubble %</th>
    <th>Warm / cool / resid</th><th>Model <span class="unit">(GB)</span></th><th>Act <span class="unit">(GB)</span></th><th>Peak <span class="unit">(GB)</span></th><th>Status</th>
  </tr></thead><tbody>`;
  for (const row of rows) {
    const layers = row.layers.join("+") + (row.layers.length > 1 ? ` = ${row.layers.reduce((a, b) => a + b, 0)}` : "");
    const status = row.feasible === undefined || row.feasible === null ? "-"
      : row.feasible ? "<span class='badge ok'>OK</span>" : "<span class='badge bad'>OOM</span>";
    html += `<tr>
      <td>D${row.rank}</td><td>[${row.stages.join(", ")}]</td><td>${layers}</td>
      <td>${fmtNum(row.comp)}</td><td>${fmtNum(row.bubble)}</td>
      <td>${(row.bubble_ratio * 100).toFixed(2)}%</td>
      <td>${fmtNum(row.warmup_bubble)} / ${fmtNum(row.cooldown_bubble)} / ${fmtNum(row.residual_bubble)}</td>
      <td>${fmtGb(row.model_state_gb)}</td><td>${fmtGb(row.activation_peak_gb)}</td>
      <td>${fmtGb(row.peak_gb)}</td><td>${status}</td>
    </tr>`;
  }
  html += "</tbody></table></div>";
  $("ranks-body").innerHTML = html;
}

/* ================= gantt: canvas renderer + region zoom ================= */
let gantt = null; // { data, canvas, ctx, range: [fromPct, toPct], ro }

const fmtPct = (v) => `${Math.round(v * 10) / 10}%`;
/* Single source of truth for the visible time range (percent of full span). */
function setViewRange(fromPct, toPct) {
  if (!gantt) return;
  fromPct = Math.max(0, Math.min(99, fromPct));
  toPct = Math.min(100, Math.max(fromPct + 1, toPct));
  gantt.range = [fromPct, toPct];
  updateRangeUI();
  drawGantt();
}
function updateRangeUI() {
  const [f, t] = gantt ? gantt.range : [0, 100];
  $("range-from").value = Math.round(f);
  $("range-to").value = Math.round(t);
  $("range-label").innerHTML = `${fmtPct(f)} &ndash; ${fmtPct(t)}`;
  const fill = $("range-fill");
  fill.style.left = f + "%";
  fill.style.width = Math.max(0, t - f) + "%";
  $("zoom-reset").style.visibility = f > 0 || t < 100 ? "visible" : "hidden";
}
function onSlider(which) {
  let f = +$("range-from").value, t = +$("range-to").value;
  if (which === "from" && f > t - 1) f = t - 1;
  if (which === "to" && t < f + 1) t = f + 1;
  setViewRange(f, t);
}
$("range-from").addEventListener("input", () => onSlider("from"));
$("range-to").addEventListener("input", () => onSlider("to"));

/* Drag the filled band to pan the visible window (span stays constant). */
$("range-fill").addEventListener("mousedown", (ev) => {
  if (!gantt) return;
  ev.preventDefault();
  ev.stopPropagation(); // keep the panel-drag handler out of this gesture
  const startX = ev.clientX;
  const [f0, t0] = gantt.range;
  const span = t0 - f0;
  const ctlW = $("range-ctl").getBoundingClientRect().width;
  function move(e) {
    const dPct = ((e.clientX - startX) / ctlW) * 100;
    const nf = Math.max(0, Math.min(100 - span, f0 + dPct));
    setViewRange(nf, nf + span);
  }
  function up() {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
  }
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
});

const GANTT_COLORS = { F: "#E8C66A", B: "#94B8E8", W: "#8FBD8C", R: "#F8CECC" };
const GANTT_GUTTER = 40, GANTT_AXIS_H = 22;

/* Geometry of the current view: time window + pixel mapping. */
function ganttGeom() {
  const { canvas, data, range } = gantt;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width / dpr, H = canvas.height / dpr;
  const t0 = (range[0] / 100) * data.max_t;
  const t1 = (range[1] / 100) * data.max_t;
  const plotW = Math.max(1, W - GANTT_GUTTER - 6);
  const plotH = Math.max(1, H - GANTT_AXIS_H - 4);
  const rowH = plotH / Math.max(1, data.devices.length);
  return {
    W, H, t0, t1, plotW, plotH, rowH,
    xOf: (tm) => GANTT_GUTTER + ((tm - t0) / (t1 - t0)) * plotW,
    tOf: (px) => t0 + ((px - GANTT_GUTTER) / plotW) * (t1 - t0),
  };
}

function niceStep(rough) {
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  for (const m of [1, 2, 5, 10]) if (m * pow >= rough) return m * pow;
  return 10 * pow;
}

function drawGantt() {
  if (!gantt) return;
  const { ctx, data } = gantt;
  const dpr = window.devicePixelRatio || 1;
  const g = ganttGeom();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, g.W, g.H);
  ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";

  // zebra row backgrounds
  for (let i = 0; i < data.devices.length; i++) {
    if (i % 2 === 0) continue;
    ctx.fillStyle = "rgba(100, 116, 139, .055)";
    ctx.fillRect(GANTT_GUTTER, GANTT_AXIS_H + i * g.rowH, g.plotW, g.rowH);
  }

  // time axis with nice ticks
  const step = niceStep((g.t1 - g.t0) / 6);
  ctx.fillStyle = "#94a3b8";
  ctx.strokeStyle = "#e8edf4";
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  for (let tm = Math.ceil(g.t0 / step) * step; tm <= g.t1; tm += step) {
    const x = g.xOf(tm);
    ctx.beginPath(); ctx.moveTo(x, GANTT_AXIS_H - 6); ctx.lineTo(x, g.H - 4); ctx.stroke();
    ctx.fillText(tm.toLocaleString(), x + 3, 12);
  }

  const bh = Math.min(g.rowH * 0.74, g.rowH - 4);
  for (let i = 0; i < data.devices.length; i++) {
    const dev = data.devices[i];
    const yMid = GANTT_AXIS_H + i * g.rowH + g.rowH / 2;
    ctx.font = "600 11px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillStyle = "#475569";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(`D${dev.did}`, 6, yMid);
    ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
    // row separator
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(GANTT_GUTTER, GANTT_AXIS_H + (i + 1) * g.rowH);
    ctx.lineTo(g.W - 6, GANTT_AXIS_H + (i + 1) * g.rowH);
    ctx.stroke();

    const y = yMid - bh / 2;
    for (const [s, e, w, mid] of dev.blocks) {
      if (e <= g.t0 || s >= g.t1) continue;
      const x0 = Math.max(g.xOf(s), GANTT_GUTTER);
      const x1 = Math.min(g.xOf(e), GANTT_GUTTER + g.plotW);
      const bw = Math.max(x1 - x0, 0.75);
      ctx.fillStyle = GANTT_COLORS[w] || "#cccccc";
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y, bw, bh, Math.min(3, bw / 3));
      else ctx.rect(x0, y, bw, bh);
      ctx.fill();
      // Fade stroke/labels in with block width instead of hard on/off
      // thresholds: hard cutoffs make text pop in and out en masse while
      // dragging the range (visible flicker around the threshold).
      const strokeA = Math.min(1, Math.max(0, (bw - 3) / 6));
      if (strokeA > 0.04) {
        ctx.strokeStyle = `rgba(15,23,42,${(0.35 * strokeA).toFixed(3)})`;
        ctx.stroke();
      }
      const textA = Math.min(1, Math.max(0, (bw - 13) / 12));
      if (textA > 0.04) {
        ctx.fillStyle = `rgba(31,41,55,${textA.toFixed(3)})`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(mid), x0 + bw / 2, yMid);
        ctx.textAlign = "left";
      }
    }
  }
}

function ganttHit(mx, my) {
  const g = ganttGeom();
  const { data } = gantt;
  if (mx < GANTT_GUTTER || my < GANTT_AXIS_H) return null;
  const row = Math.floor((my - GANTT_AXIS_H) / g.rowH);
  if (row < 0 || row >= data.devices.length) return null;
  const tm = g.tOf(mx);
  const dev = data.devices[row];
  for (const b of dev.blocks) {
    if (b[0] <= tm && tm <= b[1]) return { dev, b };
    if (b[0] > tm) break; // blocks sorted by start
  }
  return null;
}

function setupGantt(data) {
  const body = $("gantt-body");
  if (!data || !data.devices || !data.devices.length) {
    body.innerHTML = "<div class='placeholder'>No scheduling records.</div>";
    gantt = null;
    return;
  }
  if (gantt && gantt.ro) gantt.ro.disconnect();
  body.innerHTML = `<div class="gantt-host"><canvas id="gantt-canvas"></canvas><div id="select-rect"></div><div id="gantt-tip"></div></div>`;
  const host = body.querySelector(".gantt-host");
  const canvas = $("gantt-canvas");
  canvas.width = 0; // force the first size() call to resize + draw
  gantt = { data, canvas, ctx: canvas.getContext("2d"), range: [0, 100], ro: null };

  function size() {
    const rect = host.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (w === canvas.width && h === canvas.height) return; // resize is a clear
    canvas.width = w;
    canvas.height = h;
    drawGantt();
  }
  gantt.ro = new ResizeObserver(size);
  gantt.ro.observe(host);
  size();
  updateRangeUI();

  let selecting = null;
  const tip = $("gantt-tip");

  canvas.addEventListener("mousemove", (ev) => {
    if (selecting) return;
    const rect = canvas.getBoundingClientRect();
    const hit = ganttHit(ev.clientX - rect.left, ev.clientY - rect.top);
    if (!hit) { tip.style.display = "none"; return; }
    const [s, e, w, mid, sid] = hit.b;
    tip.textContent =
      `${w} mid=${mid} sid=${sid} D${hit.dev.did}  ` +
      `${Math.round(s).toLocaleString()} – ${Math.round(e).toLocaleString()} (${Math.round(e - s).toLocaleString()})`;
    const hostRect = host.getBoundingClientRect();
    let lx = ev.clientX - hostRect.left + 14, ly = ev.clientY - hostRect.top + 14;
    tip.style.display = "block";
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (lx + tw > hostRect.width) lx = Math.max(0, ev.clientX - hostRect.left - tw - 10);
    if (ly + th > hostRect.height) ly = Math.max(0, ev.clientY - hostRect.top - th - 10);
    tip.style.left = lx + "px"; tip.style.top = ly + "px";
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });

  canvas.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const hostRect = host.getBoundingClientRect();
    selecting = { startX: ev.clientX };
    tip.style.display = "none";
    const sel = $("select-rect");
    sel.style.left = (ev.clientX - hostRect.left) + "px";
    sel.style.width = "0px";
    sel.style.display = "block";

    function move(e) {
      const a = Math.min(selecting.startX, e.clientX), b = Math.max(selecting.startX, e.clientX);
      sel.style.left = (a - hostRect.left) + "px";
      sel.style.width = (b - a) + "px";
    }
    function up(e) {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
      sel.style.display = "none";
      const a = Math.min(selecting.startX, e.clientX), b = Math.max(selecting.startX, e.clientX);
      selecting = null;
      if (b - a < 12) return; // click, not a selection
      const g = ganttGeom();
      const ta = g.tOf(Math.max(a - rect.left, GANTT_GUTTER));
      const tb = g.tOf(Math.min(b - rect.left, GANTT_GUTTER + g.plotW));
      setViewRange((ta / gantt.data.max_t) * 100, (tb / gantt.data.max_t) * 100);
    }
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });
}
$("zoom-reset").addEventListener("click", () => setViewRange(0, 100));

/* ================= run ================= */
function showError(message, tb) {
  const toast = $("toast");
  toast.innerHTML = `<span class="close">&times;</span><b>${message}</b>` +
    (tb ? ` <span class="toggle-tb">traceback</span><pre style="display:none">${tb}</pre>` : "");
  toast.style.display = "block";
  toast.querySelector(".close").onclick = () => (toast.style.display = "none");
  const tg = toast.querySelector(".toggle-tb");
  if (tg) tg.onclick = () => {
    const pre = toast.querySelector("pre");
    pre.style.display = pre.style.display === "none" ? "block" : "none";
  };
}
let runInflight = false, runQueued = false;
async function run() {
  if (runInflight) { runQueued = true; return; }
  runInflight = true;
  const btn = $("run-btn");
  btn.disabled = true; btn.textContent = "Running...";
  $("toast").style.display = "none";
  try {
    await flushDump(); // form edits land in the YAML before running
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: $("config").value }),
    });
    const r = await resp.json();
    if (!r.ok) { showError(r.error || "run failed", r.traceback); return; }
    last = r;
    ppLastRun = { part: r.partition, place: r.placement };
    renderSummary(r);
    renderPartPlace();
    setupGantt(r.gantt);
    renderRanks(r.ranks);
    $("config-out").textContent = r.pipeline_config;
    $("dl-svg").disabled = false;
    $("dl-config").disabled = false;
  } catch (e) {
    showError("Request failed: " + e);
  } finally {
    btn.disabled = false; btn.innerHTML = "Run &#9654;";
    runInflight = false;
    if (runQueued) { runQueued = false; run(); } // latest edits win
  }
}
$("run-btn").addEventListener("click", run);

/* auto rerun: debounce so bursts of edits collapse into one run */
let autoRunTimer = null;
function scheduleAutoRun() {
  const box = $("auto-rerun");
  if (!box || !box.checked) return;
  clearTimeout(autoRunTimer);
  autoRunTimer = setTimeout(run, 600);
}
$("config").addEventListener("input", () => {
  // YAML edits count as config changes too (run posts the raw YAML text)
  if ($("panel-config").classList.contains("mode-form")) return;
  scheduleAutoRun();
});
document.addEventListener("keydown", (ev) => {
  if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") { ev.preventDefault(); run(); }
});

/* ================= downloads ================= */
function download(name, text, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
$("dl-svg").addEventListener("click", () => last && download("pipeline_gantt.svg", last.svg, "image/svg+xml"));
$("dl-config").addEventListener("click", () => last && download("pipeline_config.yaml", last.pipeline_config, "text/yaml"));

(async () => {
  await fetchOptions();   // dropdown values for model.name / profile_times_path
  buildForm();
  await loadExamples();   // loads the default config into form + YAML
  if ($("config").value.trim()) run(); // show results immediately on first visit
})();
