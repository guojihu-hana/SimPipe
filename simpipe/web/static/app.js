"use strict";
const $ = (id) => document.getElementById(id);
let last = null;

/* ================= grid layout: snap, push-apart, persist ================= */
const COLS = 12, ROW_H = 84, GAP = 12;
const LAYOUT_KEY = "simpipe_grid_v4";
const UNLOCK_KEY = "simpipe_layout_unlocked";
// Fixed layout: the gantt on top, config + partition/placement editor below,
// summary as a full-width strip, then per-rank stats + yaml.
const DEFAULT_GRID = {
  "panel-gantt":     { x: 0, y: 0, w: 12, h: 5 },
  "panel-config":    { x: 0, y: 5, w: 5, h: 6 },
  "panel-partplace": { x: 5, y: 5, w: 7, h: 6 },
  "panel-summary":   { x: 0, y: 11, w: 12, h: 2 },
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
    { path: "model.name", type: "dselect", optsKey: "models", def: "mock_model", noEmpty: true, label: "Config",
      desc: "Profiled model (has profiles/<name>.json) or mock_model for synthetic timings. Custom names (used with profile_times_path) can be set in the YAML view." },
    { path: "model.num_layers", type: "int", def: 32, min: 1, max: 4096, label: "#Layers",
      desc: "Transformer body layer count (embedding/head excluded)." },
    { path: "model.pattern", type: "text", ph: "ET*32L",
      desc: "Layer pattern: E embedding, L head, body types M mamba / * attn / - MLP / T transformer / # MoE; X*N repeats X N times. Editable for mock_model (num_layers follows the pattern; raising num_layers pads T). Profiled models show their pattern read-only. Hover to see the full pattern." },
    { path: "model.layer_time", type: "float", min: 0.01, max: 1e9, ph: "e.g. 100 (= 1 ms)", mockOnly: true,
      desc: "Mock timing: uniform per-layer duration in 0.01 ms ticks, F = B = W. Embedding/head cost 0." },
    { path: "model.layer_f_time", type: "float", min: 0.01, max: 1e9, mockOnly: true,
      desc: "Mock timing: forward duration override (0.01 ms ticks)." },
    { path: "model.layer_b_time", type: "float", min: 0.01, max: 1e9, mockOnly: true,
      desc: "Mock timing: backward duration override; defaults to the forward time." },
    { path: "model.layer_w_time", type: "float", min: 0.01, max: 1e9, mockOnly: true,
      desc: "Mock timing: weight-update duration override; defaults to the forward time." },
  ]},
  // compact 4-up grid; row 1 and row 2 pair up column-wise
  // (PP/EP, TP/DP, Zero/Seq len, mb size / #mbs) with per-column label widths
  { sec: "Parallel", compact: true, fields: [
    { path: "parallel.pp_size", type: "int", def: 1, min: 1, max: 1024,
      desc: "Pipeline-parallel size = number of devices." },
    { path: "parallel.tp_size", type: "int", def: 8, min: 1, max: 64,
      desc: "Tensor-parallel size; scales analytic timing and per-rank model/activation memory." },
    { path: "parallel.zero_stage", type: "int", def: 1, min: 0, max: 3,
      desc: "ZeRO stage (0-3) used by the per-rank model-state memory estimate." },
    { path: "model.micro_batch_size", type: "int", def: 1, min: 1, max: 65536, label: "Micro-batch size",
      desc: "Reference microbatch size of the profiled shape." },
    { path: "parallel.ep_size", type: "int", def: 1, min: 1, max: 512,
      desc: "Expert-parallel size for MoE models; shards experts across ranks." },
    { path: "parallel.dp_size", type: "int", def: 1, min: 1, max: 4096,
      desc: "Data-parallel size; with ZeRO it shards optimizer/gradient state in the memory estimate." },
    { path: "model.seq_len", type: "int", def: 4096, min: 1, max: 16777216,
      desc: "Reference sequence length of the profiled shape; varlen batch scales relative to it." },
    { path: "parallel.micro_batch_num", type: "int", def: 8, min: 1, max: 16384, label: "#Micro-batches",
      desc: "Microbatches per iteration. Derived from batch.microbatches / batch.time_scales when those are set." },
    { path: "parallel.bwd_split", type: "bool", desc: "Split backward into B (grad-input) and W (grad-weight) workloads (zero-bubble style)." },
    { path: "model.recompute", type: "bool", desc: "Full activation recompute: each backward re-runs the forward first." },
    { path: "parallel.chunk_num", type: "int", min: 1, max: 256, ph: "auto", label: "#Chunk",
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
      label: "Comp launch (ms)",
      desc: "Constant launch overhead in ms added to every F/B/W workload's duration (kernel dispatch, bookkeeping, chunk switch); occupies device time." },
    { path: "hardware.comm_alpha_us", type: "float", def: 5.0, min: 0, max: 100000,
      label: "Comm launch (us)",
      desc: "Per-message launch latency in us on every cross-device dependency edge (NCCL alpha); delays the consumer without occupying device time. Any nonzero value rounds up to a whole 0.01 ms tick; set 0 for back-to-back stage handoffs." },
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
    // compact: 4-up cells with the label above the input (short numerics)
    box.className = sec.compact ? "form-sec compact" : "form-sec";
    box.innerHTML = `<h4>${sec.sec}</h4>`;
    const fgrid = document.createElement("div");
    fgrid.className = "fgrid";
    for (const f of sec.fields) {
      const row = document.createElement("div");
      row.className = f.wide ? "frow wide" : "frow";
      if (f.mockOnly) row.dataset.mockonly = "1"; // hidden unless mock_model
      row.title = f.desc; // hover shows the comment
      const name = f.label || labelize(f.path.split(".").pop());
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
      // 3-up grid: row 1 = Name | #Layers | Multimodal (+Enc/+Dec buttons),
      // row 2 = Pattern (two cells, hover shows full text) | Align model
      fgrid.classList.add("model-grid");
      const addRow = document.createElement("div");
      addRow.className = "frow aux-add-row";
      addRow.innerHTML =
        `<label>Multimodal</label><span class="aux-add-btns">` +
        `<button type="button" class="small" id="add-encoder" title="Add a multimodal encoder: every microbatch runs all encoders before backbone stage 0.">+ Encoder</button>` +
        `<button type="button" class="small" id="add-decoder" title="Add a decoder behind the last stage: its F follows the last stage's F and its B gates the last stage's B.">+ Decoder</button></span>`;
      const patRow = fgrid.querySelector('[data-path="model.pattern"]').closest(".frow");
      patRow.classList.add("pattern-row");
      // fixed label widths (CSS) align the input edges column-wise:
      // Name with Pattern, Multimodal with Align model
      fgrid.querySelector('[data-path="model.name"]').closest(".frow")
        .classList.add("name-row");
      patRow.before(addRow);
      addRow.querySelector("#add-encoder").addEventListener("click", () => addAuxModule("encoders"));
      addRow.querySelector("#add-decoder").addEventListener("click", () => addAuxModule("decoders"));
      // Align model: mock only (renderModelTimes toggles it and fills options)
      const alignRow = document.createElement("div");
      alignRow.className = "frow";
      alignRow.id = "align-row";
      alignRow.innerHTML =
        `<label title="Copy the selected model's pattern, per-type times and model config into this mock; every value stays editable.">Align model</label>` +
        `<select id="mt-align"><option value="">--</option></select>`;
      patRow.after(alignRow);
      alignRow.querySelector("#mt-align").addEventListener("change", (ev) => {
        if (ev.target.value) alignMockToModel(ev.target.value);
      });

      const mt = document.createElement("div");
      mt.id = "model-times"; // per-layer-type f/b/w table, filled dynamically
      box.appendChild(mt);
      const aux = document.createElement("div");
      aux.id = "aux-modules"; // encoder/decoder cards, filled dynamically
      box.appendChild(aux);
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
    if (f.path === "schedule" && val !== undefined) {
      // only OctoPipe and ZBH exploit the B/W backward split; every other
      // schedule defaults to a whole backward (still user-editable after)
      const wantSplit = val === "octopipe" || val === "zbh";
      setPath(cfgObj, "parallel.bwd_split", wantSplit);
      const bs = document.querySelector('[data-path="parallel.bwd_split"]');
      if (bs) bs.checked = wantSplit;
      if (val === "interleaved") {
        // interleaved defaults to the deepest chunking: one layer per stage
        const nl = Number(getPath(cfgObj, "model.num_layers")) || 0;
        const pp = Number(getPath(cfgObj, "parallel.pp_size")) || 1;
        const maxChunk = Math.max(1, Math.min(256, Math.floor(nl / pp)));
        setPath(cfgObj, "parallel.chunk_num", maxChunk);
        const ck = document.querySelector('[data-path="parallel.chunk_num"]');
        if (ck) ck.value = String(maxChunk);
      }
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
  renderAuxModules();
}

/* ============ multimodal encoders / decoders ============
   cfgObj.encoders / cfgObj.decoders are lists of module mappings (name,
   model {pattern, per-type times}, placement).  Memory knobs (params_gb,
   act_gb) are YAML-only and keep whatever the config carries.  Each
   renders as a nested mini model card (like the backbone's) inside the Model
   section; edits write the arrays directly and re-dump the YAML.  Legacy
   flat forward_ms/backward_ms/weight_ms scalars are folded into an
   equivalent single-layer model on first render.  Number/name inputs update
   in place (keeping focus); add/remove, placement-shape and pattern changes
   rebuild the cards. */
const AUX_PLACEMENTS = ["first_stage", "last_stage", "replicated", "dedicated"];
const AUX_PLACE_DESC = {
  first_stage: "Runs on the device holding backbone stage 0 (partition tuning rebalances layers around it).",
  last_stage: "Runs on the device holding the last backbone stage (partition tuning rebalances layers around it).",
  replicated: "One weight copy per backbone device; microbatches are sharded round-robin, so each runs its encoder once and up to pp_size run in parallel (module DP).",
  dedicated: "Gets its own extra device after the backbone's, connected via P2P.",
  devices: "Explicit device-id list; microbatches are sharded round-robin across the copies (module DP).",
};
const AUX_FIELD_DESC = {
  name: "Display name of this module (Gantt rows and records).",
  model: "The model this module runs. mock_model: synthetic per-layer-type times below; profiled models take pattern, layers and times from their profile.",
  num_layers: "Number of body layers; editing pads the pattern with T / truncates it.",
  pattern: "Layer pattern like the backbone's: body types M mamba / * attn / - MLP / T transformer / # MoE; X*N repeats X N times. The module runs as one block whose F/B/W are the sums over these layers.",
  recompute: "Full activation recompute: this module's backward re-runs its forward first (B costs F+B, like the backbone's recompute).",
};

function auxList(role) {
  return Array.isArray(cfgObj[role]) ? cfgObj[role] : [];
}

function addAuxModule(role) {
  const list = auxList(role).slice();
  const isEnc = role === "encoders";
  list.push({
    name: `${isEnc ? "encoder" : "decoder"}${list.length + 1}`,
    model: { name: "mock_model", num_layers: 4, pattern: "T*4", forward_ms: { T: 0.5 } },
    placement: isEnc ? "first_stage" : "last_stage",
  });
  cfgObj[role] = list;
  renderAuxModules();
  scheduleDump();
  scheduleAutoRun();
}

/* An aux module is a model.  Old flat specs (total forward_ms/backward_ms/
   weight_ms scalars) are folded into an equivalent single-layer mock model,
   and a legacy top-level recompute flag moves onto model.recompute; the
   engine performs the same normalization when parsing YAML. */
function auxNormalizeModel(mod) {
  let m = mod.model;
  if (!m || (!m.pattern && !m.name)) {
    const f = typeof mod.forward_ms === "number" ? mod.forward_ms : 1.0;
    m = { name: "mock_model", num_layers: 1, pattern: "T", forward_ms: { T: f } };
    if (typeof mod.backward_ms === "number") m.backward_ms = { T: mod.backward_ms };
    if (typeof mod.weight_ms === "number") m.weight_ms = { T: mod.weight_ms };
    delete mod.forward_ms;
    delete mod.backward_ms;
    delete mod.weight_ms;
    mod.model = m;
  }
  if (!m.name) m.name = "mock_model";
  if (mod.recompute !== undefined) { // legacy top-level flag
    if (mod.recompute) m.recompute = true;
    delete mod.recompute;
  }
  return m;
}
const auxIsMock = (m) => (m.name || "mock_model") === "mock_model";

/* keep every pattern type priced, like the backbone's ensurePatternTimes */
function auxEnsureTimes(m) {
  for (const sym of new Set(patBody(m.pattern || ""))) {
    if ((m.forward_ms || {})[sym] === undefined) {
      if (!m.forward_ms) m.forward_ms = {};
      m.forward_ms[sym] = 0.5;
    }
  }
}

/* per-pass module totals in ms (sum over pattern layers, E/L included) for
   the summary; profiled models read the profile's pattern + times */
function auxTotals(m) {
  const prof = auxIsMock(m) ? null : (DYN_OPTS.model_layers || {})[m.name];
  const t = prof ? { f: prof.f || {}, b: prof.b || {}, w: prof.w || {} } : mockTables(m);
  const pattern = prof ? prof.pattern : (m.pattern || "");
  const tot = { f: 0, b: 0, w: 0 };
  for (const c of expandPat(pattern)) {
    tot.f += t.f[c] || 0;
    tot.b += t.b[c] || 0;
    tot.w += t.w[c] || 0;
  }
  return tot;
}

function removeAuxModule(role, idx) {
  const list = auxList(role).slice();
  list.splice(idx, 1);
  if (list.length) cfgObj[role] = list; else delete cfgObj[role];
  renderAuxModules();
  scheduleDump();
  scheduleAutoRun();
}

function auxCard(role, mod, idx) {
  const isEnc = role === "encoders";
  const card = document.createElement("div");
  card.className = "aux-card";

  const head = document.createElement("div");
  head.className = "aux-head";
  const badge = document.createElement("span");
  badge.className = "aux-role " + (isEnc ? "enc" : "dec");
  badge.textContent = `${isEnc ? "Encoder" : "Decoder"} ${idx + 1}`; // matches the gantt legend's Enc1/Dec1
  const name = document.createElement("input");
  name.type = "text";
  name.className = "aux-name";
  name.spellcheck = false;
  name.placeholder = isEnc ? "encoder" : "decoder";
  name.title = AUX_FIELD_DESC.name;
  if (mod.name) name.value = mod.name;
  name.addEventListener("input", () => {
    const v = name.value.trim();
    if (v) mod.name = v; else delete mod.name;
    scheduleDump();
    scheduleAutoRun();
  });
  const del = document.createElement("button");
  del.type = "button";
  del.className = "aux-del";
  del.textContent = "\u2212"; // minus sign
  del.title = `Remove this ${isEnc ? "encoder" : "decoder"}`;
  del.addEventListener("click", () => removeAuxModule(role, idx));
  head.appendChild(badge);
  head.appendChild(name);
  head.appendChild(del);
  card.appendChild(head);

  /* --- the module's own model (a mini backbone-style model card) --- */
  const m = auxNormalizeModel(mod);
  const isMock = auxIsMock(m);
  const prof = isMock ? null : (DYN_OPTS.model_layers || {})[m.name];

  const grid = document.createElement("div");
  grid.className = "fgrid";

  // model selector: mock_model or any profiled model, like the backbone's
  const mrow = document.createElement("div");
  mrow.className = "frow";
  mrow.title = AUX_FIELD_DESC.model;
  const mlab = document.createElement("label");
  mlab.textContent = "Model";
  const msel = document.createElement("select");
  const modelNames = (DYN_OPTS.models && DYN_OPTS.models.length)
    ? DYN_OPTS.models : ["mock_model"];
  for (const n of modelNames) {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n;
    msel.appendChild(o);
  }
  if (![...msel.options].some(o => o.value === m.name)) {
    const o = document.createElement("option"); // model from YAML we don't know
    o.value = m.name;
    o.textContent = m.name;
    msel.appendChild(o);
  }
  msel.value = m.name;
  msel.title = AUX_FIELD_DESC.model;
  msel.addEventListener("change", () => {
    const n = msel.value;
    const rc = m.recompute; // user intent survives the model swap
    if (n === "mock_model") {
      mod.model = { name: "mock_model", num_layers: 4, pattern: "T*4", forward_ms: { T: 0.5 } };
    } else {
      // profiled model: pattern / layers / times come from the profile, so
      // the config only records which model (engine fills in the rest)
      mod.model = { name: n };
    }
    if (rc) mod.model.recompute = true;
    renderAuxModules();
    scheduleDump();
    scheduleAutoRun();
  });
  mrow.appendChild(mlab);
  mrow.appendChild(msel);
  grid.appendChild(mrow);

  // layers <-> pattern, backbone-style: layers edits pad/truncate the
  // pattern with T, pattern edits recount layers; locked for profiled
  // models, whose shape comes from the profile
  const lrow = document.createElement("div");
  lrow.className = "frow";
  lrow.title = AUX_FIELD_DESC.num_layers;
  const llab = document.createElement("label");
  llab.textContent = "Layers";
  const linp = document.createElement("input");
  linp.type = "number";
  linp.min = "1";
  linp.step = "1";
  linp.value = prof
    ? String(patBody(prof.pattern || "").length)
    : String(patBody(m.pattern || "").length || 1);
  lrow.appendChild(llab);
  lrow.appendChild(linp);
  grid.appendChild(lrow);

  const patrow = document.createElement("div");
  patrow.className = "frow";
  patrow.title = AUX_FIELD_DESC.pattern;
  const platb = document.createElement("label");
  platb.textContent = "Pattern";
  const pinp = document.createElement("input");
  pinp.type = "text";
  pinp.spellcheck = false;
  pinp.placeholder = "T*4";
  pinp.value = prof ? compressPat(prof.pattern || "") : (m.pattern || "");
  patrow.appendChild(platb);
  patrow.appendChild(pinp);
  grid.appendChild(patrow);

  if (!isMock) {
    const lockNote = "Locked: comes from the model profile. Select mock_model to edit it.";
    setLocked(linp, true, lockNote, AUX_FIELD_DESC.num_layers);
    setLocked(pinp, true, lockNote, AUX_FIELD_DESC.pattern);
  }
  linp.addEventListener("change", () => {
    const n = Number(linp.value);
    if (!Number.isInteger(n) || n < 1) { linp.classList.add("invalid"); return; }
    linp.classList.remove("invalid");
    let body = patBody(m.pattern || "");
    body = n > body.length
      ? body.concat(Array(n - body.length).fill("T"))
      : body.slice(0, n);
    m.pattern = compressPat(body.join(""));
    m.num_layers = n;
    auxEnsureTimes(m);
    renderAuxModules(); // the times table rows may change
    scheduleDump();
    scheduleAutoRun();
  });
  pinp.addEventListener("change", () => {
    const v = pinp.value.trim();
    const exp = expandPat(v);
    if (!v || !PAT_CHARS.test(exp) || !patBody(v).length) {
      pinp.classList.add("invalid");
      return;
    }
    pinp.classList.remove("invalid");
    m.pattern = v;
    m.num_layers = patBody(v).length;
    auxEnsureTimes(m);
    renderAuxModules();
    scheduleDump();
    scheduleAutoRun();
  });

  // full activation recompute (B re-runs F first); lives on the module's
  // model, exactly like the backbone's model.recompute
  const rrow = document.createElement("div");
  rrow.className = "frow";
  rrow.title = AUX_FIELD_DESC.recompute;
  const rlab = document.createElement("label");
  rlab.textContent = "Recompute";
  const rchk = document.createElement("input");
  rchk.type = "checkbox";
  rchk.checked = m.recompute === true;
  rchk.addEventListener("change", () => {
    if (rchk.checked) m.recompute = true;
    else delete m.recompute;
    renderAuxModules(); // the module-total line reflects the B = F+B fold
    scheduleDump();
    scheduleAutoRun();
  });
  rrow.appendChild(rlab);
  rrow.appendChild(rchk);
  grid.appendChild(rrow);

  // placement select (+ device list input when explicit)
  const prow = document.createElement("div");
  prow.className = "frow";
  const isList = Array.isArray(mod.placement);
  prow.title = AUX_PLACE_DESC[isList ? "devices" : (mod.placement || "first_stage")];
  const plab = document.createElement("label");
  plab.textContent = "Placement";
  const sel = document.createElement("select");
  for (const p of AUX_PLACEMENTS.concat(["devices"])) {
    const o = document.createElement("option");
    o.value = p;
    o.textContent = p === "devices" ? "device list\u2026" : p.replace("_", " ");
    o.title = AUX_PLACE_DESC[p];
    sel.appendChild(o);
  }
  sel.value = isList ? "devices" : (mod.placement || (isEnc ? "first_stage" : "last_stage"));
  sel.addEventListener("change", () => {
    if (sel.value === "devices") mod.placement = Array.isArray(mod.placement) ? mod.placement : [0];
    else mod.placement = sel.value;
    renderAuxModules(); // device-list input appears/disappears
    scheduleDump();
    scheduleAutoRun();
  });
  prow.appendChild(plab);
  prow.appendChild(sel);
  grid.appendChild(prow);

  if (isList) {
    const drow = document.createElement("div");
    drow.className = "frow";
    drow.title = AUX_PLACE_DESC.devices;
    const dlab = document.createElement("label");
    dlab.textContent = "Devices";
    const dinp = document.createElement("input");
    dinp.type = "text";
    dinp.spellcheck = false;
    dinp.placeholder = "0, 1";
    dinp.value = mod.placement.join(", ");
    dinp.addEventListener("input", () => {
      const items = dinp.value.replace(/[\[\]]/g, "").split(",")
        .map(s => s.trim()).filter(s => s !== "");
      const ids = items.map(Number);
      if (!ids.length || ids.some(v => !Number.isInteger(v) || v < 0)) {
        dinp.classList.add("invalid");
        return;
      }
      dinp.classList.remove("invalid");
      mod.placement = ids;
      scheduleDump();
      scheduleAutoRun();
    });
    drow.appendChild(dlab);
    drow.appendChild(dinp);
    grid.appendChild(drow);
  }

  card.appendChild(grid);
  card.appendChild(auxTimesTable(mod, m));
  return card;
}

/* per-layer-type F/B/W table of an aux module's model, mirroring the
   backbone's renderModelTimes: editable ms inputs for mock_model, read-only
   profile values for profiled models */
function auxTimesTable(mod, m) {
  const wrap = document.createElement("div");
  wrap.className = "aux-times";
  const prof = auxIsMock(m) ? null : (DYN_OPTS.model_layers || {})[m.name];
  const t = prof ? { f: prof.f || {}, b: prof.b || {}, w: prof.w || {} } : mockTables(m);
  const pattern = prof ? prof.pattern : (m.pattern || "");
  const { counts, order } = typeCounts(expandPat(pattern));
  const num = (v) => v === undefined || v === null ? "" : String(v);
  const ro = prof ? " disabled" : "";
  // "N" instead of "Count": the cards sit two abreast so columns are narrow
  let html = `<div class="mt-row mt-head"><span></span><span>Type</span>
      <span title="Layer count">N</span><span>F <i class="unit">(ms)</i></span><span>B <i class="unit">(ms)</i></span><span>W <i class="unit">(ms)</i></span></div>`;
  for (const c of order) {
    const [cls, label] = SYM_INFO[c] || ["mlp", c];
    const fixed = c === "E" || c === "L";
    html += `<div class="mt-row${fixed ? " mt-fixed" : ""}">
      <span class="mt-dot sym-${cls}"></span>
      <span title="${labelize(label)} (${c})">${labelize(label)} (${c})</span>
      <span>${counts[c]}</span>
      <span><input type="number" step="any" min="0" data-sym="${c}" data-kind="forward_ms" value="${num(t.f[c])}"${ro}></span>
      <span><input type="number" step="any" min="0" data-sym="${c}" data-kind="backward_ms" value="${num(t.b[c])}"${ro}></span>
      <span><input type="number" step="any" min="0" data-sym="${c}" data-kind="weight_ms" value="${num(t.w[c])}"${ro}></span></div>`;
  }
  if (prof) html += `<div class="mt-note">Times come from the ${m.name} profile (read-only).</div>`;
  html += `<div class="mt-note aux-sum"></div>`;
  wrap.innerHTML = html;

  const sum = wrap.querySelector(".aux-sum");
  const updateSum = () => {
    const tot = auxTotals(m);
    const rc = m.recompute === true;
    const b = rc ? tot.b + tot.f : tot.b; // recompute: B re-runs F
    sum.textContent =
      `Module per microbatch: F ${fmtMs(tot.f)} \u00b7 B ${fmtMs(b)}${rc ? " (recompute)" : ""} \u00b7 W ${fmtMs(tot.w)} ms`;
  };
  updateSum();

  for (const inp of wrap.querySelectorAll("input[data-sym]:not([disabled])")) {
    inp.addEventListener("change", () => {
      const v = inp.value.trim() === "" ? undefined : Number(inp.value);
      if (v !== undefined && (!Number.isFinite(v) || v < 0)) return;
      const kind = inp.dataset.kind, sym = inp.dataset.sym;
      if (!m[kind]) m[kind] = {};
      if (v === undefined) delete m[kind][sym];
      else m[kind][sym] = v;
      if (!Object.keys(m[kind]).length) delete m[kind];
      updateSum();
      scheduleDump();
      scheduleAutoRun();
    });
  }
  return wrap;
}

function renderAuxModules() {
  const root = $("aux-modules");
  if (!root) return;
  root.innerHTML = "";
  let migrated = false;
  for (const role of ["encoders", "decoders"])
    auxList(role).forEach((mod, idx) => {
      // flat scalars -> model, or a legacy top-level recompute flag
      if (!(mod.model && (mod.model.pattern || mod.model.name)) || mod.recompute !== undefined)
        migrated = true;
      root.appendChild(auxCard(role, mod, idx));
    });
  if (migrated) scheduleDump(); // keep the YAML view in sync with the fold
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
    // the box is narrow: hovering shows the full pattern above the field help
    if (pe.value) pe.title = `${pe.value}\n\n${pe.title}`;
  }
  const nlEl = document.querySelector('[data-path="model.num_layers"]');
  if (nlEl) setLocked(nlEl, !isMock, "Locked: the layer count comes from the " +
    "model profile. Select mock_model to edit it.", fieldDesc("model.num_layers"));

  const head = `<div class="mt-row mt-head"><span></span><span>Type</span>
      <span>Count</span><span>F <i class="unit">(ms)</i></span><span>B <i class="unit">(ms)</i></span><span>W <i class="unit">(ms)</i></span></div>`;
  // Align model row (form grid, next to Pattern): mock only — the dropdown
  // copies a profiled model's config into the mock
  const alignRowEl = document.getElementById("align-row");
  if (alignRowEl) {
    alignRowEl.style.display = isMock ? "" : "none";
    const sel = alignRowEl.querySelector("#mt-align");
    const names = Object.keys(DYN_OPTS.model_layers || {});
    if (sel.options.length !== names.length + 1)
      sel.innerHTML = `<option value="">--</option>` +
        names.map(n => `<option>${n}</option>`).join("");
    sel.value = alignSource; // keep showing which model the mock mirrors
  }

  if (isMock && hasPattern) {
    const t = mockTables(m);
    const { counts, order } = typeCounts(expandPat(m.pattern));
    const num = (v) => v === undefined || v === null ? "" : String(v);
    let html = head;
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
    // legacy uniform mock (no pattern): the Align model row stays visible
    const info = ppLayerInfo();
    el.innerHTML = info && info.uniform
      ? `<div class="mt-note">All layers: F ${fmtMs(info.uniform.f)} · B ${fmtMs(info.uniform.b)}
         · W ${fmtMs(info.uniform.w)} ms — embedding/head cost 0</div>`
      : "";
    el.style.display = "";
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

/* Where each encoder/decoder copy lives, mirroring the engine's placement
   rules (first_stage / last_stage / replicated / dedicated / device list).
   Returns { byRank: Map<rank, item[]>, dedicated: item[] }. */
function ppAuxDistribution(stages, nRanks) {
  const out = { byRank: new Map(), dedicated: [] };
  const rankOf = (sid) => {
    const st = stages.find(s => s.sid === sid);
    return st ? Math.min(st.rank, nRanks - 1) : 0;
  };
  const maxSid = stages.length ? Math.max(...stages.map(s => s.sid)) : 0;
  const add = (r, item) => {
    if (r < 0 || r >= nRanks) return;
    if (!out.byRank.has(r)) out.byRank.set(r, []);
    out.byRank.get(r).push(item);
  };
  const walk = (mods, isEnc) => (mods || []).forEach((mod, i) => {
    const item = {
      label: (isEnc ? "Enc" : "Dec") + (i + 1),
      enc: isEnc,
      title: `${mod.name || (isEnc ? "encoder" : "decoder")} — ` +
        (Array.isArray(mod.placement) ? `devices [${mod.placement.join(", ")}]`
          : (mod.placement || (isEnc ? "first_stage" : "last_stage"))),
    };
    const p = mod.placement || (isEnc ? "first_stage" : "last_stage");
    if (Array.isArray(p)) p.forEach(r => add(r, item));
    else if (p === "replicated") for (let r = 0; r < nRanks; r++) add(r, item);
    else if (p === "dedicated") out.dedicated.push(item);
    else if (p === "last_stage") add(rankOf(maxSid), item);
    else add(rankOf(0), item);
  });
  walk(cfgObj.encoders, true);
  walk(cfgObj.decoders, false);
  return out;
}
function ppAuxBadges(items) {
  if (!items || !items.length) return "";
  return " " + items.map(it =>
    `<span class="aux-role ${it.enc ? "enc" : "dec"}" title="${escHtml(it.title)}">${escHtml(it.label)}</span>`
  ).join(" ");
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
  const aux = ppAuxDistribution(stages, nRanks);
  body.innerHTML = "";
  const ranksBox = document.createElement("div");
  ranksBox.className = "pp-ranks" + (ppLayout === "tiled" ? " tiled" : "");
  for (let r = 0; r < nRanks; r++) {
    const row = document.createElement("div");
    row.className = "pp-rank";
    row.innerHTML = `<div class="pp-rank-label">Rank ${r}${ppAuxBadges(aux.byRank.get(r))}</div>` +
      `<div class="pp-rank-stages"></div>`;
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
  // dedicated encoder/decoder hosts occupy extra devices past the backbone
  aux.dedicated.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "pp-rank pp-rank-aux";
    row.innerHTML = `<div class="pp-rank-label">Rank ${nRanks + i}${ppAuxBadges([item])}</div>` +
      `<div class="pp-rank-stages"><span class="pp-aux-note">dedicated device — no backbone stage</span></div>`;
    ranksBox.appendChild(row);
  });
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

/* --- import / export a local YAML file --- */
$("import-btn").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => setConfigText(String(reader.result));
  reader.readAsText(file);
  ev.target.value = "";
});
$("export-btn").addEventListener("click", async () => {
  await flushDump(); // pending form edits land in the YAML first
  let text = $("config").value;
  // A profiled model is just a name that references on-disk profile data;
  // inline its pattern + per-type times (same shape alignMockToModel makes)
  // so the exported file runs on machines without any profiles.  Aux
  // modules are models too, so their profiled models inline the same way.
  const inlineTimes = (mdl, lm2) => {
    mdl.pattern = compressPat(lm2.pattern);
    mdl.forward_ms = { ...(lm2.f || {}) };
    mdl.backward_ms = { ...(lm2.b || {}) };
    mdl.weight_ms = { ...(lm2.w || {}) };
    mdl.num_layers = patBody(mdl.pattern).length;
    mdl.name = "mock_model";
  };
  const cfg = JSON.parse(JSON.stringify(cfgObj));
  let changed = false;
  const name = getPath(cfgObj, "model.name");
  const lm = (DYN_OPTS.model_layers || {})[name];
  if (name !== "mock_model" && lm && lm.pattern) {
    Object.assign(cfg.model, (DYN_OPTS.model_meta || {})[name] || {});
    for (const k of ["layer_time", "layer_f_time", "layer_b_time", "layer_w_time"])
      delete cfg.model[k];
    inlineTimes(cfg.model, lm);
    changed = true;
  }
  for (const role of ["encoders", "decoders"])
    for (const mod of (Array.isArray(cfg[role]) ? cfg[role] : [])) {
      const mn = mod.model && mod.model.name;
      const mlm = mn && mn !== "mock_model" && (DYN_OPTS.model_layers || {})[mn];
      if (mlm && mlm.pattern) { inlineTimes(mod.model, mlm); changed = true; }
    }
  if (changed) {
    try {
      const resp = await fetch("/api/dump", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: cfg }),
      });
      const r = await resp.json();
      if (r.ok) text = r.text;
    } catch {} // fall back to exporting the config as-is
  }
  download("simpipe_config.yaml", text, "text/yaml");
});

/* ================= initial config ================= */
async function loadDefaultConfig() {
  try {
    const examples = await (await fetch("/api/examples")).json();
    const preferred = examples.find(e => e.name === "mock_model.yaml")
      || examples.find(e => e.name === "varlen.yaml") || examples[0];
    if (preferred && !$("config").value) $("config").value = preferred.content;
  } catch {}
  // default to the form view once the initial config is in place
  await setCfgMode("form");
}

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
  // Aux column only exists when some rank hosts an encoder/decoder copy
  const hasAux = rows.some(r => (r.aux || []).length);
  let html = `<div class="tbl-wrap"><table><thead><tr>
    <th>Rank</th><th>Stages</th><th>Layers</th>${hasAux ? "<th>Enc / Dec</th>" : ""}<th>Comp</th><th>Bubble</th><th>Bubble %</th>
    <th>Warm / cool / resid</th><th>Model <span class="unit">(GB)</span></th><th>Act <span class="unit">(GB)</span></th><th>Peak <span class="unit">(GB)</span></th><th>Status</th>
  </tr></thead><tbody>`;
  for (const row of rows) {
    const layers = !row.layers.length ? "-"
      : row.layers.join("+") + (row.layers.length > 1 ? ` = ${row.layers.reduce((a, b) => a + b, 0)}` : "");
    const status = row.feasible === undefined || row.feasible === null ? "-"
      : row.feasible ? "<span class='badge ok'>OK</span>" : "<span class='badge bad'>OOM</span>";
    const stages = row.stages.length ? `[${row.stages.join(", ")}]` : "aux only";
    const auxCell = hasAux
      ? `<td>${(row.aux || []).map(auxRankBadge).join(" ") || "-"}</td>`
      : "";
    html += `<tr>
      <td>D${row.rank}</td><td>${stages}</td><td>${layers}</td>${auxCell}
      <td>${fmtNum(row.comp)}</td><td>${fmtNum(row.bubble)}</td>
      <td>${row.bubble_ratio === null || row.bubble_ratio === undefined ? "-" : (row.bubble_ratio * 100).toFixed(2) + "%"}</td>
      <td>${fmtNum(row.warmup_bubble)} / ${fmtNum(row.cooldown_bubble)} / ${fmtNum(row.residual_bubble)}</td>
      <td>${fmtGb(row.model_state_gb)}</td><td>${fmtGb(row.activation_peak_gb)}</td>
      <td>${fmtGb(row.peak_gb)}</td><td>${status}</td>
    </tr>`;
  }
  html += "</tbody></table></div>";
  $("ranks-body").innerHTML = html;
}
/* "enc:vit@d0" -> small role-colored chip labeled with the module name */
function auxRankBadge(tag) {
  const enc = tag.startsWith("enc:");
  const name = escHtml(tag.slice(4));
  return `<span class="aux-role ${enc ? "enc" : "dec"}">${name}</span>`;
}
function escHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ================= gantt: canvas renderer + region zoom ================= */
let gantt = null; // { data, canvas, ctx, range: [fromPct, toPct], ro }

/* Deepest zoom: 0.001% of the run (~a handful of ticks on big traces). */
const MIN_SPAN_PCT = 1e-3;
const fmtPct = (v) => {
  const span = gantt ? gantt.range[1] - gantt.range[0] : 100;
  const digits = span < 0.1 ? 3 : span < 2 ? 2 : 1;
  return `${v.toFixed(digits)}%`;
};
/* Single source of truth for the visible time range (percent of full span). */
function setViewRange(fromPct, toPct) {
  if (!gantt) return;
  fromPct = Math.max(0, Math.min(100 - MIN_SPAN_PCT, fromPct));
  toPct = Math.min(100, Math.max(fromPct + MIN_SPAN_PCT, toPct));
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
/* Multimodal modules use their own palettes (block kind "enc:*" / "dec:*")
   so encoder / backbone / decoder blocks are distinguishable at a glance;
   within a palette F is lightest and W darkest.  Encoders draw from warm
   hues and decoders from cool violets; each module gets its own palette
   (assigned by name below, wrapping if there are more modules than hues). */
const AUX_PALETTES = {
  enc: [
    { F: "#F2A6A6", B: "#E37B7B", W: "#CE5757", R: "#FAE3E3" }, // red
    { F: "#F6C08F", B: "#EC9A4E", W: "#D97B23", R: "#FBE7D0" }, // orange
    { F: "#F2A6CD", B: "#E37BAE", W: "#CE578F", R: "#FAE3F0" }, // rose
    { F: "#E0B48C", B: "#C98F5A", W: "#A96F38", R: "#F3E4D3" }, // copper
    { F: "#F8B3A0", B: "#EF8A6C", W: "#D96846", R: "#FCE5DD" }, // coral
  ],
  dec: [
    { F: "#CDB6F2", B: "#A98BE3", W: "#8A66CE", R: "#EBE0FA" }, // purple
    { F: "#B3B8F0", B: "#8890E0", W: "#6470CC", R: "#E2E4FA" }, // indigo
    { F: "#E0AEE0", B: "#C883C8", W: "#A95FA9", R: "#F5E0F5" }, // plum
    { F: "#B9C8F2", B: "#8FA5E3", W: "#6B84CE", R: "#E4EAFA" }, // periwinkle
    { F: "#C3A6E8", B: "#9C74D4", W: "#7B52B8", R: "#EDE2F8" }, // violet
  ],
};
/* kind ("enc:vit") -> palette, rebuilt per run in setupGantt */
let auxKindPalette = {};
const blockPalette = (kind) =>
  kind ? auxKindPalette[kind] || GANTT_COLORS : GANTT_COLORS;
const GANTT_GUTTER = 40, GANTT_AXIS_H = 22;
/* Stable, well-spread accent per micro batch (used for pins and connectors). */
const pathColor = (mid) => `hsl(${(mid * 61) % 360} 60% 36%)`;

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

  // micro-batch path highlight: any number of pinned mids plus the hovered one
  const activeMids = new Set(gantt.pinnedMids);
  if (gantt.hoverMid != null) activeMids.add(gantt.hoverMid);
  const anyActive = activeMids.size > 0;

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
    for (const [s, e, w, mid, , kind] of dev.blocks) {
      if (e <= g.t0 || s >= g.t1) continue;
      const x0 = Math.max(g.xOf(s), GANTT_GUTTER);
      const x1 = Math.min(g.xOf(e), GANTT_GUTTER + g.plotW);
      const bw = Math.max(x1 - x0, 0.75);
      const isActive = activeMids.has(mid);
      const dimmed = anyActive && !isActive;
      ctx.globalAlpha = dimmed ? 0.22 : 1;
      ctx.fillStyle = blockPalette(kind)[w] || "#cccccc";
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y, bw, bh, Math.min(3, bw / 3));
      else ctx.rect(x0, y, bw, bh);
      ctx.fill();
      // Fade stroke/labels in with block width instead of hard on/off
      // thresholds: hard cutoffs make text pop in and out en masse while
      // dragging the range (visible flicker around the threshold).
      const strokeA = Math.min(1, Math.max(0, (bw - 3) / 6));
      if (isActive) {
        ctx.strokeStyle = pathColor(mid);
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.lineWidth = 1;
      } else if (strokeA > 0.04) {
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
      ctx.globalAlpha = 1;
    }
  }

  // dashed connectors tracing each active micro-batch through the pipeline
  if (anyActive) {
    const cx = (t) => Math.max(GANTT_GUTTER, Math.min(g.xOf(t), GANTT_GUTTER + g.plotW));
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    for (const am of activeMids) {
      const pts = [];
      for (let i = 0; i < data.devices.length; i++) {
        const yMid = GANTT_AXIS_H + i * g.rowH + g.rowH / 2;
        for (const b of data.devices[i].blocks) {
          if (b[3] === am) pts.push({ s: b[0], e: b[1], yMid });
        }
      }
      pts.sort((p, q) => p.s - q.s || p.e - q.e);
      ctx.strokeStyle = pathColor(am);
      ctx.beginPath();
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        if (b.s <= g.t0 && a.e <= g.t0) continue; // both left of the view
        if (a.e >= g.t1 && b.s >= g.t1) continue; // both right of the view
        ctx.moveTo(cx(a.e), a.yMid);
        ctx.lineTo(cx(b.s), b.yMid);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
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
  // one palette + legend entry per multimodal module, labeled Enc1/Enc2/
  // Dec1... to match the config card badges; the index follows the config's
  // encoders/decoders array order so colors stay stable across reruns
  // (encoders take warm hues, decoders cool ones, no repeats per role)
  const kinds = new Set();
  if (data && data.devices)
    for (const d of data.devices)
      for (const b of d.blocks) if (b[5]) kinds.add(b[5]);
  auxKindPalette = {};
  const legend = $("legend-aux");
  legend.textContent = "";
  // group replicated / device-list shards ("vit@d0", "vit@d1", ...) under
  // their module, so all copies share one legend entry and one color
  const modules = new Map(); // "enc:vit" -> {role, name, i, kinds: []}
  for (const kind of kinds) {
    const role = kind.slice(0, 3);
    const name = kind.slice(4).replace(/@d\d+$/, "");
    const key = role + ":" + name;
    if (!modules.has(key)) {
      const list = (role === "enc" ? cfgObj.encoders : cfgObj.decoders) || [];
      modules.set(key, { role, name, i: list.findIndex((m) => (m && m.name) === name), kinds: [] });
    }
    modules.get(key).kinds.push(kind);
  }
  const seen = { enc: 0, dec: 0 };
  const entries = [...modules.values()]
    .sort((a, b) => (a.role === b.role ? a.i - b.i : a.role === "enc" ? -1 : 1));
  for (const ent of entries) {
    if (ent.i < 0) ent.i = seen[ent.role]++; // config out of sync: order seen
    const pal = AUX_PALETTES[ent.role];
    if (!pal) continue;
    const colors = pal[ent.i % pal.length];
    for (const kind of ent.kinds) auxKindPalette[kind] = colors;
    const copies = ent.kinds.length > 1 ? `, ${ent.kinds.length} copies` : "";
    const key = document.createElement("span");
    key.className = "key";
    key.title = `${ent.name} (${ent.role === "enc" ? "encoder" : "decoder"} ${ent.i + 1}${copies}); ` +
      "its F/B/W share this palette";
    const sw = document.createElement("i");
    sw.style.setProperty("--c", colors.B);
    key.appendChild(sw);
    key.appendChild(document.createTextNode(`${ent.role === "enc" ? "Enc" : "Dec"}${ent.i + 1}`));
    legend.appendChild(key);
  }
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
  gantt = { data, canvas, ctx: canvas.getContext("2d"), range: [0, 100], ro: null,
            hoverX: null, hoverMid: null, pinnedMids: new Set() };

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
    gantt.hoverX = ev.clientX; // anchor for W/S keyboard zoom
    if (selecting) return;
    const rect = canvas.getBoundingClientRect();
    const hit = ganttHit(ev.clientX - rect.left, ev.clientY - rect.top);
    const hoverMid = hit ? hit.b[3] : null;
    if (hoverMid !== gantt.hoverMid) { gantt.hoverMid = hoverMid; drawGantt(); }
    if (!hit) { tip.style.display = "none"; return; }
    const [s, e, w, mid, sid, kind] = hit.b;
    const who = kind
      ? `${kind.slice(4)} (${kind.startsWith("enc") ? "encoder" : "decoder"})`
      : `sid=${sid}`;
    tip.textContent =
      `${w} mid=${mid} ${who} D${hit.dev.did}  ` +
      `${Math.round(s).toLocaleString()} – ${Math.round(e).toLocaleString()} (${Math.round(e - s).toLocaleString()})`;
    const hostRect = host.getBoundingClientRect();
    let lx = ev.clientX - hostRect.left + 14, ly = ev.clientY - hostRect.top + 14;
    tip.style.display = "block";
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (lx + tw > hostRect.width) lx = Math.max(0, ev.clientX - hostRect.left - tw - 10);
    if (ly + th > hostRect.height) ly = Math.max(0, ev.clientY - hostRect.top - th - 10);
    tip.style.left = lx + "px"; tip.style.top = ly + "px";
  });
  canvas.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    gantt.hoverX = null;
    if (gantt.hoverMid != null) { gantt.hoverMid = null; drawGantt(); }
  });

  /* perfetto-style navigation: drag pans, shift+drag selects a region to
     zoom into, wheel zooms around the cursor, horizontal wheel pans */
  canvas.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const hostRect = host.getBoundingClientRect();
    tip.style.display = "none";

    if (!ev.shiftKey) { // drag = pan; a no-move click (un)pins the micro batch
      selecting = { pan: true };
      canvas.style.cursor = "grabbing";
      const startX = ev.clientX, startY = ev.clientY;
      let moved = false;
      const [f0, t0] = gantt.range;
      const span = t0 - f0;
      const plotW = ganttGeom().plotW;
      function move(e) {
        if (Math.abs(e.clientX - startX) > 3 || Math.abs(e.clientY - startY) > 3) moved = true;
        const dPct = (-(e.clientX - startX) / plotW) * span;
        const nf = Math.max(0, Math.min(100 - span, f0 + dPct));
        setViewRange(nf, nf + span);
      }
      function up() {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        selecting = null;
        canvas.style.cursor = "";
        if (!moved) { // click: toggle that path's pin; empty space clears all
          const hit = ganttHit(startX - rect.left, startY - rect.top);
          if (hit) {
            const mid = hit.b[3];
            if (gantt.pinnedMids.has(mid)) gantt.pinnedMids.delete(mid);
            else gantt.pinnedMids.add(mid);
          } else {
            gantt.pinnedMids.clear();
          }
          drawGantt();
        }
      }
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
      return;
    }

    // shift+drag = region select-to-zoom
    selecting = { startX: ev.clientX };
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

  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault(); // keep the page/panel from scrolling
    const g = ganttGeom();
    const [f0, t0] = gantt.range;
    const span = t0 - f0;

    // trackpad horizontal scroll (or shift+wheel) pans
    const dx = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX
             : ev.shiftKey ? ev.deltaY : 0;
    if (dx !== 0) {
      const dPct = (dx / g.plotW) * span;
      const nf = Math.max(0, Math.min(100 - span, f0 + dPct));
      setViewRange(nf, nf + span);
      return;
    }

    // vertical scroll zooms, keeping the time under the cursor fixed
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(GANTT_GUTTER, Math.min(ev.clientX - rect.left, GANTT_GUTTER + g.plotW));
    const anchor = f0 + ((mx - GANTT_GUTTER) / g.plotW) * span; // pct under cursor
    const dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY; // line -> px
    const factor = Math.exp(dy * 0.0015); // <1 zoom in, >1 zoom out
    const newSpan = Math.max(MIN_SPAN_PCT, Math.min(100, span * factor));
    let nf = anchor - ((anchor - f0) / span) * newSpan;
    nf = Math.max(0, Math.min(100 - newSpan, nf));
    setViewRange(nf, nf + newSpan);
  }, { passive: false });
}
$("zoom-reset").addEventListener("click", () => setViewRange(0, 100));

$("hl-reset").addEventListener("click", () => {
  if (!gantt) return;
  gantt.pinnedMids.clear();
  drawGantt();
});

/* WASD navigation (perfetto-style): W zoom in / S zoom out around the
   hovered time (window center when the mouse is off the chart), A/D pan.
   Held keys are animated in a rAF loop with per-frame dt, so motion is
   smooth and independent of the OS key-repeat rate.  Keys are ignored
   while typing in a form control. */
const NAV_PAN_SPAN_PER_S = 1.4; // view-widths per second
const NAV_ZOOM_PER_S = 3.5;     // zoom factor per second held
const navKeys = new Set();
let navRaf = null, navLastTs = 0;

function navAnchorPct(f0, span) {
  if (gantt.hoverX == null) return f0 + span / 2;
  const g = ganttGeom();
  const rect = gantt.canvas.getBoundingClientRect();
  const mx = Math.max(GANTT_GUTTER, Math.min(gantt.hoverX - rect.left, GANTT_GUTTER + g.plotW));
  return f0 + ((mx - GANTT_GUTTER) / g.plotW) * span;
}

function navStep(now) {
  navRaf = null;
  if (!gantt || !navKeys.size) return;
  const dt = Math.min((now - navLastTs) / 1000, 0.05); // clamp hitchy frames
  navLastTs = now;
  const [f0, t0] = gantt.range;
  const span = t0 - f0;
  let newSpan = span, nf = f0;

  let zoom = 1;
  if (navKeys.has("w")) zoom /= NAV_ZOOM_PER_S;
  if (navKeys.has("s")) zoom *= NAV_ZOOM_PER_S;
  if (zoom !== 1) {
    newSpan = Math.max(MIN_SPAN_PCT, Math.min(100, span * Math.pow(zoom, dt)));
    const anchor = navAnchorPct(f0, span);
    nf = anchor - ((anchor - f0) / span) * newSpan;
  }
  if (navKeys.has("a")) nf -= newSpan * NAV_PAN_SPAN_PER_S * dt;
  if (navKeys.has("d")) nf += newSpan * NAV_PAN_SPAN_PER_S * dt;

  nf = Math.max(0, Math.min(100 - newSpan, nf));
  setViewRange(nf, nf + newSpan);
  navRaf = requestAnimationFrame(navStep);
}

document.addEventListener("keydown", (ev) => {
  if (!gantt) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key.toLowerCase();
  if (k !== "w" && k !== "a" && k !== "s" && k !== "d") return;
  const t = ev.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
  ev.preventDefault();
  if (ev.repeat) return; // the rAF loop animates held keys itself
  navKeys.add(k);
  if (!navRaf) { navLastTs = performance.now(); navRaf = requestAnimationFrame(navStep); }
});
document.addEventListener("keyup", (ev) => navKeys.delete(ev.key.toLowerCase()));
window.addEventListener("blur", () => navKeys.clear());

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

/* Session-wide run history.  The array is chronological with the current
   run last; the dropdown pins the current run on top (not clickable) and
   lists the older runs below it in time order.  Restoring an entry moves
   it back to the current slot instead of appending a duplicate. */
const runHistory = []; // { ms, yaml, at }
let pendingRestore = null; // entry the in-flight run is restoring, if any
function renderHistory() {
  $("hist-wrap").hidden = runHistory.length === 0;
  const pop = $("hist-pop");
  pop.textContent = "";
  if (!runHistory.length) return;
  const order = [runHistory.length - 1, ...runHistory.slice(0, -1).keys()];
  for (const i of order) {
    const h = runHistory[i];
    const isCur = i === runHistory.length - 1;
    const row = document.createElement("div");
    row.className = "hist-row" + (isCur ? " current" : "");
    const idx = document.createElement("span");
    idx.className = "hist-idx";
    idx.textContent = `#${i + 1}`;
    const ms = document.createElement("b");
    ms.textContent = `${h.ms.toFixed(2)} ms`;
    const at = document.createElement("span");
    at.className = "hist-at";
    at.textContent = h.at.toTimeString().slice(0, 8);
    row.append(idx, ms, at);
    if (!isCur) {
      row.title = "Restore this run's config";
      row.addEventListener("click", () => {
        pop.hidden = true;
        if (!confirm(`Restore the config of run #${i + 1} (${h.ms.toFixed(2)} ms)?`)) return;
        pendingRestore = h;
        setConfigText(h.yaml);
        clearTimeout(autoRunTimer); // one deliberate run, not the debounced one
        run();
      });
    }
    pop.appendChild(row);
  }
}
$("hist-btn").addEventListener("click", (ev) => {
  ev.stopPropagation();
  const pop = $("hist-pop");
  pop.hidden = !pop.hidden;
  if (!pop.hidden) pop.scrollTop = 0; // current sits on top
});
document.addEventListener("click", (ev) => {
  if (!$("hist-wrap").contains(ev.target)) $("hist-pop").hidden = true;
});

async function run() {
  if (runInflight) { runQueued = true; return; }
  runInflight = true;
  const restoring = pendingRestore; // consumed by this run only
  pendingRestore = null;
  const btn = $("run-btn");
  btn.disabled = true; btn.textContent = "Running...";
  $("toast").style.display = "none";
  // white veil + spinner over the gantt while the backend works; the timer
  // ticks every 0.1s and the final time lands next to the F/B/W legend
  const overlay = $("run-overlay"), timerEl = $("run-timer");
  overlay.style.display = "flex";
  const t0 = performance.now();
  const elapsed = () => ((performance.now() - t0) / 1000).toFixed(1) + " s";
  timerEl.textContent = "0.0 s";
  const timerTick = setInterval(() => { timerEl.textContent = elapsed(); }, 100);
  try {
    await flushDump(); // form edits land in the YAML before running
    const cfgSnapshot = $("config").value; // what History restores later
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: cfgSnapshot }),
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
    // wall-clock generation time + this run's simulated pipeline time; the
    // History dropdown holds every previous run for comparison / restore
    const exeMs = r.makespan / 100;
    $("gen-time").textContent =
      `Generated in ${elapsed()} \u00b7 Exe time: ${exeMs.toFixed(2)} ms`;
    if (restoring && cfgSnapshot === restoring.yaml) {
      // a restore re-promotes the old entry to the current slot, no duplicate
      const k = runHistory.indexOf(restoring);
      if (k >= 0) runHistory.splice(k, 1);
      restoring.ms = exeMs;
      restoring.at = new Date();
      runHistory.push(restoring);
    } else {
      runHistory.push({ ms: exeMs, yaml: cfgSnapshot, at: new Date() });
    }
    renderHistory();
  } catch (e) {
    showError("Request failed: " + e);
  } finally {
    clearInterval(timerTick);
    overlay.style.display = "none";
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
  await loadDefaultConfig();   // loads the default config into form + YAML
  if ($("config").value.trim()) run(); // show results immediately on first visit
})();
