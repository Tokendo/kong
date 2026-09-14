/* Kong web interface.
 *
 * No framework on purpose: the page is served by a standard-library HTTP
 * server out of the Python package, so everything it needs has to be in these
 * three files. It polls /api/state, appends what it has not seen yet, and
 * posts the buttons. */

"use strict";

const POLL_MS = 500;

/* The token is minted per process and arrives in the URL. It is kept in
 * memory and taken out of the address bar, so a screenshot of the window does
 * not hand out the run. */
const TOKEN = (() => {
  const fromUrl = new URLSearchParams(location.search).get("t");
  if (fromUrl) {
    sessionStorage.setItem("kong-token", fromUrl);
    history.replaceState(null, "", location.pathname);
    return fromUrl;
  }
  return sessionStorage.getItem("kong-token") || "";
})();

const $ = (id) => document.getElementById(id);

const ui = {
  providers: [],
  baseUrls: {},        // per provider, so switching back does not retype it
  keys: {},            // typed keys, kept for the session only
  provider: "",
  logCursor: 0,
  resultsCursor: 0,
  conflictsVersion: -1,
  results: [],
  waitBase: 0,
  waitStamp: 0,
  waiting: false,
  running: false,
  graph: null,          // nodes, edges and the adjacency built from them
  graphFocus: 0,
  graphAsked: false,    // the tab reads the file once, then on Reload
};

/* ----------------------------------------------------------------- plumbing */

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "X-Kong-Token": TOKEN, "Content-Type": "application/json",
               ...(options.headers || {}) },
  });
  if (!response.ok && response.status === 403) {
    throw new Error("This page has lost its token. Reopen the URL Kong printed.");
  }
  return response.json();
}

const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body || {}) });

let toastTimer = 0;
function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("toast--error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
}

function announce(result) {
  if (!result) return;
  if (result.message) toast(result.message, result.ok === false);
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function clock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  return m ? `${m}m ${String(total % 60).padStart(2, "0")}s` : `${total}s`;
}

function compact(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

/* ------------------------------------------------------------------ the form */

function formValue() {
  return {
    binary_path: $("binary").value,
    output_dir: $("output").value,
    formats: [...document.querySelectorAll("#formats input:checked")].map((i) => i.value),
    resume: $("resume").checked,
    provider: ui.provider,
    model: $("model").value,
    draft_model: $("draft-model").value,
    refine_below: $("refine-below").value,
    draft_only: $("draft-only").checked,
    base_url: $("base-url").value,
    api_key: $("api-key").value,
    max_prompt_chars: $("max-prompt-chars").value,
    max_chunk_functions: $("max-chunk-functions").value,
    max_output_tokens: $("max-output-tokens").value,
  };
}

function provider(value) {
  return ui.providers.find((p) => p.value === value) || {};
}

function selectProvider(value) {
  if (ui.provider && ui.provider !== value) {
    ui.baseUrls[ui.provider] = $("base-url").value;
    ui.keys[ui.provider] = $("api-key").value;
  }
  ui.provider = value;

  const chosen = provider(value);
  for (const button of document.querySelectorAll("#providers button")) {
    button.setAttribute("aria-checked", String(button.dataset.value === value));
  }

  $("base-url").value = ui.baseUrls[value] || "";
  $("base-url").disabled = !chosen.needs_base_url;
  $("detect").disabled = !chosen.detectable;
  $("api-key").value = ui.keys[value] || "";
  $("model-hint").textContent = chosen.default_model
    ? `Leave empty for ${chosen.default_model}.`
    : "A custom endpoint needs the model name it serves.";
  renderKeyHint();
}

function renderKeyHint() {
  const chosen = provider(ui.provider);
  if ($("api-key").value.trim()) {
    $("key-hint").textContent = "Save keeps this key for the next run.";
  } else if (!chosen.env_var) {
    $("key-hint").textContent = "No key needed for a local server.";
  } else if (chosen.saved_key) {
    $("key-hint").textContent = `Using the key saved for ${chosen.label}.`;
  } else if (chosen.env_key_set) {
    $("key-hint").textContent = `Using ${chosen.env_var} from the environment.`;
  } else {
    $("key-hint").textContent = `No key: set ${chosen.env_var} or save one here.`;
  }
}

function renderFormatHint() {
  const transpiled = [...document.querySelectorAll("#formats input:checked")]
    .map((input) => input.value)
    .filter((value) => value === "python" || value === "csharp");
  $("format-hint").textContent = transpiled.length
    ? "Python/C# is a readable reconstruction, not a runnable port, and costs a second LLM pass over the binary."
    : "";
}

function buildForm(boot) {
  ui.providers = boot.providers;
  ui.baseUrls = { ...boot.base_urls };

  $("providers").innerHTML = "";
  for (const item of boot.providers) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "radio";
    button.dataset.value = item.value;
    button.textContent = item.label;
    button.setAttribute("aria-checked", "false");
    button.addEventListener("click", () => selectProvider(item.value));
    $("providers").append(button);
  }

  $("formats").innerHTML = "";
  for (const format of boot.formats) {
    const chip = document.createElement("label");
    chip.className = "chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = format.value;
    input.checked = boot.default_formats.includes(format.value);
    input.addEventListener("change", renderFormatHint);
    chip.append(input, document.createTextNode(format.label));
    $("formats").append(chip);
  }

  $("binary").value = boot.binary_path || "";
  $("output").value = boot.output_dir || "";
  $("refine-below").value = boot.default_refine_below;

  restore(boot.remembered || {});
  selectProvider(ui.provider || boot.providers[0].value);
  renderFormatHint();
}

/** Put back what the last run was configured with, minus the key. */
function restore(saved) {
  const text = {
    binary_path: "binary", output_dir: "output", model: "model",
    draft_model: "draft-model", refine_below: "refine-below",
    max_prompt_chars: "max-prompt-chars",
    max_chunk_functions: "max-chunk-functions",
    max_output_tokens: "max-output-tokens",
  };
  for (const [key, id] of Object.entries(text)) {
    if (saved[key] !== undefined && saved[key] !== null) $(id).value = saved[key];
  }
  if (typeof saved.resume === "boolean") $("resume").checked = saved.resume;
  if (typeof saved.draft_only === "boolean") $("draft-only").checked = saved.draft_only;
  if (Array.isArray(saved.formats) && saved.formats.length) {
    for (const input of document.querySelectorAll("#formats input")) {
      input.checked = saved.formats.includes(input.value);
    }
  }
  if (saved.base_url && saved.provider) ui.baseUrls[saved.provider] = saved.base_url;
  if (saved.provider && provider(saved.provider).value) ui.provider = saved.provider;
}

/* ---------------------------------------------------------------- rendering */

function renderWaiting(state) {
  ui.waiting = state.llm_waiting;
  ui.waitBase = state.llm_wait_seconds || 0;
  ui.waitStamp = performance.now();

  const pill = $("llm-pill");
  const banner = $("waiting");
  pill.classList.toggle("pill--waiting", state.llm_waiting);
  pill.classList.toggle("pill--error", Boolean(state.error));
  banner.hidden = !state.llm_waiting;

  if (state.llm_waiting) {
    $("waiting-title").textContent =
      state.llm_in_flight > 1
        ? `Waiting for the model — ${plural(state.llm_in_flight, "request")} in flight`
        : "Waiting for the model";
    $("waiting-detail").textContent = state.llm_wait_label;
  } else if (state.error) {
    $("llm-pill-text").textContent = "Run failed";
  }
  tickClock();
}

/** The seconds keep moving between polls, so the page counts them itself. */
function tickClock() {
  const extra = ui.waiting ? (performance.now() - ui.waitStamp) / 1000 : 0;
  const seconds = ui.waitBase + extra;
  if (ui.waiting) {
    $("waiting-clock").textContent = clock(seconds);
    $("llm-pill-text").textContent = `Waiting on the model · ${clock(seconds)}`;
  } else if (!$("llm-pill").classList.contains("pill--error")) {
    $("llm-pill-text").textContent = ui.lastWait
      ? `Model idle · last answer in ${ui.lastWait.toFixed(1)}s`
      : "Model idle";
  }
}

function tile(label, value, sub) {
  return `<div class="tile"><span class="tile-label">${label}</span>` +
         `<span class="tile-value">${value}` +
         (sub ? ` <small>${sub}</small>` : "") + `</span></div>`;
}

function renderTiles(state) {
  const waitShare = state.elapsed_seconds > 0
    ? Math.round((state.llm_wait_total_seconds / state.elapsed_seconds) * 100)
    : 0;
  $("tiles").innerHTML = [
    tile("Functions", `${state.completed}`, `/ ${state.total}`),
    tile("Confidence", `${state.high_confidence}`,
         `${state.medium_confidence} med · ${state.low_confidence} low`),
    tile("LLM calls", `${state.llm_calls}`,
         state.llm_in_flight ? `${state.llm_in_flight} in flight` : ""),
    tile("Tokens", compact(state.input_tokens), `in · ${compact(state.output_tokens)} out`),
    tile("Cost", `$${state.cost_usd.toFixed(4)}`),
    tile("Elapsed", clock(state.elapsed_seconds)),
    tile("Waiting on model", clock(state.llm_wait_total_seconds),
         state.elapsed_seconds > 0 ? `${waitShare}%` : ""),
    tile("To finish", `${state.pending_finish}`),
  ].join("");
}

function renderStatus(state) {
  const status = $("status");
  if (state.error) {
    status.textContent = `Failed: ${state.error}`;
    status.classList.add("status--error");
    return;
  }
  status.classList.remove("status--error");

  if (!ui.running && !state.finished && state.phase === "idle") {
    status.textContent = "Idle.";
    return;
  }
  const pieces = [`Phase: ${state.phase}`];
  if (state.binary_label) pieces.push(state.binary_label);
  if (state.paused) pieces.push("PAUSED");
  if (state.finishing) pieces.push("FINISHING PASS");
  if (state.checking_coherence) pieces.push("CHECKING COHERENCE");
  status.textContent = pieces.join("   ");
}

function renderButtons(state, hasController) {
  const busy = state.finishing || state.checking_coherence;
  $("start").disabled = state.running;
  $("pause").disabled = !state.running;
  $("pause").textContent = state.paused ? "Resume" : "Pause";
  $("export").disabled = !hasController;
  $("finish").disabled = !hasController || busy || !state.pending_finish;
  $("coherence").disabled = !hasController || busy;
}

function appendLog(entries) {
  if (!entries.length) return;
  const log = $("log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    const line = document.createElement("span");
    line.className = entry.tag;
    line.textContent = entry.message;
    fragment.append(line);
  }
  log.append(fragment);
  // Only follow the tail when the reader is already there; scrolling back to
  // read an error should not be undone by the next function that finishes.
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function functionRow(result) {
  const grade = result.confidence >= 80 ? "high" : result.confidence < 50 ? "low" : "";
  const cell = (value, cls = "") => {
    const td = document.createElement("td");
    if (cls) td.className = cls;
    td.textContent = value;
    return td;
  };
  const row = document.createElement("tr");
  row.append(
    cell(result.address, "mono"),
    cell(result.original, "mono"),
    cell(result.name),
    cell(`${result.confidence}%`, grade),
    cell(result.classification),
  );
  row.dataset.haystack =
    `${result.address} ${result.original} ${result.name} ${result.classification}`.toLowerCase();
  return row;
}

function appendResults(results) {
  if (!results.length) return;
  ui.results.push(...results);
  const filter = $("filter").value.trim().toLowerCase();
  const fragment = document.createDocumentFragment();
  for (const result of results) {
    const row = functionRow(result);
    if (filter && !row.dataset.haystack.includes(filter)) row.hidden = true;
    fragment.append(row);
  }
  $("functions").append(fragment);
  $("count-functions").textContent = String(ui.results.length);
}

function renderConflicts(conflicts) {
  const body = $("conflicts");
  body.innerHTML = "";
  for (const conflict of conflicts) {
    const row = document.createElement("tr");
    for (const [value, cls] of [
      [conflict.kind, ""],
      [conflict.functions, "mono"],
      [conflict.summary, ""],
      [conflict.resolution || "pending", conflict.resolution ? "" : "muted"],
    ]) {
      const td = document.createElement("td");
      td.className = cls;
      td.textContent = value;
      row.append(td);
    }
    body.append(row);
  }
  $("count-conflicts").textContent = String(conflicts.length);
}

/* --------------------------------------------------------------------- poll */

async function poll() {
  let payload;
  try {
    payload = await api(
      `/api/state?log=${ui.logCursor}&results=${ui.resultsCursor}` +
      `&conflicts=${ui.conflictsVersion}`
    );
  } catch (error) {
    $("llm-pill-text").textContent = "Disconnected from Kong";
    $("llm-pill").classList.add("pill--error");
    return;
  }

  const state = payload.state;
  ui.running = state.running;
  ui.lastWait = state.llm_last_wait_seconds;
  ui.pendingFinish = state.pending_finish;
  ui.logCursor = payload.log_cursor;
  ui.resultsCursor = payload.results_cursor;

  appendLog(payload.log);
  appendResults(payload.results);
  if (payload.conflicts) {
    renderConflicts(payload.conflicts);
    ui.conflictsVersion = payload.conflicts_version;
  }

  $("progress-bar").style.width = `${(state.progress_fraction * 100).toFixed(1)}%`;
  renderStatus(state);
  renderButtons(state, payload.has_controller);
  renderTiles(state);
  renderWaiting(state);
}

/* ------------------------------------------------------------------- picker */

/* ------------------------------------------------------------- call graph */

/* The graph is read from analysis.json rather than from the running job: that
   is where it is complete, and a page opened long after the run gets the same
   answer as one that watched it. */

const GRAPH_NEIGHBOURS = 9;
const SVG_NS = "http://www.w3.org/2000/svg";
const svgEl = (name, attrs) => {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  return node;
};

function confidenceClass(node) {
  if (node.u || node.q < 40) return "lo";
  return node.q >= 80 ? "hi" : "md";
}

function confidenceInk(node) {
  return { hi: "#26bfb5", md: "rgba(255,255,255,.68)", lo: "#ff5772" }[
    confidenceClass(node)
  ];
}

async function loadGraph() {
  ui.graphAsked = true;
  $("graph-hint").hidden = false;
  $("graph-hint").textContent = "Reading the call graph\u2026";
  $("graph").hidden = true;

  // The output directory on the form, not just the running job's: the common
  // case is reopening the page to read a graph a finished run left on disk.
  const where = $("output").value.trim();
  let payload;
  try {
    payload = await api(
      where ? `/api/graph?path=${encodeURIComponent(where)}` : "/api/graph"
    );
  } catch (error) {
    $("graph-hint").textContent = String(error.message || error);
    return;
  }

  if (!payload.ok) {
    $("graph-hint").textContent = payload.message || "No call graph to show.";
    $("graph-filter").disabled = true;
    ui.graph = null;
    return;
  }

  const callers = payload.nodes.map(() => []);
  const callees = payload.nodes.map(() => []);
  for (const [from, to] of payload.edges) {
    callees[from].push(to);
    callers[to].push(from);
  }
  ui.graph = { ...payload, callers, callees };

  $("graph-filter").disabled = false;
  $("graph-hint").hidden = true;
  $("graph").hidden = false;

  // Open on the busiest function: the one place the shape of the program shows.
  let best = 0;
  for (let i = 0; i < payload.nodes.length; i += 1) {
    if (callers[i].length + callees[i].length >
        callers[best].length + callees[best].length) {
      best = i;
    }
  }
  focusNode(best);   // renders the list itself
}

function graphMatches() {
  const needle = $("graph-filter").value.trim().toLowerCase();
  const { nodes, callers, callees } = ui.graph;
  const kept = [];
  for (let i = 0; i < nodes.length; i += 1) {
    if (!needle
        || nodes[i].n.toLowerCase().includes(needle)
        || nodes[i].a.includes(needle)) {
      kept.push(i);
    }
  }
  kept.sort((a, b) =>
    (callers[b].length + callees[b].length) - (callers[a].length + callees[a].length)
  );
  return kept;
}

function renderGraphList() {
  if (!ui.graph) return;
  const list = $("graph-list");
  const { nodes, callers, callees } = ui.graph;
  const kept = graphMatches();
  const shown = kept.slice(0, 300);

  list.textContent = "";
  for (const i of shown) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-item";
    if (i === ui.graphFocus) button.setAttribute("aria-current", "true");

    const name = document.createElement("span");
    name.className = `graph-item-name q-${confidenceClass(nodes[i])}`;
    name.textContent = nodes[i].n;

    const meta = document.createElement("span");
    meta.className = "graph-item-meta";
    meta.textContent =
      `${nodes[i].a} \u00b7 ${callers[i].length}\u2191 ${callees[i].length}\u2193`
      + (nodes[i].u ? " \u00b7 not analyzed" : ` \u00b7 ${nodes[i].q}%`);

    button.append(name, meta);
    button.addEventListener("click", () => focusNode(i));
    item.append(button);
    list.append(item);
  }

  if (kept.length > shown.length) {
    const rest = document.createElement("li");
    rest.className = "graph-more";
    rest.textContent = `${kept.length - shown.length} more \u2014 narrow the filter.`;
    list.append(rest);
  }
  if (!kept.length) {
    const none = document.createElement("li");
    none.className = "graph-more";
    none.textContent = "Nothing matches that.";
    list.append(none);
  }
}

function focusNode(index) {
  ui.graphFocus = index;
  renderGraphDetail();
  drawGraph();
  renderGraphList();
}

function renderGraphDetail() {
  const { nodes, callers, callees } = ui.graph;
  const node = nodes[ui.graphFocus];
  const detail = $("graph-detail");
  detail.textContent = "";

  const title = document.createElement("h3");
  title.className = "graph-title";
  title.textContent = node.n;

  const meta = document.createElement("p");
  meta.className = "graph-meta";
  const bits = [node.a];
  if (node.c) bits.push(node.c);
  bits.push(node.u
    ? "never analyzed"
    : `confidence ${node.q}%`);
  bits.push(`${plural(callers[ui.graphFocus].length, "caller")}`);
  bits.push(`${plural(callees[ui.graphFocus].length, "callee")}`);
  meta.textContent = bits.join(" \u00b7 ");

  detail.append(title, meta);
}

function drawGraph() {
  const svg = $("graph-svg");
  const { nodes, callers, callees } = ui.graph;
  const focus = ui.graphFocus;
  svg.textContent = "";

  const BOX_W = 188;
  const BOX_H = 42;
  const GAP = 14;
  const ROW = 116;

  const up = callers[focus].slice(0, GRAPH_NEIGHBOURS);
  const down = callees[focus].slice(0, GRAPH_NEIGHBOURS);
  const columns = Math.max(up.length, down.length, 1);
  const width = Math.max(columns * (BOX_W + GAP) + GAP, 520);
  const height = ROW * 2 + BOX_H + 52;

  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const rowY = (which) => (which === "up" ? 24 : which === "self" ? 24 + ROW : 24 + ROW * 2);
  const place = (list, which) => {
    const span = list.length * BOX_W + (list.length - 1) * GAP;
    const left = (width - span) / 2;
    return list.map((index, position) => ({
      index, x: left + position * (BOX_W + GAP), y: rowY(which),
    }));
  };

  const upBoxes = place(up, "up");
  const downBoxes = place(down, "down");
  const self = { index: focus, x: (width - BOX_W) / 2, y: rowY("self") };

  const link = (from, to) => {
    const x1 = from.x + BOX_W / 2;
    const y1 = from.y + BOX_H;
    const x2 = to.x + BOX_W / 2;
    const y2 = to.y;
    const mid = (y1 + y2) / 2;
    svg.append(svgEl("path", {
      d: `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`,
      fill: "none", stroke: "rgba(255,255,255,.28)", "stroke-width": "1.2",
    }));
  };
  upBoxes.forEach((box) => link(box, self));
  downBoxes.forEach((box) => link(self, box));

  const box = (spot, isSelf) => {
    const node = nodes[spot.index];
    const group = svgEl("g", { class: "graph-node" });
    if (!isSelf) {
      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "button");
    }
    group.append(svgEl("rect", {
      x: spot.x, y: spot.y, width: BOX_W, height: BOX_H, rx: 6,
      fill: isSelf ? "#14425f" : "#0f3550",
      stroke: isSelf ? "#26bfb5" : "rgba(255,255,255,.28)",
      "stroke-width": isSelf ? "2" : "1",
    }));

    // The monospace face is set in the stylesheet: a presentation attribute
    // does not resolve a custom property.
    const label = svgEl("text", {
      class: "graph-node-name",
      x: spot.x + 10, y: spot.y + 18, fill: confidenceInk(node),
      "font-size": "12",
    });
    label.textContent = node.n.length > 25 ? `${node.n.slice(0, 24)}\u2026` : node.n;

    const under = svgEl("text", {
      x: spot.x + 10, y: spot.y + 33,
      fill: "rgba(255,255,255,.45)", "font-size": "10.5",
    });
    under.textContent = node.u
      ? `${node.a} \u00b7 not analyzed`
      : [node.a, node.c].filter(Boolean).join(" \u00b7 ");

    const tip = svgEl("title");
    tip.textContent = node.u ? `${node.n} — never analyzed` : `${node.n} — ${node.q}%`;

    group.append(label, under, tip);
    if (!isSelf) {
      group.addEventListener("click", () => focusNode(spot.index));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          focusNode(spot.index);
        }
      });
    }
    svg.append(group);
  };

  upBoxes.forEach((spot) => box(spot, false));
  box(self, true);
  downBoxes.forEach((spot) => box(spot, false));

  const caption = (text, y) => {
    const node = svgEl("text", {
      x: 10, y, fill: "rgba(255,255,255,.45)",
      "font-size": "10.5", "letter-spacing": "1.1",
    });
    node.textContent = text;
    svg.append(node);
  };
  const more = (count) =>
    count > GRAPH_NEIGHBOURS ? ` (${GRAPH_NEIGHBOURS} shown)` : "";
  caption(`CALLERS ${callers[focus].length}${more(callers[focus].length)}`, 15);
  caption(`CALLEES ${callees[focus].length}${more(callees[focus].length)}`, rowY("down") - 9);
}

const picker = {
  target: null,
  mode: "file",
  path: "",
};

async function openPicker(targetId, mode) {
  picker.target = targetId;
  picker.mode = mode;
  $("picker-title").textContent =
    mode === "dir" ? "Select a directory" : "Select a binary";
  $("picker-pick").hidden = mode !== "dir";
  await showDirectory($(targetId).value || "");
  $("picker").showModal();
}

async function showDirectory(path) {
  const listing = await api(`/api/browse?path=${encodeURIComponent(path)}`);
  picker.path = listing.path;
  $("picker-path").textContent = listing.path;
  $("picker-up").disabled = !listing.parent;
  $("picker-up").dataset.path = listing.parent || "";

  const list = $("picker-list");
  list.innerHTML = "";
  if (listing.error) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = listing.error;
    list.append(item);
    return;
  }
  for (const entry of listing.entries) {
    if (!entry.is_dir && picker.mode === "dir") continue;
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML =
      `<span class="${entry.is_dir ? "dir" : ""}">${entry.is_dir ? "▸ " : ""}</span>`;
    button.append(document.createTextNode(entry.name));
    if (!entry.is_dir) {
      const size = document.createElement("span");
      size.className = "size";
      size.textContent = `${compact(entry.size)}B`;
      button.append(size);
    }
    button.addEventListener("click", () => {
      if (entry.is_dir) {
        showDirectory(entry.path);
      } else {
        choose(entry.path);
      }
    });
    item.append(button);
    list.append(item);
  }
  if (!list.children.length) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = "Nothing here.";
    list.append(item);
  }
}

function choose(path) {
  $(picker.target).value = path;
  if (picker.target === "binary") {
    // The output directory follows the binary, unless it has been set by hand.
    const stem = path.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
    $("output").value = $("output").value.replace(/kong_output(_[^\\/]*)?$/, `kong_output_${stem}`);
  }
  $("picker").close();
}

/* ------------------------------------------------------------------ actions */

function wire() {
  $("start").addEventListener("click", async () => {
    $("start").disabled = true;
    const result = await post("/api/start", formValue());
    announce(result);
    if (result.ok) {
      ui.logCursor = 0;
      ui.resultsCursor = 0;
      ui.conflictsVersion = -1;
      ui.results = [];
      $("log").innerHTML = "";
      $("functions").innerHTML = "";
      $("conflicts").innerHTML = "";
      $("count-functions").textContent = "0";
      $("count-conflicts").textContent = "0";
    } else {
      $("start").disabled = false;
    }
    poll();
  });

  $("pause").addEventListener("click", async () => announce(await post("/api/pause")));
  $("export").addEventListener("click", async () => announce(await post("/api/export")));
  $("coherence").addEventListener("click", async () =>
    announce(await post("/api/coherence")));

  $("finish").addEventListener("click", async () => {
    // The expensive half of a draft run, so it is asked about rather than
    // launched by a stray click.
    const pending = ui.pendingFinish || 0;
    const question =
      `Re-read ${plural(pending, "function")} one at a time with the main ` +
      `model, then redo cleanup, synthesis and export?`;
    if (pending && !confirm(question)) return;
    announce(await post("/api/finish"));
  });

  $("detect").addEventListener("click", async () => {
    $("detect").disabled = true;
    const result = await post("/api/detect", { base_url: $("base-url").value.trim() });
    if (result.limits) {
      $("max-prompt-chars").value = result.limits.max_prompt_chars;
      $("max-chunk-functions").value = result.limits.max_chunk_functions;
      $("max-output-tokens").value = result.limits.max_output_tokens;
    }
    if (result.models && result.models.length && !$("model").value.trim()) {
      $("model").value = result.models[0];
    }
    announce(result);
    $("detect").disabled = !provider(ui.provider).detectable;
  });

  $("save-key").addEventListener("click", async () => {
    const result = await post("/api/key", {
      provider: ui.provider, key: $("api-key").value.trim(),
    });
    announce(result);
    if (result.ok) {
      provider(ui.provider).saved_key = Boolean($("api-key").value.trim());
      renderKeyHint();
    }
  });

  $("quit").addEventListener("click", async () => {
    if (ui.running && !confirm("An analysis is running. Stop it and close Kong?")) return;
    await post("/api/quit");
    document.body.innerHTML =
      '<p style="padding:40px">Kong has stopped. You can close this tab.</p>';
  });

  $("api-key").addEventListener("input", renderKeyHint);

  for (const button of document.querySelectorAll("[data-browse]")) {
    button.addEventListener("click", () =>
      openPicker(button.dataset.browse, button.dataset.mode));
  }
  $("picker-up").addEventListener("click", () => showDirectory($("picker-up").dataset.path));
  $("picker-pick").addEventListener("click", () => choose(picker.path));

  $("filter").addEventListener("input", () => {
    const needle = $("filter").value.trim().toLowerCase();
    for (const row of $("functions").children) {
      row.hidden = Boolean(needle) && !row.dataset.haystack.includes(needle);
    }
  });

  $("graph-reload").addEventListener("click", loadGraph);
  $("graph-filter").addEventListener("input", renderGraphList);

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      for (const other of document.querySelectorAll(".tab")) {
        other.classList.toggle("tab--on", other === tab);
      }
      for (const name of ["log", "functions", "coherence", "graph"]) {
        $(`pane-${name}`).hidden = name !== tab.dataset.tab;
      }
      // Read on first open rather than on every run: the file is only written
      // at export, and it can be a few hundred kilobytes.
      if (tab.dataset.tab === "graph" && !ui.graphAsked) loadGraph();
    });
  }
}

async function main() {
  wire();
  try {
    buildForm(await api("/api/bootstrap"));
  } catch (error) {
    toast(String(error.message || error), true);
    return;
  }
  await poll();
  setInterval(poll, POLL_MS);
  setInterval(tickClock, 250);
}

main();
