/* adbagent web UI. No framework: fetch + EventSource against the FastAPI backend. */

"use strict";

const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const text = await res.text();
  let body;
  try { body = JSON.parse(text); } catch { body = text; }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : text;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

let noticeTimer = null;
function notice(msg, isError = true) {
  const el = $("notice");
  el.textContent = msg;
  el.className = "notice show " + (isError ? "error" : "ok");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { el.className = "notice"; }, 6000);
}

function fmtTime(epoch) {
  if (!epoch) return "";
  const d = new Date(epoch * 1000);
  return d.toLocaleString();
}

function fmtDur(s) {
  s = Math.round(s || 0);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

/* ---------------------------------------------------------------- tabs */

const loadedTabs = new Set();
const tabLoaders = {
  run: () => {},
  history: loadRuns,
  devices: loadDevices,
  config: loadConfig,
  skills: loadSkills,
};

$("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (!btn) return;
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll("section.tab").forEach((s) =>
    s.classList.toggle("active", s.id === "tab-" + btn.dataset.tab));
  const name = btn.dataset.tab;
  if (!loadedTabs.has(name)) {
    loadedTabs.add(name);
    tabLoaders[name]().catch((err) => notice(err.message));
  }
});

/* ------------------------------------------------------- event cards */

function actionSummary(a) {
  if (!a || typeof a !== "object") return "";
  const kind = a.action || "?";
  const bits = [];
  const t = a.target;
  if (t && typeof t === "object") {
    if (t.text) bits.push(`"${t.text}"`);
    else if (t.resource_id) bits.push(`#${t.resource_id}`);
    else if (t.index != null) bits.push(`#${t.index}`);
  }
  if (a.text) bits.push(kind === "open_app" ? a.text : `"${a.text}"`);
  if (a.direction) bits.push(a.direction);
  if (a.key) bits.push(a.key);
  return `${kind}${bits.length ? " " + bits.join(" ") : ""}`;
}

const GRADE_CLASSES = { worked: "worked", no_change: "no_change" };

function renderEvent(ev) {
  const kind = ev.kind || "";
  const div = document.createElement("div");

  if (kind === "run_start") {
    div.className = "banner";
    div.innerHTML = `<b>goal</b> ${esc(ev.goal)}<br><span class="small">${esc(ev.model || "")}</span>`;
  } else if (kind === "run_resume") {
    // Where two sittings of one run join: the events above it are the failed
    // attempt, the ones below are the continuation.
    div.className = "banner";
    div.innerHTML = `<b>resumed from step ${esc(ev.resumed_at_step || 0)}</b>` +
      `<br><span class="small">${esc(ev.model || "")}</span>`;
  } else if (kind === "decide") {
    const a = ev.action || {};
    const llm = ev.llm || {};
    div.className = "card";
    div.innerHTML =
      `<div class="head"><span class="stepno">step ${esc(ev.step)}</span>` +
      `<span class="action">${esc(actionSummary(a))}</span>` +
      (ev.effort ? `<span class="chip neutral">${esc(ev.effort)}</span>` : "") +
      (ev.screenshot ? `<span class="chip neutral">vision</span>` : "") +
      `</div>` +
      (a.observation ? `<div class="obs">${esc(a.observation)}</div>` : "") +
      (a.reasoning ? `<div class="why">${esc(a.reasoning)}</div>` : "") +
      `<div class="meta">${ev.wall_s ? ev.wall_s.toFixed(1) + "s · " : ""}` +
      `${llm.n_calls || 0} call(s) · $${(llm.usd || 0).toFixed(4)}` +
      (a.confidence ? ` · ${esc(a.confidence)}` : "") + `</div>`;
  } else if (kind === "verify") {
    const cls = GRADE_CLASSES[ev.grade] || "failed";
    div.className = "card";
    div.innerHTML = `<div class="head"><span class="stepno">step ${esc(ev.step)}</span>` +
      `<span class="chip ${cls}">${esc(ev.grade)}</span></div><div class="why">${esc(ev.reason || "")}</div>`;
  } else if (kind === "judge") {
    div.className = "card";
    div.innerHTML = `<div class="head"><span class="stepno">step ${esc(ev.step)}</span>` +
      `<span class="chip ${ev.satisfied ? "worked" : "failed"}">judge: ${ev.satisfied ? "satisfied" : "not yet"}</span></div>` +
      `<div class="why">${esc(ev.evidence || "")}</div>`;
  } else if (kind === "image_analysis") {
    div.className = "card";
    div.innerHTML = `<details><summary>vision read · step ${esc(ev.step)} · ${esc(ev.model || "")}</summary>` +
      `<div class="why" style="margin-top:6px">${esc(ev.result || "")}</div></details>`;
  } else if (kind === "run_end") {
    div.className = "banner " + esc(ev.outcome || "");
    div.innerHTML = `<b>${esc((ev.outcome || "").toUpperCase())}</b> — ${esc(ev.steps)} steps, ` +
      `${esc(ev.llm_calls)} LLM calls, $${(ev.usd || 0).toFixed(4)}`;
  } else if (kind === "active_skill") {
    div.className = "small mono";
    div.textContent = `skill loaded: ${ev.name} (${ev.package})`;
  } else if (kind === "scratchpad") {
    div.className = "small mono";
    div.textContent = `collected: ${(ev.keys || []).join(", ")} (${ev.total} total)`;
  } else if (kind === "dead_ends") {
    div.className = "small mono";
    div.textContent = `dead ends avoided: ${(ev.remembered || []).join(", ")}`;
  } else {
    div.className = "small mono";
    div.textContent = kind;
  }
  return div;
}

function updateCountersFromEvent(ev) {
  if (typeof ev.step === "number") {
    live.step = Math.max(live.step, ev.step);
  }
  const llm = ev.llm;
  if (llm && typeof llm === "object") {
    live.calls += llm.n_calls || 0;
    live.cost += llm.usd || 0;
  }
  paintCounters();
}

/* --------------------------------------------------------- llm stream */

/* One collapsible panel per LLM call: it opens when the call starts, the
   model's raw thinking and response stream into it live, and it folds itself
   away the moment the call ends -- the decide/verify cards carry the result.
   Panels are keyed off order, not step: calls within a step are sequential
   (a vision read, then the decision), so at most one is ever open per feed,
   tracked as feed._llm. */

const LLM_PURPOSES = { decide: "decide", judge: "judge",
  analyze_image: "vision read", read_item: "item read" };

function fmtTok(n) {
  n = n || 0;
  return n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n);
}

function llmSummary(start, end) {
  let s = `LLM ${LLM_PURPOSES[start.purpose] || start.purpose || "decide"}`;
  if (start.step) s += ` · step ${esc(start.step)}`;
  if (start.model) s += ` · ${esc(String(start.model).split("/").pop())}`;
  if (start.screenshot) s += " · +img";
  if (end) {
    if (end.elapsed) s += ` · ${end.elapsed.toFixed(1)}s`;
    if (end.completion_tokens) {
      s += ` · ${fmtTok(end.prompt_tokens)}→${fmtTok(end.completion_tokens)} tok`;
      if (end.reasoning_tokens) s += ` (${fmtTok(end.reasoning_tokens)} thinking)`;
    }
  } else if (end === null) {
    s += " · interrupted";
  }
  return s;
}

function finalizeLlm(feed, end) {
  const p = feed._llm;
  if (!p) return;
  feed._llm = null;
  p.details.open = false;  // auto-hide: the stream is over, the verdict follows
  p.summary.textContent = llmSummary(p.start, end);
}

function handleLlmEvent(ev, feed) {
  if (ev.kind === "llm_start") {
    finalizeLlm(feed, null);  // a panel the last stream never closed
    const card = document.createElement("div");
    card.className = "card llm-live";
    card.innerHTML =
      `<details open><summary><span class="pulse"></span> ${llmSummary(ev)}</summary>` +
      `<div class="llm-sec thinking"><div class="llm-sec-label">thinking</div>` +
      `<div class="llm-text"></div></div>` +
      `<div class="llm-sec response"><div class="llm-sec-label">response</div>` +
      `<div class="llm-text"></div></div></details>`;
    feed.appendChild(card);
    feed._llm = {
      start: ev,
      details: card.querySelector("details"),
      summary: card.querySelector("summary"),
      thinking: "", response: "",
      thinkingSec: card.querySelector(".llm-sec.thinking"),
      thinkingText: card.querySelector(".llm-sec.thinking .llm-text"),
      responseSec: card.querySelector(".llm-sec.response"),
      responseText: card.querySelector(".llm-sec.response .llm-text"),
    };
  } else if (ev.kind === "llm_stream" && feed._llm) {
    const p = feed._llm;
    const thinking = ev.stream_type === "thinking";
    const key = thinking ? "thinking" : "response";
    p[key] += ev.text || "";
    const sec = thinking ? p.thinkingSec : p.responseSec;
    const text = thinking ? p.thinkingText : p.responseText;
    if (sec.style.display !== "block") sec.style.display = "block";
    text.textContent = p[key];
    text.scrollTop = text.scrollHeight;
  } else if (ev.kind === "llm_end") {
    finalizeLlm(feed, ev);
  }
}

function paintCounters() {
  $("c-step").textContent = live.step;
  $("c-calls").textContent = live.calls;
  $("c-cost").textContent = "$" + live.cost.toFixed(4);
}

/* ------------------------------------------------------------ run tab */

const live = { step: 0, calls: 0, cost: 0, source: null, startedAt: 0, timer: null };

function setRunningUI(running) {
  $("btn-start").disabled = running;
  $("btn-stop").disabled = !running;
  $("c-state").textContent = running ? "running" : "idle";
  $("c-state").style.color = running ? "var(--green)" : "var(--text-dim)";
  clearInterval(live.timer);
  if (running) {
    live.timer = setInterval(() => {
      $("c-elapsed").textContent = fmtDur((Date.now() / 1000) - live.startedAt);
    }, 1000);
  }
}

function openStream() {
  if (live.source) return;
  $("live").style.display = "block";
  const source = new EventSource("/api/runs/stream");
  live.source = source;

  source.addEventListener("run", (e) => {
    const data = JSON.parse(e.data);
    $("c-runid").textContent = data.run_id;
  });
  source.addEventListener("event", (e) => {
    const ev = JSON.parse(e.data);
    $("feed").appendChild(renderEvent(ev));
    $("feed").lastElementChild.scrollIntoView({ block: "nearest" });
    updateCountersFromEvent(ev);
  });
  source.addEventListener("llm", (e) => {
    const feed = $("feed");
    handleLlmEvent(JSON.parse(e.data), feed);
    feed.lastElementChild.scrollIntoView({ block: "nearest" });
  });
  source.addEventListener("state", (e) => {
    const st = JSON.parse(e.data);
    if (st.started_at) live.startedAt = st.started_at;
    setRunningUI(!!st.running);
    if (!st.running && st.returncode != null) {
      $("c-state").textContent = "exited (" + st.returncode + ")";
    }
  });
  source.addEventListener("end", () => {
    source.close();
    live.source = null;
    setRunningUI(false);
    finalizeLlm($("feed"), null);  // a run can end mid-call
    loadedTabs.delete("history");  // a new run may have appeared
  });
  source.onerror = () => {
    // The server closes the connection after "end"; anything else is a drop.
    if (live.source && source.readyState === EventSource.CLOSED) {
      live.source = null;
      setRunningUI(false);
    }
  };
}

/* What the form's options are currently set to -- shared by a fresh run and a
   resume, since a resume starts a new sitting under the same options. */
function runOptions() {
  const body = {
    repeat: $("opt-repeat").value.trim() || "1",
    dry_run: $("opt-dry-run").checked,
    allow_destructive: $("opt-destructive").checked,
    no_learn: $("opt-no-learn").checked,
    serial: $("opt-serial").value.trim(),
  };
  const ms = parseInt($("opt-max-steps").value, 10);
  if (ms) body.max_steps = ms;
  const budget = parseFloat($("opt-budget").value);
  if (!isNaN(budget)) body.budget_usd = budget;
  return body;
}

function beginLive() {
  live.step = 0; live.calls = 0; live.cost = 0; live.startedAt = Date.now() / 1000;
  $("feed").innerHTML = "";
  $("feed")._llm = null;
  $("c-runid").textContent = "starting…";
  paintCounters();
  setRunningUI(true);
  $("run-hint").textContent = "";
  openStream();
}

$("run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { goal: $("goal").value.trim(), ...runOptions() };
  if (!body.goal) { notice("a goal is required"); return; }

  try {
    await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    notice(err.message);
    return;
  }
  beginLive();
});

/* Continue a failed run from its checkpoint, watched in the run tab. */
async function resumeRun(id) {
  try {
    await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ resume: id, ...runOptions() }),
    });
  } catch (err) {
    notice(err.message);
    return;
  }
  document.querySelector('button[data-tab="run"]').click();
  beginLive();
  $("run-hint").textContent = `resuming ${id} from its checkpoint`;
}

$("btn-stop").addEventListener("click", async () => {
  try {
    await api("/api/runs/stop", { method: "POST" });
    $("run-hint").textContent = "stopping — the agent restores the phone first";
  } catch (err) {
    notice(err.message);
  }
});

/* -------------------------------------------------------------- status */

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    const parts = [];
    parts.push(st.device_serial ? `device ${esc(st.device_serial)}` : `<span class="warn">no device serial</span>`);
    parts.push(st.model ? esc(st.model.split("/").pop()) : `<span class="warn">no model</span>`);
    parts.push(st.api_key_present ? `<span class="ok">api key</span>` : `<span class="warn">no api key</span>`);
    if (st.run && st.run.running) parts.push(`<span class="ok">● running: ${esc(st.run.goal)}</span>`);
    $("status").innerHTML = parts.join(" · ");
    // Reattach to a run already in progress (e.g. page reloaded mid-run).
    if (st.run && st.run.running && !live.source) {
      live.step = 0; live.calls = 0; live.cost = 0;
      live.startedAt = st.run.started_at || Date.now() / 1000;
      setRunningUI(true);
      openStream();
    }
  } catch {
    $("status").textContent = "server unreachable";
  }
}

/* ------------------------------------------------------------ history */

async function loadRuns() {
  const data = await api("/api/runs");
  const tbody = document.querySelector("#runs-table tbody");
  tbody.innerHTML = "";
  $("runs-empty").style.display = data.runs.length ? "none" : "block";
  for (const r of data.runs) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="mono">${esc(r.id)}</td>` +
      `<td class="goal-cell" title="${esc(r.goal)}">${esc(r.goal)}</td>` +
      `<td><span class="chip ${esc(r.outcome)}">${esc(r.outcome)}</span></td>` +
      `<td>${esc(r.steps)}</td><td>$${(r.usd || 0).toFixed(3)}</td>` +
      `<td>${fmtDur(r.duration_s)}</td>` +
      `<td class="small">${fmtTime(r.started)}</td>` +
      `<td></td>`;
    if (r.resumable && !(data.active && data.active.running)) {
      const btn = document.createElement("button");
      btn.className = "ghost";
      btn.textContent = "resume";
      btn.title = `continue from step ${r.steps}`;
      btn.addEventListener("click", (e) => {
        e.stopPropagation();          // the row itself opens the detail view
        resumeRun(r.id);
      });
      tr.lastElementChild.appendChild(btn);
    }
    tr.addEventListener("click", () => openRunDetail(r.id));
    tbody.appendChild(tr);
  }
  $("runs-list-view").style.display = "block";
  $("run-detail-view").style.display = "none";
}

async function openRunDetail(id) {
  $("runs-list-view").style.display = "none";
  $("run-detail-view").style.display = "block";
  $("detail-title").textContent = id;
  const feed = $("detail-feed");
  feed.innerHTML = "";
  feed._llm = null;
  try {
    const d = await api("/api/runs/" + encodeURIComponent(id));
    const s = d.summary, st = d.stats;
    const resumeBtn = $("btn-resume-run");
    resumeBtn.style.display = s.resumable ? "" : "none";
    resumeBtn.onclick = () => resumeRun(id);
    $("detail-stats").innerHTML =
      `<h3>${esc(s.goal)}</h3><div class="counters">` +
      `<span>outcome <b>${esc(s.outcome)}</b></span>` +
      `<span>steps <b>${esc(s.steps)}</b></span>` +
      `<span>cost <b>$${(s.usd || 0).toFixed(4)}</b></span>` +
      `<span>decisions <b>${st.decisions}</b></span>` +
      `<span>latency/step <b>${st.latency_median_s}s med · ${st.latency_p90_s}s p90</b></span>` +
      `<span>prompt tok <b>${st.prompt_tokens.toLocaleString()}</b></span>` +
      `<span>completion tok <b>${st.completion_tokens.toLocaleString()}</b></span>` +
      (st.sweep_reads ? `<span>sweep reads <b>${st.sweep_reads}</b></span>` : "") +
      `</div>`;
    if (d.scratchpad) {
      $("detail-scratch").style.display = "block";
      $("detail-scratch-text").textContent = d.scratchpad;
    } else {
      $("detail-scratch").style.display = "none";
    }
    for (const ev of d.events) {
      // Stream records render as the same live panels, long since collapsed.
      if (typeof ev.kind === "string" && ev.kind.startsWith("llm_")) {
        handleLlmEvent(ev, feed);
      } else {
        feed.appendChild(renderEvent(ev));
      }
    }
    finalizeLlm(feed, null);  // a run that died mid-call has no llm_end
  } catch (err) {
    notice(err.message);
    return;
  }
  api("/api/runs/" + encodeURIComponent(id) + "/log")
    .then((d) => { $("detail-log").textContent = d.text || "(empty)"; })
    .catch(() => { $("detail-log").textContent = "(unavailable)"; });
}

$("btn-back-runs").addEventListener("click", () => loadRuns().catch((e) => notice(e.message)));

/* ------------------------------------------------------------ devices */

async function loadDevices() {
  const d = await api("/api/devices");
  $("dev-error").textContent = d.error ? `adb: ${d.error}` : "";
  const tbody = document.querySelector("#dev-table tbody");
  tbody.innerHTML = "";
  $("dev-empty").style.display = d.devices.length ? "none" : "block";
  $("dev-candidates").textContent =
    d.candidates.length ? `wireless debugging advertised at: ${d.candidates.join(", ")}` : "";
  for (const dev of d.devices) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="mono">${esc(dev.serial)}</td><td>${esc(dev.model)}</td>` +
      `<td>${esc(dev.android)}</td><td></td>`;
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = "screenshot";
    btn.addEventListener("click", () => {
      $("shot-panel").style.display = "block";
      $("shot-serial").textContent = dev.serial;
      $("shot-img").src = "/api/devices/screenshot?serial=" +
        encodeURIComponent(dev.serial) + "&t=" + Date.now();
    });
    tr.lastElementChild.appendChild(btn);
    tbody.appendChild(tr);
  }
}

$("btn-dev-reload").addEventListener("click", () => loadDevices().catch((e) => notice(e.message)));

/* ------------------------------------------------------------- config */

const CFG_SPEC = [
  ["llm", [
    ["provider", "text"], ["model", "text"], ["model_small", "text"],
    ["model_image", "text"], ["model_skill", "text"],
    ["temperature", "number"], ["max_tokens", "number"], ["max_tokens_image", "number"],
    ["rpm", "number"], ["base_url", "text"], ["service_tier", "text"],
    ["api_key", "password"], ["api_key_env", "text"],
    ["reasoning_effort", ["", "none", "low", "medium", "high"]],
    ["reasoning_effort_hard", ["", "none", "low", "medium", "high"]],
    ["reasoning_style", ["auto", "effort", "thinking", "off"]],
    ["vision_in_decider", "bool"],
  ]],
  ["device", [
    ["serial", "text"], ["settle_budget_s", "number"],
    ["disable_animations", "bool"], ["disable_auto_rotate", "bool"],
  ]],
  ["safety", [
    ["budget_usd", "number"], ["allow_destructive", "bool"], ["unattended", "bool"],
  ]],
  ["run", [
    ["max_steps", "number"], ["max_wall_clock_s", "number"], ["artifacts_dir", "text"],
    ["pager_sweep", "bool"], ["pager_sweep_max", "number"],
    ["always_screenshot", "bool"], ["never_screenshot", "bool"], ["dry_run", "bool"],
  ]],
  ["skills", [
    ["enabled", "bool"], ["skills_dir", "text"], ["learn_after_run", "bool"],
  ]],
  ["memory", [["db", "text"]]],
];

let cfgValues = {};

async function loadConfig() {
  const d = await api("/api/config");
  cfgValues = d.config;
  $("cfg-path").textContent = d.path || "(no config file — one will be created on save)";
  const form = $("cfg-form");
  form.innerHTML = "";
  for (const [section, fieldsSpec] of CFG_SPEC) {
    const group = document.createElement("div");
    group.className = "cfg-group";
    group.innerHTML = `<h4>${esc(section)}</h4>`;
    const grid = document.createElement("div");
    grid.className = "cfg-grid";
    for (const [key, type] of fieldsSpec) {
      const value = (cfgValues[section] || {})[key];
      const label = document.createElement("label");
      const inputId = `cfg-${section}-${key}`;
      if (type === "bool") {
        label.className = "check";
        label.innerHTML = `<input type="checkbox" id="${inputId}" ${value ? "checked" : ""}> ${esc(key)}`;
      } else if (Array.isArray(type)) {
        label.innerHTML = `${esc(key)}<select id="${inputId}">` +
          type.map((v) => `<option value="${esc(v)}" ${v === value ? "selected" : ""}>` +
            `${v === "" ? "(unset)" : esc(v)}</option>`).join("") + `</select>`;
      } else if (type === "password") {
        const set = value && value !== "";
        label.innerHTML = `${esc(key)}<input type="password" id="${inputId}" ` +
          `placeholder="${set ? "(set, hidden)" : "(unset)"}" autocomplete="off">`;
      } else {
        const step = Number.isInteger(value) ? "1" : "any";
        label.innerHTML = `${esc(key)}<input type="${type}" id="${inputId}" ` +
          `value="${esc(value ?? "")}" ${type === "number" ? `step="${step}"` : ""}>`;
      }
      grid.appendChild(label);
    }
    group.appendChild(grid);
    form.appendChild(group);
  }
}

$("btn-cfg-save").addEventListener("click", async () => {
  const sections = {};
  for (const [section, fieldsSpec] of CFG_SPEC) {
    for (const [key, type] of fieldsSpec) {
      const el = $(`cfg-${section}-${key}`);
      const original = (cfgValues[section] || {})[key];
      let value;
      if (type === "bool") {
        value = el.checked;
      } else if (Array.isArray(type)) {
        value = el.value;
        if (value === "" && original !== "") continue;  // don't clobber with unset
      } else if (type === "number") {
        if (el.value.trim() === "") continue;
        value = Number.isInteger(original) ? parseInt(el.value, 10) : parseFloat(el.value);
        if (Number.isNaN(value)) continue;
      } else if (type === "password") {
        value = el.value;
        if (value === "") continue;  // blank = leave the stored key untouched
      } else {
        value = el.value;
        if (value === "" && original !== "") continue;
      }
      (sections[section] = sections[section] || {})[key] = value;
    }
  }
  try {
    const r = await api("/api/config", { method: "PUT", body: JSON.stringify({ sections }) });
    notice("saved to " + r.path, false);
    refreshStatus();
  } catch (err) {
    notice(err.message);
  }
});

/* ------------------------------------------------------------- skills */

async function loadSkills() {
  const d = await api("/api/skills");
  const list = $("skills-list");
  list.innerHTML = "";
  if (!d.skills.length) {
    list.innerHTML = `<p class="small">No skills yet. Runs learn them automatically, or generate one.</p>`;
  }
  for (const s of d.skills) {
    const item = document.createElement("div");
    item.className = "skill-item";
    item.innerHTML = `<div class="nm">${esc(s.name)}</div>` +
      `<div class="pkg">${esc(s.packages.join(", ") || "—")}</div>` +
      `<div class="small">${s.workflows} workflows · ${s.nuances} nuances</div>`;
    item.addEventListener("click", async () => {
      document.querySelectorAll(".skill-item").forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      try {
        const full = await api("/api/skills/" + encodeURIComponent(s.name));
        $("skill-editor").style.display = "block";
        $("skill-name").textContent = full.name;
        $("skill-json").value = JSON.stringify(full, null, 2);
      } catch (err) { notice(err.message); }
    });
    list.appendChild(item);
  }
}

$("btn-skills-reload").addEventListener("click", () => loadSkills().catch((e) => notice(e.message)));

$("btn-skill-save").addEventListener("click", async () => {
  let body;
  try {
    body = JSON.parse($("skill-json").value);
  } catch (err) {
    notice("invalid JSON: " + err.message);
    return;
  }
  try {
    const r = await api("/api/skills/" + encodeURIComponent($("skill-name").textContent), {
      method: "PUT", body: JSON.stringify(body),
    });
    notice("saved to " + r.path, false);
    loadSkills().catch(() => {});
  } catch (err) {
    notice(err.message);
  }
});

$("btn-gen").addEventListener("click", async () => {
  let jobId;
  try {
    const r = await api("/api/skills/generate", {
      method: "POST",
      body: JSON.stringify({ name: $("gen-name").value, tasks: $("gen-tasks").value }),
    });
    jobId = r.job;
  } catch (err) {
    notice(err.message);
    return;
  }
  $("gen-log").style.display = "block";
  $("gen-log").textContent = "";
  $("gen-status").textContent = "exploring…";
  const poll = setInterval(async () => {
    try {
      const job = await api("/api/jobs/" + jobId);
      $("gen-log").textContent = job.output_tail.join("\n");
      $("gen-log").scrollTop = $("gen-log").scrollHeight;
      if (!job.running) {
        clearInterval(poll);
        $("gen-status").textContent =
          job.returncode === 0 ? "done" : `failed (exit ${job.returncode})`;
        loadSkills().catch(() => {});
      }
    } catch {
      clearInterval(poll);
      $("gen-status").textContent = "lost track of the job";
    }
  }, 2000);
});

/* -------------------------------------------------------------- boot */

refreshStatus();
setInterval(refreshStatus, 5000);
loadRuns().catch(() => {});  // warm the history cache for later
loadedTabs.add("run");
