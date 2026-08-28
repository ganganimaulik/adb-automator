/* adbagent web UI. No framework: fetch + EventSource against the FastAPI backend.

   Three tabs. Work is one surface -- compose a goal, watch it happen, read what
   it answered, and browse what it answered before -- because starting a run and
   reviewing one were never two different jobs. Watch is the other shape of work.
   Setup is the phone, the config and the skills.

   The feed has two densities. `story` is one row per step: what it did, what it
   saw, whether it worked. `trace` restores every line the harness ever wrote,
   which is the view worth having when the agent itself is the thing that is
   wrong. Nothing is deleted for `story` -- it is all in the DOM behind
   `.trace-only`, one class away. */

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
  // A run is minutes and a history is days, and the same function formats
  // both. Left at minutes, 26 hours of accumulated wall clock reported as
  // `1546m56s` -- a number nobody can read at a glance and nobody wants to
  // the second.
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(s / 3600);
  if (h < 24) return `${h}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  return `${Math.floor(h / 24)}d${h % 24}h`;
}

/* `09/08/2026, 19:29:14` does not say which number is the month, and nobody
   reading a list of runs wants a date anyway -- they want how long ago. The
   absolute time stays on the hover. */
function fmtRel(epoch) {
  if (!epoch) return "";
  const s = Date.now() / 1000 - epoch;
  if (s < 0) return "just now";
  const step = (n, unit) => `${n} ${unit}${n === 1 ? "" : "s"} ago`;
  if (s < 45) return "just now";
  if (s < 5400) return step(Math.max(1, Math.round(s / 60)), "minute");
  if (s < 86400) return step(Math.round(s / 3600), "hour");
  if (s < 172800) return "yesterday";
  if (s < 2592000) return step(Math.round(s / 86400), "day");
  if (s < 31536000) return step(Math.max(1, Math.round(s / 2592000)), "month");
  return step(Math.round(s / 31536000), "year");
}

/* A goal on one line. Watch prompts are a paragraph of instructions whose first
   line is the only distinctive part; long single-line goals are left whole and
   wrapped, because truncating six runs of one goal to 55 characters is what made
   them all read the same row. */
function goalTitle(goal) {
  const lines = String(goal || "").split("\n").map((l) => l.trim()).filter(Boolean);
  return lines[0] || "(no goal recorded)";
}

function goalRest(goal) {
  const lines = String(goal || "").split("\n").map((l) => l.trim()).filter(Boolean);
  return lines.length > 1 ? lines.length - 1 : 0;
}

/* Two goals are the same goal when they differ only by a number.
   Grouping on the exact string is what a history looks like to a machine, not
   what one looks like to the person who made it: "send likes on 3 new
   profiles" and "…on 7 new profiles" are one thing tried twice, and filing
   them apart split 165 of these 169 runs five ways, each group reporting its
   own success rate for what was really one practice. So the key folds every
   run of digits to a single mark and leaves every word alone — a goal that
   differs by a count is one goal, a goal that differs by a word is two. */
function goalKey(goal) {
  return goalTitle(goal).toLowerCase().replace(/\s+/g, " ").trim()
    .replace(/\d+(?:[.,]\d+)*/g, "#");
}

/* ------------------------------------------------------- follow the tail

   The boxes that scroll inside themselves -- a streaming llm panel, the child's
   exit log, the generator's log -- are read top to bottom and chase their
   newest line, but only while the reader is at the bottom of one. Scrolling up
   parks the view where it was left; scrolling back down resumes the chase.

   Decided here and now on each new line, never from a scroll listener: scroll
   events land a frame late, and a stream can outrun them. */

const TAIL_SLACK = 32;  // px short of the bottom that still counts as "at it"

function nearBottom(box) {
  return box.scrollHeight - box.scrollTop - box.clientHeight <= TAIL_SLACK;
}

/* Add live content to `box` via `grow`, keeping the newest of it in view --
   unless the reader has scrolled away from the tail, in which case the view
   stays where they put it.

   Whether to chase is decided before `grow` runs: one card can be taller than
   the slack, and once it is in the DOM there is no telling that from a reader
   having scrolled up. Sitting exactly where the last chase landed also counts
   as the tail however far the bottom has moved since -- that was content
   arriving or a screenshot finishing, not the reader. */
function followTail(box, grow) {
  const chase = Math.abs(box.scrollTop - box._autoTop) < 1  // NaN when unset: fine
                || nearBottom(box);
  grow();
  if (!chase) {
    box._autoTop = -1;  // parked; only being at the bottom resumes the chase
    return;
  }
  box.scrollTop = box.scrollHeight;
  box._autoTop = box.scrollTop;  // read back: the write above was clamped
}

/* --------------------------------------------------------- newest first

   A feed is written head-first: every step, banner and card goes in at the top
   of it. The newest of a run is then always in the same place -- just under the
   feed's heading -- instead of at the end of a page that grows all afternoon,
   and nothing has to chase it down. Only the top level is reversed: a step's own
   screenshots, reads and trace fold stay in the order they happened, because
   that is one step's story rather than the run's.

   What is left to do is hold the reader's place. Everything a live run writes
   lands at the head or in the row that is still at the head -- so a reader down
   among older steps would be pushed down the page by every step, every chip and
   every screenshot, and is given back exactly what arrived instead. A reader
   with the head in view is not moved at all: the feed grows away from them
   rather than under them.

   The browser offers to do this itself (scroll anchoring), but not in every
   browser, and it cannot tell those two cases apart: parked at the head it
   holds whatever was newest a moment ago exactly where it is, which puts the
   row that just arrived off the top of the screen. Left on alongside this it
   also pays for the same growth twice, since the anchor it settles on is an
   ancestor of the feed -- measured: 274px of scroll for a 137px answer block.
   So the sheet declines it for the document, and the arithmetic is done here,
   on the frame the content lands in. */

/* Run `grow`, and put back whatever it cost a reader who is below the head.

   Measured off the foot of the feed in document coordinates, which moves by
   exactly what an older row moves by: a step row is inserted with "thinking…"
   in it and goes on growing for as long as the step lasts -- its observation,
   its chips, its screenshots, the llm panel streaming into it -- and the
   readouts and the ledger above the feed push the whole of it down as they
   fill. All of that is one measurement rather than a tally of insertions. */
function holdPlace(feed, grow) {
  // Nothing to hold: a hidden tab, a finished run being replayed into the page
  // in one go, or a nested call -- the outermost one measures the whole event.
  if (!feed._live || !feed.offsetParent || feed._holding) { grow(); return; }
  const page = document.scrollingElement;
  const before = feed.getBoundingClientRect().bottom + page.scrollTop;
  feed._holding = true;
  try { grow(); } finally { feed._holding = false; }
  const box = feed.getBoundingClientRect();
  // A head still in view is a reader watching the newest end: nothing to give
  // back, because nothing they are looking at moved.
  if (box.top >= 0) { feed._owed = 0; return; }
  const was = page.scrollTop;
  const owed = box.bottom + was - before + (feed._owed || 0);
  page.scrollTop = was + owed;
  // The scroller lands on a device pixel, so part of a row goes unpaid every
  // time -- measured on a 2x display: 70.297px of row against 70.5px of scroll,
  // a fifth of a pixel a step. Carried to the next row rather than left to add
  // up, which over a watch's afternoon is the reader's place walking away.
  feed._owed = owed - (page.scrollTop - was);
}

/* ------------------------------------------------------------- density

   One switch for every feed on the page, kept on <body> so the whole
   restructuring is a class lookup rather than a re-render -- and remembered,
   because somebody debugging the agent is debugging it all afternoon. */

const DENSITY_KEY = "adbagent.density";

function setDensity(mode) {
  const value = mode === "trace" ? "trace" : "story";
  document.body.dataset.density = value;
  document.querySelectorAll("[data-density]").forEach((b) =>
    b.classList.toggle("on", b.dataset.density === value));
  saveValue(DENSITY_KEY, value);
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-density]");
  if (btn) setDensity(btn.dataset.density);
});

/* ---------------------------------------------------------------- tabs */

const loadedTabs = new Set();
const tabLoaders = {
  work: loadRuns,
  watch: loadWatch,
  setup: loadSetup,
};

$("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (!btn) return;
  document.querySelectorAll("nav#tabs button").forEach((b) =>
    b.classList.toggle("active", b === btn));
  document.querySelectorAll("section.tab").forEach((s) =>
    s.classList.toggle("active", s.id === "tab-" + btn.dataset.tab));
  const name = btn.dataset.tab;
  if (!loadedTabs.has(name)) {
    loadedTabs.add(name);
    tabLoaders[name]().catch((err) => notice(err.message));
  }
});

/* Setup's own three panes. Not tabs -- they are one tab's contents -- and each
   costs a fetch, so each is loaded the first time it is looked at. */

const setupLoaders = { device: loadDevices, config: loadConfig, skills: loadSkills };
const loadedPanes = new Set();

function showPane(name) {
  document.querySelectorAll("#setup-nav button").forEach((b) =>
    b.classList.toggle("active", b.dataset.pane === name));
  document.querySelectorAll("#tab-setup .pane").forEach((p) =>
    p.classList.toggle("active", p.id === "pane-" + name));
  if (loadedPanes.has(name)) return;
  loadedPanes.add(name);
  setupLoaders[name]().catch((err) => notice(err.message));
}

$("setup-nav").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-pane]");
  if (btn) showPane(btn.dataset.pane);
});

async function loadSetup() {
  const active = document.querySelector("#setup-nav button.active");
  showPane(active ? active.dataset.pane : "device");
}

/* ------------------------------------------------------- event rendering */

function trunc(s, n) {
  s = String(s ?? "");
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

/* Which control the step is about to touch.

   `tap #3` is the action as the model wrote it, and `#3` is a position in an
   element list that is not on this page -- so the row said what the harness did
   and left *what it did it to* to be guessed from the screenshot underneath.
   `el` is what the harness resolved that ordinal to on the screen the decision
   was made from (the `target_element` of the `decide` event):

     undefined  a run recorded before that was written down -- render as before
     null       nothing on the screen matched, which for a tap is the reason the
                step is about to fail, and is worth saying rather than hiding
     object     the element, led by whatever a human would call it */
function targetName(t, el) {
  const ord = t.index != null ? `#${t.index}`
    : t.resource_id ? `#${t.resource_id}`
    : t.text ? `"${t.text}"` : "";
  if (el === undefined) return t.text ? `"${t.text}"` : ord;
  if (el === null) return `${ord} (not on screen)`;
  const label = (el.text || "").trim();
  const name = label ? `"${trunc(label, 40)}"`
    : el.resource_id ? `id=${el.resource_id}`
    : el.kind || "";
  return [ord, name].filter(Boolean).join(" ");
}

/* The rest of the resolved element, for the row's tooltip: the parts that
   identify it rather than name it. A toggle's state leads, because "tap #3
   Notifications" reads the same whether it is about to turn them on or off. */
function targetTitle(el) {
  if (!el || typeof el !== "object") return "";
  const bits = [];
  if (el.checkable) bits.push(el.checked ? "checked" : "unchecked");
  if (el.selected) bits.push("selected");
  if (el.enabled === false) bits.push("disabled");
  if (el.kind) bits.push(el.kind);
  if (el.text) bits.push(`"${el.text}"`);
  if (el.resource_id) bits.push(`id=${el.resource_id}`);
  if (el.center) bits.push(`at (${el.center[0]},${el.center[1]})`);
  return bits.join(" · ");
}

function actionSummary(a, el) {
  if (!a || typeof a !== "object") return "";
  const kind = a.action || "?";
  const bits = [];
  const t = a.target;
  if (t && typeof t === "object") bits.push(targetName(t, el));
  /* A package name, a link and a Settings action are identifiers, not prose:
     quoting them the way a typed string is quoted reads as if the agent typed
     them somewhere. */
  const BARE = new Set(["open_app", "restart_app", "open_url", "list_apps"]);
  if (a.text) bits.push(BARE.has(kind) ? a.text : `"${a.text}"`);
  if (a.direction) bits.push(a.direction);
  if (a.key) bits.push(a.key);
  return `${kind}${bits.length ? " " + bits.filter(Boolean).join(" ") : ""}`;
}

/* `success` was missing here, so every `verify` the harness graded `success`
   drew a red chip reading "success" -- the default branch is `failed`. */
const GRADE_CLASSES = { worked: "worked", success: "worked", no_change: "no_change" };

/* Kinds that are a line of context rather than a card, each mapped to its text.
   A kind in neither this table nor a branch below renders as its bare name --
   which is how a sweep used to read as twenty lines saying "sweep_step" with
   everything the events carried thrown away.

   All of it is telemetry: the harness talking to itself about what it refused,
   remembered or retried. Real value when the agent is what is being debugged,
   noise when reading what a run did, so it lives inside the step's own fold. */
const NOTE_LINES = {
  sweep_step: (e) => `repeated \`${e.gesture || "the gesture"}\` · ` +
    `${e.read_count || 0} read` + (e.moved === false ? " · did not move" : ""),
  pager_retry: (e) => `retry harder: ${e.action || "the same gesture"}`,
  dismiss: (e) => `harness dismissed ${e.label} (attempt ${e.attempt})`,
  dismiss_failed: (e) =>
    `${e.label} dismisses nothing after ${e.tries} tries — handed to the model`,
  loop_break: (e) => `stuck on ${e.exact_id} — going back`,
  back_loop_escape: (e) =>
    `${e.consecutive_backs} backs in a row — asking for another approach`,
  scroll_rejected: (e) => `scroll rejected: ${e.action}`,
  refused: (e) => `refused ${e.label} — needs "allow destructive"`,
  stall_block: (e) =>
    `refused ${e.action} — tried ${e.tries}× here, nothing new for ${e.stalled} steps`,
  replan: (e) => `stalled ${e.stalled} steps — new approach: ${e.strategy || "?"}`,
  replan_failed: (e) => `replan produced nothing usable (${e.error || "?"})`,
  scratchpad: (e) => `collected: ${(e.keys || []).join(", ")} (${e.total} total)`,
  // `completed` is called out on its own because it is the only part of a plan
  // change that moved the stall ladder -- see `plan.py`.
  plan: (e) => `plan: ${e.done}/${e.total} done` +
    ((e.completed || []).length ? ` — finished ${e.completed.join(", ")}` : "") +
    (e.refused ? ` (${e.refused} step(s) refused, plan at its cap)` : ""),
  dead_ends: (e) => `dead ends avoided: ${(e.remembered || []).join(", ")}`,
  // Was reaching the feed as the bare word `goal_check`, with the model's own
  // account of what was still missing thrown away.
  goal_check: (e) => e.satisfied
    ? "goal check: already satisfied"
    : `goal check: not yet — ${e.evidence || "no reason given"}`,
};

/* How a run stops badly. Rendered rather than dropped: an `error` event used to
   reach the feed as the bare word "error", message and all discarded. */
const HALT_BANNERS = {
  error: (e) => [`failed`, `<b>error</b> ${esc(e.error || "")}`],
  gave_up: (e) => [`failed`, `<b>gave up</b> ${esc(e.reason || "")}`],
  stalled_out: (e) => [`failed`,
                       `<b>stopped going anywhere</b> nothing new for ` +
                       `${esc(String(e.stalled || "?"))} steps — last progress: ` +
                       `${esc(e.last_progress || "?")}`],
  sensitive: (e) => [`needs_user`,
                     `<b>stopped on a sensitive screen</b> ${esc(e.reason || "")}`],
};

/* --------------------------------------------------- collected data

   What the run has *got*: the ledger `scratchpad.py` maintains. The harness
   sends only the records that were new or corrected on a step -- keyed by `id`,
   the normalised key it matched them on -- so the union is kept here per feed
   and re-rendered whole. A person watching a collection run wants the ledger,
   not the delta: the delta is one line and the question is always "has it got
   everything yet".

   The union is pinned above the feed, because that question comes up twenty
   steps after the step that last answered it. The per-step panels stay in the
   trace, where the delta is the point. */

function noteRow(rec, fresh) {
  const row = document.createElement("div");
  row.className = "rec" + (fresh ? " fresh" : "");
  const was = rec.superseded || [];
  row.innerHTML =
    `<span class="k">${esc(rec.key)}</span>` +
    (rec.value ? `<span class="v">${esc(rec.value)}</span>` : "") +
    // A re-read that disagreed with the first reading. Kept by the harness
    // rather than overwritten, so showing it is the whole point of keeping it.
    (was.length ? `<span class="was">earlier: ${esc(was.join(" · "))}</span>` : "");
  return row;
}

/* The pinned ledger: the whole union, with whatever the newest step touched
   marked. `box` is the surface's own panel. */
function paintLedger(box, ledger, fresh, dropped) {
  if (!box) return;
  if (!ledger || !ledger.size) { box.hidden = true; return; }
  box.hidden = false;
  const n = ledger.size;
  box.innerHTML = `<div class="lk">collected data · ${n} ` +
    `${n === 1 ? "record" : "records"}</div><div class="notes-body"></div>`;
  const body = box.querySelector(".notes-body");
  for (const [id, rec] of ledger) body.appendChild(noteRow(rec, fresh.has(id)));
  // The harness evicts its oldest records past a ceiling and reports the count
  // rather than hiding it; the view has no such ceiling, so when the two
  // disagree it is the harness that stopped carrying them -- and the prompt the
  // model sees from here on is the shorter one.
  if (dropped > 0) {
    const note = document.createElement("div");
    note.className = "notes-note";
    note.textContent = `the harness has dropped the oldest ${dropped} for ` +
      `space — still listed here, but no longer in the model's prompt`;
    box.appendChild(note);
  }
}

function finalizeNotes(feed) {
  const card = feed && feed._notesCard;
  if (card) {
    card.open = false;
    feed._notesCard = null;
  }
}

/* The plan, in the two shapes it arrives in.

   `progress` on a `decide` action is a *delta* -- the steps that changed this
   turn -- and a checklist is the wrong picture of it: `[ ] send the summary`
   beside two steps means "these two changed", not "one of two is left". So a
   delta renders as what changed, and only the `plan` event, which carries the
   whole ledger, renders as a checklist. */
const PLAN_MARK = {done: "[x]", active: "[>]", blocked: "[!]", pending: "[ ]"};

function planDelta(steps) {
  return (steps || [])
    .map((s) => (s.status ? `${s.text || s.id}: ${s.status}` : (s.text || s.id)))
    .filter(Boolean).join("; ");
}

function planText(steps) {
  return (steps || [])
    .map((s) => `${PLAN_MARK[s.status] || PLAN_MARK.pending} ${s.text || s.id}`)
    .join("\n");
}

/* Fold a `scratchpad` event into the feed's union, and return the per-step
   delta panel for the trace.

   Only the newest panel stays open. Fifteen steps of an album walk would
   otherwise be fifteen copies of a growing list, and the one worth reading is
   always the last -- so each new panel folds the one before it, the way the
   thinking panels fold when their call ends. A panel a reader opened by hand is
   left alone: only the panel this feed opened is tracked. */
function notesPanel(ev, feed) {
  const ledger = feed._notes || (feed._notes = new Map());
  const fresh = new Set();
  for (const rec of ev.records || []) {
    const id = rec.id || rec.key;
    // Set, not delete-then-set: a corrected record keeps the position it was
    // first written in, which is the order the harness holds them in too.
    ledger.set(id, rec);
    fresh.add(id);
  }
  finalizeNotes(feed);  // this panel is the current ledger now; the last is not
  paintLedger(feed._ledgerBox, ledger, fresh, ledger.size - (ev.total || ledger.size));

  const card = document.createElement("div");
  card.className = "card notes";
  const n = fresh.size;
  card.innerHTML =
    `<details open><summary>collected data · step ${esc(ev.step)} · ` +
    `+${n} ${n === 1 ? "record" : "records"} · ${ledger.size} total</summary>` +
    `<div class="notes-body"></div></details>`;
  const body = card.querySelector(".notes-body");
  for (const [id, rec] of ledger) body.appendChild(noteRow(rec, fresh.has(id)));
  feed._notesCard = card.querySelector("details");
  return card;
}

/* --------------------------------------------------------- step rows */

/* The one way anything enters a feed: at the head, holding the reader's place.
   Every top-level row goes through here so that "newest first" is a property of
   the feed rather than a habit of nine call sites. */
function feedAdd(feed, el) {
  holdPlace(feed, () => feed.prepend(el));
}

/* The step's row, created on first sight of it -- which is the model starting to
   think about it, not the decision landing, so the row appears as soon as there
   is something to say about it. */
function feedStep(feed, step) {
  const steps = feed._steps || (feed._steps = new Map());
  const key = step == null ? "" : String(step);
  if (!key) return null;
  const found = steps.get(key);
  if (found) return found;
  const el = document.createElement("div");
  el.className = "step";
  el.innerHTML =
    `<div class="step-row">` +
      `<span class="step-no">${esc(key)}</span>` +
      `<div class="step-main">` +
        `<div class="step-act" data-placeholder="1">thinking…</div>` +
        `<div class="step-obs" hidden></div>` +
        `<div class="step-verdict" hidden></div>` +
      `</div>` +
      `<div class="step-chips"></div>` +
    `</div>` +
    `<div class="step-shots"></div>` +
    `<div class="step-reads"></div>` +
    `<div class="step-more trace-only"></div>`;
  const node = {
    el,
    act: el.querySelector(".step-act"),
    obs: el.querySelector(".step-obs"),
    verdict: el.querySelector(".step-verdict"),
    chips: el.querySelector(".step-chips"),
    shots: el.querySelector(".step-shots"),
    reads: el.querySelector(".step-reads"),
    more: el.querySelector(".step-more"),
  };
  steps.set(key, node);
  feedAdd(feed, el);
  return node;
}

/* Where a line about step N goes. Falls back to the last step seen, so events
   the harness records without one (`dead_ends` is filed under the step it is
   about to protect) still land under a step rather than loose in the feed. */
function stepFor(feed, ev) {
  const step = ev.step != null ? ev.step : feed._curStep;
  if (step != null) feed._curStep = step;
  return feedStep(feed, step);
}

function setAct(node, text, title) {
  node.act.textContent = text;
  if (title) node.act.title = title;
  delete node.act.dataset.placeholder;
}

/* Name a step whose decision has not landed. A sweep step never gets one -- the
   harness repeats the gesture in code -- and a row reading "thinking…" for a
   step that is over and did something is the placeholder outliving its purpose. */
function nameStep(node, text) {
  if (!node || node.act.dataset.placeholder !== "1") return;
  setAct(node, text);
}

/* Every step still holding the live placeholder, once there is nothing more
   coming: an interrupted run leaves its last step mid-thought, and the honest
   reading of that row is that no decision was recorded, not that something is
   still thinking about it. */
function finalizeSteps(feed) {
  for (const node of (feed._steps || new Map()).values()) {
    if (node.act.dataset.placeholder !== "1") continue;
    node.act.textContent = "no decision recorded";
    node.act.classList.add("unresolved");
    delete node.act.dataset.placeholder;
  }
}

function chip(cls, text, title) {
  return `<span class="chip ${esc(cls)}"${title ? ` title="${esc(title)}"` : ""}>` +
    `${esc(text)}</span>`;
}

function setLine(el, text) {
  el.textContent = text || "";
  el.hidden = !text;
}

/* The per-step arithmetic. Tokens as well as calls and spend, because what a
   step cost is mostly what it was *shown* -- and the cached share is the
   difference between a prompt that reused the run's context and one that paid
   for the whole of it again, which is otherwise invisible until the bill. */
function stepMeta(ev) {
  const llm = ev.llm || {};
  const a = ev.action || {};
  const bits = [];
  if (ev.wall_s) bits.push(ev.wall_s.toFixed(1) + "s");
  bits.push(`${llm.n_calls || 0} call(s)`);
  if (llm.prompt_tokens) {
    let tok = `${fmtTok(llm.prompt_tokens)}→${fmtTok(llm.completion_tokens)} tok`;
    if (llm.cached_tokens) {
      tok += ` · ${Math.round(100 * llm.cached_tokens / llm.prompt_tokens)}% cached`;
    }
    bits.push(tok);
  }
  bits.push(`$${(llm.usd || 0).toFixed(4)}`);
  if (a.confidence) bits.push(a.confidence);
  return bits.join(" · ");
}

function banner(feed, cls, html, traceOnly) {
  const div = document.createElement("div");
  div.className = "banner " + cls + (traceOnly ? " trace-only" : "");
  div.innerHTML = html;
  feedAdd(feed, div);
}

function tele(feed, ev, text) {
  const node = stepFor(feed, ev);
  const div = document.createElement("div");
  div.className = "tele";
  div.textContent = text;
  if (node) node.more.appendChild(div);
  else { div.classList.add("trace-only"); feedAdd(feed, div); }
}

/* One event, folded into the feed wherever it belongs: onto its step's row, into
   that step's trace fold, or -- for the handful that are about the run rather
   than a step -- as a rule between steps. */
function foldEvent(ev, feed) {
  const kind = ev.kind || "";

  if (kind === "decide") {
    const a = ev.action || {};
    // Whether this run stopped by *asking* something, which is the one halt an
    // answer can restart. `needs_user` is also what a sensitive screen produces
    // -- a password field the agent will not type into -- and there the answer
    // is to do it on the phone, never to type a credential into this page. Both
    // end the same way in `run_end`, so the difference has to be caught here,
    // on the action that made it.
    if (a.action === "ask_user") feed._asked = a.text || "";
    const node = stepFor(feed, ev);
    if (!node) return;
    setAct(node, actionSummary(a, ev.target_element) || "—",
           targetTitle(ev.target_element));
    setLine(node.obs, a.observation);
    node.chips.innerHTML =
      // Why the turn was given deep reasoning, on the chip that says it was:
      // `hard_because` is recorded per step and had nowhere to be read. Named
      // rather than bare -- a chip reading only `none` beside one reading
      // `success` is a value with its question missing.
      (ev.effort ? chip("neutral", `effort ${ev.effort}`, ev.hard_because || "") : "") +
      (ev.screenshot ? chip("neutral", "vision") : "") +
      // The stall ladder's own counter. Every action can be succeeding while the
      // run goes nowhere, and this is the number the harness escalates on.
      (ev.stalled ? chip("stall", `stalled ${ev.stalled}`,
        "steps since the run last learned anything — the harness starts " +
        "intervening as this climbs") : "") +
      node.chips.innerHTML;
    if (a.reasoning) {
      const why = document.createElement("div");
      why.className = "why";
      why.textContent = a.reasoning;
      node.more.appendChild(why);
    }
    // What this step changed about the plan. The plan itself is pinned above
    // the feed, off the `plan` event -- the card carrying a change is twenty
    // steps back by the time anyone asks what the run is up to.
    const delta = planDelta(a.progress);
    if (delta) {
      const plan = document.createElement("div");
      plan.className = "plan";
      plan.textContent = delta;
      node.more.appendChild(plan);
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = stepMeta(ev);
    node.more.appendChild(meta);

  } else if (kind === "verify") {
    // Folded into the row it graded rather than trailing it as a card of its
    // own: a 47px card reading `step 4 success` was a third of this feed.
    const node = stepFor(feed, ev);
    if (!node) return;
    node.chips.insertAdjacentHTML("afterbegin",
      chip(GRADE_CLASSES[ev.grade] || "failed", ev.grade || "?"));
    if (ev.reason) setLine(node.verdict, ev.reason);

  } else if (kind === "judge") {
    const node = stepFor(feed, ev);
    if (!node) return;
    node.chips.insertAdjacentHTML("afterbegin", chip(
      ev.satisfied ? "worked" : "failed",
      `judge: ${ev.satisfied ? "satisfied" : "not yet"}`));
    if (ev.evidence) setLine(node.verdict, ev.evidence);

  } else if (kind === "item_reading") {
    // The sweep's own vision call. It gets no live panel -- a prefetched read
    // runs on another thread, and streaming two of those into one view
    // interleaves them -- so this row is the whole of it: the line the model
    // read, and the frame it read it from. Kept out of the fold, because on a
    // carousel walk these *are* what the run collected.
    const node = stepFor(feed, ev);
    const row = document.createElement("div");
    row.className = "card";
    row.innerHTML =
      `<div class="head">${chip("neutral", "read")}` +
      (ev.position ? `<span class="mono">#${esc(ev.position)}</span>` : "") + `</div>` +
      `<div class="obs">${esc(ev.reading || "")}</div>`;
    const thumb = llmShot(ev, feed._runId);
    if (thumb) row.appendChild(thumb);
    if (node) node.reads.appendChild(row);
    else feedAdd(feed, row);

  } else if (kind === "image_analysis") {
    const node = stepFor(feed, ev);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML =
      `<details><summary>vision read · ${esc(ev.model || "")}</summary>` +
      `<div class="why" style="margin-top:6px">${esc(ev.result || "")}</div></details>`;
    if (node) node.more.appendChild(card);
    else { card.classList.add("trace-only"); feedAdd(feed, card); }

  } else if (kind === "scratchpad" && (ev.records || []).length) {
    // A run recorded before the event carried its records has only the keys, and
    // falls through to the one-line form in NOTE_LINES below.
    const node = stepFor(feed, ev);
    const panel = notesPanel(ev, feed);
    if (node) node.more.appendChild(panel);
    else { panel.classList.add("trace-only"); feedAdd(feed, panel); }

  } else if (kind === "active_skill") {
    // Recorded on every step -- it is the per-step record of what the prompt
    // actually carried -- so rendering each one buried the fact under eight
    // identical grey lines. A line per *change* is the information: it is where
    // the run crossed into another app's guidance, or failed to.
    const label = `${ev.name || "?"}|${ev.package || ""}`;
    if (feed._skill === label) return;
    feed._skill = label;
    banner(feed, "skill", chip("skill", "skill loaded") +
      ` <b>${esc(ev.name || "?")}</b>` +
      (ev.package ? ` <span class="small">${esc(ev.package)}</span>` : ""));

  } else if (kind === "sweep") {
    banner(feed, "", `<b>repeated \`${esc(ev.gesture)}\` ${esc(ev.swept)}×</b>, ` +
      `${esc(ev.read)} read <span class="small">· steps ${esc(ev.first_step)}–` +
      `${esc(ev.last_step)}${ev.reason ? " · " + esc(ev.reason) : ""}</span>`);

  } else if (kind === "run_start") {
    // The goal is the headline of the surface now, so this is a rule in the
    // trace rather than a third copy of it.
    banner(feed, "", `<b>goal</b> ${esc(ev.goal)}<br>` +
      `<span class="small">${esc(ev.model || "")}</span>`, true);

  } else if (kind === "run_resume") {
    // Where two sittings of one run join: the events above it are the failed
    // attempt, the ones below are the continuation.
    banner(feed, "", `<b>resumed from step ${esc(ev.resumed_at_step || 0)}</b>` +
      `<br><span class="small">${esc(ev.model || "")}</span>`);

  } else if (kind === "run_end") {
    // The answer has its own block above the feed -- large, first, and once. In
    // the trace it stays where it always was, because the trace is the record.
    banner(feed, esc(ev.outcome || ""),
      `<b>${esc((ev.outcome || "").toUpperCase())}</b> — ${esc(ev.steps)} steps, ` +
      `${esc(ev.llm_calls)} LLM calls, $${(ev.usd || 0).toFixed(4)}` +
      (ev.result ? `<div class="why">${esc(ev.result)}</div>` : "") +
      (ev.evidence ? `<div class="why">${esc(ev.evidence)}</div>` : ""), true);

  } else if (HALT_BANNERS[kind]) {
    const [cls, html] = HALT_BANNERS[kind](ev);
    banner(feed, cls, html);

  } else if (NOTE_LINES[kind]) {
    // A swept step has no decision to name it: the harness is repeating the
    // gesture the model already chose, in code, without asking again.
    if (kind === "sweep_step") {
      nameStep(stepFor(feed, ev), `repeated \`${ev.gesture || "the gesture"}\``);
    }
    tele(feed, ev, NOTE_LINES[kind](ev));

  } else if (kind) {
    tele(feed, ev, kind);
  }
}

function updateCountersFromEvent(ev, v) {
  if (typeof ev.step === "number") {
    v.step = Math.max(v.step, ev.step);
  }
  // Which skill is in the prompt *now*, not just where it last changed: the
  // feed scrolls away and the answer to "is this run being helped by the right
  // app's skill" should not need scrolling back for.
  if (ev.kind === "active_skill" && ev.name) {
    v.skill = ev.name;
  }
  // The ceilings this sitting runs under, so a step count reads as a position
  // and the spend reads against what it is allowed to reach.
  if (ev.kind === "run_start" || ev.kind === "run_resume") {
    v.maxSteps = ev.max_steps || 0;
    v.budget = ev.budget_usd || 0;
  }
  // How much the run has collected, and where it thinks it is. Both scroll away
  // in the feed, and both are the answer to "is this going anywhere".
  if (ev.kind === "scratchpad" && typeof ev.total === "number") {
    v.records = ev.total;
  }
  // The plan event carries the whole ledger, so the readout is a straight
  // assignment rather than a union the client has to maintain -- the same
  // arrangement as `scratchpad.total` above, and for the same reason: the
  // harness owns how two spellings of one step become one entry.
  if (ev.kind === "plan") {
    v.progress = planText(ev.steps);
  }
  if (ev.kind === "decide") {
    // The first of the three primary readouts: what it is doing, in the words
    // the action itself is written in.
    v.now = actionSummary(ev.action, ev.target_element) || v.now;
  }
  const llm = ev.llm;
  if (llm && typeof llm === "object") {
    v.calls += llm.n_calls || 0;
    v.cost += llm.usd || 0;
  }
  paintCounters(v);
}

/* --------------------------------------------------------- llm stream */

/* One collapsible panel per LLM call: it opens when the call starts, the
   model's raw thinking and response stream into it live, and it folds itself
   away the moment the call ends -- the step row carries the result. Panels are
   keyed off order, not step: calls within a step are sequential (a vision read,
   then the decision), so at most one is ever open per feed, tracked as
   feed._llm. They live in the step's trace fold, because the raw stream is the
   deepest layer of the machinery.

   A call that was shown a screenshot carries its file name, and the thumbnail
   goes on the step itself rather than into the fold: a vision read you cannot
   check against the frame it read is only half the record, and the filmstrip of
   what a run was actually shown belongs in the story. Which run's files to ask
   for is feed._runId -- the live feed learns it from the stream, the history
   feed from the run it opened. */

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

/* The frame at the size of the window -- with, when the caller has it, the
   geometry that was drawn over the small copy.

   This is where the element list is actually legible. Eighty numbered boxes
   over a 236px panel is a texture; over the full window it is the list the
   model was choosing `#N` out of, which is the thing a decision that went to
   the wrong control is always argued about. */
function openLightbox(src, alt, geo) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = alt || "";
  const shade = $("lightbox-shade");
  const target = geo && boxCss(geo.target && geo.target.bounds, geo.w, geo.h);
  shade.innerHTML =
    (geo ? elementBoxes(geo.els, geo.w, geo.h) : "") +
    (target ? `<div class="phone-box" style="${target}">` +
              `<span class="phone-tag">${geo.target.index == null ? ""
                : esc("#" + geo.target.index)}</span></div>` : "");
  shade.hidden = !shade.innerHTML;
  // The stage only takes the screen's shape when there is something to draw
  // over it; see `.lightbox-stage.fitted`.
  const stage = document.querySelector(".lightbox-stage");
  const fitted = !shade.hidden && geo.w > 0 && geo.h > 0;
  stage.classList.toggle("fitted", fitted);
  if (fitted) stage.style.setProperty("--shot-ratio", `${geo.w} / ${geo.h}`);
  else stage.style.removeProperty("--shot-ratio");
  $("lightbox").hidden = false;
}

function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightbox-img").removeAttribute("src");  // stop holding the decoded frame
  const shade = $("lightbox-shade");
  shade.innerHTML = "";
  shade.hidden = true;
  document.querySelector(".lightbox-stage").classList.remove("fitted");
}

$("lightbox").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

/* The thumbnail of the frame this call was shown, or nothing when it was shown
   none. */
function llmShot(ev, runId) {
  if (!ev.shot || !runId) return null;
  const wrap = document.createElement("div");
  wrap.className = "llm-shot";
  const img = document.createElement("img");
  img.src = `/api/runs/${encodeURIComponent(runId)}/shot/${encodeURIComponent(ev.shot)}`;
  img.alt = `the screen sent to the model at step ${ev.step || 0}`;
  img.title = "the frame this call was shown — click to enlarge";
  img.loading = "lazy";
  img.addEventListener("click", () => openLightbox(img.src, img.alt));
  wrap.appendChild(img);
  return wrap;
}

/* ------------------------------------------------- chunks, cheaply

   A thinking model streams in far smaller pieces than a browser can lay out. A
   measured 46-call run: 56,968 chunks, one call of it 20,120 chunks carrying
   75,920 characters. Two rules keep that affordable, and breaking either one is
   what made the page stop answering the mouse:

   Text is *appended*, never rewritten. Re-setting `textContent` to the whole of
   what has arrived costs the length of the message on every chunk, so a call
   ends up writing (chars x chunks / 2) characters -- 764 million for that one
   call, 1.3 billion over the run -- to show 76 KB.

   Layout is measured once a frame, not once a chunk. Chasing the tail reads
   `scrollHeight` on the box and on the whole document, which forces the
   pending layout of a page carrying a run's worth of rows and screenshots.
   Chunks arriving between two frames are joined into one append, so the cost
   follows the display's refresh rate rather than the model's token rate. */

function llmFlush(p) {
  if (p.frame) { cancelAnimationFrame(p.frame); p.frame = 0; }
  const ready = p.secs.filter((s) => s.buf);
  if (!ready.length) return;
  p.chase(() => {
    for (const s of ready) {
      if (s.sec.style.display !== "block") s.sec.style.display = "block";
      // Appending a text node rather than growing one keeps the write
      // proportional to the chunk. The browser coalesces them on its own.
      followTail(s.text, () => s.text.appendChild(document.createTextNode(s.buf)));
      s.buf = "";
    }
  });
}

function llmPush(p, streamType, chunk) {
  const s = p.secs[streamType === "thinking" ? 0 : 1];
  s.buf += chunk;
  if (!p.frame) p.frame = requestAnimationFrame(() => { p.frame = 0; llmFlush(p); });
}

function finalizeLlm(feed, end) {
  const p = feed._llm;
  if (!p) return;
  feed._llm = null;
  llmFlush(p);             // whatever the last frame did not get to
  p.details.open = false;  // auto-hide: the stream is over, the verdict follows
  p.summary.textContent = llmSummary(p.start, end);
}

function handleLlmEvent(ev, feed) {
  if (ev.kind === "llm_start") {
    finalizeLlm(feed, null);  // a panel the last stream never closed
    const node = stepFor(feed, ev);
    const card = document.createElement("div");
    card.className = "card llm-live";
    card.innerHTML =
      `<details open><summary><span class="pulse"></span> ${llmSummary(ev)}</summary>` +
      `<div class="llm-sec thinking"><div class="llm-sec-label">thinking</div>` +
      `<div class="llm-text"></div></div>` +
      `<div class="llm-sec response"><div class="llm-sec-label">response</div>` +
      `<div class="llm-text"></div></div></details>`;
    if (node) node.more.appendChild(card);
    else { card.classList.add("trace-only"); feedAdd(feed, card); }
    // The frame outside the fold: it is evidence, not machinery.
    const shot = llmShot(ev, feed._runId);
    if (shot) {
      if (node) node.shots.appendChild(shot);
      else feedAdd(feed, shot);
    }
    const section = (name) => ({
      sec: card.querySelector(`.llm-sec.${name}`),
      text: card.querySelector(`.llm-sec.${name} .llm-text`),
      buf: "",
    });
    feed._llm = {
      start: ev,
      details: card.querySelector("details"),
      summary: card.querySelector("summary"),
      secs: [section("thinking"), section("response")],
      frame: 0,
      // The panel is already in the feed; what streaming into it moves is
      // everything below it, which is a reader's place to hold like any other.
      chase: (grow) => holdPlace(feed, grow),
    };
  } else if (ev.kind === "llm_stream" && feed._llm) {
    llmPush(feed._llm, ev.stream_type, ev.text || "");
  } else if (ev.kind === "llm_end") {
    finalizeLlm(feed, ev);
  }
}

function paintCounters(v) {
  v.els.now.textContent = v.now || "…";
  // Against the ceiling when the run recorded one: "step 12/60" says how much
  // room is left, which is what somebody watching actually wants to know.
  v.els.step.textContent = v.maxSteps ? `${v.step}/${v.maxSteps}` : v.step;
  v.els.calls.textContent = v.calls;
  v.els.cost.textContent = "$" + v.cost.toFixed(4) +
    (v.budget ? " / $" + v.budget.toFixed(2) : "");
  v.els.skill.textContent = v.skill || "—";
  v.els.records.textContent = v.records;
  // The model's own progress note, outside the feed on purpose: it is the one
  // line that answers "what does it think it is doing", and the row carrying it
  // is twenty rows back by the time the question comes up.
  v.els.progress.textContent = v.progress;
  v.els.progress.hidden = !v.progress;
  // Only worth the space once there is more than one: a single run has no
  // iteration to speak of. A tour never repeats, so it has no such counter.
  if (v.els.iterWrap) {
    v.els.iterWrap.hidden = !(v.iteration > 1);
    v.els.iter.textContent = v.iteration;
  }
}

/* The tail of the feed while a child is shutting down.

   One card, grown a line at a time from the child's stdout. It exists because
   the shutdown is the one part of the work that leaves no events behind it: the
   loop has ended, the run directory is closed, and what is still happening --
   the phone's keyboard, animations and rotation being put back, then a model
   call folding every pass into the app's skill -- is reported nowhere else.
   Under a watch that is a minute of a feed that has stopped moving, which is
   indistinguishable from a stop that did not work. */
function appendExitLine(feed, line) {
  if (!feed._exitLog) {
    const card = document.createElement("div");
    card.className = "banner";
    card.innerHTML = "<b>stopping</b><br><span class=\"small\">the phone is "
      + "restored first, then what the passes learned about the app is written "
      + "into its skill — one model call, which can take a minute.</span>";
    const pre = document.createElement("pre");
    pre.className = "log";
    card.appendChild(pre);
    feed._exitLog = pre;
    feedAdd(feed, card);   // the newest thing there is: at the head, like a step
  }
  // The card grows downward inside itself even though the feed grows upward:
  // this is one thing happening, and its lines are in the order it says them.
  // `pre.log` scrolls internally past 400px, so a long write-up would otherwise
  // grow off the bottom of a box that stayed at its first line.
  //
  // Text nodes, not innerHTML: these are the child's own lines, and one of them
  // is a goal or a skill name it read off the phone.
  const pre = feed._exitLog;
  holdPlace(feed, () => followTail(pre, () =>
    pre.appendChild(document.createTextNode(line + "\n"))));
}

/* ------------------------------------------------------ the phone

   It automates a phone and never showed you the phone. This panel does, off
   `/api/device/frame` -- one `exec-out screencap`, which opens no device session
   and so cannot disturb whatever is driving it. Which is the whole reason it is
   a new endpoint and not the Devices screenshot: that one goes through
   `Device()`, which zeroes the animation scales and locks rotation on the way
   in, and the server refuses it while an agent holds the phone.

   Every state says which it is. A frame is stamped with how old it is, a phone
   that cannot be read says why, and nothing is ever shown unlabelled. */

/* ------------------------------------------------- element geometry

   Where the run said its elements were, drawn over a picture of the screen
   they were measured on. Percentages, never pixels: every picture here is a
   scaled copy -- a 236px panel, a full-screen lightbox, and every width the
   column takes in between -- and a box placed as a fraction of its own screen
   needs no recomputing when the scale changes under it. */

//: Whether the whole element list is drawn, or only what the step is aiming at.
//: Remembered like the feed's density: it is a way of reading the panel rather
//: than a property of any one run.
const ELEMENTS_KEY = "adbagent.phone-elements";

function boxCss(b, w, h) {
  const [l, t, r, bot] = (b || []).map(Number);
  if (!(w > 0) || !(h > 0) || !(r > l) || !(bot > t)) return "";
  const pct = (n, d) => `${(100 * n / d).toFixed(3)}%`;
  return `left:${pct(l, w)};top:${pct(t, h)};` +
         `width:${pct(r - l, w)};height:${pct(bot - t, h)}`;
}

/* Every element the model was shown, outlined and numbered.

   `#12` is a position in this list, so the list is the difference between
   reading the number and seeing what it picked out of — which is the question
   a decision that went to the wrong control always raises. */
function elementBoxes(els, w, h) {
  let html = "";
  for (const el of els || []) {
    const css = boxCss(el.b, w, h);
    if (!css) continue;
    html += `<div class="phone-el" style="${css}" ` +
            `title="${esc(`#${el.i} ` + (targetTitle(el) || ""))}">` +
            `<span class="phone-num">${esc(String(el.i))}</span></div>`;
  }
  return html;
}

const FRAME_POLL_MS = 2000;
//: A phone that cannot be read will not start being readable in two seconds, so
//: back right off rather than asking sixteen hundred times an hour. Still a
//: poll and not a stop: plugging a phone in mid-run should heal on its own.
const FRAME_RETRY_MS = 15000;

function phoneView(box, { live = false, edge = 720 } = {}) {
  box.innerHTML =
    `<div class="phone-head"><span class="lbl">the phone</span>` +
    `<span class="grow"></span>` +
    `<button type="button" class="pause" hidden>pause</button>` +
    `<button type="button" class="els" hidden>elements</button>` +
    `<button type="button" class="shot">refresh</button></div>` +
    `<div class="phone-frame"><div class="phone-empty">—</div></div>` +
    `<div class="phone-note"></div>`;
  const frame = box.querySelector(".phone-frame");
  const note = box.querySelector(".phone-note");
  const lbl = box.querySelector(".lbl");
  const pause = box.querySelector(".pause");
  const elsBtn = box.querySelector(".els");
  const v = {
    timer: null, url: "", busy: false, paused: false, at: 0, ticker: null,
    running: false, every: FRAME_POLL_MS,
    //: The last step's geometry, kept so the toggle below can redraw it
    //: without waiting for another step to arrive.
    geo: null,
    showEls: loadValue(ELEMENTS_KEY) === "1",
  };

  /* What the run says is on the screen, drawn over the picture of it.

     Built once and re-attached rather than rendered with the frame: everything
     below replaces the frame's contents wholesale -- a new image, an error
     state -- and this has to survive both. The target sits after the element
     list in the DOM so it paints over its own outline rather than under it. */
  const shade = document.createElement("div");
  shade.className = "phone-shade";
  shade.hidden = true;
  shade.innerHTML = `<div class="phone-els"></div>` +
                    `<div class="phone-box"><span class="phone-tag"></span></div>`;
  const others = shade.querySelector(".phone-els");
  const mark = shade.querySelector(".phone-box");
  const tag = shade.querySelector(".phone-tag");
  frame.appendChild(shade);

  function say(text, warn) {
    note.textContent = text || "";
    note.className = "phone-note" + (warn ? " warn" : "");
  }

  function empty(text) {
    if (v.url) { URL.revokeObjectURL(v.url); v.url = ""; }
    frame.innerHTML = `<div class="phone-empty">${esc(text)}</div>`;
    frame.appendChild(shade);
    unmark();          // nothing to be over
  }

  function unmark() { v.geo = null; paint(); }

  /* Draw one step's geometry: `{w, h, target, els}`, any part of it missing.

     The two halves arrive from two files and nothing orders them, so this is
     called again with whichever turned up -- and draws whatever it has. */
  function paint(geo) {
    if (geo !== undefined) v.geo = geo;
    const g = v.geo;
    if (!g || !frame.querySelector("img")) { shade.hidden = true; return; }
    const css = boxCss(g.target && g.target.bounds, g.w, g.h);
    mark.style.cssText = css;
    mark.hidden = !css;
    if (css) {
      tag.textContent = g.target.index == null ? "" : `#${g.target.index}`;
      tag.hidden = g.target.index == null;
      // The label the trace uses for the same element, so hovering the box and
      // reading the step row are the same identification.
      mark.title = targetTitle(g.target) || "";
    }
    others.innerHTML = v.showEls ? elementBoxes(g.els, g.w, g.h) : "";
    shade.hidden = !css && !others.innerHTML;
  }

  function syncEls() {
    elsBtn.textContent = v.showEls ? "hide elements" : "elements";
    elsBtn.classList.toggle("on", v.showEls);
  }

  function stamp() {
    if (!v.at) return;
    const age = Math.round(Date.now() / 1000 - v.at);
    lbl.textContent = v.running
      ? (age < 4 ? "the phone · live" : `the phone · ${age}s old`)
      : `the phone · ${age < 4 ? "just now" : fmtRel(v.at)}`;
  }

  async function grab() {
    if (v.busy) return;
    v.busy = true;
    const slow = (ms) => { if (v.every !== ms) { v.every = ms; schedule(); } };
    try {
      const res = await fetch(
        `/api/device/frame?max_long_edge=${edge}&t=${Date.now()}`);
      if (!res.ok) {
        let detail = res.statusText;
        try { detail = (await res.json()).detail || detail; } catch { /* text */ }
        empty("no picture");
        say(String(detail), true);
        v.at = 0;
        lbl.textContent = "the phone";
        slow(FRAME_RETRY_MS);
        return;
      }
      slow(FRAME_POLL_MS);
      const blob = await res.blob();
      const next = URL.createObjectURL(blob);
      let img = frame.querySelector("img");
      if (!img) {
        frame.innerHTML = "";
        img = document.createElement("img");
        img.alt = "the phone's screen right now";
        img.title = "click to enlarge";
        // Enlarged with whatever is drawn over it: eighty numbered boxes are
        // unreadable at the width of this column and are the whole point at
        // the width of the window.
        img.addEventListener("click", () => openLightbox(img.src, img.alt, v.geo));
        frame.appendChild(img);
        frame.appendChild(shade);   // kept on top of whatever replaced it
      }
      const old = v.url;
      img.src = next;
      v.url = next;
      if (old) URL.revokeObjectURL(old);
      v.at = Date.now() / 1000;
      stamp();
      say(v.running
        ? "a read-only screencap taken beside the run — it opens no device "
          + "session, so it cannot disturb what the agent set"
        : "");
    } catch (err) {
      say(err.message, true);
      slow(FRAME_RETRY_MS);
    } finally {
      v.busy = false;
    }
  }

  function schedule() {
    clearInterval(v.timer);
    if (v.paused || !live || !v.running) return;
    v.timer = setInterval(() => {
      // Nothing to see on a hidden tab, and screencap is not free on the phone.
      if (!document.hidden && box.offsetParent) grab();
    }, v.every);
  }

  pause.addEventListener("click", () => {
    v.paused = !v.paused;
    pause.textContent = v.paused ? "resume" : "pause";
    schedule();
  });
  elsBtn.addEventListener("click", () => {
    v.showEls = !v.showEls;
    saveValue(ELEMENTS_KEY, v.showEls ? "1" : "0");
    syncEls();
    paint();
  });
  box.querySelector(".shot").addEventListener("click", grab);

  syncEls();
  clearInterval(v.ticker);
  v.ticker = setInterval(stamp, 1000);

  return {
    /* A run is on: poll, and label the frames live. */
    start() {
      v.running = true;
      v.every = FRAME_POLL_MS;   // a new run gets a fresh benefit of the doubt
      pause.hidden = !live;
      // Offered only where there is something to draw: geometry arrives from a
      // run, so on the Setup screenshot the button would never do anything.
      elsBtn.hidden = !live;
      grab();
      schedule();
    },
    /* Over: stop polling, keep the last frame, and stop calling it live. */
    stop() {
      v.running = false;
      pause.hidden = true;
      elsBtn.hidden = true;
      clearInterval(v.timer);
      stamp();
      say("");
      unmark();   // the last frame is kept; what it was about to do is not
    },
    /* One frame, now. */
    refresh: grab,
    /* One step's geometry -- what it is aiming at, and what it chose out of --
       and taking it away again. */
    mark: paint,
    unmark,
    idle(text) { empty(text); say(""); lbl.textContent = "the phone"; v.at = 0; },
  };
}

/* ------------------------------------------------- a live surface

   The readouts, the feed under them, and the SSE connection that fills them.
   There are three: Work's, Watch's, and the one under the skill generator --
   because a generation is a run. It drives the phone through the same agent,
   spends from the same budget and writes the same events, screenshots and
   thinking stream, so it is shown with the same rows rather than as the tail of
   a subprocess's stdout.

   `prefix` names the readout ids (`c-step`, `gc-step`), `feedId` the feed. */

function makeLive(prefix, boxId, feedId) {
  const el = (name) => $(prefix + name);
  const feed = $(feedId);
  feed._live = true;  // chase the tail; the history feed does not
  feed._ledgerBox = el("ledger");
  return {
    step: 0, calls: 0, cost: 0, skill: "", iteration: 1,
    records: 0, progress: "", maxSteps: 0, budget: 0, now: "",
    source: null, startedAt: 0, timer: null,
    // Stopped, but not over: the child still holds the phone while it restores
    // it and writes up what it learned. Its own state, because showing it as
    // "running" is what made a stop look like it had been ignored.
    stopping: false,
    url: "",           // where to stream from; a job's is known only once it starts
    box: $(boxId), feed,
    els: { runid: el("runid"), now: el("now"), step: el("step"),
           calls: el("calls"), cost: el("cost"), elapsed: el("elapsed"),
           state: el("state"), skill: el("skill"), iterWrap: el("iter-wrap"),
           iter: el("iter"), records: el("records"), progress: el("progress"),
           ledger: el("ledger") },
    phone: null,             // the live device panel, when the surface has one
    marked: null,            // the step whose target the panel is drawing
    geo: null,               // what it is drawing: target, element list, scale
    screen: null,            // the newest `screens.jsonl` line seen
    passLabel: "iteration",  // a watch calls its own "pass"
    setRunning: () => {},   // what else on the page follows this surface
    onEvent: () => {},
    onEnd: () => {},
  };
}

/* A surface's readouts back to nothing: for a run starting, and for one being
   reattached to after a reload, where whatever the page last watched is not it.
   The ceilings go too -- the stream replays from the top of the events file, so
   `run_start` puts back the ones this run is actually under. */
function resetCounters(v) {
  v.step = 0;
  v.calls = 0;
  v.cost = 0;
  v.skill = "";
  v.records = 0;
  v.progress = "";
  v.maxSteps = 0;
  v.budget = 0;
  v.now = "";
  v.marked = null;
  v.geo = null;
  v.screen = null;
  if (v.els.ledger) v.els.ledger.hidden = true;
  if (v.phone && v.phone.unmark) v.phone.unmark();
}

/* Keep the phone panel's element box in step with the run.

   The box is only honest inside one window: after a `decide` has named an
   element and before the action lands on it. In there the agent has observed
   and not yet acted, so the screen the panel is polling *is* the screen the
   geometry was measured on -- and it is also the moment the box is worth
   having, since it says what is about to be touched.

   `verify` closes the window: by then the gesture has gone in and the screen
   has settled somewhere else. So does any event belonging to a later step,
   which is what covers the steps that never reach a verify -- a refused
   action, a dismissed nag, a terminal one. */
function markTarget(ev, v) {
  const phone = v.phone;
  if (!phone || !phone.mark) return;
  if (ev.kind === "decide") {
    v.marked = ev.step;
    // A decision with nothing to aim at -- `back`, `done`, a scroll with no
    // element -- clears the last one rather than leaving it up over a screen
    // it has stopped describing. The element list may already be here or may
    // be a frame behind; either way `markScreen` finishes the job.
    v.geo = { w: ev.screen_w, h: ev.screen_h, target: ev.target_element,
              els: v.screen && v.screen.step === ev.step ? v.screen.els : null };
    phone.mark(v.geo);
  } else if (ev.kind === "verify" || ev.kind === "run_end"
             || (v.marked != null && ev.step != null && ev.step !== v.marked)) {
    v.marked = null;
    v.geo = null;
    phone.unmark();
  }
}

/* One line of `screens.jsonl`: where every element the model was shown was.

   Its own SSE channel because it is its own file, and the two are read in one
   pass with the decision first -- so the list usually lands just after the
   target it belongs to, and joins it rather than replacing it. Only the newest
   is kept: the panel draws the step in flight and nothing else. */
function markScreen(rec, v) {
  const phone = v.phone;
  if (!phone || !phone.mark) return;
  v.screen = rec;
  if (v.marked !== rec.step || !v.geo) return;
  v.geo.els = rec.els;
  // The screen's own size, when the decision did not carry one -- which is
  // every run recorded before it did.
  v.geo.w = v.geo.w || rec.w;
  v.geo.h = v.geo.h || rec.h;
  phone.mark(v.geo);
}

/* Keep a surface's clock and status telling the truth.

   Three states, not two: a child that has been asked to stop is still running
   -- still holding the phone, still spending -- for as long as its shutdown
   takes, which for a watch is a model call that writes the app's skill. The
   clock keeps going through it, because that time is still being spent. */
function setLiveRunning(v, running) {
  const stopping = running && v.stopping;
  if (!running) v.stopping = false;
  v.els.state.textContent = stopping ? "stopping…" : (running ? "running" : "idle");
  v.els.state.style.color = stopping ? "var(--yellow)"
    : (running ? "var(--green)" : "var(--text-dim)");
  clearInterval(v.timer);
  if (running) {
    v.timer = setInterval(() => {
      v.els.elapsed.textContent = fmtDur((Date.now() / 1000) - v.startedAt);
    }, 1000);
  }
  if (v.phone) running ? v.phone.start() : v.phone.stop();
  v.setRunning(running, stopping);
}

/* Move a surface into the stopping phase and show it at once.

   Called on the click rather than waiting for the server to say so: the request
   answers immediately now, but a button that only responds once a round trip has
   landed is the same button that looked broken before. */
function setStopping(v) {
  v.stopping = true;
  setLiveRunning(v, true);
}

function openStream(v) {
  if (v.source) return;
  v.box.hidden = false;
  const source = new EventSource(v.url);
  v.source = source;
  const feed = v.feed;

  source.addEventListener("run", (e) => {
    const data = JSON.parse(e.data);
    // Sent again for every `--repeat` iteration, each of which is a separate
    // run in its own directory. Rule off rather than letting the next one's
    // step 1 land against the last one's step 40. The rule goes in at the head
    // like everything else, and the iteration it announces then grows above it.
    if (feed._runId && feed._runId !== data.run_id) {
      finalizeLlm(feed, null);        // an iteration can end mid-call
      const rule = document.createElement("div");
      rule.className = "banner";
      // "pass" for a watch, "iteration" for a --repeat run. The same mechanism
      // underneath -- a new run directory -- but calling a watch's sweep of the
      // inbox an "iteration" reads as though the goal were being retried.
      rule.innerHTML = `<b>${v.passLabel} ${esc(data.iteration || "?")}</b>` +
        `<br><span class="small runid">${esc(data.run_id)}</span>`;
      feedAdd(feed, rule);
      // Steps are per iteration, and so are the ledger and the progress note --
      // each iteration pursues the goal from scratch in its own run directory.
      // Calls and spend are the session's, because that is what --budget-usd
      // bounds.
      v.step = 0;
      v.records = 0;
      v.progress = "";
      feed._llm = null;
      feed._skill = "";
      feed._steps = null;
      feed._curStep = null;
      feed._asked = "";      // the last pass's question is not this one's
      v.marked = null;
      v.geo = null;
      v.screen = null;
      if (v.phone && v.phone.unmark) v.phone.unmark();
      finalizeNotes(feed);   // the last pass's ledger is not this one's
      feed._notes = null;
      if (v.els.ledger) v.els.ledger.hidden = true;
    }
    v.iteration = data.iteration || 1;
    v.els.runid.textContent = data.run_id;
    feed._runId = data.run_id;        // sent before any llm frame, so the
    paintCounters(v);                 // screenshots have a run to come from
  });
  source.addEventListener("event", (e) => {
    const ev = JSON.parse(e.data);
    // Everything one event does to the page inside one measurement: the row it
    // adds or grows, and the readouts, the answer and the ledger above the
    // feed, which push it down the page as they fill.
    holdPlace(feed, () => {
      foldEvent(ev, feed);
      updateCountersFromEvent(ev, v);
      markTarget(ev, v);
      v.onEvent(ev);
    });
  });
  // Geometry lays nothing out: it is drawn over the phone panel, which is not
  // in the feed's flow, so it needs no `holdPlace`.
  source.addEventListener("screen", (e) => {
    markScreen(JSON.parse(e.data), v);
  });
  source.addEventListener("llm", (e) => {
    const ev = JSON.parse(e.data);
    // A stream chunk is one of tens of thousands and lays out nothing: it goes
    // into a buffer, and the frame that flushes it holds the place itself.
    if (ev.kind === "llm_stream") handleLlmEvent(ev, feed);
    else holdPlace(feed, () => handleLlmEvent(ev, feed));
  });
  // The child's own account of its shutdown, which no run file carries: the
  // phone being put back, and the skill written from everything the passes saw.
  source.addEventListener("output", (e) => {
    appendExitLine(feed, JSON.parse(e.data).line);
  });
  source.addEventListener("state", (e) => {
    const st = JSON.parse(e.data);
    if (st.started_at) v.startedAt = st.started_at;
    if (st.stopping) v.stopping = true;
    setLiveRunning(v, !!st.running);
    if (!st.running && st.returncode != null) {
      v.els.state.textContent = "exited (" + st.returncode + ")";
    }
  });
  source.addEventListener("end", () => {
    source.close();
    v.source = null;
    setLiveRunning(v, false);
    finalizeLlm(feed, null);        // a run can end mid-call
    finalizeSteps(feed);            // and mid-step
    v.onEnd();
    // Every child here leaves a run directory behind it -- a goal run, a watch
    // pass, a generation's tour -- so history has something new in it, and
    // whether the one just finished can be resumed is a fact only that
    // directory has.
    loadRuns().then(syncResumeButton).catch(() => {});
  });
  source.onerror = () => {
    // The server closes the connection after "end"; anything else is a drop.
    if (v.source && source.readyState === EventSource.CLOSED) {
      v.source = null;
      setLiveRunning(v, false);
    }
  };
}

/* ----------------------------------------------------------- work tab */

const live = makeLive("c-", "live", "feed");
live.url = "/api/runs/stream";
live.phone = phoneView($("live-phone"), { live: true });
live.setRunning = (running, stopping) => {
  $("btn-start").disabled = running;
  // Shown only while there is something to stop. One SIGINT is all it takes: a
  // second lands in the shutdown -- outside the handler that catches the first
  // -- and takes down the work it is doing there.
  $("btn-stop").hidden = !running;
  $("btn-stop").disabled = stopping;
  // What to do next is only offered once there is no next iteration coming.
  $("run-actions").hidden = running;
};
live.onEvent = (ev) => {
  // The answer, the moment it exists, in the block that leads the surface.
  if (ev.kind === "run_end") showResult(ev, live.feed._runId, live.feed._asked);
};
// The hint is about something in progress -- stopping, resuming -- and none of
// it is still true once the run is over.
live.onEnd = () => { $("run-hint").textContent = ""; };

function setRunningUI(running) {
  setLiveRunning(live, running);
}

/* Compose, watch, read: phases of one surface. The result is not one of them --
   it appears the moment there is an answer, which under `--repeat` is while the
   session is still going. */
function workPhase(phase) {
  $("run-form").hidden = phase !== "compose";
  $("live").hidden = phase === "compose";
  if (phase !== "done") $("work-result").hidden = true;
}

function showResult(ev, runId, asked) {
  const box = $("result-box");
  const outcome = ev.outcome || "unknown";
  box.className = "result-box " + esc(outcome);
  const arith = [`${ev.steps ?? "?"} steps`, `${ev.llm_calls ?? 0} LLM calls`,
                 `$${(ev.usd || 0).toFixed(4)}`].join(" · ");
  box.innerHTML =
    `<div class="verdict"><b>${esc(outcome.toUpperCase())}</b>` +
    `<span>${esc(arith)}</span>` +
    (runId ? `<span class="runid">${esc(runId)}</span>` : "") + `</div>` +
    (ev.result
      ? `<div class="answer">${esc(ev.result)}</div>`
      : `<div class="answer none">Nothing was reported — the run did not reach ` +
        `an answer of its own.</div>`) +
    (ev.evidence ? `<div class="why">${esc(ev.evidence)}</div>` : "");
  box.dataset.runId = runId || "";
  $("work-result").hidden = false;
  $("btn-resume-live").hidden = true;   // until the run list says otherwise
  // Answerable only when the run asked. `syncResumeButton` runs a moment later
  // off the refreshed history and takes the checkpoint into account, which is
  // the other half of the condition: an answer with nothing to resume into has
  // nowhere to go.
  const form = $("run-answer");
  form.hidden = !(outcome === "needs_user" && asked && runId);
  if (!form.hidden) {
    $("answer-text").value = "";
    $("btn-answer").disabled = false;
  }
}

/* Whether the run just finished left a checkpoint. Read off the history list
   rather than guessed from the outcome: a failed run without a checkpoint has
   nothing to resume, and a button that 409s is worse than no button. */
function syncResumeButton() {
  const id = $("result-box").dataset.runId;
  const found = RUNS.find((r) => r.id === id);
  const btn = $("btn-resume-live");
  btn.hidden = !(found && found.resumable);
  btn.onclick = () => resumeRun(id);
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
    assert_shell: $("opt-assert-shell").value.trim(),
    assert_equals: $("opt-assert-equals").value.trim(),
    assert_text: $("opt-assert-text").value.trim(),
  };
  const ms = parseInt($("opt-max-steps").value, 10);
  if (ms) body.max_steps = ms;
  const budget = parseFloat($("opt-budget").value);
  if (!isNaN(budget)) body.budget_usd = budget;
  return body;
}

/* What the folded options currently say, on the line you fold them behind.

   Seven settings, all defaulted, and the run that changes one is rare — so
   they cost every run the space of being read and told most of them nothing.
   Folded they cost a line, and this is that line: only what is actually set,
   with the guardrails first because they are the two that spend money and
   break things. A default left alone is not mentioned, so the summary is
   empty exactly when there is nothing to know. */
function paintRunOpts() {
  const bits = [];
  const budget = $("opt-budget").value.trim();
  const steps = $("opt-max-steps").value.trim();
  const repeat = $("opt-repeat").value.trim();
  const serial = $("opt-serial").value.trim();
  if ($("opt-dry-run").checked) bits.push(`<b class="warn">dry run</b>`);
  if ($("opt-destructive").checked) bits.push(`<b class="bad">destructive allowed</b>`);
  if (budget) bits.push(`$${esc(budget)}`);
  if (steps) bits.push(`${esc(steps)} steps`);
  if (repeat && repeat !== "1") bits.push(`<b class="warn">repeat ${esc(repeat)}</b>`);
  if (serial) bits.push(esc(serial));
  if ($("opt-no-learn").checked) bits.push("no learning");
  if ($("opt-assert-shell").value.trim() || $("opt-assert-text").value.trim())
    bits.push("assertion set");
  $("runopts-summary").innerHTML = bits.length
    ? `options · ${bits.join(" · ")}`
    : `options · <span class="dim">defaults</span>`;
  return bits.length > 0;
}

$("runopts").addEventListener("input", paintRunOpts);
$("runopts").addEventListener("change", paintRunOpts);

/* The same line for a watch's ceilings. These read as "12 replies/hour" rather
   than bare numbers, because a ceiling with no unit on it is the one kind of
   number worth misreading — and the placeholder is the default, so a field
   left alone is reported as the default rather than as blank. */
function paintWatchOpts() {
  const bits = [];
  const put = (id, fmt) => {
    const v = $(id).value.trim();
    if (v) bits.push(fmt(v));
  };
  put("watch-rph", (v) => `${v} replies/hour`);
  put("watch-rpc", (v) => `${v} per conversation`);
  put("watch-cooldown", (v) => `${v}s cooldown`);
  put("watch-usd", (v) => `$${v}/hour`);
  put("watch-serial", (v) => v);
  if ($("watch-no-learn").checked) bits.push("no learning");
  $("watchopts-summary").innerHTML = bits.length
    ? `limits · ${bits.join(" · ")}`
    : `limits · <span class="dim">defaults</span>`;
  return bits.length > 0;
}

$("watchopts").addEventListener("input", paintWatchOpts);
$("watchopts").addEventListener("change", paintWatchOpts);

/* Clear a surface and attach it to whatever is starting now. */
function beginLive(v) {
  resetCounters(v);
  v.iteration = 1;
  v.stopping = false;
  v.startedAt = Date.now() / 1000;
  v.feed.innerHTML = "";
  v.feed._llm = null;
  v.feed._runId = "";
  v.feed._skill = "";
  v.feed._steps = null;
  v.feed._curStep = null;
  v.feed._notes = null;
  v.feed._notesCard = null;
  v.feed._exitLog = null;   // cleared with the feed it hung off
  v.els.runid.textContent = "starting…";
  paintCounters(v);
  setLiveRunning(v, true);
  openStream(v);
}

async function startRun(body) {
  try {
    await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    notice(err.message);
    return false;
  }
  $("run-hint").textContent = "";
  workPhase("live");
  beginLive(live);
  return true;
}

$("run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { goal: $("goal").value.trim(), ...runOptions() };
  if (!body.goal) { notice("a goal is required"); return; }
  saveGoal("adbagent.goal", body.goal);
  await startRun(body);
});

/* The goal box is a textarea because goals are paragraphs, so Enter belongs to
   the text and the run needs the modifier — the same bargain every compose box
   makes. Without it the only way to start was to leave the keyboard. */
$("goal").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    $("run-form").requestSubmit();
  }
});

$("btn-new-run").addEventListener("click", () => {
  workPhase("compose");
  $("goal").focus();
});

$("btn-run-again").addEventListener("click", () => {
  const goal = $("goal").value.trim();
  if (!goal) { workPhase("compose"); $("goal").focus(); return; }
  startRun({ goal, ...runOptions() });
});

/* Continue a failed run from its checkpoint, watched on the work surface.

   Answers whether the resume actually started, because one caller has already
   saved something by the time it asks -- see the answer form below. */
async function resumeRun(id) {
  try {
    await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ resume: id, ...runOptions() }),
    });
  } catch (err) {
    notice(err.message);
    return false;
  }
  document.querySelector('button[data-tab="work"]').click();
  $("runs-list-view").hidden = false;
  $("run-detail-view").hidden = true;
  workPhase("live");
  beginLive(live);
  $("run-hint").textContent = `resuming ${id} from its checkpoint`;
  return true;
}

/* Answer an `ask_user`, then continue.

   Two requests rather than one. They fail for different reasons and want
   different words -- there is nothing to answer, versus the phone is busy --
   and separating them means an answer typed against a busy phone is still
   saved: the Resume button beside this form picks it up whenever the phone is
   free, with nothing to type again. */
$("run-answer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = $("result-box").dataset.runId;
  const text = $("answer-text").value.trim();
  if (!id || !text) return;
  const btn = $("btn-answer");
  btn.disabled = true;
  try {
    await api(`/api/runs/${encodeURIComponent(id)}/answer`,
              { method: "POST", body: JSON.stringify({ text }) });
  } catch (err) {
    notice(err.message);
    btn.disabled = false;
    return;
  }
  // Cleared the moment it is saved rather than when the resume lands: what gets
  // typed here is usually the one-time code the agent stopped rather than
  // invent, and it has no business sitting in the page after the run has it.
  $("answer-text").value = "";
  if (await resumeRun(id)) $("run-answer").hidden = true;
  else btn.disabled = false;
});

$("btn-stop").addEventListener("click", async () => {
  setStopping(live);
  $("run-hint").textContent = "stopping — the agent restores the phone first";
  try {
    await api("/api/runs/stop", { method: "POST" });
  } catch (err) {
    notice(err.message);
  }
});

/* -------------------------------------------------------------- status

   Three facts and whatever is holding the phone. The device one used to lie:
   it printed the serial out of config.json whether or not anything was on the
   other end of it, which is the one fact that decides whether Start run can
   work at all. */

/* The fourth field is a serial the page can offer to switch to. Saying "not
   attached" is the right diagnosis and was, on its own, a dead end: the fix
   lives four navigations away in the config form, and the only thing it wants
   typed is a serial adb is already reporting on this very line. So when there
   is exactly one attached phone and it is not the configured one, the status
   pill carries the fix rather than just the finding. */
function deviceStatus(st) {
  const serials = st.devices_attached || [];
  const serial = st.device_serial || "";
  const lone = serials.length === 1 && serials[0] !== serial ? serials[0] : "";
  if (serial && st.device_attached) return ["ok", `device ${serial}`, "", ""];
  if (serial) {
    return ["warn", `${serial} · not attached`,
      "the serial in config.json is set, but adb does not see it — a run " +
      "cannot start until it is connected", lone];
  }
  if (serials.length === 1) return ["ok", `device ${serials[0]}`, "", ""];
  if (serials.length > 1) {
    return ["warn", `${serials.length} devices · none chosen`,
      "set device.serial in Config, or name one per run", ""];
  }
  return ["warn", "no device", "nothing is attached", ""];
}

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    LAST_STATUS = st;
    const parts = [];
    const [cls, text, why, fix] = deviceStatus(st);
    parts.push(`<span class="st ${cls}"${why ? ` title="${esc(why)}"` : ""}>` +
      `${esc(text)}` +
      (fix ? `<button type="button" class="st-fix" data-use="${esc(fix)}"` +
        ` title="write device.serial = ${esc(fix)} to config.json">` +
        `use ${esc(fix)}</button>` : "") +
      `</span>`);
    parts.push(st.model
      ? `<span class="st">${esc(st.model.split("/").pop())}</span>`
      : `<span class="st warn">no model</span>`);
    parts.push(st.api_key_present
      ? `<span class="st ok">api key</span>`
      : `<span class="st warn">no api key</span>`);
    if (st.run && st.run.running) {
      parts.push(st.run.stopping
        ? `<span class="st warn live">● stopping the run</span>`
        : `<span class="st ok live">● running: ${esc(goalTitle(st.run.goal))}</span>`);
    }
    if (st.watch && st.watch.running) {
      // The mode is part of the status line, not just the tab: a watch outlives
      // every reload, and "is it sending?" should be answerable at a glance from
      // any tab. A watch on its way out is neither mode -- it is no longer
      // replying to anything -- so it says that instead.
      parts.push(st.watch.stopping
        ? `<span class="st warn live">● stopping the watch — writing up what it learned</span>`
        : st.watch.draft
        ? `<span class="st ok live">● watching (draft): ${esc(goalTitle(st.watch.goal))}</span>`
        : `<span class="st bad live">● watching LIVE: ${esc(goalTitle(st.watch.goal))}</span>`);
    }
    if (st.job) parts.push(`<span class="st ok live">● generating a skill</span>`);
    $("status").innerHTML = parts.join("");
    // Reattach to a run already in progress (e.g. page reloaded mid-run).
    if (st.run && st.run.running && !live.source) {
      resetCounters(live);
      live.startedAt = st.run.started_at || Date.now() / 1000;
      live.stopping = !!st.run.stopping;
      workPhase("live");
      setRunningUI(true);
      openStream(live);
    }
    // And to a watch, which outlives a reload by days rather than minutes --
    // including a reload during the minute it takes to stop one.
    if (st.watch && st.watch.running && !watchLive.source) {
      resetCounters(watchLive);
      watchLive.startedAt = st.watch.started_at || Date.now() / 1000;
      watchLive.stopping = !!st.watch.stopping;
      $("watch-draft").checked = !!st.watch.draft;
      setLiveRunning(watchLive, true);
      openStream(watchLive);
    }
    // And to a generation, which outlives a reload just as long.
    if (st.job && !genLive.source) {
      resetCounters(genLive);
      genLive.startedAt = st.job.started_at || Date.now() / 1000;
      watchGeneration(st.job.id, { fresh: false });
    }
    if (st.live_reload) openReloadStream();
  } catch {
    $("status").innerHTML = `<span class="st bad">server unreachable</span>`;
  }
}

let LAST_STATUS = {};

/* Delegated, because the status line is rebuilt from a string on every poll
   and a listener bound to the button would be thrown away with it. */
$("status").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-use]");
  if (!btn) return;
  const serial = btn.dataset.use;
  btn.disabled = true;
  try {
    await api("/api/device/use",
      { method: "POST", body: JSON.stringify({ serial }) });
    notice(`device.serial is now ${serial}`, false);
    await refreshStatus();
    // The Device pane and the Config form both show the serial that just
    // changed; refetch whichever the reader has already opened.
    if (loadedPanes.has("device")) loadDevices().catch(() => {});
    if (loadedPanes.has("config")) loadConfig().catch(() => {});
  } catch (err) {
    notice(err.message);
    btn.disabled = false;
  }
});

/* -------------------------------------------------------------- watch */

const watchLive = makeLive("w-", "watch-live", "watch-feed");
watchLive.url = "/api/watch/stream";
watchLive.passLabel = "pass";
watchLive.phone = phoneView($("watch-phone"), { live: true });
watchLive.setRunning = (running, stopping) => {
  $("btn-watch-start").disabled = running;
  // Not clickable twice. The second SIGINT arrives while the first is being
  // acted on -- during the skill write-up, which is not inside the loop's
  // interrupt handler -- and losing that is losing everything the watch learned.
  $("btn-watch-stop").hidden = !running;
  $("btn-watch-stop").disabled = stopping;
  // The policy is read once, at startup. Editing it under a running watch would
  // take effect at no predictable moment, so the server refuses and the form
  // says so before you type into it. The picker locks with it: while a watch is
  // running there is one policy in play, and offering to switch away from it
  // would only be offering to mislead about which one that is.
  $("watch-policy").readOnly = running;
  $("btn-policy-save").disabled = running;
  $("btn-policy-new").disabled = running;
  $("watch-policy-select").disabled = running;
  paintWatchBanner(running, stopping);
};
// Every pass writes a reply or it does not; either way the ledger is what
// changed, so refresh it when one ends rather than making the reader ask.
watchLive.onEvent = (ev) => {
  if (ev.kind === "reply_attempt" || ev.kind === "reply_confirmed"
      || ev.kind === "run_end") loadLedger().catch(() => {});
};
watchLive.onEnd = () => { $("watch-hint").textContent = ""; };

let watchDefaults = {};

function paintWatchBanner(running, stopping) {
  const draft = $("watch-draft").checked;
  const el = $("watch-mode-banner");
  el.className = "banner " + (draft ? "ok" : "danger");
  el.innerHTML = draft
    ? "<b>DRAFT</b> — replies are composed and recorded, and never sent."
      + "<br><span class=\"small\">Read what it would have said in the run feed,"
      + " then uncheck “draft only” when the drafts look right.</span>"
    : "<b>LIVE</b> — replies WILL be sent to real people from this device."
      + "<br><span class=\"small\">The harness will not answer the same message"
      + " twice, whatever the policy says.</span>";
  if (stopping) {
    el.innerHTML += "<br><span class=\"small\">stopping — no more replies will "
      + "be sent; it is finishing up and writing what it learned.</span>";
  } else if (running) {
    el.innerHTML += "<br><span class=\"small\">watching…</span>";
  }
}

function watchOptions() {
  const num = (id, int) => {
    const raw = $(id).value.trim();
    if (raw === "") return null;
    const n = int ? parseInt(raw, 10) : parseFloat(raw);
    return isNaN(n) ? null : n;
  };
  return {
    goal: $("watch-goal").value.trim(),
    // Which policy, said explicitly rather than left to config: the tab may be
    // pointed at any of them, and the one on screen is the one to start.
    policy: policyPath,
    draft: $("watch-draft").checked,
    no_learn: $("watch-no-learn").checked,
    serial: $("watch-serial").value.trim(),
    interval_s: num("watch-interval"),
    sweep_s: num("watch-sweep"),
    max_steps: num("watch-steps", true),
    replies_per_hour: num("watch-rph", true),
    replies_per_conversation: num("watch-rpc", true),
    cooldown_s: num("watch-cooldown"),
    usd_per_hour: num("watch-usd"),
  };
}

/* ------------------------------------------------------------ policies

   Several of them, and each carries the goal it was written for. The pairing is
   the whole point: the Hinge policy is only correct under "work through Discover
   and reply to matches", and starting it under the goal still in the box from
   the Instagram policy is a watch doing the wrong thing carefully. So the goal
   lives in the policy file itself and choosing a policy fills the box in.

   `policyPath` is the one fact both halves of this tab read: the composer starts
   that policy, the editor below edits it. */

let POLICIES = [];
let policiesDir = "";
let policyPath = "";     // the policy this tab is pointed at
let policyGoal = "";     // the goal saved *in* it, as of the last load
const POLICY_KEY = "adbagent.watch-policy";

function policyAt(path) {
  return POLICIES.find((p) => p.path === path) || null;
}

/* The list, and which of them the tab is on.

   Precedence: whatever is already open, then the one this browser last chose,
   then `watch.policy` from config. A remembered path that has since been
   deleted falls through rather than leaving the tab pointed at nothing. */
async function loadPolicies() {
  const data = await api("/api/watch/policies");
  POLICIES = data.policies || [];
  policiesDir = data.dir || "";
  // The configured one as the *list* spells it: config may name it by a relative
  // path and the listing by an absolute one, and the server has already worked
  // out which row that is. Comparing the two strings here would not.
  const configured = POLICIES.find((p) => p.current);
  const wanted = [policyPath, loadValue(POLICY_KEY) || "",
                  configured ? configured.path : ""];
  policyPath = wanted.find((p) => p && policyAt(p))
    || (POLICIES[0] ? POLICIES[0].path : "");
  paintPolicySelect();
  return data;
}

function paintPolicySelect() {
  const sel = $("watch-policy-select");
  if (!sel) return;
  const opts = POLICIES.map((p) => {
    // The name answers "which file"; the goal answers "what is it for", which
    // is the part a filename never says.
    const what = p.goal ? ` — ${trunc(p.goal, 52)}`
                        : (p.exists ? "" : " — not written yet");
    return `<option value="${esc(p.path)}">${esc((p.name || p.path) + what)}</option>`;
  });
  if (!POLICIES.length) {
    opts.push(`<option value="">${esc(policiesDir
      ? `no policies in ${policiesDir} yet — New… writes one`
      : "no policies directory — set watch.policies_dir in Config")}</option>`);
  }
  sel.innerHTML = opts.join("");
  sel.value = policyPath;
}

/* What this policy is for, and whether the goal at the top of the tab still
   matches it. Divergence is legitimate -- a one-off start under a different
   goal -- so it is reported rather than corrected, with the two ways out named:
   adopt the saved goal, or save the new one over it. */
function paintPolicyGoalNote() {
  const el = $("policy-goal-note");
  if (!el) return;
  const goal = $("watch-goal").value.trim();
  if (!policyPath) {
    el.innerHTML = "<span class=\"warn\">No policy selected.</span> Set "
      + "<code>watch.policies_dir</code> in Config, or write one with New….";
    return;
  }
  if (!policyGoal) {
    el.innerHTML = "No goal saved with this policy yet — <b>Save policy</b> "
      + "stores whatever is in the box at the top of this tab, so choosing it "
      + "next time brings the goal back with it.";
    return;
  }
  if (goal === policyGoal) {
    el.innerHTML = `Goal saved with this policy: <b>${esc(policyGoal)}</b>`;
    return;
  }
  el.innerHTML = `<span class="warn">The goal at the top is not the one saved `
    + `here.</span> Saved: <b>${esc(policyGoal)}</b> `
    + `<button type="button" class="linkbtn" data-act="use-goal">use it</button>`
    + ` · <b>Save policy</b> stores the one at the top instead.`;
}

/* Put the policy's own goal in the box. What "the goal follows the policy"
   actually means, and the reason the two are stored together. */
function applyPolicyGoal() {
  if (!policyGoal) return false;
  $("watch-goal").value = policyGoal;
  saveValue("adbagent.watch-goal", policyGoal);
  paintPolicyGoalNote();
  return true;
}

/* Point the tab at another policy. Choosing one is a deliberate act, so it
   adopts that policy's goal -- which is the whole of what "the goal follows the
   policy" means. A policy with no goal of its own leaves the box alone rather
   than clearing it: there is nothing to replace it with. */
async function selectPolicy(path) {
  policyPath = path || "";
  saveValue(POLICY_KEY, policyPath);
  paintPolicySelect();
  await loadPolicy();
  applyPolicyGoal();
}

async function loadWatch() {
  const data = await api("/api/watch");
  watchDefaults = data.defaults || {};
  $("watch-policy-path").textContent = data.policy_path || "(no policy path set)";
  $("watch-ledger-path").textContent = data.ledger_path || "";
  // Placeholders, not values: an empty field means "whatever config says", and
  // filling them in would silently pin today's defaults into every start.
  const ph = (id, v) => { if (v !== undefined && v !== null) $(id).placeholder = String(v); };
  ph("watch-interval", watchDefaults.interval_s);
  // Only when it is on: the markup already says "off", which is what 0 means
  // and a good deal clearer than showing a 0.
  if (watchDefaults.sweep_s) ph("watch-sweep", watchDefaults.sweep_s);
  ph("watch-steps", watchDefaults.max_steps);
  ph("watch-rph", watchDefaults.max_replies_per_hour);
  ph("watch-rpc", watchDefaults.max_replies_per_thread_per_hour);
  ph("watch-cooldown", watchDefaults.thread_cooldown_s);
  const active = data.active || {};
  // An active watch's goal is the last prompt that actually ran; show it over
  // the remembered one. With no active watch, the remembered goal stays.
  if (active.goal) $("watch-goal").value = active.goal;
  if (active.running) $("watch-draft").checked = !!active.draft;
  paintWatchBanner(!!active.running, !!active.stopping);
  // A running watch pins the policy it was started with; nothing else here has
  // a claim on the selection, so the remembered one stands.
  if (active.running && active.policy) policyPath = active.policy;
  await loadPolicies();
  await loadPolicy();
  // Only into an empty box: on a page reload the remembered goal is somebody's
  // typing, and adopting over it would discard an edit nobody asked to discard.
  // A deliberate change of policy is the case that adopts -- see `selectPolicy`.
  if (!$("watch-goal").value.trim()) applyPolicyGoal();
  await loadLedger();
}

async function loadPolicy() {
  const query = policyPath ? "?path=" + encodeURIComponent(policyPath) : "";
  const data = await api("/api/watch/policy" + query);
  const box = $("watch-policy");
  // `_loaded` is what the file said. Kept so live reload can tell a box nobody
  // has touched -- safe to replace -- from one holding an unsaved edit.
  box.value = box._loaded = data.text || "";
  policyGoal = data.goal || "";
  $("policy-editing").textContent = data.name || "";
  $("watch-policy-path").textContent =
    data.path ? data.path + (data.exists ? "" : " (not written yet)")
              : "(no policy path set — set watch.policy in Config)";
  paintPolicyGoalNote();
}

async function loadLedger() {
  const data = await api("/api/watch/ledger");
  const tbody = document.querySelector("#watch-ledger-table tbody");
  tbody.innerHTML = "";
  const rows = data.threads || [];
  $("watch-ledger-empty").hidden = rows.length > 0;
  $("watch-ledger-table").hidden = rows.length === 0;
  $("watch-ledger-path").textContent = data.path || "";
  for (const t of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${esc(t.preview || t.thread_key)}</td>` +
      `<td>${t.reply_count}</td>` +
      `<td class="small" title="${esc(fmtTime(t.last_attempt_at))}">` +
      `${esc(fmtRel(t.last_attempt_at))}</td>` +
      `<td class="small">${t.confirmed
        ? "<span class=\"ok\">confirmed</span>"
        : "<span class=\"warn\">unconfirmed — in doubt</span>"}</td>`;
    tbody.appendChild(tr);
  }
}

$("watch-draft").addEventListener("change", () => paintWatchBanner(false));

// Typing a goal can put it out of step with the policy that is open. Said as it
// happens rather than discovered at the moment of starting.
$("watch-goal").addEventListener("input", paintPolicyGoalNote);

$("watch-policy-select").addEventListener("change", (e) => {
  selectPolicy(e.target.value).catch((err) => notice(err.message));
});

$("policy-goal-note").addEventListener("click", (e) => {
  if (e.target.dataset && e.target.dataset.act === "use-goal") applyPolicyGoal();
});

$("btn-policy-new").addEventListener("click", async () => {
  const name = prompt(
    policiesDir ? `Name for the new policy — a file in ${policiesDir}:`
                : "Name for the new policy:", "");
  if (name === null) return;                       // cancelled
  if (!name.trim()) { notice("a policy needs a name"); return; }
  try {
    // Starts on the goal in the box, since a new policy is usually written for
    // the thing you were just about to ask for.
    const created = await api("/api/watch/policies", {
      method: "POST",
      body: JSON.stringify({ name: name.trim(),
                             goal: $("watch-goal").value.trim() }),
    });
    policyPath = created.path;
    saveValue(POLICY_KEY, policyPath);
    await loadPolicies();
    await loadPolicy();
    notice(`created ${created.path} — write the instructions and save`, false);
    $("watch-policy").focus();
  } catch (err) { notice(err.message); }
});

$("btn-policy-save").addEventListener("click", async () => {
  try {
    // Both halves, always: the instructions and the goal they are correct
    // under. Saving one and not the other is how the pair comes apart.
    const r = await api("/api/watch/policy", {
      method: "PUT",
      body: JSON.stringify({ path: policyPath, text: $("watch-policy").value,
                             goal: $("watch-goal").value.trim() }),
    });
    notice(`policy saved to ${r.path}`, false);
    await loadPolicies();      // the list shows each policy's goal; one changed
    await loadPolicy();
  } catch (err) { notice(err.message); }
});

$("btn-ledger-reload").addEventListener("click", () =>
  loadLedger().catch((err) => notice(err.message)));

$("watch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = watchOptions();
  if (!body.goal) { notice("say what to watch"); return; }
  saveGoal("adbagent.watch-goal", body.goal);
  if (!body.draft &&
      !confirm("This will send real replies from this device.\n\n" +
               "Have you read what it drafts first?")) return;
  try {
    await api("/api/watch", { method: "POST", body: JSON.stringify(body) });
  } catch (err) { notice(err.message); return; }
  $("watch-hint").textContent = "";
  // The same clean slate a run gets, including the ledger and the notes panels:
  // a watch starting is not a continuation of the last one.
  beginLive(watchLive);
});

$("btn-watch-stop").addEventListener("click", async () => {
  // Said before the request, not after it: a watch takes as long to stop as its
  // shutdown takes, and the whole complaint about this button was that it looked
  // inert until that was over.
  setStopping(watchLive);
  $("watch-hint").textContent =
    "stopping — restoring the phone, then writing up what it learned";
  try {
    await api("/api/watch/stop", { method: "POST" });
  } catch (err) { notice(err.message); }
});

/* ------------------------------------------------------------ history

   On the work surface rather than a tab of its own: you start a run and read
   what it did, and those were never two places.

   Every row carries the answer. A list that shows outcome, steps, cost and
   duration and *not* what the run concluded makes the one interesting field the
   one you have to click for. */

let RUNS = [];
let RUNS_ACTIVE = false;
let RUN_FILTER = { q: "", outcome: "" };

async function loadRuns() {
  const data = await api("/api/runs");
  RUNS = data.runs || [];
  RUNS_ACTIVE = !!(data.active && data.active.running);
  paintRuns();
  paintRecentGoals();
  paintStanding();
}

/* The filter matches the outcome word the chip already says, so there is
   nothing to learn: `aborted` filters to the rows chipped `aborted`. The one
   that is not an outcome is `resumable`, which is the question this list was
   worst at answering — a checkpoint is the most useful thing in a history and
   it was reachable only by opening runs one at a time to see if they had one. */
function runMatches(r) {
  const f = RUN_FILTER.outcome;
  if (f === "resumable") {
    if (!r.resumable) return false;
  } else if (f && r.outcome !== f) {
    return false;
  }
  if (!RUN_FILTER.q) return true;
  const hay = `${r.goal} ${r.result} ${r.evidence} ${r.id} ${r.outcome}`.toLowerCase();
  return RUN_FILTER.q.split(/\s+/).every((word) => hay.includes(word));
}

/* One run's row. A button, so it is reachable from the keyboard -- these were
   `<tr>`s with a click listener. */
function runRow(r, { showGoal = true } = {}) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = "run";
  const rest = goalRest(r.goal);
  el.innerHTML =
    `<span class="oc">${chip(r.outcome, r.outcome)}</span>` +
    (showGoal
      ? `<div class="goal" title="${esc(r.goal)}">${esc(goalTitle(r.goal))}` +
        (rest ? ` <span class="small">+${rest} more lines</span>` : "") + `</div>`
      : "") +
    (r.result
      ? `<div class="ans">${esc(r.result)}</div>`
      : `<div class="ans none">${esc(r.evidence
          || "no answer recorded")}</div>`) +
    `<div class="meta">` +
      `<span title="${esc(fmtTime(r.started))}">${esc(fmtRel(r.started))}</span>` +
      `<span>${esc(r.steps)} steps</span>` +
      `<span>$${(r.usd || 0).toFixed(3)}</span>` +
      `<span>${esc(fmtDur(r.duration_s))}</span>` +
      `<span class="runid">${esc(r.id)}</span>` +
    `</div>`;
  // A failed or interrupted run with a checkpoint can be continued rather than
  // started over -- exactly like `run --resume`.
  if (r.resumable && !RUNS_ACTIVE) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "resume";
    btn.textContent = "resume";
    btn.title = `continue from step ${r.steps}`;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();          // the row itself opens the detail view
      resumeRun(r.id);
    });
    el.querySelector(".meta").appendChild(btn);
  }
  el.addEventListener("click", () => openRunDetail(r.id));
  return el;
}

/* Runs of the same goal, folded together. The real story of a history like this
   is one person retrying one goal six times, and fourteen rows that read the
   same is the list refusing to say so. The newest attempt is the summary, so
   folding hides nothing that was not already older news. */
function groupRuns(runs) {
  const groups = [];
  const byGoal = new Map();
  for (const r of runs) {
    const key = goalKey(r.goal);
    let g = byGoal.get(key);
    if (!g) {
      g = { key, goal: r.goal, runs: [], titles: new Set() };
      byGoal.set(key, g);
      groups.push(g);
    }
    g.runs.push(r);
    g.titles.add(goalTitle(r.goal));
  }
  return groups;
}

function paintRuns() {
  const list = $("runs-list");
  list.innerHTML = "";
  const shown = RUNS.filter(runMatches);
  $("runs-empty").hidden = RUNS.length > 0;
  const groups = groupRuns(shown);
  $("runs-count").textContent = RUNS.length
    ? `${shown.length} of ${RUNS.length} run${RUNS.length === 1 ? "" : "s"}` +
      (groups.length !== shown.length ? ` · ${groups.length} goals` : "")
    : "";
  if (RUNS.length && !shown.length) {
    const p = document.createElement("p");
    p.className = "small";
    p.textContent = "Nothing matches that.";
    list.appendChild(p);
  }
  for (const g of groups) {
    if (g.runs.length === 1) {
      list.appendChild(runRow(g.runs[0]));
      continue;
    }
    const newest = g.runs[0];
    const ok = g.runs.filter((r) => r.outcome === "success").length;
    const spend = g.runs.reduce((a, r) => a + (r.usd || 0), 0);
    const left = g.runs.filter((r) => r.resumable).length;
    const det = document.createElement("details");
    det.className = "rungroup";
    const rest = goalRest(g.goal);
    // What a group of attempts is worth knowing is what it cost to get a
    // success out of it, and a group that never got one is the one worth
    // seeing at a glance — eleven attempts and $3.62 for nothing is a goal to
    // rewrite, not to run a twelfth time.
    const each = ok ? `$${(spend / ok).toFixed(3)} each` : "nothing worked";
    det.innerHTML =
      `<summary>` +
        `<span class="oc">${chip(newest.outcome, newest.outcome)}</span>` +
        `<div class="goal" title="${esc(g.goal)}">${esc(goalTitle(g.goal))}` +
        (rest ? ` <span class="small">+${rest} more lines</span>` : "") +
        (g.titles.size > 1
          ? ` <span class="variants" title="${esc([...g.titles].join("\n\n"))}">` +
            `${g.titles.size} wordings</span>`
          : "") + `</div>` +
        (newest.result ? `<div class="ans">${esc(newest.result)}</div>` : "") +
        `<div class="tally">${g.runs.length} attempts · ` +
        `<span class="${ok ? "" : "bad"}">${ok} succeeded</span> · ` +
        `$${spend.toFixed(3)} · <span class="${ok ? "" : "bad"}">${each}</span>` +
        (left ? ` · <span class="warn">${left} resumable</span>` : "") +
        ` · last ` +
        `<span title="${esc(fmtTime(newest.started))}">${esc(fmtRel(newest.started))}</span>` +
        ` <span class="more">show each</span></div>` +
      `</summary><div class="kids"></div>`;
    const kids = det.querySelector(".kids");
    // A group that folded several wordings together has to show them, or the
    // rows inside it are indistinguishable and the fold has lost the thing it
    // folded on.
    const showGoal = g.titles.size > 1;
    for (const r of g.runs) kids.appendChild(runRow(r, { showGoal }));
    list.appendChild(det);
  }
}

$("runs-search").addEventListener("input", () => {
  RUN_FILTER.q = $("runs-search").value.trim().toLowerCase();
  paintRuns();
});

document.querySelector("#runs-list-view .seg").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-outcome]");
  if (btn) setOutcomeFilter(btn.dataset.outcome);
});

/* The five most recent distinct goals, as chips that fill the box. The blank
   textarea was a cold start on a page whose whole history is retries.

   Distinct by intent, not by string. Keyed on the exact goal these came back
   as five chips reading "First go to Discover tab in Hinge and send likes
   with 'Hey…" — the same truncation five times, twice for literally the same
   text, because what told them apart was a number thirty characters past
   where the chip ended. One chip per intent, and each chip says how that
   intent has actually gone, so picking one is a decision rather than a
   guess. */
function paintRecentGoals() {
  const wrap = $("recent-goals");
  const box = $("recent-chips");
  box.innerHTML = "";
  const seen = new Map();
  for (const r of RUNS) {
    const goal = (r.goal || "").trim();
    if (!goal) continue;
    const key = goalKey(goal);
    // The newest wording wins the chip; the rest of the group only feeds the
    // tally, so the chip fills the box with a goal that was really run.
    let g = seen.get(key);
    if (!g) {
      if (seen.size === 5) continue;
      g = { goal, ok: 0, n: 0 };
      seen.set(key, g);
    }
    g.n += 1;
    if (r.outcome === "success") g.ok += 1;
  }
  const picks = [...seen.values()];
  wrap.hidden = !picks.length;
  for (const p of picks) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.innerHTML = `<span>${esc(goalTitle(p.goal))}</span>` +
      (p.n > 1 ? `<span class="tally ${p.ok ? "" : "bad"}">${p.ok}/${p.n}</span>` : "");
    btn.title = `${p.ok} of ${p.n} succeeded\n\n${p.goal}`;
    btn.addEventListener("click", () => {
      $("goal").value = p.goal;
      saveValue("adbagent.goal", p.goal);
      $("goal").focus();
    });
    box.appendChild(btn);
  }
}

/* ------------------------------------------------------------- standing

   The sums. Every number this page showed was about one run, and the
   questions a history like this actually raises are all totals: what has it
   cost, how often does it work, and how much of it is sitting half-finished.
   169 runs, 79 of them successful and 90 of them holding a checkpoint nobody
   went back to — none of that was on the page, and none of it needs a new
   endpoint, only the addition the reader was being left to do. */
function paintStanding() {
  const box = $("standing");
  if (!RUNS.length) { box.hidden = true; return; }
  const ok = RUNS.filter((r) => r.outcome === "success").length;
  const spend = RUNS.reduce((a, r) => a + (r.usd || 0), 0);
  const secs = RUNS.reduce((a, r) => a + (r.duration_s || 0), 0);
  const left = RUNS.filter((r) => r.resumable).length;
  const rate = Math.round((ok / RUNS.length) * 100);
  const cell = (k, v, cls = "", title = "") =>
    `<span class="sc ${cls}"${title ? ` title="${esc(title)}"` : ""}>` +
    `<b>${v}</b><span class="sk">${k}</span></span>`;
  box.innerHTML =
    cell("runs", RUNS.length) +
    cell("succeeded", `${rate}%`, rate >= 50 ? "ok" : "warn",
      `${ok} of ${RUNS.length}`) +
    cell("spent", `$${spend.toFixed(2)}`, "",
      ok ? `$${(spend / ok).toFixed(2)} per success` : "nothing has succeeded yet") +
    cell("on the phone", fmtDur(secs)) +
    (left
      // The one number here that is a thing to do rather than a thing to know.
      ? `<button type="button" class="sc act" id="standing-resumable"` +
        ` title="runs that stopped with a checkpoint — each can be continued` +
        ` instead of started over">` +
        `<b>${left}</b><span class="sk">resumable</span></button>`
      : "");
  box.hidden = false;
  const btn = $("standing-resumable");
  if (btn) {
    btn.addEventListener("click", () => {
      setOutcomeFilter("resumable");
      $("runs-list-view").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
}

/* Set from the filter buttons and from the standing strip, so both agree
   about which button is lit. */
function setOutcomeFilter(outcome) {
  RUN_FILTER.outcome = outcome;
  document.querySelectorAll("#runs-list-view .seg button[data-outcome]")
    .forEach((b) => b.classList.toggle("on", b.dataset.outcome === outcome));
  paintRuns();
}

/* Every skill a run had in its prompt, in the order it picked them up.
   Consecutive repeats collapse, because `active_skill` is recorded per step.
   A goal spanning two apps whose chain names only one is the tell that the
   other had no skill on disk to load -- which is worth seeing without reading
   the whole feed. */
function skillChain(events) {
  const chain = [];
  for (const ev of events || []) {
    if (ev.kind === "active_skill" && ev.name && chain[chain.length - 1] !== ev.name) {
      chain.push(ev.name);
    }
  }
  return chain;
}

async function openRunDetail(id) {
  $("runs-list-view").hidden = true;
  $("run-detail-view").hidden = false;
  $("detail-title").textContent = id;
  const feed = $("detail-feed");
  feed.innerHTML = "";
  feed._llm = null;
  feed._runId = id;
  feed._skill = "";
  feed._steps = null;
  feed._curStep = null;
  feed._notes = null;
  feed._notesCard = null;
  feed._ledgerBox = $("detail-ledger");
  $("detail-ledger").hidden = true;
  $("run-detail-view").scrollIntoView({ block: "start" });
  try {
    const d = await api("/api/runs/" + encodeURIComponent(id));
    const s = d.summary, st = d.stats, chain = skillChain(d.events);
    const resumeBtn = $("btn-resume-run");
    resumeBtn.hidden = !s.resumable;
    resumeBtn.onclick = () => resumeRun(id);
    // The goal once, as the headline. It used to be here, in the run_start
    // banner in the feed, and on the row that opened this page.
    //
    // A watch pass's goal is twenty lines of standing instructions whose first
    // line is the whole of what distinguishes it; setting all twenty as the
    // headline pushes the answer off the screen. So: the first line, and the
    // rest one click away -- folded, not truncated.
    const rest = goalRest(s.goal);
    $("detail-goal").innerHTML =
      `<span class="gt">${esc(goalTitle(s.goal))}</span>` +
      (rest ? `<details class="goal-rest"><summary class="small">` +
        `+${rest} more line${rest === 1 ? "" : "s"} of instructions</summary>` +
        `<pre class="log">${esc(s.goal)}</pre></details>` : "");
    // The answer above the arithmetic, because "what did it conclude" outranks
    // "how many tokens did that take" for everyone who opens a finished run.
    const box = $("detail-answer");
    box.className = "result-box " + esc(s.outcome || "");
    box.innerHTML =
      `<div class="verdict"><b>${esc((s.outcome || "").toUpperCase())}</b>` +
      `<span title="${esc(fmtTime(s.started))}">${esc(fmtRel(s.started))}</span></div>` +
      (s.result
        ? `<div class="answer">${esc(s.result)}</div>`
        : `<div class="answer none">Nothing was reported — this run did not ` +
          `reach an answer of its own.</div>`) +
      (s.evidence ? `<div class="why">${esc(s.evidence)}</div>` : "");
    // Four numbers anybody wants, and the cost of thinking behind a fold: a flat
    // mono row gave `latency p90` the same weight as `outcome`.
    $("detail-metrics").innerHTML =
      `<span>steps <b class="strong">${esc(s.steps)}</b></span>` +
      `<span>cost <b class="strong">$${(s.usd || 0).toFixed(4)}</b></span>` +
      `<span>took <b class="strong">${esc(fmtDur(s.duration_s))}</b></span>` +
      `<span>model <b>${esc((s.model || "").split("/").pop() || "—")}</b></span>` +
      (chain.length ? `<span>skills <b>${esc(chain.join(" → "))}</b></span>` : "");
    $("detail-stats").innerHTML =
      `<span>decisions <b>${st.decisions}</b></span>` +
      `<span>llm calls <b>${st.llm_calls}</b></span>` +
      `<span>latency/step <b>${st.latency_median_s}s med · ${st.latency_p90_s}s p90</b></span>` +
      `<span>prompt tok <b>${st.prompt_tokens.toLocaleString()}</b></span>` +
      `<span>cached tok <b>${st.cached_tokens.toLocaleString()}</b></span>` +
      `<span>completion tok <b>${st.completion_tokens.toLocaleString()}</b></span>` +
      (st.reasoning_tokens
        ? `<span>thinking tok <b>${st.reasoning_tokens.toLocaleString()}</b></span>` : "") +
      (st.sweep_reads ? `<span>sweep reads <b>${st.sweep_reads}</b></span>` : "");
    for (const ev of d.events) {
      // Stream records render as the same live panels, long since collapsed.
      if (typeof ev.kind === "string" && ev.kind.startsWith("llm_")) {
        handleLlmEvent(ev, feed);
      } else {
        foldEvent(ev, feed);
      }
    }
    finalizeLlm(feed, null);  // a run that died mid-call has no llm_end
    finalizeSteps(feed);      // nothing is still thinking in a finished run
    // Every per-step ledger panel folded, unlike a live run's last one: this
    // page already opens with the finished ledger in full, above the feed.
    finalizeNotes(feed);
    // A run recorded before the events carried their records still has the
    // server's replay of the ledger, which is better than nothing at all.
    if ((!feed._notes || !feed._notes.size) && d.scratchpad) {
      const pin = $("detail-ledger");
      pin.hidden = false;
      pin.innerHTML = `<div class="lk">collected data</div>`;
      const pre = document.createElement("pre");
      pre.className = "log";
      pre.textContent = d.scratchpad;
      pin.appendChild(pre);
    }
  } catch (err) {
    notice(err.message);
    return;
  }
  api("/api/runs/" + encodeURIComponent(id) + "/log")
    .then((d) => { $("detail-log").textContent = d.text || "(empty)"; })
    .catch(() => { $("detail-log").textContent = "(unavailable)"; });
}

$("btn-back-runs").addEventListener("click", () => {
  $("runs-list-view").hidden = false;
  $("run-detail-view").hidden = true;
  loadRuns().catch((e) => notice(e.message));
});

/* ------------------------------------------------------------ devices */

/* Setup's own frame. Not polled -- it is grabbed when you ask for it -- so it
   can afford a sharper one than the panels that take a picture every two
   seconds while a run is going. */
const setupPhone = phoneView($("setup-phone"), { edge: 1080 });

async function loadDevices() {
  const d = await api("/api/devices");
  $("dev-error").textContent = d.error ? `adb: ${d.error}` : "";
  const tbody = document.querySelector("#dev-table tbody");
  tbody.innerHTML = "";
  $("dev-empty").hidden = d.devices.length > 0;
  $("dev-table").hidden = d.devices.length === 0;
  $("dev-candidates").textContent =
    d.candidates.length ? `wireless debugging advertised at: ${d.candidates.join(", ")}` : "";
  const configured = (LAST_STATUS.device_serial || "").trim();
  for (const dev of d.devices) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="mono">${esc(dev.serial)}</td><td>${esc(dev.model)}</td>` +
      `<td>${esc(dev.android)}</td><td></td>`;
    const cell = tr.lastElementChild;
    // The table listed every phone adb can see and offered no way to run on
    // one of them. Choosing meant reading the serial off this row and typing
    // it into the config form two panes away.
    if (dev.serial === configured) {
      const tag = document.createElement("span");
      tag.className = "small ok-text";
      tag.textContent = "in use";
      cell.appendChild(tag);
    } else {
      const use = document.createElement("button");
      use.className = "primary";
      use.textContent = "use this";
      use.title = `write device.serial = ${dev.serial} to config.json`;
      use.addEventListener("click", async () => {
        use.disabled = true;
        try {
          await api("/api/device/use",
            { method: "POST", body: JSON.stringify({ serial: dev.serial }) });
          notice(`device.serial is now ${dev.serial}`, false);
          await refreshStatus();
          await loadDevices();
          if (loadedPanes.has("config")) loadConfig().catch(() => {});
        } catch (err) {
          notice(err.message);
          use.disabled = false;
        }
      });
      cell.appendChild(use);
    }
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = "screenshot";
    btn.addEventListener("click", () => {
      $("shot-serial").textContent = dev.serial;
      setupPhone.refresh();
    });
    cell.appendChild(btn);
    tbody.appendChild(tr);
  }
  paintDeviceState(d.devices);
  // No point asking for a frame there is no phone to give.
  if (d.devices.length) setupPhone.refresh();
  else setupPhone.idle("nothing attached");
}

/* Configured, attached, or neither — said in words, because it is the fact that
   decides whether a run can start and the header used to report the first as
   though it were the second. */
function paintDeviceState(devices) {
  const serial = (LAST_STATUS.device_serial || "").trim();
  const serials = devices.map((d) => d.serial);
  let cls = "warn", what = "", why = "";
  if (serial && serials.includes(serial)) {
    cls = "ok";
    what = "attached";
    why = `<span class="mono">${esc(serial)}</span> is configured and adb sees it. Runs will use it.`;
  } else if (serial) {
    cls = "warn";
    what = "configured, not attached";
    why = `<span class="mono">${esc(serial)}</span> is set as ` +
      `<span class="mono">device.serial</span> but adb does not see it. ` +
      (serials.length
        ? `What is attached: <span class="mono">${esc(serials.join(", "))}</span>. ` +
          `Clear or change the serial in Config, or name one per run.`
        : `Nothing is attached at all — a run cannot start until it is connected.`);
  } else if (serials.length === 1) {
    cls = "ok";
    what = "attached";
    why = `<span class="mono">${esc(serials[0])}</span>, and no serial is ` +
      `configured, so it is the one that gets used.`;
  } else if (serials.length > 1) {
    cls = "warn";
    what = `${serials.length} attached, none chosen`;
    why = `Set <span class="mono">device.serial</span> in Config, or name one ` +
      `per run — otherwise adb picks and it may not be the one you meant.`;
  } else {
    cls = "bad";
    what = "nothing attached";
    why = "No phone and no serial configured. Nothing can run until one is connected.";
  }
  $("dev-state").innerHTML =
    `<div class="devstate ${cls}"><span class="dot"></span>` +
    `<span class="what">${esc(what)}</span>` +
    `<span class="why">${why}</span></div>`;
}

$("btn-dev-reload").addEventListener("click", () =>
  loadDevices().catch((e) => notice(e.message)));

/* The three read-only device commands the CLI has always had and the browser
   never did. All of them open a `Device` session, which resets animation scales
   and rotation on the way in and out, so the server refuses them while anything
   is driving the phone -- and the button says so rather than the phone changing
   under a run. */

function deviceSerial() {
  return $("opt-serial").value.trim() || $("watch-serial").value.trim();
}

async function withStatus(statusId, label, fn) {
  const el = $(statusId);
  el.textContent = label;
  try {
    await fn();
    el.textContent = "";
  } catch (err) {
    el.textContent = err.message;
  }
}

$("btn-apps-load").addEventListener("click", () =>
  withStatus("apps-status", "listing…", async () => {
    const q = new URLSearchParams({
      search: $("apps-search").value.trim(),
      third_party: $("apps-third-party").checked ? "true" : "false",
      serial: deviceSerial(),
    });
    const d = await api("/api/apps?" + q);
    const out = $("apps-out");
    out.hidden = false;
    out.textContent = d.count
      ? `${d.count} app(s)\n\n` + d.apps.join("\n")
      : "no matching apps";
  }));

$("apps-search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); $("btn-apps-load").click(); }
});

$("btn-dump").addEventListener("click", () =>
  withStatus("dump-status", "dumping…", async () => {
    const q = new URLSearchParams({
      raw: $("dump-raw").checked ? "true" : "false",
      serial: deviceSerial(),
    });
    const d = await api("/api/dump?" + q);
    $("dump-meta").textContent =
      `${d.package || "(no package)"} ${d.activity || ""} · ` +
      `${d.width}x${d.height} · ${d.elements} of ${d.nodes} nodes shown · ` +
      `skeleton ${d.skeleton_id}` + (d.keyboard_open ? " · keyboard open" : "");
    const out = $("dump-out");
    out.hidden = false;
    out.textContent = d.rendered + (d.xml ? "\n\n--- raw xml ---\n" + d.xml : "");
  }));

$("btn-doctor").addEventListener("click", () =>
  withStatus("doctor-status", "checking…", async () => {
    const d = await api("/api/doctor");
    const out = $("doctor-out");
    out.hidden = false;
    out.textContent = d.text || "(no output)";
  }));

/* ------------------------------------------------------------- config */

/* A model field is a dropdown over the provider's own catalogue (/api/models)
   rather than a text box: the ids are long, exact and unguessable, and a typo in
   one only surfaces as a failed call mid-run.

   The catalogue is advice, not law. "custom…" still takes any id, a configured
   model the catalogue does not list is kept and labelled rather than dropped,
   and a provider that cannot be reached at all leaves every field editable --
   a picker that can silently rewrite what is configured is worse than the text
   box it replaced.

   `unset` is what the empty option means for that field. The fallback chain
   lives in config.py, and it is worth saying out loud here, since taking the
   fallback is the whole reason four of these five are empty by default.

   `label` and `help` are the human reading of a key. Sixty-two raw Python
   attribute names in one flat form is a reference manual, not a settings screen:
   `stall_replan_at` and `settle_quiet_s` mean nothing until somebody tells you,
   and the README already did. */
const CFG_SPEC = [
  ["llm", [
    ["provider", "text", { label: "Provider", help: "Which API to talk to." }],
    ["model", "model", { unset: "a run needs one", label: "Model",
      help: "The model that decides what to do next. Everything else falls back to it." }],
    ["model_small", "model", { unset: "falls back to model", label: "Small model",
      help: "For the cheap side calls — verifying, judging, summarising." }],
    ["model_image", "model", { unset: "falls back to model", vision: true,
      label: "Vision model",
      help: "Describes a screenshot when the deciding model cannot see." }],
    ["model_skill", "model", { unset: "falls back to model", label: "Skill model",
      help: "Writes an app's skill from what a run learned." }],
    ["model_skill_image", "model",
      { unset: "falls back to model_image, then model", vision: true,
        label: "Skill vision model",
        help: "Looks at screens while a skill is being written." }],
    ["temperature", "number", { label: "Temperature" }],
    ["max_tokens", "number", { label: "Max output tokens" }],
    ["max_tokens_image", "number", { label: "Max output tokens (vision)" }],
    ["rpm", "number", { label: "Requests per minute",
      help: "Client-side rate limit. 0 is no limit." }],
    ["base_url", "text", { label: "Base URL",
      help: "Override the provider's endpoint." }],
    ["service_tier", "text", { label: "Service tier" }],
    ["api_key", "password", { label: "API key",
      help: "Written to config.json, which is gitignored. Blank keeps what is stored." }],
    ["api_key_env", "text", { label: "API key env var",
      help: "Read from the environment when no key is stored." }],
    ["reasoning_effort", ["", "none", "low", "medium", "high"],
      { label: "Reasoning effort",
        help: "Depth for a routine turn. Empty switches the whole feature off — and it is the single largest lever on latency." }],
    ["reasoning_effort_hard", ["", "none", "low", "medium", "high"],
      { label: "Reasoning effort — hard turns",
        help: "Depth for a turn the loop can see is hard." }],
    ["reasoning_style", ["auto", "effort", "thinking", "off"],
      { label: "Reasoning wire convention",
        help: "How to ask for it. Two incompatible conventions are in use; auto guesses from the model name." }],
    ["vision_in_decider", "bool", { label: "Deciding model can see",
      help: "Set when llm.model itself accepts images: the screenshot then goes straight to the deciding call instead of being described first — one round trip per screenshot turn instead of two." }],
  ]],
  ["device", [
    ["serial", "text", { label: "Device serial",
      help: "Which phone. Blank means whichever single device adb sees." }],
    ["settle_budget_s", "number", { label: "Settle budget (s)",
      help: "Hard ceiling on waiting for one screen to stop moving." }],
    ["settle_quiet_s", "number", { label: "Settle quiet time (s)",
      help: "How long two dumps must agree before the screen counts as settled." }],
    ["disable_animations", "bool", { label: "Turn animations off during a run",
      help: "Restored when the run ends." }],
    ["disable_auto_rotate", "bool", { label: "Lock rotation during a run",
      help: "Restored when the run ends." }],
  ]],
  ["safety", [
    ["budget_usd", "number", { label: "Budget ceiling ($)",
      help: "Session spend ceiling. The run aborts when it is reached." }],
    ["allow_destructive", "bool", { label: "Allow destructive actions",
      help: "Deleting, sending, paying, uninstalling. Off, they are refused." }],
    ["unattended", "bool", { label: "Never prompt",
      help: "Refuse anything that would need a person rather than asking." }],
  ]],
  ["run", [
    ["max_steps", "number", { label: "Max steps",
      help: "How many actions one run may take before it stops." }],
    ["max_wall_clock_s", "number", { label: "Max wall clock (s)" }],
    ["artifacts_dir", "text", { label: "Runs directory",
      help: "Where each run's events, frames and log are written." }],
    ["max_consecutive_failures", "number", { label: "Consecutive failures before giving up",
      help: "Actions that failed in a row. Separate from the stall ladder, which counts actions that worked and got nowhere." }],
    // The stall ladder. A run started from here already obeys these -- the UI
    // spawns the CLI, which reads config.json -- so leaving them off the form
    // meant the only way to see why a run stopped at 14 steps was to open the
    // file by hand. `0` switches a tier off; see `config.RunConfig`.
    ["stall_nudge_at", "number", { label: "Stall: nudge at",
      help: "Steps without learning anything before the model is told so, shown a screenshot and made to think harder. 0 switches the tier off." }],
    ["stall_block_at", "number", { label: "Stall: block at",
      help: "Steps before the harness starts refusing actions already tried twice on this screen." }],
    ["stall_replan_at", "number", { label: "Stall: replan at",
      help: "Steps before one call is spent asking for a different approach." }],
    ["stall_give_up_at", "number", { label: "Stall: give up at",
      help: "Steps before the run stops. The collected data survives." }],
    ["goal_check_every", "number", { label: "Ask if it is already done, every",
      help: "Steps between asking a model whether the goal is already satisfied. 0 switches it off." }],
    ["goal_check_hits", "number", { label: "Satisfied verdicts needed",
      help: "Consecutive satisfied verdicts before the run ends. One sample is how a run that still had work gets cut off." }],
    ["pager_sweep", "bool", { label: "Sweep carousels in code",
      help: "Page through a carousel without a model call per item, once the model has chosen to." }],
    ["pager_sweep_max", "number", { label: "Items per sweep",
      help: "Before control returns to the model." }],
    ["always_screenshot", "bool", { label: "Always pay for vision" }],
    ["never_screenshot", "bool", { label: "Never pay for vision",
      help: "Disables sweeping, which needs to read items." }],
    ["dry_run", "bool", { label: "Dry run",
      help: "Decide everything, touch nothing." }],
  ]],
  ["skills", [
    ["enabled", "bool", { label: "Use app skills",
      help: "Inject per-app guidance into the prompt for the apps the goal is about." }],
    ["skills_dir", "text", { label: "Skills directory" }],
    ["learn_after_run", "bool", { label: "Learn after every run",
      help: "One model call at the end, folding what the run saw into the app's skill." }],
  ]],
  // The Watch tab can override most of these per start, but they belong here
  // too: these are the ceilings, and a ceiling you have to retype on every start
  // is one that will be forgotten once. `policy` and `ledger` especially -- those
  // are paths, and the Watch tab reads them from here rather than asking.
  ["watch", [
    ["policy", "text", { label: "Reply policy file",
      help: "The instructions that decide what gets replied to and what it says. A watch will not start without one. The one the Watch tab opens on, and the one a bare `adbagent watch` uses." }],
    ["policies_dir", "text", { label: "Policies directory",
      help: "Where the other policies live. Every policy in here is offered in the Watch tab's picker, and each carries the goal it was written for." }],
    ["ledger", "text", { label: "Reply ledger file",
      help: "The record the never-double-reply guarantee is built on." }],
    ["interval_s", "number", { label: "Seconds between passes" }],
    ["sweep_s", "number", { label: "Sweep every (s)",
      help: "Run a pass this often even when nothing on screen has changed, for work that does not announce itself. 0 is off, and off only ever spends on a screen that changed." }],
    ["max_steps", "number", { label: "Steps per pass" }],
    ["draft", "bool", { label: "Draft only",
      help: "Compose and record replies, and never send them." }],
    ["fail_closed", "bool", { label: "Fail closed",
      help: "If the ledger cannot be written, do not send." }],
    ["thread_cooldown_s", "number", { label: "Cooldown per conversation (s)" }],
    ["max_replies_per_hour", "number", { label: "Replies per hour" }],
    ["max_replies_per_thread_per_hour", "number",
      { label: "Replies per conversation per hour" }],
    ["max_usd_per_hour", "number", { label: "Spend per hour ($)",
      help: "0 is off." }],
    ["backoff_initial_s", "number", { label: "Backoff, first wait (s)",
      help: "After a pass that changed nothing." }],
    ["backoff_max_s", "number", { label: "Backoff, longest wait (s)" }],
  ]],
  ["memory", [["db", "text", { label: "Memory database",
    help: "What it remembers about screens between runs." }]]],
];

/* The handful anyone actually sets, in the order they get set in. Everything
   else is real and reachable, one fold down, with a search box -- but a form
   that opens on sixty-two fields is a form nobody reads the top of. */
const CFG_TIER1 = [
  "llm.model", "llm.reasoning_effort", "safety.budget_usd", "run.max_steps",
  "device.serial", "safety.allow_destructive", "watch.draft",
  "skills.learn_after_run",
];

/* Section names as a person would say them. */
const CFG_SECTIONS = {
  llm: "Model & API", device: "Phone", safety: "Safety", run: "Run behaviour",
  skills: "App skills", watch: "Watch", memory: "Memory",
};

/* The value of the "custom…" option, chosen so no model id can be it. Not a NUL
   or another control character: the HTML parser rewrites U+0000 in an attribute
   to U+FFFD, and every comparison against it would then quietly fail. */
const MODEL_CUSTOM = "custom…";

let cfgValues = {};
let cfgDefaults = {};
let modelFields = [];  // the model dropdowns now on the form
let catalogue = { models: [], provider: "", error: "", loaded: false };

function modelOptionLabel(m) {
  const caps = [];
  if (m.context_length) caps.push(Math.round(m.context_length / 1024) + "k");
  if (m.vision) caps.push("vision");
  if (m.tools) caps.push("tools");
  return m.id + (caps.length ? `  ·  ${caps.join(" · ")}` : "") +
    (m.deprecated ? `  ·  deprecated ${m.deprecated}` : "");
}

/* Render one model dropdown against whatever catalogue we have: once when the
   form is built -- usually before the fetch lands, so the field shows only its
   own value -- and again when it arrives. */
function fillModelSelect(field) {
  const sel = $(field.id);
  if (!sel) return;
  // What is on screen wins over what is on disk: the catalogue can land after
  // the reader has already picked something.
  const value = sel.options.length ? sel.value : field.value;
  const match = catalogue.models.find((m) => m.value === value || m.id === value);
  const opts = [`<option value="">(unset — ${esc(field.unset)})</option>`];
  if (value && value !== MODEL_CUSTOM && !match) {
    opts.push(`<option value="${esc(value)}">${esc(value)}  ·  not in the catalogue</option>`);
  }
  // A slot that gets handed a screenshot cannot take a text-only model: the
  // whole call fails, not just the image part of it. So say which is which.
  const seeing = catalogue.models.filter((m) => m.vision);
  const blind = catalogue.models.filter((m) => !m.vision);
  const groups = field.vision && seeing.length && blind.length
    ? [["can see", seeing], ["text-only — an image call to one of these fails", blind]]
    : [["", catalogue.models]];
  for (const [name, list] of groups) {
    if (!list.length) continue;
    const body = list.map((m) =>
      `<option value="${esc(m.value)}">${esc(modelOptionLabel(m))}</option>`).join("");
    opts.push(name ? `<optgroup label="${esc(name)}">${body}</optgroup>` : body);
  }
  opts.push(`<option value="${MODEL_CUSTOM}">custom…</option>`);
  sel.innerHTML = opts.join("");
  sel.value = match ? match.value : value;
  $(field.id + "-custom").style.display = sel.value === MODEL_CUSTOM ? "" : "none";
}

async function loadModels(refresh = false) {
  const status = $("cfg-models-status");
  status.textContent = refresh ? "reloading the model catalogue…" : "loading models…";
  try {
    const d = await api("/api/models" + (refresh ? "?refresh=1" : ""));
    catalogue = { models: d.models || [], provider: d.provider || "",
                  error: d.error || "", loaded: true };
  } catch (err) {
    catalogue = { models: [], provider: "", error: err.message, loaded: true };
  }
  const n = catalogue.models.length;
  status.textContent = catalogue.error
    ? `No model catalogue: ${catalogue.error}. Pick “custom…” to type an id.`
    : `${n} model${n === 1 ? "" : "s"} from ${catalogue.provider}.`;
  modelFields.forEach(fillModelSelect);
  syncVisionHint();  // the catalogue can rewrite a short id into its long form
}

/* `vision_in_decider` is one way of saying the deciding model takes images.
   Naming one model for both slots of a pair says it too, and `decider_sees` in
   config.py is what the run actually reads -- so an unticked box beside a
   matching pair reads as a setting that does nothing, when the round trip it
   controls is already being saved. The form says which pair did it. */
function fieldValue(dotted) {
  const id = "cfg-" + dotted.replace(".", "-");
  const el = $(id);
  if (!el) return "";
  const raw = el.value === MODEL_CUSTOM ? ($(id + "-custom") || {}).value || "" : el.value;
  return (raw || "").trim();
}

/* `config.same_model`, in the browser: one id has two accepted forms, and empty
   is never a match -- an unset field is a fallback, not a decision. */
function sameModel(a, b) {
  return !!a && !!b && a.split("/").pop() === b.split("/").pop();
}

function syncVisionHint() {
  const note = $("cfg-llm-vision_in_decider-auto");
  if (!note) return;
  const model = fieldValue("llm.model"), image = fieldValue("llm.model_image");
  const skill = fieldValue("llm.model_skill"), skillImage = fieldValue("llm.model_skill_image");
  const on = [];
  if (sameModel(model, image)) on.push("model and model_image");
  // The pair a skill run resolves to (`skills.use_skill_model`), which is only
  // worth saying when one of the two was set to something of its own.
  if ((skill || skillImage) && sameModel(skill || model, skillImage || image)) {
    on.push("model_skill and model_skill_image");
  }
  note.textContent = on.length ? `· on already: ${on.join(", ")} name one model` : "";
}

/* "custom…" reveals the text box beside its dropdown. Delegated, because the
   form is rebuilt from scratch every time it loads. */
$("cfg-form").addEventListener("input", syncVisionHint);
$("cfg-form").addEventListener("change", (e) => {
  syncVisionHint();
  const field = modelFields.find((f) => f.id === e.target.id);
  if (!field) return;
  const box = $(field.id + "-custom");
  const on = e.target.value === MODEL_CUSTOM;
  box.style.display = on ? "" : "none";
  if (on) box.focus();
});

$("btn-cfg-models").addEventListener("click", () => loadModels(true));

/* One field. Returns the element, so the two tiers can place it themselves. */
function cfgField(section, key, type, opts) {
  const value = (cfgValues[section] || {})[key];
  const shipped = (cfgDefaults[section] || {})[key];
  const label = (opts && opts.label) || key.replace(/_/g, " ");
  const help = (opts && opts.help) || "";
  const inputId = `cfg-${section}-${key}`;
  const wrap = document.createElement("div");
  wrap.className = "cfg-field";
  // Marked when it differs from what ships. Nothing to compare a secret to.
  const changed = type !== "password" && shipped !== undefined
    && JSON.stringify(value) !== JSON.stringify(shipped);
  const over = changed ? `<span class="over" title="default: ${esc(
    shipped === "" ? "(unset)" : String(shipped))}">changed</span>` : "";
  const head = `<span class="lbl">${esc(label)}` +
    `<span class="key">${esc(section)}.${esc(key)}</span>${over}</span>` +
    (help ? `<span class="help">${esc(help)}</span>` : "");

  if (type === "model") {
    wrap.innerHTML = head +
      `<select class="mono" id="${inputId}"></select>` +
      `<input class="mono" type="text" id="${inputId}-custom" spellcheck="false" ` +
      `placeholder="model id" autocomplete="off" style="display:none">`;
    modelFields.push({ id: inputId, value: value ?? "",
                       unset: (opts && opts.unset) || "falls back to model",
                       vision: !!(opts && opts.vision) });
  } else if (type === "bool") {
    wrap.className = "cfg-field check";
    wrap.innerHTML = `<input type="checkbox" id="${inputId}" ${value ? "checked" : ""}>` +
      `<span class="body">${head}` +
      (key === "vision_in_decider"
        ? `<span class="cfg-auto" id="${inputId}-auto"></span>` : "") + `</span>`;
  } else if (Array.isArray(type)) {
    wrap.innerHTML = head + `<select id="${inputId}">` +
      type.map((v) => `<option value="${esc(v)}" ${v === value ? "selected" : ""}>` +
        `${v === "" ? "(unset)" : esc(v)}</option>`).join("") + `</select>`;
  } else if (type === "password") {
    const set = value && value !== "";
    wrap.innerHTML = head + `<input type="password" id="${inputId}" ` +
      `placeholder="${set ? "(set, hidden)" : "(unset)"}" autocomplete="off">`;
  } else {
    const step = Number.isInteger(value) ? "1" : "any";
    wrap.innerHTML = head + `<input type="${type}" id="${inputId}" ` +
      `value="${esc(value ?? "")}" ${type === "number" ? `step="${step}"` : ""}>`;
  }
  wrap.dataset.hay = `${label} ${section}.${key} ${help}`.toLowerCase();
  return wrap;
}

async function loadConfig() {
  const d = await api("/api/config");
  cfgValues = d.config;
  cfgDefaults = d.defaults || {};
  $("cfg-path").textContent = d.path || "(no config file — one will be created on save)";
  const tier1 = $("cfg-essentials");
  const rest = $("cfg-advanced");
  tier1.innerHTML = "";
  rest.innerHTML = "";
  modelFields = [];
  const first = new Map();  // dotted -> element, so tier 1 keeps its own order
  for (const [section, fieldsSpec] of CFG_SPEC) {
    const group = document.createElement("div");
    group.className = "cfg-group";
    group.innerHTML = `<h4>${esc(CFG_SECTIONS[section] || section)} ` +
      `<span class="mono">${esc(section)}</span></h4>`;
    const grid = document.createElement("div");
    grid.className = "cfg-grid";
    let placed = 0;
    for (const [key, type, opts] of fieldsSpec) {
      const el = cfgField(section, key, type, opts);
      if (CFG_TIER1.includes(`${section}.${key}`)) {
        first.set(`${section}.${key}`, el);
      } else {
        grid.appendChild(el);
        placed += 1;
      }
    }
    if (placed) {
      group.appendChild(grid);
      rest.appendChild(group);
    }
  }
  for (const dotted of CFG_TIER1) {
    const el = first.get(dotted);
    if (el) tier1.appendChild(el);
  }
  modelFields.forEach(fillModelSelect);  // with the current values, at least
  syncVisionHint();
  filterConfig();
  cfgDirty = false;                      // the form says what the file says
  if (!catalogue.loaded) loadModels();   // and with the catalogue when it lands
}

/* Sixty-two fields are too many to diff, so the form remembers instead whether
   anything was typed into it since it was last loaded or saved. Live reload
   asks: config.json changing on disk must not silently discard an edit that is
   still on screen. */
let cfgDirty = false;
$("cfg-form").addEventListener("input", () => { cfgDirty = true; });

/* Search over both tiers. Sixty-two settings need a way in that is not
   scrolling, and the names you would search for are the Python ones. */
function filterConfig() {
  const q = $("cfg-search").value.trim().toLowerCase();
  let hits = 0;
  document.querySelectorAll("#cfg-form .cfg-field").forEach((el) => {
    const on = !q || (el.dataset.hay || "").includes(q);
    el.hidden = !on;
    if (on) hits += 1;
  });
  document.querySelectorAll("#cfg-advanced .cfg-group").forEach((g) => {
    g.hidden = !g.querySelector(".cfg-field:not([hidden])");
  });
  $("cfg-search-status").textContent = q
    ? `${hits} setting${hits === 1 ? "" : "s"} match`
    : "";
  if (q) $("cfg-advanced-panel").open = true;
}

$("cfg-search").addEventListener("input", filterConfig);

$("btn-cfg-save").addEventListener("click", async () => {
  const sections = {};
  for (const [section, fieldsSpec] of CFG_SPEC) {
    for (const [key, type] of fieldsSpec) {
      const inputId = `cfg-${section}-${key}`;
      const el = $(inputId);
      const original = (cfgValues[section] || {})[key];
      let value;
      if (type === "bool") {
        value = el.checked;
      } else if (type === "model") {
        value = el.value;
        if (value === MODEL_CUSTOM) {
          value = $(`${inputId}-custom`).value.trim();
          if (value === "") continue;  // "custom…" with nothing typed yet
        }
        // Unlike a plain text field, an emptied model dropdown is a decision --
        // take the fallback -- so it is written rather than skipped.
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
    loadConfig().catch(() => {});   // the "changed" marks move with the save
  } catch (err) {
    notice(err.message);
  }
});

/* ------------------------------------------------------------- skills */

/* The skill the editor is showing, if any. Held so the list can be repainted
   -- by a save, or by the file changing underneath -- without losing which one
   was open. */
let openSkill = "";

function showSkill(full) {
  openSkill = full.name;
  $("skill-editor").hidden = false;
  $("skill-name").textContent = full.name;
  const box = $("skill-json");
  box.value = box._loaded = JSON.stringify(full, null, 2);
}

async function loadSkills() {
  const d = await api("/api/skills");
  const list = $("skills-list");
  list.innerHTML = "";
  if (!d.skills.length) {
    list.innerHTML = `<p class="small">No skills yet. Runs learn them automatically, or generate one.</p>`;
  }
  for (const s of d.skills) {
    const item = document.createElement("div");
    item.className = "skill-item" + (s.name === openSkill ? " active" : "");
    item.innerHTML = `<div class="nm">${esc(s.name)}</div>` +
      `<div class="pkg">${esc(s.packages.join(", ") || "—")}</div>` +
      `<div class="small">${s.workflows} workflows · ${s.nuances} nuances</div>`;
    item.addEventListener("click", async () => {
      document.querySelectorAll(".skill-item").forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      try {
        showSkill(await api("/api/skills/" + encodeURIComponent(s.name)));
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
    $("skill-json")._loaded = $("skill-json").value;  // the file says this now
    loadSkills().catch(() => {});
  } catch (err) {
    notice(err.message);
  }
});

/* A job's output arrives as the last 50 lines, re-sent whole on every poll -- a
   window that slides as the generator talks. Rewriting the box from it would
   slide the text out from under a reader who scrolled up, so keep the log
   append-only instead: stitch each window onto what is already shown, and their
   lines stay where they were. */
function growLog(pre, tail) {
  if (!tail || !tail.length) return;
  const shown = pre._lines || (pre._lines = []);

  /* The longest suffix of the log that the new window repeats is where the two
     agree; everything after it is what the generator has said since. */
  let overlap = 0;
  for (let n = Math.min(shown.length, tail.length); n > 0; n--) {
    let same = true;
    for (let i = 0; i < n && same; i++) same = shown[shown.length - n + i] === tail[i];
    if (same) { overlap = n; break; }
  }
  const fresh = tail.slice(overlap);
  if (!fresh.length) return;
  // No overlap with a non-empty log means the window slid past what we have:
  // more than 50 lines in one poll. Say so rather than splicing them together.
  if (!overlap && shown.length) fresh.unshift("… earlier lines went by unseen …");

  const first = shown.length === 0;
  shown.push(...fresh);
  pre.appendChild(document.createTextNode((first ? "" : "\n") + fresh.join("\n")));
}

function clearLog(pre) {
  pre.hidden = false;
  pre.textContent = "";
  pre._lines = null;
}

/* The generator's own live surface. A tour is a run, so it gets the run's view:
   the same step rows, thinking panels, submitted frames and readouts, off the
   same files. */
const genLive = makeLive("gc-", "gen-live", "gen-feed");
genLive.jobId = 0;
genLive.phone = phoneView($("gen-phone"), { live: true });
genLive.setRunning = (running, stopping) => {
  $("btn-gen").disabled = running;
  $("btn-gen-stop").hidden = !running;
  $("btn-gen-stop").disabled = stopping;
};
genLive.onEnd = () => loadSkills().catch(() => {});
genLive.onEvent = (ev) => {
  // The tour ending is not the job ending: the skill is written up afterwards,
  // from what the tour saw, by one more call in the same process -- and the run
  // directory it would have streamed into is closed by then. So the feed goes
  // quiet here, and only the status says why.
  if (ev.kind === "run_end") $("gen-status").textContent = "writing the skill…";
};

/* Two channels, because they carry different halves of the story. The stream is
   the tour -- every step of it, as it happens. The poll is the child's stdout,
   which is where the part *after* the tour lands: the skill written up from
   what it saw, and the refusals that come before there is any run to stream
   ("no API key", "that app is not installed"). */
function watchGeneration(jobId, { fresh = true } = {}) {
  genLive.jobId = jobId;
  genLive.url = "/api/jobs/" + jobId + "/stream";
  if (fresh) {
    clearLog($("gen-log"));
    $("gen-log-wrap").hidden = false;
    $("gen-status").textContent = "exploring…";
    beginLive(genLive);
  } else {
    // Reattached after a reload: the log's earlier lines are gone with the page.
    $("gen-log-wrap").hidden = false;
    $("gen-status").textContent = "generating…";
    setLiveRunning(genLive, true);
    openStream(genLive);
  }

  const poll = setInterval(async () => {
    try {
      const job = await api("/api/jobs/" + jobId);
      followTail($("gen-log"), () => growLog($("gen-log"), job.output_tail));
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
}

$("btn-gen-stop").addEventListener("click", async () => {
  setStopping(genLive);
  $("gen-status").textContent = "stopping — the tour restores the phone first";
  try {
    await api("/api/jobs/" + genLive.jobId + "/stop", { method: "POST" });
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
  watchGeneration(jobId);
});

/* --------------------------------------------------------- live reload

   `adbagent ui` run from a source checkout watches everything the page is made
   of and streams what changed. Three answers, because the three cost different
   things to apply:

     assets  — the page is stale as a whole: reload it.
     config, skills, policy — one panel is holding an old copy of a file the
       server reads per request: refetch that panel, and only if doing so does
       not take something out from under whoever is typing.
     code    — the server imported it once and cannot re-import it, so it
       restarts. Nothing to do here but say so and wait for it.

   The boot id is how a restart is seen at all. EventSource reconnects on its
   own, so the restart itself is invisible — but the server on the far side of
   it announces a different id, and a page still running the JS a dead process
   served is exactly the page that has to reload. */

let reloadStream = null;
let serverBoot = null;      // the process this page's assets came from
let serverRestarts = false; // whether a code change leads anywhere

function openReloadStream() {
  if (reloadStream) return;
  reloadStream = new EventSource("/api/dev/reload");
  reloadStream.addEventListener("hello", (e) => {
    const d = JSON.parse(e.data);
    if (serverBoot && d.boot !== serverBoot) { location.reload(); return; }
    serverBoot = d.boot;
    serverRestarts = !!d.restarts;
    devbar("");
  });
  reloadStream.addEventListener("reload", (e) => {
    const d = JSON.parse(e.data);
    applyChange(d).catch((err) => devbar(err.message));
  });
}

async function applyChange(d) {
  const names = (d.paths || []).join(", ");
  switch (d.kind) {
    case "assets":
      devbar(`reloading — ${names}`, true);
      location.reload();
      return;
    case "code":
      devbar(serverRestarts ? `${names} — restarting`
                            : `${names} changed; restart the server to pick it up`, true);
      return;
    case "restart":
      // Held because something is still driving the phone. Said with what, so
      // the wait reads as a decision rather than a hang.
      devbar(d.note ? `restart waiting: ${d.note}` : "restarting", true);
      return;
    case "config":  return reloadPane("config", names, refreshConfigFile);
    case "skills":  return reloadPane("skills", names, refreshSkillFiles);
    case "policy":  return refreshPolicyFile(names);
  }
}

/* Safe to overwrite: nothing is being typed into it, and what it holds is what
   the file last said. Anything else is an unsaved edit, and a file that changed
   underneath it is news to report rather than grounds to discard it. */
function beingEdited(el) {
  return !el || document.activeElement === el || el.value !== el._loaded;
}

async function reloadPane(pane, names, refresh) {
  if (!loadedPanes.has(pane)) return;   // never opened; it will load current
  await refresh(names);
}

async function refreshConfigFile(names) {
  if (cfgDirty || $("cfg-form").contains(document.activeElement)) {
    devbar(`${names} changed on disk — your unsaved edits are still here`, true);
    return;
  }
  await loadConfig();
  refreshStatus();                      // the header reads model and serial off it
  devbar(`${names} reloaded`);
}

async function refreshSkillFiles(names) {
  await loadSkills();                   // the list holds no edits, so it always goes
  const box = $("skill-json");
  if (openSkill && !beingEdited(box)) {
    try {
      showSkill(await api("/api/skills/" + encodeURIComponent(openSkill)));
    } catch { /* deleted, or renamed: the list above already says so */ }
  }
  devbar(`${names} reloaded`);
}

async function refreshPolicyFile(names) {
  if (!loadedTabs.has("watch")) return;
  // The list holds no edits, so it always goes: a policy added to the directory
  // in another window should appear in the picker either way.
  await loadPolicies();
  if (beingEdited($("watch-policy"))) {
    devbar(`${names} changed on disk — your unsaved edits are still here`, true);
    return;
  }
  await loadPolicy();
  devbar(`${names} reloaded`);
}

let devbarTimer = null;
function devbar(text, sticky = false) {
  const el = $("devbar");
  if (!el) return;
  clearTimeout(devbarTimer);
  el.textContent = text;
  el.hidden = !text;
  // A state that is waiting for something stays up until it stops waiting.
  // A change already applied is news for a moment.
  if (text && !sticky) devbarTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

/* -------------------------------------------------------- persistence

   The inputs a run or watch was set with are saved to localStorage so fields
   do not have to be retyped across page reloads or browser sessions. */

const PERSIST_FIELDS = [
  // Work surface fields
  { id: "goal", key: "adbagent.goal", type: "text" },
  { id: "opt-max-steps", key: "adbagent.opt-max-steps", type: "text" },
  { id: "opt-budget", key: "adbagent.opt-budget", type: "text" },
  { id: "opt-repeat", key: "adbagent.opt-repeat", type: "text" },
  { id: "opt-serial", key: "adbagent.opt-serial", type: "text" },
  { id: "opt-dry-run", key: "adbagent.opt-dry-run", type: "checkbox" },
  { id: "opt-destructive", key: "adbagent.opt-destructive", type: "checkbox" },
  { id: "opt-no-learn", key: "adbagent.opt-no-learn", type: "checkbox" },
  { id: "opt-assert-shell", key: "adbagent.opt-assert-shell", type: "text" },
  { id: "opt-assert-equals", key: "adbagent.opt-assert-equals", type: "text" },
  { id: "opt-assert-text", key: "adbagent.opt-assert-text", type: "text" },

  // Watch fields
  { id: "watch-goal", key: "adbagent.watch-goal", type: "text" },
  { id: "watch-draft", key: "adbagent.watch-draft", type: "checkbox" },
  { id: "watch-no-learn", key: "adbagent.watch-no-learn", type: "checkbox" },
  { id: "watch-interval", key: "adbagent.watch-interval", type: "text" },
  { id: "watch-sweep", key: "adbagent.watch-sweep", type: "text" },
  { id: "watch-steps", key: "adbagent.watch-steps", type: "text" },
  { id: "watch-serial", key: "adbagent.watch-serial", type: "text" },
  { id: "watch-rph", key: "adbagent.watch-rph", type: "text" },
  { id: "watch-rpc", key: "adbagent.watch-rpc", type: "text" },
  { id: "watch-cooldown", key: "adbagent.watch-cooldown", type: "text" },
  { id: "watch-usd", key: "adbagent.watch-usd", type: "text" },
];

function saveValue(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

function loadValue(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function bindPersistence() {
  for (const f of PERSIST_FIELDS) {
    const el = $(f.id);
    if (!el) continue;

    // Restore saved value
    const saved = loadValue(f.key);
    if (saved !== null) {
      if (f.type === "checkbox") {
        el.checked = saved === "true";
      } else {
        el.value = saved;
      }
    }

    // Save on input/change
    const eventName = f.type === "checkbox" ? "change" : "input";
    el.addEventListener(eventName, () => {
      const val = f.type === "checkbox" ? String(el.checked) : el.value;
      saveValue(f.key, val);
    });
  }
}

// Backwards compatibility functions if referenced anywhere
function saveGoal(key, value) { saveValue(key, value); }
function loadGoal(key) { return loadValue(key) || ""; }

/* -------------------------------------------------------------- boot */

setDensity(loadValue(DENSITY_KEY) || "story");
bindPersistence();
// After the saved values are back in the fields, so the folded line describes
// the options this page actually loaded with rather than the empty form. Open
// it when something is set: a page that comes back with `repeat inf` and a
// destructive tick still on it should not have to be unfolded to find out.
$("runopts").open = paintRunOpts();
$("watchopts").open = paintWatchOpts();
workPhase("compose");
paintWatchBanner(false, false);
refreshStatus();
setInterval(refreshStatus, 5000);
loadRuns().catch(() => {});   // history is on this surface, so it loads with it
loadedTabs.add("work");
