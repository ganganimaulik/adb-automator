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

/* ------------------------------------------------------- follow the tail

   Live surfaces -- the event feed, a streaming llm panel, the generator log --
   chase their newest line only while the reader is at the bottom. Scrolling up
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

/* The event feed rides the page's own scrollbar, which lives on the document. */
function followPageTail(feed, grow) {
  if (!feed.offsetParent) { grow(); return; }  // hidden tab: no tail to chase
  followTail(document.scrollingElement, grow);
}

/* Treat wherever the page sits now as the tail, so the next live line resumes
   the chase: a reader opening the run tab mid-run wants the newest of it. */
function armPageTail() {
  const page = document.scrollingElement;
  page._autoTop = page.scrollTop;
}

/* ---------------------------------------------------------------- tabs */

const loadedTabs = new Set();
const tabLoaders = {
  run: () => {},
  watch: loadWatch,
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
  // The other tab's scroll position isn't a reader's decision about this feed.
  if (name === "run" || name === "skills" || name === "watch") armPageTail();
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

/* A ledger key as a person reads it: `label:today, 9:17 am#2` is the app's own
   caption plus the ledger's bookkeeping, and the prefix is the bookkeeping. */
/* Kinds that are a line of context rather than a card, each mapped to its text.
   A kind in neither this table nor a branch below renders as its bare name --
   which is how a sweep used to read as twenty lines saying "sweep_step" with
   everything the events carried thrown away. */
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
  dead_ends: (e) => `dead ends avoided: ${(e.remembered || []).join(", ")}`,
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

   What the run has *got*: the ledger `scratchpad.py` maintains, shown under the
   step that changed it. The harness sends only the records that were new or
   corrected on that step -- keyed by `id`, the normalised key it matched them
   on -- so the union is kept here per feed and re-rendered whole. A person
   watching a collection run wants the ledger, not the delta: the delta is one
   line and the question is always "has it got everything yet".

   Only the newest panel stays open. Fifteen steps of an album walk would
   otherwise be fifteen copies of a growing list, and the one worth reading is
   always the last -- so each new panel folds the one before it, the way the
   thinking panels fold when their call ends. A panel a reader opened by hand is
   left alone: only the panel this feed opened is tracked. */

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

function finalizeNotes(feed) {
  const card = feed && feed._notesCard;
  if (card) {
    card.open = false;
    feed._notesCard = null;
  }
}

function notesPanel(ev, feed) {
  const ledger = feed ? (feed._notes || (feed._notes = new Map())) : new Map();
  const fresh = new Set();
  for (const rec of ev.records || []) {
    const id = rec.id || rec.key;
    // Set, not delete-then-set: a corrected record keeps the position it was
    // first written in, which is the order the harness holds them in too.
    ledger.set(id, rec);
    fresh.add(id);
  }
  finalizeNotes(feed);  // this panel is the current ledger now; the last is not

  const card = document.createElement("div");
  card.className = "card notes";
  const n = fresh.size;
  card.innerHTML =
    `<details open><summary>collected data · step ${esc(ev.step)} · ` +
    `+${n} ${n === 1 ? "record" : "records"} · ${ledger.size} total</summary>` +
    `<div class="notes-body"></div></details>`;
  const body = card.querySelector(".notes-body");
  for (const [id, rec] of ledger) body.appendChild(noteRow(rec, fresh.has(id)));
  // The harness evicts its oldest records past a ceiling and reports the count
  // rather than hiding it; the view has no such ceiling, so when the two
  // disagree it is the harness that stopped carrying them -- and the prompt the
  // model sees from here on is the shorter one.
  const dropped = ledger.size - (ev.total || ledger.size);
  if (dropped > 0) {
    const note = document.createElement("div");
    note.className = "notes-note";
    note.textContent = `the harness has dropped the oldest ${dropped} for ` +
      `space — still listed here, but no longer in the model's prompt`;
    card.querySelector("details").appendChild(note);
  }
  if (feed) feed._notesCard = card.querySelector("details");
  return card;
}

/* --------------------------------------------------------- step cards */

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
  if (a.confidence) bits.push(esc(a.confidence));
  return bits.join(" · ");
}

/* One feed entry, or null when the event says nothing new.
   `feed` is the element it is going into, which is where the per-feed state a
   few kinds need lives -- see `active_skill`. */
function renderEvent(ev, feed) {
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
    div.className = "card";
    div.innerHTML =
      `<div class="head"><span class="stepno">step ${esc(ev.step)}</span>` +
      `<span class="action">${esc(actionSummary(a))}</span>` +
      // Why the turn was given deep reasoning, on the chip that says it was:
      // `hard_because` is recorded per step and had nowhere to be read.
      (ev.effort ? `<span class="chip neutral"${ev.hard_because
        ? ` title="${esc(ev.hard_because)}"` : ""}>${esc(ev.effort)}</span>` : "") +
      (ev.screenshot ? `<span class="chip neutral">vision</span>` : "") +
      // The stall ladder's own counter. Every action can be succeeding while the
      // run goes nowhere, and this is the number the harness escalates on.
      (ev.stalled ? `<span class="chip stall" title="steps since the run last ` +
        `learned anything — the harness starts intervening as this climbs">` +
        `stalled ${esc(ev.stalled)}</span>` : "") +
      `</div>` +
      (a.observation ? `<div class="obs">${esc(a.observation)}</div>` : "") +
      (a.reasoning ? `<div class="why">${esc(a.reasoning)}</div>` : "") +
      // The model's own plan for the goal, which it rewrites as it goes. It has
      // been in the prompt and in the events all along and was shown nowhere.
      (a.progress ? `<div class="plan">${esc(a.progress)}</div>` : "") +
      `<div class="meta">${stepMeta(ev)}</div>`;
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
    // The answer first, the arithmetic under it. A run that read something and
    // reported it put that in the terminal action's text, and the feed card for
    // that step looks like every other step card -- so the banner is the one
    // place a person scrolling to the bottom is guaranteed to see it.
    div.innerHTML =
      (ev.result ? `<div class="result">${esc(ev.result)}</div>` : "") +
      `<b>${esc((ev.outcome || "").toUpperCase())}</b> — ${esc(ev.steps)} steps, ` +
      `${esc(ev.llm_calls)} LLM calls, $${(ev.usd || 0).toFixed(4)}` +
      (ev.evidence ? `<div class="why">${esc(ev.evidence)}</div>` : "");
  } else if (kind === "active_skill") {
    // Recorded on every step -- it is the per-step record of what the prompt
    // actually carried -- so rendering each one buried the fact under eight
    // identical grey lines. A line per *change* is the information: it is where
    // the run crossed into another app's guidance, or failed to.
    const label = `${ev.name || "?"}|${ev.package || ""}`;
    if (feed) {
      if (feed._skill === label) return null;
      feed._skill = label;
    }
    div.className = "banner skill";
    div.innerHTML = `<span class="chip skill">skill loaded</span> <b>${esc(ev.name || "?")}</b>` +
      (ev.package ? ` <span class="small">${esc(ev.package)}</span>` : "");
  } else if (kind === "item_reading") {
    // The sweep's own vision call. It gets no live panel -- a prefetched read
    // runs on another thread, and streaming two of those into one view
    // interleaves them -- so this card is the whole of it: the line the model
    // read, and the frame it read it from.
    div.className = "card";
    div.innerHTML =
      `<div class="head"><span class="stepno">step ${esc(ev.step)}</span>` +
      `<span class="chip neutral">read</span>` +
      (ev.position ? `<span class="mono">#${esc(ev.position)}</span>` : "") + `</div>` +
      `<div class="obs">${esc(ev.reading || "")}</div>`;
    const thumb = llmShot(ev, feed && feed._runId);
    if (thumb) div.appendChild(thumb);
  } else if (kind === "sweep") {
    div.className = "banner";
    div.innerHTML = `<b>repeated \`${esc(ev.gesture)}\` ${esc(ev.swept)}×</b>, ` +
      `${esc(ev.read)} read <span class="small">· steps ${esc(ev.first_step)}–` +
      `${esc(ev.last_step)}${ev.reason ? " · " + esc(ev.reason) : ""}</span>`;
  } else if (kind === "scratchpad" && (ev.records || []).length) {
    // A run recorded before the event carried its records has only the keys, and
    // falls through to the one-line form in NOTE_LINES below.
    return notesPanel(ev, feed);
  } else if (HALT_BANNERS[kind]) {
    const [cls, html] = HALT_BANNERS[kind](ev);
    div.className = "banner " + cls;
    div.innerHTML = html;
  } else if (NOTE_LINES[kind]) {
    div.className = "small mono";
    div.textContent = NOTE_LINES[kind](ev);
  } else {
    div.className = "small mono";
    div.textContent = kind;
  }
  return div;
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
  if (ev.kind === "decide" && (ev.action || {}).progress) {
    v.progress = ev.action.progress;
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
   away the moment the call ends -- the decide/verify cards carry the result.
   Panels are keyed off order, not step: calls within a step are sequential
   (a vision read, then the decision), so at most one is ever open per feed,
   tracked as feed._llm.

   A call that was shown a screenshot carries its file name, and the thumbnail
   stays visible after the panel folds: a vision read you cannot check against
   the frame it read is only half the record. Which run's files to ask for is
   feed._runId -- the live feed learns it from the stream, the history feed from
   the run it opened. */

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

function openLightbox(src, alt) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = alt || "";
  $("lightbox").style.display = "flex";
}

function closeLightbox() {
  $("lightbox").style.display = "none";
  $("lightbox-img").removeAttribute("src");  // stop holding the decoded frame
}

$("lightbox").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

/* The thumbnail of the frame this call was shown, or nothing when it was shown
   none. Lives outside the <details> so folding the panel keeps it. */
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
   pending layout of a page carrying a run's worth of cards and screenshots.
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
    const card = document.createElement("div");
    card.className = "card llm-live";
    card.innerHTML =
      `<details open><summary><span class="pulse"></span> ${llmSummary(ev)}</summary>` +
      `<div class="llm-sec thinking"><div class="llm-sec-label">thinking</div>` +
      `<div class="llm-text"></div></div>` +
      `<div class="llm-sec response"><div class="llm-sec-label">response</div>` +
      `<div class="llm-text"></div></div></details>`;
    const shot = llmShot(ev, feed._runId);
    if (shot) card.appendChild(shot);
    feed.appendChild(card);
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
      // Replaying a finished run appends hundreds of panels in one go and must
      // not drag the page down with each; a live feed chases its newest line.
      chase: feed._live ? (grow) => followPageTail(feed, grow) : (grow) => grow(),
    };
  } else if (ev.kind === "llm_stream" && feed._llm) {
    llmPush(feed._llm, ev.stream_type, ev.text || "");
  } else if (ev.kind === "llm_end") {
    finalizeLlm(feed, ev);
  }
}

function paintCounters(v) {
  // Against the ceiling when the run recorded one: "step 12/60" says how much
  // room is left, which is what somebody watching actually wants to know.
  v.els.step.textContent = v.maxSteps ? `${v.step}/${v.maxSteps}` : v.step;
  v.els.calls.textContent = v.calls;
  v.els.cost.textContent = "$" + v.cost.toFixed(4) +
    (v.budget ? " / $" + v.budget.toFixed(2) : "");
  v.els.skill.textContent = v.skill || "—";
  v.els.records.textContent = v.records;
  // The model's own progress note, outside the feed on purpose: it is the one
  // line that answers "what does it think it is doing", and the card carrying it
  // is twenty cards back by the time the question comes up.
  v.els.progress.textContent = v.progress;
  v.els.progress.style.display = v.progress ? "" : "none";
  // Only worth the space once there is more than one: a single run has no
  // iteration to speak of. A tour never repeats, so it has no such counter.
  if (v.els.iterWrap) {
    v.els.iterWrap.style.display = v.iteration > 1 ? "" : "none";
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
    pre.style.marginTop = "8px";
    card.appendChild(pre);
    feed._exitLog = pre;
    followPageTail(feed, () => feed.appendChild(card));
  }
  // Two tails to chase: the page's, and the log box's own -- `pre.log` scrolls
  // internally past 400px, and a long write-up would otherwise grow off the
  // bottom of a box that stayed at its first line.
  //
  // Text nodes, not innerHTML: these are the child's own lines, and one of them
  // is a goal or a skill name it read off the phone.
  const pre = feed._exitLog;
  followPageTail(feed, () => followTail(pre, () =>
    pre.appendChild(document.createTextNode(line + "\n"))));
}

/* ------------------------------------------------- a live surface

   The counters strip, the feed under it, and the SSE connection that fills
   them. There are two: the run tab's, and the one under the skill generator --
   because a generation is a run. It drives the phone through the same agent,
   spends from the same budget and writes the same events, screenshots and
   thinking stream, so it is shown with the same cards rather than as the tail
   of a subprocess's stdout.

   `prefix` names the counter ids (`c-step`, `gc-step`), `feedId` the feed. */

function makeLive(prefix, boxId, feedId) {
  const el = (name) => $(prefix + name);
  const feed = $(feedId);
  feed._live = true;  // chase the tail; the history feed does not
  return {
    step: 0, calls: 0, cost: 0, skill: "", iteration: 1,
    records: 0, progress: "", maxSteps: 0, budget: 0,
    source: null, startedAt: 0, timer: null,
    // Stopped, but not over: the child still holds the phone while it restores
    // it and writes up what it learned. Its own state, because showing it as
    // "running" is what made a stop look like it had been ignored.
    stopping: false,
    url: "",           // where to stream from; a job's is known only once it starts
    box: $(boxId), feed,
    els: { runid: el("runid"), step: el("step"), calls: el("calls"),
           cost: el("cost"), elapsed: el("elapsed"), state: el("state"),
           skill: el("skill"), iterWrap: el("iter-wrap"), iter: el("iter"),
           records: el("records"), progress: el("progress") },
    passLabel: "iteration",  // a watch calls its own "pass"
    setRunning: () => {},   // what else on the page follows this surface
    onEvent: () => {},
    onEnd: () => {},
  };
}

/* A surface's counters back to nothing: for a run starting, and for one being
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
  v.box.style.display = "block";
  const source = new EventSource(v.url);
  v.source = source;
  const feed = v.feed;

  source.addEventListener("run", (e) => {
    const data = JSON.parse(e.data);
    // Sent again for every `--repeat` iteration, each of which is a separate
    // run in its own directory. Rule off rather than letting the next one's
    // step 1 land under the last one's step 40.
    if (feed._runId && feed._runId !== data.run_id) {
      finalizeLlm(feed, null);        // an iteration can end mid-call
      const rule = document.createElement("div");
      rule.className = "banner";
      // "pass" for a watch, "iteration" for a --repeat run. The same mechanism
      // underneath -- a new run directory -- but calling a watch's sweep of the
      // inbox an "iteration" reads as though the goal were being retried.
      rule.innerHTML = `<b>${v.passLabel} ${esc(data.iteration || "?")}</b>` +
        `<br><span class="small">${esc(data.run_id)}</span>`;
      followPageTail(feed, () => feed.appendChild(rule));
      // Steps are per iteration, and so are the ledger and the progress note --
      // each iteration pursues the goal from scratch in its own run directory.
      // Calls and spend are the session's, because that is what --budget-usd
      // bounds.
      v.step = 0;
      v.records = 0;
      v.progress = "";
      feed._llm = null;
      feed._skill = "";
      finalizeNotes(feed);   // the last pass's ledger is not this one's
      feed._notes = null;
    }
    v.iteration = data.iteration || 1;
    v.els.runid.textContent = data.run_id;
    feed._runId = data.run_id;        // sent before any llm frame, so the
    paintCounters(v);                 // screenshots have a run to come from
  });
  source.addEventListener("event", (e) => {
    const ev = JSON.parse(e.data);
    const el = renderEvent(ev, feed);
    if (el) followPageTail(feed, () => feed.appendChild(el));
    updateCountersFromEvent(ev, v);
    v.onEvent(ev);
  });
  source.addEventListener("llm", (e) => {
    // Not wrapped in followPageTail: a stream chunk is one of tens of thousands
    // and the panel batches its own appends and scrolling by frame.
    handleLlmEvent(JSON.parse(e.data), feed);
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
    loadedTabs.delete("history");   // a new run may have appeared
    v.onEnd();
  });
  source.onerror = () => {
    // The server closes the connection after "end"; anything else is a drop.
    if (v.source && source.readyState === EventSource.CLOSED) {
      v.source = null;
      setLiveRunning(v, false);
    }
  };
}

/* ------------------------------------------------------------ run tab */

const live = makeLive("c-", "live", "feed");
live.url = "/api/runs/stream";
live.setRunning = (running, stopping) => {
  $("btn-start").disabled = running;
  // One SIGINT is all it takes. A second lands in the shutdown -- outside the
  // handler that catches the first -- and takes down the work it is doing there.
  $("btn-stop").disabled = !running || stopping;
};
// The hint is about something in progress -- stopping, resuming -- and none of
// it is still true once the run is over.
live.onEnd = () => { $("run-hint").textContent = ""; };

function setRunningUI(running) {
  setLiveRunning(live, running);
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

/* Clear a surface and attach it to whatever is starting now. */
function beginLive(v) {
  resetCounters(v);
  v.iteration = 1;
  v.stopping = false;
  v.startedAt = Date.now() / 1000;
  v.feed.innerHTML = "";
  armPageTail();  // a new run is followed however the last one was left
  v.feed._llm = null;
  v.feed._runId = "";
  v.feed._skill = "";
  v.feed._notes = null;
  v.feed._notesCard = null;
  v.feed._exitLog = null;   // cleared with the feed it hung off
  v.els.runid.textContent = "starting…";
  paintCounters(v);
  setLiveRunning(v, true);
  openStream(v);
}

$("run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { goal: $("goal").value.trim(), ...runOptions() };
  if (!body.goal) { notice("a goal is required"); return; }
  saveGoal("adbagent.goal", body.goal);

  try {
    await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    notice(err.message);
    return;
  }
  $("run-hint").textContent = "";
  beginLive(live);
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
  beginLive(live);
  $("run-hint").textContent = `resuming ${id} from its checkpoint`;
}

$("btn-stop").addEventListener("click", async () => {
  setStopping(live);
  $("run-hint").textContent = "stopping — the agent restores the phone first";
  try {
    await api("/api/runs/stop", { method: "POST" });
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
    if (st.run && st.run.running) {
      parts.push(st.run.stopping
        ? `<span class="warn">● stopping the run</span>`
        : `<span class="ok">● running: ${esc(st.run.goal)}</span>`);
    }
    if (st.watch && st.watch.running) {
      // The mode is part of the status line, not just the tab: a watch outlives
      // every reload, and "is it sending?" should be answerable at a glance from
      // any tab. A watch on its way out is neither mode -- it is no longer
      // replying to anything -- so it says that instead.
      parts.push(st.watch.stopping
        ? `<span class="warn">● stopping the watch — writing up what it learned</span>`
        : st.watch.draft
        ? `<span class="ok">● watching (draft): ${esc(st.watch.goal)}</span>`
        : `<span class="warn">● watching LIVE: ${esc(st.watch.goal)}</span>`);
    }
    if (st.job) parts.push(`<span class="ok">● generating a skill</span>`);
    $("status").innerHTML = parts.join(" · ");
    // Reattach to a run already in progress (e.g. page reloaded mid-run).
    if (st.run && st.run.running && !live.source) {
      resetCounters(live);
      live.startedAt = st.run.started_at || Date.now() / 1000;
      live.stopping = !!st.run.stopping;
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
  } catch {
    $("status").textContent = "server unreachable";
  }
}

/* -------------------------------------------------------------- watch */

const watchLive = makeLive("w-", "watch-live", "watch-feed");
watchLive.url = "/api/watch/stream";
watchLive.passLabel = "pass";
watchLive.setRunning = (running, stopping) => {
  $("btn-watch-start").disabled = running;
  // Not clickable twice. The second SIGINT arrives while the first is being
  // acted on -- during the skill write-up, which is not inside the loop's
  // interrupt handler -- and losing that is losing everything the watch learned.
  $("btn-watch-stop").disabled = !running || stopping;
  // The policy is read once, at startup. Editing it under a running watch would
  // take effect at no predictable moment, so the server refuses and the form
  // says so before you type into it.
  $("watch-policy").readOnly = running;
  $("btn-policy-save").disabled = running;
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
    draft: $("watch-draft").checked,
    serial: $("watch-serial").value.trim(),
    interval_s: num("watch-interval"),
    max_steps: num("watch-steps", true),
    replies_per_hour: num("watch-rph", true),
    replies_per_conversation: num("watch-rpc", true),
    cooldown_s: num("watch-cooldown"),
    usd_per_hour: num("watch-usd"),
  };
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
  await loadPolicy();
  await loadLedger();
}

async function loadPolicy() {
  const data = await api("/api/watch/policy");
  $("watch-policy").value = data.text || "";
  $("watch-policy-path").textContent =
    data.path ? data.path + (data.exists ? "" : " (not written yet)")
              : "(no policy path set — set watch.policy in Config)";
}

async function loadLedger() {
  const data = await api("/api/watch/ledger");
  const tbody = document.querySelector("#watch-ledger-table tbody");
  tbody.innerHTML = "";
  const rows = data.threads || [];
  $("watch-ledger-empty").style.display = rows.length ? "none" : "block";
  $("watch-ledger-path").textContent = data.path || "";
  for (const t of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${esc(t.preview || t.thread_key)}</td>` +
      `<td>${t.reply_count}</td>` +
      `<td class="small">${esc(fmtTime(t.last_attempt_at))}</td>` +
      `<td class="small">${t.confirmed
        ? "<span class=\"ok\">confirmed</span>"
        : "<span class=\"warn\">unconfirmed — in doubt</span>"}</td>`;
    tbody.appendChild(tr);
  }
}

$("watch-draft").addEventListener("change", () => paintWatchBanner(false));

$("btn-policy-save").addEventListener("click", async () => {
  try {
    const r = await api("/api/watch/policy", {
      method: "PUT",
      body: JSON.stringify({ text: $("watch-policy").value }),
    });
    notice(`policy saved to ${r.path}`, false);
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
  $("runs-list-view").style.display = "none";
  $("run-detail-view").style.display = "block";
  $("detail-title").textContent = id;
  const feed = $("detail-feed");
  feed.innerHTML = "";
  feed._llm = null;
  feed._runId = id;
  feed._skill = "";
  feed._notes = null;
  feed._notesCard = null;
  try {
    const d = await api("/api/runs/" + encodeURIComponent(id));
    const s = d.summary, st = d.stats, chain = skillChain(d.events);
    const resumeBtn = $("btn-resume-run");
    resumeBtn.style.display = s.resumable ? "" : "none";
    resumeBtn.onclick = () => resumeRun(id);
    $("detail-stats").innerHTML =
      `<h3>${esc(s.goal)}</h3>` +
      // Above the counters, because "what did it conclude" outranks "how many
      // tokens did that take" for everyone who opens a finished run.
      (s.result ? `<div class="banner ${esc(s.outcome)}">` +
        `<div class="result">${esc(s.result)}</div>` +
        (s.evidence ? `<div class="why">${esc(s.evidence)}</div>` : "") +
        `</div>` : "") +
      `<div class="counters">` +
      `<span>outcome <b>${esc(s.outcome)}</b></span>` +
      `<span>steps <b>${esc(s.steps)}</b></span>` +
      `<span>cost <b>$${(s.usd || 0).toFixed(4)}</b></span>` +
      `<span>decisions <b>${st.decisions}</b></span>` +
      `<span>latency/step <b>${st.latency_median_s}s med · ${st.latency_p90_s}s p90</b></span>` +
      `<span>prompt tok <b>${st.prompt_tokens.toLocaleString()}</b></span>` +
      `<span>completion tok <b>${st.completion_tokens.toLocaleString()}</b></span>` +
      (st.sweep_reads ? `<span>sweep reads <b>${st.sweep_reads}</b></span>` : "") +
      (chain.length ? `<span>skills <b>${esc(chain.join(" → "))}</b></span>` : "") +
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
        const el = renderEvent(ev, feed);
        if (el) feed.appendChild(el);
      }
    }
    finalizeLlm(feed, null);  // a run that died mid-call has no llm_end
    // Every ledger panel folded, unlike a live run's last one: this page already
    // opens with the finished ledger in full, above the feed.
    finalizeNotes(feed);
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
    out.style.display = "block";
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
    out.style.display = "block";
    out.textContent = d.rendered + (d.xml ? "\n\n--- raw xml ---\n" + d.xml : "");
  }));

$("btn-doctor").addEventListener("click", () =>
  withStatus("doctor-status", "checking…", async () => {
    const d = await api("/api/doctor");
    const out = $("doctor-out");
    out.style.display = "block";
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
   fallback is the whole reason four of these five are empty by default. */
const CFG_SPEC = [
  ["llm", [
    ["provider", "text"],
    ["model", "model", { unset: "a run needs one" }],
    ["model_small", "model", { unset: "falls back to model" }],
    ["model_image", "model", { unset: "falls back to model", vision: true }],
    ["model_skill", "model", { unset: "falls back to model" }],
    ["model_skill_image", "model",
      { unset: "falls back to model_image, then model", vision: true }],
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
    ["settle_quiet_s", "number"],
    ["disable_animations", "bool"], ["disable_auto_rotate", "bool"],
  ]],
  ["safety", [
    ["budget_usd", "number"], ["allow_destructive", "bool"], ["unattended", "bool"],
  ]],
  ["run", [
    ["max_steps", "number"], ["max_wall_clock_s", "number"], ["artifacts_dir", "text"],
    ["max_consecutive_failures", "number"],
    // The stall ladder. A run started from here already obeys these -- the UI
    // spawns the CLI, which reads config.json -- so leaving them off the form
    // meant the only way to see why a run stopped at 14 steps was to open the
    // file by hand. `0` switches a tier off; see `config.RunConfig`.
    ["stall_nudge_at", "number"], ["stall_block_at", "number"],
    ["stall_replan_at", "number"], ["stall_give_up_at", "number"],
    ["goal_check_every", "number"], ["goal_check_hits", "number"],
    ["pager_sweep", "bool"], ["pager_sweep_max", "number"],
    ["always_screenshot", "bool"], ["never_screenshot", "bool"], ["dry_run", "bool"],
  ]],
  ["skills", [
    ["enabled", "bool"], ["skills_dir", "text"], ["learn_after_run", "bool"],
  ]],
  // The Watch tab can override most of these per start, but they belong here
  // too: these are the ceilings, and a ceiling you have to retype on every start
  // is one that will be forgotten once. `policy` and `ledger` especially -- those
  // are paths, and the Watch tab reads them from here rather than asking.
  ["watch", [
    ["policy", "text"], ["ledger", "text"],
    ["interval_s", "number"], ["max_steps", "number"],
    ["draft", "bool"], ["fail_closed", "bool"],
    ["thread_cooldown_s", "number"],
    ["max_replies_per_hour", "number"],
    ["max_replies_per_thread_per_hour", "number"],
    ["max_usd_per_hour", "number"],
    ["backoff_initial_s", "number"], ["backoff_max_s", "number"],
  ]],
  ["memory", [["db", "text"]]],
];

/* The value of the "custom…" option, chosen so no model id can be it. Not a NUL
   or another control character: the HTML parser rewrites U+0000 in an attribute
   to U+FFFD, and every comparison against it would then quietly fail. */
const MODEL_CUSTOM = "custom…";

let cfgValues = {};
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

async function loadConfig() {
  const d = await api("/api/config");
  cfgValues = d.config;
  $("cfg-path").textContent = d.path || "(no config file — one will be created on save)";
  const form = $("cfg-form");
  form.innerHTML = "";
  modelFields = [];
  for (const [section, fieldsSpec] of CFG_SPEC) {
    const group = document.createElement("div");
    group.className = "cfg-group";
    group.innerHTML = `<h4>${esc(section)}</h4>`;
    const grid = document.createElement("div");
    grid.className = "cfg-grid";
    for (const [key, type, opts] of fieldsSpec) {
      const value = (cfgValues[section] || {})[key];
      const label = document.createElement("label");
      const inputId = `cfg-${section}-${key}`;
      if (type === "model") {
        label.innerHTML = `${esc(key)}<select class="mono" id="${inputId}"></select>` +
          `<input class="mono" type="text" id="${inputId}-custom" spellcheck="false" ` +
          `placeholder="model id" autocomplete="off" style="display:none">`;
        modelFields.push({ id: inputId, value: value ?? "",
                           unset: (opts && opts.unset) || "falls back to model",
                           vision: !!(opts && opts.vision) });
      } else if (type === "bool") {
        label.className = "check";
        label.innerHTML = `<input type="checkbox" id="${inputId}" ${value ? "checked" : ""}> ${esc(key)}` +
          (key === "vision_in_decider" ? `<span class="cfg-auto" id="${inputId}-auto"></span>` : "");
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
  modelFields.forEach(fillModelSelect);  // with the current values, at least
  syncVisionHint();
  if (!catalogue.loaded) loadModels();   // and with the catalogue when it lands
}

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
  pre.style.display = "block";
  pre.textContent = "";
  pre._lines = null;
}

/* The generator's own live surface. A tour is a run, so it gets the run's view:
   the same step cards, thinking panels, submitted frames and counters, off the
   same files. */
const genLive = makeLive("gc-", "gen-live", "gen-feed");
genLive.jobId = 0;
genLive.setRunning = (running, stopping) => {
  $("btn-gen").disabled = running;
  $("btn-gen-stop").disabled = !running || stopping;
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
    $("gen-log-wrap").style.display = "block";
    $("gen-status").textContent = "exploring…";
    beginLive(genLive);
  } else {
    // Reattached after a reload: the log's earlier lines are gone with the page.
    $("gen-log-wrap").style.display = "block";
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

/* -------------------------------------------------------- persistence

   The inputs a run or watch was set with are saved to localStorage so fields
   do not have to be retyped across page reloads or browser sessions. */

const PERSIST_FIELDS = [
  // Run page fields
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

  // Watch page fields
  { id: "watch-goal", key: "adbagent.watch-goal", type: "text" },
  { id: "watch-draft", key: "adbagent.watch-draft", type: "checkbox" },
  { id: "watch-interval", key: "adbagent.watch-interval", type: "text" },
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

bindPersistence();
refreshStatus();
setInterval(refreshStatus, 5000);
loadRuns().catch(() => {});  // warm the history cache for later
loadedTabs.add("run");

