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
  if (name === "run" || name === "skills") armPageTail();
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
  v.els.step.textContent = v.step;
  v.els.calls.textContent = v.calls;
  v.els.cost.textContent = "$" + v.cost.toFixed(4);
  v.els.skill.textContent = v.skill || "—";
  // Only worth the space once there is more than one: a single run has no
  // iteration to speak of. A tour never repeats, so it has no such counter.
  if (v.els.iterWrap) {
    v.els.iterWrap.style.display = v.iteration > 1 ? "" : "none";
    v.els.iter.textContent = v.iteration;
  }
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
    source: null, startedAt: 0, timer: null,
    url: "",           // where to stream from; a job's is known only once it starts
    box: $(boxId), feed,
    els: { runid: el("runid"), step: el("step"), calls: el("calls"),
           cost: el("cost"), elapsed: el("elapsed"), state: el("state"),
           skill: el("skill"), iterWrap: el("iter-wrap"), iter: el("iter") },
    setRunning: () => {},   // what else on the page follows this surface
    onEvent: () => {},
    onEnd: () => {},
  };
}

/* Keep a surface's clock and status telling the truth. */
function setLiveRunning(v, running) {
  v.els.state.textContent = running ? "running" : "idle";
  v.els.state.style.color = running ? "var(--green)" : "var(--text-dim)";
  clearInterval(v.timer);
  if (running) {
    v.timer = setInterval(() => {
      v.els.elapsed.textContent = fmtDur((Date.now() / 1000) - v.startedAt);
    }, 1000);
  }
  v.setRunning(running);
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
      rule.innerHTML = `<b>iteration ${esc(data.iteration || "?")}</b>` +
        `<br><span class="small">${esc(data.run_id)}</span>`;
      followPageTail(feed, () => feed.appendChild(rule));
      // Steps are per iteration; calls and spend are the session's, because
      // that is what --budget-usd bounds.
      v.step = 0;
      feed._llm = null;
      feed._skill = "";
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
  source.addEventListener("state", (e) => {
    const st = JSON.parse(e.data);
    if (st.started_at) v.startedAt = st.started_at;
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
live.setRunning = (running) => {
  $("btn-start").disabled = running;
  $("btn-stop").disabled = !running;
};

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
  };
  const ms = parseInt($("opt-max-steps").value, 10);
  if (ms) body.max_steps = ms;
  const budget = parseFloat($("opt-budget").value);
  if (!isNaN(budget)) body.budget_usd = budget;
  return body;
}

/* Clear a surface and attach it to whatever is starting now. */
function beginLive(v) {
  v.step = 0; v.calls = 0; v.cost = 0; v.skill = "";
  v.iteration = 1;
  v.startedAt = Date.now() / 1000;
  v.feed.innerHTML = "";
  armPageTail();  // a new run is followed however the last one was left
  v.feed._llm = null;
  v.feed._runId = "";
  v.feed._skill = "";
  v.els.runid.textContent = "starting…";
  paintCounters(v);
  setLiveRunning(v, true);
  openStream(v);
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
    if (st.job) parts.push(`<span class="ok">● generating a skill</span>`);
    $("status").innerHTML = parts.join(" · ");
    // Reattach to a run already in progress (e.g. page reloaded mid-run).
    if (st.run && st.run.running && !live.source) {
      live.step = 0; live.calls = 0; live.cost = 0;
      live.startedAt = st.run.started_at || Date.now() / 1000;
      setRunningUI(true);
      openStream(live);
    }
    // And to a generation, which outlives a reload just as long.
    if (st.job && !genLive.source) {
      genLive.step = 0; genLive.calls = 0; genLive.cost = 0;
      genLive.startedAt = st.job.started_at || Date.now() / 1000;
      watchGeneration(st.job.id, { fresh: false });
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
  try {
    const d = await api("/api/runs/" + encodeURIComponent(id));
    const s = d.summary, st = d.stats, chain = skillChain(d.events);
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
}

/* "custom…" reveals the text box beside its dropdown. Delegated, because the
   form is rebuilt from scratch every time it loads. */
$("cfg-form").addEventListener("change", (e) => {
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
  modelFields.forEach(fillModelSelect);  // with the current values, at least
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
genLive.setRunning = (running) => {
  $("btn-gen").disabled = running;
  $("btn-gen-stop").disabled = !running;
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
  try {
    await api("/api/jobs/" + genLive.jobId + "/stop", { method: "POST" });
    $("gen-status").textContent = "stopping — the tour restores the phone first";
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

/* -------------------------------------------------------------- boot */

refreshStatus();
setInterval(refreshStatus, 5000);
loadRuns().catch(() => {});  // warm the history cache for later
loadedTabs.add("run");
