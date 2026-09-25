// thesis-rag frontend: a chat over the papers, streamed over SSE (POST, so fetch
// + a small parser instead of EventSource). The conversation and access code live
// in sessionStorage only: this tab, gone when it closes (PLAN_ADDENDUM 14.1).
// The server is stateless; each message sends the recent turns back.
"use strict";

const $ = (id) => document.getElementById(id);
const form = $("ask"), question = $("question"), submit = $("submit");
const codeRow = $("code-row"), codeInput = $("code"), formMsg = $("form-msg");
const thread = $("thread"), newChat = $("new-chat");
const REFUSAL = "The corpus doesn't cover this.";
const CODE_KEY = "thesis-rag-code", CHAT_KEY = "thesis-rag-chat";
const HISTORY_TURNS = 4; // exchanges sent back as context (the server trims too)
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

let controller = null;
// {q, text, cited: [card], passage_ids, refused, error, latency_ms}
let turns = [];

// --- storage (may be blocked; never required) ---------------------------------
function load(key) { try { return sessionStorage.getItem(key) || ""; } catch { return ""; } }
function save(key, v) { try { v ? sessionStorage.setItem(key, v) : sessionStorage.removeItem(key); } catch {} }
const getCode = () => load(CODE_KEY);
const setCode = (v) => save(CODE_KEY, v);
function saveChat() { save(CHAT_KEY, turns.length ? JSON.stringify(turns) : ""); }

// --- composer ---------------------------------------------------------------------
function autosize() {
  question.style.height = "auto";
  question.style.height = question.scrollHeight + "px";
}
question.addEventListener("input", autosize);
// Re-fit when the width changes (rotation, narrow screens) or web fonts arrive:
// a long message otherwise stays clipped at its old height.
new ResizeObserver(autosize).observe(question);
document.fonts?.ready.then(autosize);
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  } else if (e.key === "Escape") {
    question.value = "";
    autosize();
  }
});
document.querySelectorAll("#examples a[data-q]").forEach((a) =>
  a.addEventListener("click", (e) => {
    e.preventDefault();
    question.value = a.dataset.q;
    autosize();
    form.requestSubmit();
  })
);
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = question.value.trim();
  if (!q || submit.disabled) return question.focus();
  if (!codeRow.hidden && codeInput.value.trim()) setCode(codeInput.value.trim());
  send(q);
});

function showMessage(text) {
  formMsg.textContent = text;
  formMsg.hidden = !text;
}

function setMode() {
  const chatting = turns.length > 0;
  document.body.classList.toggle("asked", chatting);
  thread.hidden = !chatting;
  newChat.hidden = !chatting;
  question.placeholder = chatting
    ? "Ask a follow-up, or say “explain that more simply”"
    : "Ask a question about the papers";
}

// --- sending ----------------------------------------------------------------------
function historyFor() {
  const done = turns.filter((t) => t.text && !t.error).slice(-HISTORY_TURNS);
  return done.flatMap((t) => [
    { role: "user", content: t.q },
    { role: "assistant", content: t.text.slice(0, 4000) },
  ]);
}

async function send(q) {
  if (controller) controller.abort();
  controller = new AbortController();
  showMessage("");
  const history = historyFor();
  const last = turns.filter((t) => t.text && !t.error).at(-1);
  const turn = { q, text: "", cited: [], passage_ids: [], refused: false, error: "" };
  turns.push(turn);
  const i = turns.length - 1;
  setMode();
  const el = turnEl(turn, i);
  thread.append(el);
  const ui = parts(el);
  ui.answer.classList.add("streaming");
  ui.answer.setAttribute("aria-busy", "true");
  ui.status.textContent = history.length ? "Thinking about the conversation…" : "Searching the papers…";
  el.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
  question.value = "";
  autosize();
  submit.disabled = true;

  let passages = new Map(); // n -> source card from the "sources" event
  let res;
  try {
    const headers = { "content-type": "application/json" };
    const code = getCode();
    if (code) headers["x-access-code"] = code;
    res = await fetch("api/chat", {
      method: "POST",
      headers,
      body: JSON.stringify({ message: q, history, reuse_ids: last ? last.passage_ids : [] }),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name !== "AbortError") fail(turn, el, "Couldn't reach the server. Check your connection and try again.");
    return;
  }

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) {
      // Not a real turn: take it back and put the message in the box again.
      turns.pop();
      el.remove();
      setMode();
      question.value = q;
      autosize();
      const hadCode = getCode() || codeInput.value;
      setCode("");
      codeRow.hidden = false;
      codeInput.focus();
      submit.disabled = false;
      showMessage(hadCode ? data.error || "That access code isn't right." : "This demo needs an access code.");
      return;
    }
    return fail(turn, el, data.error || "Something went wrong. Please try again.");
  }
  codeRow.hidden = true;

  let buffer = "", text = "";
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let cut;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const ev = parseEvent(block);
        if (!ev) continue;
        if (ev.event === "sources") {
          passages = new Map(ev.data.map((s) => [s.n, s]));
          ui.status.textContent = readingLine(ev.data);
        } else if (ev.event === "token") {
          text += ev.data;
          renderAnswer(ui.answer, text, i, passages, false);
        } else if (ev.event === "done") {
          onDone(turn, el, i, ev.data, passages);
        } else if (ev.event === "error") {
          fail(turn, el, ev.data.message);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") fail(turn, el, "The answer was interrupted. Please try again.");
  } finally {
    if (turns[i] === turn) {
      submit.disabled = false;
      ui.answer.removeAttribute("aria-busy");
      ui.answer.classList.remove("streaming");
      if (!turn.text && !turn.error) fail(turn, el, "The answer was interrupted. Please try again.");
    }
  }
}

function parseEvent(block) {
  let event = "message", data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).replace(/^ /, "");
  }
  if (!data) return null;
  try { return { event, data: JSON.parse(data) }; } catch { return null; }
}

function readingLine(list) {
  if (!list.length) return "No matching passages found";
  const cites = [...new Set(list.map((s) => s.short_cite))];
  const names = cites.length > 3 ? `${cites.slice(0, 3).join(", ")} and ${cites.length - 3} more` : cites.join(", ");
  return `Reading ${list.length} passages from ${names}`;
}

function onDone(turn, el, i, data, passages) {
  turn.text = data.text;
  turn.refused = data.refused || data.text.trim().startsWith(REFUSAL);
  turn.cited = data.cited.filter((n) => passages.has(n)).map((n) => passages.get(n));
  turn.passage_ids = data.passage_ids || [];
  turn.latency_ms = data.latency_ms;
  saveChat();
  fillTurn(el, turn, i);
}

function fail(turn, el, message) {
  turn.error = message;
  turn.text = "";
  saveChat();
  fillTurn(el, turn, turns.indexOf(turn));
  submit.disabled = false;
}

// --- rendering --------------------------------------------------------------------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Paragraphs, "- " lists, **bold**, and [n] markers -> superscript links. Markers
// are linked only when complete, so a half-streamed "[1" stays plain text.
function toHtml(text, i, cards) {
  const blocks = text.trim().split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-•*]\s+/.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*[-•*]\s+/, ""), i, cards)}</li>`).join("") + "</ul>";
    }
    return `<p>${lines.map((l) => inline(l, i, cards)).join("<br>")}</p>`;
  }).join("");
}

function inline(s, i, cards) {
  return escapeHtml(s)
    .replace(/\*\*(\S(?:[^*]*?\S)?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d{1,2})\]/g, (m, n) =>
      cards.has(Number(n))
        ? `<sup class="cite"><a href="#src-${i}-${n}" data-turn="${i}" data-n="${n}" aria-label="source ${n}">${n}</a></sup>`
        : m
    );
}

function renderAnswer(answerEl, text, i, cards, final) {
  answerEl.innerHTML = toHtml(text, i, cards);
  if (final && window.renderMathInElement) {
    // Equations are quoted from the papers as LaTeX; typeset them once complete.
    window.renderMathInElement(answerEl, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}

function turnEl(turn, i) {
  const el = document.createElement("article");
  el.className = "turn";
  el.innerHTML =
    `<p class="you"><span class="visually-hidden">You: </span></p>` +
    `<p class="status" aria-live="polite"></p>` +
    `<div class="answer" aria-live="off"></div>` +
    `<section class="sources-wrap" hidden><h2 class="sources-title">Sources</h2><ol class="sources"></ol></section>`;
  el.querySelector(".you").append(turn.q);
  return el;
}

function parts(el) {
  return {
    status: el.querySelector(".status"),
    answer: el.querySelector(".answer"),
    sourcesWrap: el.querySelector(".sources-wrap"),
    sources: el.querySelector(".sources"),
  };
}

function fillTurn(el, turn, i) {
  const ui = parts(el);
  ui.answer.classList.remove("streaming", "refusal");
  ui.answer.removeAttribute("aria-busy");
  if (turn.error) {
    ui.answer.classList.add("refusal");
    ui.answer.textContent = turn.error;
    ui.status.textContent = "";
    ui.sourcesWrap.hidden = true;
    return;
  }
  const cards = new Map(turn.cited.map((s) => [s.n, s]));
  renderAnswer(ui.answer, turn.text, i, cards, true);
  if (turn.refused) ui.answer.classList.add("refusal");
  ui.status.textContent = "";
  ui.sources.replaceChildren(...turn.cited.map((s) => sourceItem(s, i)));
  ui.sourcesWrap.hidden = turn.cited.length === 0;
}

function sourceItem(s, i) {
  const li = document.createElement("li");
  li.id = `src-${i}-${s.n}`;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "src-toggle";
  btn.setAttribute("aria-expanded", "false");
  const where = [s.short_cite, s.heading && s.heading !== "Paper overview" ? s.heading : "Overview", `p.${s.page_start}`];
  btn.innerHTML =
    `<span class="src-n">[${s.n}]</span>` +
    `<span class="src-line">${escapeHtml(where.join(" · "))}</span>` +
    `<span class="src-id">${escapeHtml(s.paper_id)}</span>`;
  const body = document.createElement("div");
  body.className = "src-body";
  body.hidden = true;
  body.id = `src-body-${i}-${s.n}`;
  btn.setAttribute("aria-controls", body.id);
  const doi = s.doi ? `<a href="https://doi.org/${encodeURI(s.doi)}" target="_blank" rel="noopener">doi.org/${escapeHtml(s.doi)}</a>` : "No DOI (venue issues none)";
  body.innerHTML = `<q>${escapeHtml(s.snippet)}</q>${doi}`;
  btn.addEventListener("click", () => toggle(btn, body));
  li.append(btn, body);
  return li;
}

function toggle(btn, body, open) {
  const show = open ?? body.hidden;
  body.hidden = !show;
  btn.setAttribute("aria-expanded", String(show));
}

// Superscript citation -> scroll to its source in the same turn and open it.
thread.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-n]");
  if (!a) return;
  e.preventDefault();
  const li = $(`src-${a.dataset.turn}-${a.dataset.n}`);
  if (!li) return;
  toggle(li.querySelector(".src-toggle"), li.querySelector(".src-body"), true);
  li.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "center" });
  li.classList.add("flash");
  setTimeout(() => li.classList.remove("flash"), 1200);
  li.querySelector(".src-toggle").focus({ preventScroll: true });
});

// --- new chat / restore -------------------------------------------------------------
function reset(e) {
  e?.preventDefault();
  if (controller) controller.abort();
  controller = null;
  turns = [];
  saveChat();
  thread.replaceChildren();
  submit.disabled = false;
  showMessage("");
  setMode();
  question.value = "";
  autosize();
  question.focus();
}
newChat.addEventListener("click", reset);
$("home").addEventListener("click", reset);

function restore() {
  try { turns = JSON.parse(load(CHAT_KEY) || "[]"); } catch { turns = []; }
  // A turn cut off by a reload has no answer: drop it rather than show it empty.
  turns = turns.filter((t) => t && t.q && (t.text || t.error));
  turns.forEach((t, i) => {
    const el = turnEl(t, i);
    thread.append(el);
    fillTurn(el, t, i);
  });
  setMode();
  if (turns.length) window.scrollTo(0, document.body.scrollHeight);
}

// --- about ------------------------------------------------------------------------
const aboutBtn = $("about-toggle"), about = $("about");
let papersLoaded = false;
aboutBtn.addEventListener("click", async () => {
  about.hidden = !about.hidden;
  aboutBtn.setAttribute("aria-expanded", String(!about.hidden));
  if (about.hidden || papersLoaded) return;
  papersLoaded = true;
  try {
    const papers = await (await fetch("api/papers")).json();
    $("papers").innerHTML = papers.map((p) =>
      `<li><span class="pid">${escapeHtml(p.id)}</span>${escapeHtml(p.short_cite)}` +
      `<span class="ptitle">${escapeHtml(p.title)}${p.doi ? ` · <a href="https://doi.org/${encodeURI(p.doi)}" target="_blank" rel="noopener">DOI</a>` : ""}</span></li>`
    ).join("");
  } catch {
    papersLoaded = false;
    $("papers").innerHTML = "<li>Couldn't load the paper list.</li>";
  }
});

// KaTeX loads deferred before this script runs (both `defer`, in order).
restore();
question.focus();
