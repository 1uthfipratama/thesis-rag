// thesis-rag frontend: one question at a time, streamed over SSE (POST, so fetch
// + a small parser instead of EventSource). No localStorage; the access code is
// kept in sessionStorage only (PLAN_ADDENDUM 14.1).
"use strict";

const $ = (id) => document.getElementById(id);
const form = $("ask"), question = $("question"), submit = $("submit");
const codeRow = $("code-row"), codeInput = $("code"), formMsg = $("form-msg");
const result = $("result"), statusEl = $("status"), answerEl = $("answer");
const sourcesWrap = $("sources-wrap"), sourcesEl = $("sources");
const REFUSAL = "The corpus doesn't cover this.";
const CODE_KEY = "thesis-rag-code";

let controller = null;
let passages = new Map(); // n -> source card from the "sources" event

// --- storage (may be blocked; never required) ---------------------------------
function getCode() { try { return sessionStorage.getItem(CODE_KEY) || ""; } catch { return ""; } }
function setCode(v) { try { v ? sessionStorage.setItem(CODE_KEY, v) : sessionStorage.removeItem(CODE_KEY); } catch {} }

// --- question box -----------------------------------------------------------------
function autosize() {
  question.style.height = "auto";
  question.style.height = question.scrollHeight + "px";
}
question.addEventListener("input", autosize);
// Re-fit when the width changes (rotation, narrow screens) or web fonts arrive:
// a long question otherwise stays clipped at its old height.
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
  if (!q) return question.focus();
  if (!codeRow.hidden && codeInput.value.trim()) setCode(codeInput.value.trim());
  ask(q);
});

function showMessage(text) {
  formMsg.textContent = text;
  formMsg.hidden = !text;
}

// --- asking -----------------------------------------------------------------------
async function ask(q) {
  if (controller) controller.abort();
  controller = new AbortController();
  showMessage("");
  document.body.classList.add("asked");
  result.hidden = false;
  sourcesWrap.hidden = true;
  sourcesEl.replaceChildren();
  answerEl.className = "answer streaming";
  answerEl.replaceChildren();
  answerEl.setAttribute("aria-busy", "true");
  statusEl.textContent = "Searching the papers…";
  submit.disabled = true;
  passages = new Map();

  let res;
  try {
    const headers = { "content-type": "application/json" };
    const code = getCode();
    if (code) headers["x-access-code"] = code;
    res = await fetch("api/ask", {
      method: "POST", headers, body: JSON.stringify({ question: q }), signal: controller.signal,
    });
  } catch (err) {
    if (err.name !== "AbortError") fail("Couldn't reach the server. Check your connection and try again.");
    return;
  }

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) {
      setCode("");
      codeRow.hidden = false;
      codeInput.focus();
      result.hidden = true;
      submit.disabled = false;
      showMessage(getCode() || codeInput.value ? data.error || "That access code isn't right." : "This demo needs an access code.");
      return;
    }
    return fail(data.error || "Something went wrong. Please try again.");
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
        if (ev.event === "sources") onSources(ev.data);
        else if (ev.event === "token") { text += ev.data; renderAnswer(text, false); }
        else if (ev.event === "done") onDone(ev.data);
        else if (ev.event === "error") fail(ev.data.message);
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") fail("The answer was interrupted. Please try again.");
  } finally {
    submit.disabled = false;
    answerEl.removeAttribute("aria-busy");
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

function onSources(list) {
  list.forEach((s) => passages.set(s.n, s));
  const cites = [...new Set(list.map((s) => s.short_cite))];
  const names = cites.length > 3 ? `${cites.slice(0, 3).join(", ")} and ${cites.length - 3} more` : cites.join(", ");
  statusEl.textContent = list.length ? `Reading ${list.length} passages from ${names}` : "No matching passages found";
}

function onDone(data) {
  answerEl.classList.remove("streaming");
  const refused = data.refused || data.text.trim().startsWith(REFUSAL);
  renderAnswer(data.text, true);
  if (refused) {
    answerEl.classList.add("refusal");
    statusEl.textContent = "";
    return;
  }
  const cited = data.cited.filter((n) => passages.has(n));
  statusEl.textContent = cited.length ? `${(data.latency_ms / 1000).toFixed(1)} s` : "";
  renderSources(cited);
}

function fail(message) {
  answerEl.classList.remove("streaming");
  answerEl.className = "answer refusal";
  answerEl.textContent = message;
  statusEl.textContent = "";
  submit.disabled = false;
}

// --- rendering --------------------------------------------------------------------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Paragraphs, "- " lists, and [n] markers -> superscript links. Markers are only
// linked when complete, so a half-streamed "[1" stays plain text.
function toHtml(text) {
  const blocks = text.trim().split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-•*]\s+/.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*[-•*]\s+/, ""))}</li>`).join("") + "</ul>";
    }
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  }).join("");
}

function inline(s) {
  return escapeHtml(s).replace(/\[(\d{1,2})\]/g, (m, n) =>
    passages.has(Number(n))
      ? `<sup class="cite"><a href="#src-${n}" data-n="${n}" aria-label="source ${n}">${n}</a></sup>`
      : m
  );
}

function renderAnswer(text, final) {
  answerEl.innerHTML = toHtml(text);
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

function renderSources(cited) {
  sourcesEl.replaceChildren(...cited.map((n) => sourceItem(passages.get(n))));
  sourcesWrap.hidden = cited.length === 0;
}

function sourceItem(s) {
  const li = document.createElement("li");
  li.id = `src-${s.n}`;
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
  body.id = `src-body-${s.n}`;
  btn.setAttribute("aria-controls", body.id);
  const doi = s.doi ? `<a href="https://doi.org/${encodeURI(s.doi)}" target="_blank" rel="noopener">doi.org/${escapeHtml(s.doi)}</a>` : "No DOI (venue issues none)";
  body.innerHTML = `<q>${escapeHtml(s.snippet)}</q>${doi}`;
  btn.addEventListener("click", () => toggle(li, btn, body));
  li.append(btn, body);
  return li;
}

function toggle(li, btn, body, open) {
  const show = open ?? body.hidden;
  body.hidden = !show;
  btn.setAttribute("aria-expanded", String(show));
}

// Superscript citation -> scroll to its source and open it.
answerEl.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-n]");
  if (!a) return;
  e.preventDefault();
  const li = $(`src-${a.dataset.n}`);
  if (!li) return;
  toggle(li, li.querySelector(".src-toggle"), li.querySelector(".src-body"), true);
  li.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  li.classList.add("flash");
  setTimeout(() => li.classList.remove("flash"), 1200);
  li.querySelector(".src-toggle").focus({ preventScroll: true });
});

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

$("home").addEventListener("click", (e) => {
  e.preventDefault();
  if (controller) controller.abort();
  document.body.classList.remove("asked");
  result.hidden = true;
  question.value = "";
  autosize();
  question.focus();
});

question.focus();
