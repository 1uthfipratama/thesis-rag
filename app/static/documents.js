// thesis-rag documents page: upload a PDF, watch it being indexed, remove uploads,
// and list the 19 core papers. Shares the access code and search scope with the
// chat page through sessionStorage (same keys as app.js).
"use strict";

const $ = (id) => document.getElementById(id);
const CODE_KEY = "thesis-rag-code", SCOPE_KEY = "thesis-rag-scope";
const codeRow = $("code-row"), codeInput = $("code"), fileInput = $("file"), drop = $("drop");
const status = $("upload-status"), docsEl = $("docs"), clearDocs = $("clear-docs");

function load(key) { try { return sessionStorage.getItem(key) || ""; } catch { return ""; } }
function save(key, v) { try { v ? sessionStorage.setItem(key, v) : sessionStorage.removeItem(key); } catch {} }

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function codeHeaders() {
  const code = codeInput.value.trim() || load(CODE_KEY);
  return code ? { "x-access-code": code } : {};
}

function needCode(error) {
  const hadCode = load(CODE_KEY) || codeInput.value;
  save(CODE_KEY, "");
  codeRow.hidden = false;
  codeInput.focus();
  status.textContent = hadCode ? error || "That access code isn't right." : "This app needs an access code. Enter it above and try again.";
}

// --- upload ---------------------------------------------------------------------------
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = "";
});
let dragDepth = 0;
document.addEventListener("dragenter", (e) => {
  if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
  dragDepth++;
  drop.classList.add("over");
});
document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; drop.classList.remove("over"); } });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  drop.classList.remove("over");
  const f = e.dataTransfer?.files?.[0];
  if (f) uploadFile(f);
});

async function uploadFile(file) {
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
    status.textContent = "That file isn't a PDF.";
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    status.textContent = "PDFs up to 20 MB only.";
    return;
  }
  if (codeInput.value.trim()) save(CODE_KEY, codeInput.value.trim());
  status.textContent = `uploading ${file.name}…`;
  const body = new FormData();
  body.append("file", file);
  let res, data;
  try {
    res = await fetch("api/upload", { method: "POST", headers: codeHeaders(), body });
    data = await res.json().catch(() => ({}));
  } catch {
    status.textContent = "Couldn't reach the server. Please try again.";
    return;
  }
  if (res.status === 401) return needCode(data.error);
  if (!res.ok) { status.textContent = data.error || "That upload didn't work."; return; }
  codeRow.hidden = true;
  if (data.duplicate) {
    status.textContent = `already uploaded as ${data.paper_id}`;
    return loadDocs();
  }
  pollJob(data.job_id, file.name);
}

async function pollJob(jobId, name) {
  // Plain text that updates in place: parsing… -> embedding 32/78 -> ready.
  while (true) {
    let job;
    try { job = await (await fetch(`api/jobs/${jobId}`)).json(); } catch { job = null; }
    if (!job || !job.status) {
      status.textContent = "Lost track of that upload; check the list below.";
      return loadDocs();
    }
    if (job.status === "ready") {
      save(SCOPE_KEY, "user"); // the chat opens scoped to uploads
      status.innerHTML = `${escapeHtml(name)}: ${escapeHtml(job.stage_detail)} · <a href="./">ask about it →</a>`;
      return loadDocs();
    }
    if (job.status === "failed") {
      status.textContent = `${name}: ${job.error}`;
      return loadDocs();
    }
    status.textContent = `${name}: ${job.stage_detail || "queued…"}`;
    await new Promise((r) => setTimeout(r, 800));
  }
}

// --- lists ----------------------------------------------------------------------------
async function loadDocs() {
  let docs = [];
  try { docs = await (await fetch("api/documents")).json(); } catch {}
  docsEl.replaceChildren(...docs.map((d) => {
    const li = document.createElement("li");
    const state = d.status === "ready" ? `${d.n_pages} pages · ${d.n_chunks} passages` : d.status;
    li.innerHTML =
      `<span class="pid">${escapeHtml(d.paper_id)}</span>${escapeHtml(d.title)}` +
      `<span class="ptitle">${escapeHtml(d.filename)} · ${escapeHtml(state)} </span>`;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "tbtn danger";
    rm.textContent = "remove";
    rm.addEventListener("click", () => removeDocs(d.paper_id, `Remove ${d.filename}?`));
    li.querySelector(".ptitle").append(rm);
    return li;
  }));
  $("no-docs").hidden = docs.length > 0;
  clearDocs.hidden = docs.length < 2;
  if (!docs.some((d) => d.status === "ready") && load(SCOPE_KEY)) save(SCOPE_KEY, "");
}

async function removeDocs(id, prompt) {
  if (!confirm(prompt)) return;
  const res = await fetch(id ? `api/documents/${id}` : "api/documents", { method: "DELETE", headers: codeHeaders() }).catch(() => null);
  if (!res) { status.textContent = "Couldn't reach the server."; return; }
  if (res.status === 401) return needCode((await res.json().catch(() => ({}))).error);
  status.textContent = res.ok ? "removed" : "Couldn't remove that.";
  loadDocs();
}
clearDocs.addEventListener("click", () => removeDocs("", "Remove all uploaded documents?"));

async function loadPapers() {
  try {
    const papers = await (await fetch("api/papers")).json();
    $("papers").innerHTML = papers.map((p) =>
      `<li><span class="pid">${escapeHtml(p.id)}</span>${escapeHtml(p.short_cite)}` +
      `<span class="ptitle">${escapeHtml(p.title)}${p.doi ? ` · <a href="https://doi.org/${encodeURI(p.doi)}" target="_blank" rel="noopener">doi</a>` : ""}</span></li>`
    ).join("");
  } catch {
    $("papers").innerHTML = "<li>Couldn't load the paper list.</li>";
  }
}

loadDocs();
loadPapers();
