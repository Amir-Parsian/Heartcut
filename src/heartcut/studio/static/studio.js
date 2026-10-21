"use strict";

/* Heartcut Studio front end: plain JavaScript, no build step, no external requests.
 *
 * Views (hash routes): #/ (projects), #/new (the five-step wizard) and #/project/<id>
 * (progress, then results). Columns and models move by drag and drop with pointer events,
 * so a mouse, a finger or a pen all work; every drag has a click or keyboard alternative.
 * Text from files is only ever inserted with textContent, never parsed as HTML.
 */

// ------------------------------------------------------------------ helpers

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const SVG_NS = "http://www.w3.org/2000/svg";

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

const ICONS = {
  check: ["M5 12.5l4.5 4.5L19 7"],
  x: ["M6 6l12 12M18 6L6 18"],
  download: ["M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5", "M5 19h14"],
  upload: ["M12 16V5m0 0-4.5 4.5M12 5l4.5 4.5", "M5 19h14"],
  file: ["M7 3h7l5 5v13H7z", "M14 3v5h5", "M10 13h6M10 17h6"],
  trophy: ["M8 4h8v5a4 4 0 0 1-8 0V4z", "M8 6H5a3 3 0 0 0 3 4M16 6h3a3 3 0 0 1-3 4", "M12 13v4M8.5 20h7M10 17h4"],
  play: ["M8 5v14l11-7z"],
  stop: ["M7 7h10v10H7z"],
  trash: ["M5 7h14M10 11v6M14 11v6", "M9 7l1-3h4l1 3", "M6.5 7l1 13h9l1-13"],
};

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  for (const d of ICONS[name]) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

const fmt = {
  int: (n) => Number(n || 0).toLocaleString(),
  pct: (v) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(1)}%`),
  money(v) {
    if (v === null || v === undefined) return "unknown";
    if (v === 0) return "$0";
    if (v < 0.01) return "under $0.01";
    return `$${v.toFixed(v < 10 ? 2 : 0)}`;
  },
  ms(v) {
    if (v === null || v === undefined) return "n/a";
    if (v < 1) return "under 1 ms";
    if (v < 100) return `${v.toFixed(1)} ms`;
    if (v < 10000) return `${Math.round(v)} ms`;
    return `${(v / 1000).toFixed(1)} s`;
  },
  mb: (v) => (v === null || v === undefined ? "n/a" : v < 1 ? `${(v * 1000).toFixed(0)} KB` : `${v.toFixed(1)} MB`),
  times(ratio) {
    if (ratio >= 100) return `${Math.round(ratio).toLocaleString()}x`;
    return `${ratio.toFixed(ratio >= 10 ? 0 : 1)}x`;
  },
  date: (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : ""),
  duration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}:${String(s % 60).padStart(2, "0")}`;
  },
};

function loading(text) {
  return h("div", { class: "loading" }, h("div", { class: "spinner" }), text);
}

function callout(kind, title, ...body) {
  return h("div", { class: `callout ${kind}` }, title ? h("strong", {}, title) : null, ...body);
}

let toastTimer;
function toast(message, kind = "") {
  const box = $("#toast");
  box.textContent = message;
  box.className = `toast ${kind}`;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), kind === "bad" ? 7000 : 4000);
}

// ------------------------------------------------------------------ API

async function errorText(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => `${(d.loc || []).slice(-1)[0] || "value"}: ${d.msg}`).join("; ");
    }
  } catch (_) {
    /* not JSON */
  }
  return `The studio answered ${response.status} ${response.statusText}.`;
}

const api = {
  async call(method, url, body) {
    const options = { method, headers: {} };
    if (body instanceof Blob) {
      options.headers["Content-Type"] = "application/octet-stream";
      options.body = body;
    } else if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(url, options);
    } catch (_) {
      throw new Error("Can't reach the studio. Is `heartcut studio` still running?");
    }
    if (!response.ok) throw new Error(await errorText(response));
    return response.json();
  },
  get: (url) => api.call("GET", url),
  post: (url, body) => api.call("POST", url, body === undefined ? {} : body),
  put: (url, body) => api.call("PUT", url, body),
  del: (url) => api.call("DELETE", url),
};

const projectUrl = (id, rest = "") => `/api/projects/${encodeURIComponent(id)}${rest}`;

// ------------------------------------------------------------------ state

const S = { catalog: null, wiz: null, project: null, poll: null };

async function loadCatalog(force = false) {
  if (!S.catalog || force) S.catalog = await api.get("/api/catalog");
  return S.catalog;
}
const teacherInfo = (id) => S.catalog.teachers.find((t) => t.id === id);
const studentInfo = (id) => S.catalog.students.find((s) => s.id === id);

function freshWizard() {
  return {
    step: 0,
    maxStep: 0,
    upload: null,
    text: [],
    label: null,
    profile: null,
    name: "",
    task: "",
    teacher: "simulated",
    apiKey: "",
    budget: 5,
    maxTrain: "",
    localModel: "",
    localUrl: "",
    students: ["tfidf-logreg"],
    thorough: false,
    draft: null,
    dirty: true,
    started: false,
  };
}

// ------------------------------------------------------------------ drag and drop

// Which zones take which kind of item.
const ZONE_ACCEPTS = { palette: "column", text: "column", label: "column", shelf: "model", lineup: "model" };

function hasFiles(event) {
  return [...(event.dataTransfer?.types || [])].includes("Files");
}

/* Make `el` draggable with pointer events; `onDrop(zoneName)` runs on a valid drop.
 * A press that moves less than a few pixels stays an ordinary click. */
function draggable(el, kind, onDrop) {
  el.addEventListener("pointerdown", (down) => {
    if (down.button !== 0 || down.target.closest(".no-drag")) return;
    const rect = el.getBoundingClientRect();
    const dx = down.clientX - rect.left;
    const dy = down.clientY - rect.top;
    let ghost = null;
    let zone = null;

    const move = (ev) => {
      if (!ghost) {
        if (Math.hypot(ev.clientX - down.clientX, ev.clientY - down.clientY) < 6) return;
        ghost = el.cloneNode(true);
        ghost.classList.add("drag-ghost");
        ghost.style.width = `${rect.width}px`;
        document.body.append(ghost);
        el.classList.add("dragging");
        document.body.classList.add("is-dragging");
        for (const z of $$("[data-zone]")) z.classList.toggle("can-drop", ZONE_ACCEPTS[z.dataset.zone] === kind);
        closeChipMenu();
      }
      ev.preventDefault();
      ghost.style.transform = `translate(${ev.clientX - dx}px, ${ev.clientY - dy}px) rotate(1.5deg)`;
      const under = document.elementFromPoint(ev.clientX, ev.clientY)?.closest("[data-zone]");
      const target = under && ZONE_ACCEPTS[under.dataset.zone] === kind ? under : null;
      if (target !== zone) {
        zone?.classList.remove("over");
        target?.classList.add("over");
        zone = target;
      }
    };
    const finish = (drop) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      if (!ghost) return;
      ghost.remove();
      el.classList.remove("dragging");
      document.body.classList.remove("is-dragging");
      for (const z of $$("[data-zone]")) z.classList.remove("can-drop", "over");
      // The click that follows a drag must not open the chip menu.
      const swallow = (e) => {
        e.stopPropagation();
        e.preventDefault();
      };
      el.addEventListener("click", swallow, { capture: true, once: true });
      setTimeout(() => el.removeEventListener("click", swallow, { capture: true }), 0);
      if (drop && zone) onDrop(zone.dataset.zone);
    };
    const up = () => finish(true);
    const cancel = () => finish(false);
    window.addEventListener("pointermove", move, { passive: false });
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", cancel);
  });
}

function setEmpty(zone, empty) {
  zone.classList.toggle("is-empty", empty);
}

/* A file drop target that is also a click-to-browse button. */
function fileDrop(zone, input, onFile) {
  zone.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    zone.classList.remove("over");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    if (input.files[0]) onFile(input.files[0]);
    input.value = "";
  });
}

// ------------------------------------------------------------------ routing

function route() {
  stopPolling();
  closeChipMenu();
  const hash = location.hash || "#/";
  for (const view of $$(".view")) view.hidden = true;
  if (hash.startsWith("#/project/")) return showProject(decodeURIComponent(hash.slice("#/project/".length)));
  if (hash === "#/new") return showWizard();
  return showHome();
}

// ------------------------------------------------------------------ home

const STATUS_TEXT = {
  draft: "Not started",
  running: "Training",
  done: "Done",
  failed: "Needs attention",
  stopped: "Paused",
};

function statusPill(status) {
  return h("span", { class: `pill ${status}` }, STATUS_TEXT[status] || status);
}

async function showHome() {
  $("#view-home").hidden = false;
  const list = $("#project-list");
  if (!list.children.length) list.replaceChildren(loading("Loading your projects..."));
  try {
    const [catalog, projects] = await Promise.all([loadCatalog(true), api.get("/api/projects")]);
    $("#workspace-note").textContent = `Projects are saved on this computer in ${catalog.workspace}. Heartcut Studio ${catalog.version}.`;
    const busy = $("#busy-banner");
    busy.hidden = !catalog.active;
    if (catalog.active) {
      busy.replaceChildren("A project is training right now. ", h("a", { href: `#/project/${encodeURIComponent(catalog.active)}` }, "Watch its progress"));
    }
    list.replaceChildren(
      ...(projects.length
        ? projects.map(projectCard)
        : [
            h(
              "div",
              { class: "empty" },
              "No projects yet. ",
              h("a", { href: "#/new" }, "Start your first one"),
              ": a practice run with the sample tickets takes under a minute.",
            ),
          ]),
    );
  } catch (err) {
    list.replaceChildren(callout("bad", "Couldn't load your projects", err.message));
  }
}

function projectCard(p) {
  const s = p.summary;
  return h(
    "a",
    { class: "project-card", href: `#/project/${encodeURIComponent(p.id)}` },
    h("div", { class: "meta" }, statusPill(p.status), " ", p.teacher_title),
    h("h3", {}, p.name),
    s ? h("div", { class: "score" }, `${shortTitle(s.best.title)} student: ${fmt.pct(s.best.accuracy)} correct`) : null,
    h("div", { class: "meta" }, `${fmt.int(p.data.examples)} examples, ${p.data.categories} categories`),
    h("div", { class: "meta" }, fmt.date(p.created_at)),
  );
}

const shortTitle = (title) => title.split(" (")[0];

// ------------------------------------------------------------------ wizard

const STEP_RENDERERS = [renderDataStep, renderColumnsStep, renderTeacherStep, renderStudentsStep, renderReviewStep];

async function showWizard() {
  $("#view-new").hidden = false;
  if (!S.wiz || S.wiz.started) S.wiz = freshWizard();
  try {
    await loadCatalog(true);
  } catch (err) {
    toast(err.message, "bad");
    return;
  }
  renderWizard();
}

function renderWizard() {
  const w = S.wiz;
  for (const step of $$("#view-new .step")) step.hidden = Number(step.dataset.step) !== w.step;
  for (const li of $$("#stepper li")) {
    const n = Number(li.dataset.step);
    li.classList.toggle("current", n === w.step);
    li.classList.toggle("done", n !== w.step && n <= w.maxStep && n < 4);
    const button = $("button", li);
    button.disabled = n > w.maxStep;
    if (n === w.step) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  $("#back").style.visibility = w.step === 0 ? "hidden" : "visible";
  $("#next").replaceChildren(...(w.step === 4 ? [icon("play"), "Start training"] : ["Next"]));
  $("#next").disabled = false;
  $("#step-error").textContent = "";
  STEP_RENDERERS[w.step]();
}

function markDirty() {
  S.wiz.dirty = true;
  $("#step-error").textContent = "";
}

function stepError(step) {
  const w = S.wiz;
  if (step === 0) return w.upload ? "" : "Drop a spreadsheet, or use the sample tickets, to continue.";
  if (step === 1) {
    if (!w.profile) return "Still checking your columns...";
    if (!w.profile.ready) return w.profile.errors[0] || "Fix the columns first.";
    if (!w.name.trim()) return "Give your project a name.";
    if (w.task.trim().length < 3) return "Describe in a few words what the sorting is for.";
    return "";
  }
  if (step === 2) {
    const t = teacherInfo(w.teacher);
    if (!t || !t.available) return "That teacher isn't installed; pick another one.";
    if (t.api_key_env && !t.has_key && !w.apiKey.trim()) return `${t.title} needs an API key: paste it in the box below the cards.`;
    if (t.kind === "cloud" && !(Number(w.budget) > 0)) return "Set a spending limit above $0.";
    if (t.kind !== "practice" && w.maxTrain !== "" && !(Number(w.maxTrain) >= 10)) {
      return "Label at least 10 messages, or leave that box empty to label them all.";
    }
    return "";
  }
  if (step === 3) return w.students.length ? "" : "Put at least one student in your line-up.";
  return "";
}

function goTo(step) {
  const w = S.wiz;
  for (let s = w.step; s < step; s += 1) {
    const err = stepError(s);
    if (err) {
      w.step = s;
      renderWizard();
      $("#step-error").textContent = err;
      return;
    }
  }
  w.step = step;
  w.maxStep = Math.max(w.maxStep, step);
  renderWizard();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nextStep() {
  const w = S.wiz;
  const err = stepError(w.step);
  if (err) {
    $("#step-error").textContent = err;
    return;
  }
  if (w.step === 4) startTraining();
  else goTo(w.step + 1);
}

// Step 1: data ---------------------------------------------------------------

function renderDataStep() {
  const u = S.wiz.upload;
  const box = $("#upload-summary");
  if (!u) {
    box.replaceChildren();
    return;
  }
  const columns = u.columns.map((c) => c.name);
  box.replaceChildren(
    h(
      "div",
      { class: "file-card" },
      icon("file"),
      h("div", { class: "grow" }, h("div", { class: "name" }, u.filename), h("div", { class: "meta" }, `${fmt.int(u.rows)} rows, ${columns.length} columns`)),
      h("span", { class: "pill done" }, "Looks good"),
    ),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ...columns.map((c) => h("th", {}, c)))),
        h("tbody", {}, ...u.preview.map((row) => h("tr", {}, ...columns.map((c) => h("td", { title: row[c] }, row[c]))))),
      ),
    ),
    h("p", { class: "footnote" }, `The first ${u.preview.length} rows. Press Next to tell Heartcut which column is which.`),
  );
}

async function sendDataFile(file) {
  const zone = $("#data-drop");
  const limit = S.catalog.limits.max_upload_mb;
  if (file.size > limit * 1e6) {
    toast(`That file is larger than ${limit} MB. A few thousand rows are plenty.`, "bad");
    return;
  }
  zone.classList.add("busy");
  const title = $("strong", zone);
  const before = title.textContent;
  title.textContent = `Reading ${file.name}...`;
  try {
    acceptUpload(await api.call("POST", `/api/uploads?filename=${encodeURIComponent(file.name)}`, file));
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    zone.classList.remove("busy");
    title.textContent = before;
  }
}

async function useSample() {
  try {
    acceptUpload(await api.post("/api/uploads/sample"));
  } catch (err) {
    toast(err.message, "bad");
  }
}

function acceptUpload(upload) {
  const w = S.wiz;
  w.upload = upload;
  w.text = [...upload.guess.text_columns];
  w.label = upload.guess.label_column;
  w.profile = upload.profile;
  w.name = niceName(upload.filename);
  w.task = upload.task || w.task;
  w.maxStep = Math.max(w.maxStep, 1);
  markDirty();
  renderWizard();
}

function niceName(filename) {
  const stem = filename.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ").trim();
  return stem ? stem[0].toUpperCase() + stem.slice(1) : "My project";
}

// Step 2: columns ------------------------------------------------------------

function columnInfo(name) {
  return S.wiz.upload.columns.find((c) => c.name === name) || { name, sample: "", unique: 0 };
}

function renderColumnsStep() {
  const w = S.wiz;
  const used = new Set([...w.text, w.label].filter(Boolean));
  const palette = $("#column-palette");
  palette.replaceChildren(...w.upload.columns.filter((c) => !used.has(c.name)).map((c) => columnChip(c, "palette")));
  $("#slot-text").replaceChildren(...w.text.map((n) => columnChip(columnInfo(n), "text")));
  $("#slot-label").replaceChildren(...(w.label ? [columnChip(columnInfo(w.label), "label")] : []));
  setEmpty(palette.closest(".zone"), !palette.children.length);
  setEmpty($("#slot-text").closest(".zone"), !w.text.length);
  setEmpty($("#slot-label").closest(".zone"), !w.label);
  $("#project-name").value = w.name;
  $("#task").value = w.task;
  renderProfile();
}

function columnChip(column, zone) {
  const chip = h(
    "button",
    {
      type: "button",
      class: "chip",
      "aria-haspopup": "menu",
      "aria-label": `${column.name}: choose where this column goes`,
      title: `${fmt.int(column.unique)} different values. Drag it, or click to choose.`,
    },
    h("span", { class: "grip", "aria-hidden": "true" }, "⋮⋮"),
    h("span", { class: "chip-text" }, h("span", { class: "chip-name" }, column.name), h("span", { class: "chip-sample" }, column.sample || "(empty)")),
    zone === "palette"
      ? null
      : h(
          "span",
          {
            class: "remove no-drag",
            "aria-hidden": "true",
            title: "Leave this column out",
            onclick: (e) => {
              e.stopPropagation();
              placeColumn(column.name, "palette");
            },
          },
          "×",
        ),
  );
  chip.addEventListener("click", () => openChipMenu(chip, column.name));
  draggable(chip, "column", (target) => placeColumn(column.name, target));
  return chip;
}

function placeColumn(name, zone) {
  const w = S.wiz;
  w.text = w.text.filter((n) => n !== name);
  if (w.label === name) w.label = null;
  if (zone === "text") w.text.push(name);
  if (zone === "label") w.label = name;
  markDirty();
  renderColumnsStep();
  refreshProfile();
}

function openChipMenu(anchor, name) {
  const w = S.wiz;
  const menu = $("#chip-menu");
  const where = w.text.includes(name) ? "text" : w.label === name ? "label" : "palette";
  const options = [];
  if (where !== "text") options.push(["text", w.text.length ? "Add to the message" : "Use as the message"]);
  if (where !== "label") options.push(["label", w.label ? "Use as the category instead" : "Use as the category"]);
  if (where !== "palette") options.push(["palette", "Leave it out"]);
  menu.replaceChildren(
    ...options.map(([zone, text]) =>
      h(
        "button",
        {
          type: "button",
          role: "menuitem",
          onclick: () => {
            closeChipMenu();
            placeColumn(name, zone);
          },
        },
        text,
      ),
    ),
  );
  const rect = anchor.getBoundingClientRect();
  menu.hidden = false;
  menu.style.left = `${Math.min(rect.left, window.innerWidth - menu.offsetWidth - 16) + window.scrollX}px`;
  menu.style.top = `${rect.bottom + window.scrollY + 6}px`;
  $("button", menu)?.focus();
}

function closeChipMenu() {
  const menu = $("#chip-menu");
  if (menu) menu.hidden = true;
}

let profileTimer;
let profileSeq = 0;
function refreshProfile() {
  const w = S.wiz;
  w.profile = null;
  renderProfile();
  clearTimeout(profileTimer);
  profileTimer = setTimeout(async () => {
    const seq = ++profileSeq;
    try {
      const result = await api.post(`/api/uploads/${encodeURIComponent(w.upload.id)}/profile`, {
        text_columns: w.text,
        label_column: w.label,
      });
      if (seq !== profileSeq || S.wiz !== w) return;
      w.profile = result;
      if (w.step === 1) renderProfile();
    } catch (err) {
      toast(err.message, "bad");
    }
  }, 120);
}

function renderProfile() {
  const p = S.wiz.profile;
  const box = $("#profile");
  if (!p) {
    box.replaceChildren(loading("Checking your columns..."));
    return;
  }
  const parts = [];
  if (p.usable) {
    parts.push(
      h(
        "div",
        { class: "profile-stats" },
        h("span", {}, h("b", {}, fmt.int(p.usable)), " examples ready"),
        h("span", {}, h("b", {}, fmt.int(p.categories.length)), " categories"),
        p.ready ? h("span", { class: "pill done" }, "Ready to train") : null,
      ),
    );
  }
  for (const message of p.errors) parts.push(callout("bad", null, message));
  for (const message of p.warnings) parts.push(callout("warn", null, message));
  if (p.categories.length && !p.errors.length) parts.push(categoryBars(p.categories));
  box.replaceChildren(...parts);
}

function categoryBars(categories, limit = 12) {
  const max = Math.max(...categories.map((c) => c.count));
  const rows = categories.slice(0, limit).map((c) =>
    h(
      "div",
      { class: `bar-row${c.count < 5 ? " small" : ""}` },
      h("span", { class: "bar-name", title: c.name }, c.name),
      h("div", { class: "bar-track" }, h("div", { class: "bar-fill", style: `width:${((100 * c.count) / max).toFixed(1)}%` })),
      h("span", { class: "bar-count" }, fmt.int(c.count)),
    ),
  );
  if (categories.length > limit) rows.push(h("p", { class: "footnote" }, `and ${categories.length - limit} more categories`));
  return h("div", { class: "bars", "aria-label": "Examples per category" }, ...rows);
}

// Step 3: teacher ------------------------------------------------------------

function renderTeacherStep() {
  const w = S.wiz;
  $("#teacher-cards").replaceChildren(
    ...S.catalog.teachers.map((t) =>
      h(
        "button",
        {
          type: "button",
          role: "radio",
          class: "choice",
          "aria-checked": String(w.teacher === t.id),
          disabled: !t.available,
          onclick: () => {
            w.teacher = t.id;
            markDirty();
            renderTeacherStep();
          },
        },
        h("div", { class: "choice-top" }, h("strong", {}, t.title), h("span", { class: `pill${t.kind === "practice" ? " practice" : ""}` }, t.price)),
        h("p", {}, t.blurb),
        t.available ? null : h("span", { class: "install" }, `Not installed. ${t.install}`),
      ),
    ),
  );
  renderTeacherOptions();
}

function field(label, input, help) {
  return h("label", { class: "field" }, h("span", {}, label), input, help ? h("small", {}, help) : null);
}

function boundInput(attrs, key, after) {
  const w = S.wiz;
  return h("input", {
    ...attrs,
    value: w[key],
    oninput: (e) => {
      w[key] = e.target.value;
      markDirty();
      if (after) after();
    },
  });
}

function renderTeacherOptions() {
  const w = S.wiz;
  const t = teacherInfo(w.teacher);
  const parts = [];
  if (t.kind === "practice") {
    parts.push(
      callout(
        "info",
        "A free dry run",
        "Nothing leaves your computer and nothing is charged. Use it to check your data and see every step, then start a new project with a real teacher for results you can rely on.",
      ),
    );
  }
  if (t.api_key_env) {
    if (t.has_key) {
      parts.push(callout("good", "API key found", `Using ${t.api_key_env} from this computer. You're all set.`));
    } else {
      parts.push(
        field(
          "Anthropic API key",
          boundInput({ type: "password", autocomplete: "off", placeholder: "sk-ant-...", spellcheck: "false" }, "apiKey"),
          "Create one at console.anthropic.com. The studio keeps it in memory on this computer only; it is never saved to disk.",
        ),
      );
    }
  }
  if (t.kind === "local") {
    parts.push(
      callout("info", "Before you start", `Install Ollama, then download the model once: ollama pull ${w.localModel || t.model}`),
      h(
        "div",
        { class: "field-row" },
        field("Model name in Ollama", boundInput({ placeholder: t.model }, "localModel")),
        field("Ollama address", boundInput({ placeholder: t.base_url }, "localUrl")),
      ),
    );
  }
  if (t.kind !== "practice") {
    const examples = S.wiz.profile?.usable || 0;
    parts.push(
      h(
        "div",
        { class: "field-row" },
        t.kind === "cloud"
          ? field(
              "Spending limit (US dollars)",
              boundInput({ type: "number", min: "0.01", step: "0.5", inputmode: "decimal" }, "budget"),
              "The teacher stops when this is spent. Answers so far are kept and never paid for twice.",
            )
          : null,
        field(
          "Most messages to label",
          boundInput({ type: "number", min: "10", step: "10", placeholder: "all", inputmode: "numeric" }, "maxTrain"),
          `Leave empty to use all of about ${fmt.int(Math.round(examples * 0.68))} learning examples. Fewer is cheaper and faster.`,
        ),
      ),
    );
  }
  $("#teacher-options").replaceChildren(...parts);
  $("#teacher-options").className = "options";
}

// Step 4: students -----------------------------------------------------------

function renderStudentsStep() {
  const w = S.wiz;
  const all = S.catalog.students;
  const lineup = w.students.map(studentInfo).filter(Boolean);
  const shelf = all.filter((s) => !w.students.includes(s.id));
  $("#model-shelf").replaceChildren(...shelf.map((s) => modelCard(s, false)));
  $("#model-lineup").replaceChildren(...lineup.map((s) => modelCard(s, true)));
  setEmpty($("#model-shelf").closest(".zone"), !shelf.length);
  setEmpty($("#model-lineup").closest(".zone"), !lineup.length);
  for (const radio of $$("input[name=thorough]")) radio.checked = (radio.value === "thorough") === w.thorough;
}

function modelCard(student, inLineup) {
  const fact = (label, value) => h("span", {}, `${label} `, h("b", {}, value));
  const card = h(
    "div",
    { class: `model-card${student.available ? "" : " unavailable"}` },
    h(
      "div",
      { class: "model-top" },
      h("strong", {}, student.title),
      h("span", { class: "model-id" }, student.id),
      student.available || inLineup
        ? h(
            "button",
            {
              type: "button",
              class: "button small no-drag",
              onclick: () => moveModel(student.id, inLineup ? "shelf" : "lineup"),
              "aria-label": `${inLineup ? "Remove" : "Add"} the ${student.title} student`,
            },
            inLineup ? "Remove" : "Add",
          )
        : null,
    ),
    h("p", {}, student.blurb),
    h("div", { class: "facts" }, fact("Size", student.size), fact("Answers in", student.speed), fact("Training", student.training)),
    student.available ? null : h("p", { class: "install" }, `Needs extra components: ${student.install}`),
  );
  if (student.available) draggable(card, "model", (zone) => moveModel(student.id, zone));
  return card;
}

function moveModel(id, zone) {
  const w = S.wiz;
  w.students = w.students.filter((s) => s !== id);
  if (zone === "lineup") w.students.push(id);
  markDirty();
  renderStudentsStep();
}

// Step 5: review -------------------------------------------------------------

function settingsBody() {
  const w = S.wiz;
  const t = teacherInfo(w.teacher);
  return {
    name: w.name.trim(),
    upload_id: w.upload.id,
    text_columns: w.text,
    label_column: w.label,
    task: w.task.trim(),
    teacher: w.teacher,
    students: w.students,
    thorough: w.thorough,
    max_budget_usd: t.kind === "cloud" ? Number(w.budget) : null,
    max_train_examples: t.kind !== "practice" && w.maxTrain !== "" ? Number(w.maxTrain) : null,
    local_model: t.kind === "local" ? w.localModel.trim() || null : null,
    local_url: t.kind === "local" ? w.localUrl.trim() || null : null,
  };
}

async function renderReviewStep() {
  const w = S.wiz;
  const box = $("#review");
  if (w.dirty || !w.draft) {
    box.replaceChildren(loading("Preparing your data and estimating the cost..."));
    $("#next").disabled = true;
    try {
      const body = settingsBody();
      const draft = w.draft ? await api.put(projectUrl(w.draft.id), body) : await api.post("/api/projects", body);
      if (S.wiz !== w) return;
      w.draft = draft;
      w.dirty = false;
    } catch (err) {
      box.replaceChildren(callout("bad", "Heartcut couldn't set up this project", err.message));
      return;
    }
  }
  if (w.step !== 4) return;
  $("#next").disabled = false;
  box.replaceChildren(...reviewContent(w.draft));
}

function costSummary(estimate, teacher) {
  if (teacher.kind === "practice") return ["Free", "Practice run: no AI is called."];
  if (teacher.kind === "local") return ["No charge", "The teacher runs on this computer."];
  if (!estimate?.total) return ["Unknown", "No published price for this model."];
  const { low_usd: low, high_usd: high } = estimate.total;
  const range = low === high ? fmt.money(low) : `${fmt.money(low)} to ${fmt.money(high)}`;
  return [range, "A rough estimate for labeling every message once. Repeat runs reuse saved answers for free."];
}

function reviewContent(draft) {
  const w = S.wiz;
  const t = teacherInfo(w.teacher);
  const d = draft.data;
  const [cost, costNote] = costSummary(draft.estimate, t);
  const item = (label, value, note, cls = "") =>
    h("div", { class: `review-item ${cls}` }, h("span", {}, label), h("strong", {}, value), note ? h("small", {}, note) : null);
  const teacherNote =
    t.kind === "practice" ? "Simulated, for practice" : t.kind === "local" ? `${w.localModel || t.model}, on this computer` : t.model;
  const out = [
    h(
      "div",
      { class: "review-grid" },
      item("Your data", draft.source_file, `${fmt.int(d.examples)} examples in ${d.categories} categories`),
      item(
        "How it is used",
        `${fmt.int(d.train)} to learn from`,
        `${fmt.int(d.validation)} to pick the winner and ${fmt.int(d.test)} for the final exam, graded with your own categories, which no model sees while learning.`,
      ),
      item("Teacher", t.title, teacherNote),
      item(
        "Students",
        w.students.map((id) => studentInfo(id).title).join(", "),
        w.thorough ? "Thorough: three runs each and a learning curve" : "Quick: one run each",
      ),
      item("Teacher cost", cost, costNote, "cost"),
    ),
  ];
  if (t.kind === "cloud") {
    out.push(
      callout(
        "warn",
        "This makes paid calls",
        `Anthropic bills your account for the teacher's answers. Heartcut stops at your limit of ${fmt.money(Number(w.budget))}.`,
      ),
    );
  }
  out.push(h("p", { class: "footnote" }, "Under the hood this is an ordinary Heartcut experiment, saved as ", h("code", {}, draft.experiment_path)));
  return out;
}

async function startTraining() {
  const w = S.wiz;
  if (!w.draft || w.dirty) return;
  const t = teacherInfo(w.teacher);
  const next = $("#next");
  next.disabled = true;
  try {
    await api.post(projectUrl(w.draft.id, "/start"), {
      api_key: t.api_key_env && !t.has_key ? w.apiKey.trim() : null,
    });
    w.started = true;
    location.hash = `#/project/${encodeURIComponent(w.draft.id)}`;
  } catch (err) {
    $("#step-error").textContent = err.message;
    next.disabled = false;
  }
}

// ------------------------------------------------------------------ project

async function showProject(id) {
  $("#view-project").hidden = false;
  const body = $("#project-body");
  if (S.project?.id !== id) {
    body.dataset.rendered = "";
    body.replaceChildren(loading("Loading the project..."));
  }
  try {
    await loadCatalog();
    renderProject(await api.get(projectUrl(id)));
  } catch (err) {
    body.replaceChildren(h("a", { class: "back", href: "#/" }, "← All projects"), callout("bad", "Couldn't open this project", err.message));
  }
}

function stopPolling() {
  clearTimeout(S.poll);
  S.poll = null;
}

function schedulePoll(id) {
  stopPolling();
  S.poll = setTimeout(async () => {
    if (location.hash !== `#/project/${encodeURIComponent(id)}`) return;
    try {
      renderProject(await api.get(projectUrl(id)));
    } catch (_) {
      schedulePoll(id);
    }
  }, 1000);
}

function renderProject(p) {
  S.project = p;
  const body = $("#project-body");
  if (p.status === "done") {
    // Keep the page (and whatever was typed into "Try it") if it is already showing.
    if (body.dataset.rendered === `${p.id}:done`) return;
    body.replaceChildren(projectHeader(p), ...resultsView(p));
    body.dataset.rendered = `${p.id}:done`;
    return;
  }
  const log = $("details.log", body);
  const logOpen = log?.open ?? false;
  const scrolled = $("pre", log || body);
  const atBottom = !scrolled || scrolled.scrollTop + scrolled.clientHeight >= scrolled.scrollHeight - 8;
  const keep = p.status !== "running" && body.dataset.rendered === `${p.id}:${p.status}`;
  if (!keep) {
    body.replaceChildren(projectHeader(p), progressView(p, logOpen));
    body.dataset.rendered = `${p.id}:${p.status}`;
    const pre = $("details.log pre", body);
    if (pre && atBottom) pre.scrollTop = pre.scrollHeight;
  }
  if (p.status === "running") schedulePoll(p.id);
}

function projectHeader(p) {
  const practice = p.settings.teacher === "simulated";
  return h(
    "div",
    { class: "project-head" },
    h(
      "div",
      { class: "titles" },
      h("a", { class: "back", href: "#/" }, "← All projects"),
      h("h1", {}, p.name),
      h(
        "div",
        { class: "meta" },
        statusPill(p.status),
        practice ? h("span", { class: "pill practice" }, "Practice run") : null,
        h("span", {}, `${p.teacher_title}, ${fmt.int(p.data.examples)} examples, ${p.data.categories} categories`),
      ),
    ),
    p.running
      ? null
      : h("button", { type: "button", class: "button ghost danger small", onclick: () => deleteProject(p) }, icon("trash"), "Delete"),
  );
}

function progressView(p, logOpen) {
  const running = p.status === "running";
  const index = p.stage_index ?? -1;
  const headings = {
    running: "Training in progress",
    failed: "Training stopped with a problem",
    stopped: "Paused",
    draft: "Not started yet",
  };
  const timeline = h(
    "ol",
    { class: "timeline" },
    ...p.stages.map((stage, i) => {
      let state = "pending";
      if (i < index) state = "done";
      else if (i === index) state = running ? "active" : p.status === "failed" ? "failed" : "pending";
      const dot =
        state === "done" ? icon("check") : state === "active" ? h("div", { class: "spinner" }) : state === "failed" ? icon("x") : String(i + 1);
      return h(
        "li",
        { class: state },
        h("div", { class: "dot" }, dot),
        h("div", { class: "label" }, stage.title, state === "active" && p.stage_detail ? h("span", { class: "detail" }, p.stage_detail) : null),
      );
    }),
  );
  const started = p.started_at ? Date.parse(p.started_at) : null;
  const ended = p.finished_at ? Date.parse(p.finished_at) : Date.now();
  const parts = [
    h(
      "div",
      { class: "progress-head" },
      h("h2", {}, headings[p.status] || "Training"),
      started ? h("span", { class: "elapsed" }, `${running ? "Running for" : "Ran for"} ${fmt.duration((ended - started) / 1000)}`) : null,
    ),
    timeline,
  ];
  if (p.error) parts.push(callout(p.status === "stopped" ? "warn" : "bad", p.error.title, h("p", {}, p.error.detail)));
  if (running) {
    parts.push(
      h(
        "div",
        { class: "actions" },
        h("button", { type: "button", class: "button", onclick: () => stopProject(p) }, icon("stop"), "Stop after this step"),
        h("span", { class: "muted" }, "You can close this tab: training continues while the studio runs."),
      ),
    );
  } else {
    parts.push(resumeControls(p));
  }
  const lines = p.log && p.log.length ? p.log.join("\n") : "Nothing logged yet.";
  parts.push(h("details", { class: "log", open: logOpen }, h("summary", {}, "Show technical details"), h("pre", {}, lines)));
  return h("div", { class: "panel" }, ...parts);
}

function resumeControls(p) {
  const t = teacherInfo(p.settings.teacher);
  const code = p.error?.code;
  let keyInput = null;
  let budgetInput = null;
  const fields = [];
  if (t.api_key_env && (!t.has_key || code === "key")) {
    keyInput = h("input", { type: "password", autocomplete: "off", placeholder: "sk-ant-...", spellcheck: "false" });
    fields.push(field("Anthropic API key", keyInput));
  }
  if (code === "budget") {
    budgetInput = h("input", { type: "number", min: "0.01", step: "0.5", value: p.settings.max_budget_usd ?? 5 });
    fields.push(field("Spending limit for this run (US dollars)", budgetInput));
  }
  const label = p.status === "draft" ? "Start training" : "Resume";
  const button = h("button", { type: "button", class: "button primary" }, icon("play"), label);
  button.addEventListener("click", async () => {
    const body = {};
    if (keyInput?.value.trim()) body.api_key = keyInput.value.trim();
    if (budgetInput) body.max_budget_usd = Number(budgetInput.value);
    button.disabled = true;
    try {
      const updated = await api.post(projectUrl(p.id, "/start"), body);
      S.catalog = null;
      await loadCatalog();
      renderProject(updated);
    } catch (err) {
      toast(err.message, "bad");
      button.disabled = false;
    }
  });
  const note =
    p.status === "draft" ? null : h("span", { class: "muted" }, "Finished steps are reused, so nothing is redone or paid for twice.");
  return h("div", { class: "actions" }, ...fields, button, note);
}

async function stopProject(p) {
  try {
    renderProject(await api.post(projectUrl(p.id, "/stop")));
    toast("Stopping after the current step...");
  } catch (err) {
    toast(err.message, "bad");
  }
}

async function deleteProject(p) {
  if (!window.confirm(`Delete "${p.name}" and everything it trained? This can't be undone.`)) return;
  try {
    await api.del(projectUrl(p.id));
    toast("Project deleted.");
    S.project = null;
    location.hash = "#/";
  } catch (err) {
    toast(err.message, "bad");
  }
}

// Results --------------------------------------------------------------------

function verdictText(s) {
  const best = s.best;
  const teacher = s.teacher;
  const gap = teacher.accuracy - best.accuracy;
  let text = `It sorted ${fmt.pct(best.accuracy)} of the exam messages correctly, `;
  if (gap > 0.0005) text += `${(100 * gap).toFixed(1)} points below the teacher (${fmt.pct(teacher.accuracy)})`;
  else if (gap < -0.0005) text += `${(-100 * gap).toFixed(1)} points above the teacher (${fmt.pct(teacher.accuracy)})`;
  else text += `the same as the teacher`;
  const extras = [];
  if (best.cost_per_1k_usd && teacher.cost_per_1k_usd) extras.push(`about ${fmt.times(teacher.cost_per_1k_usd / best.cost_per_1k_usd)} cheaper to run`);
  if (best.latency_p50_ms && teacher.latency_p50_ms) extras.push(`${fmt.times(teacher.latency_p50_ms / best.latency_p50_ms)} faster`);
  return `${text}${extras.length ? `, and it is ${extras.join(" and ")}` : ""}.`;
}

function compareStat(title, rows, note) {
  const values = rows.map((r) => r.value).filter((v) => v !== null && v !== undefined);
  const max = rows[0].scale ?? Math.max(...values, 1e-12);
  return h(
    "div",
    { class: "stat" },
    h("h3", {}, title),
    h(
      "div",
      { class: "cmp" },
      ...rows.map((r) =>
        h(
          "div",
          { class: `cmp-row ${r.cls}` },
          h("span", { class: "who" }, r.who),
          h(
            "div",
            { class: "bar-track" },
            h("div", { class: "bar-fill", style: `width:${r.value === null || r.value === undefined ? 0 : Math.max(2, (100 * r.value) / max).toFixed(1)}%` }),
          ),
          h("span", { class: "val" }, r.text),
        ),
      ),
    ),
    note ? h("p", { class: "note" }, note) : null,
  );
}

function resultsView(p) {
  const s = p.summary;
  const best = s.best;
  const teacher = s.teacher;
  const winner = shortTitle(best.title);
  const out = [];
  if (s.simulated) {
    out.push(
      callout(
        "warn",
        "Practice results",
        "The teacher was simulated, so these numbers only show how the process works. Start a new project with a real teacher, such as Claude Haiku, to measure your task.",
      ),
    );
  }
  out.push(
    h(
      "div",
      { class: "panel verdict" },
      h("div", { class: "trophy" }, icon("trophy")),
      h("div", {}, h("p", { class: "eyebrow" }, "Your winner"), h("h2", {}, `The ${winner} student`), h("p", {}, verdictText(s))),
      h(
        "div",
        { class: "actions" },
        h("a", { class: "button primary", href: projectUrl(p.id, "/download"), download: "" }, icon("download"), "Download model"),
      ),
    ),
  );

  const costNote =
    best.cost_per_1k_usd && teacher.cost_per_1k_usd
      ? `The student runs on an ordinary computer; its cost is a rented 4-core server's price at the speed measured here.`
      : "The teacher's price is unknown, so the costs can't be compared.";
  out.push(
    h(
      "div",
      { class: "stat-grid" },
      compareStat(
        "Correct answers on the final exam",
        [
          { who: winner, cls: "student", value: best.accuracy, text: fmt.pct(best.accuracy), scale: 1 },
          { who: "Teacher", cls: "teacher", value: teacher.accuracy, text: fmt.pct(teacher.accuracy) },
        ],
        `${fmt.int(s.test_examples)} messages you sorted yourself, which no model saw while learning.`,
      ),
      compareStat(
        "Cost per 1,000 messages",
        [
          { who: winner, cls: "student", value: best.cost_per_1k_usd, text: fmt.money(best.cost_per_1k_usd) },
          { who: "Teacher", cls: "teacher", value: teacher.cost_per_1k_usd, text: fmt.money(teacher.cost_per_1k_usd) },
        ],
        costNote,
      ),
      compareStat(
        "Time to answer one message",
        [
          { who: winner, cls: "student", value: best.latency_p50_ms, text: fmt.ms(best.latency_p50_ms) },
          { who: "Teacher", cls: "teacher", value: teacher.latency_p50_ms || null, text: teacher.latency_p50_ms ? fmt.ms(teacher.latency_p50_ms) : "n/a" },
        ],
        `Typical (median) time on this computer, ${s.hardware}.`,
      ),
    ),
  );

  out.push(h("div", { class: "two-col" }, tryPanel(p), batchPanel(p)));

  out.push(
    h(
      "div",
      { class: "panel" },
      h("h3", {}, "Every model"),
      h(
        "p",
        { class: "muted" },
        `The teacher agreed with your own categories on ${fmt.pct(s.teacher_label_accuracy)} of the ${fmt.int(s.teacher_labeled)} messages it labeled for the students. Students learn from those answers, so they rarely beat the teacher by much.`,
      ),
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          {},
          h(
            "thead",
            {},
            h(
              "tr",
              {},
              h("th", {}, "Model"),
              h("th", { class: "num" }, "Correct"),
              h("th", { class: "num" }, "Cost per 1,000"),
              h("th", { class: "num" }, "Time per message"),
              h("th", { class: "num" }, "Size"),
            ),
          ),
          h(
            "tbody",
            {},
            ...s.models.map((m) =>
              h(
                "tr",
                { class: m.best ? "best" : "" },
                h("td", {}, m.title, m.best ? [" ", h("span", { class: "pill done" }, "Winner")] : null),
                h("td", { class: "num" }, fmt.pct(m.accuracy)),
                h("td", { class: "num" }, fmt.money(m.cost_per_1k_usd)),
                h("td", { class: "num" }, m.latency_p50_ms ? fmt.ms(m.latency_p50_ms) : "n/a"),
                h("td", { class: "num" }, m.role === "student" ? fmt.mb(m.size_mb) : "hosted"),
              ),
            ),
          ),
        ),
      ),
    ),
  );

  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const chart = (key, caption) => {
    const name = s.charts[dark && s.charts[`${key}-dark`] ? `${key}-dark` : key];
    return name ? h("figure", {}, h("img", { src: projectUrl(p.id, `/files/${encodeURIComponent(name)}`), alt: caption, loading: "lazy" }), h("figcaption", {}, caption)) : null;
  };
  const figures = [
    chart("pareto", "Accuracy against cost per 1,000 messages: up and to the left is better."),
    chart("learning_curve", "How accuracy grows with the number of labeled examples."),
  ].filter(Boolean);
  if (figures.length) out.push(h("div", { class: "panel" }, h("h3", {}, "Charts"), h("div", { class: "charts" }, ...figures)));

  if (s.confusions.length) {
    out.push(
      h(
        "div",
        { class: "panel" },
        h("h3", {}, "Where the winner gets mixed up"),
        h("p", { class: "muted" }, `${fmt.int(s.errors)} of ${fmt.int(s.test_examples)} exam messages got a different category from yours. The most common mix-ups:`),
        h(
          "ul",
          { class: "confusions" },
          ...s.confusions.map((c) =>
            h(
              "li",
              {},
              `${c.count}x your `,
              h("span", { class: "tag" }, c.gold),
              " was sorted as ",
              h("span", { class: "tag" }, c.predicted),
              ...c.examples.slice(0, 2).map((text) => h("q", {}, text)),
            ),
          ),
        ),
      ),
    );
  }

  out.push(
    h(
      "details",
      { class: "panel dev" },
      h("summary", {}, "For developers"),
      h("p", {}, "Re-run or extend this project from the command line:"),
      h("pre", {}, `heartcut run -c ${p.experiment_path}`),
      h("p", {}, "Serve the winner as an HTTP API:"),
      h("pre", {}, `heartcut serve --model ${s.export_dir}`),
      h(
        "div",
        { class: "links" },
        ...s.reports.map((name) => h("a", { href: projectUrl(p.id, `/files/${encodeURIComponent(name)}`), target: "_blank", rel: "noopener" }, name)),
      ),
    ),
  );
  return out;
}

function tryPanel(p) {
  const textarea = h("textarea", { placeholder: "Type or paste a message, e.g. a new support email...", "aria-label": "Message to sort" });
  const answer = h("div", { "aria-live": "polite" });
  const button = h("button", { type: "button", class: "button primary" }, "Sort it");
  const run = async () => {
    const text = textarea.value.trim();
    if (!text) {
      textarea.focus();
      return;
    }
    button.disabled = true;
    try {
      const result = await api.post(projectUrl(p.id, "/predict"), { texts: [text] });
      answer.replaceChildren(answerView(result.predictions[0]));
    } catch (err) {
      answer.replaceChildren(callout("bad", null, err.message));
    } finally {
      button.disabled = false;
    }
  };
  button.addEventListener("click", run);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
  });
  return h(
    "div",
    { class: "panel" },
    h("h3", {}, "Try it"),
    h("p", { class: "muted" }, "Type a message and see which category the winner picks."),
    textarea,
    h("div", { class: "try-row" }, button, h("span", { class: "muted" }, "or press Ctrl+Enter")),
    answer,
  );
}

function answerView(prediction) {
  const c = prediction.confidence;
  const word = c >= 0.8 ? "Very sure" : c >= 0.5 ? "Fairly sure" : "Unsure: worth a human check";
  return h(
    "div",
    { class: "answer" },
    h("span", { class: "answer-label" }, prediction.label),
    h(
      "div",
      { class: `confidence${c < 0.5 ? " unsure" : ""}` },
      h("div", { class: "bar-track" }, h("div", { class: "bar-fill", style: `width:${(100 * c).toFixed(1)}%` })),
      h("span", {}, `${word} (${fmt.pct(c)})`),
    ),
    prediction.alternatives.length
      ? h("p", { class: "others" }, `Next best: ${prediction.alternatives.map((a) => `${a.label} (${fmt.pct(a.confidence)})`).join(", ")}`)
      : null,
  );
}

function batchPanel(p) {
  const input = h("input", { type: "file", hidden: true, accept: ".csv,.tsv,.tab,.txt,.jsonl,.ndjson,.json,.xlsx,.xlsm" });
  const zone = h(
    "label",
    { class: "dropzone compact", tabindex: "0" },
    input,
    icon("upload"),
    h("strong", {}, "Drop a spreadsheet of new messages"),
    h("span", {}, "or ", h("u", {}, "click to choose a file")),
  );
  const status = h("div", { "aria-live": "polite" });
  fileDrop(zone, input, (file) => labelFile(p, file, zone, status));
  return h(
    "div",
    { class: "panel" },
    h("h3", {}, "Sort a whole spreadsheet"),
    h(
      "p",
      { class: "muted" },
      `You get it back as CSV with two new columns: predicted_category and confidence. It needs the same message column${p.settings.text_columns.length > 1 ? "s" : ""} as before (${p.settings.text_columns.join(", ")}) or a single column of messages.`,
    ),
    zone,
    status,
  );
}

async function labelFile(p, file, zone, status) {
  zone.classList.add("busy");
  status.replaceChildren(loading(`Sorting the messages in ${file.name}...`));
  try {
    const response = await fetch(projectUrl(p.id, `/label-file?filename=${encodeURIComponent(file.name)}`), {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    if (!response.ok) throw new Error(await errorText(response));
    const blob = await response.blob();
    const match = /filename="([^"]+)"/.exec(response.headers.get("content-disposition") || "");
    const name = match ? match[1] : "labeled.csv";
    const link = h("a", { class: "button small", href: URL.createObjectURL(blob), download: name }, icon("download"), `Download ${name}`);
    status.replaceChildren(callout("good", "Done", h("p", {}, "Your download should start by itself. If it didn't: "), link));
    link.click();
  } catch (err) {
    status.replaceChildren(callout("bad", "Couldn't sort that file", err.message));
  } finally {
    zone.classList.remove("busy");
  }
}

// ------------------------------------------------------------------ start up

function init() {
  fileDrop($("#data-drop"), $("#data-file"), sendDataFile);
  $("#use-sample").addEventListener("click", useSample);
  $("#back").addEventListener("click", () => {
    S.wiz.step = Math.max(0, S.wiz.step - 1);
    renderWizard();
  });
  $("#next").addEventListener("click", nextStep);
  for (const li of $$("#stepper li")) {
    $("button", li).addEventListener("click", () => {
      const n = Number(li.dataset.step);
      if (n < S.wiz.step) {
        S.wiz.step = n;
        renderWizard();
      } else {
        goTo(n);
      }
    });
  }
  $("#project-name").addEventListener("input", (e) => {
    S.wiz.name = e.target.value;
    markDirty();
  });
  $("#task").addEventListener("input", (e) => {
    S.wiz.task = e.target.value;
    markDirty();
  });
  for (const radio of $$("input[name=thorough]")) {
    radio.addEventListener("change", () => {
      S.wiz.thorough = radio.value === "thorough" && radio.checked;
      markDirty();
    });
  }
  document.addEventListener("pointerdown", (e) => {
    if (!e.target.closest("#chip-menu, .chip")) closeChipMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeChipMenu();
  });
  // A file dropped anywhere on the data step counts; elsewhere the browser must not open it.
  window.addEventListener("dragover", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    const onDataStep = !$("#view-new").hidden && S.wiz?.step === 0;
    if (onDataStep && e.dataTransfer.files[0]) sendDataFile(e.dataTransfer.files[0]);
  });
  window.addEventListener("hashchange", route);
  route();
}

init();
