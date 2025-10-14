/* app.js — wire up upload, verify (QA), and twin generation + viewer */

/// ====== CONFIGURE YOUR ENDPOINTS HERE ======
/// If you only have a single RunPod handler, you can set both to the same URL.
// Example: const GENERATE_URL = "https://<your-runpod-endpoint>";
const CALIBRATE_URL = "https://YOUR_CALIBRATE_ENDPOINT"; // should return calibration payload (role_report, retake_tips, etc.)
const GENERATE_URL  = "https://YOUR_GENERATE_ENDPOINT";  // returns { ok, glb_b64?, role_report?, log? }

/// ===========================================

const state = {
  files: [],
  thumbs: [],
  b64s: [],
  lastQA: null,
  lastGLB: null
};

const $ = (id) => document.getElementById(id);

const els = {
  fileInput: $("fileInput"),
  drop: $("drop"),
  thumbs: $("thumbs"),
  qaSummary: $("qaSummary"),
  retakeTips: $("retakeTips"),
  viewer: $("viewer"),
  btnReset: $("btnReset"),
  turn: $("turn"),
  btnVerify: $("btnVerify"),
  btnGenerate: $("btnGenerate"),
  status: $("status"),
  log: $("log"),
  // form fields
  lang: $("lang"),
  preset: $("preset"),
  poseMode: $("poseMode"),
  highDetail: $("highDetail"),
  height: $("height"),
  chest: $("chest"),
  waist: $("waist"),
  hips: $("hips"),
  shoulder: $("shoulder"),
  inseam: $("inseam"),
  arm: $("arm"),
  foot_length: $("foot_length"),
  foot_width: $("foot_width"),
  foot_width_category: $("foot_width_category"),
  measureBoard: $("measureBoard")
};

function logln(msg) {
  els.log.value += (msg + "\n");
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text, kind="") {
  els.status.textContent = text || "";
  els.status.className = "hint " + (kind || "");
}

// ---- Photo handling ----
function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    state.files.push(f);
  }
  renderThumbs();
}

function renderThumbs() {
  els.thumbs.innerHTML = "";
  state.thumbs = [];
  const files = state.files.slice(0, 12); // cap to 12
  files.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `<img src="${url}"><div class="tag">#${i+1}</div>`;
    els.thumbs.appendChild(div);
    state.thumbs.push({ el: div, url });
  });
}

async function fileToBase64(file) {
  const buf = await file.arrayBuffer();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
  return b64;
}

async function collectBase64() {
  state.b64s = [];
  for (const f of state.files.slice(0, 12)) {
    state.b64s.push(await fileToBase64(f));
  }
}

// ---- Build payloads ----
function measurementsPayload() {
  const num = (v) => v ? Number(v) : undefined;
  return {
    height: num(els.height.value),      // cm or m accepted by backend
    chest: num(els.chest.value),
    waist: num(els.waist.value),
    hips: num(els.hips.value),
    shoulder: num(els.shoulder.value),
    inseam: num(els.inseam.value),
    arm: num(els.arm.value),
    foot_length: num(els.foot_length.value),
    foot_width: num(els.foot_width.value),
    foot_width_category: els.foot_width_category.value || undefined
  };
}

function commonPayload() {
  const m = measurementsPayload();
  return {
    ...m,
    lang: els.lang.value,
    preset: els.preset.value,
    poseMode: els.poseMode.value,              // "auto" | "neutral"
    highDetail: els.highDetail.value === "true",
    photos: { unordered: state.b64s }
  };
}

// ---- QA UI renderers ----
function roleChip(role, report) {
  const st = (report && report.status) || "missing";
  const label = st === "ok" ? "OK" : (st === "retry" ? "Retry" : "Missing");
  const cls = st === "ok" ? "ok" : (st === "retry" ? "status-warn" : "bad");
  return `<span class="pill ${cls}">${role}: ${label}</span>`;
}

function renderQA(res) {
  state.lastQA = res;
  const rr = res.role_report || {};
  els.qaSummary.innerHTML = `
    ${roleChip("front", rr.front)}
    ${roleChip("side", rr.side)}
    ${roleChip("back", rr.back)}
  `;
  const tips = (res.retake_tips || []).map(t => "• " + t).join("\n");
  els.retakeTips.textContent = tips;
}

// ---- Networking ----
async function postJSON(url, payload) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: payload })
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

// ---- Actions ----
async function onVerify() {
  try {
    setStatus("Verifying photos…");
    els.log.value = "";
    if (!els.height.value) { setStatus("Height is required.", "status-bad"); return; }
    if (state.files.length < 3) { setStatus("Please add at least 3 photos.", "status-bad"); return; }
    await collectBase64();

    // Prefer a dedicated CALIBRATE_URL; if you only have GENERATE_URL, you can still call it with allowPartial=false.
    const payload = { ...commonPayload(), required_roles:["front","side"], allowPartial:false };
    logln("→ /calibrate payload (truncated images)");
    const res = await postJSON(CALIBRATE_URL, payload);
    logln("← calibration response");
    if (res.ok === false && res.role_report) {
      renderQA(res);
      setStatus("Some photos need retake.", "status-warn");
      return;
    }
    // If server returns 'ok' but also produces GLB, we still just show QA here.
    renderQA(res);
    setStatus("Photos look OK.", "status-ok");
  } catch (e) {
    console.error(e);
    logln(String(e));
    setStatus("Verify failed.", "status-bad");
  }
}

async function onGenerate() {
  try {
    setStatus("Generating twin… this may take a bit.");
    if (!els.height.value) { setStatus("Height is required.", "status-bad"); return; }
    if (state.files.length < 3) { setStatus("Please add at least 3 photos.", "status-bad"); return; }
    await collectBase64();

    // Build payload for your RunPod handler
    const payload = {
      ...commonPayload(),
      required_roles:["front","side"],
      allowPartial:false,
      texRes: (els.highDetail.value === "true" ? 4096 : 2048)
    };

    logln("→ /generate payload (truncated images)");
    const res = await postJSON(GENERATE_URL, payload);
    logln("← generate response");
    if (!res.ok) {
      renderQA(res); // show role report if provided
      if (res.error) logln(res.error);
      if (res.log) logln(res.log);
      setStatus("Generation failed. Check QA and logs.", "status-bad");
      return;
    }

    // 3D: load GLB
    if (res.glb_b64) {
      state.lastGLB = res.glb_b64;
      const blob = await (await fetch(`data:application/octet-stream;base64,${res.glb_b64}`)).blob();
      const url = URL.createObjectURL(blob);
      els.viewer.src = url;
      setStatus("Twin ready ✓", "status-ok");
    }
    renderQA(res);
    if (res.log) logln(res.log);
  } catch (e) {
    console.error(e);
    logln(String(e));
    setStatus("Generate failed.", "status-bad");
  }
}

// ---- Viewer controls ----
els.btnReset.addEventListener("click", () => els.viewer.resetTurntableRotation());
els.turn.addEventListener("input", () => {
  els.viewer.turntableRotation = Number(els.turn.value) * Math.PI / 180;
  els.viewer.autoRotate = false;
});

// ---- Drag & drop ----
els.fileInput.addEventListener("change", (e) => addFiles(e.target.files));
["dragenter","dragover"].forEach(ev => els.drop.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); els.drop.style.borderColor = "#3a3f49"; }));
["dragleave","drop"].forEach(ev => els.drop.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); els.drop.style.borderColor = "#2a2e36"; }));
els.drop.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  addFiles(files);
});

// ---- Handles (guide points) — draggable UI only, doesn’t affect backend ----
(function initHandles(){
  let drag = null;
  const board = els.measureBoard;
  function onDown(e){
    if (!(e.target.classList && e.target.classList.contains("handle"))) return;
    drag = { el:e.target, startX:e.clientX, startY:e.clientY };
    e.target.style.cursor = "grabbing";
  }
  function onMove(e){
    if (!drag) return;
    const rect = board.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 100;
    const ny = ((e.clientY - rect.top) / rect.height) * 100;
    drag.el.style.left = Math.max(2, Math.min(98, nx)) + "%";
    drag.el.style.top  = Math.max(2, Math.min(98, ny)) + "%";
  }
  function onUp(){
    if (drag) drag.el.style.cursor = "grab";
    drag = null;
  }
  board.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
})();

// ---- Buttons ----
els.btnVerify.addEventListener("click", onVerify);
els.btnGenerate.addEventListener("click", onGenerate);

// ---- Startup ----
setStatus("Awaiting photos…");
