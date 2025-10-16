/* app.js — IMPROVED VERSION with enhancements */

/// ====== CONFIGURE YOUR ENDPOINTS HERE ======
/// 
/// IMPORTANT: Replace these URLs with your actual RunPod endpoints!
///
/// Option 1: Same endpoint for both (recommended for single deployment)
const ENDPOINT_ID = "YOUR_RUNPOD_ENDPOINT_ID"; // e.g., "abc123xyz"
const RUNPOD_BASE = `https://api.runpod.ai/v2/${ENDPOINT_ID}`;
const CALIBRATE_URL = `${RUNPOD_BASE}/runsync`;
const GENERATE_URL = `${RUNPOD_BASE}/runsync`;

/// Option 2: Separate endpoints (uncomment if you have two different functions)
// const CALIBRATE_URL = "https://api.runpod.ai/v2/YOUR_CALIBRATE_ID/runsync";
// const GENERATE_URL  = "https://api.runpod.ai/v2/YOUR_GENERATE_ID/runsync";

/// Optional: API Key (only needed if using authenticated endpoints)
const RUNPOD_API_KEY = ""; // Leave empty if not needed

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
  btnDownload: $("btnDownload"),
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
  const timestamp = new Date().toLocaleTimeString();
  els.log.value += `[${timestamp}] ${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text, kind="") {
  els.status.textContent = text || "";
  els.status.className = "hint " + (kind || "");
}

// IMPROVEMENT: Loading state management
function setLoading(isLoading) {
  els.btnVerify.disabled = isLoading;
  els.btnGenerate.disabled = isLoading;
  
  if (isLoading) {
    els.btnGenerate.innerHTML = '<span class="spinner active"></span> Processing...';
    els.btnVerify.innerHTML = '<span class="spinner active"></span> Verifying...';
  } else {
    els.btnGenerate.innerHTML = 'Generate Twin';
    els.btnVerify.innerHTML = 'Verify Photos';
  }
}

// ---- Photo handling ----
function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    state.files.push(f);
  }
  renderThumbs();
  logln(`Added ${files.length} photo(s). Total: ${state.files.length}`);
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
  logln("Converting images to base64...");
  state.b64s = [];
  for (const f of state.files.slice(0, 12)) {
    state.b64s.push(await fileToBase64(f));
  }
  logln(`Converted ${state.b64s.length} images`);
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
  els.retakeTips.textContent = tips || "No retake tips.";
}

// ---- Networking ----
// IMPROVEMENT: Better error messages and optional API key support
async function postJSON(url, payload) {
  const headers = { "Content-Type": "application/json" };
  
  // Add API key if configured
  if (RUNPOD_API_KEY && url.includes('runpod.ai')) {
    headers["Authorization"] = `Bearer ${RUNPOD_API_KEY}`;
  }
  
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ input: payload })
    });
    
    if (!r.ok) {
      const errorText = await r.text();
      throw new Error(`HTTP ${r.status}: ${errorText}`);
    }
    
    return await r.json();
  } catch (error) {
    // Better error messages
    if (error.message.includes('Failed to fetch')) {
      throw new Error('Network error. Check your endpoint URL and CORS settings.');
    }
    throw error;
  }
}

// ---- Actions ----
async function onVerify() {
  const startTime = Date.now();
  
  try {
    setLoading(true);
    setStatus("Verifying photos…");
    els.log.value = "";
    
    // Validation
    if (!els.height.value) { 
      setStatus("Height is required.", "status-bad"); 
      return; 
    }
    if (state.files.length < 3) { 
      setStatus("Please add at least 3 photos.", "status-bad"); 
      return; 
    }
    
    logln("Starting photo verification...");
    await collectBase64();

    // Prefer a dedicated CALIBRATE_URL; if you only have GENERATE_URL, you can still call it with allowPartial=false.
    const payload = { ...commonPayload(), required_roles:["front","side"], allowPartial:false };
    logln(`→ Calling ${CALIBRATE_URL}`);
    
    const res = await postJSON(CALIBRATE_URL, payload);
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    logln(`← Verification completed in ${elapsed}s`);
    
    if (res.ok === false && res.role_report) {
      renderQA(res);
      setStatus("Some photos need retake.", "status-warn");
      logln("❌ Photos failed quality check. See retake tips above.");
      return;
    }
    
    // If server returns 'ok' but also produces GLB, we still just show QA here.
    renderQA(res);
    setStatus(`Photos look OK ✓ (${elapsed}s)`, "status-ok");
    logln("✓ Photos passed quality check!");
    
  } catch (e) {
    console.error(e);
    logln(`❌ Error: ${e.message}`);
    setStatus("Verify failed. Check logs.", "status-bad");
  } finally {
    setLoading(false);
  }
}

async function onGenerate() {
  const startTime = Date.now();
  
  try {
    setLoading(true);
    setStatus("Generating twin… this may take 30-90 seconds.");
    
    // Validation
    if (!els.height.value) { 
      setStatus("Height is required.", "status-bad"); 
      return; 
    }
    if (state.files.length < 3) { 
      setStatus("Please add at least 3 photos.", "status-bad"); 
      return; 
    }
    
    logln("Starting twin generation...");
    await collectBase64();

    // Build payload for your RunPod handler
    const payload = {
      ...commonPayload(),
      required_roles:["front","side"],
      allowPartial:false,
      texRes: (els.highDetail.value === "true" ? 4096 : 2048)
    };

    logln(`→ Calling ${GENERATE_URL}`);
    logln(`   Settings: ${els.preset.value} preset, ${payload.texRes}px texture, ${els.poseMode.value} pose`);
    
    const res = await postJSON(GENERATE_URL, payload);
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    logln(`← Response received in ${elapsed}s`);
    
    if (!res.ok) {
      renderQA(res); // show role report if provided
      if (res.error) logln(`❌ Error: ${res.error}`);
      if (res.log) logln(res.log);
      setStatus("Generation failed. Check QA and logs.", "status-bad");
      return;
    }

    // 3D: load GLB
    if (res.glb_b64) {
      state.lastGLB = res.glb_b64;
      logln(`✓ Received GLB (${(res.glb_b64.length / 1024).toFixed(1)} KB base64)`);
      
      const blob = await (await fetch(`data:application/octet-stream;base64,${res.glb_b64}`)).blob();
      const url = URL.createObjectURL(blob);
      els.viewer.src = url;
      
      // Show download button
      if (els.btnDownload) {
        els.btnDownload.style.display = 'inline-block';
      }
      
      setStatus(`Twin ready ✓ (${elapsed}s)`, "status-ok");
      logln(`✓ 3D model loaded in viewer`);
    }
    
    renderQA(res);
    if (res.log) {
      logln("--- Blender Log ---");
      logln(res.log);
    }
    
  } catch (e) {
    console.error(e);
    logln(`❌ Error: ${e.message}`);
    setStatus("Generate failed. Check logs.", "status-bad");
  } finally {
    setLoading(false);
  }
}

// IMPROVEMENT: Download GLB function
function onDownload() {
  if (!state.lastGLB) {
    logln("❌ No GLB file available to download");
    return;
  }
  
  try {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filename = `avatar-${timestamp}.glb`;
    
    const a = document.createElement('a');
    a.href = `data:application/octet-stream;base64,${state.lastGLB}`;
    a.download = filename;
    a.click();
    
    logln(`✓ Downloaded ${filename}`);
  } catch (e) {
    logln(`❌ Download failed: ${e.message}`);
  }
}

// ---- Viewer controls ----
els.btnReset.addEventListener("click", () => {
  els.viewer.resetTurntableRotation();
  logln("Reset camera view");
});

els.turn.addEventListener("input", () => {
  els.viewer.turntableRotation = Number(els.turn.value) * Math.PI / 180;
  els.viewer.autoRotate = false;
});

// ---- Drag & drop ----
els.fileInput.addEventListener("change", (e) => addFiles(e.target.files));
["dragenter","dragover"].forEach(ev => els.drop.addEventListener(ev, (e) => { 
  e.preventDefault(); 
  e.stopPropagation(); 
  els.drop.style.borderColor = "#3a3f49"; 
}));
["dragleave","drop"].forEach(ev => els.drop.addEventListener(ev, (e) => { 
  e.preventDefault(); 
  e.stopPropagation(); 
  els.drop.style.borderColor = "#2a2e36"; 
}));
els.drop.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  addFiles(files);
});

// ---- Handles (guide points) — draggable UI only, doesn't affect backend ----
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
if (els.btnDownload) {
  els.btnDownload.addEventListener("click", onDownload);
}

// ---- Startup ----
// Validate endpoint configuration
if (ENDPOINT_ID === "YOUR_RUNPOD_ENDPOINT_ID") {
  setStatus("⚠️ Please configure your RunPod endpoint in app.js", "status-warn");
  logln("⚠️ WARNING: Endpoint not configured!");
  logln("   Edit app.js line 9 and replace YOUR_RUNPOD_ENDPOINT_ID with your actual endpoint ID");
} else {
  setStatus("Awaiting photos…");
  logln("✓ App initialized");
  logln(`   Calibrate endpoint: ${CALIBRATE_URL}`);
  logln(`   Generate endpoint: ${GENERATE_URL}`);
}
