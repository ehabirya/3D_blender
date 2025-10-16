/* Dynamic app.js with gender-based figures and natural measurement points */

let ENDPOINT_FULL = "";
let RUNPOD_API_KEY = "";
let CURRENT_GENDER = "male";

const state = {
  roleFiles: { front: null, side: null, back: null, foot: null },
  b64s: [],
  lastQA: null,
  lastGLB: null,
  measurements: {}
};

const $ = (id) => document.getElementById(id);

const els = {
  endpointFull: $("endpointFull"),
  apiKey: $("apiKey"),
  thumbsReview: $("thumbsReview"),
  qaSummary: $("qaSummary"),
  retakeTipsContainer: $("retakeTipsContainer"),
  viewer: $("viewer"),
  btnReset: $("btnReset"),
  turn: $("turn"),
  btnVerify: $("btnVerify"),
  btnGenerate: $("btnGenerate"),
  btnDownload: $("btnDownload"),
  statusContainer: $("statusContainer"),
  log: $("log"),
  figureWrapper: $("figureWrapper")
};

// Measurement point configurations for male and female
const MEASUREMENT_CONFIGS = {
  female: {
    figure: `
      <svg viewBox="0 0 300 600" style="width: 100%; height: auto;">
        <!-- Height line -->
        <line x1="20" y1="30" x2="20" y2="580" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,5" opacity="0.3"/>
        
        <!-- Female silhouette -->
        <g fill="#94a3b8" opacity="0.8">
          <!-- Head -->
          <ellipse cx="150" cy="50" rx="30" ry="38"/>
          <!-- Neck -->
          <rect x="140" y="85" width="20" height="20" rx="5"/>
          <!-- Body -->
          <path d="M 150 105 L 140 130 L 135 180 L 130 230 L 135 280 L 145 330 L 148 420 L 150 580 M 150 105 L 160 130 L 165 180 L 170 230 L 165 280 L 155 330 L 152 420 L 150 580" stroke="#94a3b8" stroke-width="25" fill="none" stroke-linecap="round"/>
          <!-- Arms -->
          <path d="M 140 130 L 100 160 L 95 240 M 160 130 L 200 160 L 205 240" stroke="#94a3b8" stroke-width="18" fill="none" stroke-linecap="round"/>
          <!-- Legs -->
          <path d="M 145 330 L 140 420 L 135 520 L 130 580 M 155 330 L 160 420 L 165 520 L 170 580" stroke="#94a3b8" stroke-width="20" fill="none" stroke-linecap="round"/>
        </g>
        
        <!-- Measurement lines -->
        <g stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.5">
          <!-- Head -->
          <line x1="120" y1="50" x2="250" y2="50"/>
          <!-- Neck -->
          <line x1="120" y1="95" x2="250" y2="95"/>
          <!-- Shoulder -->
          <line x1="100" y1="130" x2="250" y2="130"/>
          <!-- Bust -->
          <line x1="115" y1="165" x2="250" y2="165"/>
          <!-- Underbust -->
          <line x1="120" y1="185" x2="250" y2="185"/>
          <!-- Waist -->
          <line x1="115" y1="230" x2="250" y2="230"/>
          <!-- Hip -->
          <line x1="115" y1="280" x2="250" y2="280"/>
          <!-- Arm -->
          <line x1="50" y1="200" x2="95" y2="200"/>
          <!-- Hand -->
          <line x1="50" y1="240" x2="95" y2="240"/>
          <!-- Inseam -->
          <line x1="135" y1="420" x2="250" y2="420"/>
          <!-- Foot -->
          <line x1="110" y1="580" x2="190" y2="580"/>
        </g>
      </svg>
    `,
    points: [
      { id: "m_head", label: "Head", top: "8.3%", left: "83%", side: "right" },
      { id: "m_neck", label: "Neck", top: "15.8%", left: "83%", side: "right" },
      { id: "m_shoulder", label: "Shoulder", top: "21.7%", left: "83%", side: "right" },
      { id: "m_bust", label: "Bust", top: "27.5%", left: "83%", side: "right" },
      { id: "m_underbust", label: "Underbust", top: "30.8%", left: "83%", side: "right" },
      { id: "m_waist", label: "Waist", top: "38.3%", left: "83%", side: "right" },
      { id: "m_hip", label: "Hip", top: "46.7%", left: "83%", side: "right" },
      { id: "m_arm", label: "Arm", top: "33.3%", left: "16%", side: "left" },
      { id: "m_hand", label: "Hand", top: "40%", left: "16%", side: "left" },
      { id: "m_inseam", label: "Inseam", top: "70%", left: "83%", side: "right" },
      { id: "m_height", label: "Height", top: "96.7%", left: "16%", side: "left" },
      { id: "m_foot_len", label: "Foot Length", top: "96.7%", left: "50%", side: "bottom" },
      { id: "m_foot_wid", label: "Foot Width", top: "96.7%", left: "70%", side: "bottom" }
    ]
  },
  male: {
    figure: `
      <svg viewBox="0 0 300 600" style="width: 100%; height: auto;">
        <!-- Height line -->
        <line x1="20" y1="30" x2="20" y2="580" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,5" opacity="0.3"/>
        
        <!-- Male silhouette -->
        <g fill="#94a3b8" opacity="0.8">
          <!-- Head -->
          <ellipse cx="150" cy="50" rx="32" ry="40"/>
          <!-- Neck -->
          <rect x="138" y="87" width="24" height="18" rx="5"/>
          <!-- Body -->
          <path d="M 150 105 L 145 130 L 140 180 L 138 240 L 140 280 L 145 330 L 148 420 L 150 580 M 150 105 L 155 130 L 160 180 L 162 240 L 160 280 L 155 330 L 152 420 L 150 580" stroke="#94a3b8" stroke-width="32" fill="none" stroke-linecap="round"/>
          <!-- Arms -->
          <path d="M 145 130 L 95 170 L 88 250 M 155 130 L 205 170 L 212 250" stroke="#94a3b8" stroke-width="20" fill="none" stroke-linecap="round"/>
          <!-- Legs -->
          <path d="M 145 330 L 142 420 L 138 520 L 135 580 M 155 330 L 158 420 L 162 520 L 165 580" stroke="#94a3b8" stroke-width="22" fill="none" stroke-linecap="round"/>
        </g>
        
        <!-- Measurement lines -->
        <g stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.5">
          <!-- Head -->
          <line x1="118" y1="50" x2="250" y2="50"/>
          <!-- Neck -->
          <line x1="118" y1="95" x2="250" y2="95"/>
          <!-- Shoulder -->
          <line x1="95" y1="130" x2="250" y2="130"/>
          <!-- Chest -->
          <line x1="108" y1="170" x2="250" y2="170"/>
          <!-- Waist -->
          <line x1="108" y1="240" x2="250" y2="240"/>
          <!-- Hip -->
          <line x1="110" y1="285" x2="250" y2="285"/>
          <!-- Arm -->
          <line x1="50" y1="210" x2="88" y2="210"/>
          <!-- Hand -->
          <line x1="50" y1="250" x2="88" y2="250"/>
          <!-- Inseam -->
          <line x1="135" y1="420" x2="250" y2="420"/>
          <!-- Foot -->
          <line x1="115" y1="580" x2="185" y2="580"/>
        </g>
      </svg>
    `,
    points: [
      { id: "m_head", label: "Head", top: "8.3%", left: "83%", side: "right" },
      { id: "m_neck", label: "Neck", top: "15.8%", left: "83%", side: "right" },
      { id: "m_shoulder", label: "Shoulder", top: "21.7%", left: "83%", side: "right" },
      { id: "m_chest", label: "Chest", top: "28.3%", left: "83%", side: "right" },
      { id: "m_waist", label: "Waist", top: "40%", left: "83%", side: "right" },
      { id: "m_hip", label: "Hip", top: "47.5%", left: "83%", side: "right" },
      { id: "m_arm", label: "Arm", top: "35%", left: "16%", side: "left" },
      { id: "m_hand", label: "Hand", top: "41.7%", left: "16%", side: "left" },
      { id: "m_inseam", label: "Inseam", top: "70%", left: "83%", side: "right" },
      { id: "m_height", label: "Height", top: "96.7%", left: "16%", side: "left" },
      { id: "m_foot_len", label: "Foot Length", top: "96.7%", left: "50%", side: "bottom" },
      { id: "m_foot_wid", label: "Foot Width", top: "96.7%", left: "70%", side: "bottom" }
    ]
  }
};

// Utility Functions
function logln(m) {
  const t = new Date().toLocaleTimeString();
  els.log.value += `[${t}] ${m}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text, type = "info") {
  if (!text) {
    els.statusContainer.innerHTML = "";
    return;
  }
  
  const icons = {
    info: "ℹ️",
    ok: "✅",
    warn: "⚠️",
    bad: "❌"
  };
  
  els.statusContainer.innerHTML = `
    <div class="status-bar ${type}">
      <span>${icons[type] || "ℹ️"}</span>
      <span>${text}</span>
    </div>
  `;
}

function setLoading(isLoading) {
  els.btnVerify.disabled = isLoading;
  els.btnGenerate.disabled = isLoading;
  
  if (isLoading) {
    els.btnVerify.innerHTML = '<span class="spinner"></span> Verifying...';
    els.btnGenerate.innerHTML = '<span class="spinner"></span> Processing...';
  } else {
    els.btnVerify.innerHTML = '<span>✓</span> Verify Photos';
    els.btnGenerate.innerHTML = '<span>🚀</span> Generate 3D Twin';
  }
}

// Gender Selection
document.querySelectorAll(".gender-option").forEach(option => {
  option.addEventListener("click", function() {
    document.querySelectorAll(".gender-option").forEach(o => o.classList.remove("active"));
    this.classList.add("active");
    CURRENT_GENDER = this.dataset.gender;
    renderFigure();
    logln(`🔄 Switched to ${CURRENT_GENDER} figure`);
  });
});

// Render Figure with Measurement Points
function renderFigure() {
  const config = MEASUREMENT_CONFIGS[CURRENT_GENDER];
  
  els.figureWrapper.innerHTML = config.figure;
  
  // Add measurement points
  config.points.forEach((point, index) => {
    setTimeout(() => {
      const pointEl = document.createElement("div");
      pointEl.className = "measurement-point";
      pointEl.style.top = point.top;
      pointEl.style.left = point.left;
      pointEl.style.animationDelay = `${index * 0.05}s`;
      
      const dotEl = document.createElement("div");
      dotEl.className = "measurement-dot";
      dotEl.dataset.id = point.id;
      dotEl.dataset.label = point.label;
      dotEl.dataset.side = point.side;
      
      // Check if already filled
      if (state.measurements[point.id]) {
        dotEl.classList.add("filled");
      }
      
      const popupEl = document.createElement("div");
      popupEl.className = "measurement-input-popup";
      
      // Position popup based on side
      if (point.side === "right") {
        popupEl.style.left = "120%";
        popupEl.style.top = "50%";
        popupEl.style.transform = "translateY(-50%)";
      } else if (point.side === "left") {
        popupEl.style.right = "120%";
        popupEl.style.top = "50%";
        popupEl.style.transform = "translateY(-50%)";
      } else if (point.side === "bottom") {
        popupEl.style.left = "50%";
        popupEl.style.top = "120%";
        popupEl.style.transform = "translateX(-50%)";
      }
      
      popupEl.innerHTML = `
        <button class="close-btn">×</button>
        <label>${point.label} (cm)</label>
        <input type="number" step="0.1" placeholder="Enter value" value="${state.measurements[point.id] || ''}">
      `;
      
      dotEl.addEventListener("click", (e) => {
        e.stopPropagation();
        // Close other popups
        document.querySelectorAll(".measurement-input-popup.active").forEach(p => {
          if (p !== popupEl) p.classList.remove("active");
        });
        popupEl.classList.toggle("active");
      });
      
      const inputEl = popupEl.querySelector("input");
      inputEl.addEventListener("input", () => {
        const value = parseFloat(inputEl.value);
        if (!isNaN(value) && value > 0) {
          state.measurements[point.id] = value;
          dotEl.classList.add("filled");
          logln(`📏 ${point.label}: ${value} cm`);
        } else {
          delete state.measurements[point.id];
          dotEl.classList.remove("filled");
        }
      });
      
      popupEl.querySelector(".close-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        popupEl.classList.remove("active");
      });
      
      pointEl.appendChild(dotEl);
      pointEl.appendChild(popupEl);
      els.figureWrapper.appendChild(pointEl);
    }, index * 50);
  });
}

// Close popups when clicking outside
document.addEventListener("click", (e) => {
  if (!e.target.closest(".measurement-point")) {
    document.querySelectorAll(".measurement-input-popup.active").forEach(p => {
      p.classList.remove("active");
    });
  }
});

// File Upload Handling
document.querySelectorAll(".slot").forEach(slot => {
  const inp = slot.querySelector("input");
  const img = slot.querySelector(".preview");
  const role = slot.dataset.role;

  slot.addEventListener("click", (e) => {
    if (e.target !== inp) inp.click();
  });

  slot.addEventListener("dragover", (e) => {
    e.preventDefault();
    slot.style.borderColor = "var(--accent)";
    slot.style.transform = "scale(1.02)";
  });

  slot.addEventListener("dragleave", () => {
    slot.style.borderColor = "";
    slot.style.transform = "";
  });

  slot.addEventListener("drop", (e) => {
    e.preventDefault();
    slot.style.borderColor = "";
    slot.style.transform = "";
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, role, img, slot);
  });

  inp.addEventListener("change", () => {
    const f = inp.files[0];
    if (f) handleFile(f, role, img, slot);
  });
});

function handleFile(f, role, img, slot) {
  if (!f.type.startsWith("image/")) {
    logln(`❌ ${role}: Invalid file type. Please upload an image.`);
    return;
  }

  if (f.size > 10 * 1024 * 1024) {
    logln(`⚠️ ${role}: File size too large (${(f.size / 1024 / 1024).toFixed(2)}MB). Max 10MB.`);
    setStatus(`File too large: ${f.name}`, "warn");
    return;
  }

  img.src = URL.createObjectURL(f);
  img.style.display = "block";
  slot.classList.add("has-file");
  state.roleFiles[role] = f;
  
  logln(`✅ ${role.toUpperCase()}: ${f.name} uploaded (${(f.size / 1024).toFixed(2)}KB)`);
  renderThumbs();
  updateUploadStatus();
}

function updateUploadStatus() {
  const uploaded = Object.values(state.roleFiles).filter(Boolean).length;
  const required = ["front", "side", "back"].filter(role => state.roleFiles[role]).length;
  
  if (required === 3) {
    setStatus(`✅ All required photos uploaded (${uploaded}/4 total)`, "ok");
  } else if (uploaded > 0) {
    setStatus(`${required}/3 required photos uploaded`, "info");
  }
}

function renderThumbs() {
  const roles = ["front", "side", "back", "foot"];
  const uploads = roles.filter(r => state.roleFiles[r]);
  
  if (uploads.length === 0) {
    els.thumbsReview.innerHTML = '<div class="empty-state"><p>No photos uploaded yet</p></div>';
    return;
  }
  
  els.thumbsReview.innerHTML = "";
  uploads.forEach(role => {
    const f = state.roleFiles[role];
    const u = URL.createObjectURL(f);
    const d = document.createElement("div");
    d.className = "thumb";
    d.innerHTML = `
      <img src="${u}" alt="${role}">
      <div class="badge">${role}</div>
    `;
    els.thumbsReview.appendChild(d);
  });
}

// Base64 Conversion
async function fileToBase64(f) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(f);
  });
}

async function collectBase64() {
  state.b64s = [];
  const files = Object.values(state.roleFiles).filter(Boolean);
  
  if (!files.length) return;
  
  logln(`🔄 Converting ${files.length} images to base64...`);
  
  for (const f of files) {
    const b64 = await fileToBase64(f);
    state.b64s.push(b64);
  }
  
  logln(`✅ Conversion complete!`);
}

// Quality Analysis Display
function roleChip(role, report) {
  const st = (report && report.status) || "missing";
  const label = st === "ok" ? "OK" : (st === "retry" ? "Retry" : "Missing");
  const cls = st === "ok" ? "ok" : (st === "retry" ? "warn" : "bad");
  
  return `<span class="pill ${cls}">${role.toUpperCase()}: ${label}</span>`;
}

function renderQA(res) {
  state.lastQA = res;
  const rr = res.role_report || {};
  
  els.qaSummary.innerHTML = `
    ${roleChip("front", rr.front)}
    ${roleChip("side", rr.side)}
    ${roleChip("back", rr.back)}
  `;
  
  const tips = res.retake_tips || [];
  if (tips.length > 0) {
    els.retakeTipsContainer.innerHTML = `
      <div class="retake-tips">
        <h3>📋 Retake Recommendations</h3>
        <ul>
          ${tips.map(t => `<li>${t}</li>`).join('')}
        </ul>
      </div>
    `;
  } else {
    els.retakeTipsContainer.innerHTML = "";
  }
}

// Measurements Collection
function readMeasures() {
  const measurements = { ...state.measurements };
  
  // Map to API expected names
  const mapped = {
    height: measurements.m_height,
    chest: measurements.m_chest || measurements.m_bust,
    waist: measurements.m_waist,
    hips: measurements.m_hip,
    shoulder: measurements.m_shoulder,
    inseam: measurements.m_inseam,
    arm: measurements.m_arm,
    foot_length: measurements.m_foot_len,
    foot_width: measurements.m_foot_wid,
    neck: measurements.m_neck,
    head: measurements.m_head,
    hand: measurements.m_hand
  };
  
  return mapped;
}

// API Communication
async function postJSON(url, payload) {
  const headers = { "Content-Type": "application/json" };
  
  if (RUNPOD_API_KEY && url.includes('runpod.ai')) {
    headers.Authorization = `Bearer ${RUNPOD_API_KEY}`;
  }
  
  const response = await fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify({ input: payload })
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
  
  return await response.json();
}

// Verify Photos
async function onVerify() {
  try {
    setLoading(true);
    logln("━━━━━━━━━━━━━━━━━━━━━━━━");
    logln("🔍 Starting photo verification...");
    
    if (!state.roleFiles.front || !state.roleFiles.side || !state.roleFiles.back) {
      setStatus("Please upload Front, Side, and Back photos.", "warn");
      logln("⚠️ Missing required photos");
      return;
    }
    
    if (!ENDPOINT_FULL) {
      setStatus("Please enter your RunPod runsync URL.", "warn");
      logln("⚠️ No endpoint URL configured");
      return;
    }
    
    await collectBase64();
    
    const payload = {
      photos: { unordered: state.b64s },
      required_roles: ["front", "side", "back"],
      allowPartial: false
    };
    
    logln(`📤 Calling API: ${ENDPOINT_FULL}`);
    const res = await postJSON(ENDPOINT_FULL, payload);
    
    logln("📥 Response received");
    renderQA(res);
    
    if (res.ok === false) {
      setStatus("⚠️ Some photos need to be retaken.", "warn");
      logln("⚠️ Photo quality issues detected");
    } else {
      setStatus("✅ All photos passed quality check!", "ok");
      logln("✅ Photo verification successful");
    }
    
  } catch (e) {
    logln(`❌ Error: ${e.message}`);
    setStatus("Verification failed. Check the log for details.", "bad");
    console.error(e);
  } finally {
    setLoading(false);
  }
}

// Generate 3D Twin
async function onGenerate() {
  try {
    setLoading(true);
    logln("━━━━━━━━━━━━━━━━━━━━━━━━");
    logln("🚀 Starting 3D twin generation...");
    
    if (!state.roleFiles.front || !state.roleFiles.side || !state.roleFiles.back) {
      setStatus("Please upload Front, Side, and Back photos.", "warn");
      logln("⚠️ Missing required photos");
      return;
    }
    
    if (!ENDPOINT_FULL) {
      setStatus("Please enter your RunPod runsync URL.", "warn");
      logln("⚠️ No endpoint URL configured");
      return;
    }
    
    await collectBase64();
    
    const measurements = readMeasures();
    const filledCount = Object.keys(measurements).filter(k => measurements[k] !== undefined).length;
    logln(`📏 Measurements: ${filledCount} fields filled`);
    
    const payload = {
      photos: { unordered: state.b64s },
      required_roles: ["front", "side", "back"],
      allowPartial: false,
      texRes: 2048,
      preset: CURRENT_GENDER === "female" ? "female" : "male",
      ...measurements
    };
    
    logln(`📤 Calling API with ${CURRENT_GENDER} preset`);
    logln("⏳ This may take 30-60 seconds...");
    
    const res = await postJSON(ENDPOINT_FULL, payload);
    
    logln("📥 Response received");
    renderQA(res);
    
    if (res.glb_b64) {
      state.lastGLB = res.glb_b64;
      
      logln("🎨 Loading 3D model...");
      const blob = await (await fetch(`data:application/octet-stream;base64,${res.glb_b64}`)).blob();
      const url = URL.createObjectURL(blob);
      
      els.viewer.src = url;
      els.btnDownload.style.display = "inline-flex";
      
      const sizeKB = (res.glb_b64.length * 0.75 / 1024).toFixed(2);
      logln(`✅ 3D model loaded! Size: ${sizeKB}KB`);
      setStatus("🎉 Your 3D twin is ready!", "ok");
      
    } else {
      setStatus("❌ No 3D model returned. Check the log.", "bad");
      logln("❌ No model data in response");
    }
    
    if (res.log) {
      logln("📝 Server log:");
      logln(res.log);
    }
    
  } catch (e) {
    logln(`❌ Error: ${e.message}`);
    setStatus("Generation failed. Check the log for details.", "bad");
    console.error(e);
  } finally {
    setLoading(false);
  }
}

// Download GLB
function onDownload() {
  if (!state.lastGLB) {
    logln("⚠️ No model to download");
    return;
  }
  
  const ts = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
  const a = document.createElement("a");
  a.href = `data:application/octet-stream;base64,${state.lastGLB}`;
  a.download = `digital-twin-${CURRENT_GENDER}-${ts}.glb`;
  a.click();
  
  logln(`✅ Model downloaded: digital-twin-${CURRENT_GENDER}-${ts}.glb`);
}

// Viewer Controls
function resetViewer() {
  if (els.viewer.resetTurntableRotation) {
    els.viewer.resetTurntableRotation();
  }
  els.viewer.cameraOrbit = "0deg 75deg 105%";
  els.turn.value = 0;
  logln("🔄 Viewer reset to default position");
}

function updateRotation() {
  const degrees = Number(els.turn.value);
  const radians = degrees * Math.PI / 180;
  els.viewer.turntableRotation = radians;
  els.viewer.autoRotate = false;
}

// API Configuration Sync
function syncApi() {
  ENDPOINT_FULL = els.endpointFull.value.trim();
  RUNPOD_API_KEY = els.apiKey.value.trim();
  
  if (!ENDPOINT_FULL) {
    setStatus("⚠️ Please enter your RunPod runsync URL to get started.", "warn");
  } else {
    setStatus("✅ API endpoint configured. Ready to process!", "ok");
    logln(`🔗 Endpoint configured: ${ENDPOINT_FULL.substring(0, 50)}...`);
  }
}

// Event Listeners
els.btnVerify.addEventListener("click", onVerify);
els.btnGenerate.addEventListener("click", onGenerate);
els.btnDownload.addEventListener("click", onDownload);
els.btnReset.addEventListener("click", resetViewer);
els.turn.addEventListener("input", updateRotation);
els.endpointFull.addEventListener("input", syncApi);
els.apiKey.addEventListener("input", syncApi);

// Initialize
renderFigure();
syncApi();
logln("🎯 Digital Twin Generator initialized");
logln("📌 Select gender, upload photos, click measurement dots");
