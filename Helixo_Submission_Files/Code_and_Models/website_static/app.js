/* ═══════════════════════════════════════════════════════════
   Helixo V2 — Client-Side YOLOv8 Inference using ONNX Runtime Web
   
   Model:  YOLOv8-nano (best.onnx)
   Input:  [1, 3, 224, 224] (BCHW, float32, 0-1 normalized)
   Output: [1, 6, 1029]  →  6 = [x, y, w, h, helmet_conf, no_helmet_conf]
   ═══════════════════════════════════════════════════════════ */

const MODEL_URL = "./model/best.onnx";
const INPUT_SIZE = 224;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const CLASS_NAMES = ["helmet", "no_helmet"];
const CLASS_COLORS = { helmet: "#10b981", no_helmet: "#ef4444" };

let session = null;

// ─── DOM Elements ─────────────────────────────────────────
const fileInput = document.getElementById("fileInput");
const uploadZone = document.getElementById("uploadZone");
const fileCountBadge = document.getElementById("fileCountBadge");
const btnDetect = document.getElementById("btnDetect");
const btnWebcam = document.getElementById("btnWebcam");
const processingBar = document.getElementById("processingBar");
const processingText = document.getElementById("processingText");
const placeholder = document.getElementById("placeholder");
const resultsContainer = document.getElementById("resultsContainer");
const webcamBox = document.getElementById("webcamBox");
const webcamVideo = document.getElementById("webcamVideo");
const btnCapture = document.getElementById("btnCapture");
const btnCloseWc = document.getElementById("btnCloseWc");
const procCanvas = document.getElementById("processingCanvas");

let currentFiles = [];
let webcamStream = null;

// ─── Load ONNX Model ─────────────────────────────────────
async function loadModel() {
    const loadingScreen = document.getElementById("modelLoadingScreen");
    const loaderStatus = document.getElementById("loaderStatus");

    try {
        loaderStatus.textContent = "Initializing ONNX Runtime...";
        
        // Use WASM backend
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";

        loaderStatus.textContent = "Downloading YOLOv8-nano model (12 MB)...";
        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
        });

        loaderStatus.textContent = "Model loaded! ✅";
        
        // Hide loading screen
        setTimeout(() => {
            loadingScreen.classList.add("hidden");
            setTimeout(() => loadingScreen.remove(), 500);
        }, 600);

    } catch (err) {
        loaderStatus.textContent = "❌ Error: " + err.message;
        console.error("Model load error:", err);
    }
}

// ─── Image Preprocessing ─────────────────────────────────
function preprocessImage(imgElement) {
    const ctx = procCanvas.getContext("2d");
    procCanvas.width = INPUT_SIZE;
    procCanvas.height = INPUT_SIZE;

    // Letterbox resize (maintain aspect ratio with gray padding)
    const imgW = imgElement.naturalWidth || imgElement.width;
    const imgH = imgElement.naturalHeight || imgElement.height;
    const scale = Math.min(INPUT_SIZE / imgW, INPUT_SIZE / imgH);
    const newW = Math.round(imgW * scale);
    const newH = Math.round(imgH * scale);
    const padX = (INPUT_SIZE - newW) / 2;
    const padY = (INPUT_SIZE - newH) / 2;

    // Gray background (114/255 is YOLO's default padding)
    ctx.fillStyle = `rgb(114, 114, 114)`;
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(imgElement, padX, padY, newW, newH);

    // Get pixel data
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data;

    // Convert to BCHW float32 tensor, normalized 0-1
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        float32Data[i] = pixels[i * 4] / 255.0;                     // R
        float32Data[INPUT_SIZE * INPUT_SIZE + i] = pixels[i * 4 + 1] / 255.0;  // G
        float32Data[2 * INPUT_SIZE * INPUT_SIZE + i] = pixels[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor("float32", float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    return { tensor, scale, padX, padY, imgW, imgH };
}

// ─── YOLOv8 Post-Processing ──────────────────────────────
function postprocess(outputData, scale, padX, padY, imgW, imgH) {
    // outputData shape: [1, 6, 1029] → transpose to [1029, 6]
    const numClasses = CLASS_NAMES.length;  // 2
    const numBoxes = outputData.length / (4 + numClasses);  // 1029

    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
        // Read values (column-major: outputData is [6][1029])
        const cx = outputData[0 * numBoxes + i];
        const cy = outputData[1 * numBoxes + i];
        const w  = outputData[2 * numBoxes + i];
        const h  = outputData[3 * numBoxes + i];

        // Find best class
        let maxConf = 0;
        let classId = 0;
        for (let c = 0; c < numClasses; c++) {
            const conf = outputData[(4 + c) * numBoxes + i];
            if (conf > maxConf) {
                maxConf = conf;
                classId = c;
            }
        }

        if (maxConf < CONF_THRESHOLD) continue;

        // Convert from letterboxed coords to original image coords
        const x1 = ((cx - w / 2) - padX) / scale;
        const y1 = ((cy - h / 2) - padY) / scale;
        const x2 = ((cx + w / 2) - padX) / scale;
        const y2 = ((cy + h / 2) - padY) / scale;

        // Clamp to image bounds
        detections.push({
            x1: Math.max(0, x1),
            y1: Math.max(0, y1),
            x2: Math.min(imgW, x2),
            y2: Math.min(imgH, y2),
            confidence: maxConf,
            classId: classId,
            className: CLASS_NAMES[classId],
        });
    }

    // Apply NMS
    return nms(detections, IOU_THRESHOLD);
}

function nms(boxes, iouThreshold) {
    // Sort by confidence descending
    boxes.sort((a, b) => b.confidence - a.confidence);

    const kept = [];
    const suppressed = new Set();

    for (let i = 0; i < boxes.length; i++) {
        if (suppressed.has(i)) continue;
        kept.push(boxes[i]);

        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed.has(j)) continue;
            if (iou(boxes[i], boxes[j]) > iouThreshold) {
                suppressed.add(j);
            }
        }
    }
    return kept;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
}

// ─── Run Inference on One Image ───────────────────────────
async function runInference(imgElement) {
    const { tensor, scale, padX, padY, imgW, imgH } = preprocessImage(imgElement);

    const inputName = session.inputNames[0];
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    const outputData = results[outputName].data;

    return postprocess(outputData, scale, padX, padY, imgW, imgH);
}

// ─── Draw Bounding Boxes on Canvas ───────────────────────
function drawResults(imgElement, detections) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const w = imgElement.naturalWidth || imgElement.width;
    const h = imgElement.naturalHeight || imgElement.height;
    canvas.width = w;
    canvas.height = h;

    ctx.drawImage(imgElement, 0, 0, w, h);

    detections.forEach((det) => {
        const color = CLASS_COLORS[det.className] || "#ffffff";
        const bx = det.x1;
        const by = det.y1;
        const bw = det.x2 - det.x1;
        const bh = det.y2 - det.y1;

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.round(w / 150));
        ctx.strokeRect(bx, by, bw, bh);

        // Label background
        const label = `${det.className.replace("_", " ")} ${Math.round(det.confidence * 100)}%`;
        const fontSize = Math.max(12, Math.round(w / 30));
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textW = ctx.measureText(label).width;
        const labelH = fontSize + 8;

        ctx.fillStyle = color;
        ctx.fillRect(bx, by - labelH, textW + 12, labelH);

        // Label text
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, bx + 6, by - 5);
    });

    return canvas;
}

// ─── Build Result Card HTML ──────────────────────────────
function buildResultCard(filename, canvas, detections) {
    const card = document.createElement("div");
    card.className = "card result-card";

    const helmetCount = detections.filter((d) => d.className === "helmet").length;
    const noHelmetCount = detections.filter((d) => d.className === "no_helmet").length;

    // Summary
    let summaryText, summaryClass;
    if (detections.length === 0) {
        summaryText = "⚠️ No Detection — Camera blocked or no person";
        summaryClass = "yellow";
    } else if (helmetCount > 0 && noHelmetCount === 0) {
        summaryText = "✅ Helmet Detected — Ignition ENABLED";
        summaryClass = "green";
    } else if (noHelmetCount > 0 && helmetCount === 0) {
        summaryText = "❌ No Helmet — Ignition BLOCKED";
        summaryClass = "red";
    } else {
        summaryText = "⚠️ Mixed — Helmet + No-helmet detected";
        summaryClass = "orange";
    }

    // Detection items
    let detHTML = "";
    if (detections.length === 0) {
        detHTML = `<div class="det-item"><span class="dn"><span class="dd" style="background:var(--yellow)"></span>⚠️ No objects</span><span class="dc medium">0%</span></div>`;
    } else {
        detections.forEach((d) => {
            const pct = Math.round(d.confidence * 100);
            const confClass = pct >= 80 ? "high" : pct >= 50 ? "medium" : "low";
            const emoji = d.className === "helmet" ? "🪖" : "⛑️";
            detHTML += `<div class="det-item"><span class="dn"><span class="dd ${d.className}"></span>${emoji} ${d.className.replace("_", " ")}</span><span class="dc ${confClass}">${pct}%</span></div>`;
        });
    }

    card.innerHTML = `
        <div class="result-filename">📄 ${filename}</div>
        <div class="status-banner ${summaryClass}">${summaryText}</div>
        <div class="result-canvas-wrap"></div>
        <div class="stats-row">
            <div class="stat-box"><div class="sv">${detections.length}</div><div class="sl">Detections</div></div>
            <div class="stat-box"><div class="sv" style="color:var(--green)">${helmetCount}</div><div class="sl">Helmet</div></div>
            <div class="stat-box"><div class="sv" style="color:var(--red)">${noHelmetCount}</div><div class="sl">No Helmet</div></div>
        </div>
        ${detHTML}
    `;

    // Insert canvas
    card.querySelector(".result-canvas-wrap").appendChild(canvas);
    return card;
}

// ─── Process All Selected Files ──────────────────────────
async function processAllFiles() {
    if (!session || currentFiles.length === 0) return;

    btnDetect.disabled = true;
    processingBar.style.display = "flex";
    placeholder.style.display = "none";
    resultsContainer.innerHTML = "";

    for (let i = 0; i < currentFiles.length; i++) {
        processingText.textContent = `Processing ${i + 1} of ${currentFiles.length}...`;

        const file = currentFiles[i];
        const img = await loadImageFromFile(file);
        const detections = await runInference(img);
        const canvas = drawResults(img, detections);
        const card = buildResultCard(file.name, canvas, detections);
        resultsContainer.appendChild(card);
    }

    processingBar.style.display = "none";
    btnDetect.disabled = false;
}

function loadImageFromFile(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = URL.createObjectURL(file);
    });
}

// ─── Event Listeners ─────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFiles(fileInput.files);
});

function handleFiles(files) {
    currentFiles = Array.from(files);
    fileCountBadge.textContent = currentFiles.length + (currentFiles.length === 1 ? " image selected" : " images selected");
    fileCountBadge.style.display = "inline-block";
    btnDetect.disabled = false;
    if (webcamStream) closeWebcam();
}

btnDetect.addEventListener("click", processAllFiles);

// ─── Webcam ──────────────────────────────────────────────
btnWebcam.addEventListener("click", async () => {
    if (webcamStream) { closeWebcam(); return; }
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } });
        webcamVideo.srcObject = webcamStream;
        webcamBox.style.display = "block";
    } catch (err) {
        alert("Webcam access denied: " + err.message);
    }
});

btnCloseWc.addEventListener("click", closeWebcam);

function closeWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach((t) => t.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;
    webcamBox.style.display = "none";
}

btnCapture.addEventListener("click", async () => {
    if (!webcamStream) return;
    const c = document.createElement("canvas");
    c.width = webcamVideo.videoWidth;
    c.height = webcamVideo.videoHeight;
    c.getContext("2d").drawImage(webcamVideo, 0, 0);
    c.toBlob((blob) => {
        const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
        handleFiles([file]);
        processAllFiles();
    }, "image/jpeg", 0.9);
});

// ─── Init ────────────────────────────────────────────────
loadModel();
