document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const modelStatusText = document.getElementById("modelStatusText");
    const modelArchBadge = document.getElementById("modelArchBadge");
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const sampleGrid = document.getElementById("sampleGrid");
    const previewBox = document.getElementById("previewBox");
    const imagePreview = document.getElementById("imagePreview");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const analyzeSpinner = document.getElementById("analyzeSpinner");
    const resultsSection = document.getElementById("resultsSection");

    // Camera elements
    const webcamVideo = document.getElementById("webcamVideo");
    const webcamCanvas = document.getElementById("webcamCanvas");
    const startCameraBtn = document.getElementById("startCameraBtn");
    const captureBtn = document.getElementById("captureBtn");
    let stream = null;

    // Result elements
    const gradeBanner = document.getElementById("gradeBanner");
    const gradeBadge = document.getElementById("gradeBadge");
    const gradeCommodity = document.getElementById("gradeCommodity");
    const gradeDesc = document.getElementById("gradeDesc");
    const gradeCondition = document.getElementById("gradeCondition");
    const freshnessValue = document.getElementById("freshnessValue");
    const metricConfidence = document.getElementById("metricConfidence");
    const confProgress = document.getElementById("confProgress");
    const metricDefect = document.getElementById("metricDefect");
    const defectProgress = document.getElementById("defectProgress");
    const metricShelfLife = document.getElementById("metricShelfLife");
    const actionRecommendation = document.getElementById("actionRecommendation");
    const xaiImage = document.getElementById("xaiImage");
    const xaiLabel = document.getElementById("xaiLabel");
    const viewToggles = document.querySelectorAll(".toggle-btn");
    const probList = document.getElementById("probList");
    const reportTime = document.getElementById("reportTime");

    let currentImageData = null;
    let lastResult = null;

    // 1. Fetch Model Status
    fetch("/api/model-info")
        .then(res => res.json())
        .then(data => {
            if (data.status === "loaded") {
                modelStatusText.textContent = `Online (${data.num_classes} Classes)`;
                modelStatusText.style.color = "#10b981";
                if (data.metrics && data.metrics.best_val_accuracy) {
                    modelArchBadge.textContent = `Accuracy: ${data.metrics.best_val_accuracy.toFixed(1)}% | ${data.input_resolution}`;
                }
            } else {
                modelStatusText.textContent = "Model Initializing...";
                modelStatusText.style.color = "#f59e0b";
            }
        })
        .catch(() => {
            modelStatusText.textContent = "Offline";
            modelStatusText.style.color = "#ef4444";
        });

    // 2. Fetch Preset Samples
    fetch("/api/samples")
        .then(res => res.json())
        .then(data => {
            if (data.samples && data.samples.length > 0) {
                sampleGrid.innerHTML = "";
                data.samples.forEach(sample => {
                    const chip = document.createElement("button");
                    chip.className = "sample-chip";
                    const isFresh = sample.includes("fresh");
                    const icon = isFresh ? "🟢" : "🔴";
                    const label = sample.replace("_test.png", "").replace("_", " ").toUpperCase();
                    chip.innerHTML = `<span>${icon}</span> <span>${label}</span>`;
                    chip.onclick = () => loadSample(sample);
                    sampleGrid.appendChild(chip);
                });
            } else {
                sampleGrid.innerHTML = "<p class='section-hint'>No sample images found.</p>";
            }
        });

    function loadSample(sampleName) {
        fetch(`/api/sample/${sampleName}`)
            .then(res => res.json())
            .then(data => {
                setImage(data.image);
            });
    }

    // 3. Tab Switching
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");

            if (btn.dataset.tab !== "webcam-tab" && stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                startCameraBtn.textContent = "Start Camera";
                captureBtn.disabled = true;
            }
        });
    });

    // 4. Dropzone & File Input
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            setImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    function setImage(base64Uri) {
        currentImageData = base64Uri;
        imagePreview.src = base64Uri;
        previewBox.style.display = "block";
        previewBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // 5. Webcam Support
    startCameraBtn.addEventListener("click", async () => {
        if (!stream) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
                webcamVideo.srcObject = stream;
                startCameraBtn.textContent = "Stop Camera";
                captureBtn.disabled = false;
            } catch (err) {
                alert("Unable to access camera: " + err.message);
            }
        } else {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            startCameraBtn.textContent = "Start Camera";
            captureBtn.disabled = true;
        }
    });

    captureBtn.addEventListener("click", () => {
        if (!stream) return;
        webcamCanvas.width = webcamVideo.videoWidth || 320;
        webcamCanvas.height = webcamVideo.videoHeight || 240;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
        const dataUrl = webcamCanvas.toDataURL("image/png");
        setImage(dataUrl);
    });

    // 6. Run Quality Analysis
    analyzeBtn.addEventListener("click", async () => {
        if (!currentImageData) return;

        analyzeBtn.disabled = true;
        analyzeSpinner.style.display = "inline-block";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: currentImageData })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Analysis failed");
            }

            const data = await response.json();
            lastResult = data;
            renderResults(data);
        } catch (err) {
            alert("Error running analysis: " + err.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeSpinner.style.display = "none";
        }
    });

    function renderResults(data) {
        const report = data.report;
        resultsSection.style.display = "flex";
        reportTime.textContent = new Date().toLocaleTimeString();

        // Grade Banner Styling
        gradeBadge.textContent = report.grade;
        gradeBadge.style.backgroundColor = report.status_color;
        gradeBadge.style.boxShadow = `0 4px 14px ${report.status_color}66`;

        gradeBanner.style.borderColor = report.status_color;
        gradeCommodity.textContent = report.commodity;
        gradeDesc.textContent = report.grade_description;
        gradeCondition.textContent = `Condition: ${report.condition}`;
        gradeCondition.style.color = report.status_color;

        freshnessValue.textContent = `${report.freshness_score.toFixed(1)}%`;
        freshnessValue.style.color = report.status_color;

        // Metrics
        const confPct = (report.confidence * 100).toFixed(1);
        metricConfidence.textContent = `${confPct}%`;
        confProgress.style.width = `${confPct}%`;

        metricDefect.textContent = `${report.defect_percentage.toFixed(1)}%`;
        defectProgress.style.width = `${Math.min(100, report.defect_percentage)}%`;

        metricShelfLife.textContent = report.estimated_shelf_life;
        actionRecommendation.textContent = report.recommended_action;

        // Grad-CAM Visualizer
        xaiImage.src = data.gradcam_heatmap;
        xaiLabel.textContent = "Grad-CAM Defect Overlay";

        // Reset toggles to overlay
        viewToggles.forEach(t => t.classList.toggle("active", t.dataset.view === "overlay"));

        // Probabilities
        probList.innerHTML = "";
        const sortedProbs = Object.entries(report.probabilities).sort((a, b) => b[1] - a[1]);
        sortedProbs.forEach(([className, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const item = document.createElement("div");
            item.className = "prob-item";
            item.innerHTML = `
                <span class="prob-name">${className.replace("_", " ")}</span>
                <div class="prob-bar-wrapper">
                    <div class="prob-bar" style="width: ${pct}%;"></div>
                </div>
                <span class="prob-value">${pct}%</span>
            `;
            probList.appendChild(item);
        });

        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    // 7. XAI View Toggle
    viewToggles.forEach(btn => {
        btn.addEventListener("click", () => {
            if (!lastResult) return;
            viewToggles.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const mode = btn.dataset.view;
            if (mode === "overlay") {
                xaiImage.src = lastResult.gradcam_heatmap;
                xaiLabel.textContent = "Grad-CAM Overlay";
            } else if (mode === "original") {
                xaiImage.src = lastResult.original_image;
                xaiLabel.textContent = "Original Input";
            } else if (mode === "heatmap") {
                xaiImage.src = lastResult.gradcam_heatmap;
                xaiLabel.textContent = "Attention Heatmap";
            }
        });
    });
});
