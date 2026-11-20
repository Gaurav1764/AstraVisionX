/* =========================================================================
   AstraVisionX — Modern Light Dashboard Frontend Logic
   Clean card layout, dynamic thumbnail cropping, synchronized semantic 
   confidence colors, class-conditioned per-detection Grad-CAM/SHAP/LIME,
   and bottom chat bar.
   ========================================================================= */

(function () {
  "use strict";

  // --- State ---------------------------------------------------------------
  let selectedFile = null;
  let currentSessionId = null;
  let analysisData = null;
  let activeDetectionIndex = null;
  let naturalImgW = 0;
  let naturalImgH = 0;
  let currentDetections = [];
  let croppedThumbnails = [];
  let activeXaiMethod = "gradcam";
  let xaiCache = {}; // maps `global_method` or `det_INDEX_method` -> result

  // --- Filter State ---
  let activeFilterCategories = new Set(["all"]);
  let searchQuery = "";
  let minConfidenceThreshold = 10;
  let isDimMode = true; // true = Dim (~15% opacity), false = Hide non-matching boxes
  let categoryCounts = {};
  let demographicEnabled = false;
  let demographicCounts = {};
  let filterConfig = {};

  // --- DOM Refs: Navigation & Upload ---------------------------------------
  const uploadZone       = document.getElementById("uploadZone");
  const fileInput        = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage     = document.getElementById("previewImage");
  const fileNameEl       = document.getElementById("fileName");
  const analyzeBtn       = document.getElementById("analyzeBtn");
  const btnUploadNew     = document.getElementById("btnUploadNew");
  const downloadPdfBtn   = document.getElementById("downloadPdfBtn");
  const fullscreenBtn    = document.getElementById("fullscreenBtn");
  const loadingOverlay   = document.getElementById("loadingOverlay");
  const errorBanner      = document.getElementById("errorBanner");
  const brandLogoHome    = document.getElementById("brandLogoHome");

  // --- DOM Refs: Workspace & Panels ----------------------------------------
  const reportChatLayout        = document.getElementById("reportChatLayout");
  const aiSummaryStats          = document.getElementById("aiSummaryStats");
  const detectedCountBadge      = document.getElementById("detectedCountBadge");
  const resetViewBtn            = document.getElementById("resetViewBtn");
  const detectionsBody          = document.getElementById("detectionsBody");
  const classificationsBody     = document.getElementById("classificationsBody");
  const gradcamPills            = document.getElementById("gradcamPills");
  const confidenceBreakdownBox  = document.getElementById("confidenceBreakdownBox");

  // --- DOM Refs: Category Filter Bar ---
  const detectionFilterBar          = document.getElementById("detectionFilterBar");
  const filterChipsRow              = document.getElementById("filterChipsRow");
  const filterSearchInput           = document.getElementById("filterSearchInput");
  const clearSearchBtn              = document.getElementById("clearSearchBtn");
  const filterConfidenceSlider      = document.getElementById("filterConfidenceSlider");
  const confSliderValue             = document.getElementById("confSliderValue");
  const dimHideToggle               = document.getElementById("dimHideToggle");
  const dimHideLabel                = document.getElementById("dimHideLabel");
  const demographicDisclaimerBanner = document.getElementById("demographicDisclaimerBanner");
  const filterEmptyState            = document.getElementById("filterEmptyState");
  const emptyClearFiltersBtn        = document.getElementById("emptyClearFiltersBtn");
  const detectionsTableContainer    = document.getElementById("detectionsTableContainer");

  // --- DOM Refs: Image Viewer & XAI ----------------------------------------
  const imgOriginal             = document.getElementById("imgOriginal");
  const imgGradcamOverlay       = document.getElementById("imgGradcamOverlay");
  const overlayToggle           = document.getElementById("overlayToggle");
  const zoomLayer               = document.getElementById("zoomLayer");
  const bboxOverlayContainer    = document.getElementById("bboxOverlayContainer");
  const xaiMethodTabs           = document.querySelectorAll(".xai-tab");
  const currentMethodBadge      = document.getElementById("currentMethodBadge");
  const activePresetBadge       = document.getElementById("activePresetBadge");
  const presetSelect            = document.getElementById("presetSelect");
  const btnScoreInfo            = document.getElementById("btnScoreInfo");
  const btnCloseScoreInfo       = document.getElementById("btnCloseScoreInfo");
  const confidenceInfoCollapsible = document.getElementById("confidenceInfoCollapsible");

  // --- DOM Refs: Chat Bar & Drawer -----------------------------------------
  const chatFeedDrawer      = document.getElementById("chatFeedDrawer");
  const chatBrandTrigger    = document.getElementById("chatBrandTrigger");
  const chatDrawerClose     = document.getElementById("chatDrawerClose");
  const chatMessages        = document.getElementById("chatMessages");
  const chatForm            = document.getElementById("chatForm");
  const chatInput           = document.getElementById("chatInput");
  const chatSendBtn         = document.getElementById("chatSendBtn");
  const chatTyping          = document.getElementById("chatTyping");
  const chatSessionIdEl     = document.getElementById("chatSessionId");
  const micBtn              = document.getElementById("micBtn");
  const voiceToggleBtn      = document.getElementById("voiceToggleBtn");
  const suggestedChipsRow   = document.getElementById("suggestedChipsRow");

  // --- Natural Voice Selection & Speech Setup ------------------------------
  let isVoiceEnabled = true;
  let recognition = null;
  let isRecording = false;
  let selectedNaturalVoice = null;

  function loadBestVoice() {
    if (!('speechSynthesis' in window)) return;
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return;

    // Prioritized list of high-definition, neural, natural voices across Edge, Chrome, Safari, Firefox
    const preferredVoicePatterns = [
      /Jenny.*Online.*Natural/i,
      /Aria.*Online.*Natural/i,
      /Guy.*Online.*Natural/i,
      /Emma.*Online.*Natural/i,
      /Google.*US.*English/i,
      /Google.*UK.*English.*Female/i,
      /Samantha/i,
      /Ava/i,
      /Nora/i,
      /Microsoft.*Zira/i,
      /Microsoft.*David/i,
      /en-US/i,
      /en-GB/i,
    ];

    for (const pattern of preferredVoicePatterns) {
      const match = voices.find(v => pattern.test(v.name) || pattern.test(v.voiceURI));
      if (match) {
        selectedNaturalVoice = match;
        break;
      }
    }

    if (!selectedNaturalVoice) {
      selectedNaturalVoice = voices.find(v => v.lang.startsWith("en")) || voices[0];
    }
  }

  if ('speechSynthesis' in window) {
    loadBestVoice();
    window.speechSynthesis.onvoiceschanged = loadBestVoice;
  }

  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isRecording = true;
      micBtn.classList.add("recording");
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      chatInput.value = transcript;
      chatForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    };

    recognition.onerror = () => {
      isRecording = false;
      micBtn.classList.remove("recording");
    };

    recognition.onend = () => {
      isRecording = false;
      micBtn.classList.remove("recording");
    };
  }

  function cleanTextForSpeech(text) {
    if (!text) return "";
    return text
      .replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F900}-\u{1F9FF}\u{1F1E0}-\u{1F1FF}]/gu, "") // Emojis
      .replace(/```[\s\S]*?```/g, "") // Code blocks
      .replace(/`([^`]+)`/g, "$1")    // Inline code
      .replace(/https?:\/\/\S+/g, "") // URLs
      .replace(/(\*\*|__)(.*?)\1/g, "$2") // Bold
      .replace(/(\*|_)(.*?)\1/g, "$2")   // Italic
      .replace(/^#+\s+/gm, "")         // Headers
      .replace(/^[\*\-•]\s+/gm, "")    // Bullets
      .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1") // Links
      .replace(/⚠️|✅|ℹ️|🔍|🔥|📊|✨|💬|👤/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function speakText(text) {
    if (!isVoiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const cleanText = cleanTextForSpeech(text);
    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    if (!selectedNaturalVoice) loadBestVoice();
    if (selectedNaturalVoice) {
      utterance.voice = selectedNaturalVoice;
    }
    utterance.lang = 'en-US';
    utterance.rate = 1.03;  // Natural, engaging conversational cadence
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    window.speechSynthesis.speak(utterance);
  }

  // ========================================================================
  //  CONFIDENCE TIER HELPER (SYNCHRONIZED COLOR SYSTEM)
  // ========================================================================
  function getConfidenceTier(confidenceValue) {
    const pct = confidenceValue > 1 ? confidenceValue : confidenceValue * 100;
    if (pct >= 80) {
      return { tier: 'high', label: 'High', color: '#10B981', bg: '#ECFDF5', text: '#047857' };
    } else if (pct >= 65) {
      return { tier: 'med-high', label: 'Good', color: '#3B82F6', bg: '#EFF6FF', text: '#1D4ED8' };
    } else if (pct >= 50) {
      return { tier: 'med', label: 'Moderate', color: '#F59E0B', bg: '#FFFBEB', text: '#B45309' };
    } else {
      return { tier: 'low', label: 'Uncertain', color: '#EF4444', bg: '#FEF2F2', text: '#B91C1C' };
    }
  }

  // ========================================================================
  //  UPLOAD & FILE SELECTION
  // ========================================================================
  if (brandLogoHome) {
    brandLogoHome.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  uploadZone.addEventListener("click", (e) => {
    if (e.target.closest('#analyzeBtn')) return;
    fileInput.click();
  });

  if (btnUploadNew) {
    btnUploadNew.addEventListener("click", () => {
      fileInput.click();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("Please upload a valid image file (JPEG, PNG, or WebP).");
      return;
    }
    selectedFile = file;
    hideError();
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.classList.add("visible");
      fileNameEl.textContent = file.name;
    };
    reader.readAsDataURL(file);
    reportChatLayout.classList.remove("visible");
  }

  // ========================================================================
  //  ANALYZE ACTION
  // ========================================================================
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingOverlay.classList.add("visible");
    hideError();
    reportChatLayout.classList.remove("visible");

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (presetSelect && presetSelect.value) {
      formData.append("preset", presetSelect.value);
    }

    try {
      const res = await fetch("/analyze", { method: "POST", body: formData });
      const data = await res.json();

      if (!data.success) throw new Error(data.error || "Analysis failed.");

      analysisData = data;
      currentSessionId = data.session_id;
      xaiCache = {};
      
      // Store filter metadata from backend
      categoryCounts = data.category_counts || {};
      demographicEnabled = !!data.demographic_enabled;
      demographicCounts = data.demographic_counts || {};
      filterConfig = data.filter_config || {};
      
      // Reset filter controls to default
      activeFilterCategories = new Set(["all"]);
      searchQuery = "";
      if (filterSearchInput) filterSearchInput.value = "";
      if (clearSearchBtn) clearSearchBtn.style.display = "none";
      if (filterConfidenceSlider) {
        filterConfidenceSlider.value = 10;
        if (confSliderValue) confSliderValue.textContent = "10%";
      }
      minConfidenceThreshold = 10;
      
      // Store global Grad-CAM
      if (data.gradcam_image) {
        xaiCache["global_gradcam"] = {
          overlay_b64: data.gradcam_image,
          method: "gradcam",
          target_class_name: data.gradcam_info?.target_class_name || "Scene",
          confidence: data.gradcam_info?.confidence
        };
      }

      // Store pre-computed per-detection Grad-CAMs
      if (data.detections) {
        data.detections.forEach((d, idx) => {
          if (d.gradcam_image) {
            xaiCache[`det_${idx}_gradcam`] = {
              overlay_b64: d.gradcam_image,
              method: "gradcam",
              target_class_name: d.class_name?.replace(" (⚠️ Uncertain match)", ""),
              confidence: d.confidence
            };
          }
        });
      }

      renderReport(data);
      resetChat(data.nl_summary);
      if (downloadPdfBtn) downloadPdfBtn.disabled = false;
    } catch (err) {
      showError(err.message || "Something went wrong. Please try again.");
    } finally {
      loadingOverlay.classList.remove("visible");
      analyzeBtn.disabled = false;
    }
  });

  // ========================================================================
  //  DYNAMIC THUMBNAIL CROPPING GENERATOR
  // ========================================================================
  function generateCroppedThumbnails(imgElement, detections) {
    const thumbnails = [];
    if (!detections || !detections.length) return thumbnails;

    const imgW = imgElement.naturalWidth || 1;
    const imgH = imgElement.naturalHeight || 1;

    const offscreenCanvas = document.createElement("canvas");
    const ctx = offscreenCanvas.getContext("2d");

    detections.forEach((det) => {
      try {
        const [x1, y1, x2, y2] = det.bbox;
        const boxW = Math.max(x2 - x1, 1);
        const boxH = Math.max(y2 - y1, 1);

        const padX = boxW * 0.15;
        const padY = boxH * 0.15;
        const cropX = Math.max(0, x1 - padX);
        const cropY = Math.max(0, y1 - padY);
        const cropW = Math.min(imgW - cropX, boxW + padX * 2);
        const cropH = Math.min(imgH - cropY, boxH + padY * 2);

        offscreenCanvas.width = 80;
        offscreenCanvas.height = 80;

        ctx.clearRect(0, 0, 80, 80);
        ctx.drawImage(imgElement, cropX, cropY, cropW, cropH, 0, 0, 80, 80);
        thumbnails.push(offscreenCanvas.toDataURL("image/jpeg", 0.85));
      } catch (e) {
        thumbnails.push("");
      }
    });

    return thumbnails;
  }

  // ========================================================================
  //  REPORT RENDERER
  // ========================================================================
  function renderReport(data) {
    resetZoom();
    currentDetections = data.detections || [];

    // Meta & Count Badge
    const count = currentDetections.length;
    detectedCountBadge.textContent = `${count} ${count === 1 ? 'Object' : 'Objects'}`;
    chatSessionIdEl.textContent = `Session: ${data.session_id.substring(0, 8)}…`;

    // Active Preset Badge in YOLO Detections Card Header
    if (activePresetBadge) {
      if (data.preset_info?.badge_label) {
        activePresetBadge.textContent = data.preset_info.badge_label;
      }
    }
    if (presetSelect && data.preset_info?.preset) {
      presetSelect.value = data.preset_info.preset;
    }

    // Summary Stats Banner
    const uncertainDetections = currentDetections.filter(d => d.confidence < 0.6).length;
    const topClass = data.classifications && data.classifications.length ? data.classifications[0].class_name : "N/A";
    
    aiSummaryStats.innerHTML = `
      <div class="stat-chip">
        <span class="stat-chip__label">Detected:</span>
        <span class="stat-chip__value">${count} Objects</span>
      </div>
      <div class="stat-chip">
        <span class="stat-chip__label">Uncertain:</span>
        <span class="stat-chip__value" style="color:${uncertainDetections > 0 ? 'var(--conf-med-text)' : 'inherit'}">${uncertainDetections}</span>
      </div>
      <div class="stat-chip">
        <span class="stat-chip__label">ViT Scene:</span>
        <span class="stat-chip__value">${esc(topClass)}</span>
      </div>
    `;

    // Setup Images
    imgOriginal.src = "data:image/jpeg;base64," + data.original_image;
    
    if (data.gradcam_image) {
      imgGradcamOverlay.src = "data:image/jpeg;base64," + data.gradcam_image;
      overlayToggle.disabled = false;
    } else {
      imgGradcamOverlay.src = "";
      overlayToggle.disabled = true;
    }
    overlayToggle.checked = false;
    imgGradcamOverlay.classList.remove("active");
    bboxOverlayContainer.classList.remove("dimmed");
    setXaiTabActive("gradcam");

    // Once base image is loaded, generate thumbnails and bounding boxes
    const onBaseImageReady = () => {
      naturalImgW = imgOriginal.naturalWidth || 1;
      naturalImgH = imgOriginal.naturalHeight || 1;
      croppedThumbnails = generateCroppedThumbnails(imgOriginal, currentDetections);
      renderFilterChips();
      renderDetectionsTableAndBoxes(currentDetections, naturalImgW, naturalImgH);
      applyFilter();
      updateSuggestedChips(currentDetections);
    };

    if (imgOriginal.complete && imgOriginal.naturalWidth > 0) {
      onBaseImageReady();
    } else {
      imgOriginal.onload = onBaseImageReady;
    }

    // ViT Classifications Sub-table
    classificationsBody.innerHTML = "";
    if (data.classifications && data.classifications.length) {
      data.classifications.slice(0, 5).forEach((cls, i) => {
        const confNum = cls.confidence * 100;
        const tier = getConfidenceTier(confNum);
        const label = cls.class_name || `Class ${cls.class_id}`;

        const row = document.createElement("tr");
        row.className = `tier--${tier.tier}`;
        row.innerHTML = `
          <td style="text-align:center;font-weight:600;color:var(--text-muted);">${i + 1}</td>
          <td>
            <strong>${esc(label)}</strong>
            ${cls.is_general_model ? '<span class="badge-uncertain" style="margin-left:6px;">⚠️ General</span>' : ''}
          </td>
          <td>
            <div class="conf-cell-wrap">
              <div class="conf-meta-row">
                <span class="conf-tier-tag">${tier.label}</span>
                <span class="conf-pct-val">${confNum.toFixed(1)}%</span>
              </div>
              <div class="conf-bar-track">
                <div class="conf-bar-fill" style="width:${confNum.toFixed(1)}%"></div>
              </div>
            </div>
          </td>
        `;
        classificationsBody.appendChild(row);
      });
    }

    // Render initial info pills
    renderGradcamPills(data.gradcam_info || {});

    // Confidence Breakdown Details
    if (confidenceBreakdownBox && data.confidence_breakdown) {
      const cb = data.confidence_breakdown;
      confidenceBreakdownBox.innerHTML = `
        <div style="font-weight:600;margin-bottom:4px;color:var(--text-primary);">Confidence Breakdown</div>
        <div>Detector Certainty: <strong>${cb.detector_mean_confidence != null ? (cb.detector_mean_confidence * 100).toFixed(1) + '%' : 'N/A'}</strong> | ViT Match: <strong>${cb.vit_match != null ? (cb.vit_match ? 'Aligned' : 'Divergent') : 'N/A'}</strong></div>
      `;
      confidenceBreakdownBox.style.display = "block";
    }

    // Reveal Workspace
    reportChatLayout.classList.add("visible");
    reportChatLayout.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderGradcamPills(info) {
    if (!gradcamPills) return;
    gradcamPills.innerHTML = "";
    if (!info) return;

    const targetName = info.target_class_name || (info.target_class !== undefined ? `ID ${info.target_class}` : "Selected Region");
    const confStr = info.confidence != null ? (info.confidence > 1 ? info.confidence : info.confidence * 100).toFixed(1) + "%" : "N/A";
    
    gradcamPills.innerHTML = `
      <div class="gradcam-pill">Target: <strong>${esc(targetName)}</strong></div>
      <div class="gradcam-pill">Confidence: <strong>${confStr}</strong></div>
      ${info.method_description ? `<div class="gradcam-pill">${esc(info.method_description)}</div>` : ''}
    `;
  }

  // ========================================================================
  //  CATEGORY FILTER BAR LOGIC
  // ========================================================================
  function renderFilterChips() {
    if (!filterChipsRow) return;
    filterChipsRow.innerHTML = "";

    const totalCount = currentDetections.length;
    const chipsData = [
      { id: "all", label: "All", icon: "✨", count: totalCount },
      { id: "persons", label: "Persons", icon: "👤", count: categoryCounts.persons || 0 },
    ];

    // Optional Part C: Male / Female demographic chips (only if enabled)
    if (demographicEnabled) {
      chipsData.push(
        { id: "demo_male", label: "Male (apparent)", icon: "👨", count: demographicCounts.male || 0, isDemo: true },
        { id: "demo_female", label: "Female (apparent)", icon: "👩", count: demographicCounts.female || 0, isDemo: true },
        { id: "demo_unclear", label: "Unclear / not estimated", icon: "❓", count: demographicCounts.unclear || 0, isDemo: true }
      );
      if (demographicDisclaimerBanner) demographicDisclaimerBanner.style.display = "flex";
    } else {
      if (demographicDisclaimerBanner) demographicDisclaimerBanner.style.display = "none";
    }

    chipsData.push(
      { id: "clothing", label: "Clothing", icon: "👕", count: categoryCounts.clothing || 0 },
      { id: "vehicles", label: "Vehicles", icon: "🚗", count: categoryCounts.vehicles || 0 },
      { id: "license_plates", label: "License Plates", icon: "🔢", count: categoryCounts.license_plates || 0 },
      { id: "animals", label: "Animals", icon: "🐾", count: categoryCounts.animals || 0 },
      { id: "other", label: "Other", icon: "📦", count: categoryCounts.other || 0 }
    );

    chipsData.forEach((chip) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "filter-chip";
      btn.id = `filter-chip-${chip.id}`;
      btn.setAttribute("data-filter", chip.id);
      btn.setAttribute("role", "button");
      btn.setAttribute("tabindex", "0");

      const isActive = activeFilterCategories.has(chip.id);
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
      if (isActive) btn.classList.add("active");

      // Disabled if 0 items, unless it's "all"
      if (chip.count === 0 && chip.id !== "all") {
        btn.classList.add("disabled");
        btn.disabled = true;
        btn.setAttribute("title", `No ${chip.label} detected`);
      }

      btn.innerHTML = `
        <span class="filter-chip__icon">${chip.icon}</span>
        <span class="filter-chip__label">${chip.label}</span>
        <span class="filter-chip__count">${chip.count}</span>
      `;

      btn.addEventListener("click", () => toggleFilterChip(chip.id));
      btn.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleFilterChip(chip.id);
        }
      });

      filterChipsRow.appendChild(btn);
    });
  }

  function toggleFilterChip(filterId) {
    if (filterId === "all") {
      activeFilterCategories.clear();
      activeFilterCategories.add("all");
    } else {
      activeFilterCategories.delete("all");
      if (activeFilterCategories.has(filterId)) {
        activeFilterCategories.delete(filterId);
      } else {
        activeFilterCategories.add(filterId);
      }
      if (activeFilterCategories.size === 0) {
        activeFilterCategories.add("all");
      }
    }

    // Update aria-pressed and classes on all chips
    const allChips = filterChipsRow.querySelectorAll(".filter-chip");
    allChips.forEach((chip) => {
      const fid = chip.getAttribute("data-filter");
      const isSelected = activeFilterCategories.has(fid);
      chip.setAttribute("aria-pressed", isSelected ? "true" : "false");
      if (isSelected) {
        chip.classList.add("active");
      } else {
        chip.classList.remove("active");
      }
    });

    applyFilter();
  }

  function matchesFilter(det) {
    // 1. Confidence threshold check
    const conf = (det.confidence || 0) * 100;
    if (conf < minConfidenceThreshold) return false;

    // 2. Search query check
    if (searchQuery) {
      const name = (det.class_name || "").toLowerCase();
      const cat = (det.category || "").toLowerCase();
      const q = searchQuery.toLowerCase();
      if (!name.includes(q) && !cat.includes(q)) return false;
    }

    // 3. Category chips check (Union of selected categories)
    if (activeFilterCategories.has("all")) return true;

    const cat = det.category || "other";

    for (const filterId of activeFilterCategories) {
      if (filterId === "persons" && cat === "persons") return true;
      if (filterId === "clothing" && cat === "clothing") return true;
      if (filterId === "vehicles" && cat === "vehicles") return true;
      if (filterId === "license_plates" && cat === "license_plates") return true;
      if (filterId === "animals" && cat === "animals") return true;
      if (filterId === "other" && cat === "other") return true;

      // Demographic sub-checks
      if (filterId === "demo_male" && det.demographic_estimate?.presentation === "male") return true;
      if (filterId === "demo_female" && det.demographic_estimate?.presentation === "female") return true;
      if (filterId === "demo_unclear" && det.demographic_estimate?.presentation === "unclear") return true;
    }

    return false;
  }

  function applyFilter() {
    let visibleCount = 0;
    let visibleUncertain = 0;
    let activeItemVisible = false;

    currentDetections.forEach((det, i) => {
      const matches = matchesFilter(det);
      const row = document.getElementById(`det-row-${i}`);
      const bbox = document.getElementById(`bbox-item-${i}`);

      if (matches) {
        visibleCount++;
        if (det.confidence < 0.6) visibleUncertain++;
        if (activeDetectionIndex === i) activeItemVisible = true;

        if (row) {
          row.classList.remove("filtered-out");
          row.style.display = "";
        }
        if (bbox) {
          bbox.classList.remove("filtered-hidden");
          bbox.classList.remove("filtered-dimmed");
        }
      } else {
        if (row) {
          row.classList.add("filtered-out");
          row.style.display = "none";
        }
        if (bbox) {
          if (isDimMode) {
            bbox.classList.remove("filtered-hidden");
            bbox.classList.add("filtered-dimmed");
          } else {
            bbox.classList.add("filtered-hidden");
            bbox.classList.remove("filtered-dimmed");
          }
        }
      }
    });

    // If currently selected detection is filtered out, gracefully reset zoom
    if (activeDetectionIndex !== null && !activeItemVisible) {
      resetZoom();
    }

    // Update Counts & Badges
    const totalCount = currentDetections.length;
    const isFiltered = visibleCount !== totalCount || !activeFilterCategories.has("all") || searchQuery || minConfidenceThreshold > 10;
    
    if (detectedCountBadge) {
      if (isFiltered) {
        detectedCountBadge.innerHTML = `${visibleCount} of ${totalCount} <span class="filtered-indicator-tag">Filtered</span>`;
      } else {
        detectedCountBadge.textContent = `${totalCount} ${totalCount === 1 ? 'Object' : 'Objects'}`;
      }
    }

    // Update summary strip counts
    if (aiSummaryStats) {
      const topClass = analysisData?.classifications && analysisData.classifications.length ? analysisData.classifications[0].class_name : "N/A";
      aiSummaryStats.innerHTML = `
        <div class="stat-chip">
          <span class="stat-chip__label">Showing:</span>
          <span class="stat-chip__value">${visibleCount} ${isFiltered ? `(${totalCount} total)` : 'Objects'}</span>
        </div>
        <div class="stat-chip">
          <span class="stat-chip__label">Uncertain:</span>
          <span class="stat-chip__value" style="color:${visibleUncertain > 0 ? 'var(--conf-med-text)' : 'inherit'}">${visibleUncertain}</span>
        </div>
        <div class="stat-chip">
          <span class="stat-chip__label">ViT Scene:</span>
          <span class="stat-chip__value">${esc(topClass)}</span>
        </div>
      `;
    }

    // Show empty state if 0 visible
    if (visibleCount === 0) {
      if (filterEmptyState) filterEmptyState.style.display = "flex";
      if (detectionsTableContainer) detectionsTableContainer.style.display = "none";
    } else {
      if (filterEmptyState) filterEmptyState.style.display = "none";
      if (detectionsTableContainer) detectionsTableContainer.style.display = "block";
    }
  }

  // --- Filter Bar Event Listeners ---
  if (filterSearchInput) {
    filterSearchInput.addEventListener("input", (e) => {
      searchQuery = e.target.value.trim();
      if (clearSearchBtn) clearSearchBtn.style.display = searchQuery ? "block" : "none";
      applyFilter();
    });
  }

  if (clearSearchBtn) {
    clearSearchBtn.addEventListener("click", () => {
      searchQuery = "";
      if (filterSearchInput) filterSearchInput.value = "";
      clearSearchBtn.style.display = "none";
      applyFilter();
    });
  }

  if (filterConfidenceSlider) {
    filterConfidenceSlider.addEventListener("input", (e) => {
      minConfidenceThreshold = parseInt(e.target.value, 10);
      if (confSliderValue) confSliderValue.textContent = `${minConfidenceThreshold}%`;
      applyFilter();
    });
  }

  if (dimHideToggle) {
    dimHideToggle.addEventListener("change", (e) => {
      isDimMode = e.target.checked;
      if (dimHideLabel) dimHideLabel.textContent = isDimMode ? "Dim" : "Hide";
      applyFilter();
    });
  }

  if (emptyClearFiltersBtn) {
    emptyClearFiltersBtn.addEventListener("click", () => {
      activeFilterCategories.clear();
      activeFilterCategories.add("all");
      searchQuery = "";
      if (filterSearchInput) filterSearchInput.value = "";
      if (clearSearchBtn) clearSearchBtn.style.display = "none";
      minConfidenceThreshold = 10;
      if (filterConfidenceSlider) filterConfidenceSlider.value = 10;
      if (confSliderValue) confSliderValue.textContent = "10%";
      renderFilterChips();
      applyFilter();
    });
  }

  // Preset Selector change event (auto re-analyzes current image if present)
  if (presetSelect) {
    presetSelect.addEventListener("change", () => {
      if (selectedFile) {
        analyzeBtn.click();
      }
    });
  }

  // Honest confidence info note toggle
  if (btnScoreInfo && confidenceInfoCollapsible) {
    btnScoreInfo.addEventListener("click", () => {
      const isVisible = confidenceInfoCollapsible.style.display !== "none";
      confidenceInfoCollapsible.style.display = isVisible ? "none" : "block";
    });
  }

  if (btnCloseScoreInfo && confidenceInfoCollapsible) {
    btnCloseScoreInfo.addEventListener("click", () => {
      confidenceInfoCollapsible.style.display = "none";
    });
  }

  // ========================================================================
  //  RENDER DETECTIONS TABLE + SYNCHRONIZED BOUNDING BOXES
  // ========================================================================
  function renderDetectionsTableAndBoxes(detections, imgW, imgH) {
    detectionsBody.innerHTML = "";
    bboxOverlayContainer.innerHTML = "";

    if (!detections || !detections.length) {
      detectionsBody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--text-muted);padding:30px;">No objects detected</td></tr>`;
      return;
    }

    detections.forEach((det, i) => {
      const confNum = det.confidence * 100;
      const confStr = confNum.toFixed(1);
      const tier = getConfidenceTier(confNum);
      const thumbnailSrc = croppedThumbnails[i] || "";

      // 1. Create Synchronized Bounding Box on Image
      const [x1, y1, x2, y2] = det.bbox;
      const pLeft = (x1 / imgW) * 100;
      const pTop = (y1 / imgH) * 100;
      const pWidth = ((x2 - x1) / imgW) * 100;
      const pHeight = ((y2 - y1) / imgH) * 100;

      const bboxItem = document.createElement("div");
      bboxItem.className = `bbox-item tier--${tier.tier}`;
      bboxItem.id = `bbox-item-${i}`;
      bboxItem.style.left = `${pLeft}%`;
      bboxItem.style.top = `${pTop}%`;
      bboxItem.style.width = `${pWidth}%`;
      bboxItem.style.height = `${pHeight}%`;

      // Label Tag attached above bounding box
      const bboxTag = document.createElement("div");
      bboxTag.className = "bbox-tag";
      bboxTag.textContent = `${det.class_name} ${confStr}%`;
      bboxItem.appendChild(bboxTag);

      bboxOverlayContainer.appendChild(bboxItem);

      // 2. Create Table Row
      const row = document.createElement("tr");
      row.id = `det-row-${i}`;
      row.className = `tier--${tier.tier}`;
      if (det.parent_index !== null && det.parent_index !== undefined) {
        row.classList.add("nested-row");
      }
      row.setAttribute("tabindex", "0");
      row.setAttribute("role", "button");
      row.setAttribute("aria-selected", "false");

      let uncertainBadge = "";
      if (det.is_uncertain) {
        uncertainBadge = '<span class="badge-uncertain">⚠️ Uncertain</span>';
      }

      let parentTag = "";
      if (det.parent_index !== null && det.parent_index !== undefined) {
        const pType = det.parent_type === "person" ? "Person" : (det.parent_type === "vehicle" ? "Vehicle" : "Parent");
        parentTag = `<span class="nested-tag">↳ on ${pType} #${det.parent_index + 1}</span>`;
      }

      let demoTag = "";
      if (demographicEnabled && det.demographic_estimate) {
        const pres = det.demographic_estimate.presentation;
        const icon = pres === "male" ? "👨" : (pres === "female" ? "👩" : "❓");
        demoTag = `<span class="nested-tag" title="Apparent presentation, approximate">${icon} ${pres}</span>`;
      }

      // Infer honest reasons for lower confidence scores if applicable
      let scoreHint = "";
      if (confNum < 65) {
        const [bx1, by1, bx2, by2] = det.bbox;
        const bw = bx2 - bx1;
        const bh = by2 - by1;
        const areaFrac = (bw * bh) / ((imgW || 1) * (imgH || 1));
        
        // Check if significantly overlapped by another box
        let isOverlapped = false;
        detections.forEach((other, oi) => {
          if (oi !== i && other.confidence >= det.confidence) {
            const [ox1, oy1, ox2, oy2] = other.bbox;
            const ix1 = Math.max(bx1, ox1);
            const iy1 = Math.max(by1, oy1);
            const ix2 = Math.min(bx2, ox2);
            const iy2 = Math.min(by2, oy2);
            if (ix2 > ix1 && iy2 > iy1) {
              const iArea = (ix2 - ix1) * (iy2 - iy1);
              if (iArea / (bw * bh) > 0.35) isOverlapped = true;
            }
          }
        });

        if (areaFrac < 0.015) {
          scoreHint = '<span class="score-factor-hint">Likely lower score: small/distant target</span>';
        } else if (isOverlapped) {
          scoreHint = '<span class="score-factor-hint">Likely lower score: partly overlapped/crowded</span>';
        }
      }

      row.innerHTML = `
        <td class="col-index-cell">${i + 1}</td>
        <td>
          <div class="det-class-cell">
            ${thumbnailSrc ? `<img src="${thumbnailSrc}" alt="${esc(det.class_name)}" class="det-thumb" />` : ''}
            <div class="det-info-wrap">
              <span class="det-name">${esc(det.class_name)}</span>
              ${uncertainBadge}
              ${parentTag}
              ${demoTag}
              ${scoreHint}
            </div>
          </div>
        </td>
        <td>
          <div class="conf-cell-wrap">
            <div class="conf-meta-row">
              <span class="conf-tier-tag">${tier.label}</span>
              <span class="conf-pct-val">${confStr}%</span>
            </div>
            <div class="conf-bar-track">
              <div class="conf-bar-fill" style="width:${confStr}%"></div>
            </div>
          </div>
        </td>
      `;

      // 3. Bidirectional Hover Sync
      row.addEventListener("mouseenter", () => {
        bboxItem.classList.add("active");
        row.classList.add("highlight");
      });
      row.addEventListener("mouseleave", () => {
        bboxItem.classList.remove("active");
        row.classList.remove("highlight");
      });

      bboxItem.addEventListener("mouseenter", () => {
        bboxItem.classList.add("active");
        row.classList.add("highlight");
        if (activeDetectionIndex === null) {
          row.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });
      bboxItem.addEventListener("mouseleave", () => {
        bboxItem.classList.remove("active");
        row.classList.remove("highlight");
      });

      // 4. Click & Keyboard Focus Selection
      row.addEventListener("click", () => selectDetection(i));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectDetection(i);
        }
      });

      bboxItem.addEventListener("click", (e) => {
        e.stopPropagation();
        selectDetection(i);
      });

      detectionsBody.appendChild(row);
    });
  }

  // ========================================================================
  //  INTERACTIVE ZOOM & CLASS-CONDITIONED PER-DETECTION HEATMAP SWITCHING
  // ========================================================================
  function selectDetection(index) {
    // If clicking already selected object, deselect and reset to global
    if (activeDetectionIndex === index) {
      resetZoom();
      return;
    }

    if (!currentDetections || index < 0 || index >= currentDetections.length) {
      return;
    }

    activeDetectionIndex = index;
    const det = currentDetections[index];
    const imgW = naturalImgW || 1;
    const imgH = naturalImgH || 1;

    const [x1, y1, x2, y2] = det.bbox;
    const boxW = Math.max(x2 - x1, 1);
    const boxH = Math.max(y2 - y1, 1);
    const centerX = (((x1 + x2) / 2) / imgW) * 100;
    const centerY = (((y1 + y2) / 2) / imgH) * 100;

    // Smooth transform on zoom layer
    const fracW = (boxW * 1.35) / imgW;
    const fracH = (boxH * 1.35) / imgH;
    const rawScale = Math.min(1 / fracW, 1 / fracH);
    const scale = Math.min(3.8, Math.max(1.3, rawScale));

    if (zoomLayer) {
      zoomLayer.style.transformOrigin = `${centerX.toFixed(2)}% ${centerY.toFixed(2)}%`;
      zoomLayer.style.transform = `scale(${scale.toFixed(3)})`;
    }

    // Highlight row
    const allRows = detectionsBody.querySelectorAll("tr");
    allRows.forEach((r, idx) => {
      if (idx === index) {
        r.classList.add("selected");
        r.setAttribute("aria-selected", "true");
        r.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        r.classList.remove("selected");
        r.setAttribute("aria-selected", "false");
      }
    });

    // Highlight bounding box
    const allBoxes = bboxOverlayContainer.querySelectorAll(".bbox-item");
    allBoxes.forEach((b, idx) => {
      if (idx === index) {
        b.classList.add("selected");
        b.classList.remove("dimmed");
      } else {
        b.classList.remove("selected");
        b.classList.add("dimmed");
      }
    });

    if (resetViewBtn) {
      resetViewBtn.classList.add("active");
    }

    const classificationsTable = document.getElementById("classificationsTable");
    if (classificationsTable) classificationsTable.style.display = "none";
    
    const confidenceBreakdownBox = document.getElementById("confidenceBreakdownBox");
    if (confidenceBreakdownBox) confidenceBreakdownBox.style.display = "none";
    
    const accordionTitle = document.querySelector("#sceneAccordion .scene-context-summary__title span");
    if (accordionTitle) accordionTitle.textContent = "Selected Object Explainability Details";

    // Ensure the accordion is open when viewing object details
    const sceneAccordion = document.getElementById("sceneAccordion");
    if (sceneAccordion) sceneAccordion.setAttribute("open", "");

    // Update XAI heatmap to focus on the selected detection
    updateXaiForSelection(index);
  }

  function resetZoom() {
    activeDetectionIndex = null;
    if (zoomLayer) {
      zoomLayer.style.transformOrigin = "50% 50%";
      zoomLayer.style.transform = "scale(1)";
    }

    const allRows = detectionsBody.querySelectorAll("tr");
    allRows.forEach((r) => {
      r.classList.remove("selected");
      r.setAttribute("aria-selected", "false");
    });

    const allBoxes = bboxOverlayContainer.querySelectorAll(".bbox-item");
    allBoxes.forEach((b) => {
      b.classList.remove("selected");
      b.classList.remove("dimmed");
    });

    if (resetViewBtn) {
      resetViewBtn.classList.remove("active");
    }

    const classificationsTable = document.getElementById("classificationsTable");
    if (classificationsTable) classificationsTable.style.display = "table";
    
    const confidenceBreakdownBox = document.getElementById("confidenceBreakdownBox");
    if (confidenceBreakdownBox) confidenceBreakdownBox.style.display = "block";
    
    const accordionTitle = document.querySelector("#sceneAccordion .scene-context-summary__title span");
    if (accordionTitle) accordionTitle.textContent = "ViT Scene Classification & Explainability Details";

    // Revert to global overview heatmap
    updateXaiForSelection(null);
  }

  if (resetViewBtn) {
    resetViewBtn.addEventListener("click", resetZoom);
  }

  // ========================================================================
  //  XAI METHOD SWITCHER (GRAD-CAM, SHAP, LIME) & SELECTION SYNC
  // ========================================================================
  function setXaiTabActive(method) {
    activeXaiMethod = method;
    xaiMethodTabs.forEach(tab => {
      if (tab.dataset.method === method) {
        tab.classList.add("active");
      } else {
        tab.classList.remove("active");
      }
    });
    if (currentMethodBadge) {
      const labels = { gradcam: "Grad-CAM Overlay", shap: "SHAP Diverging Map", lime: "LIME Superpixels" };
      currentMethodBadge.textContent = labels[method] || "Explainability Overlay";
    }
  }

  async function updateXaiForSelection(detectionIndex) {
    if (!currentSessionId) return;

    const cacheKey = detectionIndex !== null && detectionIndex !== undefined
      ? `det_${detectionIndex}_${activeXaiMethod}`
      : `global_${activeXaiMethod}`;

    // 1. Check client cache
    if (xaiCache[cacheKey] && xaiCache[cacheKey].overlay_b64) {
      imgGradcamOverlay.src = "data:image/jpeg;base64," + xaiCache[cacheKey].overlay_b64;
      renderGradcamPills(xaiCache[cacheKey]);
      return;
    }

    // 2. If Grad-CAM on detection with precomputed data
    if (activeXaiMethod === "gradcam" && detectionIndex !== null && detectionIndex !== undefined && currentDetections[detectionIndex]?.gradcam_image) {
      const det = currentDetections[detectionIndex];
      imgGradcamOverlay.src = "data:image/jpeg;base64," + det.gradcam_image;
      renderGradcamPills({
        target_class_name: det.class_name?.replace(" (⚠️ Uncertain match)", ""),
        confidence: det.confidence,
        method_description: `Grad-CAM spatial activation focused on ${det.class_name}`
      });
      return;
    }

    // 3. Fetch from backend on demand
    loadingOverlay.classList.add("visible");
    try {
      const formData = new FormData();
      formData.append("session_id", currentSessionId);
      formData.append("method", activeXaiMethod);
      if (detectionIndex !== null && detectionIndex !== undefined) {
        formData.append("detection_index", detectionIndex);
      }

      const res = await fetch("/explain/method", { method: "POST", body: formData });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Failed to load explanation.");

      const resData = data.result || {};
      xaiCache[cacheKey] = resData;

      if (resData.overlay_b64) {
        imgGradcamOverlay.src = "data:image/jpeg;base64," + resData.overlay_b64;
        renderGradcamPills(resData);
      }
    } catch (err) {
      showError(err.message);
    } finally {
      loadingOverlay.classList.remove("visible");
    }
  }

  async function switchXaiMethod(method) {
    setXaiTabActive(method);
    await updateXaiForSelection(activeDetectionIndex);
  }

  xaiMethodTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const method = tab.dataset.method;
      switchXaiMethod(method);
    });
  });

  overlayToggle.addEventListener("change", (e) => {
    if (e.target.checked) {
      imgGradcamOverlay.classList.add("active");
      bboxOverlayContainer.classList.add("dimmed");
    } else {
      imgGradcamOverlay.classList.remove("active");
      bboxOverlayContainer.classList.remove("dimmed");
    }
  });

  // ========================================================================
  //  CONTEXTUAL PROMPT CHIPS
  // ========================================================================
  function updateSuggestedChips(detections) {
    if (!suggestedChipsRow) return;

    let chips = [
      { label: "Why was this detected?", prompt: "Why were these specific objects detected?" },
      { label: "Explain Grad-CAM overlay", prompt: "What visual features is Grad-CAM highlighting on this image?" },
      { label: "Confidence breakdown", prompt: "Can you break down the confidence scores and any uncertainty flags?" }
    ];

    if (detections && detections.length > 0) {
      const topDet = detections[0];
      chips.unshift({
        label: `Why detected as ${topDet.class_name}?`,
        prompt: `Why is the object with ${(topDet.confidence * 100).toFixed(0)}% confidence classified as ${topDet.class_name}?`
      });

      if (detections.length > 1) {
        chips.push({
          label: `Compare ${detections[0].class_name} vs ${detections[1].class_name}`,
          prompt: `Compare the predictions for ${detections[0].class_name} and ${detections[1].class_name}.`
        });
      }
    }

    suggestedChipsRow.innerHTML = `<span class="chips-label">Suggested:</span>` +
      chips.map(c => `
        <button type="button" class="chip-btn" data-prompt="${esc(c.prompt)}">
          <span>✨</span> ${esc(c.label)}
        </button>
      `).join("");

    suggestedChipsRow.querySelectorAll(".chip-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const prompt = btn.getAttribute("data-prompt");
        if (prompt) {
          chatInput.value = prompt;
          chatForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
        }
      });
    });
  }

  // ========================================================================
  //  CHAT DRAWER & FORM SUBMISSION
  // ========================================================================
  if (chatBrandTrigger) {
    chatBrandTrigger.addEventListener("click", () => {
      chatFeedDrawer.classList.toggle("open");
      scrollChat();
    });
  }

  if (chatDrawerClose) {
    chatDrawerClose.addEventListener("click", () => {
      chatFeedDrawer.classList.remove("open");
    });
  }

  voiceToggleBtn.addEventListener("click", () => {
    isVoiceEnabled = !isVoiceEnabled;
    if (isVoiceEnabled) {
      voiceToggleBtn.classList.add("active");
      voiceToggleBtn.textContent = "🔊";
      voiceToggleBtn.title = "Mute Assistant Voice";
    } else {
      voiceToggleBtn.classList.remove("active");
      voiceToggleBtn.textContent = "🔈";
      voiceToggleBtn.title = "Unmute Assistant Voice";
      window.speechSynthesis.cancel();
    }
  });

  micBtn.addEventListener("click", () => {
    if (!recognition) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }
    if (isRecording) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    if (!currentSessionId) {
      showError("Please analyze an image first before chatting with Astra AI.");
      return;
    }

    chatFeedDrawer.classList.add("open");

    appendMessage("user", message);
    chatInput.value = "";
    chatSendBtn.disabled = true;
    chatTyping.classList.add("visible");
    scrollChat();

    try {
      const formData = new FormData();
      formData.append("session_id", currentSessionId);
      formData.append("message", message);
      
      // Pass active filter state to chatbot so responses respect the filtered view
      const filtersState = {
        categories: Array.from(activeFilterCategories),
        search_query: searchQuery,
        min_confidence: minConfidenceThreshold / 100.0,
      };
      formData.append("filters_json", JSON.stringify(filtersState));

      const res = await fetch("/chat", { method: "POST", body: formData });
      const data = await res.json();

      if (!data.success) throw new Error(data.error || "Chat request failed.");

      appendMessage("assistant", data.response);
      speakText(data.response);
    } catch (err) {
      appendMessage("assistant", `⚠️ Error: ${err.message}`);
    } finally {
      chatTyping.classList.remove("visible");
      chatSendBtn.disabled = false;
      chatInput.focus();
      scrollChat();
    }
  });

  function resetChat(summaryText) {
    chatMessages.innerHTML = "";
    if (summaryText) {
      appendMessage("assistant", summaryText);
      chatFeedDrawer.classList.add("open");
      scrollChat();
    }
  }

  function appendMessage(role, content) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-msg chat-msg--${role}`;

    const avatar = document.createElement("div");
    avatar.className = "chat-msg__avatar";
    avatar.textContent = role === "user" ? "👤" : "✨";

    const contentWrapper = document.createElement("div");
    contentWrapper.className = "chat-msg__content";

    const bubble = document.createElement("div");
    bubble.className = "chat-msg__bubble";

    let html = esc(content);
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/__(.*?)__/g, "<strong>$1</strong>");
    html = html.replace(/\n/g, "<br>");

    bubble.innerHTML = html;
    contentWrapper.appendChild(bubble);

    wrapper.appendChild(avatar);
    wrapper.appendChild(contentWrapper);

    chatMessages.appendChild(wrapper);
    scrollChat();
  }

  function scrollChat() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  // ========================================================================
  //  PDF REPORT & FULLSCREEN
  // ========================================================================
  if (downloadPdfBtn) {
    downloadPdfBtn.addEventListener("click", async () => {
      if (!currentSessionId) return;

      downloadPdfBtn.disabled = true;
      const origText = downloadPdfBtn.innerHTML;
      downloadPdfBtn.innerHTML = `<span>⏳ Exporting…</span>`;

      try {
        const formData = new FormData();
        formData.append("session_id", currentSessionId);

        const res = await fetch("/report", { method: "POST", body: formData });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || "Report generation failed.");
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `AstraVisionX_Report_${currentSessionId}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (err) {
        showError(err.message);
      } finally {
        downloadPdfBtn.disabled = false;
        downloadPdfBtn.innerHTML = origText;
      }
    });
  }

  if (fullscreenBtn) {
    fullscreenBtn.addEventListener("click", () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });
  }

  // ========================================================================
  //  HELPERS
  // ========================================================================
  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.add("visible");
  }

  function hideError() {
    errorBanner.classList.remove("visible");
  }

  function esc(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
