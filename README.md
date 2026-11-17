<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<h1 align="center">🔭 AstraVisionX</h1>

<p align="center">
  <strong>Multimodal Computer Vision Platform with Explainable AI & Conversational Intelligence</strong>
</p>

<p align="center">
  Real-time object detection (YOLOv8) · Scene classification (ViT) · Multi-method XAI (Grad-CAM, Integrated Gradients, LIME) · Confidence-calibrated reliability analysis · Gemini-powered conversational assistant
</p>

> **⚠️ Disclaimer:** AstraVisionX is a research and educational tool — **not a diagnostic instrument**. All outputs are exploratory and should not be used for clinical, legal, or safety-critical decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Research Paper](#research-paper)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

AstraVisionX is an end-to-end multimodal computer vision platform that combines:

1. **Object Detection** — YOLOv8 with Open Images V7 (601 categories) for real-time bounding-box detection
2. **Scene Classification** — Vision Transformer (ViT-B/16) for ImageNet-1K scene-level classification
3. **Explainable AI (XAI)** — Three complementary explanation methods on a shared CNN backbone (ResNet-18):
   - **Grad-CAM** — Gradient-weighted class activation mapping
   - **Integrated Gradients (IG)** — Axiomatic pixel-level attribution
   - **LIME** — Model-agnostic superpixel perturbation
4. **Multi-Method Consensus** — Pairwise IoU/Dice agreement scoring across XAI methods
5. **Reliability Analysis** — Decision-boundary margin tiers with calibration metrics
6. **Conversational Assistant** — Gemini-powered Q&A grounded in structured analysis context
7. **PDF Report Generation** — Downloadable reports with detections, explanations, and confidence breakdowns

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend (SPA)                     │
│              static/index.html + app.js + style.css      │
└──────────────────────────┬──────────────────────────────┘
                           │  REST API (FastAPI)
┌──────────────────────────▼──────────────────────────────┐
│                       app.py                             │
│  /detect  /analyze  /explain/method  /chat  /report      │
└───┬───────┬────────┬────────────┬──────────┬────────────┘
    │       │        │            │          │
    ▼       ▼        ▼            ▼          ▼
┌──────┐ ┌─────┐ ┌──────────┐ ┌──────┐ ┌────────┐
│ YOLO │ │ ViT │ │ XAI      │ │Gemini│ │  PDF   │
│ v8   │ │B/16 │ │ GC+IG+   │ │ Chat │ │ Report │
│      │ │     │ │ LIME     │ │      │ │        │
└──────┘ └─────┘ └──────────┘ └──────┘ └────────┘
   ↓        ↓         ↓
 [OIV7]  [IN-1K]  [ResNet-18]
```

---

## Features

### 🎯 Real-Time Object Detection
- YOLOv8s with Open Images V7 checkpoint (601 object categories)
- Configurable confidence thresholds with uncertainty flagging
- Annotated bounding-box overlays

### 🧠 Scene Classification
- ViT-B/16 pretrained on ImageNet-1K (1,000 classes)
- Top-5 predictions with softmax probabilities
- Decision-boundary margin analysis (p₁ − p₂)

### 🔍 Multi-Method Explainability
- **Grad-CAM**: Fast, coarse spatial activation maps from convolutional layers
- **Integrated Gradients**: Axiomatic, pixel-level attribution with configurable integration steps
- **LIME**: Model-agnostic superpixel perturbation explanations
- **Per-detection XAI**: Region-targeted explanations for individual YOLO detections with contextual padding

### 📊 Consensus & Reliability
- Pairwise IoU and Dice agreement across all XAI method pairs
- Three-tier reliability badges: High / Moderate / Narrow Boundary
- Calibration-aware confidence breakdown

### 💬 Conversational Intelligence
- Gemini-powered session-scoped Q&A
- Responses grounded in structured detection, classification, and XAI context
- Natural-language summaries of analysis results

### 📄 PDF Reports
- One-click downloadable PDF with all analysis results
- Includes detections, classifications, XAI overlays, and confidence metrics

---

## Demo

<p align="center">
  <em>Upload an image → Detect objects → Classify scene → Explain predictions → Chat about results</em>
</p>

**Example outputs** (generated by `run_inference.py`):

| Detection | Grad-CAM | Integrated Gradients | LIME |
|-----------|----------|---------------------|------|
| Bounding boxes with class labels | Heatmap highlighting key CNN activations | Pixel-level attribution map | Superpixel contribution overlay |

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip / venv
- (Optional) CUDA-compatible GPU for faster inference

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AstraVisionX.git
cd AstraVisionX

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your Gemini API key (free at https://aistudio.google.com)
```

### Model Weights

YOLO weights are **not included** in the repository (too large for Git). They will be **auto-downloaded** by Ultralytics on first run, or you can manually download:

```bash
# Default model (recommended)
# YOLOv8s with Open Images V7 — auto-downloads on first use

# Or manually download specific variants:
# yolov8n-oiv7.pt  (7 MB)   — Nano (fastest)
# yolov8s-oiv7.pt  (23 MB)  — Small (default, good balance)
# yolov8m-oiv7.pt  (53 MB)  — Medium
# yolov8x-oiv7.pt  (138 MB) — Extra-large (most accurate)
```

---

## Quick Start

### Web UI

```bash
python app.py
# Open http://localhost:8000 in your browser
```

### CLI Pipeline

```bash
python run_inference.py
# Runs the full pipeline on sample.jpg and saves output images
```

### Python API

```python
from astravisionx.models.yolo_detector import YoloDetector
from astravisionx.models.vit_classifier import ViTClassifier
from astravisionx.xai import GradCamExplainer, ShapExplainer, LimeExplainer
import cv2
import timm
from PIL import Image

# Load image
image_rgb = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)

# 1. Detect objects
yolo = YoloDetector()
detections = yolo.detect(image_rgb)

# 2. Classify scene
vit = ViTClassifier()
classifications = vit.classify(image_rgb)

# 3. Explain predictions
cnn = timm.create_model("resnet18", pretrained=True)
cnn.eval()
tensor = vit.transforms(Image.fromarray(image_rgb)).unsqueeze(0)

gc = GradCamExplainer()
result = gc.explain(tensor, cnn)
print(f"Top class: {result['target_class']}, Confidence: {result['confidence']:.2%}")
```

---

## Project Structure

```
AstraVisionX/
├── app.py                          # FastAPI web application (all endpoints)
├── run_inference.py                # CLI demo — full pipeline on a single image
├── test_confidence.py              # Pytest suite for API confidence validation
├── requirements.txt                # Python dependencies
├── .env.example                    # Template environment configuration
├── LICENSE                         # MIT License
│
├── astravisionx/                   # Core Python package
│   ├── __init__.py
│   ├── config.py                   # Pydantic settings (env-based configuration)
│   │
│   ├── models/                     # Detection & classification models
│   │   ├── yolo_detector.py        # YOLOv8 wrapper (Open Images V7)
│   │   └── vit_classifier.py       # ViT-B/16 wrapper (ImageNet-1K)
│   │
│   ├── xai/                        # Explainability modules
│   │   ├── __init__.py             # Public API exports
│   │   ├── gradcam_explainer.py    # Grad-CAM implementation
│   │   ├── shap_explainer.py       # Integrated Gradients implementation
│   │   ├── lime_explainer.py       # LIME superpixel implementation
│   │   └── metrics.py             # Agreement metrics (IoU, Dice, consensus)
│   │
│   ├── chat/                       # Conversational intelligence
│   │   ├── __init__.py
│   │   ├── chatbot.py              # Gemini-powered session Q&A
│   │   ├── gemini_service.py       # Gemini API client
│   │   └── summary.py             # Natural-language summary generator
│   │
│   ├── core/                       # Session & state management
│   │   ├── interfaces.py           # Data models & type definitions
│   │   └── session.py             # Session manager with TTL
│   │
│   ├── report/                     # PDF report generation
│   │   ├── __init__.py
│   │   └── pdf_generator.py        # ReportLab-based PDF builder
│   │
│   └── utils/                      # Shared utilities
│       └── image_utils.py          # Image processing, overlays, encoding
│
├── static/                         # Frontend SPA
│   ├── index.html                  # Main HTML page
│   ├── style.css                   # Styles
│   └── app.js                      # Frontend JavaScript
│
└── docs/                           # Documentation & research
    ├── AstraVisionX_Research_Paper.tex   # IEEE-format LaTeX paper
    └── AstraVisionX_Project_Report.txt   # Project report
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the web frontend |
| `/health` | GET | System & model health check |
| `/detect` | POST | Upload image → YOLO detections (bounding boxes + labels) |
| `/explain` | POST | Upload image → Grad-CAM heatmap overlay (legacy) |
| `/explain/method` | POST | Multi-method XAI: `gradcam`, `shap`, `lime`, or `compare_all` |
| `/analyze` | POST | Full pipeline: detect → classify → explain → confidence breakdown |
| `/chat` | POST | Session-scoped conversational Q&A |
| `/report` | POST | Generate downloadable PDF report |

### Example: Full Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample.jpg"
```

### Example: Multi-Method XAI Comparison

```bash
curl -X POST http://localhost:8000/explain/method \
  -F "session_id=YOUR_SESSION_ID" \
  -F "method=compare_all"
```

---

## Configuration

All settings are managed via environment variables (`.env` file). See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `True` | Enable debug mode |
| `USE_CUDA` | `False` | Use GPU if available |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `YOLO_MODEL_PATH` | `yolov8s-oiv7.pt` | YOLO model checkpoint |
| `VIT_MODEL_NAME` | `vit_base_patch16_224` | ViT model name (timm) |
| `DETECTOR_CONF_THRESHOLD` | `0.10` | YOLO detection threshold |
| `MODEL_MODE` | `general` | `general` or `medical` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `SESSION_TTL_MINUTES` | `30` | Session expiry time |

---

## Evaluation & Benchmarks

AstraVisionX was evaluated on 100 diverse ImageNet images spanning 100 classes. Key results:

### XAI Agreement (top-20% masks, N=100)

| Pair | IoU (%) | Dice (%) |
|------|---------|----------|
| Grad-CAM ↔ IG | 23.40 ± 2.21 | 36.63 ± 2.81 |
| Grad-CAM ↔ LIME | 23.05 ± 2.32 | 36.01 ± 3.02 |
| IG ↔ LIME | 17.55 ± 1.36 | 29.28 ± 1.94 |
| **Global Mean** | **21.33 ± 1.44** | **33.97 ± 1.90** |
| Analytical Chance | 11.11 | — |

### Faithfulness (Deletion/Insertion AUC, N=25)

| Method | Deletion AUC ↓ | Insertion AUC ↑ |
|--------|---------------|-----------------|
| Integrated Gradients | **0.152** | 0.545 |
| Grad-CAM | 0.296 | **0.701** |
| Random Baseline | 0.204 | 0.214 |

### Reliability Tiers (N=100)

| Tier | Margin (m) | Count | Accuracy |
|------|-----------|-------|----------|
| High | m ≥ 0.50 | 85 | 100% |
| Moderate | 0.20 ≤ m < 0.50 | 9 | 100% |
| Narrow Boundary | m < 0.20 | 6 | 16.7% |

For full methodology and analysis, see the [research paper](docs/AstraVisionX_Research_Paper.tex).

---

## Research Paper

The accompanying IEEE-format research paper is available in the [`docs/`](docs/) directory:

- **LaTeX source**: [`docs/AstraVisionX_Research_Paper.tex`](docs/AstraVisionX_Research_Paper.tex)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest

# Run tests
pytest test_confidence.py -v

# Start dev server
python app.py
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [timm](https://github.com/huggingface/pytorch-image-models) — ViT & ResNet model zoo
- [Google Gemini](https://ai.google.dev/) — Conversational AI
- [LIME](https://github.com/marcotcr/lime) — Model-agnostic explanations
- [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391) — Grad-CAM
- [Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365) — Integrated Gradients
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) — Detection dataset

---

<p align="center">
  Built with ❤️ for Explainable AI research
</p>

