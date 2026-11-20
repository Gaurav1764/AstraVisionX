# AstraVisionX — Vercel Deployment & Architectural Guide

> **Important Architectural Reality Check**
> Vercel is a **serverless platform tailored for frontend and lightweight edge/serverless microservices**.
> Serverless platforms like Vercel have hard constraints:
> - **Serverless Function Payload & Execution Limits:** Standard tier limits serverless bundle sizes (max 50MB – 250MB uncompressed) and function timeout limits (10s – 60s max execution).
> - **Heavy ML Dependency Footprint:** AstraVisionX depends on PyTorch (`torch`, `torchvision`), HuggingFace Transformers, Timm, OpenCV, SHAP, LIME, and YOLO weights (`yolov8m-oiv7.pt` alone is ~52MB, `yolov8x` is ~131MB, and `torch` + dependencies exceeds 1.5GB).
> - **Inference Times:** Deep analysis, Grad-CAM, SHAP, and LIME on CPU take between 5s to 45s per image, which easily exceeds or borders serverless execution thresholds.

Because of this, trying to bundle PyTorch + YOLO weights directly into a single Vercel serverless function will exceed Vercel's zip artifact size limits and trigger cold-start memory timeouts.

Below is the **production-grade hybrid architecture for Vercel**, along with step-by-step setup guides for both the recommended decoupled deployment and a direct standalone container / cloud backend option.

---

## Architecture: The Recommended Production Strategy

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │                     USER / BROWSER                     │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                            ┌───────────────────┴───────────────────┐
                                            │                                       │
                                   Static Assets / UI                        API Invocations
                                            │                                       │
                                            ▼                                       ▼
                     ┌──────────────────────────────────────────────┐     ┌───────────────────────────────────┐
                     │            VERCEL (Frontend / Edge)          │     │    DEDICATED ML BACKEND           │
                     │  • HTML5 / CSS / Vanilla JS (or Next.js)     │────▶│    • Modal / Render / Railway /     │
                     │  • Instant Global CDN delivery               │     │      RunPod / AWS EC2 / GCP Cloud │
                     │  • Reverse Proxy via `vercel.json` rewrites  │     │      Run with persistent CPU/GPU  │
                     │  • Zero-config SSL & custom domain           │     │    • Full PyTorch, YOLO, XAI stack│
                     └──────────────────────────────────────────────┘     └───────────────────────────────────┘
```

---

## Option 1: Frontend on Vercel + Backend on Render / Railway / Cloud Run (Recommended)

This gives you the best of both worlds: lightning-fast worldwide CDN caching for the UI on Vercel, and dedicated memory/CPU/GPU runtime for PyTorch and YOLO inference.

### Step 1: Prepare Vercel Reverse Proxy (`vercel.json`)
Create a `vercel.json` file in your repository root so that Vercel routes frontend visitors to the UI, while transparently proxying API calls to your hosted ML backend without CORS issues:

```json
{
  "version": 2,
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://your-backend-service.onrender.com/$1"
    },
    {
      "source": "/analyze",
      "destination": "https://your-backend-service.onrender.com/analyze"
    },
    {
      "source": "/detect",
      "destination": "https://your-backend-service.onrender.com/detect"
    },
    {
      "source": "/chat",
      "destination": "https://your-backend-service.onrender.com/chat"
    },
    {
      "source": "/health",
      "destination": "https://your-backend-service.onrender.com/health"
    },
    {
      "source": "/(.*)",
      "destination": "/static/$1"
    }
  ]
}
```

### Step 2: Deploy Backend to Render, Railway, or Fly.io
Deploy the FastAPI backend where memory and runtime limits are not constrained by serverless boundaries.

#### Example: Deploying to Render
1. Push your repository to GitHub.
2. Log into [Render.com](https://render.com) and click **New + Web Service**.
3. Connect your repository `AstraVisionX`.
4. Configure the service:
   - **Environment:** `Python 3` (or `Docker`)
   - **Build Command:** `pip install -e .`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables in Render:
   - `GEMINI_API_KEY`: `<your_gemini_api_key>`
   - `MODEL_MODE`: `general`
   - `INFERENCE_PRESET`: `fast` or `balanced`
   - `YOLO_MODEL_PATH`: `yolov8s-oiv7.pt` (or `yolov8m-oiv7.pt`)
6. Copy the assigned backend URL (e.g. `https://astravisionx-api.onrender.com`).

### Step 3: Connect and Deploy on Vercel
1. Install Vercel CLI (optional) or use the Web Dashboard:
   ```bash
   npm install -g vercel
   vercel login
   ```
2. In your `vercel.json`, replace `https://your-backend-service.onrender.com` with your live backend URL.
3. Deploy to Vercel:
   ```bash
   vercel --prod
   ```
   Or link your GitHub repo directly on [vercel.com/new](https://vercel.com/new).

---

## Option 2: Direct Vercel Serverless Function (WSGI / ASGI Gateway)

If you strictly wish to host on Vercel's Python Serverless Runtime (`@vercel/python`), you must use a lightweight configuration:

> [!WARNING]
> Because Vercel serverless has a 250MB uncompressed limit, standard full `torch` + `ultralytics` + `timm` + `scikit-learn` packages will easily exceed the quota on Vercel build steps. To attempt standalone Vercel hosting, you must use CPU-optimized wheels and strip heavy non-essential dependencies.

### Step 1: Create `api/index.py` (Vercel Entrypoint)
Create a file at `api/index.py`:

```python
import sys
import os

# Add root directory to sys.path so astravisionx module resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
```

### Step 2: Configure `vercel.json` for Serverless Python

```json
{
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python",
      "config": {
        "maxDuration": 60
      }
    },
    {
      "src": "static/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/(favicon.ico|robots.txt)",
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ]
}
```

### Step 3: Configure Environment Variables in Vercel Dashboard
In the Vercel project dashboard under **Settings ➔ Environment Variables**, configure:
- `GEMINI_API_KEY`: `<your-api-key>`
- `INFERENCE_PRESET`: `fast` (critical to keep runtime and memory under serverless limits)
- `YOLO_MODEL_PATH`: `yolov8n-oiv7.pt` (nano checkpoint, smallest footprint)
- `USE_CUDA`: `False`
- `DEBUG`: `False`

---

## Deployment Checklist Summary

| Checkpoint | Status / Recommendation |
|---|---|
| **Best Architecture** | Decoupled: Vercel (Frontend & CDN) + Dedicated Cloud Service (FastAPI / PyTorch Backend). |
| **Model Footprint** | Keep model to `yolov8n-oiv7.pt` or `yolov8s-oiv7.pt` for resource-constrained cloud tiers. |
| **Secrets Management** | Never commit `.env` or `api_keys.txt` to GitHub. Always set `GEMINI_API_KEY` via dashboard settings. |
| **Cold Starts** | Set `INFERENCE_PRESET=fast` to minimize startup time and memory footprint during cold-boot invocations. |

---

*AstraVisionX Deployment Guide · Compatible with Vercel CLI v38+ & FastAPI 0.110+*
