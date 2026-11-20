"""
AstraVisionX — FastAPI Web Application

Serves the frontend and provides the full API surface:
  GET  /health          — system / model health check
  POST /detect          — upload image → YOLO bounding boxes + class labels
  POST /explain         — legacy Grad-CAM endpoint
  POST /explain/method  — multi-method XAI (Grad-CAM, SHAP, LIME, or Compare All)
  POST /analyze         — full pipeline: detect → classify → Grad-CAM → confidence breakdown
  POST /report          — generate downloadable PDF report for a session
  POST /chat            — session-scoped conversational Q&A
  GET  /                — serve the frontend
"""

import io
import json
import base64
import cv2
import numpy as np
import torch
import timm
from PIL import Image
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from astravisionx.models.yolo_detector import YoloDetector
from astravisionx.models.vit_classifier import ViTClassifier
from astravisionx.xai.gradcam_explainer import GradCamExplainer
from astravisionx.xai.shap_explainer import ShapExplainer
from astravisionx.xai.lime_explainer import LimeExplainer
from astravisionx.xai.metrics import compute_explanation_agreement, binarize_heatmap
from astravisionx.models.category_map import enrich_detections
from astravisionx.utils.image_utils import (
    apply_colormap_on_image,
    apply_diverging_colormap,
    apply_lime_superpixel_overlay,
    encode_image_base64,
    draw_detections,
)
from astravisionx.config import settings
from astravisionx.core.session import session_manager
from astravisionx.chat.summary import generate_summary
from astravisionx.chat.chatbot import chat as chatbot_chat
from astravisionx.report.pdf_generator import generate_pdf

# ---------------------------------------------------------------------------
# ImageNet class labels (top-level)
# ---------------------------------------------------------------------------
IMAGENET_LABELS: dict = {}

def _load_imagenet_labels():
    """Load ImageNet labels from a well-known URL or fall back to empty."""
    global IMAGENET_LABELS
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            labels = json.loads(resp.read().decode())
            IMAGENET_LABELS = {i: label for i, label in enumerate(labels)}
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Lazy-loaded models & explainers
# ---------------------------------------------------------------------------
_yolo: Optional[YoloDetector] = None
_vit: Optional[ViTClassifier] = None
_cnn = None  # ResNet18 backbone
_gradcam: Optional[GradCamExplainer] = None
_shap: Optional[ShapExplainer] = None
_lime: Optional[LimeExplainer] = None

def _get_models(preset: Optional[str] = None):
    global _yolo, _vit, _cnn, _gradcam, _shap, _lime
    if preset and _yolo is not None and getattr(_yolo, "preset", None) != preset:
        # Re-initialize detector with requested preset
        _yolo = YoloDetector(preset=preset)
    elif _yolo is None:
        _yolo = YoloDetector(preset=preset)
    if _vit is None:
        _vit = ViTClassifier()
    if _cnn is None:
        _cnn = timm.create_model("resnet18", pretrained=True)
        _cnn.eval()
    if _gradcam is None:
        _gradcam = GradCamExplainer()
    if _shap is None:
        _shap = ShapExplainer(num_steps=8, num_background_samples=2)
    if _lime is None:
        _lime = LimeExplainer(num_samples=35, num_segments=25)
    return _yolo, _vit, _cnn, _gradcam, _shap, _lime


def _read_image(contents: bytes):
    """Convert uploaded bytes to PIL Image and numpy RGB array."""
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_rgb = np.array(pil_image)
    return pil_image, image_rgb


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    import threading
    threading.Thread(target=_load_imagenet_labels, daemon=True).start()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AstraVisionX", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ========================== ENDPOINTS ======================================

@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """Health check — returns model / system status."""
    yolo_status = "not loaded (lazy)"
    if _yolo is not None:
        yolo_status = f"loaded ({_yolo.model_path}, {_yolo.num_classes} categories)"
    else:
        yolo_status = f"configured ({settings.yolo_model_path})"

    return JSONResponse({
        "status": "ok",
        "models": {
            "yolo": yolo_status,
            "vit": "loaded" if _vit is not None else "not loaded (lazy)",
            "cnn_backbone": "loaded" if _cnn is not None else "not loaded (lazy)",
            "explainers": ["gradcam", "shap", "lime"],
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_enabled": settings.use_cuda,
        "gemini_configured": bool(settings.gemini_api_key),
    })


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    preset: Optional[str] = Form(None),
):
    """Upload image → YOLO bounding boxes + class labels + confidence."""
    try:
        contents = await file.read()
        _, image_rgb = _read_image(contents)
        yolo, _, _, _, _, _ = _get_models(preset=preset)
        detections = yolo.detect(image_rgb)
        detections = _relabel_detections(detections)
        filter_meta = enrich_detections(
            detections,
            enable_demographics=settings.enable_demographic_estimates
        )
        annotated = draw_detections(image_rgb, detections)

        return JSONResponse({
            "success": True,
            "detections": detections,
            "annotated_image": encode_image_base64(annotated),
            "disclaimer": settings.medical_disclaimer,
            "category_counts": filter_meta["category_counts"],
            "demographic_enabled": filter_meta["demographic_enabled"],
            "demographic_counts": filter_meta["demographic_counts"],
            "filter_config": filter_meta["filter_config"],
            "preset_info": {
                "preset": yolo.preset,
                "model_path": yolo.model_path,
                "imgsz": yolo.imgsz,
                "iou": yolo.iou,
                "augment": yolo.augment,
                "device": yolo.device,
                "badge_label": f"YOLOv8{'x' if 'yolov8x' in yolo.model_path else ('m' if 'yolov8m' in yolo.model_path else ('n' if 'yolov8n' in yolo.model_path else 's'))} · {yolo.preset.replace('_', ' ').title()} · {yolo.imgsz}px"
            },
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """Upload image → Grad-CAM heatmap overlay (base64) + metadata (Legacy)."""
    try:
        contents = await file.read()
        pil_image, image_rgb = _read_image(contents)
        _, vit, cnn, gradcam, _, _ = _get_models()

        tensor = vit.transforms(pil_image).unsqueeze(0).to(vit.device)
        result = gradcam.explain(tensor, cnn)

        heatmap = result["heatmap"]
        overlay = apply_colormap_on_image(image_rgb, heatmap)

        return JSONResponse({
            "success": True,
            "gradcam_image": encode_image_base64(overlay),
            "gradcam_info": {
                "target_class": result["target_class"],
                "target_class_name": IMAGENET_LABELS.get(result["target_class"], f"Class {result['target_class']}"),
                "confidence": result["confidence"],
            },
            "disclaimer": settings.medical_disclaimer,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


def _compute_single_xai(method: str, pil_image: Image.Image, image_rgb: np.ndarray, target_class: Optional[int] = None) -> dict:
    """Helper to compute explanation for a single method across the full image."""
    _, vit, cnn, gradcam, shap_exp, lime_exp = _get_models()
    tensor = vit.transforms(pil_image).unsqueeze(0).to(vit.device)
    
    if method == "gradcam":
        res = gradcam.explain(tensor, cnn, target_class=target_class)
        heatmap = res["heatmap"]
        overlay = apply_colormap_on_image(image_rgb, heatmap)
        salient_mask = binarize_heatmap(heatmap, threshold_percentile=75.0)
        target_cls = res["target_class"]
        return {
            "method": "gradcam",
            "overlay_b64": encode_image_base64(overlay),
            "confidence": res["confidence"],
            "target_class": target_cls,
            "target_class_name": IMAGENET_LABELS.get(target_cls, f"Class {target_cls}"),
            "attribution_score": float(np.mean(heatmap)),
            "salient_mask": salient_mask,
            "method_description": "Grad-CAM: Class Activation Mapping highlighting positive CNN gradient activations."
        }
        
    elif method == "shap":
        res = shap_exp.explain(tensor, cnn, target_class=target_class, org_image_rgb=image_rgb)
        target_cls = res["target_class"]
        return {
            "method": "shap",
            "overlay_b64": res["overlay_b64"],
            "confidence": res["confidence"],
            "target_class": target_cls,
            "target_class_name": IMAGENET_LABELS.get(target_cls, f"Class {target_cls}"),
            "positive_ratio": res["positive_ratio"],
            "negative_ratio": res["negative_ratio"],
            "attribution_score": res["attribution_score"],
            "salient_mask": res["salient_mask"],
            "method_description": res["method_description"]
        }
        
    elif method == "lime":
        res = lime_exp.explain(tensor, cnn, target_class=target_class, org_image_rgb=image_rgb, transform_fn=vit.transforms)
        target_cls = res["target_class"]
        return {
            "method": "lime",
            "overlay_b64": res["overlay_b64"],
            "confidence": res["confidence"],
            "target_class": target_cls,
            "target_class_name": IMAGENET_LABELS.get(target_cls, f"Class {target_cls}"),
            "top_segments_count": res["top_segments_count"],
            "top_segments": res["top_segments"],
            "attribution_score": res["attribution_score"],
            "salient_mask": res["salient_mask"],
            "method_description": res["method_description"]
        }
    else:
        raise ValueError(f"Unknown explanation method: {method}")


def _compute_detection_xai(
    method: str,
    pil_image: Image.Image,
    image_rgb: np.ndarray,
    bbox: list,
    class_name: str,
    target_class: Optional[int] = None
) -> dict:
    """
    Computes class-conditioned, region-targeted explainability (Grad-CAM, SHAP, LIME)
    for a specific YOLO detection. Extracts the object's spatial receptive field with
    contextual padding and generates a focused activation map mapped onto the full image.
    """
    h_full, w_full, _ = image_rgb.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    # 20% contextual boundary padding
    pad_x = int(box_w * 0.20)
    pad_y = int(box_h * 0.20)
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w_full, x2 + pad_x)
    crop_y2 = min(h_full, y2 + pad_y)

    crop_rgb = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_rgb.size == 0:
        crop_rgb = image_rgb
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, w_full, h_full

    crop_pil = Image.fromarray(crop_rgb)
    _, vit, cnn, gradcam, shap_exp, lime_exp = _get_models()
    tensor = vit.transforms(crop_pil).unsqueeze(0).to(vit.device)

    clean_name = class_name.replace(" (⚠️ Uncertain match)", "")

    if method == "gradcam":
        res = gradcam.explain(tensor, cnn, target_class=target_class)
        crop_heatmap = res["heatmap"]
        
        crop_h_px = crop_y2 - crop_y1
        crop_w_px = crop_x2 - crop_x1
        crop_cam_resized = cv2.resize(crop_heatmap, (crop_w_px, crop_h_px), interpolation=cv2.INTER_LINEAR)

        full_heatmap = np.zeros((h_full, w_full), dtype=np.float32)
        full_heatmap[crop_y1:crop_y2, crop_x1:crop_x2] = crop_cam_resized

        # Smooth boundary blend
        k = max(5, int(min(box_w, box_h) * 0.15))
        if k % 2 == 0:
            k += 1
        full_heatmap = cv2.GaussianBlur(full_heatmap, (k, k), 0)
        max_val = np.max(full_heatmap)
        if max_val > 0:
            full_heatmap = full_heatmap / max_val

        overlay = apply_colormap_on_image(image_rgb, full_heatmap)
        salient_mask = binarize_heatmap(full_heatmap, threshold_percentile=75.0)

        return {
            "method": "gradcam",
            "overlay_b64": encode_image_base64(overlay),
            "confidence": res["confidence"],
            "target_class_name": clean_name,
            "attribution_score": float(np.mean(full_heatmap)),
            "salient_mask": salient_mask,
            "method_description": f"Grad-CAM: Spatial gradient activations focused on detected '{clean_name}'."
        }

    elif method == "shap":
        res = shap_exp.explain(tensor, cnn, target_class=target_class, org_image_rgb=crop_rgb)
        raw_map = res.get("raw_map")
        if raw_map is not None:
            raw_map_resized = cv2.resize(raw_map, (crop_x2 - crop_x1, crop_y2 - crop_y1), interpolation=cv2.INTER_LINEAR)
        else:
            raw_map_resized = np.ones((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype=np.float32)

        full_shap = np.zeros((h_full, w_full), dtype=np.float32)
        full_shap[crop_y1:crop_y2, crop_x1:crop_x2] = raw_map_resized
        overlay = apply_diverging_colormap(image_rgb, full_shap)

        return {
            "method": "shap",
            "overlay_b64": encode_image_base64(overlay),
            "confidence": res["confidence"],
            "target_class_name": clean_name,
            "positive_ratio": res.get("positive_ratio", 75.0),
            "negative_ratio": res.get("negative_ratio", 25.0),
            "attribution_score": float(np.mean(np.abs(full_shap))),
            "salient_mask": binarize_heatmap(np.abs(full_shap), threshold_percentile=75.0),
            "method_description": f"SHAP: Feature attributions for detected '{clean_name}'."
        }

    elif method == "lime":
        res = lime_exp.explain(tensor, cnn, target_class=target_class, org_image_rgb=crop_rgb, transform_fn=vit.transforms)
        segments = res.get("raw_map")
        top_positive_segments = res.get("top_segments", [])
        
        full_segments = np.zeros((h_full, w_full), dtype=np.int32)
        if segments is not None:
            segments_resized = cv2.resize(segments.astype(np.int32), (crop_x2 - crop_x1, crop_y2 - crop_y1), interpolation=cv2.INTER_NEAREST)
            full_segments[crop_y1:crop_y2, crop_x1:crop_x2] = segments_resized + 1
            shifted_top = [(s_id + 1, weight) for s_id, weight in top_positive_segments]
        else:
            shifted_top = []

        overlay = apply_lime_superpixel_overlay(image_rgb, full_segments, shifted_top)

        return {
            "method": "lime",
            "overlay_b64": encode_image_base64(overlay),
            "confidence": res["confidence"],
            "target_class_name": clean_name,
            "top_segments_count": len(top_positive_segments),
            "top_segments": top_positive_segments,
            "attribution_score": res.get("attribution_score", 0.5),
            "salient_mask": binarize_heatmap(cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY) / 255.0, threshold_percentile=75.0),
            "method_description": f"LIME: Superpixel contributions for detected '{clean_name}'."
        }
    else:
        raise ValueError(f"Unknown method: {method}")


@app.post("/explain/method")
async def explain_method(
    session_id: str = Form(...),
    method: str = Form("gradcam"),
    target_class: Optional[int] = Form(None),
    detection_index: Optional[int] = Form(None),
):
    """
    Unified Explainability Endpoint:
    Computes or retrieves cached XAI for Grad-CAM, SHAP, LIME (global or per-detection).
    """
    session = session_manager.get_session(session_id)
    if session is None:
        return JSONResponse({"success": False, "error": "Session not found or expired."}, status_code=404)

    try:
        # Reconstruct image from session
        if session.image_bytes is not None:
            pil_image, image_rgb = _read_image(session.image_bytes)
        else:
            img_b64 = session.original_image_b64
            img_bytes = base64.b64decode(img_b64)
            pil_image, image_rgb = _read_image(img_bytes)

        # Handle Per-Detection Explainability
        if detection_index is not None and 0 <= detection_index < len(session.detections):
            det = session.detections[detection_index]
            det_xai_map = session.detection_xai.setdefault(detection_index, {})
            
            if method not in det_xai_map or target_class is not None:
                xai_res = _compute_detection_xai(
                    method=method,
                    pil_image=pil_image,
                    image_rgb=image_rgb,
                    bbox=det["bbox"],
                    class_name=det.get("class_name", "object"),
                    target_class=target_class
                )
                json_safe_res = {k: v for k, v in xai_res.items() if k != "salient_mask"}
                det_xai_map[method] = json_safe_res

            return JSONResponse({
                "success": True,
                "session_id": session_id,
                "detection_index": detection_index,
                "method": method,
                "result": det_xai_map[method],
                "available_methods": list(det_xai_map.keys()),
            })

        # Global Full-Image Explainability
        methods_to_run = ["gradcam", "shap", "lime"] if method in ["all", "compare_all"] else [method]
        
        for m in methods_to_run:
            if m not in session.xai_cache or target_class is not None:
                xai_res = _compute_single_xai(m, pil_image, image_rgb, target_class=target_class)
                session.salient_masks[m] = xai_res["salient_mask"]
                
                json_safe_res = {k: v for k, v in xai_res.items() if k != "salient_mask"}
                session.xai_cache[m] = json_safe_res

        agreement = compute_explanation_agreement(session.salient_masks)
        session.agreement_metrics = agreement

        if method in ["all", "compare_all"]:
            return JSONResponse({
                "success": True,
                "session_id": session_id,
                "mode": "compare_all",
                "results": session.xai_cache,
                "agreement": agreement,
                "available_methods": list(session.xai_cache.keys()),
            })
        else:
            return JSONResponse({
                "success": True,
                "session_id": session_id,
                "method": method,
                "result": session.xai_cache[method],
                "agreement": agreement,
                "available_methods": list(session.xai_cache.keys()),
            })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


def _relabel_detections(detections: list) -> list:
    """Relabel low-confidence YOLO detections based on config threshold."""
    for d in detections:
        conf = d.get("confidence", 0)
        if conf < settings.confidence_threshold:
            raw_name = d.get("class_name", "unknown")
            d["class_name"] = f"{raw_name} (⚠️ Uncertain match)"
            d["raw_label"] = raw_name
            d["is_uncertain"] = True
        else:
            d["is_uncertain"] = False
    return detections


def _relabel_classifications(classifications: list) -> list:
    """Relabel classifier output based on MODEL_MODE."""
    if not settings.has_medical_checkpoint:
        caveat = "⚠️ General-purpose model" if settings.is_medical_mode else "non-medical general classifier"
        for c in classifications:
            raw_name = c.get("class_name", "unknown")
            c["class_name"] = f"{raw_name} ({caveat})"
            c["raw_label"] = raw_name
            c["label_source"] = "imagenet-1k"
            c["is_medical"] = False
    else:
        for c in classifications:
            c["label_source"] = "medical-checkpoint"
            c["is_medical"] = True
    return classifications


def _build_confidence_breakdown(detections: list, classifications: list, agreement_metrics: dict) -> dict:
    """
    Constructs an honest, comprehensive confidence & reliability breakdown.
    Includes raw model confidence, decision boundary margin, alternative classes,
    uncertainty status, and multi-method consistency.
    """
    top_1 = classifications[0] if classifications else {}
    top_2 = classifications[1] if len(classifications) > 1 else {}
    
    top_1_conf = float(top_1.get("confidence", 0.0))
    top_2_conf = float(top_2.get("confidence", 0.0))
    margin = round(top_1_conf - top_2_conf, 4)
    
    if margin >= 0.50:
        margin_label = "Decisive Separation (Clear Boundary)"
        margin_level = "high"
    elif margin >= 0.20:
        margin_label = "Moderate Separation"
        margin_level = "med"
    else:
        margin_label = "Narrow Boundary (Potential Ambiguity)"
        margin_level = "low"
        
    has_uncertain_detection = any(d.get("is_uncertain", False) for d in detections)
    is_low_conf = top_1_conf < settings.confidence_threshold
    
    if is_low_conf or has_uncertain_detection:
        reliability_status = "⚠️ Uncertain / Potential Out-of-Distribution"
        reliability_level = "warning"
    elif margin_level == "high" and top_1_conf >= 0.70:
        reliability_status = "✅ High Decision Reliability"
        reliability_level = "high"
    else:
        reliability_status = "ℹ️ Moderate Decision Reliability"
        reliability_level = "med"

    return {
        "raw_confidence": top_1_conf,
        "raw_confidence_pct": round(top_1_conf * 100, 1),
        "top_class_name": top_1.get("raw_label", top_1.get("class_name", "N/A")),
        "decision_margin": margin,
        "decision_margin_pct": round(margin * 100, 1),
        "decision_margin_label": margin_label,
        "margin_level": margin_level,
        "alternatives": classifications[:5],
        "has_uncertain_detection": has_uncertain_detection,
        "is_low_conf": is_low_conf,
        "reliability_status": reliability_status,
        "reliability_level": reliability_level,
        "agreement": agreement_metrics
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    preset: Optional[str] = Form(None),
):
    """
    Full pipeline: detect → classify → default Grad-CAM → Confidence Breakdown.
    Creates a session for conversational Q&A, multi-method XAI, and report generation.
    Supports selectable inference presets: 'fast', 'balanced', 'high_accuracy'.
    """
    try:
        contents = await file.read()
        pil_image, image_rgb = _read_image(contents)
        yolo, vit, cnn, gradcam, _, _ = _get_models(preset=preset)

        # 1. YOLO Detection
        detections = yolo.detect(image_rgb)
        detections = _relabel_detections(detections)
        filter_meta = enrich_detections(
            detections,
            enable_demographics=settings.enable_demographic_estimates
        )

        # 2. ViT Classification
        classifications = vit.classify(image_rgb)
        for c in classifications:
            c["class_name"] = IMAGENET_LABELS.get(c["class_id"], f"Class {c['class_id']}")
        classifications = _relabel_classifications(classifications)

        # 3. Default Fast Grad-CAM
        gradcam_b64 = None
        gradcam_info = {}
        salient_mask_gc = None
        try:
            tensor = vit.transforms(pil_image).unsqueeze(0).to(vit.device)
            gc_res = gradcam.explain(tensor, cnn)
            heatmap = gc_res["heatmap"]
            overlay = apply_colormap_on_image(image_rgb, heatmap)
            gradcam_b64 = encode_image_base64(overlay)
            salient_mask_gc = binarize_heatmap(heatmap, threshold_percentile=75.0)
            
            raw_target_name = IMAGENET_LABELS.get(gc_res["target_class"], f"Class {gc_res['target_class']}")
            gradcam_info = {
                "target_class": gc_res["target_class"],
                "target_class_name": f"{raw_target_name} (⚠️ general-purpose label)" if (settings.is_medical_mode and not settings.has_medical_checkpoint) else raw_target_name,
                "raw_target_class_name": raw_target_name,
                "confidence": gc_res["confidence"],
                "overlay_b64": gradcam_b64,
                "method_description": "Grad-CAM: Class Activation Mapping highlighting positive CNN gradient activations."
            }
        except Exception as e:
            gradcam_info = {"error": str(e)}

        # 3b. Pre-compute Per-Detection Grad-CAM for instant click switching
        for idx, det in enumerate(detections):
            try:
                det_xai = _compute_detection_xai(
                    method="gradcam",
                    pil_image=pil_image,
                    image_rgb=image_rgb,
                    bbox=det["bbox"],
                    class_name=det.get("class_name", "object")
                )
                det["gradcam_image"] = det_xai["overlay_b64"]
                det["gradcam_info"] = {
                    "target_class_name": det.get("class_name", "object"),
                    "confidence": det.get("confidence", 0.0),
                    "attribution_score": det_xai.get("attribution_score", 0.5),
                    "method_description": det_xai.get("method_description", "Grad-CAM")
                }
            except Exception:
                det["gradcam_image"] = gradcam_b64

        # 4. Annotated YOLO image
        annotated = draw_detections(image_rgb, detections)

        # 5. Agreement and Confidence Breakdown
        masks = {"gradcam": salient_mask_gc} if salient_mask_gc is not None else {}
        agreement_metrics = compute_explanation_agreement(masks)
        confidence_breakdown = _build_confidence_breakdown(detections, classifications, agreement_metrics)

        # 6. NL Summary
        nl_summary = generate_summary(
            detections, classifications, gradcam_info,
            filename=file.filename or "uploaded image",
        )

        # 7. Create session & cache initial state
        session = session_manager.create_session()
        session.filename = file.filename or "uploaded image"
        session.image_bytes = contents
        session.original_image_b64 = encode_image_base64(image_rgb)
        session.annotated_image_b64 = encode_image_base64(annotated)
        session.gradcam_image_b64 = gradcam_b64
        session.detections = detections
        session.classifications = classifications
        session.gradcam_info = gradcam_info
        session.nl_summary = nl_summary
        session.chat_history.append({"role": "assistant", "content": nl_summary})
        
        # Populate session detection_xai
        for idx, det in enumerate(detections):
            if "gradcam_image" in det:
                session.detection_xai[idx] = {
                    "gradcam": {
                        "method": "gradcam",
                        "overlay_b64": det["gradcam_image"],
                        "confidence": det.get("confidence", 0.0),
                        "target_class_name": det.get("class_name", "object"),
                    }
                }

        # Cache Global Grad-CAM into xai_cache and salient_masks
        if "error" not in gradcam_info:
            session.xai_cache["gradcam"] = {
                "method": "gradcam",
                "overlay_b64": gradcam_b64,
                "confidence": gradcam_info.get("confidence", 0.0),
                "target_class": gradcam_info.get("target_class", 0),
                "target_class_name": gradcam_info.get("target_class_name", "N/A"),
                "attribution_score": float(np.mean(heatmap)) if 'heatmap' in locals() else 0.5,
                "method_description": gradcam_info.get("method_description", "Grad-CAM")
            }
            session.salient_masks["gradcam"] = salient_mask_gc
            
        session.agreement_metrics = agreement_metrics
        session.confidence_breakdown = confidence_breakdown

        return JSONResponse({
            "success": True,
            "session_id": session.session_id,
            "model_mode": settings.model_mode,
            "has_medical_checkpoint": settings.has_medical_checkpoint,
            "original_image": session.original_image_b64,
            "annotated_image": session.annotated_image_b64,
            "gradcam_image": gradcam_b64,
            "detections": detections,
            "classifications": classifications,
            "gradcam_info": gradcam_info,
            "confidence_breakdown": confidence_breakdown,
            "xai_cache": session.xai_cache,
            "agreement": agreement_metrics,
            "available_methods": list(session.xai_cache.keys()),
            "nl_summary": nl_summary,
            "disclaimer": settings.medical_disclaimer,
            "category_counts": filter_meta["category_counts"],
            "demographic_enabled": filter_meta["demographic_enabled"],
            "demographic_counts": filter_meta["demographic_counts"],
            "filter_config": filter_meta["filter_config"],
            "preset_info": {
                "preset": yolo.preset,
                "model_path": yolo.model_path,
                "imgsz": yolo.imgsz,
                "iou": yolo.iou,
                "augment": yolo.augment,
                "device": yolo.device,
                "badge_label": f"YOLOv8{'x' if 'yolov8x' in yolo.model_path else ('m' if 'yolov8m' in yolo.model_path else ('n' if 'yolov8n' in yolo.model_path else 's'))} · {yolo.preset.replace('_', ' ').title()} · {yolo.imgsz}px"
            },
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/report")
async def report(session_id: str = Form(...)):
    """Generate a downloadable PDF report for a given session."""
    session = session_manager.get_session(session_id)
    if session is None:
        return JSONResponse({"success": False, "error": "Session not found or expired."}, status_code=404)

    try:
        pdf_bytes = generate_pdf(session)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="AstraVisionX_Report_{session_id}.pdf"'},
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    message: str = Form(...),
    filters_json: Optional[str] = Form(None),
):
    """
    Session-scoped conversational Q&A.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        return JSONResponse(
            {"success": False, "error": "Session not found or expired. Please re-analyze the image."},
            status_code=404,
        )

    if filters_json:
        try:
            session.active_filters = json.loads(filters_json)
        except Exception:
            pass

    try:
        response = chatbot_chat(session, message)
        return JSONResponse({
            "success": True,
            "response": response,
            "session_id": session_id,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting AstraVisionX Web UI …", flush=True)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
