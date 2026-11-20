from typing import Any, Dict, List, Optional, Tuple
import os
import cv2
import numpy as np
from ultralytics import YOLO
from astravisionx.core.interfaces import DetectorInterface
from astravisionx.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Quality / Speed Profiles Definition
# ─────────────────────────────────────────────────────────────────────────────
PRESET_CONFIGS = {
    "fast": {
        "model_path": "yolov8n-oiv7.pt",
        "imgsz": 640,
        "iou": 0.60,
        "augment": False,
        "description": "Fast Preset: Nano model (7MB), 640px, single-scale. Lowest latency.",
    },
    "balanced": {
        "model_path": "yolov8m-oiv7.pt",
        "imgsz": 960,
        "iou": 0.55,
        "augment": False,
        "description": "Balanced Preset: Medium model (53MB), 960px, tuned NMS (0.55). Stable scores.",
    },
    "high_accuracy": {
        "model_path": "yolov8x-oiv7.pt",
        "imgsz": 960,
        "iou": 0.55,
        "augment": True,
        "description": "High Accuracy Preset: X-Large model (138MB), 960px, Test-Time Augmentation (TTA). Maximum recall.",
    },
}


class YoloDetector(DetectorInterface):
    def __init__(self, model_path: str = None, preset: str = None):
        """
        Initialize the YOLOv8 detector with explicit verification and preset support.
        """
        self.preset = preset or settings.inference_preset
        preset_cfg = PRESET_CONFIGS.get(self.preset, PRESET_CONFIGS["balanced"])
        
        # If user explicitly passed a custom model path, respect it; otherwise use preset's model
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = preset_cfg["model_path"]
            
        # Fallback if checkpoint does not exist on disk
        if not os.path.exists(self.model_path):
            fallback = "yolov8s-oiv7.pt"
            print(f"[AstraVisionX] Notice: '{self.model_path}' not found, falling back to '{fallback}'", flush=True)
            self.model_path = fallback

        self.imgsz = preset_cfg.get("imgsz", settings.detector_imgsz)
        self.iou = preset_cfg.get("iou", settings.detector_iou_threshold)
        self.augment = preset_cfg.get("augment", settings.detector_augment)
        self.device = "cuda" if settings.use_cuda else "cpu"
        self.num_classes = 0
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.num_classes = len(self.model.names) if hasattr(self.model, "names") else 0
            
            if self.num_classes == 0:
                raise ValueError(f"Loaded YOLO model '{self.model_path}' has 0 class names.")

            # Required startup log line printing active model, class count, imgsz, conf, iou, augment, and device
            print(
                f"[AstraVisionX] YOLO Detector Active | Model: '{self.model_path}' | "
                f"Preset: '{self.preset}' | Classes: {self.num_classes} | imgsz: {self.imgsz} | "
                f"conf: {settings.detector_conf_threshold} | iou: {self.iou} | "
                f"augment: {self.augment} | Device: {self.device}",
                flush=True
            )
        except Exception as e:
            err_msg = (
                f"[AstraVisionX] CRITICAL: Failed to load YOLO checkpoint '{self.model_path}'. "
                f"Ensure the checkpoint file exists and is valid. Error: {str(e)}"
            )
            print(err_msg, flush=True)
            raise RuntimeError(err_msg) from e

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Optional gentle input quality handling (auto-contrast/CLAHE) if enabled in config.
        """
        if not settings.enable_auto_contrast:
            return image
        try:
            # Convert to LAB, apply CLAHE on L-channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        except Exception:
            return image

    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        iou: Optional[float] = None,
        augment: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in an RGB image using YOLO.
        Calibrated to use settings.detector_conf_threshold (default 0.10) to capture candidates.
        """
        if self.model is None:
            raise RuntimeError("YOLO detector model is not initialized.")
        
        threshold = conf if conf is not None else settings.detector_conf_threshold
        infer_imgsz = imgsz if imgsz is not None else self.imgsz
        infer_iou = iou if iou is not None else self.iou
        infer_augment = augment if augment is not None else self.augment

        # Optional input quality handling
        processed_image = self.preprocess_image(image)

        # When passing a numpy array to Ultralytics YOLO, it expects BGR order by default.
        # Ensure BGR conversion so Ultralytics restores correct RGB internally.
        if isinstance(processed_image, np.ndarray) and len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            infer_image = processed_image[:, :, ::-1]
        else:
            infer_image = processed_image

        try:
            results = self.model(
                infer_image,
                device=self.device,
                conf=threshold,
                imgsz=infer_imgsz,
                iou=infer_iou,
                augment=infer_augment,
                verbose=False
            )
        except Exception as e:
            print(f"[AstraVisionX] YOLO detection inference error: {e}", flush=True)
            raise RuntimeError(f"YOLO detection inference failed: {str(e)}") from e
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # box.xyxy is [xmin, ymin, xmax, ymax]
                b = box.xyxy[0].cpu().numpy().tolist()
                box_conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"Class_{cls_id}")
                
                detections.append({
                    "bbox": b,
                    "confidence": box_conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })
                
        return detections
