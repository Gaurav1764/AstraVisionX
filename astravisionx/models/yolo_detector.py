from typing import Any, Dict, List, Optional
import os
import numpy as np
from ultralytics import YOLO
from astravisionx.core.interfaces import DetectorInterface
from astravisionx.config import settings

class YoloDetector(DetectorInterface):
    def __init__(self, model_path: str = None):
        """
        Initialize the YOLOv8 detector with explicit verification and error reporting.
        """
        self.model_path = model_path or settings.yolo_model_path
        self.device = "cuda" if settings.use_cuda else "cpu"
        self.num_classes = 0
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.num_classes = len(self.model.names) if hasattr(self.model, "names") else 0
            
            if self.num_classes == 0:
                raise ValueError(f"Loaded YOLO model '{self.model_path}' has 0 class names.")

            print(
                f"[AstraVisionX] YOLO Detector loaded successfully: '{self.model_path}' "
                f"({self.num_classes} categories on {self.device})",
                flush=True
            )
        except Exception as e:
            err_msg = (
                f"[AstraVisionX] CRITICAL: Failed to load YOLO checkpoint '{self.model_path}'. "
                f"Ensure the checkpoint file exists and is valid. Error: {str(e)}"
            )
            print(err_msg, flush=True)
            raise RuntimeError(err_msg) from e

    def detect(self, image: np.ndarray, conf: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detect objects in an RGB image using YOLO.
        Calibrated to use settings.detector_conf_threshold (default 0.10) to capture candidates.
        """
        if self.model is None:
            raise RuntimeError("YOLO detector model is not initialized.")
        
        threshold = conf if conf is not None else settings.detector_conf_threshold
        
        # When passing a numpy array to Ultralytics YOLO, it expects BGR order by default.
        # Ensure BGR conversion so Ultralytics restores correct RGB internally.
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            infer_image = image[:, :, ::-1]
        else:
            infer_image = image

        try:
            results = self.model(infer_image, device=self.device, conf=threshold, verbose=False)
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
