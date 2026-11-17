import pytest
import io
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from app import app
from astravisionx.config import settings

client = TestClient(app)

def create_dummy_image_bytes():
    image = Image.new('RGB', (224, 224), color = 'white')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_confidence_bounds_analyze():
    """Test that all confidence values returned by /analyze are within [0, 1]"""
    # Force general medical mode for testing
    settings.model_mode = "medical"
    
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/analyze", 
        files={"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    
    # Check YOLO Detections
    for det in data.get("detections", []):
        conf = det.get("confidence", -1)
        assert 0.0 <= conf <= 1.0, f"Detection confidence out of bounds: {conf}"
        
    # Check ViT Classifications
    for cls in data.get("classifications", []):
        conf = cls.get("confidence", -1)
        assert 0.0 <= conf <= 1.0, f"Classification confidence out of bounds: {conf}"
        # Also check relabeling logic
        assert "⚠️ General-purpose model" in cls.get("class_name", "") or "non-medical general classifier" in cls.get("class_name", "")
        
    # Check Grad-CAM
    gradcam_info = data.get("gradcam_info", {})
    if "error" not in gradcam_info:
        conf = gradcam_info.get("confidence", -1)
        assert 0.0 <= conf <= 1.0, f"Grad-CAM confidence out of bounds: {conf}"
        # Check relabeling logic
        assert "general-purpose label" in gradcam_info.get("target_class_name", "") or "general classifier" in gradcam_info.get("target_class_name", "")

if __name__ == "__main__":
    pytest.main(["-v", "test_confidence.py"])
