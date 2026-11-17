import cv2
import numpy as np
import base64

def apply_colormap_on_image(org_im: np.ndarray, activation: np.ndarray, colormap_name: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply heatmap on image.
    
    Args:
        org_im: Original image (H, W, 3) in RGB format.
        activation: Heatmap (H, W) in range [0, 1].
        colormap_name: OpenCV colormap.
        
    Returns:
        Overlaid image (H, W, 3) in RGB format.
    """
    # Resize heatmap to match image size
    activation = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(activation, 0, 1)), colormap_name)
    
    # Convert heatmap to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    alpha = 0.5
    overlay = cv2.addWeighted(org_im, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def apply_diverging_colormap(org_im: np.ndarray, shap_values: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Apply a diverging colormap for signed attributions (SHAP).
    Positive values -> Crimson / Red (pushes towards class)
    Negative values -> Cyan / Blue (pushes against class)
    Near zero -> transparent/neutral.
    
    Args:
        org_im: Original image (H, W, 3) in RGB format.
        shap_values: Signed 2D array (H, W) or (H, W, C) where values are in [-max, +max].
        alpha: Blending weight for high attribution pixels.
        
    Returns:
        RGB image with overlaid diverging attribution map.
    """
    if len(shap_values.shape) == 3:
        # Sum or mean across color channels to get 2D spatial attribution
        shap_2d = np.mean(shap_values, axis=2) if shap_values.shape[2] == 3 else shap_values[0]
    else:
        shap_2d = shap_values.copy()

    # Resize to match original image
    shap_2d = cv2.resize(shap_2d, (org_im.shape[1], org_im.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Symmetric normalization around 0
    max_abs = np.percentile(np.abs(shap_2d), 99)
    if max_abs < 1e-7:
        max_abs = np.max(np.abs(shap_2d))
    if max_abs < 1e-7:
        max_abs = 1.0
        
    normalized = np.clip(shap_2d / max_abs, -1.0, 1.0)
    
    # Create RGB color map:
    # Negative (-1..0): Blue/Cyan (0, 180, 255)
    # Neutral (0): original image tint
    # Positive (0..+1): Red/Coral (255, 45, 85)
    h, w = shap_2d.shape
    overlay = org_im.astype(np.float32).copy()
    
    # Weight map based on attribution magnitude
    abs_weight = np.abs(normalized)
    # Smooth thresholding so background remains visible
    blend_factor = np.clip(abs_weight * alpha, 0.0, alpha)
    blend_factor = np.expand_dims(blend_factor, axis=-1)
    
    pos_mask = normalized > 0
    neg_mask = normalized < 0
    
    color_layer = np.zeros((h, w, 3), dtype=np.float32)
    # Positives: Vibrant Coral Red [245, 60, 60]
    color_layer[pos_mask] = [245.0, 50.0, 60.0]
    # Negatives: Vibrant Cyan Blue [20, 140, 245]
    color_layer[neg_mask] = [20.0, 140.0, 245.0]
    
    blended = (1.0 - blend_factor) * overlay + blend_factor * color_layer
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_lime_superpixel_overlay(
    org_im: np.ndarray,
    segments: np.ndarray,
    top_segments: list,
    boundary_color: tuple = (255, 215, 0),
    positive_fill_color: tuple = (16, 185, 129),
    negative_fill_color: tuple = (239, 68, 68),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Apply LIME superpixel segment boundary and fill overlay.
    
    Args:
        org_im: Original image (H, W, 3) in RGB format.
        segments: 2D integer array (H, W) of superpixel labels.
        top_segments: List of tuples (segment_id, weight) or list of positive segment ids.
        boundary_color: RGB tuple for superpixel boundary lines (e.g. Gold/Yellow).
        positive_fill_color: RGB tuple for positive segments (e.g. Emerald Green).
        negative_fill_color: RGB tuple for negative segments (e.g. Rose/Red).
        alpha: Transparency for fill overlay.
        
    Returns:
        RGB image with highlighted superpixel segments.
    """
    h, w, _ = org_im.shape
    if segments.shape != (h, w):
        segments = cv2.resize(segments.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        
    overlay = org_im.copy().astype(np.float32)
    fill_layer = org_im.copy().astype(np.float32)
    mask_to_blend = np.zeros((h, w), dtype=bool)
    
    # Process segment weights
    for item in top_segments:
        if isinstance(item, (tuple, list)):
            seg_id, weight = item[0], item[1]
        else:
            seg_id, weight = item, 1.0
            
        seg_mask = (segments == seg_id)
        if not np.any(seg_mask):
            continue
            
        mask_to_blend[seg_mask] = True
        if weight >= 0:
            fill_layer[seg_mask] = positive_fill_color
        else:
            fill_layer[seg_mask] = negative_fill_color

    # Blend filled regions
    mask_3d = np.expand_dims(mask_to_blend, axis=-1)
    blended = np.where(mask_3d, (1.0 - alpha) * overlay + alpha * fill_layer, overlay)
    result = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Draw boundary outlines around superpixels using morphological gradient / Sobel
    grad_x = cv2.Sobel(segments.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(segments.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    boundaries = (np.abs(grad_x) + np.abs(grad_y)) > 0.1
    
    # Dilate boundaries slightly for crisp visibility
    kernel = np.ones((2, 2), np.uint8)
    boundaries = cv2.dilate(boundaries.astype(np.uint8), kernel).astype(bool)
    
    result[boundaries] = boundary_color
    return result


def encode_image_base64(image: np.ndarray) -> str:
    """
    Encode an image to base64 string.
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image for each detection.
    """
    annotated = image.copy()
    
    colors = [
        (99, 102, 241),   # Indigo
        (16, 185, 129),   # Emerald
        (245, 158, 11),   # Amber
        (239, 68, 68),    # Red
        (139, 92, 246),   # Violet
        (236, 72, 153),   # Pink
        (6, 182, 212),    # Cyan
        (251, 191, 36),   # Yellow
    ]
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', 0.0)
        
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 4), (x1 + label_w + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated
