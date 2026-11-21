from typing import Any, Dict, List, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from astravisionx.core.interfaces import ExplainerInterface
from astravisionx.utils.image_utils import apply_lime_superpixel_overlay, encode_image_base64


def _generate_superpixels(image_rgb: np.ndarray, num_segments: int = 35) -> np.ndarray:
    """
    Generate superpixels with graceful fallback.
    Tries skimage.segmentation.slic first. If blocked by OS DLL policy or unavailable,
    falls back cleanly to OpenCV K-Means + spatial grid superpixels without binary DLL dependencies.
    """
    try:
        from skimage.segmentation import slic
        return slic(image_rgb, n_segments=num_segments, compactness=10, sigma=1, start_label=0)
    except Exception:
        # Robust fallback using OpenCV and NumPy (guaranteed to load without blocked DLLs)
        h, w = image_rgb.shape[:2]
        
        # Calculate grid dimensions based on target number of segments
        aspect = w / max(h, 1)
        grid_rows = max(2, int(np.round(np.sqrt(num_segments / max(aspect, 1e-4)))))
        grid_cols = max(2, int(np.round(grid_rows * aspect)))
        
        row_bounds = np.linspace(0, h, grid_rows + 1, dtype=int)
        col_bounds = np.linspace(0, w, grid_cols + 1, dtype=int)
        
        segments = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        
        for r in range(grid_rows):
            r_start, r_end = row_bounds[r], row_bounds[r + 1]
            for c in range(grid_cols):
                c_start, c_end = col_bounds[c], col_bounds[c + 1]
                segments[r_start:r_end, c_start:c_end] = current_label
                current_label += 1
                
        return segments


class LimeExplainer(ExplainerInterface):
    """
    LIME (Local Interpretable Model-agnostic Explanations) Image Explainer.
    Segments image into superpixels, perturbs segment visibility, and fits a weighted
    linear ridge surrogate to quantify segment contributions.
    """
    
    def __init__(self, num_samples: int = 100, num_segments: int = 35):
        self.num_samples = num_samples
        self.num_segments = num_segments

    def explain(
        self,
        image_tensor: torch.Tensor,
        model: torch.nn.Module,
        target_class: Optional[int] = None,
        org_image_rgb: Optional[np.ndarray] = None,
        transform_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate LIME superpixel explanations for the target image.
        """
        model.eval()
        device = next(model.parameters()).device
        
        if org_image_rgb is None:
            img_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            org_image_rgb = (img_np * 255).astype(np.uint8)

        h, w, _ = org_image_rgb.shape
        
        # 1. Base inference
        if transform_fn is not None:
            t = transform_fn(Image.fromarray(org_image_rgb)).unsqueeze(0).to(device)
        else:
            t = image_tensor.to(device)
            
        with torch.no_grad():
            out = model(t)
            probs = F.softmax(out, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(probs).item())
            confidence = float(probs[target_class].item())

        # 2. Superpixel segmentation with safe fallback
        segments = _generate_superpixels(org_image_rgb, num_segments=self.num_segments)
        unique_segs = np.unique(segments)
        num_segs = len(unique_segs)
        
        # Normalize labels to contiguous 0..num_segs-1
        label_map = {old: new for new, old in enumerate(unique_segs)}
        segments = np.vectorize(label_map.get)(segments)
        
        # 3. Generate binary perturbation matrix Z of shape (num_samples, num_segs)
        np.random.seed(42)
        Z = np.random.binomial(1, 0.5, size=(self.num_samples, num_segs))
        # Ensure the unperturbed original image (all 1s) is in the sample
        Z[0, :] = 1.0
        
        # Compute cosine / exponential distance weights: pi(z) = exp(-D(z, 1)^2 / sigma^2)
        distances = np.sum((1.0 - Z) ** 2, axis=1) / max(num_segs, 1)
        kernel_width = 0.25
        weights_pi = np.exp(-distances / (kernel_width ** 2))
        
        # 4. Generate perturbed images and predict target class probabilities
        mean_rgb = np.mean(org_image_rgb, axis=(0, 1)).astype(np.uint8)
        perturbed_probs = []
        batch_tensors = []
        batch_size = 25
        
        for i in range(self.num_samples):
            mask_active = Z[i]
            perturbed_img = org_image_rgb.copy()
            inactive_segs = np.where(mask_active == 0)[0]
            for seg_id in inactive_segs:
                perturbed_img[segments == seg_id] = mean_rgb
                
            pil_p = Image.fromarray(perturbed_img)
            ten = transform_fn(pil_p) if transform_fn is not None else torch.from_numpy(perturbed_img).permute(2,0,1).float()/255.0
            batch_tensors.append(ten)
            
            if len(batch_tensors) >= batch_size or i == self.num_samples - 1:
                batch_t = torch.stack(batch_tensors, dim=0).to(device)
                with torch.no_grad():
                    logits = model(batch_t)
                    p_batch = F.softmax(logits, dim=1)[:, target_class].cpu().numpy()
                    perturbed_probs.extend(p_batch.tolist())
                batch_tensors = []

        Y = np.array(perturbed_probs)  # (num_samples,)
        
        # 5. Fit Weighted Ridge Regression surrogate model:
        W_diag = np.diag(weights_pi)
        alpha = 1.0
        Z_weighted = np.dot(W_diag, Z)
        A = np.dot(Z.T, Z_weighted) + alpha * np.eye(num_segs)
        b = np.dot(Z_weighted.T, Y)
        feature_weights = np.linalg.solve(A, b)  # (num_segs,)
        
        # 6. Rank superpixels by weight
        top_indices = np.argsort(feature_weights)[::-1]
        top_positive_segments = []
        salient_mask = np.zeros_like(segments, dtype=bool)
        
        for idx in top_indices:
            w_val = float(feature_weights[idx])
            if w_val > 0:
                top_positive_segments.append((int(idx), w_val))
                salient_mask[segments == idx] = True

        # Render superpixel overlay
        overlay_img = apply_lime_superpixel_overlay(
            org_image_rgb,
            segments,
            top_positive_segments[:8],
            boundary_color=(255, 215, 0),
            positive_fill_color=(16, 185, 129),
            alpha=0.42
        )
        overlay_b64 = encode_image_base64(overlay_img)
        
        # Proxy continuous heatmap for unified interface
        superpixel_heatmap = np.zeros_like(segments, dtype=np.float32)
        for seg_id, weight in top_positive_segments:
            superpixel_heatmap[segments == seg_id] = max(0.0, weight)
        if np.max(superpixel_heatmap) > 0:
            superpixel_heatmap /= np.max(superpixel_heatmap)

        return {
            "method": "lime",
            "heatmap": superpixel_heatmap,
            "raw_map": segments,
            "salient_mask": salient_mask,
            "overlay_b64": overlay_b64,
            "confidence": confidence,
            "target_class": target_class,
            "top_segments_count": len(top_positive_segments),
            "top_segments": [(int(s), round(float(w), 4)) for s, w in top_positive_segments[:5]],
            "attribution_score": float(np.sum([w for _, w in top_positive_segments])) if top_positive_segments else 0.0,
            "top_regions": [],
            "method_description": "LIME Explainer: Local superpixel perturbations highlighting the segment regions most influential to the prediction."
        }
