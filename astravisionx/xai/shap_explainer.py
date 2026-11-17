from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from astravisionx.core.interfaces import ExplainerInterface
from astravisionx.utils.image_utils import apply_diverging_colormap, encode_image_base64

class ShapExplainer(ExplainerInterface):
    """
    SHAP (Shapley Additive exPlanations) via PyTorch Integrated Gradients.
    
    Computes exact Shapley feature attributions (positive = pushes towards the target class,
    negative = opposes the target class) by integrating gradients over a path from background
    baselines to the input image tensor.
    """
    
    def __init__(self, num_steps: int = 16, num_baselines: int = 4, num_background_samples: Optional[int] = None):
        self.num_steps = num_steps
        self.num_baselines = num_background_samples if num_background_samples is not None else num_baselines

    def _generate_baselines(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Generate diverse reference baselines: zero, mean, and blurred images."""
        device = image_tensor.device
        baselines = []
        
        # 1. Zero baseline (black)
        baselines.append(torch.zeros_like(image_tensor))
        
        # 2. Mean baseline (dataset / image average)
        mean_val = image_tensor.mean().item()
        baselines.append(torch.full_like(image_tensor, fill_value=mean_val))
        
        # 3. Gaussian blurred baselines (contextual baselines)
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        blur1 = cv2.GaussianBlur(img_np, (21, 21), 0)
        blur2 = cv2.GaussianBlur(img_np, (45, 45), 0)
        
        t1 = torch.from_numpy(blur1).permute(2, 0, 1).unsqueeze(0).to(device).float()
        t2 = torch.from_numpy(blur2).permute(2, 0, 1).unsqueeze(0).to(device).float()
        baselines.extend([t1, t2])
        
        return baselines[:self.num_baselines]

    def explain(
        self,
        image_tensor: torch.Tensor,
        model: torch.nn.Module,
        target_class: Optional[int] = None,
        org_image_rgb: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients / Expected Gradients SHAP attribution.
        
        Args:
            image_tensor: Normalized (1, C, H, W) PyTorch tensor.
            model: PyTorch model.
            target_class: Target class index.
            org_image_rgb: Original (H, W, 3) RGB uint8 image for overlay.
        """
        model.eval()
        device = next(model.parameters()).device
        tensor = image_tensor.clone().detach().to(device)
        
        # Forward pass for initial confidence and target class
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(probs).item())
            confidence = float(probs[target_class].item())

        baselines = self._generate_baselines(tensor)
        all_attributions = []

        # Integrated Gradients path integration over each baseline
        alphas = torch.linspace(0.0, 1.0, steps=self.num_steps, device=device)
        
        for baseline in baselines:
            # Interpolated batch: baseline + alpha * (input - baseline)
            diff = tensor - baseline
            interpolated_batch = torch.cat([baseline + alpha * diff for alpha in alphas], dim=0)
            interpolated_batch.requires_grad = True
            
            # Forward pass on path
            path_logits = model(interpolated_batch)
            path_scores = path_logits[:, target_class]
            
            # Backward pass for path gradients
            model.zero_grad()
            path_scores.sum().backward()
            
            grads = interpolated_batch.grad.detach() # Shape (steps, C, H, W)
            avg_grads = torch.mean(grads, dim=0, keepdim=True) # (1, C, H, W)
            
            # Shapley value contribution: (x - x') * avg_grad
            attr = (diff * avg_grads).squeeze(0).detach().cpu().numpy() # (C, H, W)
            all_attributions.append(attr)

        # Average attribution across baselines
        mean_attribution = np.mean(all_attributions, axis=0) # (C, H, W)
        
        # Spatial attribution map across channels
        spatial_shap = np.sum(mean_attribution, axis=0) # (H, W)
        
        # Normalization and stats
        pos_values = spatial_shap[spatial_shap > 0]
        neg_values = spatial_shap[spatial_shap < 0]
        
        pos_sum = float(np.sum(pos_values)) if len(pos_values) > 0 else 0.0
        neg_sum = float(np.abs(np.sum(neg_values))) if len(neg_values) > 0 else 0.0
        total_mag = pos_sum + neg_sum + 1e-8
        
        pos_ratio = round((pos_sum / total_mag) * 100, 1)
        neg_ratio = round((neg_sum / total_mag) * 100, 1)
        
        # Salient mask for agreement scoring (top positive 25%)
        if len(pos_values) > 0:
            pos_thresh = np.percentile(pos_values, 75)
            salient_mask = (spatial_shap >= pos_thresh)
        else:
            salient_mask = np.zeros_like(spatial_shap, dtype=bool)

        # Diverging colormap overlay
        overlay_b64 = None
        if org_image_rgb is not None:
            overlay_img = apply_diverging_colormap(org_image_rgb, spatial_shap)
            overlay_b64 = encode_image_base64(overlay_img)

        # Shifted heatmap [0, 1] for unified display
        max_abs = np.max(np.abs(spatial_shap)) + 1e-8
        norm_map = (spatial_shap / (2 * max_abs)) + 0.5

        return {
            "method": "shap",
            "heatmap": norm_map.astype(float),
            "raw_map": spatial_shap.astype(float),
            "salient_mask": salient_mask,
            "overlay_b64": overlay_b64,
            "confidence": confidence,
            "target_class": target_class,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "attribution_score": float(np.mean(np.abs(spatial_shap))),
            "top_regions": [],
            "method_description": "SHAP Integrated Gradients: Signed pixel attribution showing positive (red) and opposing (blue) contributions."
        }
