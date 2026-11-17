from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from astravisionx.core.interfaces import ExplainerInterface

class GradCamExplainer(ExplainerInterface):
    def __init__(self, target_layer_name: str = None):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            target_layer_name: The name of the layer to attach hooks to. 
                               If None, we'll try to automatically find the last spatial layer.
        """
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
    def _save_gradient(self, module, grad_input, grad_output):
        """
        Hook to save the gradients during the backward pass.
        Why: Grad-CAM needs the gradients of the target class score with respect to the feature maps 
        to understand how sensitive the prediction is to each feature channel.
        """
        # grad_output is a tuple containing the gradients wrt the output of the layer
        self.gradients = grad_output[0].detach()
        
    def _save_activation(self, module, input, output):
        """
        Hook to save the activations during the forward pass.
        Why: We need the actual feature maps (activations) to know which spatial regions 
        were activated by the input image.
        """
        self.activations = output.detach()
        
    def explain(self, image_tensor: torch.Tensor, model: torch.nn.Module, target_class: int = None) -> Dict[str, Any]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_tensor: The preprocessed image tensor (1, C, H, W).
            model: The PyTorch model to explain.
            target_class: The class index to explain. If None, uses the model's top prediction.
        """
        model.eval()
        
        # 1. Attach hooks to the target layer.
        # For this example, we assume model is a CNN where we can find a layer.
        # If it's a ViT, we need to reshape the token sequence to a 2D grid.
        # Here we do a generic CNN approach, but we can extend it.
        
        # We need to find the target layer if not provided. 
        # A simple heuristic for timm models (like ResNet or CNNs) is finding the last Conv2d.
        target_layer = None
        if self.target_layer_name:
            for name, module in model.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break
        else:
            # Fallback: just find the last Conv2d layer or the last block
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    
        if target_layer is None:
            raise ValueError("Could not find a target layer for Grad-CAM.")
            
        # Register hooks
        handle_forward = target_layer.register_forward_hook(self._save_activation)
        handle_backward = target_layer.register_full_backward_hook(self._save_gradient)
        
        # 2. Forward pass
        # Why: We need to get the class scores to determine what the model predicted,
        # and to trigger the forward hook which saves our activations.
        image_tensor.requires_grad = True
        output = model(image_tensor)
        
        if target_class is None:
            # If no target class specified, explain the top prediction
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        
        # 3. Backward pass
        # Why: We backpropagate from the specific class score. This sends gradients 
        # flowing backwards through the network, triggering our backward hook to save 
        # the gradients at the target layer.
        model.zero_grad()
        score.backward()
        
        # Remove hooks to avoid memory leaks
        handle_forward.remove()
        handle_backward.remove()
        
        # 4. Compute weights (Global Average Pooling of gradients)
        # Why: The gradients represent the importance of each feature map. By pooling them 
        # spatially, we get a single weight (alpha) per channel, indicating how important 
        # that entire channel is for predicting the target class.
        gradients = self.gradients[0] # Shape: (C, H, W)
        activations = self.activations[0] # Shape: (C, H, W)
        
        # If the model is a ViT, the shape might be (Num_Tokens, Embed_Dim) instead of (C, H, W).
        # We handle the CNN case (C, H, W) here.
        if len(gradients.shape) == 3:
            weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # Shape: (C, 1, 1)
            
            # 5. Weighted combination of activations
            # Why: We multiply each feature map by its importance weight and sum them up. 
            # This highlights regions that positively contributed to the class prediction.
            cam = torch.sum(weights * activations, dim=0) # Shape: (H, W)
            
            # 6. ReLU and Normalization
            # Why: ReLU ensures we only consider features that have a *positive* influence 
            # on the class score. We don't care about regions that suppress the class.
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            
            # Normalize between 0 and 1
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            
            # Normalize confidence: the raw score is a logit which can be negative.
            # We apply softmax to the full output vector to get a proper probability
            # in [0, 1] that is meaningful to display to the user.
            raw_logit = float(score.item())
            probabilities = F.softmax(output, dim=1)[0]
            normalized_confidence = float(probabilities[target_class].item())
            # Clamp to [0, 1] as a safety net
            normalized_confidence = max(0.0, min(1.0, normalized_confidence))
                
            return {
                "heatmap": cam,
                "confidence": normalized_confidence,
                "raw_logit": raw_logit,  # kept for debugging
                "target_class": target_class,
                "top_regions": [] # Can be populated using contour detection on the heatmap later
            }
        else:
            raise NotImplementedError("Grad-CAM for non-spatial outputs (like raw ViT tokens) requires reshaping logic.")
