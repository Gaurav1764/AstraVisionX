from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

from astravisionx.core.interfaces import ClassifierInterface
from astravisionx.config import settings

class ViTClassifier(ClassifierInterface):
    def __init__(self, model_name: str = None):
        """
        Initialize the ViT classifier using timm.
        """
        model_name = model_name or settings.vit_model_name
        self.device = torch.device("cuda" if settings.use_cuda and torch.cuda.is_available() else "cpu")
        
        # We load a pretrained Vision Transformer model.
        # This will download the weights if not cached.
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get ImageNet class labels (Timm doesn't provide them directly easily, 
        # but we can fetch them or just return indices for now).
        # We will use dummy labels or just the class IDs.
        
        # Standard ImageNet transforms
        # For ViT base patch16 224
        data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def classify(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify the scene/image.
        """
        # Convert numpy array (BGR/RGB) to PIL Image
        # Assuming input is RGB (or we handle conversion outside)
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
            
        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
            
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probs, 5)
        
        results = []
        for i in range(top5_prob.size(0)):
            results.append({
                "class_id": int(top5_catid[i].item()),
                "confidence": float(top5_prob[i].item()),
                # "class_name": "TODO" # Needs imagenet classes mapping
            })
            
        return results
