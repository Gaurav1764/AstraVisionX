from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np

class DetectorInterface(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: numpy array (H, W, C) representing the image.
            
        Returns:
            List of dictionaries containing bounding box, class name, and confidence.
        """
        pass

class ClassifierInterface(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify the scene/image.
        
        Args:
            image: numpy array (H, W, C) representing the image.
            
        Returns:
            List of dictionaries containing class name and confidence.
        """
        pass

class ExplainerInterface(ABC):
    @abstractmethod
    def explain(self, image: np.ndarray, model: Any, target_class: int = None) -> Dict[str, Any]:
        """
        Explain the model's prediction on the image.
        
        Args:
            image: numpy array (H, W, C) representing the image.
            model: The model to explain.
            target_class: Target class index for explanation.
            
        Returns:
            Dictionary containing heatmap array, confidence, and top contributing regions.
        """
        pass
