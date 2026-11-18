from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Literal, Union

class Settings(BaseSettings):
    debug: bool = True
    use_cuda: bool = False
    
    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            if v.lower() in ("true", "1", "t", "yes", "y"):
                return True
            if v.lower() in ("false", "0", "f", "no", "n"):
                return False
            # Handle log levels like 'DEBUG', 'WARN', 'INFO', etc.
            return v.upper() == "DEBUG"
        return bool(v)
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Model Settings (Expanded Open Images V7 601-class checkpoint)
    yolo_model_path: str = "yolov8s-oiv7.pt"
    vit_model_name: str = "vit_base_patch16_224"
    confidence_threshold: float = 0.60
    detector_conf_threshold: float = 0.10
    
    # Model mode: "general" uses stock ImageNet/COCO models with appropriate caveats.
    # "medical" would use domain-specific checkpoints (not yet available — stubs
    # a TODO and enforces stronger disclaimers until a real checkpoint is wired in).
    model_mode: str = "medical"
    
    # LLM / Chatbot Settings
    gemini_api_key: str = ""
    
    # Session Settings
    session_ttl_minutes: int = 30
    
    # Category Filter & Demographic Settings
    enable_demographic_estimates: bool = False
    
    # Disclaimer
    medical_disclaimer: str = "For research and educational purposes only — not a diagnostic tool."
    
    @property
    def is_medical_mode(self) -> bool:
        return self.model_mode.lower() == "medical"
    
    @property
    def has_medical_checkpoint(self) -> bool:
        """
        Returns True only when a real medical-domain model checkpoint is loaded.
        Currently always False — will flip to True once we wire in CheXNet or
        a LoRA-fine-tuned model.
        """
        # TODO: Set to True once a medical checkpoint (e.g. CheXNet, LoRA-tuned
        # DenseNet on ChestX-ray14) is loaded via vit_model_name or a dedicated
        # medical_classifier_path setting.
        return False
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
        
settings = Settings()
