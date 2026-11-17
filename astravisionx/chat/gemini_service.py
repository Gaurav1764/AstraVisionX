import os
from typing import List, Optional
from google import genai
from google.genai import types
from astravisionx.config import settings

# Candidate models in priority order of speed, stability & availability
CANDIDATE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]

# Cache of last successful model & key to avoid failover latency on subsequent calls
_cached_working_model: Optional[str] = None
_cached_working_key: Optional[str] = None

def get_all_gemini_keys() -> List[str]:
    """Retrieve all configured Gemini API keys from settings and api_keys.txt."""
    keys = []
    
    # 1. Key from settings / .env
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        keys.append(settings.gemini_api_key.strip())
        
    # 2. Keys from api_keys.txt
    api_keys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "api_keys.txt")
    if os.path.exists(api_keys_path):
        try:
            with open(api_keys_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and line not in keys:
                        keys.append(line)
        except Exception as e:
            print(f"Error reading api_keys.txt: {e}")
            
    return keys


def generate_with_fallback(
    contents: any,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 350,
) -> str:
    """
    Executes generate_content across candidate keys and models with automatic
    failover. Prioritizes the last successful model/key for sub-second latency.
    """
    global _cached_working_model, _cached_working_key
    keys = get_all_gemini_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY found in .env or api_keys.txt.")

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=0.25,
    ) if system_instruction else types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        temperature=0.25,
    )

    # 1. Try cached working key + model first for instant response
    if _cached_working_model and _cached_working_key and _cached_working_key in keys:
        try:
            client = genai.Client(api_key=_cached_working_key)
            resp = client.models.generate_content(
                model=_cached_working_model,
                contents=contents,
                config=config,
            )
            if resp and resp.text:
                return resp.text.strip()
        except Exception:
            # Invalidate cache if failed
            _cached_working_model = None
            _cached_working_key = None

    # 2. Cascade across candidate models and keys
    last_error = None
    for model_name in CANDIDATE_MODELS:
        for key in keys:
            try:
                client = genai.Client(api_key=key)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                if resp and resp.text:
                    # Save working configuration
                    _cached_working_model = model_name
                    _cached_working_key = key
                    return resp.text.strip()
            except Exception as err:
                last_error = err
                continue

    if last_error:
        raise last_error
    raise RuntimeError("All Gemini candidate models and keys failed.")
