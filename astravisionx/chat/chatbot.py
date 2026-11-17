"""
Session-scoped Chatbot for AstraVisionX.

Provides conversational Q&A grounded in a specific image's analysis results.
Uses Google Gemini as the LLM when an API key is configured; otherwise
falls back to a smart context-aware response engine that answers common
questions about detections, classifications, and Grad-CAM without an LLM.

Why per-session?  Each uploaded image has its own analysis context (detections,
classifications, heatmaps).  The chatbot must answer questions about *that*
specific image, not act as a general-purpose assistant.  Conversation memory
lets follow-up questions ("what about the second one?") resolve correctly.
"""

from __future__ import annotations
from astravisionx.config import settings
from astravisionx.core.session import Session


def chat(session: Session, user_message: str) -> str:
    """
    Process a user message within the given session and return the assistant's
    response.  Appends both the user message and the response to the session's
    chat_history.
    """
    # Record the user message
    session.chat_history.append({"role": "user", "content": user_message})

    if not settings.gemini_api_key:
        response = "I'm currently unable to reach the reasoning model — check that GEMINI_API_KEY is set in your .env file."
    else:
        response = _chat_with_gemini(session, user_message)

    # Record the assistant response
    session.chat_history.append({"role": "assistant", "content": response})
    return response


# ---------------------------------------------------------------------------
# Gemini-powered chat
# ---------------------------------------------------------------------------

def _get_optimized_image_bytes(session: Session) -> Optional[bytes]:
    """Generates and caches a lightweight 512px JPEG thumbnail for fast vision API calls."""
    if hasattr(session, "chat_thumbnail_bytes") and session.chat_thumbnail_bytes:
        return session.chat_thumbnail_bytes

    try:
        import io
        import base64
        from PIL import Image
        raw_bytes = session.image_bytes
        if not raw_bytes and session.original_image_b64:
            b64_str = session.original_image_b64
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)

        if raw_bytes:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=75, optimize=True)
            session.chat_thumbnail_bytes = buf.getvalue()
            return session.chat_thumbnail_bytes
    except Exception as e:
        print(f"Error optimizing chat image thumbnail: {e}")
    return session.image_bytes


def _chat_with_gemini(session: Session, user_message: str) -> str:
    """Use Google Gemini with full conversation history + analysis context + optimized image payload."""
    try:
        from google.genai import types
        from astravisionx.chat.gemini_service import generate_with_fallback

        system_instruction = (
            "You are the AstraVisionX AI assistant — a friendly, natural, expert multimodal vision chatbot. "
            "Answer the user's questions concisely (2-4 clear sentences) based on the image analysis data and attached image.\n\n"
            "GUIDELINES:\n"
            "1. Speak naturally and directly. Avoid repeating disclaimers on every turn.\n"
            "2. If a detection is tagged 'Uncertain match', explain that the detector's vocabulary may have misclassified it.\n"
            "3. If asked about colors, positions, or visual details, inspect the attached image accurately.\n"
            "4. Format responses cleanly for text and voice readouts.\n\n"
            "--- ANALYSIS CONTEXT ---\n"
            f"{session.to_context_string()}\n"
            "--- END ANALYSIS CONTEXT ---"
        )

        # Build conversation history for Gemini (keep last 6 messages for speed and context)
        contents = []
        recent_history = session.chat_history[-7:-1] if len(session.chat_history) > 1 else []
        for msg in recent_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

        # Add the current user message AND the lightweight thumbnail image
        image_parts = [types.Part.from_text(text=user_message)]
        opt_bytes = _get_optimized_image_bytes(session)
        if opt_bytes:
            image_parts.append(types.Part.from_bytes(data=opt_bytes, mime_type="image/jpeg"))

        contents.append(types.Content(role="user", parts=image_parts))

        return generate_with_fallback(contents, system_instruction=system_instruction, max_output_tokens=300)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"I'm currently unable to reach the reasoning model. API Error: {str(e)}"
