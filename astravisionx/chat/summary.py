"""
Natural-Language Summary Generator for AstraVisionX.

Generates a readable narrative from structured analysis results.
Uses Google Gemini if an API key is configured; otherwise falls back
to a template-based approach that still produces useful output.

MODE-AWARE:  When MODEL_MODE is "medical" or no medical checkpoint is
loaded, the summary leads with an explicit caveat about general-purpose
models.  Zero-YOLO-detection on medical images is framed honestly (COCO
model has no medical classes), not presented as "nothing found."
"""

from astravisionx.config import settings


def _model_caveat() -> str:
    """
    Build the leading caveat sentence based on model mode and checkpoint status.
    """
    if settings.has_medical_checkpoint:
        return ""

    if settings.is_medical_mode:
        return "ℹ️ NOTE: This analysis uses general-purpose models for reference.\n\n"
    else:
        return "ℹ️ NOTE: This analysis uses general-purpose models for reference.\n\n"


def generate_summary(
    detections: list,
    classifications: list,
    gradcam_info: dict,
    filename: str = "uploaded image",
) -> str:
    """
    Produce a natural-language summary of the analysis results.

    Why two paths?  We want the system to work out-of-the-box without
    an API key (template mode), but produce richer prose when an LLM
    is available (Gemini mode).
    """
    if settings.gemini_api_key:
        return _generate_with_gemini(detections, classifications, gradcam_info, filename)
    return _generate_template(detections, classifications, gradcam_info, filename)


def _generate_with_gemini(
    detections: list,
    classifications: list,
    gradcam_info: dict,
    filename: str,
) -> str:
    """Use Google Gemini to produce a rich narrative summary."""
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)

        context = _build_data_block(detections, classifications, gradcam_info, filename)
        caveat = _model_caveat()

        # The LLM prompt instructs Gemini to lead with the caveat and frame
        # zero detections honestly when using general-purpose models.
        prompt = (
            "You are an AI image analysis assistant for AstraVisionX. "
            "Given the structured analysis results below, write a concise, "
            "professional natural-language summary (4-6 sentences).\n\n"
        )

        if not settings.has_medical_checkpoint:
            prompt += (
                "CRITICAL INSTRUCTIONS:\n"
                "1. Briefly mention the model used is general-purpose (ImageNet/COCO).\n"
                "2. Confidently summarize the detected objects and their confidence scores.\n"
                "3. Summarize the scene classification and Grad-CAM focus areas.\n"
                "4. Be factual, concise, and professional without sounding apologetic or overly negative.\n\n"
            )
        else:
            prompt += "Be factual — do not speculate beyond the data.\n\n"

        prompt += (
            f"{context}\n\n"
            f"⚠️ {settings.medical_disclaimer}"
        )

        from astravisionx.chat.gemini_service import generate_with_fallback
        return generate_with_fallback(prompt)

    except Exception as e:
        # Fall back to template if Gemini fails
        return _generate_template(detections, classifications, gradcam_info, filename)


def _generate_template(
    detections: list,
    classifications: list,
    gradcam_info: dict,
    filename: str,
) -> str:
    """
    Template-based fallback summary.  No external API needed.

    Why templates?  They guarantee deterministic, readable output even
    when no LLM is available.  The structure mirrors what a human analyst
    would write: caveat → detections → classification → explainability.
    """
    parts = []

    # --- Leading caveat (always first, never buried) ---
    caveat = _model_caveat()
    if caveat:
        parts.append(caveat.strip())

    # --- Detections summary ---
    if detections:
        # Group by class name and count occurrences
        class_counts: dict[str, int] = {}
        max_conf = 0.0
        for d in detections:
            name = d.get("class_name", "unknown")
            class_counts[name] = class_counts.get(name, 0) + 1
            max_conf = max(max_conf, d.get("confidence", 0))

        obj_parts = []
        for name, count in class_counts.items():
            obj_parts.append(f"{count} {name}{'s' if count > 1 else ''}")
        objects_str = ", ".join(obj_parts)

        parts.append(
            f"The YOLO object detector identified {len(detections)} object(s) in "
            f"the image: {objects_str}. "
            f"The highest detection confidence was {max_conf:.1%}."
        )
    else:
        # Zero detections — frame honestly based on mode
        if settings.is_medical_mode and not settings.has_medical_checkpoint:
            parts.append(
                "The object detector (YOLOv8, trained on the COCO dataset of everyday "
                "objects) returned zero detections. This is expected — the COCO model "
                "has no medical object classes (fractures, lesions, foreign bodies, etc.) "
                "and cannot identify clinical findings. Zero detections here is not a "
                "negative diagnostic result."
            )
        else:
            parts.append(
                "The YOLO object detector (COCO-trained) did not identify any objects "
                "in the image. Note: this detector is trained on general-purpose "
                "categories and may not cover domain-specific objects."
            )

    # --- Classification summary ---
    if classifications:
        top = classifications[0]
        label = top.get("class_name", f"Class {top.get('class_id', '?')}")
        conf = top.get("confidence", 0)

        if not settings.has_medical_checkpoint:
            # Emphasize that these are ImageNet labels, not clinical findings
            parts.append(
                f"The ViT classifier (ImageNet-1k) returned \"{label}\" as the "
                f"top prediction with {conf:.1%} confidence. This is a general-purpose "
                f"label from the ImageNet dataset and is not a medical finding."
            )
        else:
            parts.append(
                f"The classifier's top prediction is \"{label}\" "
                f"with {conf:.1%} confidence."
            )

    # --- Grad-CAM summary ---
    if gradcam_info and not gradcam_info.get("error"):
        target = gradcam_info.get("target_class_name", "the predicted class")
        gc_conf = gradcam_info.get("confidence", 0)
        parts.append(
            f"Grad-CAM explainability highlights the image regions most influential "
            f"for the model's prediction of \"{target}\" (confidence {gc_conf:.1%}). "
            f"The heatmap overlay shows where the model focused its attention."
        )

    return "\n\n".join(parts)


def _build_data_block(
    detections: list,
    classifications: list,
    gradcam_info: dict,
    filename: str,
) -> str:
    """Build a structured text block for the LLM prompt."""
    lines = [f"Filename: {filename}"]
    lines.append(f"Model mode: {settings.model_mode}")
    lines.append(f"Medical checkpoint loaded: {settings.has_medical_checkpoint}")
    lines.append("")

    lines.append("YOLO Detections:")
    if detections:
        for i, d in enumerate(detections, 1):
            lines.append(f"  {i}. {d.get('class_name', '?')} (confidence: {d.get('confidence', 0):.2%})")
    else:
        lines.append("  Zero detections")

    lines.append("")
    lines.append("ViT Classifications:")
    for i, c in enumerate(classifications, 1):
        label = c.get("class_name", f"Class {c.get('class_id', '?')}")
        lines.append(f"  {i}. {label} (confidence: {c.get('confidence', 0):.2%})")

    lines.append("")
    lines.append("Grad-CAM:")
    if gradcam_info.get("error"):
        lines.append(f"  Error: {gradcam_info['error']}")
    else:
        lines.append(f"  Target: {gradcam_info.get('target_class_name', 'N/A')}")
        lines.append(f"  Confidence: {gradcam_info.get('confidence', 0):.4f}")

    return "\n".join(lines)
