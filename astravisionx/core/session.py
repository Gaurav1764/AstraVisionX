"""
Session Manager for AstraVisionX.

Maintains an in-memory cache of per-image analysis results so that
the /chat, /explain/method, and /report endpoints can retrieve them without
re-running inference. Each session is tied to one uploaded image.
"""

import uuid
import time
import threading
from typing import Any, Dict, Optional

from astravisionx.config import settings


class Session:
    """Represents one analysis session tied to a single uploaded image."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()

        # Populated by /analyze
        self.filename: str = ""
        self.original_image_b64: str = ""
        self.annotated_image_b64: str = ""
        self.gradcam_image_b64: Optional[str] = None
        self.detections: list = []
        self.classifications: list = []
        self.gradcam_info: dict = {}
        self.nl_summary: str = ""
        self.active_filters: Optional[dict] = None

        # Explainability Cache: maps method name ('gradcam', 'shap', 'lime') -> result dict
        self.xai_cache: Dict[str, dict] = {}
        # Per-detection Explainability Cache: maps detection_index -> {method: result_dict}
        self.detection_xai: Dict[int, Dict[str, dict]] = {}
        # Cached binary salient masks for consistency computation
        self.salient_masks: Dict[str, Any] = {}
        # Computed multi-method agreement metrics
        self.agreement_metrics: Dict[str, Any] = {}
        # Comprehensive confidence breakdown
        self.confidence_breakdown: Dict[str, Any] = {}

        # Raw image buffer bytes for on-demand XAI computation
        self.image_bytes: Optional[bytes] = None

        # Chat history — list of {"role": "user"|"assistant", "content": str}
        self.chat_history: list = []

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > (settings.session_ttl_minutes * 60)

    def to_context_string(self) -> str:
        """
        Build a plain-text context block that summarises the analysis results.
        This is injected into the chatbot's system prompt so the LLM can
        answer user questions grounded in the actual data.
        """
        lines = [
            f"Image filename: {self.filename}",
            "",
        ]
        if self.active_filters:
            cats = self.active_filters.get("categories", [])
            query = self.active_filters.get("search_query", "")
            min_c = self.active_filters.get("min_confidence", 0.0)
            filter_desc = []
            if cats:
                filter_desc.append(f"Categories: {', '.join(cats)}")
            if query:
                filter_desc.append(f"Search: '{query}'")
            if min_c > 0:
                filter_desc.append(f"Min Confidence: {min_c:.0%}")
            if filter_desc:
                lines.append(f"=== Active User View Filter: {' | '.join(filter_desc)} ===")
                lines.append("")

        lines.append("=== YOLO Object Detections ===")
        if self.detections:
            for i, d in enumerate(self.detections, 1):
                parent_info = f", parent: det_{d['parent_index']}" if d.get('parent_index') is not None else ""
                cat_info = f", category: {d.get('category', 'unknown')}"
                lines.append(
                    f"  {i}. {d.get('class_name', 'unknown')} — "
                    f"confidence {d.get('confidence', 0):.1%}, "
                    f"bbox {[round(v) for v in d.get('bbox', [])]}"
                    f"{cat_info}{parent_info}"
                )
        else:
            lines.append("  No objects detected.")

        lines.append("")
        lines.append("=== Scene Classifications (Top-5) ===")
        for i, c in enumerate(self.classifications, 1):
            label = c.get("class_name", f"Class {c.get('class_id', '?')}")
            lines.append(f"  {i}. {label} — confidence {c.get('confidence', 0):.1%}")

        lines.append("")
        lines.append("=== Multi-Method Explainability (XAI) ===")
        for method, res in self.xai_cache.items():
            lines.append(f"  [{method.upper()}]: Target Class '{res.get('target_class_name', 'N/A')}', Score/Conf: {res.get('confidence', 0):.2%}")
            if "positive_ratio" in res:
                lines.append(f"    - Positive Attributions: {res.get('positive_ratio')}%, Negative: {res.get('negative_ratio')}%")
            if "top_segments_count" in res:
                lines.append(f"    - Contributing Superpixels: {res.get('top_segments_count')}")

        if self.agreement_metrics and "agreement_score_pct" in self.agreement_metrics:
            lines.append("")
            lines.append("=== Explainability Agreement & Consistency ===")
            lines.append(f"  Consistency Score (IoU): {self.agreement_metrics.get('agreement_score_pct')}% ({self.agreement_metrics.get('consistency_level')})")
            if "agreement_dice_pct" in self.agreement_metrics:
                lines.append(f"  Dice Overlap Score: {self.agreement_metrics.get('agreement_dice_pct')}%")
            lines.append(f"  Note: {self.agreement_metrics.get('summary_note')}")
            if self.agreement_metrics.get("pairwise_iou"):
                for pair, val in self.agreement_metrics["pairwise_iou"].items():
                    dice_val = self.agreement_metrics.get("pairwise_dice", {}).get(pair, "N/A")
                    lines.append(f"    - {pair}: {val}% IoU, {dice_val}% Dice")

        if self.confidence_breakdown:
            lines.append("")
            lines.append("=== Confidence & Reliability Breakdown ===")
            lines.append(f"  Top Class Confidence: {self.confidence_breakdown.get('raw_confidence', 0):.1%}")
            lines.append(f"  Decision Boundary Margin (Gap to #2): {self.confidence_breakdown.get('decision_margin', 0):.1%}")
            lines.append(f"  Reliability Status: {self.confidence_breakdown.get('reliability_status', 'N/A')}")

        if self.nl_summary:
            lines.append("")
            lines.append("=== Natural-Language Summary ===")
            lines.append(self.nl_summary)

        return "\n".join(lines)


class SessionManager:
    """Thread-safe in-memory session store with auto-expiry."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create_session(self) -> Session:
        """Create a new session and return it."""
        session_id = uuid.uuid4().hex[:12]
        session = Session(session_id)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID, or None if expired / not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                return None
            return session

    def cleanup_expired(self):
        """Remove all expired sessions. Called periodically."""
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
            for sid in expired:
                del self._sessions[sid]


# Global singleton
session_manager = SessionManager()
