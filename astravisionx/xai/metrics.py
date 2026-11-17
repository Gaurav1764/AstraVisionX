from typing import Dict, Any, List, Optional
import numpy as np

def binarize_heatmap(heatmap: np.ndarray, threshold_percentile: float = 75.0) -> np.ndarray:
    """
    Binarize a 2D continuous heatmap (values in [0, 1]) by taking top percentile.
    
    Returns:
        Boolean numpy array of shape (H, W).
    """
    if heatmap is None or heatmap.size == 0:
        return np.zeros((1, 1), dtype=bool)
    
    # Check if heatmap has variation
    max_val = np.max(heatmap)
    min_val = np.min(heatmap)
    if max_val <= min_val + 1e-6:
        return np.zeros_like(heatmap, dtype=bool)
        
    thresh = np.percentile(heatmap, threshold_percentile)
    # Ensure threshold isn't capturing the entire image
    thresh = max(thresh, min_val + 0.25 * (max_val - min_val))
    return heatmap >= thresh


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index) between two boolean masks.
    Both masks are resized/aligned to same shape if necessary.
    """
    if mask_a is None or mask_b is None:
        return 0.0
    
    if mask_a.shape != mask_b.shape:
        import cv2
        mask_b = cv2.resize(mask_b.astype(np.uint8), (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient between two boolean masks.
    """
    if mask_a is None or mask_b is None:
        return 0.0
        
    if mask_a.shape != mask_b.shape:
        import cv2
        mask_b = cv2.resize(mask_b.astype(np.uint8), (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_explanation_agreement(
    masks: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Computes pairwise and overall consistency metrics between explanation methods.
    
    Args:
        masks: Dict mapping method name ('gradcam', 'shap', 'lime') to boolean masks (H, W).
        
    Returns:
        Dict containing pairwise IoU, overall agreement score (0-100%), and qualitative description.
    """
    available_methods = [k for k, v in masks.items() if v is not None]
    
    if len(available_methods) < 2:
        return {
            "agreement_score_pct": 100.0,
            "consistency_level": "Single Method",
            "pairwise_iou": {},
            "summary_note": "Single explanation method active.",
            "methods_compared": available_methods
        }
        
    pairwise_iou = {}
    pairwise_dice = {}
    ious = []
    dices = []
    
    for i in range(len(available_methods)):
        for j in range(i + 1, len(available_methods)):
            m1 = available_methods[i]
            m2 = available_methods[j]
            iou = compute_mask_iou(masks[m1], masks[m2])
            dice = compute_mask_dice(masks[m1], masks[m2])
            pair_key = f"{m1}_vs_{m2}"
            pairwise_iou[pair_key] = round(iou * 100, 1)
            pairwise_dice[pair_key] = round(dice * 100, 1)
            ious.append(iou)
            dices.append(dice)
            
    avg_iou = float(np.mean(ious)) if ious else 0.0
    avg_dice = float(np.mean(dices)) if dices else 0.0
    agreement_score_pct = round(avg_iou * 100, 1)
    agreement_dice_pct = round(avg_dice * 100, 1)
    
    # Qualitative analysis based on genuine computed metrics
    m_names = [m.upper() for m in available_methods]
    if agreement_score_pct >= 50.0:
        level = "High"
        note = f"High visual agreement ({agreement_score_pct}% region overlap) across {', '.join(m_names)}. All methods isolate closely overlapping salient features, indicating high decision stability."
    elif agreement_score_pct >= 25.0:
        level = "Moderate"
        lowest_pair = min(pairwise_iou.items(), key=lambda x: x[1])
        p1, p2 = [p.upper() for p in lowest_pair[0].split("_vs_")]
        note = f"Moderate agreement ({agreement_score_pct}% average overlap). {p1} and {p2} show some spatial divergence ({lowest_pair[1]}% overlap), indicating partial sensitivity to surrounding context."
    else:
        level = "Low"
        note = f"Low explanation consistency ({agreement_score_pct}% overlap). Different explanation techniques attribute weight to distinct visual regions; prediction should be interpreted with caution."
        
    return {
        "agreement_score_pct": agreement_score_pct,
        "agreement_dice_pct": agreement_dice_pct,
        "consistency_level": level,
        "pairwise_iou": pairwise_iou,
        "pairwise_dice": pairwise_dice,
        "summary_note": note,
        "methods_compared": available_methods
    }
