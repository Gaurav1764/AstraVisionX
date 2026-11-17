"""
AstraVisionX — Example: Basic Image Analysis Pipeline

This example demonstrates how to use AstraVisionX programmatically:
  1. Detect objects with YOLOv8
  2. Classify the scene with ViT
  3. Generate Grad-CAM, Integrated Gradients, and LIME explanations
  4. Compute multi-method agreement

Usage:
    python examples/basic_pipeline.py [path/to/image.jpg]
"""

import sys
import cv2
import numpy as np
import timm
import torch
from PIL import Image

from astravisionx.models.yolo_detector import YoloDetector
from astravisionx.models.vit_classifier import ViTClassifier
from astravisionx.xai import (
    GradCamExplainer,
    ShapExplainer,
    LimeExplainer,
    compute_explanation_agreement,
)
from astravisionx.xai.metrics import binarize_heatmap
from astravisionx.utils.image_utils import (
    apply_colormap_on_image,
    apply_diverging_colormap,
    apply_lime_superpixel_overlay,
    draw_detections,
)


def main():
    # --- Load image ---
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        print("Usage: python examples/basic_pipeline.py [path/to/image.jpg]")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    print(f"Loaded image: {image_path} ({image_rgb.shape[1]}x{image_rgb.shape[0]})")

    # --- 1. Object Detection ---
    print("\n[1/5] Running YOLOv8 object detection...")
    yolo = YoloDetector()
    detections = yolo.detect(image_rgb)
    print(f"  Found {len(detections)} objects:")
    for d in detections:
        print(f"    • {d['class_name']}: {d['confidence']:.1%}")

    # Save annotated image
    annotated = draw_detections(image_rgb, detections)
    cv2.imwrite("output_detections.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # --- 2. Scene Classification ---
    print("\n[2/5] Running ViT scene classification...")
    vit = ViTClassifier()
    classifications = vit.classify(image_rgb)
    print("  Top-5 predictions:")
    for i, c in enumerate(classifications[:5], 1):
        print(f"    {i}. Class {c['class_id']} — {c['confidence']:.2%}")

    # --- 3. XAI Explanations ---
    cnn = timm.create_model("resnet18", pretrained=True)
    cnn.eval()
    tensor = vit.transforms(pil_image).unsqueeze(0).to(vit.device)
    target_class = classifications[0]["class_id"]

    # Grad-CAM
    print("\n[3/5] Generating Grad-CAM explanation...")
    gc = GradCamExplainer()
    res_gc = gc.explain(tensor, cnn, target_class=target_class)
    gc_overlay = apply_colormap_on_image(image_rgb, res_gc["heatmap"])
    cv2.imwrite("output_gradcam.jpg", cv2.cvtColor(gc_overlay, cv2.COLOR_RGB2BGR))
    print(f"  Confidence: {res_gc['confidence']:.2%}")

    # Integrated Gradients
    print("\n[4/5] Generating Integrated Gradients explanation...")
    ig = ShapExplainer(num_steps=8, num_baselines=2)
    res_ig = ig.explain(tensor, cnn, target_class=target_class, org_image_rgb=image_rgb)
    ig_overlay = apply_diverging_colormap(image_rgb, res_ig["raw_map"])
    cv2.imwrite("output_ig.jpg", cv2.cvtColor(ig_overlay, cv2.COLOR_RGB2BGR))
    print(f"  Positive attribution: {res_ig['positive_ratio']}%")

    # LIME
    print("\n[5/5] Generating LIME explanation...")
    lime_exp = LimeExplainer(num_samples=25, num_segments=20)
    res_lime = lime_exp.explain(
        tensor, cnn, target_class=target_class,
        org_image_rgb=image_rgb, transform_fn=vit.transforms,
    )
    lime_overlay = apply_lime_superpixel_overlay(
        image_rgb, res_lime["raw_map"], res_lime["top_segments"],
    )
    cv2.imwrite("output_lime.jpg", cv2.cvtColor(lime_overlay, cv2.COLOR_RGB2BGR))
    print(f"  Top segments: {res_lime['top_segments_count']}")

    # --- 4. Multi-Method Agreement ---
    print("\n━" * 50)
    masks = {
        "gradcam": binarize_heatmap(res_gc["heatmap"]),
        "ig": res_ig["salient_mask"],
        "lime": res_lime["salient_mask"],
    }
    agreement = compute_explanation_agreement(masks)
    print(f"Agreement Score: {agreement['agreement_score_pct']}%"
          f" ({agreement['consistency_level']} consistency)")
    print(f"Pairwise IoU: {agreement['pairwise_iou']}")

    print("\n✅ All outputs saved: output_detections.jpg, output_gradcam.jpg, "
          "output_ig.jpg, output_lime.jpg")


if __name__ == "__main__":
    main()

