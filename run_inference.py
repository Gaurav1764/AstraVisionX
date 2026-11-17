import cv2
import urllib.request
import numpy as np
import torch
import timm
from PIL import Image

from astravisionx.models.yolo_detector import YoloDetector
from astravisionx.models.vit_classifier import ViTClassifier
from astravisionx.xai import GradCamExplainer, ShapExplainer, LimeExplainer, compute_explanation_agreement
from astravisionx.utils.image_utils import (
    apply_colormap_on_image,
    apply_diverging_colormap,
    apply_lime_superpixel_overlay,
    draw_detections,
)

def download_image(url, filename="sample.jpg"):
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Could not download image: {e}")

def main():
    filename = "sample.jpg"
    
    # Load image
    image = cv2.imread(filename)
    if image is None:
        url = "https://ultralytics.com/images/bus.jpg"
        download_image(url, filename)
        image = cv2.imread(filename)
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("=" * 60)
    print("AstraVisionX — Multi-Method Pipeline Execution")
    print("=" * 60)
    
    # 1. YOLO Detection
    print("\n1. Running YOLO Detector...")
    yolo = YoloDetector()
    detections = yolo.detect(image_rgb)
    print(f"   Detections found: {len(detections)}")
    for d in detections:
        print(f"   - {d['class_name']}: {d['confidence']:.1%}")
    annotated = draw_detections(image_rgb, detections)
    cv2.imwrite("detections_output.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    # 2. ViT Classification
    print("\n2. Running ViT Scene Classifier...")
    vit = ViTClassifier()
    classifications = vit.classify(image_rgb)
    print("   Top Classifications:")
    for i, c in enumerate(classifications, 1):
        print(f"   {i}. Class ID {c['class_id']} — {c['confidence']:.1%}")
        
    # Prepare backbone model for XAI
    cnn = timm.create_model('resnet18', pretrained=True)
    cnn.eval()
    tensor = vit.transforms(Image.fromarray(image_rgb)).unsqueeze(0).to(vit.device)
    target_class = classifications[0]['class_id']
    
    # 3. Grad-CAM
    print("\n3. Running Grad-CAM Explainer...")
    gc = GradCamExplainer()
    res_gc = gc.explain(tensor, cnn, target_class=target_class)
    gc_overlay = apply_colormap_on_image(image_rgb, res_gc['heatmap'])
    cv2.imwrite("gradcam_output.jpg", cv2.cvtColor(gc_overlay, cv2.COLOR_RGB2BGR))
    print(f"   Grad-CAM saved to gradcam_output.jpg (Target Class: {res_gc['target_class']}, Conf: {res_gc['confidence']:.2%})")
    
    # 4. SHAP
    print("\n4. Running SHAP Integrated Gradients Explainer...")
    shap_exp = ShapExplainer(num_steps=16, num_baselines=4)
    res_shap = shap_exp.explain(tensor, cnn, target_class=target_class, org_image_rgb=image_rgb)
    shap_overlay = apply_diverging_colormap(image_rgb, res_shap['raw_map'])
    cv2.imwrite("shap_output.jpg", cv2.cvtColor(shap_overlay, cv2.COLOR_RGB2BGR))
    print(f"   SHAP saved to shap_output.jpg (Positive: {res_shap['positive_ratio']}%, Negative: {res_shap['negative_ratio']}%)")
    
    # 5. LIME
    print("\n5. Running LIME Superpixel Explainer...")
    lime_exp = LimeExplainer(num_samples=100, num_segments=35)
    res_lime = lime_exp.explain(tensor, cnn, target_class=target_class, org_image_rgb=image_rgb, transform_fn=vit.transforms)
    lime_overlay = apply_lime_superpixel_overlay(image_rgb, res_lime['raw_map'], res_lime['top_segments'])
    cv2.imwrite("lime_output.jpg", cv2.cvtColor(lime_overlay, cv2.COLOR_RGB2BGR))
    print(f"   LIME saved to lime_output.jpg (Top positive segments: {res_lime['top_segments_count']})")
    
    # 6. Explanation Agreement Metric
    print("\n6. Computing Multi-Method Agreement Metric...")
    masks = {
        'gradcam': res_gc['heatmap'] > 0.4,
        'shap': res_shap['salient_mask'],
        'lime': res_lime['salient_mask'],
    }
    agreement = compute_explanation_agreement(masks)
    print(f"   Agreement Score: {agreement['agreement_score_pct']}% ({agreement['consistency_level']} Consistency)")
    print(f"   Pairwise Overlap: {agreement['pairwise_iou']}")
    print(f"   Summary Note: {agreement['summary_note']}")
    print("\n" + "=" * 60)
    print("Pipeline completed successfully! Visualizations saved.")
    print("=" * 60)

if __name__ == "__main__":
    main()
