#!/usr/bin/env python3
"""
gradio_app.py

Unified demo: классификация (битый/грязный) + детекция/сегментация (Mask R-CNN).
- Загружает два чекпойнта: classification (best.pt из train_eval.py) и detection (best.pth из maskrcnn).
- Пользователь загружает изображение, выбирает порог score, получает:
  * наложение боксов/масок,
  * вероятности классификации (damage prob, cleanliness).
"""
import argparse
import json
from pathlib import Path

import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import gradio as gr

try:
    import timm
except ImportError:
    timm = None
import torchvision

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
CLASSES = ["__background__", "car", "dent", "scratch", "rust", "dirt"]
PALETTE = {
    "car":     (46, 204, 113),
    "dent":    (231, 76, 60),
    "scratch": (155, 89, 182),
    "rust":    (243, 156, 18),
    "dirt":    (52, 152, 219),
}

# ---- Classification model (must match train_eval.py) ----
class ConditionNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", clean_levels=3):
        super().__init__()
        if timm is None:
            raise ImportError("Install timm")
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.damage_head = nn.Linear(feat_dim, 1)
        self.clean_head  = nn.Linear(feat_dim, clean_levels if clean_levels>2 else 1)
        self.clean_levels = clean_levels
    def forward(self, x):
        f = self.backbone(x)
        d = self.damage_head(f).squeeze(1)
        c = self.clean_head(f)
        if self.clean_levels == 2: c = c.squeeze(1)
        return d, c

def load_classification(ckpt_path: str, backbone="convnext_tiny", clean_levels=3, device="cpu"):
    model = ConditionNet(backbone=backbone, clean_levels=clean_levels).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model

def classify_image(model, img: Image.Image, clean_levels=3, img_size=384, device="cpu"):
    tf = T.Compose([T.Resize(int(img_size*1.1)), T.CenterCrop(img_size),
                    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = tf(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        d_log, c_log = model(x)
        d_prob = torch.sigmoid(d_log).item()
        if clean_levels==2:
            c_prob = torch.sigmoid(c_log).item()  # this is prob(positive class); if you trained positive=dirty, treat accordingly
            clean_text = f"clean(binary prob)={c_prob:.2f}"
        else:
            c_prob = torch.softmax(c_log, dim=1).cpu().numpy()[0]
            clean_text = "cleanliness_probs=" + ",".join(f"{p:.2f}" for p in c_prob.tolist())
    return d_prob, clean_text

# ---- Detection model ----
def get_maskrcnn(num_classes=6):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT", box_detections_per_img=200)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model

def load_detection(ckpt_path: str, device="cpu"):
    model = get_maskrcnn(num_classes=len(CLASSES)).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model

def detect_image(model, img: Image.Image, score_th=0.5, device="cpu"):
    tf = T.Compose([T.ToTensor()])
    x = tf(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = model(x)[0]
    boxes  = pred.get("boxes", torch.empty(0,4)).cpu().numpy()
    labels = pred.get("labels", torch.empty(0)).cpu().numpy().astype(int)
    scores = pred.get("scores", torch.empty(0)).cpu().numpy()
    masks  = pred.get("masks", torch.empty(0)).cpu().numpy()

    keep = [i for i,s in enumerate(scores) if s>=score_th]
    order = sorted(keep, key=lambda i: float(scores[i]), reverse=True)

    canvas = img.copy()
    dr = ImageDraw.Draw(canvas, 'RGBA')
    rows = []
    for i in order:
        b, l, s = boxes[i], labels[i], scores[i]
        if l<=0 or l>=len(CLASSES): continue
        cname = CLASSES[l]
        color = PALETTE.get(cname, (255,255,255))
        x1,y1,x2,y2 = b.astype(int)
        dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
        dr.text((x1, max(0,y1-14)), f"{cname}:{s:.2f}", fill=color)
        if masks is not None and len(masks)>i:
            m = (masks[i,0] > 0.5).astype(np.uint8)*120
            overlay = Image.fromarray(np.stack([m*0, m, m*0], axis=-1).astype(np.uint8), 'RGB').resize(img.size)
            canvas = Image.blend(canvas, overlay, alpha=0.30)
        rows.append([cname, float(s), int(x1),int(y1),int(x2),int(y2)])
    return canvas, rows

def build_demo(class_ckpt, det_ckpt, backbone="convnext_tiny", clean_levels=3, device="cpu"):
    cls_model = load_classification(class_ckpt, backbone=backbone, clean_levels=clean_levels, device=device) if class_ckpt else None
    det_model = load_detection(det_ckpt, device=device) if det_ckpt else None

    def infer(img, score_th=0.5):
        img = Image.fromarray(img)
        dmg_prob, clean_text = (None, "—")
        if cls_model is not None:
            dmg_prob, clean_text = classify_image(cls_model, img, clean_levels=clean_levels, device=device)
        canvas, rows = (img, [])
        if det_model is not None:
            canvas, rows = detect_image(det_model, img, score_th=score_th, device=device)
        # text panel
        text = f"damage_p={dmg_prob:.2f} | {clean_text}" if dmg_prob is not None else clean_text
        return canvas, rows, text

    return infer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-ckpt", type=str, required=False, help="Path to classification checkpoint (best.pt)")
    ap.add_argument("--det-ckpt", type=str, required=True, help="Path to detection/segmentation checkpoint (best.pth)")
    ap.add_argument("--backbone", type=str, default="convnext_tiny")
    ap.add_argument("--clean-levels", type=int, default=3, choices=[2,3])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    infer = build_demo(args.class_ckpt, args.det_ckpt, backbone=args.backbone, clean_levels=args.clean_levels, device=args.device)

    with gr.Blocks() as demo:
        gr.Markdown("# Car Condition Demo — Classification + Detection/Segmentation")
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload image", height=480)
            with gr.Column():
                th = gr.Slider(0.0, 0.95, value=0.5, step=0.05, label="Score threshold (detection)")
                btn = gr.Button("Run")
                text_out = gr.Textbox(label="Classification output")
        img_out = gr.Image(type="pil", label="Overlay")
        table = gr.Dataframe(headers=["class","score","x1","y1","x2","y2"], datatype=["str","number","number","number","number","number"])

        def _run(img, score):
            canvas, rows, text = infer(img, score)
            return canvas, rows, text

        btn.click(_run, inputs=[img_in, th], outputs=[img_out, table, text_out])

    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
