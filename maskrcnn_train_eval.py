#!/usr/bin/env python3
"""
maskrcnn_train_eval.py

- Обучение Mask R-CNN на COCO-подобном датасете (bbox + опционально маски).
- Качественная валидация (визуализации) + количественная оценка **mAP** через COCOEval (bbox/segm).
- Рисуем боксы/маски с легендой и фильтром по score, сортируем по убыванию score.

Пример:
  pip install torch torchvision pycocotools matplotlib
  python maskrcnn_train_eval.py --coco-root /path/to/_merged_coco --epochs 15 --batch-size 4 --lr 5e-4 --out runs_maskrcnn

Оценка mAP (валид):
  python maskrcnn_train_eval.py --coco-root /path/to/_merged_coco --eval-split valid --checkpoint runs_maskrcnn/best.pth --out runs_maskrcnn_eval

Инференс на папке:
  python maskrcnn_train_eval.py --predict /path/to/dir --checkpoint runs_maskrcnn/best.pth --out runs_maskrcnn_pred --score-th 0.5
"""
import argparse, random, time, json, math
from pathlib import Path
from typing import Tuple, Dict, Any, List

from tqdm import tqdm
import torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.ops import box_convert
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt

CLASSES = ["__background__", "car", "dent", "scratch", "rust", "dirt"]
NAME2ID = {n:i for i,n in enumerate(CLASSES)}  # model labels 0..5
PALETTE = {
    "car":     (46, 204, 113),  # green
    "dent":    (231, 76, 60),   # red
    "scratch": (155, 89, 182),  # purple
    "rust":    (243, 156, 18),  # orange
    "dirt":    (52, 152, 219),  # blue
}

def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class CocoDetDataset(Dataset):
    def __init__(self, root: Path, split: str, augment=False):
        ann_path = root / split / "_annotations.coco.json"
        self.coco = COCO(str(ann_path))
        self.root = root / split
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.augment = augment
        self.tf = T.Compose([T.ToTensor()])

    def __len__(self): return len(self.ids)

    def _load_img(self, img_info):
        p = self.root / img_info["file_name"]
        if not p.exists():
            alt = self.root / "images" / Path(img_info["file_name"]).name
            p = alt
        img = Image.open(p).convert("RGB")
        return img

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        img = self._load_img(img_info)
        W, H = img.size

        boxes, labels, masks = [], [], []
        for a in anns:
            name = self.coco.cats[a["category_id"]]["name"].lower()
            # normalize typos
            if "scracth" in name: 
                name = "scratch"
            if "dunt" in name: 
                name = "dent"
            if name.startswith("car"): 
                name = "car"
            if name not in NAME2ID: 
                continue
            cid = NAME2ID[name]

            x,y,w,h = a.get("bbox", [0,0,0,0])
            if w<=0 or h<=0: continue
            boxes.append([x,y,x+w,y+h])
            labels.append(cid)
            m = Image.new("L", (W, H), 0)
            ImageDraw.Draw(m).rectangle([x, y, x+w, y+h], outline=1, fill=1)
            masks.append(np.array(m, dtype=np.uint8))

            seg = a.get("segmentation", None)
            if isinstance(seg, list) and len(seg)>0 and isinstance(seg[0], list) and len(seg[0])>=6:
                try:
                    mask = np.zeros((H,W), dtype=np.uint8)
                    for poly in seg:
                        xs = poly[0::2]; ys = poly[1::2]
                        pil_mask = Image.new("L", (W,H), 0)
                        ImageDraw.Draw(pil_mask).polygon(list(zip(xs, ys)), outline=1, fill=1)
                        mask = np.maximum(mask, np.array(pil_mask))
                    masks.append(mask)
                except Exception:
                    pass

        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target: Dict[str, Any] = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        if len(masks) > 0:
            target["masks"] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            target["masks"] = torch.zeros((0, H, W), dtype=torch.uint8)

        img = self.tf(img)
        return img, target

def collate(batch): return tuple(zip(*batch))

def get_model(num_classes=6):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT", box_detections_per_img=200)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

def draw_legend(canvas: Image.Image, classes: List[str]):
    # small legend box in top-left
    dr = ImageDraw.Draw(canvas)
    x0, y0 = 10, 10
    y = y0
    for cname in classes:
        if cname not in PALETTE: continue
        color = PALETTE[cname]
        dr.rectangle([x0, y, x0+16, y+16], fill=color, outline=None)
        dr.text((x0+20, y), cname, fill=(255,255,255))
        y += 18
    return canvas

@torch.inference_mode()
def visualize(model, loader, device, out_dir: Path, score_th=0.5, max_images=24):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        preds = model(imgs)
        for im, pred in zip(imgs, preds):
            im_np = (im.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            h,w,_ = im_np.shape
            canvas = Image.fromarray(im_np)
            dr = ImageDraw.Draw(canvas, 'RGBA')

            boxes  = pred.get("boxes", torch.empty(0,4)).cpu().numpy()
            labels = pred.get("labels", torch.empty(0)).cpu().numpy().astype(int)
            scores = pred.get("scores", torch.empty(0)).cpu().numpy()
            masks  = pred.get("masks", torch.empty(0)).cpu().numpy()  # [N,1,H,W]

            # filter & sort
            keep = [i for i,s in enumerate(scores) if s>=score_th]
            order = sorted(keep, key=lambda i: float(scores[i]), reverse=True)
            for i in order:
                b = boxes[i]; l = labels[i]; s = scores[i]
                if l<=0 or l>=len(CLASSES): continue
                cname = CLASSES[l]
                color = PALETTE.get(cname, (255,255,255))
                x1,y1,x2,y2 = b.astype(int)
                dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
                dr.text((x1, max(0,y1-14)), f"{cname}:{s:.2f}", fill=color)

                if masks is not None and len(masks)>i:
                    m = (masks[i,0] > 0.5).astype(np.uint8)*120
                    overlay = Image.fromarray(np.stack([m*0, m, m*0], axis=-1).astype(np.uint8), 'RGB')
                    overlay = overlay.resize((w,h))
                    canvas = Image.blend(canvas, overlay, alpha=0.30)

            canvas = draw_legend(canvas, ["car","dent","scratch","rust","dirt"])
            canvas.save(out_dir/f"pred_{saved:05d}.png")
            saved += 1
            if saved >= max_images:
                return

def coco_eval(model, dataset: CocoDetDataset, device, split: str, out_dir: Path, score_th=0.001):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)

    cocoGt = dataset.coco

    # ✅ Робастность к неполному COCO (Roboflow и пр.)
    cocoGt.dataset.setdefault("info", {"description": "auto-added"})
    cocoGt.dataset.setdefault("licenses", [])
    # Маппинг имя -> реальный catId из GT
    name_to_catId = {c["name"].lower(): c["id"] for c in cocoGt.dataset.get("categories", [])}

    dets_bbox, dets_segm = [], []
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        img_ids = [int(t["image_id"][0]) for t in targets]
        preds = model(imgs)
        for pid, pred in zip(img_ids, preds):
            boxes  = pred.get("boxes", torch.empty(0,4)).detach().cpu().numpy()
            labels = pred.get("labels", torch.empty(0)).detach().cpu().numpy().astype(int)
            scores = pred.get("scores", torch.empty(0)).detach().cpu().numpy()
            masks  = pred.get("masks", torch.empty(0)).detach().cpu().numpy()  # [N,1,H,W]

            keep = scores >= score_th
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            masks = masks[keep] if masks is not None and len(masks)>0 else masks

            # 📌 Преобразуем наш индекс класса -> имя -> catId из GT
            for b, l, s in zip(boxes, labels, scores):
                if l == 0:  # background
                    continue
                cname = CLASSES[l].lower()
                cat_id = name_to_catId.get(cname)
                if cat_id is None:
                    continue
                x1, y1, x2, y2 = b.tolist()
                dets_bbox.append({
                    "image_id": pid,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })

            if masks is not None and len(masks) > 0:
                from pycocotools import mask as maskUtils
                H, W = masks.shape[-2:]
                for m, l, s in zip(masks[:,0], labels, scores):
                    if l == 0:
                        continue
                    cname = CLASSES[l].lower()
                    cat_id = name_to_catId.get(cname)
                    if cat_id is None:
                        continue
                    m = (m > 0.5).astype(np.uint8)
                    rle = maskUtils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("ascii")
                    dets_segm.append({
                        "image_id": pid,
                        "category_id": int(cat_id),
                        "segmentation": rle,
                        "score": float(s),
                    })

    stats = {}
    def _eval(results, iouType):
        if len(results) == 0:
            return None
        cocoDt = cocoGt.loadRes(results)  # больше не упадёт из-за 'info'
        E = COCOeval(cocoGt, cocoDt, iouType)
        E.evaluate(); E.accumulate(); E.summarize()
        keys = ["AP","AP50","AP75","APs","APm","APl"]
        vals = [float(v) for v in E.stats[:6]]
        return dict(zip(keys, vals))

    bbox_stats = _eval(dets_bbox, "bbox")
    if bbox_stats is not None:
        stats["bbox"] = bbox_stats
    segm_stats = _eval(dets_segm, "segm") if len(dets_segm) > 0 else None
    if segm_stats is not None:
        stats["segm"] = segm_stats

    (out_dir / f"coco_metrics_{split}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[COCO EVAL]", split, stats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", type=str, required=False, help="Path to merged COCO root with train/valid/test")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", type=str, default="runs_maskrcnn")
    ap.add_argument("--score-th", type=float, default=0.5, help="score threshold for visualization and predict")
    ap.add_argument("--eval-split", type=str, default=None, choices=["valid","test"], help="Run COCO mAP eval on split")
    ap.add_argument("--predict", type=str, default=None, help="Path to image or folder (inference only mode)")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.predict or args.eval_split:
        # eval/predict mode
        assert args.checkpoint is not None, "--checkpoint required for predict/eval"
        model = get_model(num_classes=len(CLASSES))
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.to(device).eval()

        if args.predict:
            p = Path(args.predict)
            paths = []
            if p.is_dir():
                for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                    paths.extend(list(p.rglob(ext)))
            else:
                paths = [p]
            vis_dir = out_dir / "predict"
            vis_dir.mkdir(parents=True, exist_ok=True)
            tf = T.Compose([T.ToTensor()])
            for i, ip in enumerate(paths):
                im = Image.open(ip).convert("RGB")
                x = tf(im).to(device).unsqueeze(0)
                with torch.inference_mode():
                    pred = model(x)[0]
                canvas = im.copy()
                dr = ImageDraw.Draw(canvas, 'RGBA')
                boxes  = pred.get("boxes", torch.empty(0,4)).cpu().numpy()
                labels = pred.get("labels", torch.empty(0)).cpu().numpy().astype(int)
                scores = pred.get("scores", torch.empty(0)).cpu().numpy()
                masks  = pred.get("masks", torch.empty(0)).cpu().numpy()
                keep = [i for i,s in enumerate(scores) if s>=args.score_th]
                order = sorted(keep, key=lambda i: float(scores[i]), reverse=True)
                for j in order:
                    b, l, s = boxes[j], labels[j], scores[j]
                    if l<=0 or l>=len(CLASSES): continue
                    cname = CLASSES[l]
                    color = PALETTE.get(cname, (255,255,255))
                    x1,y1,x2,y2 = b.astype(int)
                    dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
                    dr.text((x1, max(0,y1-14)), f"{cname}:{s:.2f}", fill=color)
                    if masks is not None and len(masks)>j:
                        m = (masks[j,0] > 0.5).astype(np.uint8)*120
                        overlay = Image.fromarray(np.stack([m*0, m, m*0], axis=-1).astype(np.uint8), 'RGB').resize(im.size)
                        canvas = Image.blend(canvas, overlay, alpha=0.30)
                canvas = draw_legend(canvas, ["car","dent","scratch","rust","dirt"])
                canvas.save(vis_dir / f"pred_{i:05d}.png")
            print(f"[DONE] Saved predictions to {vis_dir}")

        if args.eval_split:
            ds = CocoDetDataset(Path(args.coco_root), args.eval_split, augment=False)
            coco_eval(model, ds, device, split=args.eval_split, out_dir=out_dir, score_th=0.0001)
        return

    # Training mode
    assert args.coco_root is not None, "--coco-root required for training"

    train_ds = CocoDetDataset(Path(args.coco_root), "train", augment=True)
    val_ds   = CocoDetDataset(Path(args.coco_root), "valid", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=4, collate_fn=collate)

    model = get_model(num_classes=len(CLASSES)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    best_loss = float("inf")
    start_epoch = 1
    
    # Load existing checkpoint if available
    best_checkpoint_path = Path(args.out) / "best.pth"
    if best_checkpoint_path.exists():
        print(f"[INFO] Loading existing checkpoint from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            print(f"[INFO] Resuming from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("[INFO] Loaded model weights, starting from epoch 1")
        
        # Try to load optimizer state if available
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("[INFO] Loaded optimizer state")
        
        # Load best loss if available
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
            print(f"[INFO] Previous best loss: {best_loss:.4f}")
    else:
        print("[INFO] No existing checkpoint found, starting training from scratch")
    
    for epoch in tqdm(range(start_epoch, args.epochs+1)):
        model.train()
        total = 0.0
        for imgs, targets in train_loader:
            imgs = [im.to(device) for im in imgs]
            tgts = [{k: v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
            losses = model(imgs, tgts)  # dict of losses
            loss = sum(v for v in losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())

        avg = total / max(1, len(train_loader))
        print(f"Epoch {epoch:03d} | train_loss={avg:.4f}")

        # qualitative val
        visualize(model, val_loader, device, Path(args.out)/"samples_val", score_th=args.score_th, max_images=16)

        # save best by train loss (простая эвристика)
        if avg < best_loss:
            best_loss = avg
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss
            }
            torch.save(checkpoint, Path(args.out)/"best.pth")

    print(f"[DONE] Best model saved to {Path(args.out)/'best.pth'}")
    # mAP on valid split
    coco_eval(model, val_ds, device, split="valid", out_dir=Path(args.out))
if __name__ == "__main__":
    main()
