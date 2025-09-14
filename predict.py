
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    timm = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ConditionNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", clean_levels=3):
        super().__init__()
        if timm is None:
            raise ImportError("Please install timm: pip install timm")
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.damage_head = nn.Linear(feat_dim, 1)
        self.clean_head = nn.Linear(feat_dim, clean_levels if clean_levels>2 else 1)
        self.clean_levels = clean_levels

    def forward(self, x):
        f = self.backbone(x)
        d = self.damage_head(f).squeeze(1)
        c = self.clean_head(f)
        if self.clean_levels == 2:
            c = c.squeeze(1)
        return d, c

def load_images(paths):
    all_paths = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            all_paths.extend([str(x) for x in p.glob("**/*") if x.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
        else:
            all_paths.append(str(p))
    return all_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--backbone", type=str, default="convnext_tiny")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--clean-levels", type=int, default=3, choices=[2,3])
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    outdir = Path(args.out) 
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionNet(backbone=args.backbone, clean_levels=args.clean_levels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(int(args.img_size*1.1)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_paths = load_images(args.images)

    for p in img_paths:
        with Image.open(p).convert("RGB") as im:
            x = tf(im).unsqueeze(0).to(device)
        with torch.no_grad():
            d_log, c_log = model(x)
            d_prob = torch.sigmoid(d_log).item()
            if args.clean_levels == 2:
                c_prob = torch.sigmoid(c_log).item()
                cl_str = f"clean_p={c_prob:.2f}"
            else:
                c_prob = torch.softmax(c_log, dim=1).cpu().numpy()[0]
                cl_str = "clean_probs=" + ",".join(f"{v:.2f}" for v in c_prob.tolist())

        # Save preview
        plt.figure()
        plt.imshow(np.asarray(Image.open(p).convert("RGB")))
        plt.axis("off")
        plt.title(f"damage_p={d_prob:.2f} | {cl_str}")
        out_path = outdir / (Path(p).stem + "_pred.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    print(f"Saved previews to {outdir}")

if __name__ == "__main__":
    main()
