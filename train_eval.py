import argparse
from pathlib import Path
import time
import math
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    timm = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def to_jsonable(o):
    import numpy as np
    from pathlib import Path

    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    # на всякий случай — не возвращаем объект «как есть», чтобы не зациклиться
    raise TypeError(f"Not JSON serializable: {type(o)}")


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        split_dir: Path,
        img_size=384,
        augment=False,
    ):
        self.df = pd.read_csv(csv_path)
        self.split_dir = split_dir
        self.augment = augment

        if augment:
            self.tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)), # Изменение размера и обрезка
                    transforms.RandomHorizontalFlip(), # Случайное горизонтальное отражение
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), # Случайное изменение яркости, контрастности, насыщенности и оттенка
                    transforms.RandomAdjustSharpness(1.2, p=0.3), # Случайное изменение резкости
                    transforms.RandomGrayscale(p=0.1), # Добавляем RandomGrayscale для разнообразия
                    transforms.ToTensor(), # Преобразование изображения в тензор
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), # Нормализация изображений
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(int(img_size * 1.1)),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = row["image_path"]
        img_path = self.split_dir / img_rel
        with Image.open(img_path).convert("RGB") as im:
            x = self.tf(im)

        damage = int(row["damage"])  # 1=damaged, 0=not damaged
        
        # Для чистоты: clean=1 в CSV означает чистый, clean=0 означает грязный
        # Мы хотим dirty=1 как позитивный класс, поэтому инвертируем
        clean_binary = int(row["clean"])  # 1=clean, 0=dirty из CSV
        dirty = 1 - clean_binary  # инвертируем: 1=dirty, 0=clean

        return (
            x,
            torch.tensor([damage], dtype=torch.float32),
            torch.tensor([dirty], dtype=torch.float32),
            str(img_path),
        )


class ConditionNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0"):
        super().__init__()
        if timm is None:
            raise ImportError("Please install timm: pip install timm")

        # Берем предобученную модель без головы классификации
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features

        # Две бинарные головы
        self.damage_head = nn.Linear(feat_dim, 1)  # P(damaged)
        self.clean_head = nn.Linear(feat_dim, 1)   # P(dirty) - позитивный класс для dirty

    def forward(self, x):
        f = self.backbone(x)
        damage_logits = self.damage_head(f).squeeze(1)  # [batch_size]
        clean_logits = self.clean_head(f).squeeze(1)    # [batch_size] 
        return damage_logits, clean_logits


def compute_pos_weight(labels):
    # labels: tensor of shape [N, 1] for binary task (damage)
    y = labels.view(-1).float()
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(1.0, neg / max(1, pos)))


def plot_pr_curve(y_true, y_score, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Damage PR curve (AP={ap:.3f})")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, out_path, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(cm)))
    plt.yticks(range(len(cm)))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_sample_preds(
    images, paths, y_damage, p_damage, y_clean, p_clean, out_dir: Path, max_samples=16
):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(images), max_samples)
    # Create a simple grid 4x4
    cols = 4
    rows = int(math.ceil(n / cols))

    fig_w = 12
    fig_h = 3 * rows
    plt.figure(figsize=(fig_w, fig_h))

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        dmg_t = int(y_damage[i])
        dmg_p = float(p_damage[i])
        if isinstance(y_clean[i], (list, np.ndarray)) or (
            hasattr(y_clean[i], "shape") and len(np.array(y_clean[i]).shape)
        ):
            cl_t = int(y_clean[i])
        else:
            cl_t = int(y_clean[i])
        cl_p = p_clean[i]
        if isinstance(cl_p, (list, np.ndarray)) and len(np.array(cl_p).shape) > 0:
            cl_p_str = ",".join(f"{float(v):.2f}" for v in np.array(cl_p).tolist())
        else:
            cl_p_str = f"{float(cl_p):.2f}"

        ax.set_title(f"Dmg t={dmg_t} p={dmg_p:.2f}\nDirty t={cl_t} p={cl_p_str}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "sample_preds.png", bbox_inches="tight")
    plt.close()


def train_one_epoch(
    model, loader, optimizer, device, bce_loss_damage, bce_loss_clean, alpha=1.0, beta=1.0
):
    model.train()
    total_loss = 0.0
    total_damage_loss = 0.0
    total_clean_loss = 0.0

    for x, y_damage, y_dirty, _ in loader:
        x = x.to(device, non_blocking=True)
        y_damage = y_damage.to(device)
        y_dirty = y_dirty.to(device)

        optimizer.zero_grad()
        d_logits, c_logits = model(x)

        l_damage = bce_loss_damage(d_logits, y_damage.view(-1))
        l_clean = bce_loss_clean(c_logits, y_dirty.view(-1))
        
        # Комбинированный лосс с весами
        loss = alpha * l_damage + beta * l_clean
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_damage_loss += l_damage.item() * x.size(0)
        total_clean_loss += l_clean.item() * x.size(0)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "damage_loss": total_damage_loss / n,
        "clean_loss": total_clean_loss / n,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, damage_threshold=0.5, dirty_threshold=0.5):
    model.eval()
    y_dmg_true = []
    y_dmg_score = []
    y_dmg_pred = []
    
    y_dirty_true = []
    y_dirty_score = []
    y_dirty_pred = []

    for x, y_damage, y_dirty, _ in loader:
        x = x.to(device, non_blocking=True)
        d_logits, c_logits = model(x)

        # Damage predictions
        d_prob = torch.sigmoid(d_logits).cpu().numpy()
        d_pred = (d_prob >= damage_threshold).astype(int)
        y_dmg_score.extend(d_prob.tolist())
        y_dmg_pred.extend(d_pred.tolist())
        y_dmg_true.extend(y_damage.view(-1).cpu().numpy().tolist())

        # Dirty predictions  
        c_prob = torch.sigmoid(c_logits).cpu().numpy()
        c_pred = (c_prob >= dirty_threshold).astype(int)
        y_dirty_score.extend(c_prob.tolist())
        y_dirty_pred.extend(c_pred.tolist())
        y_dirty_true.extend(y_dirty.view(-1).cpu().numpy().tolist())

    # Damage metrics
    y_dmg_true = np.array(y_dmg_true).astype(int)
    y_dmg_score = np.array(y_dmg_score)
    y_dmg_pred = np.array(y_dmg_pred).astype(int)

    # Валидация: убедимся, что у нас есть оба класса
    unique_dmg = np.unique(y_dmg_true)
    if len(unique_dmg) > 1:
        dmg_acc = accuracy_score(y_dmg_true, y_dmg_pred)
        dmg_bacc = balanced_accuracy_score(y_dmg_true, y_dmg_pred)
        dmg_f1 = f1_score(y_dmg_true, y_dmg_pred, average="macro")
        dmg_ap = average_precision_score(y_dmg_true, y_dmg_score)
    else:
        # Если только один класс, используем простые метрики
        dmg_acc = accuracy_score(y_dmg_true, y_dmg_pred)
        dmg_bacc = dmg_acc
        dmg_f1 = 0.0  # F1 не определен для одного класса
        dmg_ap = 0.0

    # Dirty metrics
    y_dirty_true = np.array(y_dirty_true).astype(int)
    y_dirty_score = np.array(y_dirty_score)
    y_dirty_pred = np.array(y_dirty_pred).astype(int)
    
    # Валидация: убедимся, что у нас есть оба класса
    unique_dirty = np.unique(y_dirty_true)
    if len(unique_dirty) > 1:
        dirty_acc = accuracy_score(y_dirty_true, y_dirty_pred)
        dirty_bacc = balanced_accuracy_score(y_dirty_true, y_dirty_pred)
        dirty_f1 = f1_score(y_dirty_true, y_dirty_pred, average="macro")
        dirty_ap = average_precision_score(y_dirty_true, y_dirty_score)
    else:
        # Если только один класс, используем простые метрики
        dirty_acc = accuracy_score(y_dirty_true, y_dirty_pred)
        dirty_bacc = dirty_acc
        dirty_f1 = 0.0  # F1 не определен для одного класса
        dirty_ap = 0.0

    return {
        "dmg": {
            "acc": dmg_acc,
            "bacc": dmg_bacc,
            "f1": dmg_f1,
            "ap": dmg_ap,
            "y_true": y_dmg_true,
            "y_pred": y_dmg_pred,
            "y_score": y_dmg_score,
        },
        "dirty": {
            "acc": dirty_acc,
            "bacc": dirty_bacc,
            "f1": dirty_f1,
            "ap": dirty_ap,
            "y_true": y_dirty_true,
            "y_pred": y_dirty_pred,
            "y_score": y_dirty_score,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--backbone", type=str, default="convnext_tiny")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for damage loss")
    ap.add_argument("--beta", type=float, default=1.0, help="Weight for cleanliness loss")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir) if args.outdir else Path("runs") / timestamp()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_train = data_root / "multitask_labels_train.csv"
    csv_valid = data_root / "multitask_labels_valid.csv"
    csv_test = data_root / "multitask_labels_test.csv"

    train_ds = MultiTaskDataset(
        csv_train,
        data_root / "train",
        img_size=args.img_size,
        augment=True,
    )
    valid_ds = MultiTaskDataset(
        csv_valid,
        data_root / "valid",
        img_size=args.img_size,
        augment=False,
    )
    test_ds = MultiTaskDataset(
        csv_test,
        data_root / "test",
        img_size=args.img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionNet(backbone=args.backbone).to(device)

    # Compute pos_weights for both tasks
    all_damage = torch.tensor(train_ds.df["damage"].values).float().view(-1, 1)
    pos_weight_damage = compute_pos_weight(all_damage).to(device)
    
    # Для dirty: инвертируем clean для подсчета pos_weight  
    all_clean = torch.tensor(train_ds.df["clean"].values).float().view(-1, 1)
    all_dirty = 1 - all_clean  # инвертируем: 1=dirty, 0=clean
    pos_weight_dirty = compute_pos_weight(all_dirty).to(device)

    # Отладочная информация о распределении классов
    damage_counts = train_ds.df["damage"].value_counts().sort_index()
    clean_counts = train_ds.df["clean"].value_counts().sort_index()
    dirty_counts = (1 - train_ds.df["clean"]).value_counts().sort_index()
    
    print(f"[INFO] Damage distribution: {dict(damage_counts)}")
    print(f"[INFO] Clean distribution: {dict(clean_counts)} (1=clean, 0=dirty)")
    print(f"[INFO] Dirty distribution: {dict(dirty_counts)} (1=dirty, 0=clean)")
    print(f"[INFO] Pos weight damage: {pos_weight_damage.item():.3f}")
    print(f"[INFO] Pos weight dirty: {pos_weight_dirty.item():.3f}")

    bce_loss_damage = nn.BCEWithLogitsLoss(pos_weight=pos_weight_damage)
    bce_loss_clean = nn.BCEWithLogitsLoss(pos_weight=pos_weight_dirty)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = {"train": [], "valid": []}
    best_score = -1.0
    best_path = outdir / "best.pt"
    start_epoch = 1

    # Load existing checkpoint if available
    if best_path.exists():
        print(f"[INFO] Loading existing checkpoint from {best_path}")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print("[INFO] Loaded model weights")
        else:
            model.load_state_dict(checkpoint)
            print("[INFO] Loaded model weights (legacy format)")
        
        # Try to load optimizer state if available
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("[INFO] Loaded optimizer state")
        
        # Load training history if available
        if "history" in checkpoint:
            history = checkpoint["history"]
            start_epoch = len(history["train"]) + 1
            print(f"[INFO] Loaded training history, resuming from epoch {start_epoch}")
        
        # Load best score if available
        if "best_score" in checkpoint:
            best_score = checkpoint["best_score"]
            print(f"[INFO] Previous best score: {best_score:.4f}")
            
        # Load epoch info if available
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"[INFO] Resuming from epoch {start_epoch}")
    else:
        print("[INFO] No existing checkpoint found, starting training from scratch")

    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            bce_loss_damage,
            bce_loss_clean,
            args.alpha,
            args.beta,
        )
        ev = eval_epoch(model, valid_loader, device)

        history["train"].append(tr)
        history["valid"].append(ev)

        # composite score: macro F1 damage + macro F1 dirty
        score = ev["dmg"]["f1"] + ev["dirty"]["f1"]
        if score > best_score:
            best_score = score
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "history": history,
                "args": vars(args)
            }
            torch.save(checkpoint, best_path)

        # logging
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr['loss']:.4f} "
            f"| val_dmg_f1={ev['dmg']['f1']:.3f} ap={ev['dmg']['ap']:.3f} "
            f"| val_dirty_f1={ev['dirty']['f1']:.3f} ap={ev['dirty']['ap']:.3f}"
        )

    # Save history
    with open(outdir/"history.json", "w") as f:
        json.dump(history, f, indent=2, default=to_jsonable)

    # Plots
    # 1) PR curve for damage on validation
    plot_pr_curve(
        history["valid"][-1]["dmg"]["y_true"],
        history["valid"][-1]["dmg"]["y_score"],
        outdir / "val_damage_pr.png",
    )
    
    # 2) PR curve for dirty on validation
    def plot_pr_curve_dirty(y_true, y_score, out_path):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Dirty PR curve (AP={ap:.3f})")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        
    plot_pr_curve_dirty(
        history["valid"][-1]["dirty"]["y_true"],
        history["valid"][-1]["dirty"]["y_score"],
        outdir / "val_dirty_pr.png",
    )

    # 3) Confusion matrices
    plot_confusion(
        history["valid"][-1]["dmg"]["y_true"],
        history["valid"][-1]["dmg"]["y_pred"],
        outdir / "val_damage_confusion.png",
        labels=[0, 1],
        title="Damage Confusion (val)",
    )

    plot_confusion(
        history["valid"][-1]["dirty"]["y_true"],
        history["valid"][-1]["dirty"]["y_pred"],
        outdir / "val_dirty_confusion.png",
        labels=[0, 1],
        title="Dirty Confusion (val)",
    )

    # 4) Sample predictions from validation
    # Take first batch
    with torch.no_grad():
        for x, y_d, y_dirty, pth in valid_loader:
            x = x.to(device)
            d_logits, c_logits = model(x)
            d_prob = torch.sigmoid(d_logits).cpu().numpy()
            c_prob = torch.sigmoid(c_logits).cpu().numpy()

            save_sample_preds(
                x.cpu(),
                pth,
                y_d.view(-1).numpy(),
                d_prob,
                y_dirty.numpy(),
                c_prob,
                outdir / "samples",
                max_samples=16,
            )
            break

    # Final test evaluation on best model
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    test_ev = eval_epoch(model, test_loader, device)
    with open(outdir/"test_report.json", "w") as f:
        json.dump(test_ev, f, indent=2, default=to_jsonable)

    print(
        "Test metrics:",
        f"damage F1={test_ev['dmg']['f1']:.3f} AP={test_ev['dmg']['ap']:.3f} | ",
        f"dirty F1={test_ev['dirty']['f1']:.3f} AP={test_ev['dirty']['ap']:.3f}",
    )


if __name__ == "__main__":
    main()
