#!/usr/bin/env python3
"""
Train a binary classification model to distinguish between damaged and undamaged cars.
This script focuses solely on damage detection (dents, scratches, rust), removing the 
cleanliness classification complexity from the original training script.

⚠️ Key Features
- Binary classification: damaged (1) vs undamaged (0)
- Detects various damage types: dents, scratches, rust
- Uses timm backbone models (ConvNeXt, EfficientNet, etc.)
- Supports data augmentation for training set
- Comprehensive evaluation with metrics and visualizations
- Checkpoint saving/loading with training resume capability
- Configurable thresholds and hyperparameters

Dataset Structure Expected:
    data_root/
      multitask_labels_train.csv
      multitask_labels_valid.csv  
      multitask_labels_test.csv
      train/
        images/...
      valid/
        images/...
      test/
        images/...

CSV Format:
    image_path,damage,dirt_present,dirt_ratio,clean,clean_level,damage_kinds
    images/damaged_car1.jpg,1,0,0.0,1,0,dent;scratch
    images/clean_car2.jpg,0,0,0.0,1,0,

Usage:
    python train_damage_detection.py --data-root ./data/processed --backbone convnext_tiny \
        --img-size 384 --batch-size 16 --epochs 20 --lr 3e-4

Advanced Usage:
    python train_damage_detection.py --data-root ./data/processed --backbone efficientnet_b2 \
        --img-size 512 --batch-size 8 --epochs 50 --lr 1e-4 --outdir ./runs/damage_exp1 \
        --num-workers 8 --augment-prob 0.8

Requirements:
    pip install torch torchvision timm pillow pandas scikit-learn matplotlib tqdm

Output Structure:
    runs/TIMESTAMP/
      best.pt              # Best model checkpoint
      history.json         # Training history
      test_report.json     # Final test metrics
      val_pr_curve.png     # Precision-Recall curve
      val_confusion.png    # Confusion matrix
      samples/
        sample_preds.png   # Sample predictions visualization
"""

import argparse
import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    print("⚠️  Warning: timm not installed. Please run: pip install timm")

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def timestamp():
    """Generate timestamp string for experiment naming"""
    return time.strftime("%Y%m%d_%H%M%S")


def to_jsonable(obj):
    """Convert numpy/torch objects to JSON-serializable format"""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DamageDataset(Dataset):
    """
    Dataset for binary damage classification.
    
    Args:
        csv_path: Path to CSV file with labels
        split_dir: Directory containing images for this split
        img_size: Target image size for training
        augment: Whether to apply data augmentation
        augment_prob: Probability of applying each augmentation
    """
    
    def __init__(
        self,
        csv_path,
        split_dir,
        img_size=384,
        augment=False,
        augment_prob=0.5,
    ):
        self.df = pd.read_csv(csv_path)
        self.split_dir = split_dir
        self.augment = augment
        self.img_size = img_size
        
        print(f"📊 Loaded {len(self.df)} samples from {csv_path}")
        
        # Check class distribution
        damage_dist = self.df["damage"].value_counts().sort_index()
        print(f"📈 Damage distribution: {dict(damage_dist)} (1=damaged, 0=undamaged)")
        
        # Check damage types distribution
        if "damage_kinds" in self.df.columns:
            damage_kinds = self.df[self.df["damage"] == 1]["damage_kinds"].dropna()
            if len(damage_kinds) > 0:
                kinds_expanded = damage_kinds.apply(lambda x: x.split(';') if x else []).explode()
                kinds_dist = kinds_expanded.value_counts()
                print(f"🔧 Damage types: {dict(kinds_dist)}")
        
        # Data augmentation pipeline - more aggressive for damage detection
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=augment_prob),
                transforms.RandomVerticalFlip(p=0.1),  # Sometimes damage is visible from different angles
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                transforms.RandomAdjustSharpness(1.5, p=0.4),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomRotation(degrees=15),  # Small rotations to see damage from different angles
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            print(f"🔄 Using enhanced data augmentation for damage detection (prob={augment_prob})")
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.1)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            print("📸 Using standard transforms (no augmentation)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        img_path = self.split_dir / img_rel_path
        
        # Load and transform image
        try:
            with Image.open(img_path).convert("RGB") as image:
                x = self.transform(image)
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return a black image as fallback
            x = torch.zeros(3, self.img_size, self.img_size)
        
        # Extract damage label (1=damaged, 0=undamaged)
        damage_label = int(row["damage"])
        
        # Get damage types for debugging/analysis
        damage_kinds = row.get("damage_kinds", "")
        
        return (
            x,
            torch.tensor(damage_label, dtype=torch.float32),
        )


class DamageDetectionNet(nn.Module):
    """
    Binary classification network for car damage detection.
    
    Args:
        backbone: timm model name (e.g., 'convnext_tiny', 'efficientnet_b0')
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate for the classifier head
    """
    
    def __init__(self, backbone="convnext_tiny", pretrained=True, dropout=0.2):
        super().__init__()
        
        if timm is None:
            raise ImportError("timm is required. Install with: pip install timm")
        
        # Load backbone without classification head
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            global_pool="avg"
        )
        
        # Get feature dimension
        feat_dim = self.backbone.num_features
        
        # Binary classification head with more capacity for damage detection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feat_dim // 2, 1)  # Binary classification
        )
        
        print(f"🏗️  Created {backbone} with {feat_dim} features")
        print(f"📦 Using {'pretrained' if pretrained else 'random'} weights")
        print(f"🧠 Classifier: {feat_dim} -> {feat_dim//2} -> 1 (dropout={dropout})")
    
    def forward(self, x):
        """Forward pass returning logits for binary classification"""
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(1)  # [batch_size]
        return logits


def compute_pos_weight(labels):
    """
    Compute positive class weight for BCEWithLogitsLoss to handle class imbalance.
    
    Args:
        labels: Binary labels tensor
        
    Returns:
        Positive weight tensor
    """
    y = labels.view(-1).float()
    pos_count = (y == 1).sum().item()
    neg_count = (y == 0).sum().item()
    
    if pos_count == 0:
        return torch.tensor(1.0)
    
    pos_weight = max(1.0, neg_count / pos_count)
    print(f"⚖️  Class balance - Damaged: {pos_count}, Undamaged: {neg_count}")
    print(f"📊 Positive weight for damaged class: {pos_weight:.3f}")
    
    return torch.tensor(pos_weight)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        num_samples += x.size(0)
        
        # Log progress occasionally
        if batch_idx % 50 == 0:
            current_loss = total_loss / num_samples
            print(f"    Batch {batch_idx:4d}/{len(loader):4d} | Loss: {current_loss:.4f}")
    
    return {"loss": total_loss / num_samples}


@torch.no_grad()
def evaluate_model(model, loader, device, threshold=0.5):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    
    all_labels = []
    all_scores = []
    all_predictions = []
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        
        # Convert to probabilities and predictions
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        labels = y.cpu().numpy().astype(int)
        
        all_scores.extend(probs.tolist())
        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    y_pred = np.array(all_predictions)
    
    # Compute metrics
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="binary")
    
    # Only compute AP if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics["average_precision"] = average_precision_score(y_true, y_score)
    else:
        metrics["average_precision"] = 0.0
    
    # Store raw predictions for plotting
    metrics["y_true"] = y_true
    metrics["y_score"] = y_score  
    metrics["y_pred"] = y_pred
    
    return metrics


def plot_precision_recall_curve(y_true, y_score, save_path):
    """Plot and save precision-recall curve"""
    if len(np.unique(y_true)) <= 1:
        print("⚠️  Skipping PR curve - only one class present")
        return
        
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Damaged vs Undamaged)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📈 Saved PR curve to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Reds")
    plt.title("Confusion Matrix - Damage Detection")
    plt.colorbar()
    
    # Add labels
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Undamaged", "Damaged"])
    plt.yticks(tick_marks, ["Undamaged", "Damaged"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), 
                    ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Saved confusion matrix to {save_path}")


def save_sample_predictions(model, loader, device, save_dir, max_samples=16):
    """Save visualization of sample predictions"""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get one batch
    for images, labels in loader:
        break
    
    with torch.no_grad():
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        predictions = (probs >= 0.5).astype(int)
    
    # Denormalize images for display
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    n_samples = min(len(images), max_samples)
    cols = 4
    rows = math.ceil(n_samples / cols)
    
    plt.figure(figsize=(16, 4 * rows))
    
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        
        # Denormalize image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        # Create title with prediction info
        true_label = int(labels[i])
        pred_label = int(predictions[i])
        confidence = float(probs[i])
        
        true_class = "Damaged" if true_label == 1 else "Undamaged"
        pred_class = "Damaged" if pred_label == 1 else "Undamaged"
        
        color = "green" if true_label == pred_label else "red"
        title = f"True: {true_class}\nPred: {pred_class} ({confidence:.2f})"
        
        plt.title(title, color=color, fontsize=10)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"🖼️  Saved sample predictions to {save_dir}")


def save_checkpoint(model, optimizer, epoch, best_score, history, save_path):
    """Save training checkpoint"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "history": history,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load training checkpoint"""
    print(f"📁 Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -1.0)
    history = checkpoint.get("history", {"train": [], "valid": []})
    
    print(f"✅ Loaded checkpoint from epoch {epoch}, best score: {best_score:.4f}")
    
    return epoch, best_score, history


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train damage detection classification model")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, required=True,
                       help="Root directory containing processed dataset")
    
    # Model arguments  
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                       help="Backbone model from timm (default: convnext_tiny)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained weights")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate for classifier head")
    
    # Training arguments
    parser.add_argument("--img-size", type=int, default=384,
                       help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=25,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay for optimizer")
    
    # Data loading arguments
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--augment-prob", type=float, default=0.7,
                       help="Probability for data augmentation transforms")
    
    # Output arguments
    parser.add_argument("--outdir", type=str, default=None,
                       help="Output directory (default: runs/damage_TIMESTAMP)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Evaluation arguments
    parser.add_argument("--eval-threshold", type=float, default=0.5,
                       help="Threshold for binary classification")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path("runs") / f"damage_{timestamp()}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Starting damage detection training")
    print(f"📂 Output directory: {outdir}")
    print(f"🏗️  Backbone: {args.backbone}")
    print(f"📏 Image size: {args.img_size}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.lr}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    # Load datasets
    data_root = Path(args.data_root)
    
    csv_train = data_root / "multitask_labels_train.csv"
    csv_valid = data_root / "multitask_labels_valid.csv"
    csv_test = data_root / "multitask_labels_test.csv"
    
    train_dataset = DamageDataset(
        csv_train, data_root / "train",
        img_size=args.img_size, augment=True, augment_prob=args.augment_prob
    )
    valid_dataset = DamageDataset(
        csv_valid, data_root / "valid",
        img_size=args.img_size, augment=False
    )
    test_dataset = DamageDataset(
        csv_test, data_root / "test",
        img_size=args.img_size, augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"📊 Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"📊 Valid: {len(valid_dataset)} samples, {len(valid_loader)} batches")
    print(f"📊 Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Create model
    model = DamageDetectionNet(
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout
    ).to(device)
    
    # Setup loss function with class weighting
    all_labels = torch.tensor(train_dataset.df["damage"].values).float()
    pos_weight = compute_pos_weight(all_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize training state
    start_epoch = 1
    best_score = -1.0
    history = {"train": [], "valid": []}
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, best_score, history = load_checkpoint(
                model, optimizer, resume_path, device
            )
            start_epoch += 1
        else:
            print(f"⚠️  Checkpoint not found: {resume_path}")
    
    # Check for existing checkpoint in output directory
    best_checkpoint = outdir / "best.pt"
    if best_checkpoint.exists() and not args.resume:
        print(f"📁 Found existing checkpoint: {best_checkpoint}")
        start_epoch, best_score, history = load_checkpoint(
            model, optimizer, best_checkpoint, device
        )
        start_epoch += 1
    
    print(f"🚀 Starting training from epoch {start_epoch}")
    print("-" * 60)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        print("🔍 Evaluating on validation set...")
        valid_metrics = evaluate_model(model, valid_loader, device, args.eval_threshold)
        
        # Update history
        history["train"].append(train_metrics)
        history["valid"].append({
            k: v for k, v in valid_metrics.items()
            if k not in ["y_true", "y_score", "y_pred"]  # Don't store arrays
        })
        
        # Compute composite score (F1 score for binary classification)
        current_score = valid_metrics["f1_score"]
        
        # Log metrics
        print(f"📊 Train Loss: {train_metrics['loss']:.4f}")
        print(f"📊 Valid Acc: {valid_metrics['accuracy']:.4f} | "
              f"Balanced Acc: {valid_metrics['balanced_accuracy']:.4f}")
        print(f"📊 Valid F1: {valid_metrics['f1_score']:.4f} | "
              f"AP: {valid_metrics['average_precision']:.4f}")
        
        # Save best model
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(model, optimizer, epoch, best_score, history, best_checkpoint)
            print(f"💾 New best model saved! F1 score: {best_score:.4f}")
        
        print("-" * 60)
    
    print("🎉 Training completed!")
    
    # Save final training history
    history_path = outdir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=to_jsonable)
    print(f"📊 Saved training history to {history_path}")
    
    # Load best model for final evaluation
    print("\n🔍 Final evaluation on test set...")
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(f"✅ Loaded best model (F1: {best_score:.4f})")
    
    # Test evaluation
    test_metrics = evaluate_model(model, test_loader, device, args.eval_threshold)
    
    # Save test results
    test_report_path = outdir / "test_report.json"
    test_report = {
        k: v for k, v in test_metrics.items()
        if k not in ["y_true", "y_score", "y_pred"]
    }
    with open(test_report_path, "w") as f:
        json.dump(test_report, f, indent=2, default=to_jsonable)
    
    print("🎯 Final Test Results:")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"   F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"   Average Precision: {test_metrics['average_precision']:.4f}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # Use validation metrics for plots
    val_metrics = evaluate_model(model, valid_loader, device, args.eval_threshold)
    
    # PR curve
    plot_precision_recall_curve(
        val_metrics["y_true"],
        val_metrics["y_score"],
        outdir / "val_pr_curve.png"
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        val_metrics["y_true"],
        val_metrics["y_pred"],
        outdir / "val_confusion.png"
    )
    
    # Sample predictions
    samples_dir = outdir / "samples"
    save_sample_predictions(model, valid_loader, device, samples_dir)
    
    print(f"\n✅ All results saved to: {outdir}")
    print("🔧 Damage detection training completed successfully!")


if __name__ == "__main__":
    main()
