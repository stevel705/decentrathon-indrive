#!/usr/bin/env python3
"""
Prediction script for car condition analysis models.
Supports both cleanliness classification and damage detection models.

⚠️ Key Features
- Binary classification inference for cleanliness (clean vs dirty)
- Binary classification inference for damage detection (damaged vs undamaged)
- Batch processing of multiple images
- Visualization of predictions with confidence scores
- Supports both individual images and directories
- Model architecture auto-detection
- Comprehensive output with statistics

Usage Examples:
    # Cleanliness prediction on single image
    python predict.py --checkpoint runs/cleanliness_exp/best.pt --images test_car.jpg --task cleanliness

    # Damage detection on directory of images
    python predict.py --checkpoint runs/damage_exp/best.pt --images ./test_images/ --task damage --threshold 0.7

    # Batch processing with custom output directory
    python predict.py --checkpoint weights/dirt_weights.pt --images ./cars/ --out ./predictions/ --batch-size 8

    # Auto-detect task from checkpoint name
    python predict.py --checkpoint runs/20240914_120000/best.pt --images ./test_data/

Model Support:
    - CleanlinessNet: Binary classification (clean=1, dirty=0)
    - DamageDetectionNet: Binary classification (damaged=1, undamaged=0)
    - Auto-detection based on checkpoint structure

Output:
    - Individual prediction images with confidence scores
    - Summary statistics (accuracy, confidence distribution)
    - CSV file with detailed predictions
    - Visualization grid showing sample predictions

Requirements:
    pip install torch torchvision timm pillow pandas matplotlib tqdm
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import timm
except ImportError:
    timm = None
    print("⚠️  Warning: timm not installed. Please run: pip install timm")

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Task-specific configurations
TASK_CONFIGS = {
    'cleanliness': {
        'classes': ['Dirty', 'Clean'],
        'colors': ['#8B4513', '#008000'],  # Brown for dirty, green for clean
        'model_class': 'CleanlinessNet'
    },
    'damage': {
        'classes': ['Undamaged', 'Damaged'], 
        'colors': ['#008000', '#DC143C'],  # Green for undamaged, red for damaged
        'model_class': 'DamageDetectionNet'
    }
}


class CleanlinessNet(nn.Module):
    """Binary classification network for car cleanliness detection"""
    
    def __init__(self, backbone="convnext_tiny", pretrained=False, dropout=0.1):
        super().__init__()
        
        if timm is None:
            raise ImportError("timm is required. Install with: pip install timm")
        
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool="avg"
        )
        
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(1)
        return logits


class DamageDetectionNet(nn.Module):
    """Binary classification network for car damage detection"""
    
    def __init__(self, backbone="convnext_tiny", pretrained=False, dropout=0.2):
        super().__init__()
        
        if timm is None:
            raise ImportError("timm is required. Install with: pip install timm")
        
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool="avg"
        )
        
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feat_dim // 2, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(1)
        return logits


def detect_task_from_checkpoint(checkpoint_path: Path) -> str:
    """Auto-detect task type from checkpoint path or content"""
    path_str = str(checkpoint_path).lower()
    
    # Check path for task indicators
    if 'clean' in path_str or 'dirt' in path_str:
        return 'cleanliness'
    elif 'damage' in path_str:
        return 'damage'
    
    # Try to load checkpoint and inspect model structure
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Check for damage detection specific layers (deeper classifier)
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
        if any('classifier.2' in k for k in classifier_keys):  # Has ReLU layer
            return 'damage'
        else:
            return 'cleanliness'
            
    except Exception:
        pass
    
    # Default fallback
    return 'cleanliness'


def detect_backbone_from_checkpoint(checkpoint_path: Path) -> str:
    """Detect backbone architecture from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Check for backbone-specific layer patterns
        backbone_keys = [k for k in state_dict.keys() if 'backbone' in k]
        
        if any('convnext' in k.lower() for k in backbone_keys):
            if 'stages.0.downsample' in str(backbone_keys):
                return 'convnext_tiny'
            else:
                return 'convnext_base'
        elif any('efficientnet' in k.lower() for k in backbone_keys):
            return 'efficientnet_b2'
        elif any('resnet' in k.lower() for k in backbone_keys):
            return 'resnet50'
        else:
            return 'convnext_tiny'  # Default
            
    except Exception:
        return 'convnext_tiny'  # Default fallback


def load_model(checkpoint_path: Path, task: str, backbone: str, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint"""
    print(f"📁 Loading {task} model from {checkpoint_path}")
    print(f"🏗️  Using backbone: {backbone}")
    
    # Create model based on task
    if task == 'cleanliness':
        model = CleanlinessNet(backbone=backbone, pretrained=False)
    elif task == 'damage':
        model = DamageDetectionNet(backbone=backbone, pretrained=False)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        if 'epoch' in checkpoint:
            print(f"📊 Loaded model from epoch {checkpoint['epoch']}")
        if 'best_score' in checkpoint:
            print(f"🎯 Best validation score: {checkpoint['best_score']:.4f}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def get_image_paths(input_path: Path) -> List[Path]:
    """Get list of image paths from input (file or directory)"""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.rglob(f'*{ext}'))
            image_paths.extend(input_path.rglob(f'*{ext.upper()}'))
        return sorted(list(set(image_paths)))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def create_transforms(img_size: int) -> transforms.Compose:
    """Create image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def predict_batch(model: nn.Module, images: List[Path], transform: transforms.Compose, 
                 device: torch.device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run prediction on a batch of images"""
    all_probs = []
    all_predictions = []
    
    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i + batch_size]
        batch_tensors = []
        
        # Load and preprocess batch
        for img_path in batch_paths:
            try:
                with Image.open(img_path).convert('RGB') as img:
                    tensor = transform(img)
                    batch_tensors.append(tensor)
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
                # Use zero tensor as fallback
                batch_tensors.append(torch.zeros(3, 384, 384))
        
        if not batch_tensors:
            continue
            
        # Stack batch and move to device
        batch = torch.stack(batch_tensors).to(device)
        
        # Get predictions
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_probs.extend(probs.tolist())
        all_predictions.extend(preds.tolist())
    
    return np.array(all_probs), np.array(all_predictions)


def create_prediction_image(image_path: Path, prediction: int, confidence: float, 
                          task: str, save_path: Path) -> None:
    """Create annotated image with prediction and confidence"""
    config = TASK_CONFIGS[task]
    
    # Load original image
    with Image.open(image_path).convert('RGB') as img:
        # Resize if too large
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(s * ratio) for s in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create drawing context
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font_large = ImageFont.truetype("arial.ttf", 36)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
                font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except Exception:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        # Get prediction info
        class_name = config['classes'][prediction]
        color = config['colors'][prediction]
        
        # Draw prediction banner
        banner_height = 80
        draw.rectangle([(0, 0), (img.width, banner_height)], fill=color)
        
        # Draw prediction text
        main_text = f"{class_name}"
        conf_text = f"Confidence: {confidence:.1%}"
        
        # Calculate text positions
        main_bbox = draw.textbbox((0, 0), main_text, font=font_large)
        conf_bbox = draw.textbbox((0, 0), conf_text, font=font_small)
        
        main_x = (img.width - (main_bbox[2] - main_bbox[0])) // 2
        conf_x = (img.width - (conf_bbox[2] - conf_bbox[0])) // 2
        
        # Draw text
        draw.text((main_x, 10), main_text, fill='white', font=font_large)
        draw.text((conf_x, 50), conf_text, fill='white', font=font_small)
        
        # Save annotated image
        img.save(save_path, quality=95)


def create_summary_visualization(results_df: pd.DataFrame, task: str, save_path: Path) -> None:
    """Create summary visualization of all predictions"""
    config = TASK_CONFIGS[task]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{task.title()} Prediction Summary', fontsize=16)
    
    # 1. Class distribution
    class_counts = results_df['prediction'].value_counts().sort_index()
    class_names = [config['classes'][i] for i in class_counts.index]
    colors = [config['colors'][i] for i in class_counts.index]
    
    axes[0, 0].bar(class_names, class_counts.values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # 2. Confidence distribution
    axes[0, 1].hist(results_df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(results_df['confidence'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {results_df["confidence"].mean():.2f}')
    axes[0, 1].legend()
    
    # 3. Confidence by class
    for i, class_name in enumerate(config['classes']):
        class_data = results_df[results_df['prediction'] == i]['confidence']
        if len(class_data) > 0:
            axes[1, 0].hist(class_data, bins=15, alpha=0.6, 
                          label=f'{class_name} (n={len(class_data)})',
                          color=config['colors'][i])
    axes[1, 0].set_title('Confidence by Class')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 4. Statistics table
    axes[1, 1].axis('off')
    stats_text = f"""
    Total Images: {len(results_df)}
    
    Class Distribution:
    """
    for i, class_name in enumerate(config['classes']):
        count = (results_df['prediction'] == i).sum()
        percentage = count / len(results_df) * 100
        stats_text += f"  {class_name}: {count} ({percentage:.1f}%)\n"
    
    stats_text += f"""
    Confidence Statistics:
      Mean: {results_df['confidence'].mean():.3f}
      Std:  {results_df['confidence'].std():.3f}
      Min:  {results_df['confidence'].min():.3f}
      Max:  {results_df['confidence'].max():.3f}
    
    High Confidence (>0.8): {(results_df['confidence'] > 0.8).sum()}
    Low Confidence (<0.6):  {(results_df['confidence'] < 0.6).sum()}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved summary visualization to {save_path}")


def create_sample_grid(image_paths: List[Path], predictions: np.ndarray, 
                      confidences: np.ndarray, task: str, save_path: Path, 
                      max_samples: int = 16) -> None:
    """Create grid visualization of sample predictions"""
    config = TASK_CONFIGS[task]
    
    n_samples = min(len(image_paths), max_samples)
    cols = 4
    rows = math.ceil(n_samples / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Sample {task.title()} Predictions', fontsize=16)
    
    for i in range(n_samples):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Load and display image
        try:
            with Image.open(image_paths[i]).convert('RGB') as img:
                # Resize for display
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'Error loading\nimage', ha='center', va='center', 
                   transform=ax.transAxes)
        
        # Add prediction info
        pred = predictions[i]
        conf = confidences[i]
        class_name = config['classes'][pred]
        color = config['colors'][pred]
        
        title = f'{class_name}\n{conf:.1%} confidence'
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️  Saved sample grid to {save_path}")


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description="Predict car condition from images")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--images", type=str, required=True,
                       help="Path to image file or directory of images")
    
    # Model arguments
    parser.add_argument("--task", type=str, choices=['cleanliness', 'damage'], default=None,
                       help="Task type (auto-detected if not specified)")
    parser.add_argument("--backbone", type=str, default=None,
                       help="Backbone architecture (auto-detected if not specified)")
    parser.add_argument("--img-size", type=int, default=384,
                       help="Input image size")
    
    # Prediction arguments
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    
    # Output arguments
    parser.add_argument("--out", type=str, default=None,
                       help="Output directory (default: predictions_TIMESTAMP)")
    parser.add_argument("--save-individual", action="store_true",
                       help="Save individual annotated images")
    parser.add_argument("--max-samples", type=int, default=16,
                       help="Maximum samples for visualization grid")
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    images_path = Path(args.images)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images path not found: {images_path}")
    
    # Auto-detect task and backbone if not specified
    task = args.task if args.task else detect_task_from_checkpoint(checkpoint_path)
    backbone = args.backbone if args.backbone else detect_backbone_from_checkpoint(checkpoint_path)
    
    print(f"🎯 Task: {task}")
    print(f"🏗️  Backbone: {backbone}")
    print(f"📏 Image size: {args.img_size}")
    print(f"🎚️  Threshold: {args.threshold}")
    
    # Setup output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"predictions_{task}_{timestamp}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {out_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, task, backbone, device)
    
    # Get image paths
    image_paths = get_image_paths(images_path)
    print(f"📸 Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    # Create transforms
    transform = create_transforms(args.img_size)
    
    # Run predictions
    print("🔍 Running predictions...")
    probabilities, predictions = predict_batch(
        model, image_paths, transform, device, args.batch_size
    )
    
    # Convert probabilities to confidences (max of prob and 1-prob)
    confidences = np.maximum(probabilities, 1 - probabilities)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'image_path': [str(p) for p in image_paths],
        'prediction': predictions,
        'probability': probabilities,
        'confidence': confidences,
        'class_name': [TASK_CONFIGS[task]['classes'][p] for p in predictions]
    })
    
    # Save results CSV
    csv_path = out_dir / 'predictions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"💾 Saved predictions to {csv_path}")
    
    # Print summary statistics
    print("\n📊 Prediction Summary:")
    class_counts = results_df['prediction'].value_counts().sort_index()
    for class_idx, count in class_counts.items():
        class_name = TASK_CONFIGS[task]['classes'][class_idx]
        percentage = count / len(results_df) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    print("\n🎯 Confidence Statistics:")
    print(f"   Mean: {confidences.mean():.3f}")
    print(f"   Std:  {confidences.std():.3f}")
    print(f"   Min:  {confidences.min():.3f}")
    print(f"   Max:  {confidences.max():.3f}")
    
    high_conf = (confidences > 0.8).sum()
    low_conf = (confidences < 0.6).sum()
    print(f"   High confidence (>0.8): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"   Low confidence (<0.6):  {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    # Summary visualization
    create_summary_visualization(results_df, task, out_dir / 'summary.png')
    
    # Sample grid
    create_sample_grid(image_paths, predictions, confidences, task, 
                      out_dir / 'sample_grid.png', args.max_samples)
    
    # Individual annotated images (if requested)
    if args.save_individual:
        print("🖼️  Creating individual annotated images...")
        individual_dir = out_dir / 'individual'
        individual_dir.mkdir(exist_ok=True)
        
        for i, (img_path, pred, conf) in enumerate(tqdm(
            zip(image_paths, predictions, confidences), 
            total=len(image_paths), desc="Annotating images"
        )):
            output_name = f"{img_path.stem}_pred_{TASK_CONFIGS[task]['classes'][pred].lower()}.jpg"
            output_path = individual_dir / output_name
            create_prediction_image(img_path, pred, conf, task, output_path)
        
        print(f"💾 Saved {len(image_paths)} annotated images to {individual_dir}")
    
    # Save prediction metadata
    metadata = {
        'checkpoint': str(checkpoint_path),
        'task': task,
        'backbone': backbone,
        'img_size': args.img_size,
        'threshold': args.threshold,
        'num_images': len(image_paths),
        'class_distribution': {
            TASK_CONFIGS[task]['classes'][i]: int(count) 
            for i, count in class_counts.items()
        },
        'confidence_stats': {
            'mean': float(confidences.mean()),
            'std': float(confidences.std()),
            'min': float(confidences.min()),
            'max': float(confidences.max())
        }
    }
    
    metadata_path = out_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Prediction completed successfully!")
    print(f"📂 All results saved to: {out_dir}")
    print(f"🎯 Processed {len(image_paths)} images with {task} model")


if __name__ == "__main__":
    main()
