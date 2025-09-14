#!/usr/bin/env python3
"""
Example usage of predict.py for car condition analysis

This script demonstrates how to use the prediction module
for both cleanliness and damage detection tasks.
"""

import subprocess
import sys
from pathlib import Path

def run_prediction_example():
    """Run prediction examples for both tasks"""
    
    print("🚗 Car Condition Prediction Examples")
    print("=" * 50)
    
    # Check if we have test data
    test_images = Path("./data/test")  # Adjust path as needed
    single_image = Path("./data/test.jpg")  # Adjust path as needed
    
    if not test_images.exists() and not single_image.exists():
        print("⚠️  No test images found. Please ensure you have test data available.")
        print("Expected paths:")
        print(f"  - Directory: {test_images}")
        print(f"  - Single image: {single_image}")
        return
    
    # Example 1: Cleanliness prediction with auto-detection
    print("\n1️⃣  Cleanliness Prediction (Auto-detection)")
    print("-" * 40)
    
    if Path("weights/dirt_weights.pt").exists():
        cmd = [
            sys.executable, "predict.py",
            "--checkpoint", "weights/dirt_weights.pt",
            "--images", str(single_image if single_image.exists() else test_images),
            "--save-individual"
        ]
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Cleanliness prediction completed successfully!")
                print(result.stdout[-500:])  # Show last 500 chars
            else:
                print("❌ Error in cleanliness prediction:")
                print(result.stderr[-500:])
        except subprocess.TimeoutExpired:
            print("⏰ Prediction timed out")
        except Exception as e:
            print(f"❌ Error running prediction: {e}")
    else:
        print("⚠️  Cleanliness model not found at weights/dirt_weights.pt")
    
    # Example 2: Damage detection with explicit task specification
    print("\n\n2️⃣  Damage Detection (Explicit task)")
    print("-" * 40)
    
    damage_model = None
    for path in ["runs/damage_*/best.pt", "weights/damage_weights.pt"]:
        matches = list(Path(".").glob(path))
        if matches:
            damage_model = matches[0]
            break
    
    if damage_model and damage_model.exists():
        cmd = [
            sys.executable, "predict.py",
            "--checkpoint", str(damage_model),
            "--images", str(single_image if single_image.exists() else test_images),
            "--task", "damage",
            "--threshold", "0.7",
            "--batch-size", "8"
        ]
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Damage detection completed successfully!")
                print(result.stdout[-500:])
            else:
                print("❌ Error in damage detection:")
                print(result.stderr[-500:])
        except subprocess.TimeoutExpired:
            print("⏰ Prediction timed out")
        except Exception as e:
            print(f"❌ Error running prediction: {e}")
    else:
        print("⚠️  Damage detection model not found")
    
    # Example 3: Batch processing with custom output
    print("\n\n3️⃣  Batch Processing Example")
    print("-" * 40)
    
    if Path("weights/dirt_weights.pt").exists() and test_images.exists():
        cmd = [
            sys.executable, "predict.py",
            "--checkpoint", "weights/dirt_weights.pt",
            "--images", str(test_images),
            "--out", "./batch_predictions",
            "--batch-size", "16",
            "--max-samples", "20"
        ]
        print(f"Command: {' '.join(cmd)}")
        print("💡 This would process all images in the test directory")
    else:
        print("⚠️  Requirements for batch processing not met")
    
    print("\n\n📚 Usage Tips:")
    print("=" * 50)
    print("1. The script auto-detects task type from checkpoint name/structure")
    print("2. Use --save-individual to get annotated images for each prediction")
    print("3. Adjust --threshold to change sensitivity (0.5 = balanced, >0.5 = stricter)")
    print("4. Use --batch-size to control memory usage during processing")
    print("5. Output includes CSV with predictions, summary plots, and sample grid")
    
    print("\n🎯 Available Models:")
    print("-" * 20)
    
    # Check for available models
    model_paths = [
        "weights/dirt_weights.pt",
        "runs/*/best.pt",
        "runs_*/best.pt"
    ]
    
    found_models = []
    for pattern in model_paths:
        found_models.extend(Path(".").glob(pattern))
    
    if found_models:
        for model in found_models:
            print(f"  ✅ {model}")
    else:
        print("  ❌ No trained models found")
        print("  💡 Train models first using train_cleanliness.py or train_damage_detection.py")

if __name__ == "__main__":
    run_prediction_example()
