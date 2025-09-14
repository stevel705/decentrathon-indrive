#!/usr/bin/env python3
"""
System health check script for Car Condition Analysis System.
Verifies installation, dependencies, models, and basic functionality.
"""

import sys
import subprocess
import importlib
from pathlib import Path
import torch

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False

def check_uv_installation():
    """Check if uv is installed and working"""
    print("📦 Checking uv installation...")
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ uv {result.stdout.strip()}")
            return True
        else:
            print("   ❌ uv not working properly")
            return False
    except FileNotFoundError:
        print("   ❌ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📚 Checking dependencies...")
    required_packages = [
        'torch',
        'torchvision', 
        'timm',
        'fastapi',
        'uvicorn',
        'ultralytics',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages: uv add {' '.join(missing_packages)}")
        return False
    return True

def check_cuda_support():
    """Check CUDA availability"""
    print("🖥️  Checking CUDA support...")
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"   ✅ CUDA available: {device_count} device(s)")
            print(f"   🎮 Primary GPU: {device_name}")
            return True
        else:
            print("   ⚠️  CUDA not available (will use CPU)")
            return False
    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print("📁 Checking project structure...")
    
    required_files = [
        'app.py',
        'train_cleanliness.py',
        'train_damage_detection.py', 
        'predict.py',
        'pyproject.toml',
        'requirements_web.txt'
    ]
    
    required_dirs = [
        'data',
        'weights',
        'runs'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
            missing_files.append(file)
    
    missing_dirs = []
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ⚠️  {dir_name}/ (missing, will be created)")
            missing_dirs.append(dir_name)
            
    # Create missing directories
    for dir_name in missing_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   📁 Created {dir_name}/")
    
    return len(missing_files) == 0

def check_models():
    """Check if trained models are available"""
    print("🤖 Checking trained models...")
    
    model_paths = [
        'weights/dirt_weights.pt',
        'weights/best_yolo_damage.pt',
        'runs/*/best.pt'
    ]
    
    found_models = []
    for pattern in model_paths:
        if '*' in pattern:
            matches = list(Path('.').glob(pattern))
            if matches:
                found_models.extend(matches)
                for match in matches[:3]:  # Show first 3 matches
                    print(f"   ✅ {match}")
                if len(matches) > 3:
                    print(f"   ... and {len(matches) - 3} more")
        else:
            if Path(pattern).exists():
                print(f"   ✅ {pattern}")
                found_models.append(Path(pattern))
            else:
                print(f"   ❌ {pattern} (missing)")
    
    if not found_models:
        print("   💡 No trained models found. Train models first using:")
        print("      uv run python train_cleanliness.py --data-root ./dataset_prepared")
        print("      uv run python train_damage_detection.py --data-root ./dataset_prepared")
        return False
    
    return True

def check_data():
    """Check if data is available"""
    print("📊 Checking data availability...")
    
    data_paths = [
        'data/',
        'dataset/',
        'dataset_prepared/'
    ]
    
    has_data = False
    for path in data_paths:
        p = Path(path)
        if p.exists() and any(p.iterdir()):
            print(f"   ✅ {path} (contains data)")
            has_data = True
        else:
            print(f"   ⚠️  {path} (empty or missing)")
    
    if not has_data:
        print("   💡 No data found. Prepare dataset using:")
        print("      uv run python scripts/prepare_dataset.py --input-dir ./dataset --output-dir ./dataset_prepared")
    
    return has_data

def run_basic_functionality_test():
    """Test basic functionality"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test model import
        from train_cleanliness import CleanlinessNet
        model = CleanlinessNet()
        print("   ✅ Model creation successful")
        
        # Test prediction script
        import predict
        print("   ✅ Prediction module import successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def main():
    """Run comprehensive system check"""
    print("🚗 Car Condition Analysis System - Health Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("uv Installation", check_uv_installation), 
        ("Dependencies", check_dependencies),
        ("CUDA Support", check_cuda_support),
        ("Project Structure", check_project_structure),
        ("Trained Models", check_models),
        ("Data Availability", check_data),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 30)
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"   ❌ Error during {name.lower()}: {e}")
            results[name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 System is ready! You can now:")
        print("   • Train models: uv run python train_cleanliness.py")
        print("   • Run predictions: uv run python predict.py")
        print("   • Start web app: uv run python app.py")
    else:
        print("⚠️  System needs attention. Address failing checks above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
