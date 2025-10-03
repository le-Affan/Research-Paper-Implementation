#!/usr/bin/env python3
"""
Brain Tumor Detection Training Script
=====================================

This script provides an easy-to-use interface for training YOLOv7 models
on brain tumor detection datasets with different MRI views.

Usage:
    python scripts/train_brain_tumor.py --view axial --epochs 100
    python scripts/train_brain_tumor.py --view coronal --epochs 150
    python scripts/train_brain_tumor.py --view sagittal --epochs 120
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def get_dataset_path(view):
    """Get the dataset configuration path for the specified view."""
    dataset_paths = {
        'axial': 'datasets/axial_t1wce_2_class/axial_t1wce_2_class.yaml',
        'coronal': 'datasets/coronal_t1wce_2_class/coronal_t1wce_2_class.yaml',
        'sagittal': 'datasets/sagittal_t1wce_2_class/sagittal_t1wce_2_class.yaml'
    }
    return dataset_paths.get(view.lower())

def train_model(view, epochs=100, batch_size=16, img_size=640, device='', project='runs/train'):
    """Train YOLOv7 model on brain tumor detection dataset."""
    
    # Get dataset path
    data_path = get_dataset_path(view)
    if not data_path or not os.path.exists(data_path):
        print(f"❌ Dataset configuration not found for view: {view}")
        print(f"Expected path: {data_path}")
        return False
    
    # Change to yolov7 directory
    yolov7_dir = Path(__file__).parent.parent / 'yolov7'
    if not yolov7_dir.exists():
        print(f"❌ YOLOv7 directory not found: {yolov7_dir}")
        return False
    
    # Prepare training command
    cmd = [
        'python', 'train.py',
        '--weights', 'yolov7.pt',
        '--data', f'../{data_path}',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--img', str(img_size),
        '--project', f'../{project}',
        '--name', f'brain_tumor_{view}',
        '--save-period', '10'
    ]
    
    if device:
        cmd.extend(['--device', device])
    
    print(f"🚀 Starting training for {view} view...")
    print(f"📊 Dataset: {data_path}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖼️  Image size: {img_size}")
    print(f"💾 Project: {project}/brain_tumor_{view}")
    print()
    
    try:
        # Change to yolov7 directory and run training
        result = subprocess.run(cmd, cwd=yolov7_dir, check=True)
        print(f"✅ Training completed successfully for {view} view!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {view} view: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Training interrupted for {view} view")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv7 for brain tumor detection')
    parser.add_argument('--view', type=str, required=True, 
                       choices=['axial', 'coronal', 'sagittal'],
                       help='MRI view to train on')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use for training (e.g., 0 for GPU 0)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory for saving results')
    
    args = parser.parse_args()
    
    print("🧠 Brain Tumor Detection Training")
    print("=" * 40)
    
    success = train_model(
        view=args.view,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project
    )
    
    if success:
        print(f"\n🎉 Training completed! Check results in {args.project}/brain_tumor_{args.view}/")
        print(f"📈 View training metrics with: tensorboard --logdir {args.project}/brain_tumor_{args.view}/")
    else:
        print(f"\n❌ Training failed for {args.view} view")
        sys.exit(1)

if __name__ == '__main__':
    main()
