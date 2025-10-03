#!/usr/bin/env python3
"""
Brain Tumor Detection Model Evaluation Script
============================================

This script provides comprehensive evaluation of trained YOLOv7 models
for brain tumor detection across different MRI views.

Usage:
    python scripts/evaluate_model.py --weights runs/train/brain_tumor_axial/weights/best.pt --data datasets/axial_t1wce_2_class/axial_t1wce_2_class.yaml
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def evaluate_model(weights_path, data_path, device='', project='runs/val'):
    """Evaluate trained YOLOv7 model on test dataset."""
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        return False
    
    # Check if data configuration exists
    if not os.path.exists(data_path):
        print(f"❌ Data configuration not found: {data_path}")
        return False
    
    # Change to yolov7 directory
    yolov7_dir = Path(__file__).parent.parent / 'yolov7'
    if not yolov7_dir.exists():
        print(f"❌ YOLOv7 directory not found: {yolov7_dir}")
        return False
    
    # Prepare evaluation command
    cmd = [
        'python', 'test.py',
        '--weights', f'../{weights_path}',
        '--data', f'../{data_path}',
        '--project', f'../{project}',
        '--name', 'brain_tumor_evaluation',
        '--save-txt',
        '--save-conf',
        '--verbose'
    ]
    
    if device:
        cmd.extend(['--device', device])
    
    print(f"📊 Evaluating brain tumor detection model...")
    print(f"⚖️  Weights: {weights_path}")
    print(f"📁 Data: {data_path}")
    print(f"💾 Output: {project}/brain_tumor_evaluation/")
    print()
    
    try:
        # Change to yolov7 directory and run evaluation
        result = subprocess.run(cmd, cwd=yolov7_dir, check=True)
        print(f"✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Evaluation interrupted")
        return False

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained YOLOv7 model for brain tumor detection')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration file (.yaml)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use for evaluation (e.g., 0 for GPU 0)')
    parser.add_argument('--project', type=str, default='runs/val',
                       help='Project directory for saving results')
    
    args = parser.parse_args()
    
    print("🧠 Brain Tumor Detection Model Evaluation")
    print("=" * 45)
    
    success = evaluate_model(
        weights_path=args.weights,
        data_path=args.data,
        device=args.device,
        project=args.project
    )
    
    if success:
        print(f"\n🎉 Evaluation completed! Check results in {args.project}/brain_tumor_evaluation/")
        print(f"📈 Key metrics available:")
        print(f"   - mAP@0.5: Mean Average Precision at IoU 0.5")
        print(f"   - mAP@0.5:0.95: Mean Average Precision across IoU thresholds")
        print(f"   - Precision and Recall for each class")
        print(f"   - Confusion matrix")
        print(f"   - Detection results with confidence scores")
    else:
        print(f"\n❌ Evaluation failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
