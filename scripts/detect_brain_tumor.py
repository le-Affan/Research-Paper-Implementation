#!/usr/bin/env python3
"""
Brain Tumor Detection Inference Script
======================================

This script provides an easy-to-use interface for running inference
on brain tumor detection using trained YOLOv7 models.

Usage:
    python scripts/detect_brain_tumor.py --weights runs/train/brain_tumor_axial/weights/best.pt --source test_image.jpg
    python scripts/detect_brain_tumor.py --weights runs/train/brain_tumor_axial/weights/best.pt --source test_folder/ --conf 0.5
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def detect_tumors(weights_path, source, conf_threshold=0.5, device='', project='runs/detect'):
    """Run brain tumor detection on images."""
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        return False
    
    # Check if source exists
    if not os.path.exists(source):
        print(f"❌ Source not found: {source}")
        return False
    
    # Change to yolov7 directory
    yolov7_dir = Path(__file__).parent.parent / 'yolov7'
    if not yolov7_dir.exists():
        print(f"❌ YOLOv7 directory not found: {yolov7_dir}")
        return False
    
    # Prepare detection command
    cmd = [
        'python', 'detect.py',
        '--weights', f'../{weights_path}',
        '--source', f'../{source}',
        '--conf', str(conf_threshold),
        '--project', f'../{project}',
        '--name', 'brain_tumor_detection',
        '--save-txt',
        '--save-conf'
    ]
    
    if device:
        cmd.extend(['--device', device])
    
    print(f"🔍 Running brain tumor detection...")
    print(f"⚖️  Weights: {weights_path}")
    print(f"📁 Source: {source}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    print(f"💾 Output: {project}/brain_tumor_detection/")
    print()
    
    try:
        # Change to yolov7 directory and run detection
        result = subprocess.run(cmd, cwd=yolov7_dir, check=True)
        print(f"✅ Detection completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Detection interrupted")
        return False

def main():
    parser = argparse.ArgumentParser(description='Detect brain tumors using trained YOLOv7 model')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, folder, or video for detection')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use for inference (e.g., 0 for GPU 0)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory for saving results')
    
    args = parser.parse_args()
    
    print("🧠 Brain Tumor Detection")
    print("=" * 30)
    
    success = detect_tumors(
        weights_path=args.weights,
        source=args.source,
        conf_threshold=args.conf,
        device=args.device,
        project=args.project
    )
    
    if success:
        print(f"\n🎉 Detection completed! Check results in {args.project}/brain_tumor_detection/")
        print(f"📊 Results include:")
        print(f"   - Annotated images with bounding boxes")
        print(f"   - Text files with detection coordinates")
        print(f"   - Confidence scores for each detection")
    else:
        print(f"\n❌ Detection failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
