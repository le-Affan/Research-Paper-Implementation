# Brain Tumor Detection with YOLOv7

A comprehensive implementation of brain tumor detection using YOLOv7 deep learning framework on MRI images. This project demonstrates the application of state-of-the-art object detection techniques for medical imaging analysis.

## 🧠 Project Overview

This project implements a brain tumor detection system using YOLOv7, trained on MRI scans from multiple anatomical views (axial, coronal, and sagittal). The model can identify and localize brain tumors with high accuracy, making it a valuable tool for medical diagnosis assistance.

### Key Features

- **Multi-view Analysis**: Supports axial, coronal, and sagittal MRI views
- **Binary Classification**: Distinguishes between positive (tumor present) and negative (no tumor) cases
- **High Performance**: Leverages YOLOv7's advanced architecture for accurate detection
- **Medical Grade**: Designed specifically for medical imaging applications

## 📁 Project Structure

```
├── datasets/                    # Dataset configurations and data
│   ├── axial_t1wce_2_class/    # Axial view dataset
│   ├── coronal_t1wce_2_class/  # Coronal view dataset
│   └── sagittal_t1wce_2_class/ # Sagittal view dataset
├── yolov7/                     # YOLOv7 implementation
│   ├── models/                 # Model definitions
│   ├── utils/                  # Utility functions
│   ├── train.py               # Training script
│   ├── detect.py              # Detection script
│   └── requirements.txt       # Dependencies
├── docs/                      # Documentation
│   └── EXP-8_Miniproject.pdf  # Project report
├── scripts/                   # Training and evaluation scripts
├── requirements.txt           # Main project dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/le-Affan/Research-Paper-Implementation.git
   cd Research-Paper-Implementation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to YOLOv7 directory**
   ```bash
   cd yolov7
   pip install -r requirements.txt
   ```

### Training

1. **Prepare your dataset**
   - Organize MRI images in YOLO format
   - Update dataset configuration files in `datasets/` directory

2. **Train the model**
   ```bash
   python train.py --weights yolov7.pt --data ../datasets/axial_t1wce_2_class/axial_t1wce_2_class.yaml --epochs 100 --batch-size 16 --img 640
   ```

3. **Monitor training**
   - View training progress in `runs/train/exp/`
   - Use TensorBoard for detailed metrics

### Inference

1. **Run detection on new images**
   ```bash
   python detect.py --weights runs/train/exp/weights/best.pt --source /path/to/test/image.jpg --conf 0.5
   ```

2. **Batch processing**
   ```bash
   python detect.py --weights runs/train/exp/weights/best.pt --source /path/to/test/folder/ --conf 0.5
   ```

## 📊 Dataset Information

### Dataset Structure
- **Total Classes**: 2 (positive, negative)
- **Image Views**: Axial, Coronal, Sagittal
- **Format**: YOLO format with bounding box annotations
- **Split**: Train/Test (approximately 80/20)

### Dataset Statistics
- **Axial View**: 310 training images, 70 test images
- **Coronal View**: 319 training images, 78 test images  
- **Sagittal View**: 264 training images, 70 test images

## 🔧 Configuration

### Model Configuration
- **Architecture**: YOLOv7
- **Input Size**: 640x640 pixels
- **Classes**: 2 (negative, positive)
- **Confidence Threshold**: 0.5 (adjustable)

### Training Parameters
- **Epochs**: 100 (configurable)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Learning Rate**: Auto-scaled by YOLOv7
- **Optimizer**: AdamW

## 📈 Performance Metrics

The model performance is evaluated using standard object detection metrics:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)

## 🛠️ Advanced Usage

### Custom Dataset Training

1. **Prepare your data**
   ```bash
   # Organize images in YOLO format
   your_dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. **Create dataset configuration**
   ```yaml
   # your_dataset.yaml
   path: /path/to/your_dataset
   train: images/train
   val: images/val
   nc: 2
   names: ['negative', 'positive']
   ```

3. **Train with custom data**
   ```bash
   python train.py --data your_dataset.yaml --weights yolov7.pt --epochs 100
   ```

### Model Export

Export trained models to different formats:

```bash
# Export to ONNX
python export.py --weights runs/train/exp/weights/best.pt --include onnx

# Export to TensorRT
python export.py --weights runs/train/exp/weights/best.pt --include engine
```

## 📚 Documentation

- **Project Report**: See `docs/EXP-8_Miniproject.pdf` for detailed methodology
- **YOLOv7 Paper**: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
- **Original Implementation**: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WongKinYiu](https://github.com/WongKinYiu) for the original YOLOv7 implementation
- Medical imaging community for dataset contributions
- PyTorch team for the deep learning framework

## 📞 Contact

For questions and support, please open an issue or contact [@le-Affan](https://github.com/le-Affan).

---

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used as the sole basis for medical diagnosis. Always consult with qualified medical professionals for clinical decisions.