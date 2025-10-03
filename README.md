# Research Paper Implementation: Brain Tumor Detection with YOLOv7

This repository contains the implementation for the paper "Brain Tumor Detection via Deep Learning Framework on MRI Images."

## Project Overview

This project first reproduces the original paper's methodology, which uses the YOLOv7 object detection model to identify and locate brain tumors in MRI scans.

Secondly, the project extends the framework to a new medical imaging domain: detecting pneumonia in chest X-ray images. This demonstrates the adaptability and robustness of the YOLOv7 model for various diagnostic tasks.

## How to Run the Implementation

### 1. Setup
- Clone the official YOLOv7 repository:
  `git clone https://github.com/WongKinYiu/yolov7.git`
- Navigate into the cloned directory:
  `cd yolov7`
- Install the required dependencies:
  `pip install -r requirements.txt`

### 2. Prepare Data
- Organize your dataset (e.g., Brain Tumors or Pneumonia X-rays) into the YOLO format.
- Create a `custom_data.yaml` file in the `data/` directory pointing to your train/validation sets and defining your class names.

### 3. Train the Model
- Run the training script using pre-trained weights:
  `python train.py --weights yolov7.pt --data data/custom_data.yaml --epochs 100 --batch-size 16 --img 640`

### 4. Run Detection
- Use the newly trained weights (found in `runs/train/exp/weights/best.pt`) to run inference on new images:
  `python detect.py --weights runs/train/exp/weights/best.pt --source /path/to/your/test_image.jpg`
