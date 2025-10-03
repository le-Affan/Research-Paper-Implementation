# Mini-Project: Brain Tumor Detection with YOLOv7

This project reproduces the paper "Brain Tumor Detection via Deep Learning Framework on MRI Images."

## Description
The project uses the YOLOv7 object detection model to identify and locate brain tumors in MRI scans. The methodology from the original paper was reproduced, and the framework was then extended to a new task: detecting pneumonia in chest X-ray images.

## How to Run
1. Clone the YOLOv7 repository: `git clone https://github.com/WongKinYiu/yolov7.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Place the sample dataset (images and labels) in the `/data` directory.
4. To train on the custom dataset, run: `python train.py --weights yolov7.pt --data data/custom.yaml`
5. To run detection on an image, run: `python detect.py --weights best.pt --source /path/to/image.jpg`
