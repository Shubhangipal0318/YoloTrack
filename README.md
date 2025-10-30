# 🤖 YOLOv8 Real-Time Object Detection using OpenCV

This project demonstrates a **real-time object detection system** powered by **Ultralytics YOLOv8** and **OpenCV**.  
It captures live video from a webcam, detects multiple objects within each frame, and displays bounding boxes, class labels, and confidence scores dynamically.

The implementation serves as a foundation for **AI-based vision systems**, robotics projects like the **Mopping Robot**, and autonomous navigation.

---

## 🧠 Overview

This Python-based project leverages the power of **deep learning** for object detection tasks in real time.  
Using the **YOLOv8** model (You Only Look Once, version 8), the system performs **fast and accurate detection** of common objects directly from live video input.

YOLOv8 is pre-trained on the COCO dataset, capable of detecting **80+ object categories** including people, vehicles, household items, and more.

---

## 🏗️ System Architecture

[ Webcam ] ---> [ Frame Capture (OpenCV) ]--->[ YOLOv8 Model Inference ]--->[ Object Bounding Boxes + Labels ]--->[ Visualization (OpenCV Window) ]


---

## 🚀 Features

- 🎥 **Real-Time Detection:** Processes live video streams from the webcam with minimal latency.  
- 🧭 **Accurate Bounding Boxes:** Displays class names and confidence scores for each object.  
- ⚡ **Lightweight Inference:** Uses YOLOv8-Small (`yolov8s.pt`) for fast performance on CPUs/GPUs.  
- 🧩 **Customizable:** Easily switch to other YOLOv8 variants (`n`, `m`, `l`, `x`) or custom-trained weights.  
- 🧠 **Expandable:** Ideal for integration into robotics (e.g., obstacle avoidance in cleaning robots).  

---

## 🧩 Requirements

Make sure you have Python 3.8+ installed.

Install the required libraries:

```bash
pip install ultralytics opencv-python
pip install torch torchvision
