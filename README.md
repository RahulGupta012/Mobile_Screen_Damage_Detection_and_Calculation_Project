# 📱 Refurbished Phone Screen Damage Assessment System

An AI-powered Computer Vision application designed to assist buyers, sellers, and refurbishment businesses in evaluating the physical condition of smartphone screens from images. The system automatically detects a phone, isolates it from the background, and estimates screen damage severity by analyzing visible cracks and scratches.

---

## Features

* Automated smartphone detection using YOLO
* Precise phone segmentation using Mask R-CNN
* Background removal for accurate damage analysis
* Crack and scratch extraction using Canny Edge Detection
* Screen damage percentage estimation
* Interactive Streamlit-based graphical interface
* End-to-end automated image processing pipeline
* Suitable for refurbished device quality assessment workflows

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red)
![Mask R-CNN](https://img.shields.io/badge/Mask%20R--CNN-Instance%20Segmentation-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blue?logo=opencv)
![Canny Edge Detection](https://img.shields.io/badge/Canny-Edge%20Detection-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-GUI-FF4B4B?logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-AI-purple)

---

##  System Pipeline

Image Input → YOLO Detection → Phone Localization → Mask R-CNN Segmentation → Background Removal → Edge Detection → Damage Analysis → Damage Percentage Estimation

---

## Methodology

### 1. Phone Detection

A YOLO-based object detection model is used to identify and localize smartphones within the input image.

### 2. Phone Segmentation

The detected phone region is passed through a pre-trained Mask R-CNN model to generate a precise segmentation mask, effectively removing irrelevant background information.

### 3. Damage Analysis

The segmented phone screen is processed using Canny Edge Detection to identify crack-like and scratch-like patterns.

### 4. Damage Estimation

The detected damaged regions are analyzed to estimate the percentage of affected screen area, providing an objective assessment of screen condition.

---

##  Demo

A demonstration video has been included in the repository showcasing:

* End-to-end inference pipeline
* Model performance
* Example predictions
* Usage workflow

---

##  Running the Application

### Launch GUI Interface

```bash
streamlit run streamlit.py
```

### Run Damage Calculation Pipeline

```bash
python seg_img_damage_calculation.py
```

---

## 📂 Dataset Acknowledgement

Special thanks to **DATACLUSTER LABS** on Kaggle for providing the dataset used to train the phone detection model.

---

##  Potential Applications

* Refurbished smartphone marketplaces
* Mobile device trade-in platforms
* Automated quality inspection systems
* E-commerce product verification
* Insurance claim assessment
* Repair cost estimation workflows

---

## Key Highlights

* Multi-stage Computer Vision pipeline
* Object Detection + Instance Segmentation integration
* Automated damage quantification
* Real-time user interface with Streamlit
* Scalable architecture for refurbishment and inspection use cases

