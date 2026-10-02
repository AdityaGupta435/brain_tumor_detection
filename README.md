# 🧠 Brain Tumor Classifier

A deep learning-based brain MRI image classification web application built using **PyTorch** and **Streamlit**.

## 📌 Features

- Upload brain MRI images
- Classify images into 4 categories
- Display prediction confidence
- Display probability for all classes
- Simple and responsive Streamlit interface
- CPU and CUDA support
- Educational medical AI project

## 🧠 Supported Classes

The model supports four classes:

1. Glioma Tumor
2. Meningioma Tumor
3. No Tumor
4. Pituitary Tumor

## 🛠️ Technology Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow

## 📐 Model Input

The model expects:

- Image size: `250 × 250`
- Channels: `RGB`
- Normalization:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
