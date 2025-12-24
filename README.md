# 🦴 BoneSentinel  
**Automated Abnormality Detection in X-ray Images Using Deep Learning**

BoneSentinel is an end-to-end medical imaging system designed to detect abnormal X-ray images using deep learning. The project implements a complete pipeline—from data preprocessing and model training to deployment using a FastAPI backend and a lightweight frontend interface.

The primary objective is to demonstrate a **stable, explainable, and deployable medical AI workflow**, rather than maximizing benchmark performance at this stage.


## 📌 Problem Statement

Manual interpretation of X-ray images is time-consuming and prone to inter-observer variability, especially under high clinical workload. BoneSentinel aims to assist clinicians by providing an automated abnormality detection system that can serve as a **decision-support tool**, improving efficiency and consistency.


## 🧠 Model Overview

- Architecture: **EfficientNetB0** (ImageNet pretrained)
- Task: Binary classification (Normal vs Abnormal)
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy, AUC (ROC)


## 🏗️ System Architecture

Frontend (HTML / CSS / JavaScript)  
↓  
FastAPI Backend (REST API)  
↓  
TensorFlow / Keras Model  

- The frontend allows users to upload X-ray images.
- The backend handles preprocessing and inference.
- The trained deep learning model performs abnormality detection in real time.


## 📂 Model Weights

The trained model file (`best_model.keras`) is **intentionally excluded** from this repository due to size constraints and best practices.

### To run the backend locally:

1. Train the model using the provided training script.
2. Place the generated model file at: backend/model/best_model.keras

> ⚠️ Note: Without this file, the FastAPI server will not start.



## 📊 Dataset Choice & Rationale

### Initial Dataset: MURA (Musculoskeletal Radiographs)

The project initially experimented with the **MURA dataset**, which is clinically relevant for bone abnormality detection. However, several practical challenges were encountered:

- Labels are provided at the **study level**, not per image  
- Many images in abnormal studies appear visually normal  
- High label noise leads to unstable training on limited compute resources  
- Effective training requires study-level aggregation and large-scale GPU usage  

### Final Dataset: NIH ChestX-ray14

To ensure a **reliable, reproducible, and well-evaluated prototype**, the project was transitioned to the **NIH ChestX-ray14 dataset**, which offers:

- Clean **image-level labels**
- Large-scale and diverse data
- Well-established benchmarks
- Better suitability for:
  - Transfer learning
  - Explainability techniques (e.g., Grad-CAM)
  - Rapid experimentation and deployment

The pipeline is **dataset-agnostic**, and extending it back to MURA using study-level aggregation is planned as future work.



## 🚀 Training & Performance

The model was trained using **transfer learning with EfficientNetB0** on a balanced subset of the NIH ChestX-ray14 dataset.

### Training Configuration

- Image Size: 224 × 224
- Backbone: Frozen EfficientNetB0
- Training Environment: Kaggle GPU
- Epochs: Limited (prototype phase)
- Dataset Size: Subsampled for rapid iteration

### Current Prototype Results

| Metric              | Value |
|---------------------|-------|
| Validation Accuracy | ~0.58 |
| Validation AUC      | ~0.58 |

These results reflect an **early-stage prototype**, prioritizing correctness and pipeline stability over raw performance.

### Expected Performance (Full Training)

Based on prior research and established benchmarks:

- **AUC of 0.75–0.85** is achievable with:
  - Larger training subsets
  - Fine-tuning deeper EfficientNet layers
  - Longer GPU training
- Further improvements using:
  - Data augmentation
  - Class-aware loss functions
  - Multi-label or ensemble approaches



## 🧪 Backend API

The backend is implemented using **FastAPI**, serving the trained deep learning model via a REST API.

### Key Features

- `/predict` endpoint for inference
- Multipart image upload support
- Real-time model predictions
- Auto-generated API documentation via Swagger UI

Once the server is running, the interactive API can be accessed at: http://localhost:8000/docs


---

## 🖥️ Frontend Interface

The frontend provides a simple web interface to upload X-ray images and view predictions, enabling easy testing and demonstration of the complete pipeline.

(Screenshots included in the repository)

---

## 🔮 Future Work

- Full-scale training on NIH ChestX-ray14
- Fine-tuning EfficientNet backbone
- Grad-CAM visual explanations
- Study-level aggregation for MURA dataset
- Cloud deployment (AWS / Azure)
- Multi-label abnormality classification

---

## 📎 Disclaimer

This project is intended for **educational and research purposes only** and is **not approved for clinical use**.

---

## 👨‍💻 Author

Developed as a deep learning and medical imaging project to demonstrate applied AI/ML skills, model deployment, and system design.





