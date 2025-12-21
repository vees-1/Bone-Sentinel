# BoneSentinel 🦴  
**Automated Fracture Detection from X-ray Images using Deep Learning**

BoneSentinel is an end-to-end medical imaging project that detects abnormal bone X-rays using a convolutional neural network trained on the **NIH ChestX-ray14 dataset**.

---

## 🔍 Problem Statement
Manual fracture detection from X-ray images is time-consuming and subject to inter-observer variability. BoneSentinel aims to assist clinicians by providing an automated abnormality detection system.

---

## Model Weights

The trained model file (`best_model.keras`) is intentionally not included in this repository
due to size constraints and best practices.

To run the backend locally:

1. Train the model using the provided training script
2. Place the generated model at:

   backend/model/best_model.keras

Without this file, the FastAPI server will not start.

---

## 📌 Dataset Choice & Rationale

This project initially experimented with the **MURA (Musculoskeletal Radiographs) dataset** for bone abnormality detection.  
While MURA is clinically relevant, it presents several practical challenges for efficient prototyping:

- Labels are provided at the **study level**, not the individual image level  
- Many images within an abnormal study may appear visually normal  
- High label noise makes training unstable on limited compute resources  

As a result, achieving reliable convergence without large-scale GPU training and study-level aggregation is difficult.

To ensure a **stable, reproducible, and well-evaluated pipeline**, the project was transitioned to the **NIH ChestX-ray14 dataset**, which offers:

- **Image-level labels** (clean supervision)
- Large-scale, diverse data
- Well-established benchmarks
- Better suitability for transfer learning and explainability (Grad-CAM)

This switch allows the project to focus on:
- Model performance
- Explainability
- End-to-end deployment (FastAPI + frontend)

The pipeline remains **dataset-agnostic**, and extending it back to MURA using study-level aggregation is planned as future work.

---

## 🧠 Model
- **EfficientNetB0** (transfer learning)
- Binary classification head
- Loss: Binary Cross-Entropy
- Metrics: Accuracy, AUC

---

## 🏗️ System Architecture
Frontend (HTML/CSS/JS) -->

FastAPI Backend -->

TensorFlow Model


---

## 🧪 Training & Performance

The model was trained using **transfer learning with EfficientNetB0** on a balanced subset of the **NIH ChestX-ray14 dataset** for binary classification (Normal vs Abnormal).

### 🔹 Training Setup
- Architecture: EfficientNetB0 (ImageNet pretrained)
- Image size: 224 × 224
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Training environment: Kaggle GPU
- Dataset: NIH ChestX-ray14 (subset for prototyping)

### 🔹 Current Results (Prototype)
| Metric | Value |
|------|------|
| Validation Accuracy | ~0.58 |
| Validation AUC | ~0.58 |

These results correspond to an **early-stage prototype**, trained with:
- Frozen backbone
- Limited epochs
- Subsampled data

### 🔹 Expected Performance with Full Training
Based on established benchmarks and prior research on NIH ChestX-ray14:

- **AUC 0.75–0.85** is achievable with:
  - Larger training subsets
  - Fine-tuning of deeper EfficientNet layers
  - Longer GPU training
- Further improvements are possible using:
  - Class-aware loss functions
  - Data augmentation
  - Ensemble or multi-label learning

The current implementation prioritizes **correctness, stability, and explainability** over peak performance, with full-scale training planned as future work

---
## Project Demo

### Frontend Interface
<img width="1767" height="910" alt="image" src="https://github.com/user-attachments/assets/bb54f61f-c43e-46d8-b5dc-62d6812c2d0a" />
<img width="1767" height="910" alt="image" src="https://github.com/user-attachments/assets/744a7789-9022-442e-8a02-db0e560e0625" />

### Backend API (FastAPI)

The backend is implemented using **FastAPI** and serves a deep learning model for
bone abnormality detection from X-ray images.

Key features:
- REST API with `/predict` endpoint
- Multipart image upload support
- Real-time inference using a trained CNN model
- Interactive API documentation via Swagger UI

The screenshot above shows the auto-generated OpenAPI interface,
allowing easy testing and integration.
<img width="1767" height="910" alt="image" src="https://github.com/user-attachments/assets/2a374dc9-4362-4d8f-a2ed-8e016686fc20" />






