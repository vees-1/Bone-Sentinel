# BoneSentinel 🦴  
**Automated Fracture Detection from X-ray Images using Deep Learning**

BoneSentinel is an end-to-end medical imaging project that detects abnormal bone X-rays using a convolutional neural network trained on the **MURA (Musculoskeletal Radiographs) dataset**.

---

## 🔍 Problem Statement
Manual fracture detection from X-ray images is time-consuming and subject to inter-observer variability. BoneSentinel aims to assist clinicians by providing an automated abnormality detection system.

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
Frontend (HTML/CSS/JS)
↓
FastAPI Backend
↓
TensorFlow Model


---

## ⚙️ Training
- Local CPU training for prototyping
- Safe configuration (low batch size, reduced dataset)
- Validation AUC ≈ **0.68**
- Model saved as `best_model.keras`

> Full GPU training (Colab/Kaggle) recommended for higher performance.

---

## 🚀 How to Run Locally

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python -m uvicorn backend.main:app --reload



