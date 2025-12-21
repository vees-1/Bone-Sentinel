# BoneSentinel 🦴  
**Automated Fracture Detection from X-ray Images using Deep Learning**

BoneSentinel is an end-to-end medical imaging project that detects abnormal bone X-rays using a convolutional neural network trained on the **MURA (Musculoskeletal Radiographs) dataset**.

---

## 🔍 Problem Statement
Manual fracture detection from X-ray images is time-consuming and subject to inter-observer variability. BoneSentinel aims to assist clinicians by providing an automated abnormality detection system.

---

## 📊 Dataset
- **MURA v1.1** (Stanford ML Group)
- Binary classification: **Normal vs Abnormal**
- X-ray studies of upper extremities

> Dataset is not included in this repository.

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



