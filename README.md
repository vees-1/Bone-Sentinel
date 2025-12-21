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

The current implementation prioritizes **correctness, stability, and explainability** over peak performance, with full-scale training planned as future work.
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



