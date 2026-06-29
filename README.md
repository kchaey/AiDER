# 🩺 AiDER: Multimodal Skin Lesion Analysis System

> **AiDER** is an end-to-end multimodal AI pipeline designed to comprehensively analyze skin lesions and predict acne severity grades. By fusing computer vision features extracted from medical imagery with clinical and lifestyle patient questionnaire data, AiDER provides structured, evidence-based dermatological insights.

---

## 🌟 Core Architecture & Pipeline Steps

The system seamlessly synchronizes an 11-step complete medical analysis pipeline down to a unified web interface, divided into the following operational stages:

1. **Image Quality Verification**
   - Inspects input imagery for resolution, illumination, and blurring thresholds using Laplacian variance metrics to prevent uninterpretable low-quality data from passing down the pipeline.
2. **YOLOv8-based Lesion Detection & ROI Localization**
   - Localizes active skin abnormalities and blemishes in real-time, instantly extracting bounding boxes and counting independent regions of interest (ROIs) along with total pixel surface area.
3. **CNN Classification & Feature Aggregation**
   - Passes each cropped ROI through a custom Deep Convolutional Neural Network (PyTorch) to classify exact condition probabilities, combining them via a weighted confidence-aggregation logic.
4. **Tabular Lifestyle Fusion (XGBoost)**
   - Combines the extracted visual metrics (acne probability, lesion count, total surface area) with vital clinical/behavioral demographics (age, sex, sleep hours, stress levels, cosmetics adjustments, recent hormonal/lifestyle variables) to accurately evaluate the final **Hayashi Severity Grade (0–3)**.

---

## 🛠️ Tech Stacks & System Specifications

| Component / Layer | Technologies Used |
| :--- | :--- |
| **Frontend & Deployment** | Streamlit Community Cloud, Python Web Server Architecture |
| **Computer Vision (Detection)** | YOLOv8 (Ultralytics Framework), OpenCV (Image Preprocessing) |
| **Computer Vision (Classification)** | PyTorch, Torchvision Models, Deep Learning ROI Classifier |
| **Multimodal Tabular Fusion** | XGBoost (Extreme Gradient Boosting Classifier), Scikit-Learn |
| **Data Engineering** | Pandas DataFrames, NumPy Arrays, Synthetic Profiling Matrix |

---

## 📦 Installation & Environment Setup

To deploy this application to **Streamlit Community Cloud** or host it locally, ensure you position a `requirements.txt` file in your repository base containing the correct headless dependencies to bypass GUI Linux server restrictions:

```text
streamlit
xgboost
scikit-learn
pandas
numpy
opencv-python-headless
torch
torchvision
ultralytics
Pillow
