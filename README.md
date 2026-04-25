# HealthGuard — Real-Time Healthcare Fraud Detection using Random Forest

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![SMOTE](https://img.shields.io/badge/Imbalanced--learn-SMOTE-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A multi-model healthcare fraud detection system with **three independent Random Forest classifiers** — one per fraud type — each with domain-specific feature engineering, SMOTE-based class balancing, and 5-Fold Stratified Cross-Validation. Deployed as a Flask web application for real-time CSV-based fraud screening.

---

## 🎯 Key Features

- **3 independent fraud classifiers** — Phantom Billing, Ghost Patients, Wrong Diagnoses
- **Domain-specific feature engineering** per fraud type (billing anomalies, date inconsistencies, diagnosis-gender conflicts)
- **SMOTE oversampling** for handling severe class imbalance in Ghost Patient and Wrong Diagnoses datasets
- **5-Fold Stratified Cross-Validation** on all three models
- **Modular preprocessing pipeline** — separate preprocessor per fraud type
- **Flask web interface** — upload CSV, select fraud type, get instant prediction table

---

## 🏗️ System Architecture

```
User Uploads CSV + Selects Fraud Type
              │
              ▼
┌─────────────────────────────────────┐
│      Domain-Specific Preprocessor  │
│  (feature engineering per type)    │  ← Billing Per Day, Same Day
└─────────────────────────────────────┘    Discharge, Unrealistic Age,
              │                            Invalid Diagnosis Combo, etc.
              ▼
┌─────────────────────────────────────┐
│        StandardScaler               │  ← Normalizes feature values
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     Random Forest Classifier        │  ← Trained model per fraud type
│   (fraud_detector.pkl per type)     │    class_weight='balanced'
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Label Decoder                  │  ← LabelEncoder inverse transform
└─────────────────────────────────────┘
              │
              ▼
     Prediction Table (HTML)
  Row-level: Fraud / No Fraud verdict
```

---

## 🧠 Fraud Types & Model Details

### 1. Phantom Billing
Detects claims billed for services never rendered.

**Key engineered features:**
- `Billing Per Day` — Amount Billed / Length of Stay
- `Same Day Discharge` — binary flag for 0-day stays
- `High Billed` — billing above ₹2,50,000 threshold
- `Invalid Diagnosis Combo` — pregnancy diagnosis for male patients

**Model config:** RandomForest (n_estimators=100, class_weight='balanced')

---

### 2. Ghost Patients
Detects claims filed for non-existent or fabricated patients.

**Key engineered features:**
- `Same Day Discharge` — 0-day hospital stay flag
- `No Dates Present` — missing admission/discharge dates
- `Unrealistic Age` — age < 1 or > 110
- `Low Billing But Long Stay` — billing < ₹10,000 with stay > 5 days

**Model config:** RandomForest (n_estimators=150, max_depth=10, SMOTE applied)

---

### 3. Wrong Diagnoses
Detects fraudulent or incorrect diagnosis coding for inflated reimbursements.

**Key engineered features:**
- `Billing Per Day` — rate-based anomaly detection
- `Length of Stay` — unusually short or long relative to diagnosis
- `Gender` encoded numerically for cross-feature interaction

**Model config:** RandomForest (n_estimators=300, max_depth=None, SMOTE applied)

---

## 📊 Training Pipeline

For each fraud type:

```
1. Load dataset (CSV)
2. Filter relevant labels: ["No Fraud", <fraud_type>]
3. Domain-specific feature engineering
4. LabelEncoder + StandardScaler
5. 5-Fold Stratified Cross-Validation (F1 scoring)
6. Train-Test Split (80/20, stratified)
7. SMOTE oversampling on training set (Ghost Patient & Wrong Diagnoses)
8. Train final RandomForest model
9. Save: fraud_detector.pkl, scaler.pkl, label_encoder.pkl
10. Generate confusion matrix
```

---

## 📁 Project Structure

```
HealthGuard/
│
├── real_time_healthcare_fraud_detection/
│   ├── app.py                              # Flask app + prediction pipeline
│   ├── train_models_phantom_billing.py     # Phantom Billing training script
│   ├── train_models_ghost_patient.py       # Ghost Patient training script
│   ├── train_models_wrong_diagnoses.py     # Wrong Diagnoses training script
│   ├── evaluate_model.py                   # Phantom Billing evaluator
│   ├── evaluate_model_ghost_patient.py     # Ghost Patient evaluator
│   ├── evaluate_model_wrong_diagnoses.py   # Wrong Diagnoses evaluator
│   │
│   ├── utils/
│   │   ├── preprocess.py                   # Ghost Patient preprocessor
│   │   ├── preprocess_phantom_billing.py   # Phantom Billing preprocessor
│   │   ├── preprocess_wrong_diagnoses.py   # Wrong Diagnoses preprocessor
│   │   └── preprocess_loader.py            # Dynamic preprocessor router
│   │
│   ├── dataset/
│   │   ├── phantom_billing.csv
│   │   ├── ghost_patient.csv
│   │   ├── wrong_diagnoses.csv
│   │   └── (train/test splits per type)
│   │
│   ├── models/                             # Saved .pkl files (generated after training)
│   │   ├── phantom_billing/
│   │   ├── ghost_patient/
│   │   └── wrong_diagnoses/
│   │
│   ├── static/style.css
│   └── templates/
│       ├── index.html
│       └── results.html
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AmudhanManimaran/HealthGuard.git
cd HealthGuard/real_time_healthcare_fraud_detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
Run each training script to generate `.pkl` model files:
```bash
python train_models_phantom_billing.py
python train_models_ghost_patient.py
python train_models_wrong_diagnoses.py
```
This creates the `models/` directory with trained classifiers, scalers, and label encoders.

### 5. Run the Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 🚀 Usage

1. Open `http://localhost:5000`
2. Upload a **CSV file** with patient/billing records
3. Select the **fraud type** to screen for: Phantom Billing, Ghost Patients, or Wrong Diagnoses
4. Click **Submit**
5. Review the prediction table — each row is labeled **Fraud** or **No Fraud**

---

## 📦 Requirements

```
Flask
pandas
scikit-learn
imbalanced-learn
joblib
matplotlib
seaborn
numpy
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Amudhan Manimaran**
- 🌐 Portfolio: [amudhanmanimaran.github.io/Portfolio](https://amudhanmanimaran.github.io/Portfolio/)
- 💼 LinkedIn: [linkedin.com/in/amudhan-manimaran-3621bb32a](https://www.linkedin.com/in/amudhan-manimaran-3621bb32a)
- 🐙 GitHub: [github.com/AmudhanManimaran](https://github.com/AmudhanManimaran)
