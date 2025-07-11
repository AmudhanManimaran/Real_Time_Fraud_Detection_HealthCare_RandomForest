import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
from utils.preprocess_wrong_diagnoses import preprocess_data  # Custom preprocessing

# === CONFIG ===
FRAUD_TYPE = "Wrong Diagnoses"
LABEL_COL = "Fraud Type"
DATASET_PATH = "dataset/wrong_diagnoses.csv"
TRAIN_PATH = "dataset/wrong_diagnoses_train.csv"
TEST_PATH = "dataset/wrong_diagnoses_test.csv"
MODEL_PATH = "models/wrong_diagnoses"

# === SETUP ===
os.makedirs(MODEL_PATH, exist_ok=True)
print(f"\n🔄 Training model for: {FRAUD_TYPE}")

try:
    # === LOAD DATA ===
    df = pd.read_csv(DATASET_PATH)
    df = df[df[LABEL_COL].isin(["No Fraud", FRAUD_TYPE])].dropna(subset=[LABEL_COL])
    df_raw = df.copy()

    # === PREPROCESS FEATURES ===
    X, y = preprocess_data(df, label_column=LABEL_COL, is_training=True)

    # === ENCODE LABELS ===
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # === SCALE FEATURES ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === CROSS VALIDATION ===
    print("\n🔍 5-Fold Cross-Validation (F1 Score)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_temp = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    )
    cv_scores = cross_val_score(rf_temp, X_scaled, y_encoded, cv=skf, scoring="f1")
    print("📊 F1 scores:", np.round(cv_scores, 4))
    print("📈 Avg F1 score:", np.round(np.mean(cv_scores), 4))

    # === TRAIN-TEST SPLIT ===
    X_train, X_test, y_train, y_test, df_train_raw, df_test_raw = train_test_split(
        X_scaled, y_encoded, df_raw, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # === APPLY SMOTE ===
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ✅ Save RAW data before SMOTE
    df_train_raw.to_csv(TRAIN_PATH, index=False)
    df_test_raw.to_csv(TEST_PATH, index=False)

    # === TRAIN FINAL MODEL ===
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # === EVALUATE ===
    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    # === CONFUSION MATRIX ===
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Purples", values_format="d")
    plt.title(f"Confusion Matrix – {FRAUD_TYPE}")
    plt.show()

    # === SAVE COMPONENTS ===
    joblib.dump(model, os.path.join(MODEL_PATH, "fraud_detector.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))

    print(f"\n✅ Model and encoders saved to: {MODEL_PATH}")
    print(f"📁 Training set saved to: {TRAIN_PATH}")
    print(f"📁 Testing set saved to: {TEST_PATH}")
    print(f"📊 Label Classes: {label_encoder.classes_}")

except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
