import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from utils.preprocess import preprocess_data

# === CONSTANTS ===
FRAUD_TYPE = "Ghost Patients"
DATASET_PATH = "dataset/ghost_patient.csv"
MODEL_PATH = "models/ghost_patient"
TRAIN_PATH = "dataset/ghost_patient_train.csv"
TEST_PATH = "dataset/ghost_patient_test.csv"
LABEL_COL = "Fraud Type"

# === CREATE MODEL DIR ===
os.makedirs(MODEL_PATH, exist_ok=True)

print(f"\n🔄 Training model for: {FRAUD_TYPE}")

try:
    # === LOAD DATASET ===
    df = pd.read_csv(DATASET_PATH)

    # === FILTER FOR RELEVANT LABELS ONLY ===
    df = df[df[LABEL_COL].isin(["No Fraud", FRAUD_TYPE])].dropna(subset=[LABEL_COL])
    df_raw = df.copy()

    # === PREPROCESS FEATURES ===
    X, y = preprocess_data(df, label_column=LABEL_COL, is_training=True)

    # === ENCODING AND SCALING ===
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === CROSS VALIDATION ===
    print("\n🔍 Performing 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    temp_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    cv_scores = cross_val_score(temp_model, X_scaled, y_encoded, cv=skf, scoring='f1')
    print("📊 F1 scores from 5 folds:", cv_scores)
    print("📈 Average F1 score:", np.mean(cv_scores))

    # === TRAIN-TEST SPLIT ===
    X_train, X_test, y_train, y_test, df_train_raw, df_test_raw = train_test_split(
        X_scaled, y_encoded, df_raw, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # === SMOTE TO BALANCE TRAINING DATA ===
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # === RESTORE LABELS TO RAW DF ===
    df_train_raw[LABEL_COL] = label_encoder.inverse_transform(y_train[:len(df_train_raw)])
    df_test_raw[LABEL_COL] = label_encoder.inverse_transform(y_test)

    # === SAVE SPLITS ===
    df_train_raw.to_csv(TRAIN_PATH, index=False)
    df_test_raw.to_csv(TEST_PATH, index=False)

    # === FINAL TRAINING ===
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # === SAVE MODEL + TOOLS ===
    joblib.dump(model, os.path.join(MODEL_PATH, "fraud_detector.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

    print(f"\n✅ Model and encoders saved in: {MODEL_PATH}")
    print(f"📊 Classes: {label_encoder.classes_}")
    print(f"📁 Training data saved to: {TRAIN_PATH}")
    print(f"📁 Testing data saved to: {TEST_PATH}")

    # === CONFUSION MATRIX ===
    print("\n📈 Generating confusion matrix...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Oranges", values_format='d')
    plt.title(f"Confusion Matrix - {FRAUD_TYPE}")
    plt.show()

except Exception as e:
    print(f"\n❌ Error during training: {e}")
