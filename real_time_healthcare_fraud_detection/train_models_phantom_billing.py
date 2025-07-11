import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from utils.preprocess_phantom_billing import preprocess_data

# === Constants ===
FRAUD_TYPE = "Phantom Billing"
DATASET_PATH = "dataset/phantom_billing.csv"
MODEL_PATH = "models/phantom_billing"
TRAIN_PATH = "dataset/phantom_billing_train.csv"
TEST_PATH = "dataset/phantom_billing_test.csv"
LABEL_COL = "Fraud Type"

# === Create folders if not present ===
os.makedirs(MODEL_PATH, exist_ok=True)

print(f"\n🔄 Training model for: {FRAUD_TYPE}")

try:
    # === Load and Filter Data ===
    df = pd.read_csv(DATASET_PATH)
    df = df[df[LABEL_COL].isin(["No Fraud", FRAUD_TYPE])].dropna(subset=[LABEL_COL])
    df_raw = df.copy()

    # === Preprocess ===
    X, y = preprocess_data(df, label_column=LABEL_COL, is_training=True)

    # === Encode Labels ===
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # === Scale Features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Cross-Validation ===
    print("\n🔎 Performing 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    temp_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    cv_scores = cross_val_score(temp_model, X_scaled, y_encoded, cv=skf, scoring='f1')
    print("📊 F1 scores from 5 folds:", cv_scores)
    print("📈 Average F1 score:", np.mean(cv_scores))

    # === Train-Test Split ===
    X_train, X_test, y_train, y_test, df_train_raw, df_test_raw = train_test_split(
        X_scaled, y_encoded, df_raw, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # === Save raw train/test CSVs ===
    df_train_raw[LABEL_COL] = label_encoder.inverse_transform(y_train)
    df_test_raw[LABEL_COL] = label_encoder.inverse_transform(y_test)
    df_train_raw.to_csv(TRAIN_PATH, index=False)
    df_test_raw.to_csv(TEST_PATH, index=False)

    # === Train Final Model ===
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # === Save Model + Tools ===
    joblib.dump(model, os.path.join(MODEL_PATH, "fraud_detector.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

    print(f"\n✅ Model and encoders saved in: {MODEL_PATH}")
    print(f"📊 Classes: {label_encoder.classes_}")
    print(f"📁 Training data saved to: {TRAIN_PATH}")
    print(f"📁 Testing data saved to: {TEST_PATH}")

    # === Confusion Matrix ===
    print("\n📈 Generating confusion matrix...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix - {FRAUD_TYPE}")
    plt.show()

except Exception as e:
    print(f"\n❌ Error during model training: {e}")
