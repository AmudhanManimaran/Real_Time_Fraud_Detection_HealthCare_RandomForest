import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from utils.preprocess import preprocess_data

# === CONFIG ===
DATA_PATH = "dataset/ghost_patient_test.csv"
MODEL_PATH = "models/ghost_patient/fraud_detector.pkl"
SCALER_PATH = "models/ghost_patient/scaler.pkl"
ENCODER_PATH = "models/ghost_patient/label_encoder.pkl"
LABEL_COL = "Fraud Type"

print("\n🔍 Evaluating model on ghost_patient_test.csv")

try:
    # === LOAD TEST DATA ===
    df = pd.read_csv(DATA_PATH)
    df_original = df.copy()

    # === PREPROCESS FEATURES ===
    X, y, _ = preprocess_data(df, is_training=True, label_column=LABEL_COL, return_meta=True)

    # === LOAD MODEL COMPONENTS ===
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # === TRANSFORM AND PREDICT ===
    X_scaled = scaler.transform(X)
    y_encoded = encoder.transform(y)
    y_pred = model.predict(X_scaled)
    predicted_labels = encoder.inverse_transform(y_pred)

    # === METRICS REPORT ===
    print("\n=== Classification Report ===")
    print(classification_report(y_encoded, y_pred, target_names=encoder.classes_))

    precision = precision_score(y_encoded, y_pred, average='weighted')
    recall = recall_score(y_encoded, y_pred, average='weighted')
    f1 = f1_score(y_encoded, y_pred, average='weighted')

    print(f"\n📊 Precision: {precision:.4f}")
    print(f"📊 Recall:    {recall:.4f}")
    print(f"📊 F1 Score:  {f1:.4f}")

    # === CONFUSION MATRIX ===
    cm = confusion_matrix(y_encoded, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix - Ghost Patients")
    plt.tight_layout()
    plt.show()

    # === CROSS-VALIDATION ON TEST SET ===
    print("\n🔁 Running 5-Fold Cross-Validation on Test Data...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    temp_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    cv_scores = cross_val_score(temp_model, X_scaled, y_encoded, cv=skf, scoring='f1_weighted')
    print("✅ F1 Scores from each fold:", [round(score, 4) for score in cv_scores])
    print("📈 Average F1 Score:", round(cv_scores.mean(), 4))

    # === SAVE FINAL PREDICTIONS ===
    try:
        df_original = df_original.copy()
        if len(df_original) != len(predicted_labels):
            print(f"\n⚠️ Row mismatch: {len(df_original)} rows in original vs {len(predicted_labels)} predictions.")
        else:
            df_original["Fraud Type"] = predicted_labels
            df_original.to_csv("ghost_patient_predictions.csv", index=False)
            print("\n✅ Saved: ghost_patient_predictions.csv")
    except Exception as save_err:
        print(f"\n❌ Failed to save predictions: {save_err}")


except Exception as e:
    print(f"\n❌ Error during evaluation: {e}")
