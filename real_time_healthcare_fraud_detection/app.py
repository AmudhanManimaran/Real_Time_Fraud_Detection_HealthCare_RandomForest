# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from utils.preprocess_loader import get_preprocessor

app = Flask(__name__)

# Only supervised models
FRAUD_TYPES = {
    "phantom_billing": "Phantom Billing",
    "ghost_patient": "Ghost Patients",
    "wrong_diagnoses": "Wrong Diagnoses"
}

@app.route("/", methods=["GET", "POST"])
def index():
    result_table = None
    error_message = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        selected_model = request.form.get("fraud_type")

        if file and file.filename.endswith(".csv") and selected_model:
            try:
                df = pd.read_csv(file)
                df_display = df.copy()

                # Load model, scaler, and label encoder
                model_dir = os.path.join("models", selected_model)
                model = joblib.load(os.path.join(model_dir, "fraud_detector.pkl"))
                scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
                label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

                # Preprocess features
                preprocess_data = get_preprocessor(selected_model)
                X_scaled = preprocess_data(df, is_training=False)
                X_scaled = scaler.transform(X_scaled)

                # Make predictions
                preds = model.predict(X_scaled)
                labels = label_encoder.inverse_transform(preds)
                df_display["Prediction"] = labels

                # Add length of stay
                df_display["Length of Stay"] = (
                    pd.to_datetime(df_display["Date Discharged"]) -
                    pd.to_datetime(df_display["Date Admitted"])
                ).dt.days

                # Add Yes/No column
                yes_label = FRAUD_TYPES[selected_model]
                df_display["Is " + yes_label + "?"] = df_display["Prediction"].apply(
                    lambda x: "Yes" if x == yes_label else "No"
                )

                # Convert to HTML table
                result_table = df_display.to_html(classes="table table-bordered", index=False)

            except Exception as e:
                error_message = f"❌ Error processing file: {str(e)}"
        else:
            error_message = "⚠️ Please upload a valid CSV file and select a fraud type."

    return render_template("index.html", table=result_table, error=error_message, fraud_types=FRAUD_TYPES)

if __name__ == "__main__":
    app.run(debug=True)
