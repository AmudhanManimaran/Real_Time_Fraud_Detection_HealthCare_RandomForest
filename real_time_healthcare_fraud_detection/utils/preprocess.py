import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(df, is_training=False, label_column=None, return_meta=False, visualize=False):
    df = df.copy()

    # === Clean Gender ===
    df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).fillna(2)

    # === Parse Dates and Compute Length of Stay ===
    df["Date Admitted"] = pd.to_datetime(df["Date Admitted"], errors="coerce")
    df["Date Discharged"] = pd.to_datetime(df["Date Discharged"], errors="coerce")
    df["Length of Stay"] = (df["Date Discharged"] - df["Date Admitted"]).dt.days.fillna(0).clip(lower=0)

    # === Clean numeric fields ===
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)
    df["Amount Billed"] = pd.to_numeric(df["Amount Billed"], errors="coerce").fillna(0)
    df["Billing Per Day"] = (df["Amount Billed"] / df["Length of Stay"].replace(0, 1)).fillna(0)

    # === Ghost-Patient-Specific Feature Engineering ===
    df["Same Day Discharge"] = (df["Length of Stay"] == 0).astype(int)
    df["No Dates Present"] = df[["Date Admitted", "Date Discharged"]].isnull().any(axis=1).astype(int)
    df["Unrealistic Age"] = ((df["Age"] < 1) | (df["Age"] > 110)).astype(int)
    df["Low Billing But Long Stay"] = ((df["Amount Billed"] < 10000) & (df["Length of Stay"] > 5)).astype(int)

    # === Optional Visual Analysis ===
    if visualize and "Fraud Type" in df.columns:
        print("📊 Plotting feature distributions by Fraud Type...")
        os.makedirs("plots", exist_ok=True)

        for feature in ["Amount Billed", "Length of Stay", "Billing Per Day"]:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df, x=feature, hue="Fraud Type", bins=50, kde=True, palette="coolwarm", element="step")
            plt.title(f"{feature} Distribution by Fraud Type")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plot_path = f"plots/{feature.replace(' ', '_').lower()}_ghost_by_fraud_type.png"
            plt.savefig(plot_path)
            print(f"✅ Saved plot: {plot_path}")
            plt.close()

    # === Final Feature Set ===
    X = df[[
        "Age",
        "Gender",
        "Amount Billed",
        "Length of Stay",
        "Billing Per Day",
        "Same Day Discharge",
        "No Dates Present",
        "Unrealistic Age",
        "Low Billing But Long Stay"
    ]]

    # === Metadata Output (Optional) ===
    if return_meta:
        meta = df[["Patient ID", "Diagnosis", "Date Admitted", "Date Discharged", "Amount Billed"]].copy()
        if is_training:
            y = df[label_column].astype(str)
            return X, y, meta
        return X, meta

    # === Training Mode Output ===
    if is_training:
        if label_column is None or label_column not in df.columns:
            raise ValueError("Training requires a valid label column.")
        y = df[label_column].astype(str)
        return X, y

    return X
