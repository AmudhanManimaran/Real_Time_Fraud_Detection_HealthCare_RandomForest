import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(df, is_training=False, label_column=None, return_meta=False, visualize=False):
    df = df.copy()

    # === Standard Cleaning ===
    df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).fillna(2)

    df["Date Admitted"] = pd.to_datetime(df["Date Admitted"], errors="coerce")
    df["Date Discharged"] = pd.to_datetime(df["Date Discharged"], errors="coerce")
    df["Length of Stay"] = (df["Date Discharged"] - df["Date Admitted"]).dt.days.fillna(0).clip(lower=0)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)
    df["Amount Billed"] = pd.to_numeric(df["Amount Billed"], errors="coerce").fillna(0)
    df["Billing Per Day"] = (df["Amount Billed"] / df["Length of Stay"].replace(0, 1)).fillna(0)

    # === Wrong Diagnosis Specific Features ===

    # Flag: Diagnosis does not match gender (e.g., Pregnancy for Male)
    df["Diagnosis Mismatch Gender"] = ((df["Diagnosis"].str.lower().str.contains("pregnancy")) & (df["Gender"] == 0)).astype(int)

    # Flag: Diagnosis unrealistic for age (e.g., Cataract in very young)
    df["Unrealistic Diagnosis Age"] = (
        ((df["Diagnosis"].str.lower().str.contains("cataract")) & (df["Age"] < 10)) |
        ((df["Diagnosis"].str.lower().str.contains("arthritis")) & (df["Age"] < 15)) |
        ((df["Diagnosis"].str.lower().str.contains("dementia")) & (df["Age"] < 30))
    ).astype(int)

    # Flag: High cost for low-complexity diagnosis
    df["High Bill Mild Diagnosis"] = (
        ((df["Diagnosis"].str.lower().str.contains("fever|cold|flu|routine")) & (df["Amount Billed"] > 50000))
    ).astype(int)

    # Flag: Long hospital stay for minor illnesses
    df["Long Stay Minor Issue"] = (
        ((df["Diagnosis"].str.lower().str.contains("flu|cold|routine|migraine")) & (df["Length of Stay"] > 5))
    ).astype(int)

    # Flag: Diagnosis code is rare (proxy for suspicious repetition)
    rare_diagnoses = df["Diagnosis"].value_counts()[df["Diagnosis"].value_counts() < 5].index
    df["Rare Diagnosis"] = df["Diagnosis"].isin(rare_diagnoses).astype(int)

    # === Optional Visualizations ===
    if visualize and "Fraud Type" in df.columns:
        print("📊 Plotting feature distributions by Fraud Type...")
        os.makedirs("plots", exist_ok=True)
        for feature in ["Amount Billed", "Length of Stay", "Billing Per Day"]:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df, x=feature, hue="Fraud Type", bins=40, kde=True, palette="coolwarm", element="step")
            plt.title(f"{feature} Distribution by Fraud Type")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plot_path = f"plots/{feature.replace(' ', '_').lower()}_wrongdiag_by_fraud_type.png"
            plt.savefig(plot_path)
            print(f"✅ Saved plot: {plot_path}")
            plt.close()

    # === Final Features ===
    X = df[[
        "Age",
        "Gender",
        "Amount Billed",
        "Length of Stay",
        "Billing Per Day",
        "Diagnosis Mismatch Gender",
        "Unrealistic Diagnosis Age",
        "High Bill Mild Diagnosis",
        "Long Stay Minor Issue",
        "Rare Diagnosis"
    ]]

    # === Metadata Output ===
    if return_meta:
        meta = df[["Patient ID", "Diagnosis", "Date Admitted", "Date Discharged", "Amount Billed"]].copy()
        if is_training:
            y = df[label_column].astype(str)
            return X, y, meta
        return X, meta

    # === Training Output ===
    if is_training:
        if label_column is None or label_column not in df.columns:
            raise ValueError("Training requires a valid label column.")
        y = df[label_column].astype(str)
        return X, y

    return X
