# utils/preprocess_loader.py

def get_preprocessor(fraud_type):
    if fraud_type == "phantom_billing":
        from utils.preprocess_phantom_billing import preprocess_data
    elif fraud_type == "ghost_patient":
        from utils.preprocess import preprocess_data
    elif fraud_type == "wrong_diagnoses":
        from utils.preprocess_wrong_diagnoses import preprocess_data
    else:
        raise ValueError(f"No preprocessor found for fraud type: {fraud_type}")
    
    return preprocess_data
