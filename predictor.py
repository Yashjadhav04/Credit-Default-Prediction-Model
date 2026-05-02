# predictor.py
"""
Helper module for scoring PDs from the trained statsmodels Logit model.
Uses 10 enhanced financial ratios optimized for horse race competition.
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib

from preprocessing import Preprocessing_Pipeline

ARTIFACT_DIR = "artifacts"
MODEL_FILE = "model.joblib"
META_FILE = "meta.json"


def load_artifacts():
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILE)
    meta_path = os.path.join(ARTIFACT_DIR, META_FILE)

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta


def predict_pd(df_raw: pd.DataFrame, use_calibration=False):
    """
    Generate PD predictions using the enhanced 10-feature model.
    
    Parameters:
    - df_raw: raw input dataframe
    - use_calibration: if True, apply probability calibration (if available)
    """
    model, meta = load_artifacts()
    preproc_details = meta["preprocessing_details"]
    feature_columns = meta["feature_columns"]

    print(f"Using ENHANCED model with {len(feature_columns)} features:")
    for feat in feature_columns:
        print(f"  - {feat}")

    # Transform only (no fitting)
    df_proc = Preprocessing_Pipeline.transform_preprocessing(df_raw, preproc_details)

    # Build design matrix
    X = df_proc[feature_columns]
    X_df = pd.DataFrame(X, columns=feature_columns)
    X_df = sm.add_constant(X_df, has_constant="add")

    # Predict PDs
    pd_hat = model.predict(X_df)
    pd_hat = np.asarray(pd_hat)
    
    # Apply calibration if requested and available
    if use_calibration:
        calibrator_path = os.path.join(ARTIFACT_DIR, "calibrator.joblib")
        if os.path.exists(calibrator_path):
            # Load the isotonic regressor directly
            isotonic_regressor = joblib.load(calibrator_path)
            pd_hat = isotonic_regressor.transform(pd_hat)
            print("Applied probability calibration (isotonic)")
        else:
            print("⚠️  Calibration requested but calibrator not found, using raw predictions")
    
    return pd_hat
