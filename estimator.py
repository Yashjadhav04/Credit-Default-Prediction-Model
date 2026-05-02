#!/usr/bin/env python3
"""
OPTIMIZED Estimator for Horse Race Competition (v3.0)
Uses 15 strategically selected financial ratios
Removed 7 non-significant/multicollinear features
Added 3 strategic features (tangible assets, debt maturity, size)
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

# ---------------------------------------------------------------------
# CONFIG - OPTIMIZED FOR MAXIMUM AUC
# ---------------------------------------------------------------------

DATA_PATH = "train_data_full copy.csv"
ARTIFACT_DIR = "artifacts"

TARGET_COLUMN = "default_12m"
AVAIL_COLUMN = "avail_date"
AVAIL_CUTOFF = "2012-05-01"

# OPTIMIZED: 15 features (12 proven + 3 strategic additions)
FEATURE_COLUMNS: List[str] = [
    # === CORE 12 PROVEN FEATURES ===
    "leverage_w",              # z=17.46, p<0.001
    "roe_w",                   # z=17.99, p<0.001
    "roa_w",                   # z=-17.73, p<0.001
    "cash_to_assets_w",        # z=-1.87, p=0.062
    "asset_turnover_w",        # z=4.34, p<0.001
    "interest_coverage_w",     # z=1.75, p=0.081
    "ebitda_margin_w",         # z=-1.07, p=0.286
    "working_capital_ratio_w", # z=-10.65, p<0.001
    "missing_roe_flag",        # z=39.08, p<0.001 ⭐⭐⭐
    "cfroa_w",                 # z=-31.03, p<0.001 ⭐⭐⭐
    "ar_turnover_w",           # z=-7.42, p<0.001
    "fin_debt_ratio_w",        # z=-6.47, p<0.001
    # === STRATEGIC ADDITIONS (3 features) ===
    "tangible_asset_ratio_w",  # Collateral value (critical for secured lending)
    "debt_maturity_ratio_w",   # Refinancing risk (short-term debt crisis indicator)
    "log_assets",              # Size effect (small firms = higher risk)
]

# Import preprocessing
import sys
sys.path.append(os.path.dirname(__file__))
from preprocessing import Preprocessing_Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------
# CALIBRATION CLASSES
# ---------------------------------------------------------------------

class PDModelCalibrator:
    """Integrated calibration for probability predictions"""
    def __init__(self):
        self.calibrator_method = None
        self.calibrator = None
        self.is_fitted = False
    
    def fit_calibration(self, y_true, y_pred_proba, method='isotonic'):
        """Fit calibration on validation data"""
        self.calibrator_method = method
        
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
        elif method == 'sigmoid':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
        else:
            raise ValueError("Method must be 'isotonic' or 'sigmoid'")
        
        self.is_fitted = True
        print(f"✓ Calibration fitted using {method} method")
        
        calibrated_probs = self.transform(y_pred_proba)
        brier_raw = brier_score_loss(y_true, y_pred_proba)
        brier_cal = brier_score_loss(y_true, calibrated_probs)
        
        return {
            'method': method,
            'brier_score_raw': float(brier_raw),
            'brier_score_calibrated': float(brier_cal),
            'brier_improvement': float(brier_raw - brier_cal)
        }
    
    def transform(self, y_pred_proba):
        """Apply calibration transformation"""
        if not self.is_fitted:
            return y_pred_proba
        
        if self.calibrator_method == 'isotonic':
            return self.calibrator.transform(y_pred_proba)
        elif self.calibrator_method == 'sigmoid':
            return self.calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------
# HELPER FUNCTIONS (same as before)
# ---------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[AVAIL_COLUMN] = pd.to_datetime(df[AVAIL_COLUMN])
    return df


def split_train_test(df: pd.DataFrame, cutoff=AVAIL_CUTOFF) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_dt = pd.to_datetime(cutoff)
    train = df[df[AVAIL_COLUMN] < cutoff_dt].copy()
    test = df[df[AVAIL_COLUMN] >= cutoff_dt].copy()
    return train, test


def walk_forward_evaluation(df: pd.DataFrame, window_years=1) -> pd.DataFrame:
    print("Walk-forward evaluation with May-to-May windows:")
    print(f"Using OPTIMIZED {len(FEATURE_COLUMNS)} features")
    print(f"Features: {FEATURE_COLUMNS}")
    
    df = df.copy()
    df = df.sort_values(by=AVAIL_COLUMN)
    min_date = df[AVAIL_COLUMN].min()
    max_date = df[AVAIL_COLUMN].max()
    
    print(f"Min avail_date in train: {min_date}")
    print(f"Global cutoff (train/test split): {AVAIL_CUTOFF}")
    
    start_year = min_date.year
    end_year = max_date.year
    
    results = []
    
    for year in range(start_year, end_year):
        train_end = pd.Timestamp(year=year, month=5, day=1)
        val_start = train_end
        val_end = train_end + pd.DateOffset(years=window_years)
        
        train_fold = df[df[AVAIL_COLUMN] < train_end].copy()
        val_fold = df[(df[AVAIL_COLUMN] >= val_start) & (df[AVAIL_COLUMN] < val_end)].copy()
        
        if len(train_fold) < 1000 or len(val_fold) < 100:
            continue
        
        print(f"\n=== Fold for year {year}: ===")
        print(f"  Train: avail < {train_end.date()} (n={len(train_fold)})")
        print(f"  Val:   {val_start.date()} <= avail < {val_end.date()} (n={len(val_fold)})")
        
        # Preprocessing
        train_fold, details = Preprocessing_Pipeline.train_preprocessing(train_fold)
        val_fold = Preprocessing_Pipeline.transform_preprocessing(val_fold, details)
        
        X_tr = train_fold[FEATURE_COLUMNS]
        y_tr = train_fold[TARGET_COLUMN]
        X_val = val_fold[FEATURE_COLUMNS]
        y_val = val_fold[TARGET_COLUMN]
        
        # Train logistic regression
        X_tr_sm = sm.add_constant(X_tr, has_constant='add')
        logit_model = sm.Logit(y_tr, X_tr_sm)
        fit = logit_model.fit(disp=0, maxiter=100)
        
        # Predict
        X_val_sm = sm.add_constant(X_val, has_constant='add')
        probs = fit.predict(X_val_sm)
        
        auc = roc_auc_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        
        print(f"  AUC={auc:.4f}, Brier={brier:.4f}")
        
        results.append({
            'train_end': train_end.date(),
            'val_start': val_start.date(),
            'val_end': val_end.date(),
            'n_train': len(train_fold),
            'n_val': len(val_fold),
            'auc': auc,
            'brier': brier
        })
    
    res_df = pd.DataFrame(results)
    print("\nWalk-forward summary:")
    print(res_df)
    print(f"\nOptimized Model Performance:")
    print(f"Mean AUC: {res_df['auc'].mean():.4f}")
    print(f"Mean Brier: {res_df['brier'].mean():.4f}")
    
    return res_df


def train_final_model(df: pd.DataFrame):
    print(f"\nTraining OPTIMIZED final model with {len(FEATURE_COLUMNS)} features...")
    print(f"Features: {FEATURE_COLUMNS}")
    
    df_prep, details = Preprocessing_Pipeline.train_preprocessing(df)
    X = df_prep[FEATURE_COLUMNS]
    y = df_prep[TARGET_COLUMN]
    
    X_sm = sm.add_constant(X, has_constant='add')
    logit_model = sm.Logit(y, X_sm)
    fit = logit_model.fit(disp=0, maxiter=100)
    
    probs_train = fit.predict(X_sm)
    auc_train = roc_auc_score(y, probs_train)
    brier_train = brier_score_loss(y, probs_train)
    gini = 2 * auc_train - 1
    
    print(f"\nFinal Training Performance:")
    print(f"  Training AUC: {auc_train:.4f}")
    print(f"  Training Brier: {brier_train:.4f}")
    print(f"  Training Gini: {gini:.4f}")
    
    print("\n===== OPTIMIZED FINAL MODEL SUMMARY (statsmodels Logit) =====")
    print(fit.summary())
    
    print("\n===== FEATURE COEFFICIENTS =====")
    for feat, coef in zip(FEATURE_COLUMNS, fit.params[1:]):
        print(f"{feat:25s}: {coef:8.4f}")
    
    return fit, details, FEATURE_COLUMNS


def train_and_save_calibrator(df: pd.DataFrame, model, preproc_details):
    print("\n" + "="*70)
    print("TRAINING PROBABILITY CALIBRATOR")
    print("="*70)
    
    # Use 43% of training data for calibration
    df_prep = Preprocessing_Pipeline.transform_preprocessing(df.copy(), preproc_details)
    X = df_prep[FEATURE_COLUMNS]
    y = df_prep[TARGET_COLUMN]
    
    X_sm = sm.add_constant(X, has_constant='add')
    probs_raw = model.predict(X_sm)
    
    n_cal = int(len(df) * 0.43)
    indices = np.random.RandomState(42).choice(len(df), size=n_cal, replace=False)
    y_cal = y.iloc[indices]
    probs_cal = probs_raw.iloc[indices]
    
    print(f"Calibration set size: {len(y_cal):,} records")
    
    calibrator = PDModelCalibrator()
    cal_metrics = calibrator.fit_calibration(y_cal, probs_cal, method='isotonic')
    
    print(f"Brier Score - Raw: {cal_metrics['brier_score_raw']:.6f}")
    print(f"Brier Score - Calibrated: {cal_metrics['brier_score_calibrated']:.6f}")
    print(f"Brier Improvement: {cal_metrics['brier_improvement']:.6f}")
    
    return calibrator, cal_metrics


def save_artifacts(model, preproc_details: dict, feature_columns: List[str], train_df: pd.DataFrame, 
                   calibrator=None, cal_metrics=None):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # Save calibrator
    if calibrator:
        cal_path = os.path.join(ARTIFACT_DIR, "calibrator.joblib")
        joblib.dump(calibrator, cal_path)
        print(f"Saved calibrator to     {cal_path}")
    
    # Save model
    model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved OPTIMIZED model to   {model_path}")
    
    # Save metadata
    meta = {
        "model_version": "3.0_optimized",
        "feature_columns": feature_columns,
        "preprocessing_details": preproc_details,
        "target_column": TARGET_COLUMN,
        "avail_column": AVAIL_COLUMN,
        "avail_cutoff": AVAIL_CUTOFF,
        "training_records": len(train_df),
        "calibration_metrics": cal_metrics
    }
    
    meta_path = os.path.join(ARTIFACT_DIR, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def main():
    print("="*70)
    print("OPTIMIZED PD MODEL FOR HORSE RACE COMPETITION (v3.0)")
    print("="*70)
    print(f"Using {len(FEATURE_COLUMNS)} strategically selected features:")
    print("✅ Removed: 7 non-significant/multicollinear features")
    print("✅ Added: 3 strategic features (tangible assets, debt maturity, size)")
    for i, feat in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i}. {feat}")
    print("="*70)
    
    print("\nLoading data...")
    df = load_data(DATA_PATH)
    print(f"Total rows with non-missing {TARGET_COLUMN}: {len(df):,}")

    train_df, test_df = split_train_test(df)
    print(f"Train rows (avail_date < {AVAIL_CUTOFF}): {len(train_df):,}")

    # 1) Walk-forward evaluation
    wf_results = walk_forward_evaluation(train_df)

    # Save walk-forward metrics
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    wf_path = os.path.join(ARTIFACT_DIR, "walk_forward_results.csv")
    wf_results.to_csv(wf_path, index=False)
    print(f"\nSaved walk-forward results to {wf_path}")

    # 2) Train FINAL model
    model, preproc_details, feature_columns = train_final_model(train_df)

    # 3) Train calibrator
    calibrator, cal_metrics = train_and_save_calibrator(train_df, model, preproc_details)

    # 4) Save artifacts
    save_artifacts(model, preproc_details, feature_columns, train_df, calibrator, cal_metrics)

    print("\n" + "="*70)
    print("OPTIMIZED MODEL TRAINING COMPLETE (v3.0)")
    print("="*70)
    print("Key improvements:")
    print("  ✓ Removed 7 non-significant/multicollinear features")
    print("  ✓ Added 3 strategic features (tangible assets, debt maturity, size)")
    print("  ✓ 15 features total (12 proven + 3 strategic)")
    print("  ✓ Cleaner model, better interpretability")
    print("  ✓ Probability calibration included")
    print("="*70)
    print("\nUse predictor_optimized.py + harness_optimized.py to generate predictions")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
