#!/usr/bin/env python3
"""
ENHANCED Harness script for Horse Race Competition PD model.
Uses 10 high-impact financial ratios optimized for AUC/ROC performance.

Usage:
    python3 harness.py --input_csv <input file> --output_csv <output file>
    python3 harness.py --input_csv <input file> --output_csv <output file> --calibrated

Produces:
    A CSV with a single column of PDs (no header).
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

from predictor import predict_pd


def main():
    """
    Main function for generating PD predictions on test data.
    """
    parser = argparse.ArgumentParser(description='ENHANCED PD Model Harness - Horse Race Competition')
    parser.add_argument("--input_csv", required=True, 
                        help='Path to input CSV file')
    parser.add_argument("--output_csv", required=True,
                        help='Path to output CSV file for predictions')
    parser.add_argument("--calibrated", action="store_true",
                        help='Use calibrated predictions (improves probability reliability)')
    args = parser.parse_args()

    print("ENHANCED PD MODEL HARNESS - HORSE RACE COMPETITION")
    print("Optimized for AUC/ROC performance with comprehensive financial ratios")
    print("Uses 10 high-impact features from credit risk cheatsheets")
    if args.calibrated:
        print("Calibrated predictions enabled (improved probability reliability)")
    print(f"Input:  {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Calibration: {'ENABLED' if args.calibrated else 'DISABLED (use --calibrated to enable)'}")
    
    try:
        # Validate input file exists
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input file not found: {args.input_csv}")
        
        # Load input data
        print(f"\nLoading input data...")
        df = pd.read_csv(args.input_csv, low_memory=False)
        print(f"Records loaded: {len(df):,}")
        
        # Check for required artifacts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(script_dir, 'artifacts')
        model_path = os.path.join(artifacts_dir, 'model.joblib')
        meta_path = os.path.join(artifacts_dir, 'meta.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")
        
        print(f"Model artifacts found in: {artifacts_dir}")
        
        # Generate predictions (with or without calibration)
        print("\nGenerating predictions with ENHANCED model...")
        pd_hat = predict_pd(df, use_calibration=args.calibrated)
        
        # Validate predictions
        if pd_hat is None or len(pd_hat) == 0:
            raise ValueError("predict_pd returned empty predictions")
        
        if len(pd_hat) != len(df):
            raise ValueError(f"Prediction length mismatch: {len(pd_hat)} predictions for {len(df)} records")
        
        # Convert to numpy array if needed
        if isinstance(pd_hat, pd.Series):
            pd_hat = pd_hat.values
        
        # Check for invalid values
        if np.any(np.isnan(pd_hat)):
            n_nan = np.sum(np.isnan(pd_hat))
            raise ValueError(f"Found {n_nan} NaN values in predictions")
        
        if np.any((pd_hat < 0) | (pd_hat > 1)):
            n_invalid = np.sum((pd_hat < 0) | (pd_hat > 1))
            raise ValueError(f"Found {n_invalid} predictions outside [0, 1] range")
        
        # Save predictions (single column, no header)
        print(f"\nSaving predictions to: {args.output_csv}")
        np.savetxt(args.output_csv, pd_hat, fmt='%.10f', delimiter=',')
        
        # Summary statistics
        print("\nENHANCED PREDICTION SUMMARY")
        print(f"Total predictions: {len(pd_hat):,}")
        print(f"PD Range:  [{pd_hat.min():.6f}, {pd_hat.max():.6f}]")
        print(f"Mean PD:   {pd_hat.mean():.6f}")
        print(f"Median PD: {np.median(pd_hat):.6f}")
        print(f"Std PD:    {pd_hat.std():.6f}")
        print("HORSE RACE READY: Enhanced model with comprehensive financial ratios")
        if args.calibrated:
            print("Calibrated probabilities for improved reliability")
        print(f"Success: Predictions saved to {args.output_csv}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        print("Please ensure you have run estimator.py to train the model first.")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
