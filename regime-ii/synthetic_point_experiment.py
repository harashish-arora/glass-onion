#!/usr/bin/env python3
"""
Synthetic Point Interpolation Experiment

For train data pairs:
1. Plot experimental data points
2. Fit Apelblat equation
3. Generate ONE synthetic temperature (midpoint)
4. Predict with model for that synthetic point
5. Mark both Apelblat expected value and model prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

from apelblat_analysis import (generate_features_for_prediction, 
                                fit_apelblat, apelblat_equation)

# Config
TRAIN_FILE = 'data/train.csv'
MODEL_PATH = 'model/model.joblib'
SELECTOR_PATH = 'model/selector.joblib'
PLOTS_DIR = 'plots'

MIN_TEMP_POINTS = 7
MIN_APELBLAT_R2 = 0.8
MAX_MODEL_RMSE = 0.5


def run_experiment():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("="*60)
    print("SYNTHETIC POINT INTERPOLATION EXPERIMENT")
    print("="*60)
    
    # Load
    print("\nLoading train data and model...")
    df = pd.read_csv(TRAIN_FILE)
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    
    print(f"Train samples: {len(df)}")
    
    # Group by pair
    df['pair_id'] = df['Solute'] + '||' + df['Solvent']
    pairs = df.groupby('pair_id')
    print(f"Total pairs: {len(pairs)}")
    
    # Process pairs
    count = 0
    for pair_id, pair_df in pairs:
        # Filter: minimum temperature points
        if len(pair_df) < MIN_TEMP_POINTS:
            continue
        
        # Sort by temperature
        pair_df = pair_df.sort_values('Temperature').reset_index(drop=True)
        
        T = pair_df['Temperature'].values
        logS = pair_df['LogS'].values
        ln_x = logS * np.log(10)
        
        # Fit Apelblat
        A, B, C, apelblat_r2 = fit_apelblat(T, ln_x)
        if A is None or apelblat_r2 < MIN_APELBLAT_R2:
            continue
        
        # Check model RMSE on actual data
        X = generate_features_for_prediction(pair_df)
        X = selector.transform(X)
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(logS, preds))
        
        if rmse > MAX_MODEL_RMSE:
            continue
        
        # ============================================
        # GENERATE SYNTHETIC POINT
        # ============================================
        # Pick a temperature in the middle (between 2 existing points)
        mid_idx = len(T) // 2
        T_synthetic = (T[mid_idx] + T[mid_idx + 1]) / 2  # Midpoint
        
        # What Apelblat says the LogS should be at this temp
        ln_x_apelblat = apelblat_equation(T_synthetic, A, B, C)
        logS_apelblat = ln_x_apelblat / np.log(10)
        
        # What the model predicts at this temp
        synth_df = pd.DataFrame({
            'Solute': [pair_df['Solute'].iloc[0]],
            'Solvent': [pair_df['Solvent'].iloc[0]],
            'Temperature': [T_synthetic]
        })
        X_synth = generate_features_for_prediction(synth_df)
        X_synth = selector.transform(X_synth)
        logS_model = model.predict(X_synth)[0]
        
        # ============================================
        # PLOT
        # ============================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apelblat curve (for reference)
        T_curve = np.linspace(T.min() - 2, T.max() + 2, 100)
        logS_curve = apelblat_equation(T_curve, A, B, C) / np.log(10)
        ax.plot(T_curve, logS_curve, 'r--', linewidth=2, label='Apelblat Fit', zorder=1)
        
        # Experimental data points
        ax.scatter(T, logS, s=80, c='blue', edgecolors='black', 
                   label='Train Data (actual)', zorder=3)
        
        # SYNTHETIC POINT - Apelblat expectation
        ax.scatter(T_synthetic, logS_apelblat, s=200, c='red', marker='*',
                   edgecolors='black', linewidth=1.5,
                   label=f'Apelblat Expected: {logS_apelblat:.3f}', zorder=5)
        
        # SYNTHETIC POINT - Model prediction
        ax.scatter(T_synthetic, logS_model, s=200, c='lime', marker='D',
                   edgecolors='black', linewidth=1.5,
                   label=f'Model Prediction: {logS_model:.3f}', zorder=5)
        
        # Draw line connecting the two predictions
        ax.plot([T_synthetic, T_synthetic], [logS_apelblat, logS_model], 
                'k-', linewidth=2, alpha=0.7)
        
        # Annotate the difference
        diff = logS_model - logS_apelblat
        ax.annotate(f'Δ = {diff:+.3f}', 
                    xy=(T_synthetic + 1, (logS_apelblat + logS_model) / 2),
                    fontsize=12, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('LogS (log₁₀ mol/L)', fontsize=12)
        ax.set_title(f'TRAIN Pair {count}: Synthetic Point @ T={T_synthetic:.1f}K\n'
                     f'Apelblat R²={apelblat_r2:.4f}, Model RMSE={rmse:.4f}', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'synthetic_train_{count}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Pair {count}: T_synth={T_synthetic:.1f}K, "
              f"Apelblat={logS_apelblat:.3f}, Model={logS_model:.3f}, Δ={diff:+.3f}")
        
        count += 1
        if count >= 5:
            break
    
    print(f"\nSaved {count} plots to {PLOTS_DIR}/synthetic_train_*.png")
    print("\nDone!")


if __name__ == "__main__":
    run_experiment()
