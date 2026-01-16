# ablations_bulk_features.py
# Ablation study targeting bulk feature groups (Morgan, MACCS, Abraham, AUTOCORR2D, MOSE)

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from featurizer import MoleculeFeaturizer

# Config
SEED = 101
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ============================================================================
# BULK FEATURE GROUP DEFINITIONS
# ============================================================================

BULK_GROUPS = {
    'Morgan': 'Morgan_',       # 1024 features
    'MACCS': 'MACCS_',         # 167 features
    'Abraham': 'abraham_',      # 5 features
    'AUTOCORR2D': 'AUTOCORR2D_', # 192 features
    'MOSE': 'mose_'             # 13 features
}


def get_bulk_feature_indices(columns):
    """Get indices for each bulk feature group."""
    groups = {}
    for group_name, prefix in BULK_GROUPS.items():
        groups[group_name] = set(i for i, col in enumerate(columns) if col.startswith(prefix))
    return groups


def train_and_evaluate(X_train, X_test, y_train, y_test, kept_indices, experiment_name):
    """Train model on subset of features and evaluate."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {len(kept_indices)} / {X_train.shape[1]}")
    
    # Subset features
    X_tr = X_train.iloc[:, list(kept_indices)]
    X_te = X_test.iloc[:, list(kept_indices)]
    
    # Train model (same params as train.py)
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        verbose=200,
        random_state=SEED,
        allow_writing_files=False,
        thread_count=-1
    )
    model.fit(X_tr, y_train)
    
    # Evaluate on test set
    preds = model.predict(X_te)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Test R²: {r2:.6f}, RMSE: {rmse:.6f}")
    
    return {'Experiment': experiment_name, 'R2': r2, 'RMSE': rmse, 'N_Features': len(kept_indices)}


def run_ablation_study():
    """Run ablation study on bulk feature groups."""
    print("="*60)
    print("BULK FEATURE ABLATION STUDY")
    print("Morgan, MACCS, Abraham, AUTOCORR2D, MOSE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Featurize
    print("Featurizing molecules...")
    featurizer = MoleculeFeaturizer()
    X_train = featurizer.transform(train_df['SMILES'])
    X_test = featurizer.transform(test_df['SMILES'])
    y_train = train_df['LogS']
    y_test = test_df['LogS']
    
    n_features = X_train.shape[1]
    columns = list(X_train.columns)
    all_indices = set(range(n_features))
    
    print(f"\nTotal features: {n_features}")
    
    # Get bulk feature groups
    groups = get_bulk_feature_indices(columns)
    print("\nBulk Feature Groups:")
    for name, indices in groups.items():
        print(f"  {name}: {len(indices)} features")
    
    results = []
    
    # =========================================================================
    # 1. FULL MODEL (Baseline)
    # =========================================================================
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        all_indices, "FULL MODEL (Baseline)"
    ))
    baseline_r2 = results[0]['R2']
    baseline_rmse = results[0]['RMSE']
    
    # =========================================================================
    # 2. REMOVE Individual Bulk Groups
    # =========================================================================
    print("\n" + "="*60)
    print("REMOVING INDIVIDUAL BULK GROUPS")
    print("="*60)
    
    for group_name in BULK_GROUPS.keys():
        remove_indices = groups[group_name]
        kept = all_indices - remove_indices
        results.append(train_and_evaluate(
            X_train, X_test, y_train, y_test,
            kept, f"REMOVE: {group_name}"
        ))
    
    # =========================================================================
    # 3. REMOVE Combinations of Bulk Groups
    # =========================================================================
    print("\n" + "="*60)
    print("REMOVING COMBINATIONS OF BULK GROUPS")
    print("="*60)
    
    # Remove all fingerprints (Morgan + MACCS)
    remove_fps = groups['Morgan'] | groups['MACCS']
    kept = all_indices - remove_fps
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        kept, "REMOVE: All Fingerprints (Morgan + MACCS)"
    ))
    
    # Remove all structural (Morgan + MACCS + MOSE)
    remove_struct = groups['Morgan'] | groups['MACCS'] | groups['MOSE']
    kept = all_indices - remove_struct
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        kept, "REMOVE: All Structural (Morgan + MACCS + MOSE)"
    ))
    
    # Remove all bulk features
    all_bulk = groups['Morgan'] | groups['MACCS'] | groups['Abraham'] | groups['AUTOCORR2D'] | groups['MOSE']
    kept = all_indices - all_bulk
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        kept, "REMOVE: ALL BULK FEATURES"
    ))
    
    # =========================================================================
    # 4. ONLY Experiments (using only one group)
    # =========================================================================
    print("\n" + "="*60)
    print("USING ONLY INDIVIDUAL BULK GROUPS")
    print("="*60)
    
    for group_name in BULK_GROUPS.keys():
        kept = groups[group_name]
        if len(kept) > 0:
            results.append(train_and_evaluate(
                X_train, X_test, y_train, y_test,
                kept, f"ONLY: {group_name}"
            ))
    
    # =========================================================================
    # 5. Summary and Analysis
    # =========================================================================
    df_results = pd.DataFrame(results)
    
    # Add delta columns
    df_results['Delta_R2'] = df_results['R2'] - baseline_r2
    df_results['Delta_RMSE'] = df_results['RMSE'] - baseline_rmse
    
    # Save results
    df_results.to_csv("ablation_bulk_features_results.csv", index=False)
    
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Highlight significant impacts
    print("\n" + "="*60)
    print("IMPACT ANALYSIS")
    print("="*60)
    
    print(f"\nBaseline: R² = {baseline_r2:.6f}, RMSE = {baseline_rmse:.6f}")
    print("\nPerformance drop when removing each group:")
    
    for group_name in BULK_GROUPS.keys():
        row = df_results[df_results['Experiment'] == f"REMOVE: {group_name}"].iloc[0]
        delta_r2 = row['Delta_R2']
        delta_rmse = row['Delta_RMSE']
        
        impact = "SIGNIFICANT" if abs(delta_r2) >= 0.01 or delta_rmse >= 0.02 else "MARGINAL"
        print(f"  {group_name}: ΔR² = {delta_r2:+.6f}, ΔRMSE = {delta_rmse:+.6f} [{impact}]")
    
    print("\n" + "="*60)
    print(f"Results saved to ablation_bulk_features_results.csv")
    print("="*60)
    
    return df_results


if __name__ == "__main__":
    run_ablation_study()
