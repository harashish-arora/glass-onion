# ablations_filtered_physchem.py
# Ablation with highly-correlated features REMOVED from PHYSICOCHEMICAL

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from featurizer import MoleculeFeaturizer

# Config
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
SEEDS = [101]  # Single seed for faster testing

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Import original category function
from ablations import get_column_categories


def get_filtered_physicochemical_indices(columns, categories):
    """
    Get PHYSICOCHEMICAL indices with highly-correlated features REMOVED.
    
    Removing:
    - BCUT2D_* (correlated with AUTOCORR2D, mose_*)
    - SMR_VSA* (correlated with num_O, fr_nitrile, NumAromaticCarbocycles)
    - SlogP_VSA* (correlated with Morgan, MACCS, AUTOCORR2D)
    - VSA_EState* (correlated with fr_benzene, num_Cl, MACCS)
    - abraham_* (correlated with Chi, NumHDonors, NOCount, NumHeteroatoms)
    - LabuteASA (0.98 correlated with total_atoms)
    - MolMR (0.98 correlated with Chi0v)
    """
    phys_indices = categories['PHYSICOCHEMICAL']
    
    # Features to REMOVE (highly correlated with other categories)
    remove_prefixes = ['BCUT2D_', 'SMR_VSA', 'SlogP_VSA', 'VSA_EState']
    remove_exact = ['LabuteASA', 'MolMR']
    
    filtered_indices = []
    removed_features = []
    kept_features = []
    
    for idx in phys_indices:
        col = columns[idx]
        
        # Check if should be removed
        should_remove = False
        for prefix in remove_prefixes:
            if col.startswith(prefix):
                should_remove = True
                break
        if col in remove_exact:
            should_remove = True
        
        if should_remove:
            removed_features.append(col)
        else:
            filtered_indices.append(idx)
            kept_features.append(col)
    
    return filtered_indices, kept_features, removed_features


def run_experiment(X_train, X_test, y_train, y_test, kept_indices, seed, exp_name):
    """Train model on subset of features and evaluate."""
    X_tr = X_train.iloc[:, kept_indices]
    X_te = X_test.iloc[:, kept_indices]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        verbose=0,
        random_state=seed,
        allow_writing_files=False,
        thread_count=-1
    )
    model.fit(X_tr, y_train)
    
    preds = model.predict(X_te)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return {'Experiment': exp_name, 'Seed': seed, 'R2': r2, 'RMSE': rmse}


def main():
    print("="*70)
    print("FILTERED PHYSICOCHEMICAL ABLATION")
    print("(Removing highly-correlated features)")
    print("="*70)
    
    # Load and featurize
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print("Featurizing molecules...")
    featurizer = MoleculeFeaturizer()
    X_train = featurizer.transform(train_df['SMILES'])
    X_test = featurizer.transform(test_df['SMILES'])
    y_train = train_df['LogS']
    y_test = test_df['LogS']
    
    columns = list(X_train.columns)
    categories = get_column_categories(columns)
    
    # Get filtered physicochemical indices
    filtered_phys_indices, kept_features, removed_features = get_filtered_physicochemical_indices(
        columns, categories
    )
    
    # Calculate full model indices (all features)
    all_indices = list(range(len(columns)))
    
    # Calculate filtered full model (remove correlated physchem from everywhere)
    removed_set = set()
    for idx in categories['PHYSICOCHEMICAL']:
        col = columns[idx]
        should_remove = False
        for prefix in ['BCUT2D_', 'SMR_VSA', 'SlogP_VSA', 'VSA_EState']:
            if col.startswith(prefix):
                should_remove = True
                break
        if col in ['LabuteASA', 'MolMR']:
            should_remove = True
        if should_remove:
            removed_set.add(idx)
    
    filtered_all_indices = [i for i in all_indices if i not in removed_set]
    
    print(f"\nOriginal PHYSICOCHEMICAL: {len(categories['PHYSICOCHEMICAL'])} features")
    print(f"Filtered PHYSICOCHEMICAL: {len(filtered_phys_indices)} features")
    print(f"Removed: {len(removed_features)} features")
    
    print(f"\nFULL MODEL: {len(all_indices)} features")
    print(f"FULL MODEL (filtered): {len(filtered_all_indices)} features")
    
    print("\n--- REMOVED FEATURES ---")
    for f in sorted(removed_features):
        print(f"  {f}")
    
    print("\n--- KEPT PHYSICOCHEMICAL FEATURES ---")
    for f in sorted(kept_features):
        print(f"  {f}")
    
    # Run experiments
    results = []
    
    # 1. FULL MODEL (original)
    print(f"\n{'='*70}")
    print("Running: FULL MODEL (original)")
    print(f"{'='*70}")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_experiment(X_train, X_test, y_train, y_test, all_indices, seed, "FULL MODEL")
        results.append(result)
        print(f"R²={result['R2']:.6f}")
    
    # 2. FULL MODEL (filtered - without correlated physchem)
    print(f"\n{'='*70}")
    print("Running: FULL MODEL (filtered)")
    print(f"{'='*70}")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_experiment(X_train, X_test, y_train, y_test, filtered_all_indices, seed, "FULL MODEL (filtered)")
        results.append(result)
        print(f"R²={result['R2']:.6f}")
    
    # 3. ONLY: PHYSICOCHEMICAL (filtered)
    print(f"\n{'='*70}")
    print("Running: ONLY PHYSICOCHEMICAL (filtered)")
    print(f"{'='*70}")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_experiment(X_train, X_test, y_train, y_test, filtered_phys_indices, seed, "ONLY: PHYSICOCHEMICAL (filtered)")
        results.append(result)
        print(f"R²={result['R2']:.6f}")
    
    # Summary
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for exp_name in ["FULL MODEL", "FULL MODEL (filtered)", "ONLY: PHYSICOCHEMICAL (filtered)"]:
        exp_data = df_results[df_results['Experiment'] == exp_name]
        r2_mean = exp_data['R2'].mean()
        r2_std = exp_data['R2'].std()
        rmse_mean = exp_data['RMSE'].mean()
        rmse_std = exp_data['RMSE'].std()
        
        n_features = len(all_indices) if exp_name == "FULL MODEL" else (
            len(filtered_all_indices) if exp_name == "FULL MODEL (filtered)" else len(filtered_phys_indices)
        )
        
        print(f"\n{exp_name} ({n_features} features):")
        print(f"  R²:   {r2_mean:.6f} ± {r2_std:.6f}")
        print(f"  RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}")
    
    df_results.to_csv("filtered_physchem_results.csv", index=False)
    print(f"\nResults saved to filtered_physchem_results.csv")
    print("="*70)


if __name__ == "__main__":
    main()
