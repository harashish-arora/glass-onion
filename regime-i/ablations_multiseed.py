# ablations_multiseed.py
# Multi-seed ablation to check variance in ONLY: PHYSICOCHEMICAL results

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
SEEDS = [42, 101, 202, 303, 404]  # Multiple seeds to test variance

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Import category function from ablations.py
from ablations import get_column_categories


def run_experiment(X_train, X_test, y_train, y_test, kept_indices, seed, exp_name):
    """Train model on subset of features and evaluate."""
    X_tr = X_train.iloc[:, kept_indices]
    X_te = X_test.iloc[:, kept_indices]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        verbose=0,  # Silent for multi-seed
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
    print("="*60)
    print("MULTI-SEED ABLATION STUDY")
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
    
    columns = list(X_train.columns)
    categories = get_column_categories(columns)
    
    all_indices = list(range(len(columns)))
    phys_indices = categories['PHYSICOCHEMICAL']
    
    print(f"\nTotal features: {len(columns)}")
    print(f"PHYSICOCHEMICAL features: {len(phys_indices)}")
    print(f"\nRunning {len(SEEDS)} seeds for each experiment...")
    
    results = []
    
    # Run FULL MODEL with multiple seeds
    print("\n--- FULL MODEL ---")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_experiment(X_train, X_test, y_train, y_test, all_indices, seed, "FULL MODEL")
        results.append(result)
        print(f"R²={result['R2']:.6f}")
    
    # Run ONLY: PHYSICOCHEMICAL with multiple seeds
    print("\n--- ONLY: PHYSICOCHEMICAL ---")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_experiment(X_train, X_test, y_train, y_test, phys_indices, seed, "ONLY: PHYSICOCHEMICAL")
        results.append(result)
        print(f"R²={result['R2']:.6f}")
    
    # Summarize
    df_results = pd.DataFrame(results)
    df_results.to_csv("multiseed_results.csv", index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for exp_name in ["FULL MODEL", "ONLY: PHYSICOCHEMICAL"]:
        exp_data = df_results[df_results['Experiment'] == exp_name]
        r2_mean = exp_data['R2'].mean()
        r2_std = exp_data['R2'].std()
        rmse_mean = exp_data['RMSE'].mean()
        rmse_std = exp_data['RMSE'].std()
        
        print(f"\n{exp_name}:")
        print(f"  R²:   {r2_mean:.6f} ± {r2_std:.6f}")
        print(f"  RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}")
    
    print("\n" + "="*60)
    print("Detailed results saved to multiseed_results.csv")
    print("="*60)


if __name__ == "__main__":
    main()
