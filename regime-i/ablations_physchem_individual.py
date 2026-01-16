# ablations_physchem_individual.py
# Leave-one-out ablation study on PHYSICOCHEMICAL features
# Excludes: Morgan Fingerprints, MACCS keys, and features from removed_features.csv

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
REMOVED_FEATURES_PATH = "./removed_features.csv"
FEATURE_CATEGORIES_PATH = "./feature_categories_filtered.csv"

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def load_removed_features():
    """Load list of features to exclude from removed_features.csv."""
    try:
        df_removed = pd.read_csv(REMOVED_FEATURES_PATH)
        removed_features = set(df_removed['Feature'].dropna().tolist())
        return removed_features
    except FileNotFoundError:
        print(f"Warning: {REMOVED_FEATURES_PATH} not found. No features will be excluded.")
        return set()


def load_physicochemical_features():
    """Load PHYSICOCHEMICAL features from feature_categories_filtered.csv."""
    try:
        df_cat = pd.read_csv(FEATURE_CATEGORIES_PATH)
        physchem_features = df_cat[df_cat['Category'] == 'PHYSICOCHEMICAL']['Feature'].tolist()
        return physchem_features
    except FileNotFoundError:
        print(f"Warning: {FEATURE_CATEGORIES_PATH} not found.")
        return []


def get_valid_feature_indices(columns, removed_features):
    """
    Get indices of valid features, excluding:
    - Morgan Fingerprints (Morgan_*)
    - MACCS keys (MACCS_*)
    - Features from removed_features.csv
    """
    valid_indices = []
    excluded_morgan = 0
    excluded_maccs = 0
    excluded_removed = 0
    
    for i, col in enumerate(columns):
        if col.startswith('Morgan_'):
            excluded_morgan += 1
            continue
        if col.startswith('MACCS_'):
            excluded_maccs += 1
            continue
        if col in removed_features:
            excluded_removed += 1
            continue
        valid_indices.append(i)
    
    print(f"  Excluded Morgan Fingerprints: {excluded_morgan}")
    print(f"  Excluded MACCS Keys: {excluded_maccs}")
    print(f"  Excluded from removed_features.csv: {excluded_removed}")
    print(f"  Valid features remaining: {len(valid_indices)}")
    
    return valid_indices


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_indices, verbose=True):
    """Train model on subset of features and evaluate."""
    X_tr = X_train.iloc[:, feature_indices]
    X_te = X_test.iloc[:, feature_indices]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        verbose=200 if verbose else 0,
        random_state=SEED,
        allow_writing_files=False,
        thread_count=-1
    )
    model.fit(X_tr, y_train)
    
    preds = model.predict(X_te)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return r2, rmse


def run_physchem_ablation():
    """Run leave-one-out ablation on PHYSICOCHEMICAL features."""
    print("=" * 70)
    print("PHYSICOCHEMICAL FEATURE-BY-FEATURE ABLATION STUDY")
    print("Excluding: Morgan Fingerprints, MACCS Keys, removed_features.csv")
    print("=" * 70)
    
    # Load exclusion lists
    print("\n--- Loading Exclusion Lists ---")
    removed_features = load_removed_features()
    print(f"Loaded {len(removed_features)} features from removed_features.csv")
    
    # Load PHYSICOCHEMICAL features to ablate
    physchem_features = load_physicochemical_features()
    print(f"Loaded {len(physchem_features)} PHYSICOCHEMICAL features to ablate")
    
    # Load and featurize data
    print("\n--- Loading Data ---")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print("Featurizing molecules...")
    featurizer = MoleculeFeaturizer()
    X_train = featurizer.transform(train_df['SMILES'])
    X_test = featurizer.transform(test_df['SMILES'])
    y_train = train_df['LogS']
    y_test = test_df['LogS']
    
    columns = list(X_train.columns)
    print(f"\nTotal features from featurizer: {len(columns)}")
    
    # Get valid feature indices (excluding fingerprints + removed features)
    print("\n--- Filtering Features ---")
    valid_indices = get_valid_feature_indices(columns, removed_features)
    valid_columns = [columns[i] for i in valid_indices]
    
    # Create mapping from column name to valid index
    col_to_valid_idx = {col: i for i, col in enumerate(valid_columns)}
    
    # Identify PHYSICOCHEMICAL features that are in our valid set
    physchem_in_valid = [f for f in physchem_features if f in col_to_valid_idx]
    print(f"\nPHYSICOCHEMICAL features in valid set: {len(physchem_in_valid)}")
    
    if not physchem_in_valid:
        print("ERROR: No PHYSICOCHEMICAL features found in valid feature set!")
        return None
    
    print("\nFeatures to ablate:")
    for f in physchem_in_valid:
        print(f"  - {f}")
    
    results = []
    
    # =========================================================================
    # 1. BASELINE: All valid features
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: All valid features")
    print(f"Features: {len(valid_indices)}")
    print("=" * 70)
    
    baseline_r2, baseline_rmse = train_and_evaluate(
        X_train, X_test, y_train, y_test, valid_indices, verbose=True
    )
    
    print(f"\nBaseline R²: {baseline_r2:.6f}, RMSE: {baseline_rmse:.6f}")
    
    results.append({
        'Feature': 'BASELINE (all valid features)',
        'N_Features': len(valid_indices),
        'R2': baseline_r2,
        'RMSE': baseline_rmse,
        'Delta_R2': 0.0,
        'Delta_RMSE': 0.0,
        'Impact': 'BASELINE'
    })
    
    # =========================================================================
    # 2. LEAVE-ONE-OUT: Remove each PHYSICOCHEMICAL feature individually
    # =========================================================================
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT ABLATION ON PHYSICOCHEMICAL FEATURES")
    print("=" * 70)
    
    for i, feature_name in enumerate(physchem_in_valid):
        print(f"\n[{i+1}/{len(physchem_in_valid)}] Removing: {feature_name}")
        
        # Get the original column index
        original_idx = columns.index(feature_name)
        
        # Create feature set without this feature
        ablated_indices = [idx for idx in valid_indices if idx != original_idx]
        
        print(f"  Features: {len(ablated_indices)}")
        
        r2, rmse = train_and_evaluate(
            X_train, X_test, y_train, y_test, ablated_indices, verbose=False
        )
        
        delta_r2 = r2 - baseline_r2
        delta_rmse = rmse - baseline_rmse
        
        # Determine impact level
        if delta_r2 <= -0.01 or delta_rmse >= 0.02:
            impact = "HIGH"
        elif delta_r2 <= -0.005 or delta_rmse >= 0.01:
            impact = "MEDIUM"
        else:
            impact = "LOW"
        
        print(f"  R²: {r2:.6f} (Δ: {delta_r2:+.6f}), RMSE: {rmse:.6f} (Δ: {delta_rmse:+.6f}) [{impact}]")
        
        results.append({
            'Feature': feature_name,
            'N_Features': len(ablated_indices),
            'R2': r2,
            'RMSE': rmse,
            'Delta_R2': delta_r2,
            'Delta_RMSE': delta_rmse,
            'Impact': impact
        })
    
    # =========================================================================
    # 3. Save and Summarize Results
    # =========================================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv("ablation_physchem_individual_results.csv", index=False)
    
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nBaseline: R² = {baseline_r2:.6f}, RMSE = {baseline_rmse:.6f}")
    print(f"Features ablated: {len(physchem_in_valid)}")
    
    # Sort by impact (most impactful = largest R² drop)
    feature_results = df_results[df_results['Feature'] != 'BASELINE (all valid features)'].copy()
    feature_results_sorted = feature_results.sort_values('Delta_R2', ascending=True)
    
    print("\n--- Features Ranked by Impact (most important first) ---")
    print(f"{'Feature':<25} {'R²':>10} {'ΔRMSE':>10} {'ΔR²':>10} {'Impact':>8}")
    print("-" * 65)
    
    for _, row in feature_results_sorted.iterrows():
        print(f"{row['Feature']:<25} {row['R2']:>10.6f} {row['Delta_RMSE']:>+10.6f} {row['Delta_R2']:>+10.6f} {row['Impact']:>8}")
    
    # Count by impact level
    print("\n--- Impact Summary ---")
    for impact in ['HIGH', 'MEDIUM', 'LOW']:
        count = len(feature_results[feature_results['Impact'] == impact])
        print(f"  {impact}: {count} features")
    
    print(f"\nResults saved to: ablation_physchem_individual_results.csv")
    print("=" * 70)
    
    return df_results


if __name__ == "__main__":
    run_physchem_ablation()
