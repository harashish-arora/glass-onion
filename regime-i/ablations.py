# ablations_filtered.py
# Ablation study with features from removed_features.csv excluded

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

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def load_removed_features():
    """Load list of features to exclude from removed_features.csv."""
    try:
        df_removed = pd.read_csv(REMOVED_FEATURES_PATH)
        removed_features = set(df_removed['Feature'].tolist())
        print(f"Loaded {len(removed_features)} features to exclude from {REMOVED_FEATURES_PATH}")
        return removed_features
    except FileNotFoundError:
        print(f"Warning: {REMOVED_FEATURES_PATH} not found. No features will be excluded.")
        return set()


def get_column_categories(columns, removed_features):
    """
    Classify columns into categories for ablation, excluding removed features.
    Also excludes Morgan and MACCS fingerprints.
    """
    categories = {
        'COMPOSITIONAL': [],
        'TOPOLOGICAL': [],
        'ENERGETIC': [],
        'PHYSICOCHEMICAL': [],
    }
    
    # Count-based features (COMPOSITIONAL)
    count_features = {
        'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
        'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
        'RingCount', 'HeavyAtomCount', 'total_atoms',
        'MolWt', 'ExactMolWt', 'HeavyAtomMolWt'
    }
    
    # Topological features (excluding Morgan and MACCS)
    topological_features = {
        'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
        'BalabanJ', 'BertzCT', 'Kappa1', 'Kappa2', 'HallKierAlpha',
        'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v'
    }
    
    # Property-based RDKit descriptors (PHYSICOCHEMICAL)
    property_descriptors = {
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
        'FractionCSP3',
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 
        'MolLogP',
        'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
        'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
        'TPSA', 'qed'
    }
    
    for i, col in enumerate(columns):
        # Skip removed features
        if col in removed_features:
            continue
        
        # Skip Morgan and MACCS fingerprints
        if col.startswith('Morgan_') or col.startswith('MACCS_'):
            continue
        
        # COMPOSITIONAL: atom counts, functional group counts, ring/atom counts
        if (col.startswith('num_') or col.startswith('fr_') or 
            col in count_features):
            categories['COMPOSITIONAL'].append(i)
        # TOPOLOGICAL: AUTOCORR2D, motifs, graph indices (excluding fingerprints)
        elif (col.startswith('mose_') or col.startswith('AUTOCORR2D_') or
              col in topological_features):
            categories['TOPOLOGICAL'].append(i)
        # ENERGETIC: pred_Tm and Abraham descriptors
        elif col == 'pred_Tm' or col.startswith('abraham_'):
            categories['ENERGETIC'].append(i)
        # PHYSICOCHEMICAL: property descriptors
        elif col in property_descriptors:
            categories['PHYSICOCHEMICAL'].append(i)
    
    return categories


def train_and_evaluate(X_train, X_test, y_train, y_test, kept_indices, experiment_name):
    """Train model on subset of features and evaluate."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {len(kept_indices)} / {X_train.shape[1]}")
    
    # Subset features
    X_tr = X_train.iloc[:, kept_indices]
    X_te = X_test.iloc[:, kept_indices]
    
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
    
    return {'Experiment': experiment_name, 'R2': r2, 'RMSE': rmse}


def run_ablation_study():
    """Run comprehensive ablation study with filtered features."""
    print("="*60)
    print("FILTERED ABLATION STUDY")
    print("Excluding Morgan/MACCS and removed features")
    print("="*60)
    
    # Load removed features
    removed_features = load_removed_features()
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Featurize
    print("Featurizing molecules...")
    featurizer = MoleculeFeaturizer()
    X_train_full = featurizer.transform(train_df['SMILES'])
    X_test_full = featurizer.transform(test_df['SMILES'])
    y_train = train_df['LogS']
    y_test = test_df['LogS']
    
    original_columns = list(X_train_full.columns)
    print(f"\nOriginal features: {len(original_columns)}")
    
    # Get category indices (this excludes removed features and Morgan/MACCS)
    categories = get_column_categories(original_columns, removed_features)
    
    # Calculate total kept features
    all_kept_indices = set()
    for indices in categories.values():
        all_kept_indices.update(indices)
    all_kept_indices = sorted(list(all_kept_indices))
    
    # Filter to only kept features
    kept_columns = [original_columns[i] for i in all_kept_indices]
    X_train = X_train_full.iloc[:, all_kept_indices]
    X_test = X_test_full.iloc[:, all_kept_indices]
    
    n_features = X_train.shape[1]
    
    print(f"\nFiltered features: {n_features}")
    print(f"Removed: {len(original_columns) - n_features} features")
    print(f"  - Morgan fingerprints: {sum(1 for c in original_columns if c.startswith('Morgan_'))}")
    print(f"  - MACCS keys: {sum(1 for c in original_columns if c.startswith('MACCS_'))}")
    print(f"  - Redundant features: {len(removed_features)}")
    
    print("\nCategory breakdown:")
    for cat, indices in categories.items():
        # Remap indices to filtered dataset
        remapped = [all_kept_indices.index(i) for i in indices]
        categories[cat] = remapped
        print(f"  {cat}: {len(remapped)} features")
    
    results = []
    all_indices = list(range(n_features))
    
    # =========================================================================
    # FULL MODEL (Baseline with filtered features)
    # =========================================================================
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        all_indices, "FULL MODEL (Filtered Features)"
    ))
    
    # =========================================================================
    # REMOVE EXPERIMENTS (Leave-One-Out)
    # =========================================================================
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        remove_indices = set(categories[category])
        kept = [i for i in all_indices if i not in remove_indices]
        if len(kept) > 0:
            results.append(train_and_evaluate(
                X_train, X_test, y_train, y_test,
                kept, f"REMOVE: {category}"
            ))
        else:
            print(f"\nSkipping REMOVE: {category} (no features remaining)")
    
    # =========================================================================
    # ONLY EXPERIMENTS
    # =========================================================================
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        kept = categories[category]
        if len(kept) > 0:
            results.append(train_and_evaluate(
                X_train, X_test, y_train, y_test,
                kept, f"ONLY: {category}"
            ))
        else:
            print(f"\nSkipping ONLY: {category} (no features)")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv("ablation_results_filtered.csv", index=False)
    print("\n" + "="*60)
    print("Report saved to ablation_results_filtered.csv")
    print(df_results.to_string(index=False))
    print("="*60)
    
    # Save feature breakdown
    feature_breakdown = []
    for cat, indices in categories.items():
        for idx in indices:
            feature_breakdown.append({
                'Feature': kept_columns[idx],
                'Category': cat
            })
    
    pd.DataFrame(feature_breakdown).to_csv("feature_categories_filtered.csv", index=False)
    print("\nFeature categories saved to feature_categories_filtered.csv")
    
    return df_results


if __name__ == "__main__":
    run_ablation_study()