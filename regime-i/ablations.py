# ablations.py
# Ablation study for Glass Onion Regime-I model

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from featurizer import MoleculeFeaturizer

# Config (same as train.py)
SEED = 101
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ============================================================================
# FEATURE CATEGORY DEFINITIONS
# ============================================================================

def get_column_categories(columns):
    """Classify columns into categories for ablation."""
    categories = {
        'COMPOSITIONAL': [],    # Counts: num_*, fr_*, Num*, *Count, RingCount, HeavyAtomCount
        'TOPOLOGICAL': [],      # Morgan_*, MACCS_*, mose_*, AUTOCORR2D_*
        'ENERGETIC': [],        # pred_Tm
        'PHYSICOCHEMICAL': [],  # Properties: abraham_*, BCUT2D_*, MolLogP, TPSA, Chi*, etc.
    }
    
    # Count-based features (go to COMPOSITIONAL)
    # Includes: atom counts, ring counts, H-bond counts, weight-based features
    count_features = {
        'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
        'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
        'RingCount', 'HeavyAtomCount', 'total_atoms',
        # Weight-based (sum of atomic weights = count-derived)
        'MolWt', 'ExactMolWt', 'HeavyAtomMolWt'
    }
    
    # Topological features that don't start with standard prefixes
    topological_features = {
        'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',  # fingerprint density
        'BalabanJ', 'BertzCT',  # graph-theoretic indices
        'Kappa1', 'Kappa2', 'HallKierAlpha',  # Kappa shape indices
        # Chi (Kier-Hall molecular connectivity) - graph-based branching descriptors
        'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v'
    }
    
    # Property-based RDKit descriptors (go to PHYSICOCHEMICAL)
    # Note: BCUT2D, SMR_VSA, SlogP_VSA, VSA_EState, LabuteASA, MolMR removed from featurizer
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
        'TPSA', 'qed'  # drug-likeness score
    }
    
    for i, col in enumerate(columns):
        # COMPOSITIONAL: atom counts, functional group counts, ring/atom counts
        if (col.startswith('num_') or col.startswith('fr_') or 
            col in count_features):
            categories['COMPOSITIONAL'].append(i)
        # TOPOLOGICAL: fingerprints, motifs, autocorrelation, graph indices
        elif (col.startswith('Morgan_') or col.startswith('MACCS_') or 
              col.startswith('mose_') or col.startswith('AUTOCORR2D_') or
              col in topological_features):
            categories['TOPOLOGICAL'].append(i)
        # ENERGETIC: pred_Tm only
        elif col == 'pred_Tm':
            categories['ENERGETIC'].append(i)
        # PHYSICOCHEMICAL: abraham_*, BCUT2D_*, property descriptors
        elif (col.startswith('abraham_') or col.startswith('BCUT2D_') or 
              col in property_descriptors):
            categories['PHYSICOCHEMICAL'].append(i)
    
    return categories


# ============================================================================
# ABLATION EXPERIMENTS
# ============================================================================

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
    """Run comprehensive ablation study."""
    print("="*60)
    print("GLASS ONION REGIME-I ABLATION STUDY")
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
    
    print(f"\nTotal features: {n_features}")
    
    # Get category indices
    categories = get_column_categories(columns)
    for cat, indices in categories.items():
        print(f"  {cat}: {len(indices)} features")
    
    results = []
    all_indices = list(range(n_features))
    
    # =========================================================================
    # FULL MODEL (Baseline)
    # =========================================================================
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        all_indices, "FULL MODEL (Original)"
    ))
    
    # =========================================================================
    # REMOVE EXPERIMENTS (Leave-One-Out)
    # =========================================================================
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        remove_indices = set(categories[category])
        kept = [i for i in all_indices if i not in remove_indices]
        results.append(train_and_evaluate(
            X_train, X_test, y_train, y_test,
            kept, f"REMOVE: {category}"
        ))
    
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
    df_results.to_csv("ablation_results.csv", index=False)
    print("\n" + "="*60)
    print("Report saved to ablation_results.csv")
    print(df_results.to_string(index=False))
    print("="*60)
    
    return df_results


if __name__ == "__main__":
    run_ablation_study()
