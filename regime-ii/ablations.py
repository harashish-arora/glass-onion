# ablations.py
# Comprehensive ablation study for Glass Onion regime-II model
# Run with: conda activate molmerger && python ablations.py

import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor

from train_transformer import InteractionTransformer, DEVICE

# Config (same as train.py)
DATA_DIR, STORE_DIR, MODEL_DIR = "data", "feature_store", "model"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
OOF_EMBED_FILE = "train_embeddings.csv"
TRANSFORMER_PATH = "transformer.pth"
SEED = 42

warnings.filterwarnings("ignore")

# ============================================================================
# FEATURE CATEGORY DEFINITIONS
# ============================================================================

def get_column_categories(columns):
    """Classify columns into categories for ablation."""
    categories = {
        'COMPOSITIONAL': [],    # Counts: num_*, fr_*, Num*, *Count, weights
        'TOPOLOGICAL': [],      # Morgan_*, MACCS_*, mose_*, AUTOCORR2D_*, graph indices
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
# FEATURE LOADING (adapted from train.py)
# ============================================================================

def load_raw_feature_dataframes():
    """Load raw feature parquet files."""
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    return sol_raw, solv_raw


def build_feature_matrix_with_labels(df, embed_df, sol_raw, solv_raw):
    """
    Build feature matrix and return feature labels for ablation.
    Returns: X_full, feature_info dict with indices for each group
    """
    # Get raw column names
    sol_cols = list(sol_raw.columns)
    solv_cols = list(solv_raw.columns)
    
    # 1. Raw features (x_A and x_B)
    X_sol = sol_raw.loc[df['Solute']].values
    X_solv = solv_raw.loc[df['Solvent']].values
    X_raw = np.hstack([X_sol, X_solv])
    n_sol = X_sol.shape[1]
    n_solv = X_solv.shape[1]
    
    # 2. Transformer embeddings (Z_A->B)
    X_embed = embed_df[[c for c in embed_df.columns if c.startswith("Learned_")]].values
    n_embed = X_embed.shape[1]
    
    # 3. Temperature data for interactions
    Tm = sol_raw.loc[df['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_raw = df['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df['Temperature'].values).reshape(-1, 1).astype(np.float32)
    T_red = (T_raw / Tm).astype(np.float32)
    
    # 4. Interaction terms (I)
    X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
    X_modulus = np.linalg.norm(X_reshaped, axis=2)
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_modulus) * T_inv
    n_interact = X_interact.shape[1]
    
    # Stack all features
    X_full = np.hstack([X_raw, X_embed, X_interact, Tm, T_red, T_raw, T_inv])
    
    # Calculate indices for each feature group
    idx = 0
    feature_info = {
        'x_A': {'start': idx, 'end': idx + n_sol, 'cols': sol_cols},
        'x_B': {'start': idx + n_sol, 'end': idx + n_sol + n_solv, 'cols': solv_cols},
    }
    idx += n_sol + n_solv
    
    feature_info['Z_AB'] = {'start': idx, 'end': idx + n_embed}
    idx += n_embed
    
    feature_info['I'] = {'start': idx, 'end': idx + n_interact}
    idx += n_interact
    
    feature_info['f_T'] = {
        'start': idx, 'end': idx + 4,
        'Tm_idx': idx,       # pred_Tm
        'T_red_idx': idx + 1,  # T_red
        'T_idx': idx + 2,      # T (raw temperature - KEEP THIS)
        'T_inv_idx': idx + 3   # 1/T
    }
    
    # Get category indices for x_A and x_B
    sol_categories = get_column_categories(sol_cols)
    solv_categories = get_column_categories(solv_cols)
    
    # Adjust category indices to global indices
    feature_info['x_A']['categories'] = {
        cat: [i + feature_info['x_A']['start'] for i in indices]
        for cat, indices in sol_categories.items()
    }
    feature_info['x_B']['categories'] = {
        cat: [i + feature_info['x_B']['start'] for i in indices]
        for cat, indices in solv_categories.items()
    }
    
    return X_full, feature_info


def generate_test_features(df_test, sol_raw, solv_raw):
    """Generate test features (adapted from train.py)."""
    sol_c = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index('SMILES_KEY')
    solv_c = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index('SMILES_KEY')
    
    # Load Transformer
    model = InteractionTransformer().to(DEVICE)
    model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
    model.eval()
    
    # Batch Processing
    X_sol_all = sol_c.loc[df_test['Solute']].values.astype(np.float32)
    X_solv_all = solv_c.loc[df_test['Solvent']].values.astype(np.float32)
    
    batch_size = 512
    embed_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_sol_all), batch_size):
            b_sol = torch.tensor(X_sol_all[i : i+batch_size]).to(DEVICE)
            b_solv = torch.tensor(X_solv_all[i : i+batch_size]).to(DEVICE)
            _, feats, _ = model(b_sol, b_solv)
            embed_list.append(feats.cpu().numpy())
            
    X_embed = np.vstack(embed_list)
    
    # Temperature features
    T = df_test['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df_test['Temperature'].values).reshape(-1, 1).astype(np.float32)
    Tm = sol_raw.loc[df_test['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)
    
    # Interaction terms
    X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
    X_interact = (np.sign(X_reshaped.mean(axis=2)) * np.linalg.norm(X_reshaped, axis=2)) * T_inv
    
    # Raw features
    X_raw = np.hstack([sol_raw.loc[df_test['Solute']].values, solv_raw.loc[df_test['Solvent']].values])
    
    return np.hstack([X_raw, X_embed, X_interact, Tm, T_red, T, T_inv])


# ============================================================================
# ABLATION EXPERIMENTS
# ============================================================================

def get_monotone_constraints(n_features, feature_info, kept_indices):
    """
    Calculate monotone constraints for the given feature subset.
    T_red: +1, T: +1, T_inv: -1
    """
    mono = [0] * len(kept_indices)
    
    # Map original indices to new indices
    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(kept_indices)}
    
    f_T = feature_info['f_T']
    
    # T_red (index 1 in f_T block)
    if f_T['T_red_idx'] in idx_map:
        mono[idx_map[f_T['T_red_idx']]] = 1
    
    # T (index 2 in f_T block)
    if f_T['T_idx'] in idx_map:
        mono[idx_map[f_T['T_idx']]] = 1
    
    # T_inv (index 3 in f_T block)
    if f_T['T_inv_idx'] in idx_map:
        mono[idx_map[f_T['T_inv_idx']]] = -1
    
    return mono


def train_and_evaluate(X_train_full, X_test_full, y_train, y_test, kept_indices, 
                       feature_info, experiment_name):
    """Train model on subset of features and evaluate."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {len(kept_indices)} / {X_train_full.shape[1]}")
    
    # Subset features
    X_train = X_train_full[:, kept_indices]
    X_test = X_test_full[:, kept_indices]
    
    # Variance pruning on this subset
    selector = VarianceThreshold(threshold=0.0001)
    X_train_pruned = selector.fit_transform(X_train)
    X_test_pruned = selector.transform(X_test)
    
    # Map kept_indices through variance selector
    kept_after_var = [kept_indices[i] for i, keep in enumerate(selector.get_support()) if keep]
    
    print(f"After variance pruning: {X_train_pruned.shape[1]}")
    
    # Get monotone constraints
    mono = get_monotone_constraints(X_train_full.shape[1], feature_info, kept_after_var)
    
    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_pruned, y_train, test_size=0.05, random_state=SEED
    )
    
    # Train model (same params as train.py)
    model = CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=5,
        monotone_constraints=mono, early_stopping_rounds=100, 
        random_seed=SEED, verbose=200
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    # Evaluate on test set
    preds = model.predict(X_test_pruned)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Test R²: {r2:.6f}, RMSE: {rmse:.6f}")
    
    return {'Experiment': experiment_name, 'R2': r2, 'RMSE': rmse}


def run_ablation_study():
    """Run comprehensive ablation study."""
    print("="*60)
    print("GLASS ONION ABLATION STUDY")
    print("="*60)
    
    # Load data
    print("\nLoading data and features...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    df_oof = pd.read_csv(OOF_EMBED_FILE)
    sol_raw, solv_raw = load_raw_feature_dataframes()
    
    y_train = df_train['LogS'].values
    y_test = df_test['LogS'].values
    
    # Build feature matrices with labels
    X_train_full, feature_info = build_feature_matrix_with_labels(
        df_train, df_oof, sol_raw, solv_raw
    )
    X_test_full = generate_test_features(df_test, sol_raw, solv_raw)
    
    n_features = X_train_full.shape[1]
    print(f"\nTotal features: {n_features}")
    print(f"x_A: {feature_info['x_A']['start']}-{feature_info['x_A']['end']} ({feature_info['x_A']['end'] - feature_info['x_A']['start']} features)")
    print(f"x_B: {feature_info['x_B']['start']}-{feature_info['x_B']['end']} ({feature_info['x_B']['end'] - feature_info['x_B']['start']} features)")
    print(f"Z_AB: {feature_info['Z_AB']['start']}-{feature_info['Z_AB']['end']} ({feature_info['Z_AB']['end'] - feature_info['Z_AB']['start']} features)")
    print(f"I: {feature_info['I']['start']}-{feature_info['I']['end']} ({feature_info['I']['end'] - feature_info['I']['start']} features)")
    print(f"f(T): {feature_info['f_T']['start']}-{feature_info['f_T']['end']} (4 features: Tm, T_red, T, T_inv)")
    
    results = []
    
    # Helper to get all indices
    all_indices = list(range(n_features))
    
    # Get category indices across x_A and x_B
    def get_category_indices(category):
        indices = []
        indices.extend(feature_info['x_A']['categories'].get(category, []))
        indices.extend(feature_info['x_B']['categories'].get(category, []))
        return indices
    
    # =========================================================================
    # FULL MODEL (Baseline) - SKIPPED (run separately via train.py)
    # =========================================================================
    # results.append(train_and_evaluate(
    #     X_train_full, X_test_full, y_train, y_test,
    #     all_indices, feature_info, "FULL MODEL (Original)"
    # ))
    
    # =========================================================================
    # REMOVE EXPERIMENTS (Leave-One-Out)
    # =========================================================================
    
    # 1. Remove x_A and x_B entirely (keep only Z_AB, I, f_T)
    kept = list(range(feature_info['Z_AB']['start'], n_features))
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "REMOVE: x_A, x_B Entirely"
    ))
    
    # 2. Remove categories from x_A and x_B
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        remove_indices = set(get_category_indices(category))
        kept = [i for i in all_indices if i not in remove_indices]
        results.append(train_and_evaluate(
            X_train_full, X_test_full, y_train, y_test,
            kept, feature_info, f"REMOVE: Category {category}"
        ))
    
    # 3. Remove Z_A->B (Latent)
    remove_indices = set(range(feature_info['Z_AB']['start'], feature_info['Z_AB']['end']))
    kept = [i for i in all_indices if i not in remove_indices]
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "REMOVE: Z_A->B (Latent)"
    ))
    
    # 4. Remove I (Interaction Terms)
    remove_indices = set(range(feature_info['I']['start'], feature_info['I']['end']))
    kept = [i for i in all_indices if i not in remove_indices]
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "REMOVE: I (Interaction Terms)"
    ))
    
    # 5. Remove f(T) (Engineered State) - BUT KEEP raw T
    f_T = feature_info['f_T']
    remove_indices = {f_T['Tm_idx'], f_T['T_red_idx'], f_T['T_inv_idx']}  # Keep T_idx
    kept = [i for i in all_indices if i not in remove_indices]
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "REMOVE: f(T) (Engineered State)"
    ))
    
    # =========================================================================
    # ONLY EXPERIMENTS
    # =========================================================================
    
    # Always include raw T for ONLY experiments (since it's required)
    T_idx = feature_info['f_T']['T_idx']
    
    # ONLY categories
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        kept = get_category_indices(category) + [T_idx]
        kept = sorted(set(kept))
        results.append(train_and_evaluate(
            X_train_full, X_test_full, y_train, y_test,
            kept, feature_info, f"ONLY: Category {category}"
        ))
    
    # ONLY Z_A->B
    kept = list(range(feature_info['Z_AB']['start'], feature_info['Z_AB']['end'])) + [T_idx]
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "ONLY: Z_A->B (Latent)"
    ))
    
    # ONLY I (Interaction Terms)
    kept = list(range(feature_info['I']['start'], feature_info['I']['end'])) + [T_idx]
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "ONLY: I (Interaction Terms)"
    ))
    
    # ONLY f(T) (all temperature features)
    kept = list(range(feature_info['f_T']['start'], feature_info['f_T']['end']))
    results.append(train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        kept, feature_info, "ONLY: f(T) (Engineered State)"
    ))
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv("comprehensive_ablation_results.csv", index=False)
    print("\n" + "="*60)
    print("Report saved to comprehensive_ablation_results.csv")
    print(df_results.to_string(index=False))
    print("="*60)
    
    return df_results


if __name__ == "__main__":
    run_ablation_study()
