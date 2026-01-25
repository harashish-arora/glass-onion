# solvent_interaction_analysis.py
# Analysis: Do interaction terms matter more for polar solvents?
# Run with: conda activate molmerger && python solvent_interaction_analysis.py

import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor

from train_transformer import InteractionTransformer, DEVICE

# Config
DATA_DIR, STORE_DIR = "data", "feature_store"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
OOF_EMBED_FILE = "train_embeddings.csv"
TRANSFORMER_PATH = "transformer.pth"
SEED = 123

warnings.filterwarnings("ignore")

# =============================================================================
# SOLVENT POLARITY CLASSIFICATION
# =============================================================================

# Key solvents from test set mapped by polarity class
# Dielectric constants: https://depts.washington.edu/eooptic/linkfiles/dielectric_chart%5B1%5D.pdf
SOLVENT_INFO = {
    # SMILES: (Name, Polarity_Class, Dielectric_Constant)
    "CCCCCC": ("Hexane", "non-polar", 1.88),
    "c1ccccc1": ("Benzene", "non-polar", 2.28),
    "CCCCCCC": ("Heptane", "non-polar", 1.92),
    "ClCCl": ("Dichloromethane", "polar-aprotic", 8.93),
    "ClCCCl": ("1,2-Dichloroethane", "polar-aprotic", 10.4),
    "ClC(Cl)(Cl)Cl": ("Carbon Tetrachloride", "non-polar", 2.24),
    "CS(C)=O": ("DMSO", "polar-aprotic", 47.2),
    "CN1CCCC1=O": ("NMP", "polar-aprotic", 32.2),
    "CC(=O)N(C)C": ("DMAc", "polar-aprotic", 37.8),
    "CCCCOC(C)=O": ("Butyl Acetate", "polar-aprotic", 5.01),
    "CCCOC(C)=O": ("Propyl Acetate", "polar-aprotic", 5.6),
    "O=C1CCCCC1": ("Cyclohexanone", "polar-aprotic", 18.3),
    "CCC(C)O": ("2-Butanol", "polar-protic", 17.3),
    "CCCCCCCCO": ("1-Octanol", "polar-protic", 10.3),
    "OCCO": ("Ethylene Glycol", "polar-protic", 37.7),
    "CCOCCO": ("2-Ethoxyethanol", "polar-protic", 13.3),
    "CC(O)CO": ("Propylene Glycol", "polar-protic", 32.0),
    "CC(C)CCO": ("3-Methyl-1-butanol", "polar-protic", 14.7),
    "CCCCCCO": ("1-Hexanol", "polar-protic", 13.3),
    "CC(=O)O": ("Acetic Acid", "polar-protic", 6.2),
}

def classify_solvent(smiles):
    """Classify solvent by polarity."""
    if smiles in SOLVENT_INFO:
        return SOLVENT_INFO[smiles][1]
    # Heuristic fallback based on functional groups
    if "O" in smiles and ("O=" not in smiles):
        return "polar-protic"
    elif "O=" in smiles or "N" in smiles or "S" in smiles:
        return "polar-aprotic"
    else:
        return "non-polar"

def get_solvent_name(smiles):
    """Get human-readable solvent name."""
    if smiles in SOLVENT_INFO:
        return SOLVENT_INFO[smiles][0]
    return smiles[:20] + "..." if len(smiles) > 20 else smiles


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def load_raw_feature_dataframes():
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    return sol_raw, solv_raw


def build_train_features(df, embed_df, sol_raw, solv_raw, include_interaction=True):
    """Build feature matrix with option to exclude interaction terms."""
    X_sol = sol_raw.loc[df['Solute']].values
    X_solv = solv_raw.loc[df['Solvent']].values
    X_raw = np.hstack([X_sol, X_solv])
    
    # Temperature features
    Tm = sol_raw.loc[df['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_raw = df['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df['Temperature'].values).reshape(-1, 1).astype(np.float32)
    T_red = (T_raw / Tm).astype(np.float32)
    
    if include_interaction:
        # Interaction terms from transformer
        X_embed = embed_df[[c for c in embed_df.columns if c.startswith("Learned_")]].values
        X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
        X_modulus = np.linalg.norm(X_reshaped, axis=2)
        X_sign = np.sign(X_reshaped.mean(axis=2))
        X_interact = (X_sign * X_modulus) * T_inv
        return np.hstack([X_raw, X_interact, Tm, T_red, T_raw, T_inv])
    else:
        return np.hstack([X_raw, Tm, T_red, T_raw, T_inv])


def build_test_features(df_test, sol_raw, solv_raw, include_interaction=True):
    """Generate test features with option to exclude interaction terms."""
    sol_c = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index('SMILES_KEY')
    solv_c = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index('SMILES_KEY')
    
    X_raw = np.hstack([sol_raw.loc[df_test['Solute']].values, solv_raw.loc[df_test['Solvent']].values])
    
    T = df_test['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df_test['Temperature'].values).reshape(-1, 1).astype(np.float32)
    Tm = sol_raw.loc[df_test['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)
    
    if include_interaction:
        model = InteractionTransformer().to(DEVICE)
        model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
        model.eval()
        
        X_sol_all = sol_c.loc[df_test['Solute']].values.astype(np.float32)
        X_solv_all = solv_c.loc[df_test['Solvent']].values.astype(np.float32)
        
        batch_size = 512
        embed_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_sol_all), batch_size):
                b_sol = torch.tensor(X_sol_all[i:i+batch_size]).to(DEVICE)
                b_solv = torch.tensor(X_solv_all[i:i+batch_size]).to(DEVICE)
                _, feats, _ = model(b_sol, b_solv)
                embed_list.append(feats.cpu().numpy())
                
        X_embed = np.vstack(embed_list)
        X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
        X_interact = (np.sign(X_reshaped.mean(axis=2)) * np.linalg.norm(X_reshaped, axis=2)) * T_inv
        return np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
    else:
        return np.hstack([X_raw, Tm, T_red, T, T_inv])


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(X_train, y_train, seed=SEED):
    """Train CatBoost model."""
    selector = VarianceThreshold(threshold=0.0001)
    X_pruned = selector.fit_transform(X_train)
    
    # Monotone constraints for temperature features (last 4)
    mono = [0] * X_pruned.shape[1]
    mono[-3], mono[-2], mono[-1] = 1, 1, -1  # T_red, T, 1/T
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pruned, y_train, test_size=0.05, random_state=seed
    )
    
    model = CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=5,
        monotone_constraints=mono, early_stopping_rounds=100,
        random_seed=seed, verbose=200, thread_count=-1
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    return model, selector


def compute_per_solvent_rmse(df_test, y_test, preds):
    """Compute RMSE for each solvent in test set."""
    results = {}
    for solvent in df_test['Solvent'].unique():
        mask = df_test['Solvent'] == solvent
        if mask.sum() >= 10:  # Only include solvents with enough samples
            rmse = np.sqrt(mean_squared_error(y_test[mask], preds[mask]))
            results[solvent] = {
                'rmse': rmse,
                'count': mask.sum(),
                'name': get_solvent_name(solvent),
                'polarity': classify_solvent(solvent)
            }
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis():
    print("=" * 70)
    print("ANALYSIS: Interaction Terms Impact by Solvent Polarity")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    df_oof = pd.read_csv(OOF_EMBED_FILE)
    sol_raw, solv_raw = load_raw_feature_dataframes()
    
    y_train = df_train['LogS'].values
    y_test = df_test['LogS'].values
    
    # =========================================================================
    # Train WITH interaction terms
    # =========================================================================
    print("\n--- Training Model WITH Interaction Terms (24 features) ---")
    X_train_with = build_train_features(df_train, df_oof, sol_raw, solv_raw, include_interaction=True)
    X_test_with = build_test_features(df_test, sol_raw, solv_raw, include_interaction=True)
    print(f"Feature dimensions: {X_train_with.shape[1]}")
    
    model_with, selector_with = train_model(X_train_with, y_train)
    preds_with = model_with.predict(selector_with.transform(X_test_with))
    rmse_with = np.sqrt(mean_squared_error(y_test, preds_with))
    print(f"Overall Test RMSE: {rmse_with:.4f}")
    
    per_solvent_with = compute_per_solvent_rmse(df_test, y_test, preds_with)
    
    # =========================================================================
    # Train WITHOUT interaction terms
    # =========================================================================
    print("\n--- Training Model WITHOUT Interaction Terms ---")
    X_train_without = build_train_features(df_train, df_oof, sol_raw, solv_raw, include_interaction=False)
    X_test_without = build_test_features(df_test, sol_raw, solv_raw, include_interaction=False)
    print(f"Feature dimensions: {X_train_without.shape[1]}")
    
    model_without, selector_without = train_model(X_train_without, y_train)
    preds_without = model_without.predict(selector_without.transform(X_test_without))
    rmse_without = np.sqrt(mean_squared_error(y_test, preds_without))
    print(f"Overall Test RMSE: {rmse_without:.4f}")
    
    per_solvent_without = compute_per_solvent_rmse(df_test, y_test, preds_without)
    
    # =========================================================================
    # Compute Delta RMSE per solvent
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Delta RMSE per Solvent (WITHOUT - WITH interaction terms)")
    print("Positive = Interaction terms HELP (reduce RMSE)")
    print("=" * 70)
    
    results = []
    for solvent in per_solvent_with:
        if solvent in per_solvent_without:
            delta = per_solvent_without[solvent]['rmse'] - per_solvent_with[solvent]['rmse']
            results.append({
                'Solvent': solvent,
                'Name': per_solvent_with[solvent]['name'],
                'Polarity': per_solvent_with[solvent]['polarity'],
                'Count': per_solvent_with[solvent]['count'],
                'RMSE_With': per_solvent_with[solvent]['rmse'],
                'RMSE_Without': per_solvent_without[solvent]['rmse'],
                'Delta_RMSE': delta,
                'Pct_Improvement': (delta / per_solvent_without[solvent]['rmse']) * 100
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Delta_RMSE', ascending=False)
    
    # Print results by polarity class
    for polarity in ['polar-protic', 'polar-aprotic', 'non-polar']:
        subset = df_results[df_results['Polarity'] == polarity]
        if len(subset) > 0:
            print(f"\n--- {polarity.upper()} SOLVENTS ---")
            print(f"{'Solvent':<25} {'N':>5} {'RMSE_w/':>8} {'RMSE_w/o':>9} {'Δ_RMSE':>8} {'%Impr':>7}")
            print("-" * 65)
            for _, row in subset.iterrows():
                sign = "+" if row['Delta_RMSE'] > 0 else ""
                print(f"{row['Name']:<25} {row['Count']:>5} {row['RMSE_With']:>8.4f} "
                      f"{row['RMSE_Without']:>9.4f} {sign}{row['Delta_RMSE']:>7.4f} {row['Pct_Improvement']:>6.1f}%")
            
            avg_delta = subset['Delta_RMSE'].mean()
            avg_improvement = subset['Pct_Improvement'].mean()
            print(f"{'AVERAGE':<25} {'':>5} {'':>8} {'':>9} {avg_delta:>+8.4f} {avg_improvement:>6.1f}%")
    
    # Summary by polarity class
    print("\n" + "=" * 70)
    print("SUMMARY: Average Delta RMSE by Polarity Class")
    print("=" * 70)
    
    summary = df_results.groupby('Polarity').agg({
        'Delta_RMSE': ['mean', 'std', 'count'],
        'Pct_Improvement': 'mean'
    }).round(4)
    
    print(f"\n{'Polarity Class':<20} {'Mean ΔRMSE':>12} {'Std':>8} {'N':>5} {'Avg %Impr':>10}")
    print("-" * 55)
    for polarity in ['polar-protic', 'polar-aprotic', 'non-polar']:
        if polarity in summary.index:
            row = summary.loc[polarity]
            mean_delta = row[('Delta_RMSE', 'mean')]
            std_delta = row[('Delta_RMSE', 'std')]
            n = int(row[('Delta_RMSE', 'count')])
            pct = row[('Pct_Improvement', 'mean')]
            sign = "+" if mean_delta > 0 else ""
            print(f"{polarity:<20} {sign}{mean_delta:>11.4f} {std_delta:>8.4f} {n:>5} {pct:>9.1f}%")
    
    # Overall comparison
    print(f"\n--- OVERALL ---")
    print(f"With interaction terms:    RMSE = {rmse_with:.4f}")
    print(f"Without interaction terms: RMSE = {rmse_without:.4f}")
    print(f"Delta (benefit of I):      ΔRMSE = {rmse_without - rmse_with:+.4f}")
    
    # Save results
    output_file = "solvent_interaction_analysis_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    polar_delta = df_results[df_results['Polarity'].isin(['polar-protic', 'polar-aprotic'])]['Delta_RMSE'].mean()
    nonpolar_delta = df_results[df_results['Polarity'] == 'non-polar']['Delta_RMSE'].mean()
    
    print(f"Average ΔRMSE for polar solvents:     {polar_delta:+.4f}")
    print(f"Average ΔRMSE for non-polar solvents: {nonpolar_delta:+.4f}")
    
    if polar_delta > nonpolar_delta:
        print("\n✓ HYPOTHESIS SUPPORTED: Interaction terms provide more benefit for")
        print("  polar solvents (requiring explicit interaction modeling) than")
        print("  non-polar solvents (predictable from solute descriptors alone).")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED: Interaction terms benefit non-polar")
        print("  solvents more than polar solvents in this test set.")
    
    return df_results


if __name__ == "__main__":
    run_analysis()
