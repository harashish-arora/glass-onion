# correlation_analysis_extended.py
# Analyze correlations for COMPOSITIONAL and ENERGETIC features

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from featurizer import MoleculeFeaturizer

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from ablations import get_column_categories

TRAIN_PATH = "./data/train.csv"


def main():
    print("="*70)
    print("EXTENDED CORRELATION ANALYSIS")
    print("COMPOSITIONAL & ENERGETIC vs OTHER CATEGORIES")
    print("="*70)
    
    # Load and featurize
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    featurizer = MoleculeFeaturizer()
    X = featurizer.transform(train_df['SMILES'])
    
    columns = list(X.columns)
    categories = get_column_categories(columns)
    
    print(f"\nCategory sizes:")
    for cat, indices in categories.items():
        print(f"  {cat}: {len(indices)} features")
    
    # =========================================================================
    # 1. ENERGETIC (pred_Tm) correlations
    # =========================================================================
    print("\n" + "="*70)
    print("ENERGETIC (pred_Tm) CORRELATIONS")
    print("="*70)
    
    tm_idx = categories['ENERGETIC'][0]  # Should be just pred_Tm
    tm_values = X.iloc[:, tm_idx].values
    tm_col = columns[tm_idx]
    print(f"\nAnalyzing: {tm_col}")
    
    # Correlate with all other categories
    tm_corrs = []
    for cat in ['COMPOSITIONAL', 'TOPOLOGICAL', 'PHYSICOCHEMICAL']:
        for idx in categories[cat]:
            col = columns[idx]
            try:
                corr, _ = spearmanr(tm_values, X.iloc[:, idx].values)
                tm_corrs.append({
                    'Feature': col,
                    'Category': cat,
                    'Correlation': corr,
                    'Abs_Corr': abs(corr)
                })
            except:
                pass
    
    df_tm = pd.DataFrame(tm_corrs).sort_values('Abs_Corr', ascending=False)
    print("\nTop 20 features correlated with pred_Tm:")
    print(df_tm.head(20).to_string(index=False))
    
    high_tm = df_tm[df_tm['Abs_Corr'] >= 0.7]
    print(f"\nFeatures with |r| >= 0.7: {len(high_tm)}")
    if len(high_tm) > 0:
        print(high_tm.to_string(index=False))
    
    # =========================================================================
    # 2. COMPOSITIONAL correlations with other categories
    # =========================================================================
    print("\n" + "="*70)
    print("COMPOSITIONAL CORRELATIONS WITH OTHER CATEGORIES")
    print("="*70)
    
    comp_indices = categories['COMPOSITIONAL']
    topo_indices = categories['TOPOLOGICAL']
    phys_indices = categories['PHYSICOCHEMICAL']
    
    other_cols = [(columns[i], 'TOPOLOGICAL', i) for i in topo_indices]
    other_cols += [(columns[i], 'PHYSICOCHEMICAL', i) for i in phys_indices]
    
    # For each compositional feature, find best correlation
    comp_results = []
    for comp_idx in comp_indices:
        comp_col = columns[comp_idx]
        comp_values = X.iloc[:, comp_idx].values
        
        best_corr = 0
        best_other = None
        best_cat = None
        
        for other_col, other_cat, other_idx in other_cols:
            try:
                corr, _ = spearmanr(comp_values, X.iloc[:, other_idx].values)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_other = other_col
                    best_cat = other_cat
            except:
                pass
        
        comp_results.append({
            'Compositional_Feature': comp_col,
            'Best_Correlated': best_other,
            'Category': best_cat,
            'Correlation': best_corr,
            'Abs_Corr': abs(best_corr)
        })
    
    df_comp = pd.DataFrame(comp_results).sort_values('Abs_Corr', ascending=False)
    
    print("\nTop 30 COMPOSITIONAL features by correlation with TOPOLOGICAL/PHYSICOCHEMICAL:")
    print(df_comp.head(30).to_string(index=False))
    
    # Statistics
    high_comp = df_comp[df_comp['Abs_Corr'] >= 0.8]
    med_comp = df_comp[(df_comp['Abs_Corr'] >= 0.5) & (df_comp['Abs_Corr'] < 0.8)]
    low_comp = df_comp[df_comp['Abs_Corr'] < 0.5]
    
    print(f"\n--- COMPOSITIONAL CORRELATION SUMMARY ---")
    print(f"High (|r| >= 0.8): {len(high_comp)} features")
    print(f"Medium (0.5 <= |r| < 0.8): {len(med_comp)} features")
    print(f"Low (|r| < 0.5): {len(low_comp)} features")
    
    if len(high_comp) > 0:
        print("\n--- HIGHLY CORRELATED COMPOSITIONAL FEATURES (|r| >= 0.8) ---")
        print(high_comp.to_string(index=False))
    
    # Save results
    df_tm.to_csv("correlation_energetic.csv", index=False)
    df_comp.to_csv("correlation_compositional.csv", index=False)
    
    print("\n" + "="*70)
    print("Results saved to:")
    print("  - correlation_energetic.csv")
    print("  - correlation_compositional.csv")
    print("="*70)


if __name__ == "__main__":
    main()
