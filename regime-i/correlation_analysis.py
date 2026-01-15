# correlation_analysis.py
# Analyze correlations between PHYSICOCHEMICAL features and other categories

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

# Import category function from ablations.py
from ablations import get_column_categories

TRAIN_PATH = "./data/train.csv"


def main():
    print("="*70)
    print("CORRELATION ANALYSIS: PHYSICOCHEMICAL vs OTHER CATEGORIES")
    print("="*70)
    
    # Load and featurize
    print("\nLoading data and generating features...")
    train_df = pd.read_csv(TRAIN_PATH)
    featurizer = MoleculeFeaturizer()
    X = featurizer.transform(train_df['SMILES'])
    
    columns = list(X.columns)
    categories = get_column_categories(columns)
    
    print(f"\nCategory sizes:")
    for cat, indices in categories.items():
        print(f"  {cat}: {len(indices)} features")
    
    # Get feature names by category
    phys_cols = [columns[i] for i in categories['PHYSICOCHEMICAL']]
    comp_cols = [columns[i] for i in categories['COMPOSITIONAL']]
    topo_cols = [columns[i] for i in categories['TOPOLOGICAL']]
    ener_cols = [columns[i] for i in categories['ENERGETIC']]
    
    other_cols = comp_cols + topo_cols + ener_cols
    
    print(f"\nAnalyzing correlations for {len(phys_cols)} PHYSICOCHEMICAL features")
    print(f"Against {len(other_cols)} features in other categories...")
    
    # For each physicochemical feature, find highest correlation with other categories
    results = []
    
    for phys_col in phys_cols:
        phys_values = X[phys_col].values
        
        best_corr = 0
        best_other = None
        best_category = None
        
        # Check against COMPOSITIONAL
        for other_col in comp_cols:
            try:
                corr, _ = spearmanr(phys_values, X[other_col].values)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_other = other_col
                    best_category = 'COMPOSITIONAL'
            except:
                pass
        
        # Check against TOPOLOGICAL
        for other_col in topo_cols:
            try:
                corr, _ = spearmanr(phys_values, X[other_col].values)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_other = other_col
                    best_category = 'TOPOLOGICAL'
            except:
                pass
        
        # Check against ENERGETIC
        for other_col in ener_cols:
            try:
                corr, _ = spearmanr(phys_values, X[other_col].values)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_other = other_col
                    best_category = 'ENERGETIC'
            except:
                pass
        
        results.append({
            'PhysChem_Feature': phys_col,
            'Best_Correlated_Feature': best_other,
            'Best_Category': best_category,
            'Spearman_Correlation': best_corr,
            'Abs_Correlation': abs(best_corr)
        })
    
    # Sort by absolute correlation
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Abs_Correlation', ascending=False)
    
    # Save full results
    df_results.to_csv("correlation_analysis.csv", index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("TOP 20 PHYSICOCHEMICAL FEATURES BY CORRELATION WITH OTHER CATEGORIES")
    print("="*70)
    print(df_results.head(20).to_string(index=False))
    
    # Statistics
    print("\n" + "="*70)
    print("CORRELATION STATISTICS")
    print("="*70)
    
    high_corr = df_results[df_results['Abs_Correlation'] >= 0.8]
    med_corr = df_results[(df_results['Abs_Correlation'] >= 0.5) & (df_results['Abs_Correlation'] < 0.8)]
    low_corr = df_results[df_results['Abs_Correlation'] < 0.5]
    
    print(f"\nHigh correlation (|r| >= 0.8): {len(high_corr)} features")
    print(f"Medium correlation (0.5 <= |r| < 0.8): {len(med_corr)} features")
    print(f"Low correlation (|r| < 0.5): {len(low_corr)} features")
    
    print("\n" + "="*70)
    print("HIGH CORRELATION FEATURES (|r| >= 0.8)")
    print("="*70)
    if len(high_corr) > 0:
        print(high_corr.to_string(index=False))
    else:
        print("None")
    
    print("\n" + "="*70)
    print("FEATURES WITH LOW CORRELATION (|r| < 0.5) - UNIQUE TO PHYSICOCHEMICAL")
    print("="*70)
    print(f"\n{len(low_corr)} features have low correlation with other categories:")
    for _, row in low_corr.iterrows():
        print(f"  {row['PhysChem_Feature']}: |r|={row['Abs_Correlation']:.3f}")
    
    # Category breakdown
    print("\n" + "="*70)
    print("BEST CORRELATIONS BY SOURCE CATEGORY")
    print("="*70)
    for cat in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC']:
        cat_data = df_results[df_results['Best_Category'] == cat]
        if len(cat_data) > 0:
            mean_corr = cat_data['Abs_Correlation'].mean()
            max_corr = cat_data['Abs_Correlation'].max()
            print(f"\n{cat}:")
            print(f"  {len(cat_data)} PhysChem features best correlate with {cat}")
            print(f"  Mean |r|: {mean_corr:.3f}, Max |r|: {max_corr:.3f}")
    
    print("\n" + "="*70)
    print(f"Full results saved to correlation_analysis.csv")
    print("="*70)


if __name__ == "__main__":
    main()
