# feature_correlation_pca.py
# Feature correlation analysis with PCA on bulk feature groups

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from featurizer import MoleculeFeaturizer

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Config
TRAIN_PATH = "./data/train.csv"

# ============================================================================
# BULK FEATURE GROUP DEFINITIONS
# ============================================================================

BULK_GROUPS = {
    'Morgan': 'Morgan_',
    'MACCS': 'MACCS_',
    'Abraham': 'abraham_',
    'AUTOCORR2D': 'AUTOCORR2D_',
    'MOSE': 'mose_'
}


def get_bulk_feature_indices(columns):
    """Get indices for each bulk feature group."""
    groups = {}
    for group_name, prefix in BULK_GROUPS.items():
        groups[group_name] = [i for i, col in enumerate(columns) if col.startswith(prefix)]
    return groups


def compute_linear_correlations(X, y, columns):
    """Compute Pearson and Spearman correlations for each feature."""
    results = []
    for i, col in enumerate(columns):
        try:
            pearson_r, pearson_p = pearsonr(X.iloc[:, i], y)
            spearman_r, spearman_p = spearmanr(X.iloc[:, i], y)
        except:
            pearson_r, pearson_p = 0, 1
            spearman_r, spearman_p = 0, 1
        
        results.append({
            'Feature': col,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Abs_Pearson': abs(pearson_r),
            'Abs_Spearman': abs(spearman_r)
        })
    return pd.DataFrame(results)


def compute_quadratic_correlation(X_col, y):
    """Compute quadratic (polynomial degree 2) correlation with target."""
    try:
        X_col = X_col.values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_col)
        
        reg = LinearRegression()
        reg.fit(X_poly, y)
        y_pred = reg.predict(X_poly)
        return r2_score(y, y_pred)
    except:
        return 0.0


def compute_all_correlations(X, y, columns):
    """Compute linear and quadratic correlations for all features."""
    print("Computing linear correlations...")
    linear_df = compute_linear_correlations(X, y, columns)
    
    print("Computing quadratic correlations...")
    quad_r2 = []
    for i, col in enumerate(columns):
        r2 = compute_quadratic_correlation(X.iloc[:, i], y)
        quad_r2.append(r2)
    
    linear_df['Quadratic_R2'] = quad_r2
    return linear_df


def analyze_bulk_pca(X, y, columns, groups):
    """Apply PCA to each bulk feature group and analyze correlation with target."""
    print("\n" + "="*70)
    print("PCA ANALYSIS FOR BULK FEATURE GROUPS")
    print("="*70)
    
    results = []
    
    for group_name, indices in groups.items():
        if len(indices) == 0:
            print(f"\n{group_name}: No features found")
            continue
        
        print(f"\n{group_name}: {len(indices)} features")
        
        # Extract and scale features
        X_group = X.iloc[:, indices].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_group)
        
        # Determine number of PCA components (up to 10 or max features)
        n_components = min(10, len(indices), X_scaled.shape[0])
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"  Total variance explained by {n_components} PCs: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        # Analyze each PC's correlation with target
        for pc_idx in range(n_components):
            pc_values = X_pca[:, pc_idx]
            pearson_r, _ = pearsonr(pc_values, y)
            spearman_r, _ = spearmanr(pc_values, y)
            
            # Quadratic correlation
            quad_r2 = compute_quadratic_correlation(pd.Series(pc_values), y)
            
            results.append({
                'Group': group_name,
                'Component': f'PC{pc_idx+1}',
                'Variance_Explained_%': pca.explained_variance_ratio_[pc_idx] * 100,
                'Cumulative_Variance_%': sum(pca.explained_variance_ratio_[:pc_idx+1]) * 100,
                'Pearson_r_with_LogS': pearson_r,
                'Spearman_r_with_LogS': spearman_r,
                'Quadratic_R2_with_LogS': quad_r2,
                'Abs_Pearson': abs(pearson_r)
            })
            
            if pc_idx < 3:  # Show first 3 PCs
                print(f"    PC{pc_idx+1}: Var={pca.explained_variance_ratio_[pc_idx]*100:.2f}%, "
                      f"Pearson={pearson_r:.4f}, Quadratic R²={quad_r2:.4f}")
    
    return pd.DataFrame(results)


def main():
    print("="*70)
    print("FEATURE CORRELATION ANALYSIS WITH PCA")
    print("="*70)
    
    # Load and featurize
    print("\nLoading data and generating features...")
    train_df = pd.read_csv(TRAIN_PATH)
    featurizer = MoleculeFeaturizer()
    X = featurizer.transform(train_df['SMILES'])
    y = train_df['LogS']
    
    columns = list(X.columns)
    n_features = len(columns)
    print(f"Total features: {n_features}")
    
    # Get bulk feature groups
    groups = get_bulk_feature_indices(columns)
    print("\nBulk Feature Groups:")
    for name, indices in groups.items():
        print(f"  {name}: {len(indices)} features")
    
    # =========================================================================
    # 1. Linear and Quadratic Correlations for ALL features
    # =========================================================================
    print("\n" + "="*70)
    print("COMPUTING ALL FEATURE CORRELATIONS")
    print("="*70)
    
    corr_df = compute_all_correlations(X, y, columns)
    corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
    
    print("\nTop 30 features by Pearson correlation with LogS:")
    print(corr_df.head(30).to_string(index=False))
    
    # Summary by group
    print("\n" + "="*70)
    print("CORRELATION SUMMARY BY FEATURE GROUP")
    print("="*70)
    
    for group_name, indices in groups.items():
        if len(indices) == 0:
            continue
        group_cols = [columns[i] for i in indices]
        group_data = corr_df[corr_df['Feature'].isin(group_cols)]
        
        mean_pearson = group_data['Abs_Pearson'].mean()
        max_pearson = group_data['Abs_Pearson'].max()
        mean_quad = group_data['Quadratic_R2'].mean()
        max_quad = group_data['Quadratic_R2'].max()
        
        print(f"\n{group_name} ({len(indices)} features):")
        print(f"  Mean |Pearson|: {mean_pearson:.4f}, Max: {max_pearson:.4f}")
        print(f"  Mean Quadratic R²: {mean_quad:.4f}, Max: {max_quad:.4f}")
    
    # =========================================================================
    # 2. PCA Analysis for Bulk Groups
    # =========================================================================
    pca_df = analyze_bulk_pca(X, y, columns, groups)
    
    # =========================================================================
    # 3. Save Results
    # =========================================================================
    corr_df.to_csv("feature_correlation_pca_results.csv", index=False)
    pca_df.to_csv("bulk_pca_analysis.csv", index=False)
    
    print("\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print("  - feature_correlation_pca_results.csv: All feature correlations")
    print("  - bulk_pca_analysis.csv: PCA analysis for bulk groups")
    
    # =========================================================================
    # 4. Key Insights Summary
    # =========================================================================
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Best PC per group
    print("\nBest PCA component per group (by |Pearson| with LogS):")
    for group_name in groups.keys():
        group_pca = pca_df[pca_df['Group'] == group_name]
        if len(group_pca) > 0:
            best = group_pca.loc[group_pca['Abs_Pearson'].idxmax()]
            print(f"  {group_name}: {best['Component']} (|r|={best['Abs_Pearson']:.4f}, "
                  f"Var={best['Variance_Explained_%']:.2f}%)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
