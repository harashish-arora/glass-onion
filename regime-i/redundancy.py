# feature_cleanup_analysis.py
# Remove Morgan/MACCS, identify redundancies, and document physical significance

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

TRAIN_PATH = "./data/train.csv"


def get_feature_categories(columns):
    """Categorize features after removing Morgan and MACCS."""
    categories = {
        'COMPOSITIONAL': [],
        'TOPOLOGICAL': [],
        'ENERGETIC': [],
        'PHYSICOCHEMICAL': []
    }
    
    # Count-based features
    count_features = {
        'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
        'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
        'RingCount', 'HeavyAtomCount', 'total_atoms', 'MolWt', 'ExactMolWt', 'HeavyAtomMolWt'
    }
    
    # Topological features (excluding Morgan and MACCS)
    topological_features = {
        'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
        'BalabanJ', 'BertzCT', 'Kappa1', 'Kappa2', 'HallKierAlpha',
        'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v'
    }
    
    # Property-based descriptors
    property_descriptors = {
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
        'FractionCSP3', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex',
        'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
        'MinPartialCharge', 'MolLogP',
        'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
        'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
        'TPSA', 'qed'
    }
    
    for i, col in enumerate(columns):
        # Skip Morgan and MACCS
        if col.startswith('Morgan_') or col.startswith('MACCS_'):
            continue
            
        # COMPOSITIONAL
        if (col.startswith('num_') or col.startswith('fr_') or col in count_features):
            categories['COMPOSITIONAL'].append((i, col))
        # TOPOLOGICAL
        elif (col.startswith('mose_') or col.startswith('AUTOCORR2D_') or col in topological_features):
            categories['TOPOLOGICAL'].append((i, col))
        # ENERGETIC
        elif col == 'pred_Tm' or col.startswith('abraham_'):
            categories['ENERGETIC'].append((i, col))
        # PHYSICOCHEMICAL
        elif col in property_descriptors:
            categories['PHYSICOCHEMICAL'].append((i, col))
    
    return categories


def find_redundancies(X, categories, threshold=0.90):
    """Find highly correlated features within and across categories."""
    print("\n" + "="*80)
    print("FINDING REDUNDANT FEATURES (|r| >= 0.90)")
    print("="*80)
    
    redundancies = []
    
    # Get all features as flat list
    all_features = []
    for cat_name, features in categories.items():
        for idx, name in features:
            all_features.append((idx, name, cat_name))
    
    # Check all pairs
    for i in range(len(all_features)):
        idx_i, name_i, cat_i = all_features[i]
        for j in range(i + 1, len(all_features)):
            idx_j, name_j, cat_j = all_features[j]
            
            try:
                corr, _ = spearmanr(X.iloc[:, idx_i], X.iloc[:, idx_j])
                if abs(corr) >= threshold:
                    redundancies.append({
                        'Feature_1': name_i,
                        'Category_1': cat_i,
                        'Feature_2': name_j,
                        'Category_2': cat_j,
                        'Correlation': corr,
                        'Abs_Correlation': abs(corr)
                    })
            except:
                pass
    
    df_redundancies = pd.DataFrame(redundancies)
    if len(df_redundancies) > 0:
        df_redundancies = df_redundancies.sort_values('Abs_Correlation', ascending=False)
    
    return df_redundancies


def select_features_to_remove(redundancies, categories):
    """Select which features to remove based on redundancy and category priority."""
    print("\n" + "="*80)
    print("SELECTING FEATURES TO REMOVE")
    print("="*80)
    print("\nPriority: Remove from PHYSICOCHEMICAL > COMPOSITIONAL > TOPOLOGICAL > ENERGETIC")
    
    to_remove = set()
    category_priority = {'PHYSICOCHEMICAL': 4, 'COMPOSITIONAL': 3, 'TOPOLOGICAL': 2, 'ENERGETIC': 1}
    
    for _, row in redundancies.iterrows():
        feat1, cat1 = row['Feature_1'], row['Category_1']
        feat2, cat2 = row['Feature_2'], row['Category_2']
        
        # Skip if both already marked for removal
        if feat1 in to_remove and feat2 in to_remove:
            continue
        
        # Skip if one is already marked
        if feat1 in to_remove:
            continue
        if feat2 in to_remove:
            continue
        
        # Remove based on priority
        if category_priority[cat1] > category_priority[cat2]:
            to_remove.add(feat1)
            print(f"  Remove: {feat1} ({cat1}) - corr {row['Abs_Correlation']:.3f} with {feat2} ({cat2})")
        elif category_priority[cat2] > category_priority[cat1]:
            to_remove.add(feat2)
            print(f"  Remove: {feat2} ({cat2}) - corr {row['Abs_Correlation']:.3f} with {feat1} ({cat1})")
        else:
            # Same category - keep first alphabetically (arbitrary but consistent)
            if feat1 < feat2:
                to_remove.add(feat2)
                print(f"  Remove: {feat2} ({cat2}) - corr {row['Abs_Correlation']:.3f} with {feat1} ({cat1})")
            else:
                to_remove.add(feat1)
                print(f"  Remove: {feat1} ({cat1}) - corr {row['Abs_Correlation']:.3f} with {feat2} ({cat2})")
    
    return to_remove


def get_physical_significance():
    """Document physical significance of feature groups."""
    return {
        'COMPOSITIONAL': {
            'description': 'Atom counts, functional group counts, and molecular weight descriptors',
            'features': {
                'num_*': 'Count of specific atoms (C, O, N, Cl, S, F, P, Br, I, etc.)',
                'fr_*': 'Functional group counts (from RDKit fragment descriptors)',
                'total_atoms': 'Total number of atoms in molecule',
                'HeavyAtomCount': 'Number of non-hydrogen atoms',
                'MolWt': 'Molecular weight (sum of atomic weights)',
                'ExactMolWt': 'Exact molecular weight using isotopic masses',
                'HeavyAtomMolWt': 'Molecular weight excluding hydrogen',
                'NumValenceElectrons': 'Total valence electrons',
                'NumHDonors': 'Number of hydrogen bond donors',
                'NumHAcceptors': 'Number of hydrogen bond acceptors',
                'NumHeteroatoms': 'Number of heteroatoms (non-C, non-H)',
                'NumRotatableBonds': 'Number of rotatable bonds (flexibility)',
                'NumAromaticRings': 'Number of aromatic ring systems',
                'NumAliphaticRings': 'Number of non-aromatic rings',
                'RingCount': 'Total number of rings',
            }
        },
        'TOPOLOGICAL': {
            'description': 'Graph-based descriptors capturing molecular connectivity and shape',
            'features': {
                'AUTOCORR2D_*': 'Autocorrelation descriptors - encode spatial distribution of properties',
                'mose_*': 'Molecular substructure motifs (cycles, branching, paths)',
                'Chi*': 'Kier-Hall molecular connectivity indices - branching and cyclicity',
                'Kappa1/Kappa2': 'Kappa shape indices - molecular shape (linear vs spherical)',
                'BalabanJ': 'Balaban J index - average distance sum connectivity',
                'BertzCT': 'Bertz complexity index - molecular complexity',
                'HallKierAlpha': 'Hall-Kier alpha value - molecular flexibility',
                'FpDensityMorgan*': 'Fingerprint density metrics',
            }
        },
        'ENERGETIC': {
            'description': 'Thermodynamic and physicochemical property proxies',
            'features': {
                'pred_Tm': 'Predicted melting temperature (Joback method)',
                'abraham_A': 'Abraham H-bond acidity (proton donor strength)',
                'abraham_B': 'Abraham H-bond basicity (proton acceptor strength)',
                'abraham_S': 'Abraham dipolarity/polarizability',
                'abraham_E': 'Abraham excess molar refraction (electron lone pairs)',
                'abraham_V': 'Abraham McGowan volume (molecular size)',
            }
        },
        'PHYSICOCHEMICAL': {
            'description': 'Electronic properties, charge distribution, and drug-likeness',
            'features': {
                'EState_VSA*': 'Electrotopological state indices - electronic distribution',
                'PEOE_VSA*': 'Partial charge descriptors (PEOE = Partial Equalization of Orbital Electronegativities)',
                'MaxPartialCharge': 'Maximum partial charge on any atom',
                'MinPartialCharge': 'Minimum partial charge on any atom',
                'MaxEStateIndex': 'Maximum E-state value',
                'MinEStateIndex': 'Minimum E-state value',
                'MolLogP': 'Octanol-water partition coefficient (lipophilicity)',
                'TPSA': 'Topological polar surface area (membrane permeability)',
                'FractionCSP3': 'Fraction of sp3 hybridized carbons (saturation)',
                'qed': 'Quantitative Estimate of Drug-likeness',
            }
        }
    }


def main():
    print("="*80)
    print("FEATURE CLEANUP AND ANALYSIS")
    print("Step 1: Remove Morgan fingerprints and MACCS keys")
    print("Step 2: Identify and remove redundant features")
    print("Step 3: Document remaining features")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Featurize
    print("Generating features...")
    featurizer = MoleculeFeaturizer()
    X = featurizer.transform(train_df['SMILES'])
    
    original_columns = list(X.columns)
    print(f"\nOriginal features: {len(original_columns)}")
    
    # Step 1: Remove Morgan and MACCS
    print("\n" + "="*80)
    print("STEP 1: REMOVING MORGAN AND MACCS FINGERPRINTS")
    print("="*80)
    
    non_fingerprint_cols = [col for col in original_columns 
                           if not col.startswith('Morgan_') and not col.startswith('MACCS_')]
    
    morgan_count = sum(1 for col in original_columns if col.startswith('Morgan_'))
    maccs_count = sum(1 for col in original_columns if col.startswith('MACCS_'))
    
    print(f"\nRemoved:")
    print(f"  Morgan fingerprints: {morgan_count}")
    print(f"  MACCS keys: {maccs_count}")
    print(f"  Total removed: {morgan_count + maccs_count}")
    print(f"\nRemaining features: {len(non_fingerprint_cols)}")
    
    X_filtered = X[non_fingerprint_cols]
    
    # Categorize remaining features
    categories = get_feature_categories(non_fingerprint_cols)
    
    print("\nFeature breakdown by category:")
    for cat_name, features in categories.items():
        print(f"  {cat_name}: {len(features)} features")
    
    # Step 2: Find redundancies
    redundancies = find_redundancies(X_filtered, categories, threshold=0.90)
    
    print(f"\nFound {len(redundancies)} redundant feature pairs")
    
    if len(redundancies) > 0:
        print("\nTop 20 redundant pairs:")
        print(redundancies.head(20).to_string(index=False))
        redundancies.to_csv('feature_redundancies.csv', index=False)
        print("\nFull list saved to: feature_redundancies.csv")
        
        # Select features to remove
        to_remove = select_features_to_remove(redundancies, categories)
        
        print(f"\nTotal features to remove: {len(to_remove)}")
        
        # Create final feature list
        final_features = [col for col in non_fingerprint_cols if col not in to_remove]
    else:
        print("\nNo significant redundancies found!")
        final_features = non_fingerprint_cols
        to_remove = set()
    
    # Step 3: Document final features
    print("\n" + "="*80)
    print("FINAL FEATURE SET")
    print("="*80)
    
    final_categories = get_feature_categories(final_features)
    
    print("\nFinal feature counts:")
    for cat_name, features in final_categories.items():
        print(f"  {cat_name}: {len(features)} features")
    
    print(f"\nTotal features: {len(final_features)}")
    print(f"Reduction: {len(original_columns)} → {len(final_features)} ({(1-len(final_features)/len(original_columns))*100:.1f}% reduction)")
    
    # Save results
    pd.DataFrame({'Feature': final_features}).to_csv('final_features.csv', index=False)
    pd.DataFrame({'Feature': list(to_remove)}).to_csv('removed_features.csv', index=False)
    
    # Create detailed documentation
    significance = get_physical_significance()
    
    print("\n" + "="*80)
    print("PHYSICAL SIGNIFICANCE OF FEATURE CATEGORIES")
    print("="*80)
    
    for cat_name, features in final_categories.items():
        if len(features) == 0:
            continue
            
        print(f"\n{cat_name} ({len(features)} features)")
        print("-" * 80)
        print(f"Description: {significance[cat_name]['description']}\n")
        
        print("Features:")
        for idx, feat_name in features:
            print(f"  • {feat_name}")
        
        print("\nPhysical Meaning:")
        for key, meaning in significance[cat_name]['features'].items():
            print(f"  {key}: {meaning}")
    
    print("\n" + "="*80)
    print("FILES SAVED")
    print("="*80)
    print("  - final_features.csv: List of features to keep")
    print("  - removed_features.csv: List of removed redundant features")
    print("  - feature_redundancies.csv: Full redundancy analysis")
    print("="*80)


if __name__ == "__main__":
    main()