#!/usr/bin/env python
"""
Split BigSolDB cleaned data into train/test sets.
Usage: python split_bigsol.py --input bigsol_clean.csv
"""
import os
import argparse
import pandas as pd

# config
DATA_DIR = "."
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TOP_N_SOLVENTS = 19

def main():
    parser = argparse.ArgumentParser(description="Split cleaned BigSolDB into train/test sets")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned BigSolDB CSV (bigsol_clean.csv)")
    args = parser.parse_args()
    
    print(f"Loading cleaned data from {args.input}...")
    clean_df = pd.read_csv(args.input)
    
    # Verify expected columns
    expected_cols = ['Solute', 'Solvent', 'Temperature', 'LogS']
    if not all(col in clean_df.columns for col in expected_cols):
        print(f"Error: Expected columns {expected_cols}, but found {list(clean_df.columns)}")
        return
    
    print(f"Loaded {len(clean_df)} data points")
    
    # Perform the train/test split
    print(f"\nPerforming Train/Test Split...")
    
    # Identify Top N Solvents
    solvent_counts = clean_df['Solvent'].value_counts()
    train_solvents_list = solvent_counts.head(TOP_N_SOLVENTS).index.tolist()
    
    print(f"Top {TOP_N_SOLVENTS} Solvents identified for Training Set:")
    for i, solvent in enumerate(train_solvents_list, 1):
        count = solvent_counts[solvent]
        print(f"  {i}. {solvent[:50]:50s} ({count:5d} points)")
    
    # Create Split Masks
    train_mask = clean_df['Solvent'].isin(train_solvents_list)
    train_df = clean_df[train_mask].copy()
    test_df = clean_df[~train_mask].copy()
    
    # Save Split Files
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    
    print(f"\n✓ Train Set Saved: {len(train_df)} rows ({TRAIN_PATH})")
    print(f"✓ Test Set Saved:  {len(test_df)} rows ({TEST_PATH})")
    print(f"\nSplit ratio: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% train / {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test")

if __name__ == "__main__":
    main()
