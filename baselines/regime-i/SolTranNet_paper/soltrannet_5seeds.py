# soltrannet_5seeds.py
"""
SolTranNet Baseline - 5 Seed Evaluation
Molecule Attention Transformer for Aqueous Solubility

Uses the SolTranNet transformer model from the paper.
Reports RMSE ± std and R² across 5 seeds with timing.

Run from inside the SolTranNet_paper directory:
    cd baselines/regime-i/SolTranNet_paper
    python soltrannet_5seeds.py
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import time
import pickle

# Add src to path for imports
sys.path.append('src')

# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]

DATASETS = {
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}

# SolTranNet hyperparameters (paper defaults)
EPOCHS = 100        # Can increase for better results
DYNAMIC_STOP = 10   # Early stopping: stop if no improvement for N epochs
LOSS = "huber"
OPTIMIZER = "sgd"
LEARNING_RATE = 1e-4


def run_soltrannet_training(train_path, test_path, seed, output_dir):
    """Train SolTranNet model using train.py with subprocess"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python", "train.py",
        "--trainfile", train_path,
        "--testfile", test_path,
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--dynamic", str(DYNAMIC_STOP),
        "--loss", LOSS,
        "--optimizer", OPTIMIZER,
        "--lr", str(LEARNING_RATE),
        "--datadir", output_dir,
        "--twod",       # Use 2D coords - faster featurization
        "--savemodel"
    ]
    
    print(f"    Training seed {seed}...")
    print(f"    Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def extract_results_from_pickle(output_dir, seed):
    """Extract RMSE and R2 from the saved pickle file"""
    
    # Find the testdic pickle file
    for f in os.listdir(output_dir):
        if f.endswith('_testdic.pi') and f'seed{seed}' in f:
            pickle_path = os.path.join(output_dir, f)
            with open(pickle_path, 'rb') as pf:
                testdic = pickle.load(pf)
            return testdic['RMSE'], testdic['R2']
    
    return None, None


def evaluate_dataset(name, train_path, test_path):
    """Evaluate on a dataset across 5 seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    # Read data info
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    output_dir = f"soltrannet_outputs/{name.lower()}"
    
    results = []
    times = []
    
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        start_time = time.time()
        
        success = run_soltrannet_training(train_path, test_path, seed, output_dir)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if success:
            # Try to extract results from pickle
            rmse, r2 = extract_results_from_pickle(output_dir, seed)
            if rmse is not None:
                results.append((rmse, r2))
                print(f"    RMSE: {rmse:.4f}, R²: {r2:.4f}")
            else:
                print(f"    Could not extract results from pickle")
        else:
            print(f"    Training failed!")
        
        print(f"    Time: {elapsed:.1f}s")
    
    if not results:
        return None
    
    return {
        "name": name,
        "rmse_mean": np.mean([r[0] for r in results]),
        "rmse_std": np.std([r[0] for r in results]),
        "r2_mean": np.mean([r[1] for r in results]),
        "r2_std": np.std([r[1] for r in results]),
        "avg_time": np.mean(times),
        "total_time": np.sum(times)
    }


def main():
    print("=" * 60)
    print("SolTranNet Baseline - 5 Seed Evaluation")
    print("Molecule Attention Transformer")
    print("=" * 60)
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU (will be slow!)")
    
    all_results = []
    total_start = time.time()
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        if result:
            all_results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - SolTranNet Baseline")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s | Total: {r['total_time']:.1f}s")
    
    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
