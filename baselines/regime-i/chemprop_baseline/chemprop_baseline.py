#!/usr/bin/env python3
"""
Chemprop 2.x Baseline for Regime-I (Aqueous Solubility)
Runs 5-seed training and reports mean ± std for all datasets.
Uses Chemprop 2.x CLI via subprocess.
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
SEEDS = [42, 101, 123, 456, 789]
EPOCHS = 60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../regime-i/all_datasets")

# Dataset configurations
DATASETS = {
    "ESOL": {
        "train": os.path.join(DATASETS_DIR, "esol/train.csv"),
        "test": os.path.join(DATASETS_DIR, "esol/test.csv"),
        "smiles_col": "SMILES",
        "target_col": "LogS"
    },
    "AqSolDB": {
        "train": os.path.join(DATASETS_DIR, "aqsoldb/train.csv"),
        "test": os.path.join(DATASETS_DIR, "aqsoldb/test.csv"),
        "smiles_col": "SMILES",
        "target_col": "LogS"
    },
    "SC2": {
        "train": os.path.join(DATASETS_DIR, "sc2/train.csv"),
        "test": os.path.join(DATASETS_DIR, "sc2/test.csv"),
        "smiles_col": "SMILES",
        "target_col": "LogS"
    }
}


def run_chemprop_seed(train_path, test_path, smiles_col, target_col, seed, work_dir):
    """Train and evaluate Chemprop for a single seed using CLI."""
    checkpoint_dir = os.path.join(work_dir, f"checkpoint_seed_{seed}")
    pred_path = os.path.join(work_dir, f"predictions_seed_{seed}.csv")
    
    # Chemprop 2.x train command
    train_cmd = [
        "chemprop", "train",
        "--data-path", train_path,
        "--task-type", "regression",
        "--smiles-columns", smiles_col,
        "--target-columns", target_col,
        "--output-dir", checkpoint_dir,
        "--epochs", str(EPOCHS),
        "--accelerator", "cpu",
        "--num-workers", "8",
        "--pytorch-seed", str(seed)
    ]
    
    print(f"    Training with seed {seed}...", end=" ", flush=True)
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"TRAIN FAILED")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return None, None
    
    # Chemprop 2.x predict command
    pred_cmd = [
        "chemprop", "predict",
        "--test-path", test_path,
        "--model-paths", checkpoint_dir,
        "--preds-path", pred_path,
        "--accelerator", "cpu",
        "--num-workers", "8"
    ]
    
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PREDICT FAILED")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return None, None
    
    # Load predictions and calculate metrics
    try:
        test_df = pd.read_csv(test_path)
        pred_df = pd.read_csv(pred_path)
        
        # Get the prediction column
        pred_cols = [c for c in pred_df.columns if c != smiles_col]
        pred_col = pred_cols[0] if pred_cols else pred_df.columns[-1]
        
        y_true = test_df[target_col].values
        y_pred = pred_df[pred_col].values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"RMSE={rmse:.4f}, R²={r2:.4f}")
        return rmse, r2
    except Exception as e:
        print(f"METRICS FAILED: {e}")
        return None, None


def run_dataset(name, config):
    """Run 5-seed training for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    train_path = os.path.abspath(config["train"])
    test_path = os.path.abspath(config["test"])
    smiles_col = config["smiles_col"]
    target_col = config["target_col"]
    
    if not os.path.exists(train_path):
        print(f"  ERROR: Train file not found: {train_path}")
        return None
    if not os.path.exists(test_path):
        print(f"  ERROR: Test file not found: {test_path}")
        return None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    rmse_scores = []
    r2_scores = []
    
    with tempfile.TemporaryDirectory() as work_dir:
        for seed in SEEDS:
            rmse, r2 = run_chemprop_seed(
                train_path, test_path, smiles_col, target_col, seed, work_dir
            )
            if rmse is not None:
                rmse_scores.append(rmse)
                r2_scores.append(r2)
    
    if len(rmse_scores) == 0:
        return None
    
    results = {
        "dataset": name,
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "n_seeds": len(rmse_scores)
    }
    
    print(f"\n  RESULTS (n={len(rmse_scores)} seeds):")
    print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    
    return results


def main():
    print("="*60)
    print("CHEMPROP 2.x BASELINE FOR REGIME-I")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)
    
    all_results = []
    
    for name, config in DATASETS.items():
        result = run_dataset(name, config)
        if result:
            all_results.append(result)
    
    # Summary table
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'RMSE':>20} {'R²':>20}")
    print("-"*55)
    for r in all_results:
        rmse_str = f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
        r2_str = f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}"
        print(f"{r['dataset']:<15} {rmse_str:>20} {r2_str:>20}")
    
    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(BASE_DIR, "chemprop_baseline_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
