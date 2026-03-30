import os
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from fastprop.data import fastpropDataLoader
from fastprop.defaults import ALL_2D
from model_architecture import SolubilityDataset, fastpropSolubility

# Configuration
SEEDS = [42, 101, 123, 456, 789]
NUM_REPLICATES = 4
MODEL_DIR = Path("model")
TEST_FILE = Path("data/leeds_test_features.csv")
SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]

def load_ensemble(seed):
    seed_dir = MODEL_DIR / f"seed_{seed}"
    models = []
    for rep_idx in range(1, NUM_REPLICATES + 1):
        model_path = seed_dir / f"replicate_{rep_idx}.pt"
        # Using weights_only=False because fastpropSolubility might have custom objects
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        models.append(model)
    return models

def run_benchmark():
    print(f"Loading test set from {TEST_FILE}...")
    df_test = pd.read_csv(TEST_FILE)

    # Prepare tensors once
    test_solute_arr = torch.tensor(df_test[SOLUTE_COLUMNS].values, dtype=torch.float32)
    test_solvent_arr = torch.tensor(df_test[SOLVENT_COLUMNS].values, dtype=torch.float32)
    test_temp_arr = torch.tensor(df_test["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    test_logS_arr = torch.tensor(df_test["logS"].values.reshape(-1, 1), dtype=torch.float32)

    test_ds = SolubilityDataset(
        test_solute_arr,
        test_solvent_arr,
        test_temp_arr,
        test_logS_arr,
        torch.zeros_like(test_logS_arr),
    )
    # Using a larger batch size for inference efficiency
    test_loader = fastpropDataLoader(test_ds, batch_size=1024, shuffle=False)

    results = []

    for seed in SEEDS:
        print(f"\nBenchmarking Seed {seed}...")
        models = load_ensemble(seed)

        # Warmup (optional but good for timing)
        print("  Warmup...")
        with torch.no_grad():
            for batch in test_loader:
                for model in models:
                    _ = model.predict_step(batch)
                break # Just one batch for warmup

        print("  Running inference...")
        start_time = time.time()

        all_preds = []
        with torch.no_grad():
            # We want to time the ensemble prediction
            # Each predict_step handles scaling and unscaling

            # To be fair, we should probably aggregate per-replicate
            # but let's do it efficiently

            ensemble_preds = np.zeros((len(df_test), 1))

            for model in models:
                model_preds = []
                for batch in test_loader:
                    preds = model.predict_step(batch)
                    model_preds.append(preds.numpy())
                ensemble_preds += np.vstack(model_preds)

            ensemble_preds /= len(models)

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = len(df_test) / elapsed

        print(f"  Done in {elapsed:.4f}s ({throughput:.1f} samples/s)")

        results.append({
            "seed": seed,
            "inference_time_seconds": elapsed,
            "samples_per_second": throughput,
            "num_samples": len(df_test)
        })

        # Free memory
        del models
        import gc
        gc.collect()

    # Save summary
    summary_file = Path("inference_timing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {summary_file}")

    # Print table
    print("\n" + "="*50)
    print(f"{'Seed':<10} | {'Time (s)':<12} | {'Throughput (s/s)':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['seed']:<10} | {r['inference_time_seconds']:<12.4f} | {r['samples_per_second']:<15.1f}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
