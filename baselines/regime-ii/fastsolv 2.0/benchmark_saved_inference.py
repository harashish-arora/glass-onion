# benchmark_saved_inference.py
"""
FastSolv 2.0 saved-model inference benchmark for one seed.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from pytorch_lightning import Trainer
from fastprop.data import fastpropDataLoader
from fastprop.defaults import ALL_2D

from model_architecture import SolubilityDataset


NUM_REPLICATES = 4
MODEL_DIR = Path("model")
TEST_FILE = "data/baseline_bigsol2/baseline_test.csv"
SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Run one saved seed.")
    return parser.parse_args()


def load_model(seed, rep_idx):
    seed_dir = MODEL_DIR / f"seed_{seed}"
    model_path = seed_dir / f"replicate_{rep_idx}.pt"
    print(f"  Loading replicate {rep_idx}/{NUM_REPLICATES}: {model_path.name}", flush=True)
    load_start = time.time()
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model, time.time() - load_start


def evaluate_seed(seed):
    print("Loading test set...", flush=True)
    df_test = pd.read_csv(TEST_FILE)
    test_var = df_test["logS"].var()

    print("Preparing test tensors...", flush=True)
    test_solute_arr = torch.tensor(df_test[SOLUTE_COLUMNS].values, dtype=torch.float32)
    test_solvent_arr = torch.tensor(df_test[SOLVENT_COLUMNS].values, dtype=torch.float32)
    test_temp_arr = torch.tensor(df_test["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    test_logS_arr = torch.tensor(df_test["logS"].values.reshape(-1, 1), dtype=torch.float32)

    print("Building test loader...", flush=True)
    test_ds = SolubilityDataset(
        test_solute_arr.clone().detach(),
        test_solvent_arr.clone().detach(),
        test_temp_arr.clone().detach(),
        test_logS_arr.clone().detach(),
        torch.zeros_like(test_logS_arr),
    )
    test_loader = fastpropDataLoader(test_ds, batch_size=256)

    print("Running saved ensemble inference...", flush=True)
    pred_trainer = Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    all_preds = []
    model_load_seconds = 0.0
    inference_start = time.time()

    for rep_idx in range(1, NUM_REPLICATES + 1):
        model, load_seconds = load_model(seed, rep_idx)
        model_load_seconds += load_seconds
        preds = torch.cat(pred_trainer.predict(model, test_loader))
        all_preds.append(preds.numpy())
        del model
        gc.collect()

    final_preds = np.mean(all_preds, axis=0)
    elapsed = time.time() - inference_start

    rmse = np.sqrt(mean_squared_error(df_test["logS"], final_preds))
    r2 = 1 - (rmse ** 2 / test_var)
    throughput = len(df_test) / elapsed

    result = {
        "seed": seed,
        "num_samples": int(len(df_test)),
        "test_file": TEST_FILE,
        "num_replicates": NUM_REPLICATES,
        "model_load_seconds": float(model_load_seconds),
        "rmse": float(rmse),
        "r2": float(r2),
        "inference_time_seconds": float(elapsed),
        "throughput_samples_per_sec": float(throughput),
    }
    seed_file = MODEL_DIR / f"inference_seed_{seed}.json"
    with open(seed_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Seed {seed}")
    print(f"  Model load time: {model_load_seconds:.4f} s")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Inference time: {elapsed:.4f} s")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print(f"  Saved seed summary: {seed_file}")


def main():
    args = parse_args()
    evaluate_seed(args.seed)


if __name__ == "__main__":
    main()
