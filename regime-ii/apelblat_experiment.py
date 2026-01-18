"""
Apelblat Equation Comparison Experiment (Enhanced)
===================================================

This experiment evaluates how well the Glass-Onion model's predictions follow
the Apelblat equation for temperature-dependent solubility:

    ln(x) = A + B/T + C*ln(T)

Features:
- Approach A: Hold-out validation (for pairs with ≥7 temps)
- Approach B: Synthetic temperature testing (for all pairs)
- R² comparison: Known temps vs Unknown temps

Author: Glass-Onion Team
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the Apelblat experiment."""
    
    # Data paths
    test_file: str = "data/test.csv"
    train_file: str = "data/train.csv"
    store_dir: str = "feature_store"
    model_dir: str = "model"
    transformer_path: str = "transformer.pth"
    
    # Output paths
    output_dir: str = "apelblat_results"
    
    # Filtering thresholds
    min_temp_points: int = 5  # Minimum temps for Apelblat fitting
    min_temp_points_holdout: int = 7  # Minimum temps for hold-out validation
    
    # Synthetic temperature settings
    synthetic_step: float = 5.0  # Generate synthetic temps every 5K
    extrapolation_range: float = 10.0  # Test 10K beyond min/max temps
    
    # Hold-out settings
    holdout_interpolation_count: int = 2  # Number of middle temps to hold out
    holdout_extrapolation_count: int = 2  # Number of edge temps to hold out (1 from each end)
    
    # Classification thresholds
    excellent_threshold: float = 0.95
    good_threshold: float = 0.85
    moderate_threshold: float = 0.70
    
    # Visualization settings
    num_sample_plots: int = 20
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# APELBLAT EQUATION
# ============================================================================

def apelblat_equation(T: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Apelblat equation: LogS = A + B/T + C*ln(T)"""
    return A + B / T + C * np.log(T)


def fit_apelblat(temps: np.ndarray, logs: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Fit Apelblat equation to temperature-LogS data."""
    try:
        p0 = [logs.mean(), -1000.0, 0.1]
        bounds = ([-100, -50000, -10], [100, 50000, 10])
        
        params, _ = curve_fit(
            apelblat_equation, temps, logs, p0=p0, bounds=bounds, maxfev=5000
        )
        
        logs_pred = apelblat_equation(temps, *params)
        r2 = r2_score(logs, logs_pred)
        
        return params, r2, True
    except Exception:
        return np.array([0.0, 0.0, 0.0]), 0.0, False


# ============================================================================
# MODEL PREDICTOR
# ============================================================================

class ModelPredictor:
    """Wrapper for the Glass-Onion model to generate predictions."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self._load_models()
        
    def _load_models(self):
        print("Loading models and feature stores...")
        
        self.sol_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_raw.parquet")
        ).set_index("SMILES_KEY")
        self.solv_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_raw.parquet")
        ).set_index("SMILES_KEY")
        self.sol_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_council.parquet")
        ).set_index("SMILES_KEY")
        self.solv_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_council.parquet")
        ).set_index("SMILES_KEY")
        
        from train_transformer import InteractionTransformer
        self.transformer = InteractionTransformer().to(self.device)
        self.transformer.load_state_dict(
            torch.load(self.config.transformer_path, map_location=self.device)
        )
        self.transformer.eval()
        
        self.catboost_model = joblib.load(
            os.path.join(self.config.model_dir, "model.joblib")
        )
        self.selector = joblib.load(
            os.path.join(self.config.model_dir, "selector.joblib")
        )
        
        print("  ✓ Models loaded successfully")
    
    def predict(self, solutes: List[str], solvents: List[str], 
                temperatures: List[float]) -> np.ndarray:
        df = pd.DataFrame({
            "Solute": solutes, "Solvent": solvents, "Temperature": temperatures
        })
        
        X_sol = self.sol_council.loc[df["Solute"]].values.astype(np.float32)
        X_solv = self.solv_council.loc[df["Solvent"]].values.astype(np.float32)
        
        embeds = []
        with torch.no_grad():
            for i in range(len(df)):
                sol = torch.tensor(X_sol[i:i+1]).to(self.device)
                solv = torch.tensor(X_solv[i:i+1]).to(self.device)
                _, feats, _ = self.transformer(sol, solv)
                embeds.append(feats.cpu().numpy())
        
        X_embed = np.vstack(embeds)
        
        T = df["Temperature"].values.reshape(-1, 1)
        T_inv = (1000 / df["Temperature"]).values.reshape(-1, 1)
        Tm = self.sol_raw.loc[df["Solute"], "pred_Tm"].values.reshape(-1, 1)
        T_red = T / Tm
        
        X_reshaped = X_embed.reshape(-1, 24, 32)
        X_mod = np.linalg.norm(X_reshaped, axis=2)
        X_sign = np.sign(X_reshaped.mean(axis=2))
        X_interact = (X_sign * X_mod) * T_inv
        
        X_raw = np.hstack([
            self.sol_raw.loc[df["Solute"]].values,
            self.solv_raw.loc[df["Solvent"]].values
        ])
        
        X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
        X_pruned = self.selector.transform(X_full)
        
        return self.catboost_model.predict(X_pruned)


# ============================================================================
# PAIR ANALYSIS RESULT
# ============================================================================

@dataclass
class PairAnalysisResult:
    """Results for a single solute-solvent pair analysis."""
    solute: str
    solvent: str
    num_temps: int
    temp_range: Tuple[float, float]
    
    # Experimental Apelblat fit (on full data)
    exp_params: np.ndarray
    exp_r2: float
    exp_fit_success: bool
    
    # Prediction Apelblat fit
    pred_params: np.ndarray
    pred_r2: float
    pred_fit_success: bool
    
    # ========== R² at KNOWN temperatures ==========
    # Model predictions vs experimental at original temps
    r2_known_pred_vs_exp: float
    rmse_known: float
    
    # ========== Approach B: Synthetic Temperature Testing ==========
    # R² of predictions at synthetic temps vs Apelblat curve
    synthetic_temps: np.ndarray
    synthetic_preds: np.ndarray
    synthetic_apelblat: np.ndarray
    r2_synthetic_vs_apelblat: float
    rmse_synthetic: float
    
    # ========== Approach A: Hold-Out Validation (if applicable) ==========
    holdout_applicable: bool  # True if num_temps >= min_temp_points_holdout
    
    # Interpolation (middle temps held out)
    holdout_interp_temps: np.ndarray
    holdout_interp_exp: np.ndarray
    holdout_interp_preds: np.ndarray
    holdout_interp_apelblat: np.ndarray
    r2_holdout_interp_pred_vs_exp: float
    r2_holdout_interp_pred_vs_apelblat: float
    
    # Extrapolation (edge temps held out)
    holdout_extrap_temps: np.ndarray
    holdout_extrap_exp: np.ndarray
    holdout_extrap_preds: np.ndarray
    holdout_extrap_apelblat: np.ndarray
    r2_holdout_extrap_pred_vs_exp: float
    r2_holdout_extrap_pred_vs_apelblat: float
    
    # Apelblat params from reduced dataset (for hold-out)
    holdout_exp_params: np.ndarray
    holdout_exp_r2: float
    
    # Comparison metrics
    shape_correlation: float
    trend_score: float
    category: str
    
    # Original data
    original_temps: np.ndarray
    original_logs: np.ndarray
    original_preds: np.ndarray


# ============================================================================
# APELBLAT EXPERIMENT
# ============================================================================

class ApelblatExperiment:
    """Main experiment class for Apelblat equation comparison."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.predictor = ModelPredictor(config)
        self.results: List[PairAnalysisResult] = []
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "extrapolation_plots").mkdir(exist_ok=True)
        
    def load_data(self, use_train: bool = False) -> pd.DataFrame:
        print("\nLoading data...")
        df = pd.read_csv(self.config.test_file)
        if use_train:
            df_train = pd.read_csv(self.config.train_file)
            df = pd.concat([df, df_train], ignore_index=True)
        print(f"  ✓ Loaded {len(df)} samples")
        return df
    
    def group_by_pairs(self, df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
        groups = {}
        for (solute, solvent), group in df.groupby(["Solute", "Solvent"]):
            group = group.sort_values("Temperature").reset_index(drop=True)
            groups[(solute, solvent)] = group
        print(f"  ✓ Found {len(groups)} unique solute-solvent pairs")
        return groups
    
    def generate_synthetic_temps(self, temps: np.ndarray) -> np.ndarray:
        """Generate synthetic temperatures not in original data."""
        t_min, t_max = temps.min(), temps.max()
        step = self.config.synthetic_step
        extrap = self.config.extrapolation_range
        
        synthetic_range = np.arange(t_min - extrap, t_max + extrap + step, step)
        
        synthetic = [t for t in synthetic_range if np.min(np.abs(temps - t)) > 2.0]
        return np.array(synthetic)
    
    def get_holdout_indices(self, n_temps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get indices for hold-out validation.
        
        Returns:
            train_idx: indices for training (fitting Apelblat)
            interp_idx: indices for interpolation test (middle temps)
            extrap_idx: indices for extrapolation test (edge temps)
        """
        all_idx = np.arange(n_temps)
        
        # Extrapolation: first and last temps
        extrap_idx = np.array([0, n_temps - 1])
        
        # Interpolation: middle temps (evenly spaced)
        middle_range = all_idx[1:-1]  # Exclude first and last
        if len(middle_range) >= 2:
            step = max(1, len(middle_range) // (self.config.holdout_interpolation_count + 1))
            interp_idx = middle_range[step::step][:self.config.holdout_interpolation_count]
        else:
            interp_idx = np.array([], dtype=int)
        
        # Training: everything except held out
        holdout_idx = np.concatenate([extrap_idx, interp_idx])
        train_idx = np.array([i for i in all_idx if i not in holdout_idx])
        
        return train_idx, interp_idx, extrap_idx
    
    def analyze_pair(self, solute: str, solvent: str, 
                     group: pd.DataFrame) -> Optional[PairAnalysisResult]:
        """Analyze a single solute-solvent pair."""
        temps = group["Temperature"].values
        logs_exp = group["LogS"].values
        n_temps = len(temps)
        
        if n_temps < self.config.min_temp_points:
            return None
        
        # ========== Predictions at KNOWN (original) temperatures ==========
        preds_orig = self.predictor.predict(
            [solute] * n_temps, [solvent] * n_temps, temps.tolist()
        )
        
        # R² at known temps: predictions vs experimental
        r2_known = r2_score(logs_exp, preds_orig)
        rmse_known = np.sqrt(mean_squared_error(logs_exp, preds_orig))
        
        # Fit Apelblat to full experimental data
        exp_params, exp_r2, exp_success = fit_apelblat(temps, logs_exp)
        
        # Fit Apelblat to predictions
        pred_params, pred_r2, pred_success = fit_apelblat(temps, preds_orig)
        
        # ========== Approach B: Synthetic Temperature Testing ==========
        synthetic_temps = self.generate_synthetic_temps(temps)
        
        if len(synthetic_temps) > 0:
            synthetic_preds = self.predictor.predict(
                [solute] * len(synthetic_temps),
                [solvent] * len(synthetic_temps),
                synthetic_temps.tolist()
            )
            synthetic_apelblat = apelblat_equation(synthetic_temps, *exp_params) if exp_success else np.zeros_like(synthetic_temps)
            
            if exp_success and len(synthetic_temps) > 1:
                r2_synthetic = r2_score(synthetic_apelblat, synthetic_preds)
                rmse_synthetic = np.sqrt(mean_squared_error(synthetic_apelblat, synthetic_preds))
            else:
                r2_synthetic = 0.0
                rmse_synthetic = float('inf')
        else:
            synthetic_temps = np.array([])
            synthetic_preds = np.array([])
            synthetic_apelblat = np.array([])
            r2_synthetic = 0.0
            rmse_synthetic = float('inf')
        
        # ========== Approach A: Hold-Out Validation ==========
        holdout_applicable = n_temps >= self.config.min_temp_points_holdout
        
        if holdout_applicable:
            train_idx, interp_idx, extrap_idx = self.get_holdout_indices(n_temps)
            
            # Fit Apelblat on training subset
            train_temps = temps[train_idx]
            train_logs = logs_exp[train_idx]
            holdout_params, holdout_r2, holdout_success = fit_apelblat(train_temps, train_logs)
            
            # Interpolation test
            if len(interp_idx) > 0:
                interp_temps = temps[interp_idx]
                interp_exp = logs_exp[interp_idx]
                interp_preds = self.predictor.predict(
                    [solute] * len(interp_idx),
                    [solvent] * len(interp_idx),
                    interp_temps.tolist()
                )
                interp_apelblat = apelblat_equation(interp_temps, *holdout_params) if holdout_success else np.zeros_like(interp_temps)
                
                if len(interp_idx) > 1:
                    r2_interp_pred_vs_exp = r2_score(interp_exp, interp_preds)
                    r2_interp_pred_vs_apel = r2_score(interp_apelblat, interp_preds) if holdout_success else 0.0
                else:
                    # Single point - use error ratio
                    r2_interp_pred_vs_exp = 1 - (interp_preds[0] - interp_exp[0])**2 / max(0.001, np.var(logs_exp))
                    r2_interp_pred_vs_apel = 1 - (interp_preds[0] - interp_apelblat[0])**2 / max(0.001, np.var(logs_exp)) if holdout_success else 0.0
            else:
                interp_temps = np.array([])
                interp_exp = np.array([])
                interp_preds = np.array([])
                interp_apelblat = np.array([])
                r2_interp_pred_vs_exp = np.nan
                r2_interp_pred_vs_apel = np.nan
            
            # Extrapolation test
            extrap_temps = temps[extrap_idx]
            extrap_exp = logs_exp[extrap_idx]
            extrap_preds = self.predictor.predict(
                [solute] * len(extrap_idx),
                [solvent] * len(extrap_idx),
                extrap_temps.tolist()
            )
            extrap_apelblat = apelblat_equation(extrap_temps, *holdout_params) if holdout_success else np.zeros_like(extrap_temps)
            
            if len(extrap_idx) > 1:
                r2_extrap_pred_vs_exp = r2_score(extrap_exp, extrap_preds)
                r2_extrap_pred_vs_apel = r2_score(extrap_apelblat, extrap_preds) if holdout_success else 0.0
            else:
                r2_extrap_pred_vs_exp = 1 - (extrap_preds[0] - extrap_exp[0])**2 / max(0.001, np.var(logs_exp))
                r2_extrap_pred_vs_apel = 1 - (extrap_preds[0] - extrap_apelblat[0])**2 / max(0.001, np.var(logs_exp)) if holdout_success else 0.0
        else:
            # Not enough temps for hold-out
            holdout_params = np.array([0.0, 0.0, 0.0])
            holdout_r2 = 0.0
            interp_temps = np.array([])
            interp_exp = np.array([])
            interp_preds = np.array([])
            interp_apelblat = np.array([])
            r2_interp_pred_vs_exp = np.nan
            r2_interp_pred_vs_apel = np.nan
            extrap_temps = np.array([])
            extrap_exp = np.array([])
            extrap_preds = np.array([])
            extrap_apelblat = np.array([])
            r2_extrap_pred_vs_exp = np.nan
            r2_extrap_pred_vs_apel = np.nan
        
        # ========== Shape correlation ==========
        if exp_success and pred_success:
            eval_temps = np.linspace(temps.min(), temps.max(), 50)
            exp_curve = apelblat_equation(eval_temps, *exp_params)
            pred_curve = apelblat_equation(eval_temps, *pred_params)
            shape_corr, _ = pearsonr(exp_curve, pred_curve)
        else:
            shape_corr = 0.0
        
        # ========== Trend score ==========
        if exp_success:
            trend_score = (
                0.3 * max(0, r2_known) +
                0.3 * max(0, r2_synthetic) +
                0.2 * max(0, shape_corr) +
                0.2 * max(0, pred_r2)
            )
        else:
            trend_score = 0.0
        
        # Classification
        if trend_score >= self.config.excellent_threshold:
            category = "Excellent"
        elif trend_score >= self.config.good_threshold:
            category = "Good"
        elif trend_score >= self.config.moderate_threshold:
            category = "Moderate"
        else:
            category = "Poor"
        
        return PairAnalysisResult(
            solute=solute,
            solvent=solvent,
            num_temps=n_temps,
            temp_range=(temps.min(), temps.max()),
            exp_params=exp_params,
            exp_r2=exp_r2,
            exp_fit_success=exp_success,
            pred_params=pred_params,
            pred_r2=pred_r2,
            pred_fit_success=pred_success,
            r2_known_pred_vs_exp=r2_known,
            rmse_known=rmse_known,
            synthetic_temps=synthetic_temps,
            synthetic_preds=synthetic_preds,
            synthetic_apelblat=synthetic_apelblat,
            r2_synthetic_vs_apelblat=r2_synthetic,
            rmse_synthetic=rmse_synthetic,
            holdout_applicable=holdout_applicable,
            holdout_interp_temps=interp_temps,
            holdout_interp_exp=interp_exp,
            holdout_interp_preds=interp_preds,
            holdout_interp_apelblat=interp_apelblat,
            r2_holdout_interp_pred_vs_exp=r2_interp_pred_vs_exp,
            r2_holdout_interp_pred_vs_apelblat=r2_interp_pred_vs_apel,
            holdout_extrap_temps=extrap_temps,
            holdout_extrap_exp=extrap_exp,
            holdout_extrap_preds=extrap_preds,
            holdout_extrap_apelblat=extrap_apelblat,
            r2_holdout_extrap_pred_vs_exp=r2_extrap_pred_vs_exp,
            r2_holdout_extrap_pred_vs_apelblat=r2_extrap_pred_vs_apel,
            holdout_exp_params=holdout_params,
            holdout_exp_r2=holdout_r2,
            shape_correlation=shape_corr,
            trend_score=trend_score,
            category=category,
            original_temps=temps,
            original_logs=logs_exp,
            original_preds=preds_orig
        )
    
    def run_experiment(self, use_train: bool = False):
        print("\n" + "="*70)
        print("APELBLAT EQUATION COMPARISON EXPERIMENT (Enhanced)")
        print("="*70)
        
        df = self.load_data(use_train)
        pairs = self.group_by_pairs(df)
        
        print("\nAnalyzing pairs...")
        self.results = []
        
        for (solute, solvent), group in tqdm(pairs.items(), desc="Processing pairs"):
            result = self.analyze_pair(solute, solvent, group)
            if result is not None:
                self.results.append(result)
        
        print(f"\n  ✓ Analyzed {len(self.results)} pairs with sufficient data")
        
        self.save_results()
        self.generate_summary()
        self.generate_comparison_report()
        self.generate_plots()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)
    
    def save_results(self):
        print("\nSaving results...")
        
        rows = []
        for r in self.results:
            rows.append({
                "Solute": r.solute,
                "Solvent": r.solvent,
                "Num_Temps": r.num_temps,
                "Temp_Min": r.temp_range[0],
                "Temp_Max": r.temp_range[1],
                # Known temps
                "R2_Known_Pred_vs_Exp": r.r2_known_pred_vs_exp,
                "RMSE_Known": r.rmse_known,
                # Approach B: Synthetic
                "R2_Synthetic_vs_Apelblat": r.r2_synthetic_vs_apelblat,
                "RMSE_Synthetic": r.rmse_synthetic,
                "Num_Synthetic_Temps": len(r.synthetic_temps),
                # Approach A: Hold-out
                "Holdout_Applicable": r.holdout_applicable,
                "R2_Holdout_Interp_Pred_vs_Exp": r.r2_holdout_interp_pred_vs_exp,
                "R2_Holdout_Interp_Pred_vs_Apelblat": r.r2_holdout_interp_pred_vs_apelblat,
                "R2_Holdout_Extrap_Pred_vs_Exp": r.r2_holdout_extrap_pred_vs_exp,
                "R2_Holdout_Extrap_Pred_vs_Apelblat": r.r2_holdout_extrap_pred_vs_apelblat,
                # Apelblat fits
                "Exp_R2": r.exp_r2,
                "Pred_R2": r.pred_r2,
                "Shape_Correlation": r.shape_correlation,
                "Trend_Score": r.trend_score,
                "Category": r.category
            })
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(self.output_dir / "pair_analysis.csv", index=False)
        
        df_sorted = df_results.sort_values("Trend_Score", ascending=False)
        df_sorted.head(50).to_csv(self.output_dir / "best_pairs.csv", index=False)
        df_sorted.tail(50).to_csv(self.output_dir / "worst_pairs.csv", index=False)
        
        print(f"  ✓ Saved pair_analysis.csv ({len(rows)} pairs)")
    
    def generate_summary(self):
        print("\nGenerating summary statistics...")
        
        categories = [r.category for r in self.results]
        category_counts = {
            "Excellent": categories.count("Excellent"),
            "Good": categories.count("Good"),
            "Moderate": categories.count("Moderate"),
            "Poor": categories.count("Poor")
        }
        
        # R² statistics
        r2_known = [r.r2_known_pred_vs_exp for r in self.results]
        r2_synthetic = [r.r2_synthetic_vs_apelblat for r in self.results if r.r2_synthetic_vs_apelblat > -10]
        
        # Hold-out results (only for applicable pairs)
        holdout_results = [r for r in self.results if r.holdout_applicable]
        r2_interp = [r.r2_holdout_interp_pred_vs_exp for r in holdout_results if not np.isnan(r.r2_holdout_interp_pred_vs_exp)]
        r2_extrap = [r.r2_holdout_extrap_pred_vs_exp for r in holdout_results if not np.isnan(r.r2_holdout_extrap_pred_vs_exp)]
        
        summary = {
            "total_pairs_analyzed": len(self.results),
            "pairs_with_holdout": len(holdout_results),
            "category_distribution": category_counts,
            "category_percentages": {
                k: round(v / len(self.results) * 100, 2) for k, v in category_counts.items()
            },
            "r2_known_temps": {
                "mean": round(np.mean(r2_known), 4),
                "std": round(np.std(r2_known), 4),
                "min": round(np.min(r2_known), 4),
                "max": round(np.max(r2_known), 4)
            },
            "r2_synthetic_temps_approach_b": {
                "mean": round(np.mean(r2_synthetic), 4),
                "std": round(np.std(r2_synthetic), 4),
                "min": round(np.min(r2_synthetic), 4),
                "max": round(np.max(r2_synthetic), 4)
            },
            "r2_holdout_interpolation_approach_a": {
                "mean": round(np.mean(r2_interp), 4) if r2_interp else None,
                "std": round(np.std(r2_interp), 4) if r2_interp else None,
                "count": len(r2_interp)
            },
            "r2_holdout_extrapolation_approach_a": {
                "mean": round(np.mean(r2_extrap), 4) if r2_extrap else None,
                "std": round(np.std(r2_extrap), 4) if r2_extrap else None,
                "count": len(r2_extrap)
            }
        }
        
        with open(self.output_dir / "summary_statistics.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "-"*50)
        print("SUMMARY")
        print("-"*50)
        print(f"Total pairs analyzed: {summary['total_pairs_analyzed']}")
        print(f"Pairs with hold-out validation: {summary['pairs_with_holdout']}")
        print(f"\nCategory Distribution:")
        for cat, count in category_counts.items():
            pct = summary['category_percentages'][cat]
            print(f"  {cat:12s}: {count:4d} ({pct:.1f}%)")
        print("-"*50)
    
    def generate_comparison_report(self):
        """Generate comparison between known vs unknown temperature performance."""
        print("\nGenerating comparison report...")
        
        # Collect RMSE data for each pair
        all_pairs = []
        for r in self.results:
            row = {
                "Solute": r.solute[:30],
                "Solvent": r.solvent[:20],
                "Num_Temps": r.num_temps,
                "RMSE_Known": r.rmse_known,
                "RMSE_Synthetic": r.rmse_synthetic if r.rmse_synthetic != float('inf') else np.nan,
                "R2_Known": max(-1, r.r2_known_pred_vs_exp),  # Clamp R² for stats
                "R2_Synthetic": max(-1, r.r2_synthetic_vs_apelblat),
            }
            if r.holdout_applicable:
                row["R2_Holdout_Interp"] = max(-1, r.r2_holdout_interp_pred_vs_exp) if not np.isnan(r.r2_holdout_interp_pred_vs_exp) else np.nan
                row["R2_Holdout_Extrap"] = max(-1, r.r2_holdout_extrap_pred_vs_exp) if not np.isnan(r.r2_holdout_extrap_pred_vs_exp) else np.nan
            all_pairs.append(row)
        
        df_compare = pd.DataFrame(all_pairs)
        df_compare.to_csv(self.output_dir / "r2_comparison.csv", index=False)
        
        # Filter to reasonable R² values for statistics (between -1 and 1)
        r2_known = [max(-1, min(1, r.r2_known_pred_vs_exp)) for r in self.results]
        r2_synthetic = [max(-1, min(1, r.r2_synthetic_vs_apelblat)) for r in self.results]
        
        rmse_known = [r.rmse_known for r in self.results]
        rmse_synthetic = [r.rmse_synthetic for r in self.results if r.rmse_synthetic != float('inf') and r.rmse_synthetic < 10]
        
        holdout_results = [r for r in self.results if r.holdout_applicable]
        r2_interp = [max(-1, min(1, r.r2_holdout_interp_pred_vs_exp)) for r in holdout_results 
                     if not np.isnan(r.r2_holdout_interp_pred_vs_exp)]
        r2_extrap = [max(-1, min(1, r.r2_holdout_extrap_pred_vs_exp)) for r in holdout_results 
                     if not np.isnan(r.r2_holdout_extrap_pred_vs_exp)]
        
        print("\n" + "="*75)
        print("COMPARISON: KNOWN vs UNKNOWN TEMPERATURES")
        print("="*75)
        
        # RMSE comparison (primary metric - always interpretable)
        print(f"\n{'='*75}")
        print("RMSE COMPARISON (lower is better)")
        print(f"{'='*75}")
        print(f"\n{'Metric':<45} {'Mean RMSE':>12} {'Std':>10}")
        print("-"*75)
        print(f"{'Known Temperatures (original data)':<45} {np.mean(rmse_known):>12.4f} {np.std(rmse_known):>10.4f}")
        if rmse_synthetic:
            print(f"{'Approach B: Synthetic Temps':<45} {np.mean(rmse_synthetic):>12.4f} {np.std(rmse_synthetic):>10.4f}")
        print("-"*75)
        
        # R² comparison (clamped to [-1, 1])
        print(f"\n{'='*75}")
        print("R² COMPARISON (clamped to [-1, 1], higher is better)")
        print(f"{'='*75}")
        print(f"\n{'Metric':<45} {'Mean R²':>12} {'Std':>10} {'N':>6}")
        print("-"*75)
        print(f"{'Known Temperatures (Pred vs Exp)':<45} {np.mean(r2_known):>12.4f} {np.std(r2_known):>10.4f} {len(r2_known):>6}")
        print(f"{'Approach B: Synthetic (Pred vs Apelblat)':<45} {np.mean(r2_synthetic):>12.4f} {np.std(r2_synthetic):>10.4f} {len(r2_synthetic):>6}")
        if r2_interp:
            print(f"{'Approach A: Hold-out Interpolation':<45} {np.mean(r2_interp):>12.4f} {np.std(r2_interp):>10.4f} {len(r2_interp):>6}")
        if r2_extrap:
            print(f"{'Approach A: Hold-out Extrapolation':<45} {np.mean(r2_extrap):>12.4f} {np.std(r2_extrap):>10.4f} {len(r2_extrap):>6}")
        print("-"*75)
        
        # Summary statistics
        print(f"\n📊 KEY FINDINGS:")
        print(f"   • RMSE increases {np.mean(rmse_synthetic)/np.mean(rmse_known):.2f}x from known to synthetic temps" if rmse_synthetic else "")
        print(f"   • R² drops from {np.mean(r2_known):.3f} (known) to {np.mean(r2_synthetic):.3f} (synthetic)")
        if r2_interp:
            print(f"   • Hold-out interpolation R²: {np.mean(r2_interp):.3f}")
        if r2_extrap:
            print(f"   • Hold-out extrapolation R²: {np.mean(r2_extrap):.3f}")
        print("="*75)
        
        # Save comparison summary
        comparison_summary = {
            "rmse_known_temps": {
                "mean": round(np.mean(rmse_known), 4),
                "std": round(np.std(rmse_known), 4)
            },
            "rmse_synthetic_temps": {
                "mean": round(np.mean(rmse_synthetic), 4) if rmse_synthetic else None,
                "std": round(np.std(rmse_synthetic), 4) if rmse_synthetic else None,
                "rmse_ratio": round(np.mean(rmse_synthetic)/np.mean(rmse_known), 4) if rmse_synthetic else None
            },
            "r2_known_temps_clamped": {
                "mean": round(np.mean(r2_known), 4),
                "std": round(np.std(r2_known), 4)
            },
            "approach_b_synthetic_r2_clamped": {
                "mean": round(np.mean(r2_synthetic), 4),
                "std": round(np.std(r2_synthetic), 4),
                "r2_drop_from_known": round(np.mean(r2_known) - np.mean(r2_synthetic), 4)
            },
            "approach_a_holdout_interpolation": {
                "mean_r2": round(np.mean(r2_interp), 4) if r2_interp else None,
                "std_r2": round(np.std(r2_interp), 4) if r2_interp else None,
                "r2_drop_from_known": round(np.mean(r2_known) - np.mean(r2_interp), 4) if r2_interp else None,
                "num_pairs": len(r2_interp)
            },
            "approach_a_holdout_extrapolation": {
                "mean_r2": round(np.mean(r2_extrap), 4) if r2_extrap else None,
                "std_r2": round(np.std(r2_extrap), 4) if r2_extrap else None,
                "r2_drop_from_known": round(np.mean(r2_known) - np.mean(r2_extrap), 4) if r2_extrap else None,
                "num_pairs": len(r2_extrap)
            }
        }
        
        with open(self.output_dir / "r2_comparison_summary.json", "w") as f:
            json.dump(comparison_summary, f, indent=2)
    
    def generate_plots(self):
        print("\nGenerating plots...")
        
        self._plot_r2_comparison()
        self._plot_approach_comparison()
        self._plot_sample_pairs()
        
        print(f"  ✓ Plots saved to {self.output_dir / 'plots'}")
    
    def _plot_r2_comparison(self):
        """Plot R² at known vs unknown temperatures."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        r2_known = [r.r2_known_pred_vs_exp for r in self.results]
        r2_synthetic = [r.r2_synthetic_vs_apelblat for r in self.results]
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(r2_known, r2_synthetic, alpha=0.5, s=30)
        ax.plot([0, 1], [0, 1], 'r--', label='y=x')
        ax.set_xlabel("R² at Known Temps", fontsize=12)
        ax.set_ylabel("R² at Synthetic Temps (vs Apelblat)", fontsize=12)
        ax.set_title("Approach B: Known vs Synthetic Temperature R²", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, 1.05)
        ax.set_ylim(-1, 1.05)
        
        # Box plot comparison
        ax = axes[1]
        holdout_results = [r for r in self.results if r.holdout_applicable]
        
        data = [
            r2_known,
            r2_synthetic,
            [r.r2_holdout_interp_pred_vs_exp for r in holdout_results if not np.isnan(r.r2_holdout_interp_pred_vs_exp)],
            [r.r2_holdout_extrap_pred_vs_exp for r in holdout_results if not np.isnan(r.r2_holdout_extrap_pred_vs_exp)]
        ]
        labels = ['Known\nTemps', 'Synthetic\n(Approach B)', 'Holdout Interp\n(Approach A)', 'Holdout Extrap\n(Approach A)']
        
        bp = ax.boxplot([d for d in data if len(d) > 0], labels=[l for l, d in zip(labels, data) if len(d) > 0])
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("R² Distribution: Known vs Unknown Temperatures", fontsize=12)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "r2_comparison.png", dpi=150)
        plt.close()
    
    def _plot_approach_comparison(self):
        """Plot comparison between Approach A and Approach B."""
        holdout_results = [r for r in self.results if r.holdout_applicable]
        
        if len(holdout_results) < 10:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Approach A vs B for interpolation
        ax = axes[0]
        r2_synthetic = [r.r2_synthetic_vs_apelblat for r in holdout_results]
        r2_interp = [r.r2_holdout_interp_pred_vs_exp for r in holdout_results]
        
        valid_idx = [i for i, v in enumerate(r2_interp) if not np.isnan(v)]
        if valid_idx:
            ax.scatter([r2_synthetic[i] for i in valid_idx], 
                       [r2_interp[i] for i in valid_idx], alpha=0.5, s=30)
            ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
            ax.set_xlabel("R² Synthetic (Approach B)", fontsize=12)
            ax.set_ylabel("R² Hold-out Interpolation (Approach A)", fontsize=12)
            ax.set_title("Approach A vs B: Interpolation", fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Approach A vs B for extrapolation
        ax = axes[1]
        r2_extrap = [r.r2_holdout_extrap_pred_vs_exp for r in holdout_results]
        
        valid_idx = [i for i, v in enumerate(r2_extrap) if not np.isnan(v)]
        if valid_idx:
            ax.scatter([r2_synthetic[i] for i in valid_idx],
                       [r2_extrap[i] for i in valid_idx], alpha=0.5, s=30)
            ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
            ax.set_xlabel("R² Synthetic (Approach B)", fontsize=12)
            ax.set_ylabel("R² Hold-out Extrapolation (Approach A)", fontsize=12)
            ax.set_title("Approach A vs B: Extrapolation", fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "approach_comparison.png", dpi=150)
        plt.close()
    
    def _plot_sample_pairs(self):
        """Generate detailed plots for sample pairs."""
        sorted_results = sorted(self.results, key=lambda r: r.trend_score, reverse=True)
        samples = sorted_results[:5] + sorted_results[-5:]
        
        for i, result in enumerate(samples[:10]):
            self._plot_single_pair(result, i)
    
    def _plot_single_pair(self, result: PairAnalysisResult, idx: int):
        """Generate detailed plot for a single pair with all approaches."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Original experimental data
        ax.scatter(result.original_temps, result.original_logs, 
                   s=100, c='blue', marker='o', label='Experimental', zorder=5)
        
        # Model predictions at known temps
        ax.scatter(result.original_temps, result.original_preds,
                   s=100, c='red', marker='s', 
                   label=f'Predictions @Known (R²={result.r2_known_pred_vs_exp:.3f})', zorder=5)
        
        # Apelblat curve
        if result.exp_fit_success:
            t_range = np.linspace(result.temp_range[0] - 15, result.temp_range[1] + 15, 100)
            exp_curve = apelblat_equation(t_range, *result.exp_params)
            ax.plot(t_range, exp_curve, 'b-', linewidth=2, alpha=0.7,
                    label=f'Apelblat Curve (R²={result.exp_r2:.3f})')
        
        # Approach B: Synthetic temps
        if len(result.synthetic_temps) > 0:
            ax.scatter(result.synthetic_temps, result.synthetic_preds,
                       s=80, c='green', marker='x', linewidths=2,
                       label=f'Approach B: Synthetic (R²={result.r2_synthetic_vs_apelblat:.3f})', 
                       zorder=4)
        
        # Approach A: Hold-out temps (if applicable)
        if result.holdout_applicable:
            if len(result.holdout_interp_temps) > 0:
                ax.scatter(result.holdout_interp_temps, result.holdout_interp_preds,
                           s=100, c='orange', marker='^', linewidths=2,
                           label=f'Approach A: Interp (R²={result.r2_holdout_interp_pred_vs_exp:.3f})',
                           zorder=4)
            if len(result.holdout_extrap_temps) > 0:
                ax.scatter(result.holdout_extrap_temps, result.holdout_extrap_preds,
                           s=100, c='purple', marker='v', linewidths=2,
                           label=f'Approach A: Extrap (R²={result.r2_holdout_extrap_pred_vs_exp:.3f})',
                           zorder=4)
        
        ax.set_xlabel("Temperature (K)", fontsize=12)
        ax.set_ylabel("LogS", fontsize=12)
        ax.set_title(f"Apelblat Analysis: {result.category} | Trend Score: {result.trend_score:.3f}", fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        folder = "plots" if result.category in ["Excellent", "Good"] else "extrapolation_plots"
        plt.savefig(self.output_dir / folder / f"pair_{idx:03d}_{result.category.lower()}.png", dpi=150)
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = ExperimentConfig()
    experiment = ApelblatExperiment(config)
    experiment.run_experiment(use_train=False)


if __name__ == "__main__":
    main()
