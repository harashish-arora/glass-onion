
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
- Enhanced Metrics: MAPE, Slope/Sensitivity analysis, Parameter Correlation

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
    min_temp_points: int = 7  # Minimum temps for Apelblat fitting
    min_temp_points_holdout: int = 7  # Minimum temps for hold-out validation
    min_exp_apelblat_r2: float = 0.8  # Minimum R² for experimental Apelblat fit
    
    # Quality stratification
    stratify_by_quality: bool = True  # Enable RMSE-based filtering
    max_rmse_known: float = 0.5  # Maximum RMSE at known temps to include pair
    
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
    
    # RMSE Tolerance Analysis
    train_rmse_tolerance: float = 0.216  # Expected model RMSE on train data
    test_rmse_tolerance: float = 0.6775  # Expected model RMSE on test data
    
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
    r2_known_pred_vs_exp: float
    rmse_known: float
    
    # ========== Approach B: Synthetic Temperature Testing ==========
    synthetic_temps: np.ndarray
    synthetic_preds: np.ndarray
    synthetic_apelblat: np.ndarray
    r2_synthetic_vs_apelblat: float
    rmse_synthetic: float
    
    # ========== Approach A: Hold-Out Validation ==========
    holdout_applicable: bool
    
    # Interpolation
    holdout_interp_temps: np.ndarray
    holdout_interp_exp: np.ndarray
    holdout_interp_preds: np.ndarray
    holdout_interp_apelblat: np.ndarray
    r2_holdout_interp_pred_vs_exp: float
    r2_holdout_interp_pred_vs_apelblat: float
    
    # Extrapolation
    holdout_extrap_temps: np.ndarray
    holdout_extrap_exp: np.ndarray
    holdout_extrap_preds: np.ndarray
    holdout_extrap_apelblat: np.ndarray
    r2_holdout_extrap_pred_vs_exp: float
    r2_holdout_extrap_pred_vs_apelblat: float
    
    holdout_exp_params: np.ndarray
    holdout_exp_r2: float
    
    # Comparison metrics
    shape_correlation: float
    trend_score: float
    category: str
    
    # ========== Enhanced Metrics ==========
    param_correlation: float
    param_rmse: float
    mape: float
    exp_slope: float
    pred_slope: float
    slope_ratio: float
    exp_range: float
    pred_range: float
    sensitivity_ratio: float
    
    # ========== RMSE Tolerance Analysis ==========
    curve_max_deviation: float  # Max |pred_apelblat - exp_apelblat| across temps
    curve_mean_deviation: float  # Mean |pred_apelblat - exp_apelblat|
    curve_within_tolerance: bool  # Is max_deviation <= model_rmse?
    pct_points_within_tolerance: float  # % of temp points within tolerance
    
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
        t_min, t_max = temps.min(), temps.max()
        step = self.config.synthetic_step
        extrap = self.config.extrapolation_range
        
        synthetic_range = np.arange(t_min - extrap, t_max + extrap + step, step)
        synthetic = [t for t in synthetic_range if np.min(np.abs(temps - t)) > 2.0]
        return np.array(synthetic)
    
    def get_holdout_indices(self, n_temps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_idx = np.arange(n_temps)
        
        # Extrapolation: first and last
        extrap_idx = np.array([0, n_temps - 1])
        
        # Interpolation: middle temps
        middle_range = all_idx[1:-1]
        if len(middle_range) >= 2:
            step = max(1, len(middle_range) // (self.config.holdout_interpolation_count + 1))
            interp_idx = middle_range[step::step][:self.config.holdout_interpolation_count]
        else:
            interp_idx = np.array([], dtype=int)
        
        holdout_idx = np.concatenate([extrap_idx, interp_idx])
        train_idx = np.array([i for i in all_idx if i not in holdout_idx])
        
        return train_idx, interp_idx, extrap_idx
    
    def analyze_pair(self, solute: str, solvent: str, 
                     group: pd.DataFrame) -> Optional[PairAnalysisResult]:
        temps = group["Temperature"].values
        logs_exp = group["LogS"].values
        n_temps = len(temps)
        
        if n_temps < self.config.min_temp_points:
            return None
        
        preds_orig = self.predictor.predict(
            [solute] * n_temps, [solvent] * n_temps, temps.tolist()
        )
        
        r2_known = r2_score(logs_exp, preds_orig)
        rmse_known = np.sqrt(mean_squared_error(logs_exp, preds_orig))
        
        # Quality Stratification
        if self.config.stratify_by_quality and rmse_known > self.config.max_rmse_known:
            return None
        
        exp_params, exp_r2, exp_success = fit_apelblat(temps, logs_exp)
        
        if not exp_success or exp_r2 < self.config.min_exp_apelblat_r2:
            return None
        
        pred_params, pred_r2, pred_success = fit_apelblat(temps, preds_orig)
        
        # Approach B: Synthetic
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
        
        # Approach A: Holdout
        holdout_applicable = n_temps >= self.config.min_temp_points_holdout
        if holdout_applicable:
            train_idx, interp_idx, extrap_idx = self.get_holdout_indices(n_temps)
            
            train_temps = temps[train_idx]
            train_logs = logs_exp[train_idx]
            holdout_params, holdout_r2, holdout_success = fit_apelblat(train_temps, train_logs)
            
            # Interpolation
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
                    denominator = max(0.001, np.var(logs_exp))
                    r2_interp_pred_vs_exp = 1 - (interp_preds[0] - interp_exp[0])**2 / denominator
                    r2_interp_pred_vs_apel = 1 - (interp_preds[0] - interp_apelblat[0])**2 / denominator if holdout_success else 0.0
            else:
                interp_temps = np.array([])
                interp_exp = np.array([])
                interp_preds = np.array([])
                interp_apelblat = np.array([])
                r2_interp_pred_vs_exp = np.nan
                r2_interp_pred_vs_apel = np.nan
            
            # Extrapolation
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
                denominator = max(0.001, np.var(logs_exp))
                r2_extrap_pred_vs_exp = 1 - (extrap_preds[0] - extrap_exp[0])**2 / denominator
                r2_extrap_pred_vs_apel = 1 - (extrap_preds[0] - extrap_apelblat[0])**2 / denominator if holdout_success else 0.0
        else:
            holdout_params = np.zeros(3)
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
        
        # Shape correlation
        if exp_success and pred_success:
            eval_temps = np.linspace(temps.min(), temps.max(), 50)
            exp_curve = apelblat_equation(eval_temps, *exp_params)
            pred_curve = apelblat_equation(eval_temps, *pred_params)
            shape_corr, _ = pearsonr(exp_curve, pred_curve)
            
            # Param correlation
            try:
                param_correlation = np.corrcoef(exp_params, pred_params)[0, 1]
                if np.isnan(param_correlation): param_correlation = 0.0
            except:
                param_correlation = 0.0
            param_rmse = np.sqrt(np.mean((exp_params - pred_params)**2))
            
            # ========== RMSE Tolerance Analysis ==========
            # Compare Apelblat curves at original temperature points
            exp_apelblat_at_temps = apelblat_equation(temps, *exp_params)
            pred_apelblat_at_temps = apelblat_equation(temps, *pred_params)
            curve_deviations = np.abs(pred_apelblat_at_temps - exp_apelblat_at_temps)
            
            curve_max_deviation = np.max(curve_deviations)
            curve_mean_deviation = np.mean(curve_deviations)
            curve_within_tolerance = curve_max_deviation <= self.config.test_rmse_tolerance  # Will be recalculated per-dataset
            pct_points_within_tolerance = np.mean(curve_deviations <= self.config.test_rmse_tolerance) * 100
        else:
            shape_corr = 0.0
            param_correlation = 0.0
            param_rmse = float('inf')
            curve_max_deviation = float('inf')
            curve_mean_deviation = float('inf')
            curve_within_tolerance = False
            pct_points_within_tolerance = 0.0
        
        # Enhanced Metrics
        non_zero_mask = np.abs(logs_exp) > 0.01
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((preds_orig[non_zero_mask] - logs_exp[non_zero_mask]) / logs_exp[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        exp_slope = np.polyfit(temps, logs_exp, 1)[0]
        pred_slope = np.polyfit(temps, preds_orig, 1)[0]
        slope_ratio = pred_slope / exp_slope if abs(exp_slope) > 1e-6 else 0.0
        
        exp_range = logs_exp.max() - logs_exp.min()
        pred_range = preds_orig.max() - preds_orig.min()
        sensitivity_ratio = pred_range / max(0.01, exp_range)
        
        trend_score = (
            0.3 * max(0, r2_known) +
            0.3 * max(0, r2_synthetic) +
            0.2 * max(0, shape_corr) +
            0.2 * max(0, pred_r2)
        )
        
        if trend_score >= self.config.excellent_threshold: category = "Excellent"
        elif trend_score >= self.config.good_threshold: category = "Good"
        elif trend_score >= self.config.moderate_threshold: category = "Moderate"
        else: category = "Poor"
        
        return PairAnalysisResult(
            solute=solute, solvent=solvent, num_temps=n_temps, temp_range=(temps.min(), temps.max()),
            exp_params=exp_params, exp_r2=exp_r2, exp_fit_success=exp_success,
            pred_params=pred_params, pred_r2=pred_r2, pred_fit_success=pred_success,
            r2_known_pred_vs_exp=r2_known, rmse_known=rmse_known,
            synthetic_temps=synthetic_temps, synthetic_preds=synthetic_preds, synthetic_apelblat=synthetic_apelblat,
            r2_synthetic_vs_apelblat=r2_synthetic, rmse_synthetic=rmse_synthetic,
            holdout_applicable=holdout_applicable,
            holdout_interp_temps=interp_temps, holdout_interp_exp=interp_exp, holdout_interp_preds=interp_preds, holdout_interp_apelblat=interp_apelblat,
            r2_holdout_interp_pred_vs_exp=r2_interp_pred_vs_exp, r2_holdout_interp_pred_vs_apelblat=r2_interp_pred_vs_apel,
            holdout_extrap_temps=extrap_temps, holdout_extrap_exp=extrap_exp, holdout_extrap_preds=extrap_preds, holdout_extrap_apelblat=extrap_apelblat,
            r2_holdout_extrap_pred_vs_exp=r2_extrap_pred_vs_exp, r2_holdout_extrap_pred_vs_apelblat=r2_extrap_pred_vs_apel,
            holdout_exp_params=holdout_params, holdout_exp_r2=holdout_r2,
            shape_correlation=shape_corr, trend_score=trend_score, category=category,
            param_correlation=param_correlation, param_rmse=param_rmse, mape=mape,
            exp_slope=exp_slope, pred_slope=pred_slope, slope_ratio=slope_ratio,
            exp_range=exp_range, pred_range=pred_range, sensitivity_ratio=sensitivity_ratio,
            # RMSE Tolerance Analysis
            curve_max_deviation=curve_max_deviation, curve_mean_deviation=curve_mean_deviation,
            curve_within_tolerance=curve_within_tolerance, pct_points_within_tolerance=pct_points_within_tolerance,
            original_temps=temps, original_logs=logs_exp, original_preds=preds_orig
        )
    
    def run_experiment(self, use_train: bool = False, label: str = ""):
        """Run experiment on a single dataset (train or test)."""
        df = self.load_data(use_train)
        pairs = self.group_by_pairs(df)
        
        print(f"\nAnalyzing {label} pairs...")
        results = []
        
        for (solute, solvent), group in tqdm(pairs.items(), desc=f"Processing {label}"):
            result = self.analyze_pair(solute, solvent, group)
            if result is not None:
                results.append(result)
        
        print(f"  ✓ Analyzed {len(results)}/{len(pairs)} pairs")
        return results
    
    def run_full_experiment(self):
        """Run complete experiment on both train and test datasets."""
        print("\n" + "="*70)
        print("APELBLAT EQUATION COMPARISON EXPERIMENT")
        print("Separate Train/Test Analysis with Holdout Breakdown")
        print("="*70)
        
        # Run on train data
        print("\n" + "-"*70)
        print("TRAIN DATA ANALYSIS")
        print("-"*70)
        self.train_results = self.run_experiment(use_train=True, label="TRAIN")
        
        # Run on test data
        print("\n" + "-"*70)
        print("TEST DATA ANALYSIS")
        print("-"*70)
        self.test_results = self.run_experiment(use_train=False, label="TEST")
        
        # Store combined for backward compatibility
        self.results = self.train_results + self.test_results
        
        # Generate comprehensive summary
        self.save_split_results()
        self.generate_split_summary()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)
    
    def compute_split_metrics(self, results: List[PairAnalysisResult], label: str, rmse_tolerance: float) -> dict:
        """Compute metrics separately for synthetic, interpolation, extrapolation."""
        if not results:
            return {"total": 0}
        
        n = len(results)
        
        # Recompute tolerance compliance using the appropriate tolerance
        n_within_tol = sum(1 for r in results if r.curve_max_deviation <= rmse_tolerance)
        mean_shape = np.mean([r.shape_correlation for r in results])
        n_both = sum(1 for r in results if r.curve_max_deviation <= rmse_tolerance and r.shape_correlation > 0.9)
        
        # Synthetic temperature metrics
        synth_rmses = [r.rmse_synthetic for r in results if r.rmse_synthetic < 100]
        synth_r2s = [r.r2_synthetic_vs_apelblat for r in results]
        
        # Holdout Interpolation metrics  
        interp_r2s = [r.r2_holdout_interp_pred_vs_exp for r in results 
                      if r.holdout_applicable and not np.isnan(r.r2_holdout_interp_pred_vs_exp)]
        
        # Holdout Extrapolation metrics
        extrap_r2s = [r.r2_holdout_extrap_pred_vs_exp for r in results 
                      if r.holdout_applicable and not np.isnan(r.r2_holdout_extrap_pred_vs_exp)]
        
        # Known temperature metrics
        known_rmses = [r.rmse_known for r in results]
        known_r2s = [r.r2_known_pred_vs_exp for r in results]
        
        return {
            "label": label,
            "total_pairs": n,
            "rmse_tolerance": rmse_tolerance,
            
            # Overall Apelblat compliance
            "apelblat_compliance": {
                "pct_within_tolerance": round(n_within_tol / n * 100, 2),
                "mean_shape_correlation": round(mean_shape, 4),
                "pct_both_criteria": round(n_both / n * 100, 2)
            },
            
            # Known temperatures (training points)
            "known_temps": {
                "mean_rmse": round(np.mean(known_rmses), 4),
                "mean_r2": round(np.mean(known_r2s), 4)
            },
            
            # Synthetic temperature testing
            "synthetic_temps": {
                "mean_rmse": round(np.mean(synth_rmses), 4) if synth_rmses else None,
                "mean_r2": round(np.mean(synth_r2s), 4) if synth_r2s else None,
                "n_pairs": len(synth_rmses)
            },
            
            # Holdout interpolation
            "holdout_interpolation": {
                "mean_r2": round(np.mean(interp_r2s), 4) if interp_r2s else None,
                "std_r2": round(np.std(interp_r2s), 4) if interp_r2s else None,
                "n_pairs": len(interp_r2s)
            },
            
            # Holdout extrapolation
            "holdout_extrapolation": {
                "mean_r2": round(np.mean(extrap_r2s), 4) if extrap_r2s else None,
                "std_r2": round(np.std(extrap_r2s), 4) if extrap_r2s else None,
                "n_pairs": len(extrap_r2s)
            }
        }
    
    def save_split_results(self):
        """Save results with train/test labels."""
        print("\nSaving results...")
        rows = []
        
        for source, results in [("train", self.train_results), ("test", self.test_results)]:
            for r in results:
                rows.append({
                    "Source": source,
                    "Solute": r.solute, "Solvent": r.solvent,
                    "Num_Temps": r.num_temps,
                    "RMSE_Known": r.rmse_known,
                    "R2_Known": r.r2_known_pred_vs_exp,
                    "R2_Synthetic": r.r2_synthetic_vs_apelblat,
                    "RMSE_Synthetic": r.rmse_synthetic,
                    "R2_Holdout_Interp": r.r2_holdout_interp_pred_vs_exp,
                    "R2_Holdout_Extrap": r.r2_holdout_extrap_pred_vs_exp,
                    "Shape_Correlation": r.shape_correlation,
                    "Curve_Max_Deviation": r.curve_max_deviation,
                    "Curve_Within_Tolerance": r.curve_within_tolerance,
                    "Category": r.category,
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "pair_analysis.csv", index=False)
        print(f"  ✓ Saved pair_analysis.csv ({len(rows)} pairs total)")
    
    def generate_split_summary(self):
        """Generate comprehensive split summary with final combined metric."""
        print("\nGenerating split summary...")
        
        train_metrics = self.compute_split_metrics(self.train_results, "TRAIN", self.config.train_rmse_tolerance)
        test_metrics = self.compute_split_metrics(self.test_results, "TEST", self.config.test_rmse_tolerance)
        
        # Compute combined final metric
        # Weight: Shape (25%), Tolerance (25%), Synthetic R² (25%), Holdout (25%)
        def compute_final_score(m):
            if m["total_pairs"] == 0:
                return 0
            shape = m["apelblat_compliance"]["mean_shape_correlation"]
            tol = m["apelblat_compliance"]["pct_within_tolerance"] / 100
            synth = max(0, m["synthetic_temps"]["mean_r2"] or 0)
            interp = max(0, m["holdout_interpolation"]["mean_r2"] or 0)
            extrap = max(0, m["holdout_extrapolation"]["mean_r2"] or 0)
            holdout = (interp + extrap) / 2
            return round(0.25 * shape + 0.25 * tol + 0.25 * synth + 0.25 * holdout, 4)
        
        train_final = compute_final_score(train_metrics)
        test_final = compute_final_score(test_metrics)
        
        # Build comprehensive modular summary
        summary = {
            "experiment_info": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "train_pairs": train_metrics["total_pairs"],
                "test_pairs": test_metrics["total_pairs"],
            },
            "config": {
                "train_rmse_tolerance": self.config.train_rmse_tolerance,
                "test_rmse_tolerance": self.config.test_rmse_tolerance,
                "min_exp_apelblat_r2": self.config.min_exp_apelblat_r2,
                "min_temp_points": self.config.min_temp_points,
                "max_rmse_known": self.config.max_rmse_known,
            },
            "final_scores": {
                "train": train_final,
                "test": test_final,
                "generalization_gap": round(train_final - test_final, 4),
                "score_weights": "Shape 25% + Tolerance 25% + Synthetic R² 25% + Holdout 25%"
            },
            "train": train_metrics,
            "test": test_metrics,
        }
        
        # Save main summary
        with open(self.output_dir / "summary_statistics.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save modular metric files
        modular_dir = self.output_dir / "metrics"
        modular_dir.mkdir(exist_ok=True)
        
        # Save individual metric files
        for label, metrics in [("train", train_metrics), ("test", test_metrics)]:
            with open(modular_dir / f"{label}_apelblat_compliance.json", "w") as f:
                json.dump({"label": label, "rmse_tolerance": metrics["rmse_tolerance"], 
                          **metrics["apelblat_compliance"]}, f, indent=2)
            
            with open(modular_dir / f"{label}_known_temps.json", "w") as f:
                json.dump({"label": label, **metrics["known_temps"]}, f, indent=2)
            
            with open(modular_dir / f"{label}_synthetic_temps.json", "w") as f:
                json.dump({"label": label, **metrics["synthetic_temps"]}, f, indent=2)
            
            with open(modular_dir / f"{label}_holdout_interpolation.json", "w") as f:
                json.dump({"label": label, **metrics["holdout_interpolation"]}, f, indent=2)
            
            with open(modular_dir / f"{label}_holdout_extrapolation.json", "w") as f:
                json.dump({"label": label, **metrics["holdout_extrapolation"]}, f, indent=2)
        
        # Save final scores
        with open(modular_dir / "final_scores.json", "w") as f:
            json.dump(summary["final_scores"], f, indent=2)
        
        print(f"  ✓ Saved modular metrics to {modular_dir}")
        
        # Print summary
        for label, metrics in [("TRAIN", train_metrics), ("TEST", test_metrics)]:
            print(f"\n{'='*70}")
            print(f"{label} DATA SUMMARY ({metrics['total_pairs']} pairs)")
            print("="*70)
            
            print(f"\n📊 APELBLAT COMPLIANCE (tolerance = {metrics['rmse_tolerance']})")
            print(f"   Within tolerance:     {metrics['apelblat_compliance']['pct_within_tolerance']:.1f}%")
            print(f"   Mean shape corr:      {metrics['apelblat_compliance']['mean_shape_correlation']:.4f}")
            print(f"   Both criteria:        {metrics['apelblat_compliance']['pct_both_criteria']:.1f}%")
            
            print(f"\n📍 KNOWN TEMPERATURES (training points)")
            print(f"   Mean RMSE:            {metrics['known_temps']['mean_rmse']:.4f}")
            print(f"   Mean R²:              {metrics['known_temps']['mean_r2']:.4f}")
            
            print(f"\n🔮 SYNTHETIC TEMPERATURES ({metrics['synthetic_temps']['n_pairs']} pairs)")
            if metrics['synthetic_temps']['mean_rmse']:
                print(f"   Mean RMSE:            {metrics['synthetic_temps']['mean_rmse']:.4f}")
                print(f"   Mean R²:              {metrics['synthetic_temps']['mean_r2']:.4f}")
            
            print(f"\n📈 HOLDOUT INTERPOLATION ({metrics['holdout_interpolation']['n_pairs']} pairs)")
            if metrics['holdout_interpolation']['mean_r2']:
                print(f"   Mean R²:              {metrics['holdout_interpolation']['mean_r2']:.4f} ± {metrics['holdout_interpolation']['std_r2']:.4f}")
            
            print(f"\n📉 HOLDOUT EXTRAPOLATION ({metrics['holdout_extrapolation']['n_pairs']} pairs)")
            if metrics['holdout_extrapolation']['mean_r2']:
                print(f"   Mean R²:              {metrics['holdout_extrapolation']['mean_r2']:.4f} ± {metrics['holdout_extrapolation']['std_r2']:.4f}")
        
        print(f"\n{'='*70}")
        print("FINAL COMBINED SCORES")
        print("="*70)
        print(f"(Weighted: Shape 25% + Tolerance 25% + Synthetic R² 25% + Holdout 25%)")
        print(f"\n   TRAIN Score:          {train_final:.4f}")
        print(f"   TEST Score:           {test_final:.4f}")
        print(f"   Generalization Gap:   {train_final - test_final:.4f}")
        print("="*70)

    def generate_plots(self):
        print("\nGenerating plots...")
        pass

def main():
    config = ExperimentConfig()
    experiment = ApelblatExperiment(config)
    experiment.run_full_experiment()

if __name__ == "__main__":
    main()
