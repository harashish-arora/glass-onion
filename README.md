# [Paper Title TBD] - Anonymous ICML Submission

> **Code and data for the paper: "[Paper Title]"**

---

## Overview

This repository contains the complete codebase for our molecular solubility prediction framework. We address solubility prediction across two distinct regimes:

- **Regime I**: Aqueous solubility prediction (single solvent - water)
- **Regime II**: Multi-solvent solubility prediction with temperature dependence

Our approach combines physics-informed molecular descriptors with learned interaction representations to achieve state-of-the-art performance on both tasks.

---

## Repository Structure

```
├── regime-i/                   # Aqueous solubility (single solvent)
│   ├── train.py               # Main training script
│   ├── featurizer.py          # Molecular feature extraction
│   ├── data/                  # Train/test datasets
│   └── all_datasets/          # Additional benchmark datasets (AqSolDB, ESOL, SC2)
│
├── regime-ii/                  # Multi-solvent solubility with temperature
│   ├── train.py               # Main training pipeline
│   ├── train_transformer.py   # Interaction Transformer training
│   ├── council.py             # Council feature extraction (24 members)
│   ├── featurizer.py          # Full molecular featurization
│   ├── ablations.py           # Ablation study scripts
│   ├── data/                  # Train/test datasets
│   └── all_datasets/          # BigSol 1.0, BigSol 2.0, Leeds datasets
│
├── baselines/                  # Baseline implementations
│   ├── regime-i/              # Baselines for aqueous solubility
│   │   ├── SolTranNet_paper/  # SolTranNet baseline
│   │   ├── SolubNetD/         # SolubNetD baseline
│   │   ├── aqsolpred/         # AqSolPred baseline
│   │   ├── chemprop_baseline/ # ChemProp baseline
│   │   ├── fastprop_baseline/ # FastProp baseline
│   │   ├── ulrich/            # Ulrich et al. baseline
│   │   ├── tayyebi/           # Tayyebi baseline
│   │   └── gnn-benchmarks/    # GNN baselines (GIN, GCN, GAT, MPNN)
│   └── regime-ii/             # Baselines for multi-solvent
│       ├── fastsolv_1.0/      # FastSolv on BigSol 1.0
│       ├── fastsolv_2.0/      # FastSolv on BigSol 2.0
│       ├── fastsolv_leeds/    # FastSolv on Leeds
│       ├── chemprop_baseline/ # ChemProp baseline
│       └── rilood.py          # RiLOOD baseline
│
├── llm/                        # LLM Expert Evaluation Pipeline
│   ├── survey.py              # Main LLM survey pipeline
│   ├── survey_gpt.py          # GPT-4 evaluation
│   ├── survey_claude.py       # Claude evaluation
│   ├── survey_deepseek.py     # DeepSeek evaluation
│   └── llm_solubility_analysis.py  # Analysis scripts
│
├── apelblat/                   # Apelblat Equation Validation
│   ├── apelblat_experiment.py # Temperature-dependence validation
│   └── APELBLAT_EXPERIMENT_EXPLAINED.md
│
└── requirements.txt            # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 12.x (for GPU acceleration, optional)

### Setup

```bash
# Clone the repository
git clone [ANONYMOUS_URL]
cd [REPO_NAME]

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.1+ (with CUDA support for GPU)
- RDKit 2024+
- CatBoost
- scikit-learn
- pandas, numpy

---

## Quick Start

### Regime I: Aqueous Solubility

```bash
cd regime-i

# Train the model
python train.py

# Evaluate on test set
python eval_test.py
```

### Regime II: Multi-Solvent Solubility

```bash
cd regime-ii

# Step 1: Generate features (if not already done)
python generate_features.py

# Step 2: Train the Interaction Transformer
python train_transformer.py

# Step 3: Train the final model
python train.py

# Evaluate on test set
python eval_test.py
```

---

## Method Overview

### Core Architecture

1. **Council of Features (24 Members)**: Physics-informed molecular descriptors covering:
   - Global properties (MolLogP, TPSA, H-bond donors/acceptors, etc.)
   - Functional super-groups (Acidic, Basic, Protic, Polar, Halogen, Aromatic)
   - Thermodynamic proxies (Joback T_m, Abraham parameters)

2. **Interaction Transformer**: Cross-attention mechanism that learns solute-solvent interactions:
   - Projects solute and solvent council features into embedding space
   - Multi-head cross-attention captures molecular compatibility
   - Outputs learned interaction features

3. **Gradient Boosting Predictor**: CatBoost model with:
   - Physical feature backbone
   - Learned interaction embeddings
   - Temperature-modulated interaction terms
   - Monotonicity constraints for thermodynamic consistency

---

## Datasets

### Regime I
| Dataset | Train | Test | Description |
|---------|-------|------|-------------|
| AqSolDB | 8,000 | 1,000 | Curated aqueous solubility |
| ESOL | 1,024 | 128 | Delaney ESOL benchmark |
| SC2 | 2,300 | 287 | Solubility Challenge 2 |

### Regime II
| Dataset | Train | Test | Description |
|---------|-------|------|-------------|
| BigSol 1.0 | ~50,000 | ~6,000 | Multi-solvent solubility |
| BigSol 2.0 | ~60,000 | ~7,500 | Extended BigSol |
| Leeds | ~3,000 | ~400 | Temperature-varied data |

---

## Experiments

### Running Ablations

```bash
cd regime-ii
python ablations.py
```

This runs comprehensive ablations on:
- Council feature groups
- Interaction Transformer components
- Temperature encoding strategies
- Monotonicity constraints

### Running Baselines

```bash
cd baselines/regime-i
python baselining_generic_methods.py  # ML baselines
python gnn-benchmarks-regime-i/baselining_gnn_methods.py  # GNN baselines

cd baselines/regime-ii
python baselining_generic_methods.py  # ML baselines
python baselining_gnn_methods.py  # GNN baselines
```

### LLM Expert Evaluation

```bash
cd llm

# Set API keys
export GEMINI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run evaluation
python survey.py  # Uses Gemini by default
python survey_claude.py
python survey_gpt.py
```

### Apelblat Validation

```bash
cd apelblat
python apelblat_experiment.py
```

Validates that model predictions follow the Apelblat equation for temperature-dependent solubility.

---

## Results

Results are saved in respective directories:
- `regime-i/model/` - Trained models
- `regime-ii/model/` - Trained models and transformer weights
- `baselines/*/benchmark_results/` - Baseline comparison results
- `llm/llm_survey_pipeline_results*.json` - LLM evaluation results
- `apelblat/apelblat_results/` - Apelblat validation metrics

---

## Citation

```bibtex
@inproceedings{anonymous2026solubility,
  title={[Paper Title TBD]},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## License

This code is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

We thank the creators of the open-source tools and datasets used in this work, including RDKit, CatBoost, and the various solubility benchmark datasets.
