# ShiftDet 📡

**Distribution Shift-Aware Robust Signal Detection via OOD Generalization**

> Official simulation code for the IEEE journal paper:  
> *"ShiftDet: Distribution Shift-Aware Robust Signal Detection via OOD Generalization"*

---

## Overview

ShiftDet is a signal detection framework that treats classical wireless
parameter mismatches (SNR mismatch, channel model variation, hardware
impairments, waveform novelty) as instances of **distribution shift**,
and addresses them jointly using **out-of-distribution (OOD) generalization**
theory.

The three core components are:

| Component | Role |
|-----------|------|
| **Invariant Feature Extractor** | CNN backbone trained with IRM across multiple channel environments |
| **Energy-Based OOD Monitor** | Flags test signals deviating from the training distribution |
| **Test-Time Adaptation (TTA)** | Recalibrates BN statistics on flagged OOD batches via entropy minimization |

---

## Repository Structure

```
shiftdet/
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── data/
│   └── channel_dataset.py    # Synthetic channel data generator
├── models/
│   ├── backbone.py           # CNN invariant feature extractor
│   ├── ood_monitor.py        # Energy-based OOD scoring
│   └── shiftdet.py           # Full ShiftDet model wrapper
├── trainers/
│   ├── erm_trainer.py        # Standard ERM baseline trainer
│   └── irm_trainer.py        # IRM multi-environment trainer
├── utils/
│   ├── metrics.py            # PD, PFA, ROC, AUROC helpers
│   └── tta.py                # Test-time adaptation (TENT)
├── experiments/
│   ├── train.py              # Main training entry point
│   ├── evaluate.py           # Evaluation across environments
│   └── ablation.py           # Ablation study runner
├── figures/
│   └── plot_results.py       # Reproduce all paper figures
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourname/shiftdet.git
cd shiftdet
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, NumPy, SciPy, Matplotlib,
PyYAML, scikit-learn, tqdm

---

## Quick Start

### 1. Train ShiftDet (IRM backbone)
```bash
python experiments/train.py --config configs/default.yaml --method irm
```

### 2. Train ERM-CNN baseline
```bash
python experiments/train.py --config configs/default.yaml --method erm
```

### 3. Evaluate all methods on unseen environments
```bash
python experiments/evaluate.py --config configs/default.yaml
```

### 4. Run ablation study
```bash
python experiments/ablation.py --config configs/default.yaml
```

### 5. Generate all paper figures
```bash
python figures/plot_results.py
```

---

## Environments

| Environment | Split | Channel | Notes |
|-------------|-------|---------|-------|
| AWGN | Train | AWGN | SNR ∈ [−5, 20] dB |
| Rayleigh | Train | Flat fading | fd ∈ {0, 50, 200} Hz |
| Rician | **Test (unseen)** | Rician K=5 | fd = 100 Hz |
| MIMO | **Test (unseen)** | 2×2 Rayleigh | Spatial multiplexing |
| HW-Impaired | **Test (unseen)** | AWGN + IQ imbalance + phase noise | ε=0.1, φ=5° |

---

## Citation

```bibtex
@article{shiftdet2025,
  title   = {{ShiftDet}: Distribution Shift-Aware Robust Signal Detection
             via {OOD} Generalization},
  author  = {Author One and Author Two and Author Three},
  journal = {IEEE Transactions on Communications},
  year    = {2025}
}
```
