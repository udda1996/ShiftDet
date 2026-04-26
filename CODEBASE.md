# ShiftDet — Full Codebase Documentation

> **How every script works, what it does, and how they are all connected.**

This document is a complete technical reference for the ShiftDet repository.
It describes every file's purpose, internal logic, key functions, and the
precise data flows between modules. Read this before modifying any file.

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Data Flow Diagram](#2-data-flow-diagram)
3. [configs/default.yaml](#3-configsdefaultyaml)
4. [data/channel_dataset.py](#4-datachannel_datasetpy)
5. [models/backbone.py](#5-modelsbackbonepy)
6. [models/ood_monitor.py](#6-modelsood_monitorpy)
7. [models/shiftdet.py](#7-modelsshiftdetpy)
8. [trainers/irm_trainer.py](#8-trainersirm_trainerpy)
9. [trainers/erm_trainer.py](#9-trainerserm_trainerpy)
10. [utils/tta.py](#10-utilsttapy)
11. [utils/metrics.py](#11-utilsmetricspy)
12. [experiments/train.py](#12-experimentstrainpy)
13. [experiments/evaluate.py](#13-experimentsevaulatepy)
14. [experiments/ablation.py](#14-experimentsablationpy)
15. [figures/plot_results.py](#15-figuresplot_resultspy)
16. [Cross-Module Dependency Map](#16-cross-module-dependency-map)
17. [Execution Order for a Full Experiment](#17-execution-order-for-a-full-experiment)

---

## 1. Repository Overview

ShiftDet is a signal detection framework built around three ideas:

| Idea | Implementation |
|------|---------------|
| Wireless mismatches are distribution shifts | `data/channel_dataset.py` — simulates 5 channel types |
| IRM learns shift-invariant features | `trainers/irm_trainer.py` — multi-env training |
| OOD monitor + TTA recovers at test time | `models/ood_monitor.py` + `utils/tta.py` |

The repository separates concerns into four layers:

```
Layer 1 (Data)       data/channel_dataset.py
Layer 2 (Models)     models/backbone.py  ←→  models/ood_monitor.py  ←→  models/shiftdet.py
Layer 3 (Training)   trainers/irm_trainer.py  |  trainers/erm_trainer.py
Layer 4 (Execution)  experiments/train.py  →  experiments/evaluate.py  →  experiments/ablation.py
                     figures/plot_results.py
```

Everything is configured from a single file: `configs/default.yaml`.

---

## 2. Data Flow Diagram

```
configs/default.yaml
        │
        ▼
data/channel_dataset.py
  ├── awgn_channel()
  ├── rayleigh_channel()
  ├── rician_channel()         ← unseen at training time
  ├── mimo_2x2_channel()       ← unseen at training time
  ├── hw_impaired_channel()    ← unseen at training time
  ├── ChannelDataset           → PyTorch Dataset  (N, 2, L) I/Q tensors
  └── build_dataloaders()      → {env_name: DataLoader}
                                         │
        ┌────────────────────────────────┘
        │         TRAINING PHASE
        ▼
trainers/irm_trainer.py                trainers/erm_trainer.py
  IRMTrainer.train()                     ERMTrainer.train()
  └── _irm_penalty() per env            └── merged ConcatDataset
  └── L_ShiftDet = ERM + λ·IRM         └── plain cross-entropy
        │                                        │
        ▼                                        ▼
  checkpoints/best.pt               checkpoints/erm_best.pt
        │
        │         CALIBRATION PHASE
        ▼
models/ood_monitor.py
  EnergyOODMonitor.calibrate()
  └── energy_score(logits) on ID val set
  └── threshold λ = percentile(1 - target_fpr)
        │
        ▼
  checkpoints/ood_threshold.json
        │
        │         INFERENCE PHASE
        ▼
models/shiftdet.py
  ShiftDetInference.predict(y)
  ├── phi_theta(y)       → z            [models/backbone.py]
  ├── S(y) = energy(z)                  [models/ood_monitor.py]
  ├── S(y) > λ?
  │     YES → utils/tta.py             TTAAdapter.adapt()
  │             └── entropy_loss → BN update → logits
  │     NO  → logits from forward pass
  └── decision = argmax_c logits
        │
        ▼
  experiments/evaluate.py
  └── compute_pd_at_pfa()               [utils/metrics.py]
  └── compute_auroc()
  └── compute_fpr_at_tpr()
  └── results/eval_results.json
        │
        ▼
  figures/plot_results.py
  └── fig1_roc_curves.pdf
  └── fig2_energy_histogram.pdf
  └── fig3_pd_vs_snr.pdf
  └── fig4_ablation_lambda.pdf
```

---

## 3. `configs/default.yaml`

**Purpose:** Single source of truth for all hyperparameters. Every other
script loads this file at startup. Changing a value here propagates to
every module without editing any Python file.

**Key sections and what they control:**

```yaml
data:
  n_samples_per_env: 10000   # Training samples per channel environment
  signal_length: 128         # N: complex I/Q samples per received block
  snr_range_train: [-5, 20]  # SNR range during training (dB)
  snr_range_test:  [-10, 20] # Wider range at test — tests generalisation
```

```yaml
environments:
  train:                     # These envs are seen during IRM training
    - name: awgn
    - name: rayleigh_slow    # Doppler 0 Hz
    - name: rayleigh_medium  # Doppler 50 Hz
    - name: rayleigh_fast    # Doppler 200 Hz
  test_unseen:               # These envs are NEVER seen during training
    - name: rician            # Rician K=5, Doppler 100 Hz
    - name: mimo_2x2         # 2x2 MIMO Rayleigh
    - name: hw_impaired      # IQ imbalance + phase noise
```

```yaml
model:
  backbone_channels: [32, 64, 128]  # CNN filter progression
  feature_dim: 256                   # Embedding dimensionality
  n_classes: 2                       # Binary: H0 or H1
```

```yaml
training:
  irm_lambda: 100.0         # IRM penalty weight (λ_IRM in paper Eq. 5)
  irm_anneal_epochs: 10     # Ramp λ from 1→100 over first 10 epochs
```

```yaml
ood:
  target_fpr: 0.05          # 5% of ID val samples may be falsely flagged OOD
  score: energy              # 'energy' | 'msp' | 'knn'
```

```yaml
tta:
  steps: 10                 # Gradient steps for BN recalibration per batch
  lr: 1.0e-4                # Learning rate for BN affine param update
```

**Connected to:** Every script imports this via `yaml.safe_load()`.

---

## 4. `data/channel_dataset.py`

**Purpose:** Simulates all wireless channel environments and provides
PyTorch-compatible datasets for every experiment.

**Responsibility boundary:** This file owns all signal generation and
channel simulation. No channel physics appear anywhere else in the repo.

### Signal Generation

```python
generate_bpsk(n, length)  # Returns complex (n, length): bits ∈ {-1, +1}
generate_qpsk(n, length)  # Returns complex (n, length): phase ∈ {π/4, 3π/4, 5π/4, 7π/4}
```

Both return constant-envelope complex baseband signals. Used for H1 samples.

### Channel Functions

Each function has the same signature pattern:
```python
channel_fn(signals: np.ndarray,   # (N, L) complex transmitted signals
           snr_db:  np.ndarray,   # (N,)   per-sample SNR
           **kwargs) -> np.ndarray # (N, L) complex received signals
```

| Function | Channel model | Training? | Key parameters |
|----------|--------------|-----------|----------------|
| `awgn_channel` | Pure AWGN | ✅ Train | None |
| `rayleigh_channel` | Flat Rayleigh fading | ✅ Train | `doppler_hz`, `fs` |
| `rician_channel` | Rician fading | ❌ Unseen | `k_factor`, `doppler_hz` |
| `mimo_2x2_channel` | 2×2 MIMO + MRC | ❌ Unseen | None |
| `hw_impaired_channel` | AWGN + IQ imb. + phase noise | ❌ Unseen | `eps`, `phi_deg`, `pn_lw_khz` |

**How noise power is computed** (same in every channel):
```
SNR_linear = 10^(SNR_dB / 10)
signal_power = mean(|s|²)
noise_variance = signal_power / SNR_linear
noise_std = sqrt(noise_variance / 2)   # per real component
noise = noise_std * (N(0,1) + j·N(0,1))
```

**Hardware impairment model** (important for unseen env):
```
IQ imbalance:  I_out = α·I - β·Q,   Q_out = β·I + α·Q
               α = (1+ε/2)cos(φ/2),  β = (1-ε/2)sin(φ/2)
Phase noise:   Wiener process increments ~ N(0, 2π·linewidth/fs)
               cumulative sum gives time-varying phase rotation
```

### ChannelDataset

```python
ChannelDataset(channel_type, n_samples, signal_length,
               snr_range_db, channel_kwargs, modulation, seed)
```

Produces balanced H0/H1 splits:
- **H0 samples:** zeros passed through `awgn_channel` only (noise only)
- **H1 samples:** BPSK/QPSK signals passed through the specified channel

Output tensor shape: `(2, L)` where dim 0 = [I, Q] real channels.
This 2-channel format is what the CNN backbone expects.

### build_dataloaders()

```python
build_dataloaders(config, split='train') -> Dict[str, DataLoader]
```

Reads `config['environments'][split]`, instantiates one `ChannelDataset`
per environment, wraps each in a `DataLoader`, and returns a dict keyed
by environment name. This is the primary interface used by both trainers
and the evaluation scripts.

**Connected to:**
- `trainers/irm_trainer.py` — calls `build_dataloaders(config, 'train')`
- `trainers/erm_trainer.py` — same
- `experiments/train.py` — orchestrates the call
- `experiments/evaluate.py` — calls `build_dataloaders(config, 'test_unseen')`
- `experiments/ablation.py` — calls both splits
- `figures/plot_results.py` — builds small datasets for SNR curves

---

## 5. `models/backbone.py`

**Purpose:** Defines the CNN feature extractor and the complete model
with classifier head. This is the neural network that all trainers train
and all inference scripts run.

### ConvBlock

```python
ConvBlock(in_channels, out_channels, kernel_size=3, pool_size=2)
```

The atomic building block:
```
Conv1d → BatchNorm1d → ReLU → MaxPool1d
```

The `BatchNorm1d` layer is architecturally deliberate — its `weight`
(γ) and `bias` (β) affine parameters are the **only parameters updated
during test-time adaptation**. The BN layer bridges training-time
statistics and test-time recalibration.

### SignalBackbone

```python
SignalBackbone(in_channels=2, backbone_channels=[32,64,128],
               kernel_size=3, feature_dim=256)
```

Stack of `ConvBlock` layers followed by:
- `AdaptiveAvgPool1d(1)` — global average pool, collapses time axis
- `Linear(backbone_channels[-1], feature_dim)` + `BatchNorm1d` + `ReLU`

Input: `(B, 2, L)` — batch of I/Q signal blocks
Output: `(B, feature_dim)` — fixed-length embedding vector z

The global average pool makes the backbone length-agnostic: it works
for any signal_length ≥ kernel_size without architecture changes.

### ShiftDetModel

```python
ShiftDetModel(backbone, n_classes=2, feature_dim=256)
```

Adds a single `Linear(feature_dim, n_classes)` classifier head on top
of the backbone. The head `w` is kept deliberately simple because the
IRM training objective (Eq. 5 in the paper) requires that `w` be
simultaneously optimal across all training environments.

Key methods used by other modules:

| Method | Used by | Purpose |
|--------|---------|---------|
| `forward(x, return_features=False)` | All inference scripts | Standard prediction |
| `get_bn_params()` | `utils/tta.py` | Returns BN γ, β for TTA optimizer |
| `freeze_except_bn()` | `utils/tta.py` | Freezes all params except BN affine |
| `unfreeze_all()` | `utils/tta.py` | Restores full gradient after TTA |

### build_model()

```python
build_model(config) -> ShiftDetModel
```

Factory function that reads `config['model']` and instantiates the
full model. This is the **only** model construction call in the entire
repo — all scripts call `build_model(config)` rather than constructing
the model directly.

**Connected to:**
- `trainers/irm_trainer.py` — trains the model
- `trainers/erm_trainer.py` — trains the baseline model
- `experiments/train.py` — calls `build_model(config)`
- `experiments/evaluate.py` — loads checkpoint into `build_model(config)`
- `experiments/ablation.py` — same
- `figures/plot_results.py` — same
- `models/shiftdet.py` — wraps the model in the inference engine
- `models/ood_monitor.py` — receives logits from the model

---

## 6. `models/ood_monitor.py`

**Purpose:** Computes a scalar out-of-distribution score for each
received signal block, and calibrates the detection threshold on
in-distribution validation data.

This module implements three OOD scores, selectable via config:

### Energy Score (default, recommended)

```
S(y) = -log Σ_c exp(f_c(y))
     = -logsumexp(logits, dim=1)
```

Derived from the log-partition function of the model.
- ID samples → large logits → low (negative) energy score
- OOD samples → small, flat logits → high (less negative) energy score

This is theoretically grounded (Liu et al., NeurIPS 2020) and
outperforms softmax confidence for detecting distribution shift.

### MSP Score (baseline comparison)

```
S(y) = 1 - max_c softmax(f(y))_c
```

From Hendrycks & Gimpel (ICLR 2017). Simple but overconfident on OOD.
Available for ablation comparison via `score: msp` in config.

### KNN Score

```
S(y) = distance to k-th nearest neighbour in ID training embeddings
```

From Sun et al. (ICML 2022). Non-parametric, requires storing all
training embeddings. Enabled via `score: knn` in config.

### Calibration

```python
EnergyOODMonitor.calibrate(model, val_loader, device) -> float
```

Runs the model over the ID validation set, collects energy scores,
and sets `self.threshold` to the `(1 - target_fpr)` quantile:

```
threshold λ = percentile(ID scores, 100 × (1 - target_fpr))
```

This guarantees that at most `target_fpr` fraction of true ID samples
will be falsely flagged as OOD. The threshold is saved to
`checkpoints/ood_threshold.json` and loaded at evaluation time.

### Inference

```python
monitor.is_ood(scores)       # Per-sample binary flag (B,) bool
monitor.batch_is_ood(scores) # True if majority of batch is OOD
```

`batch_is_ood()` is the primary decision gate used in
`models/shiftdet.py` to decide whether to trigger TTA.

**Connected to:**
- `models/backbone.py` — receives `logits` from `ShiftDetModel.forward()`
- `experiments/train.py` — calibration after training
- `experiments/evaluate.py` — loads threshold, computes OOD AUROC
- `models/shiftdet.py` — called inside `ShiftDetInference.predict()`
- `figures/plot_results.py` — energy scores used for histogram plot

---

## 7. `models/shiftdet.py`

**Purpose:** The high-level inference engine that ties everything
together into a single `predict()` call. This is the module a
downstream user would import to deploy ShiftDet in a real system.

### ShiftDetInference

```python
ShiftDetInference(model, monitor, tta, device)
```

Wraps:
- `ShiftDetModel` from `models/backbone.py`
- `EnergyOODMonitor` from `models/ood_monitor.py`
- `TTAAdapter` from `utils/tta.py`

### predict()

```python
result = shiftdet.predict(x)  # x: (B, 2, L)
```

Implements Algorithm 1 from the paper:

```
Step 1: z, logits = phi_theta(y)         # CNN forward pass
Step 2: S(y) = -logsumexp(logits)        # Energy OOD score
Step 3: if majority of batch has S > λ:  # OOD decision
            logits = TTA.adapt(y)        # BN recalibration
Step 4: h_hat = argmax_c logits          # Detection decision
```

Returns a dict with `decisions`, `logits`, `energy`, `ood_flag`,
`tta_applied`. The `ood_flag` and `tta_applied` fields are useful
for logging and runtime diagnostics.

Also tracks `n_batches_total` and `n_batches_ood` for a running
OOD rate: `shiftdet.ood_rate()` — useful for monitoring deployment
environments.

### build_shiftdet_inference()

```python
build_shiftdet_inference(config, ckpt_path, thresh_path,
                          device, enable_tta=True)
```

Convenience factory that loads a checkpoint, loads the OOD threshold
JSON, and returns a fully configured `ShiftDetInference` object ready
for `predict()` calls. Setting `enable_tta=False` creates the
"ShiftDet (no TTA)" ablation variant.

**Connected to:**
- `models/backbone.py` — owns the `ShiftDetModel`
- `models/ood_monitor.py` — owns the `EnergyOODMonitor`
- `utils/tta.py` — owns the `TTAAdapter`
- `experiments/evaluate.py` — uses this for the full ShiftDet eval path

---

## 8. `trainers/irm_trainer.py`

**Purpose:** Implements the multi-environment IRM training objective
that is the core algorithmic contribution of ShiftDet.

### The IRM Objective

Standard ERM minimises the average loss across all training data.
IRM adds an invariance penalty:

```
L_ShiftDet = (1/K) Σ_e L^e(w ∘ φ_θ)              [ERM term]
           + λ_IRM · Σ_e ||∇_{w|w=1} L^e(w ∘ φ_θ)||²  [IRM penalty]
```

The IRM penalty measures, for each environment e, how much the loss
gradient w.r.t. a **dummy scalar classifier w=1** deviates from zero.
A zero gradient means w=1 is already optimal for environment e, which
means the representation φ_θ has already made the detection problem
identical across environments — the invariance condition.

### _irm_penalty()

```python
def _irm_penalty(self, logits, labels):
    w_dummy = torch.ones(1, requires_grad=True)
    loss    = CrossEntropyLoss(logits * w_dummy, labels)
    grad    = torch.autograd.grad(loss, w_dummy, create_graph=True)[0]
    return grad.pow(2).sum()
```

The `create_graph=True` flag is essential — it keeps the computation
graph alive so that the outer `loss.backward()` can propagate gradients
through the penalty all the way back to φ_θ's parameters.

### Lambda Annealing

```python
def _get_irm_weight(self, epoch):
    if epoch < anneal_epochs:
        return 1.0 + (irm_lambda - 1.0) * (epoch / anneal_epochs)
    return irm_lambda
```

During the first `irm_anneal_epochs`, λ is ramped linearly from 1.0
to its target value. This is critical for stability: at epoch 0 the
model is random, and applying the full IRM penalty immediately causes
training instability because the gradients are very large.

### train_epoch()

For each gradient step:
1. Sample one batch from **every** training environment simultaneously
2. Compute per-environment ERM loss and IRM penalty
3. Average across environments
4. Combine: `loss = ERM_avg + λ · penalty_avg`
5. Backward + clip gradients (max norm 5.0) + step

Gradient clipping is important here because the IRM penalty involves
second-order gradients (`create_graph=True`) which can be larger than
first-order gradients.

### Checkpointing

Saves `best.pt` whenever validation accuracy improves, and `final.pt`
at the end of training. The `best.pt` checkpoint is what all
downstream scripts load.

**Connected to:**
- `data/channel_dataset.py` — receives `{env_name: DataLoader}` dicts
- `models/backbone.py` — trains `ShiftDetModel`
- `experiments/train.py` — instantiated and called here

---

## 9. `trainers/erm_trainer.py`

**Purpose:** Implements the ERM-CNN baseline described in Table I of
the paper. Same architecture as ShiftDet, trained with standard
cross-entropy on a merged dataset.

### Key difference from IRMTrainer

```python
# IRMTrainer: separate loaders per environment, IRM penalty
for env_name, loader in loaders.items():
    logits = model(x_env)
    erm_loss += CrossEntropy(logits, y_env)
    irm_penalty += grad_penalty(logits, y_env)
loss = erm_avg + lambda * irm_avg

# ERMTrainer: ALL environments merged into ONE ConcatDataset
merged_ds = ConcatDataset([loader.dataset for loader in loaders.values()])
logits = model(x_batch)   # batch may mix any environment
loss   = CrossEntropy(logits, y_batch)
```

By merging environments, ERM loses all knowledge of which environment
each sample came from. The model can and does learn spurious
correlations specific to individual environments (e.g., Rayleigh
fading statistics), which break when tested on unseen environments.

Saves checkpoints to `erm_best.pt` and `erm_final.pt` to avoid
overwriting ShiftDet's `best.pt`.

**Connected to:**
- `data/channel_dataset.py` — same loaders as IRMTrainer
- `models/backbone.py` — trains the same `ShiftDetModel` architecture
- `experiments/train.py` — instantiated when `--method erm` is passed

---

## 10. `utils/tta.py`

**Purpose:** Implements Test-Time Adaptation (TENT algorithm, Wang et al.
ICLR 2021) for BN parameter recalibration when OOD is detected.

### Why only BatchNorm parameters?

The BN running statistics (mean μ, variance σ²) encode the marginal
distribution of intermediate feature activations. When the channel
environment shifts, these statistics misalign — the BN layers normalize
using statistics from the training distribution, which are wrong for
the test distribution. Updating only BN parameters:

1. Preserves the invariant convolutional weights learned during IRM training
2. Is fast (few parameters: 2 × feature_dim per BN layer)
3. Is stable (BN parameters are smooth functions of batch statistics)

### State Snapshot and Reset

```python
self._bn_state_snapshot = self._snapshot_bn_state()  # captured at init
# ... after adaptation ...
self._restore_bn_state()  # resets γ, β, running_mean, running_var
```

After adapting to batch t, the BN state is reset to the pre-adaptation
snapshot before processing batch t+1. This makes TTA **stateless across
batches** — each batch is adapted independently. Without this reset,
BN statistics would drift progressively across batches, which is harmful
when the test distribution is non-stationary.

### adapt()

```python
def adapt(self, x, device) -> torch.Tensor:
    # 1. Freeze all parameters except BN affine (γ, β)
    self.model.freeze_except_bn()
    self.model.train()      # BN must be in train mode to update stats

    # 2. T gradient steps minimising entropy
    for step in range(self.steps):
        logits = self.model(x)
        loss   = entropy_loss(logits)   # H(x) = -Σ p_c log p_c
        loss.backward()
        optimizer.step()

    # 3. Final prediction with adapted BN
    self.model.eval()
    logits = self.model(x)

    # 4. Reset BN state (stateless across batches)
    self._restore_bn_state()
    self.model.unfreeze_all()
    return logits
```

### entropy_loss()

```
H(x) = -Σ_c p_c log(p_c),  p_c = softmax(f_c(z))
```

Minimising entropy pushes the model toward confident (low-entropy)
predictions. On the test distribution this forces BN statistics to
align with the incoming data. The intuition: if the model is uncertain
about every test sample, the BN normalisation is wrong — entropy
minimisation corrects it.

**Connected to:**
- `models/backbone.py` — calls `freeze_except_bn()`, `get_bn_params()`, `unfreeze_all()`
- `models/ood_monitor.py` — TTA is triggered only when OOD flag is raised
- `models/shiftdet.py` — `TTAAdapter` is held as `self.tta`
- `experiments/evaluate.py` — `TTAAdapter` instantiated here for eval
- `experiments/ablation.py` — TTA steps swept in Ablation B

---

## 11. `utils/metrics.py`

**Purpose:** All quantitative evaluation metrics used in the paper.
This module has no dependencies on other ShiftDet modules — it is
purely numerical (NumPy + scikit-learn).

### compute_pd_at_pfa()

```python
compute_pd_at_pfa(scores, labels, pfa_target=1e-3) -> float
```

Primary paper metric. Sets the detection threshold at the
`(1 - pfa_target)` quantile of H0 scores, then measures the fraction
of H1 scores that exceed it. This is the Neyman-Pearson operating
point: fix P_FA, maximise P_D.

### compute_roc()

```python
compute_roc(scores, labels, n_points=200) -> (pfa_arr, pd_arr)
```

Evaluates (P_FA, P_D) at 200 threshold values to produce the full ROC
curve. Used by `figures/plot_results.py` for Figure 1.

### compute_auroc()

```python
compute_auroc(ood_scores, ood_labels) -> float
```

Area Under the ROC Curve for OOD detection quality (not detection
performance). `ood_labels` is 1 for OOD test samples, 0 for ID
validation samples. A score of 1.0 means perfect OOD-vs-ID separation.

### compute_fpr_at_tpr()

```python
compute_fpr_at_tpr(ood_scores, ood_labels, tpr_target=0.95) -> float
```

FPR@95TPR: the standard OOD detection benchmark metric. At the threshold
where 95% of OOD samples are correctly flagged, what fraction of ID
samples are incorrectly flagged? Lower is better.

### compute_detection_error()

```
min_threshold [ 0.5 × (FPR + FNR) ]
```

Balanced error metric useful when ID and OOD sets are equal size.

### summarize_results() and snr_curve_summary()

Pretty-printing utilities that format result dicts into ASCII tables
matching the layout of Table I and the SNR curves in the paper.

**Connected to:**
- `experiments/evaluate.py` — imports all metric functions
- `experiments/ablation.py` — imports `compute_pd_at_pfa`
- `figures/plot_results.py` — imports `compute_roc` for ROC plots

---

## 12. `experiments/train.py`

**Purpose:** The main entry point for training. Orchestrates the full
training pipeline from data loading through model training to OOD
monitor calibration.

### Command line

```bash
python experiments/train.py --config configs/default.yaml --method irm
python experiments/train.py --config configs/default.yaml --method erm
```

### Execution sequence

```
1. load_config(args.config)         # Load YAML
2. set_seed(config['training']['seed'])  # Reproducibility
3. build_dataloaders(config, 'train')    # All training environments
4. build_dataloaders(val_config, 'train') # ID validation set (2000 samples)
5. build_model(config)              # Instantiate CNN
6. IRMTrainer(model, config, device).train(...)    # if --method irm
   ERMTrainer(model, config, device).train(...)    # if --method erm
7. Load best.pt checkpoint
8. EnergyOODMonitor.calibrate(model, val_loader)  # Only for IRM
9. Save threshold to checkpoints/ood_threshold.json
```

### set_seed()

Sets seeds for `torch`, `torch.cuda`, `numpy`, `random`, and fixes
`cudnn.deterministic = True`. This is essential for reproducibility
of the IRM training, which is sensitive to initialisation due to the
non-convex IRM penalty.

**Connected to:**
- `data/channel_dataset.py` — `build_dataloaders()`
- `models/backbone.py` — `build_model()`
- `models/ood_monitor.py` — `EnergyOODMonitor.calibrate()`
- `trainers/irm_trainer.py` — `IRMTrainer`
- `trainers/erm_trainer.py` — `ERMTrainer`

---

## 13. `experiments/evaluate.py`

**Purpose:** Loads trained checkpoints and evaluates ShiftDet and all
baselines across every unseen test environment. Produces Table I from
the paper and saves `results/eval_results.json`.

### Two evaluation paths

**ShiftDet (full)** — uses `evaluate_with_tta()`:
```
for each batch in test_loader:
    logits = model(x)
    energy = OOD_monitor.energy_score(logits)
    if batch_is_ood(energy):
        logits = TTA_adapter.adapt(x)   # recalibrated logits
    scores.append(logits[:, 1])         # H1 logit as detection score
```

**ShiftDet (no TTA) and ERM-CNN** — uses `collect_scores_and_labels()`:
```
for each batch in test_loader:
    logits = model(x)             # standard forward pass, no TTA
    scores.append(logits[:, 1])
```

### OOD AUROC computation

For each test environment, ID energy scores (from AWGN test set) and
OOD energy scores (from the unseen environment) are concatenated with
binary labels (0=ID, 1=OOD) and passed to `compute_auroc()`. This
measures how well the energy score separates the two distributions —
a high AUROC means the OOD monitor can reliably detect the shift.

**Connected to:**
- `data/channel_dataset.py` — `build_dataloaders(config, 'test_unseen')`
- `models/backbone.py` — `build_model()` + checkpoint loading
- `models/ood_monitor.py` — loads threshold JSON, computes energy scores
- `utils/tta.py` — `TTAAdapter` for full ShiftDet path
- `utils/metrics.py` — `compute_pd_at_pfa`, `compute_auroc`, `compute_fpr_at_tpr`

---

## 14. `experiments/ablation.py`

**Purpose:** Reproduces Figure 4 of the paper by sweeping three
hyperparameters and measuring their effect on average P_D.

### Ablation A — IRM lambda sweep

```
lambdas = [0, 1, 10, 50, 100, 200, 500]
```

For each lambda: trains a new model from scratch (fast mode: 20 epochs),
calibrates OOD monitor, evaluates on all unseen envs, records avg P_D.

Lambda=0 is equivalent to ERM (no IRM penalty). The sweep shows the
optimal lambda and the sensitivity of the method to this hyperparameter.

**Important:** This ablation requires training K separate models, making
it the most compute-intensive script in the repo. Use `--full` flag for
full 100-epoch training per lambda, or omit it for fast 20-epoch mode.

### Ablation B — TTA steps sweep

```
tta_steps = [0, 1, 5, 10, 20, 50]
```

Reuses the existing `best.pt` checkpoint. Only changes the number of
gradient steps T in `TTAAdapter`. Steps=0 disables TTA entirely.
Shows the tradeoff between adaptation quality and inference latency.

### Ablation C — OOD threshold FPR sweep

```
fpr_values = [0.01, 0.05, 0.10, 0.20, 0.50]
```

Reuses the existing `best.pt` checkpoint. Recalibrates the OOD monitor
with different `target_fpr` values. Lower FPR = stricter threshold =
fewer batches trigger TTA. Higher FPR = looser = more TTA triggers.
Shows how OOD monitor aggressiveness affects end-to-end detection.

All ablation results are saved as JSON to `results/` for use by
`figures/plot_results.py`.

**Connected to:**
- `data/channel_dataset.py` — builds test loaders
- `models/backbone.py` — `build_model()` for lambda sweep
- `models/ood_monitor.py` — recalibrated for each sweep point
- `trainers/irm_trainer.py` — used in lambda sweep (Ablation A)
- `utils/tta.py` — `TTAAdapter` with varying `steps`
- `utils/metrics.py` — `compute_pd_at_pfa`

---

## 15. `figures/plot_results.py`

**Purpose:** Generates all publication-quality PDF figures from
the paper. Run after `evaluate.py` has produced `eval_results.json`.

### Figure 1 — ROC Curves

`plot_roc_curves(config, device, out_dir)`

Loads ShiftDet (full), ShiftDet (no TTA), and ERM-CNN from their
respective checkpoints. Evaluates all on the hardware-impaired channel
(hardest unseen env). Computes ROC curves via `compute_roc()` and
plots P_D vs P_FA on a semi-log x-axis. A vertical grey line marks
the paper's operating point P_FA=10⁻³.

### Figure 2 — Energy Score Histogram

`plot_energy_histogram(config, device, out_dir)`

Runs ShiftDet's encoder on AWGN (ID), Rician (OOD), and HW-impaired
(OOD) test sets. Plots overlapping histograms of energy scores with
density normalisation. The calibrated threshold λ is shown as a
vertical dashed line. Clear bimodal separation validates the OOD
monitor's ability to distinguish training from unseen environments.

### Figure 3 — P_D vs SNR Curves

`plot_pd_vs_snr(config, device, out_dir)`

Evaluates ShiftDet and ERM-CNN at fixed SNR points from -10 to +20 dB
on the Rician channel. For each SNR, a fresh dataset is generated with
`snr_range_db=(snr, snr)`. Plots the resulting P_D curves to show at
what SNR each method reaches P_D=0.9 (the standard benchmark target).

### Figure 4 — Ablation: Lambda Sweep

`plot_ablation_lambda(out_dir)`

Loads pre-computed results from `results/ablation_lambda.json` (produced
by `ablation.py`) and plots Avg P_D vs λ_IRM for each test environment.
The selected λ=100 is marked with a vertical dotted line.

**Connected to:**
- `data/channel_dataset.py` — builds evaluation datasets per figure
- `models/backbone.py` — loads checkpoints for each method
- `models/ood_monitor.py` — computes energy scores for Fig 2
- `utils/tta.py` — used in Fig 1 full ShiftDet path
- `utils/metrics.py` — `compute_roc`, `compute_pd_at_pfa`

---

## 16. Cross-Module Dependency Map

```
configs/default.yaml
    └──► ALL scripts (loaded by every experiment entry point)

data/channel_dataset.py
    └──► trainers/irm_trainer.py
    └──► trainers/erm_trainer.py
    └──► experiments/train.py
    └──► experiments/evaluate.py
    └──► experiments/ablation.py
    └──► figures/plot_results.py

models/backbone.py
    └──► models/shiftdet.py         (owns ShiftDetModel instance)
    └──► models/ood_monitor.py      (receives logits)
    └──► utils/tta.py               (calls freeze_except_bn, get_bn_params)
    └──► trainers/irm_trainer.py    (trains the model)
    └──► trainers/erm_trainer.py    (trains the model)
    └──► experiments/train.py       (build_model)
    └──► experiments/evaluate.py    (build_model + load ckpt)
    └──► experiments/ablation.py    (build_model + load ckpt)
    └──► figures/plot_results.py    (build_model + load ckpt)

models/ood_monitor.py
    └──► models/shiftdet.py         (held as self.monitor)
    └──► experiments/train.py       (calibration after training)
    └──► experiments/evaluate.py    (OOD AUROC computation)
    └──► experiments/ablation.py    (recalibrated per sweep point)
    └──► figures/plot_results.py    (energy scores for histogram)

models/shiftdet.py
    └──► models/backbone.py         (imports ShiftDetModel, build_model)
    └──► models/ood_monitor.py      (imports EnergyOODMonitor)
    └──► utils/tta.py               (imports TTAAdapter)
    └──► experiments/evaluate.py    (build_shiftdet_inference)

trainers/irm_trainer.py
    └──► experiments/train.py       (instantiated here)
    └──► experiments/ablation.py    (re-instantiated per lambda)

trainers/erm_trainer.py
    └──► experiments/train.py       (instantiated when --method erm)

utils/tta.py
    └──► models/backbone.py         (calls model methods)
    └──► models/shiftdet.py         (TTAAdapter held as self.tta)
    └──► experiments/evaluate.py    (instantiated for eval)
    └──► experiments/ablation.py    (steps swept in Ablation B)

utils/metrics.py
    └──► experiments/evaluate.py    (PD, AUROC, FPR@95)
    └──► experiments/ablation.py    (PD per sweep point)
    └──► figures/plot_results.py    (ROC curves, PD vs SNR)
```

---

## 17. Execution Order for a Full Experiment

The scripts must be run in this order. Each step produces outputs
that the next step depends on.

```
Step 1 — Train ShiftDet (IRM)
    python experiments/train.py --config configs/default.yaml --method irm
    Produces: checkpoints/best.pt
              checkpoints/ood_threshold.json

Step 2 — Train ERM-CNN baseline
    python experiments/train.py --config configs/default.yaml --method erm
    Produces: checkpoints/erm_best.pt

Step 3 — Evaluate all methods on unseen environments
    python experiments/evaluate.py --config configs/default.yaml
    Produces: results/eval_results.json
    Requires: checkpoints/best.pt
              checkpoints/erm_best.pt
              checkpoints/ood_threshold.json

Step 4 — Run ablation studies
    python experiments/ablation.py --ablation lambda    # trains K models
    python experiments/ablation.py --ablation tta_steps # reuses best.pt
    python experiments/ablation.py --ablation threshold  # reuses best.pt
    Produces: results/ablation_lambda.json
              results/ablation_tta_steps.json
              results/ablation_threshold.json

Step 5 — Generate paper figures
    python figures/plot_results.py --config configs/default.yaml
    Produces: figures/output/fig1_roc_curves.pdf
              figures/output/fig2_energy_histogram.pdf
              figures/output/fig3_pd_vs_snr.pdf
              figures/output/fig4_ablation_lambda.pdf
    Requires: checkpoints/best.pt
              checkpoints/erm_best.pt
              checkpoints/ood_threshold.json
              results/ablation_lambda.json
```

**Dependency summary:**

```
train.py  ──► evaluate.py ──► plot_results.py
          ──► ablation.py ──► plot_results.py
```

`train.py` must complete before any other script. `evaluate.py` and
`ablation.py` can be run in parallel after `train.py` finishes.
`plot_results.py` requires both to have completed.

---

*End of ShiftDet Codebase Documentation*
