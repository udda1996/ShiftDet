"""
utils/tta.py
============
Test-Time Adaptation (TTA) for ShiftDet.

When the OOD monitor flags an incoming batch as out-of-distribution,
this module recalibrates the model's BatchNorm affine parameters
(gamma, beta) via entropy minimization (TENT, Wang et al., ICLR 2021).

Why only BatchNorm parameters?
-------------------------------
- BN running statistics (mean, variance) encode the marginal
  distribution of intermediate features, which is precisely what
  changes under distribution shift.
- Updating only BN parameters (not conv weights) preserves the
  invariant representations learned during IRM training.
- BN affine parameters are few (2 × feature_dim per layer), making
  adaptation fast and stable even on small test batches.

Why entropy minimization?
--------------------------
On in-distribution data, a well-trained model makes confident
(low-entropy) predictions. Distribution shift increases prediction
entropy. Minimizing entropy on the test batch pushes the BN
statistics to align with the new distribution, recovering confidence.

The BN parameters are RESET after each batch adaptation, so the
adaptation is local and does not accumulate across batches.

Author: ShiftDet Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional


class TTAAdapter:
    """
    Test-Time Adaptation via entropy minimization on BN parameters.

    Parameters
    ----------
    model      : ShiftDetModel (IRM-trained)
    lr         : float  learning rate for BN parameter adaptation
    steps      : int    number of gradient steps per test batch
    reset_each_batch : bool
        If True (default), BN params are reset to their pre-adaptation
        values after each batch. This prevents drift across batches.
    """

    def __init__(self,
                 model:             nn.Module,
                 lr:                float = 1e-4,
                 steps:             int   = 10,
                 reset_each_batch:  bool  = True):
        self.model            = model
        self.lr               = lr
        self.steps            = steps
        self.reset_each_batch = reset_each_batch

        # Store a snapshot of BN parameters for resetting
        self._bn_state_snapshot = self._snapshot_bn_state()

    # ---------------------------------------------------------------- #
    # BN state management                                                #
    # ---------------------------------------------------------------- #

    def _snapshot_bn_state(self) -> dict:
        """
        Capture current BN layer state (weight, bias, running stats).
        This snapshot is restored after each batch if reset_each_batch=True.
        """
        snapshot = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                snapshot[name] = {
                    'weight':       module.weight.data.clone(),
                    'bias':         module.bias.data.clone(),
                    'running_mean': module.running_mean.data.clone(),
                    'running_var':  module.running_var.data.clone(),
                }
        return snapshot

    def _restore_bn_state(self):
        """Restore BN parameters to the pre-adaptation snapshot."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                s = self._bn_state_snapshot[name]
                module.weight.data.copy_(s['weight'])
                module.bias.data.copy_(s['bias'])
                module.running_mean.data.copy_(s['running_mean'])
                module.running_var.data.copy_(s['running_var'])

    # ---------------------------------------------------------------- #
    # Entropy loss                                                       #
    # ---------------------------------------------------------------- #

    @staticmethod
    def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        """
        Shannon entropy of the softmax distribution.

        H(x) = -Σ_c p_c log(p_c)

        Minimizing this encourages confident (low-entropy) predictions
        on the test batch, aligning the model to the new distribution.

        Parameters
        ----------
        logits : (B, C)

        Returns
        -------
        mean entropy : scalar tensor
        """
        probs = F.softmax(logits, dim=1)
        # Clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-8))
        entropy   = -(probs * log_probs).sum(dim=1)
        return entropy.mean()

    # ---------------------------------------------------------------- #
    # Adaptation                                                         #
    # ---------------------------------------------------------------- #

    def adapt(self,
              x:      torch.Tensor,
              device: torch.device) -> torch.Tensor:
        """
        Adapt BN parameters to the test batch x, then make a prediction.

        Steps
        -----
        1. Freeze all parameters except BN affine params (gamma, beta).
        2. Run `steps` gradient steps minimizing H(model(x)).
        3. Make prediction with adapted model.
        4. If reset_each_batch, restore BN params to pre-adaptation state.

        Parameters
        ----------
        x      : (B, 2, L)  test batch (already on device)
        device : torch.device

        Returns
        -------
        logits : (B, C)  predictions after adaptation
        """
        # 1. Freeze all except BN
        self.model.freeze_except_bn()
        self.model.train()   # BN must be in train mode to update stats

        bn_params = self.model.get_bn_params()
        optimizer = torch.optim.Adam(bn_params, lr=self.lr)

        # 2. Gradient steps
        for step in range(self.steps):
            optimizer.zero_grad()
            logits = self.model(x)
            loss   = self.entropy_loss(logits)
            loss.backward()
            optimizer.step()

        # 3. Final prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        # 4. Reset BN if required
        if self.reset_each_batch:
            self._restore_bn_state()

        # Restore full gradient computation
        self.model.unfreeze_all()

        return logits


"""
utils/metrics.py
================
Evaluation metrics for ShiftDet.

Implements:
- Detection probability PD at a fixed PFA
- ROC curve computation
- AUROC (Area Under ROC Curve)
- FPR@95TPR (standard OOD detection metric)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict


def compute_pd_at_pfa(scores:    np.ndarray,
                      labels:    np.ndarray,
                      pfa_target: float = 1e-3) -> float:
    """
    Compute detection probability PD at a fixed false-alarm rate PFA.

    Convention: label=1 means signal present (H1), label=0 means noise (H0).
    The detector threshold is set so that PFA = pfa_target on the H0 samples.

    Parameters
    ----------
    scores     : (N,)  detector scores (higher → more likely H1)
    labels     : (N,)  ground truth {0, 1}
    pfa_target : float  desired false-alarm probability

    Returns
    -------
    pd : float  detection probability at the specified PFA
    """
    h0_scores = scores[labels == 0]
    h1_scores = scores[labels == 1]

    # Threshold: (1 - pfa_target) quantile of H0 scores
    # → fraction pfa_target of H0 scores exceed threshold
    threshold = np.percentile(h0_scores, 100 * (1 - pfa_target))

    pd = (h1_scores > threshold).mean()
    return float(pd)


def compute_roc(scores:  np.ndarray,
                labels:  np.ndarray,
                n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ROC curve (PD vs PFA).

    Parameters
    ----------
    scores   : (N,)  detector scores
    labels   : (N,)  ground truth
    n_points : number of operating points to evaluate

    Returns
    -------
    pfa_arr : (n_points,) false-alarm probabilities
    pd_arr  : (n_points,) detection probabilities
    """
    pfa_arr = np.linspace(1e-4, 1.0, n_points)
    pd_arr  = np.zeros(n_points)

    h0_scores = scores[labels == 0]
    h1_scores = scores[labels == 1]

    for i, pfa in enumerate(pfa_arr):
        threshold  = np.percentile(h0_scores, 100 * (1 - pfa))
        pd_arr[i]  = (h1_scores > threshold).mean()

    return pfa_arr, pd_arr


def compute_auroc(ood_scores: np.ndarray,
                  ood_labels: np.ndarray) -> float:
    """
    AUROC for OOD detection.

    Parameters
    ----------
    ood_scores : (N,)  OOD scores (higher = more OOD)
    ood_labels : (N,)  1 if OOD, 0 if ID

    Returns
    -------
    auroc : float
    """
    return float(roc_auc_score(ood_labels, ood_scores))


def compute_fpr_at_tpr(ood_scores: np.ndarray,
                       ood_labels: np.ndarray,
                       tpr_target: float = 0.95) -> float:
    """
    FPR at a fixed TPR (standard OOD benchmark metric).

    Parameters
    ----------
    ood_scores : (N,)  higher = more OOD
    ood_labels : (N,)  1=OOD, 0=ID
    tpr_target : target true positive rate (default: 95%)

    Returns
    -------
    fpr : float
    """
    fpr_arr, tpr_arr, _ = roc_curve(ood_labels, ood_scores)
    # Find FPR at the first point where TPR >= tpr_target
    idx = np.searchsorted(tpr_arr, tpr_target)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])


def summarize_results(results: Dict[str, dict]) -> str:
    """
    Format evaluation results as a printable table.

    Parameters
    ----------
    results : {env_name: {'pd': float, 'auroc': float, ...}}

    Returns
    -------
    table_str : formatted string
    """
    header = f"{'Environment':<20} {'PD@PFA=1e-3':>12} {'AUROC':>8}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for env, m in results.items():
        rows.append(f"{env:<20} {m.get('pd', float('nan')):>12.4f} "
                    f"{m.get('auroc', float('nan')):>8.4f}")
    rows.append(sep)
    return "\n".join(rows)
