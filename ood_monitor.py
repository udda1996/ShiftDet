"""
models/ood_monitor.py
=====================
Energy-Based OOD Monitor for ShiftDet.

Theory
------
The energy score (Liu et al., NeurIPS 2020) for a sample x is:

    E(x; f) = -log Σ_c exp(f_c(x))

where f_c(x) is the c-th logit output of the model.
- In-distribution (ID) samples produce high logits
  → low (negative) energy score
- Out-of-distribution (OOD) samples produce low, flat logits
  → high (less negative) energy score

A threshold λ is calibrated on a held-out validation set from
the training environments to achieve a target false OOD alarm
rate (e.g., 5%).  At test time:
  - E(x) ≤ λ  →  IN-distribution   (no adaptation needed)
  - E(x) >  λ  →  OUT-of-distribution (trigger TTA)

Additional scores (MSP, KNN) are also provided for comparison.

Author: ShiftDet Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EnergyOODMonitor:
    """
    Energy-based OOD detector.

    Calibration is performed once on a validation set from the
    training environments, after which the threshold lambda is
    fixed for all test batches.

    Parameters
    ----------
    target_fpr : float
        Desired false OOD alarm rate on in-distribution data.
        The threshold lambda is chosen so that this fraction of
        ID validation samples are incorrectly flagged as OOD.
        Default: 0.05 (5%)
    score_type : str
        'energy' (default) | 'msp' | 'knn'
    """

    def __init__(self,
                 target_fpr: float = 0.05,
                 score_type: str   = 'energy'):
        self.target_fpr = target_fpr
        self.score_type = score_type
        self.threshold  = None          # set after calibration
        self._id_embeddings = None      # for KNN score

    # ---------------------------------------------------------------- #
    # Score computation                                                  #
    # ---------------------------------------------------------------- #

    @staticmethod
    def energy_score(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score from logits.

        E(x) = -log Σ_c exp(f_c(x))
             = -logsumexp(logits, dim=1)

        Higher energy  →  more OOD.

        Parameters
        ----------
        logits : (B, C)

        Returns
        -------
        scores : (B,)  energy scores (higher = more OOD)
        """
        return -torch.logsumexp(logits, dim=1)

    @staticmethod
    def msp_score(logits: torch.Tensor) -> torch.Tensor:
        """
        Maximum Softmax Probability (MSP) score.
        Baseline from Hendrycks & Gimpel (ICLR 2017).

        MSP(x) = 1 - max_c softmax(f(x))_c
        Higher score → more OOD.

        Parameters
        ----------
        logits : (B, C)

        Returns
        -------
        scores : (B,)
        """
        probs = F.softmax(logits, dim=1)
        return 1.0 - probs.max(dim=1).values

    @staticmethod
    def knn_score(embeddings: torch.Tensor,
                  id_embeddings: torch.Tensor,
                  k: int = 10) -> torch.Tensor:
        """
        K-Nearest Neighbour distance score.
        From Sun et al. (ICML 2022): "Out-of-Distribution Detection
        with Deep Nearest Neighbors".

        OOD score = distance to k-th nearest ID training embedding.
        Higher distance → more OOD.

        Parameters
        ----------
        embeddings    : (B, D)  test embeddings
        id_embeddings : (N, D)  stored training embeddings
        k             : number of neighbours

        Returns
        -------
        scores : (B,)
        """
        # Compute pairwise L2 distances: (B, N)
        diff  = (embeddings.unsqueeze(1) -
                 id_embeddings.unsqueeze(0))           # (B, N, D)
        dists = torch.norm(diff, dim=-1)               # (B, N)
        knn_dists, _ = torch.topk(dists, k,
                                   dim=1, largest=False)  # (B, k)
        return knn_dists[:, -1]  # distance to k-th neighbor

    def compute_score(self,
                      logits:     torch.Tensor,
                      embeddings: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """
        Dispatch to the configured score function.

        Parameters
        ----------
        logits     : (B, C)
        embeddings : (B, D)  required only for score_type='knn'

        Returns
        -------
        scores : (B,)  higher = more OOD
        """
        if self.score_type == 'energy':
            return self.energy_score(logits)
        elif self.score_type == 'msp':
            return self.msp_score(logits)
        elif self.score_type == 'knn':
            assert embeddings is not None, \
                "KNN score requires embeddings"
            assert self._id_embeddings is not None, \
                "Call store_id_embeddings() before using KNN score"
            return self.knn_score(embeddings, self._id_embeddings)
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")

    # ---------------------------------------------------------------- #
    # Calibration                                                        #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def calibrate(self,
                  model:      nn.Module,
                  val_loader: torch.utils.data.DataLoader,
                  device:     torch.device) -> float:
        """
        Calibrate the OOD threshold on in-distribution validation data.

        Sets self.threshold such that `target_fpr` fraction of
        ID validation samples have score > threshold.

        Parameters
        ----------
        model      : trained ShiftDetModel (in eval mode)
        val_loader : DataLoader for ID validation set
        device     : torch device

        Returns
        -------
        threshold : float  (also stored as self.threshold)
        """
        model.eval()
        all_scores   = []
        all_embeddings = []

        for x, _ in val_loader:
            x = x.to(device)
            if self.score_type == 'knn':
                logits, z = model(x, return_features=True)
                all_embeddings.append(z.cpu())
            else:
                logits = model(x)
            scores = self.compute_score(logits)
            all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores).numpy()

        if self.score_type == 'knn':
            self._id_embeddings = torch.cat(all_embeddings)

        # Threshold at the (1 - target_fpr) quantile of ID scores
        # → target_fpr fraction of ID samples exceed the threshold
        self.threshold = float(
            np.percentile(all_scores, 100 * (1 - self.target_fpr)))

        print(f"[OOD Monitor] Calibrated threshold = {self.threshold:.4f} "
              f"(target FPR = {self.target_fpr:.1%})")
        return self.threshold

    # ---------------------------------------------------------------- #
    # Inference                                                          #
    # ---------------------------------------------------------------- #

    def is_ood(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Binary OOD decision for a batch of score values.

        Parameters
        ----------
        scores : (B,) OOD scores

        Returns
        -------
        flags : (B,) bool tensor, True = OOD
        """
        assert self.threshold is not None, \
            "Call calibrate() before is_ood()"
        return scores > self.threshold

    def batch_is_ood(self, scores: torch.Tensor) -> bool:
        """
        Returns True if the MAJORITY of samples in the batch are OOD.
        Used to decide whether to trigger TTA for the entire batch.

        Parameters
        ----------
        scores : (B,)
        """
        return self.is_ood(scores).float().mean().item() > 0.5
