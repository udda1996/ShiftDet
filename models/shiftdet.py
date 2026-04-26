"""
models/shiftdet.py
==================
ShiftDet: full inference wrapper.

This module ties together the three components of ShiftDet:

  1. SignalBackbone + classifier head  (models/backbone.py)
  2. EnergyOODMonitor                  (models/ood_monitor.py)
  3. TTAAdapter                         (utils/tta.py)

At inference time, ShiftDetInference.predict() implements the
two-path decision described in the paper (Algorithm 1):

  Given received signal block y:
    z      = phi_theta(y)           # feature extraction
    S(y)   = -log sum_c exp(f_c(z)) # energy OOD score
    if S(y) > lambda:               # OOD detected
        recalibrate BN via TTA
    h_hat  = argmax_c f_c(z)        # detection decision

Author: ShiftDet Team
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from models.backbone    import ShiftDetModel, build_model
from models.ood_monitor import EnergyOODMonitor
from utils.tta          import TTAAdapter


class ShiftDetInference:
    """
    Full ShiftDet inference engine.

    Wraps the trained model, calibrated OOD monitor, and TTA
    adapter into a single predict() call that mirrors Algorithm 1
    in the paper.

    Parameters
    ----------
    model      : trained ShiftDetModel
    monitor    : calibrated EnergyOODMonitor
    tta        : TTAAdapter (can be None to disable TTA)
    device     : torch.device
    """

    def __init__(self,
                 model:   ShiftDetModel,
                 monitor: EnergyOODMonitor,
                 tta:     Optional[TTAAdapter],
                 device:  torch.device):
        self.model   = model.to(device).eval()
        self.monitor = monitor
        self.tta     = tta
        self.device  = device

        # Counters for runtime statistics
        self.n_batches_total = 0
        self.n_batches_ood   = 0

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass (no TTA).

        Returns
        -------
        logits  : (B, C)
        energy  : (B,)
        """
        self.model.eval()
        logits = self.model(x)
        energy = EnergyOODMonitor.energy_score(logits)
        return logits, energy

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full ShiftDet inference for one batch.

        Parameters
        ----------
        x : (B, 2, L)  received I/Q signal batch (already on device)

        Returns
        -------
        dict with keys:
          'decisions'  : (B,)  int   predicted class {0=H0, 1=H1}
          'logits'     : (B, C) float final logits
          'energy'     : (B,)  float OOD energy scores
          'ood_flag'   : bool   True if this batch was flagged as OOD
          'tta_applied': bool   True if TTA was triggered
        """
        self.n_batches_total += 1
        x = x.to(self.device)

        # Step 1: initial forward pass to get energy score
        logits, energy = self._forward(x)

        # Step 2: OOD decision
        ood_flag   = self.monitor.batch_is_ood(energy)
        tta_applied = False

        # Step 3: TTA recalibration if OOD and TTA is enabled
        if ood_flag and self.tta is not None:
            logits      = self.tta.adapt(x, self.device)
            tta_applied = True
            self.n_batches_ood += 1

        # Step 4: detection decision
        decisions = logits.argmax(dim=1)

        return {
            'decisions':   decisions,
            'logits':      logits,
            'energy':      energy,
            'ood_flag':    ood_flag,
            'tta_applied': tta_applied,
        }

    def ood_rate(self) -> float:
        """Fraction of batches flagged as OOD so far."""
        if self.n_batches_total == 0:
            return 0.0
        return self.n_batches_ood / self.n_batches_total

    def reset_stats(self):
        """Reset runtime OOD counters."""
        self.n_batches_total = 0
        self.n_batches_ood   = 0


def build_shiftdet_inference(config:   dict,
                              ckpt_path: str,
                              thresh_path: str,
                              device:   torch.device,
                              enable_tta: bool = True
                              ) -> ShiftDetInference:
    """
    Convenience factory: load checkpoint + OOD threshold and
    return a ready-to-use ShiftDetInference object.

    Parameters
    ----------
    config      : dict from default.yaml
    ckpt_path   : path to model checkpoint (.pt)
    thresh_path : path to OOD threshold JSON
    device      : torch.device
    enable_tta  : if False, TTA is disabled (ShiftDet no-TTA ablation)

    Returns
    -------
    ShiftDetInference
    """
    import json

    # Load model
    model = build_model(config)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Load OOD monitor
    with open(thresh_path) as f:
        t = json.load(f)
    monitor           = EnergyOODMonitor(t['target_fpr'], t['score_type'])
    monitor.threshold = t['threshold']

    # Build TTA adapter
    tta = None
    if enable_tta:
        tcfg = config['tta']
        tta  = TTAAdapter(model, lr=tcfg['lr'], steps=tcfg['steps'])

    return ShiftDetInference(model, monitor, tta, device)
