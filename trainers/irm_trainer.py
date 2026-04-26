"""
trainers/irm_trainer.py
========================
Invariant Risk Minimization (IRM) Trainer for ShiftDet.

This module implements the multi-environment training loop that
is the heart of ShiftDet's distribution-shift robustness.

IRM Objective (Arjovsky et al., 2019)
--------------------------------------
Given K training environments, IRM jointly minimizes:

  L_ShiftDet = (1/K) Σ_e L^e(w ∘ φ_θ)          [ERM term]
             + λ_IRM Σ_e ||∇_{w|w=1} L^e(w ∘ φ_θ)||²  [IRM penalty]

The IRM penalty measures how much the gradient of the loss
w.r.t. the fixed scalar classifier w (at w=1) deviates from
zero in each environment. A zero gradient means w=1 is optimal
for that environment — i.e., the representation already makes
the classification problem "the same" across environments.

λ_IRM is annealed from 1 to its full value over the first
`irm_anneal_epochs` epochs, following the schedule of Arjovsky
et al. to stabilise early training.

Practical note on the IRM dummy classifier
-------------------------------------------
In the original IRM formulation, the penalty uses a *scalar*
dummy classifier w ∈ R (not the real classifier head). This is
because a scalar multiplier is the simplest test of whether the
linear head on top of phi is optimal. ShiftDet follows this
convention: the real `model.classifier` is trained normally via
the ERM term, while the IRM penalty is computed using a
separate scalar `w_dummy` fixed at 1.0.

Author: ShiftDet Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List
import os
import time


class IRMTrainer:
    """
    Multi-environment IRM trainer for ShiftDet.

    Parameters
    ----------
    model     : ShiftDetModel
    config    : dict from default.yaml
    device    : torch.device
    """

    def __init__(self,
                 model:  nn.Module,
                 config: dict,
                 device: torch.device):
        self.model  = model.to(device)
        self.config = config
        self.device = device

        tcfg = config['training']
        self.epochs        = tcfg['epochs']
        self.irm_lambda    = tcfg['irm_lambda']
        self.anneal_epochs = tcfg['irm_anneal_epochs']
        self.ckpt_dir      = config['paths']['checkpoint_dir']

        self.optimizer = optim.Adam(
            model.parameters(),
            lr           = tcfg['learning_rate'],
            weight_decay = tcfg['weight_decay'],
        )
        self.criterion = nn.CrossEntropyLoss()

        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    # IRM penalty computation                                            #
    # ---------------------------------------------------------------- #

    def _irm_penalty(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the IRM invariance penalty for one environment.

        We use the gradient-based surrogate penalty from Arjovsky (2019):

            penalty = ||∇_{w|w=1} L(w · logits, labels)||²

        where w is a scalar dummy variable initialized to 1.0.
        The gradient of the CE loss w.r.t. w tells us how much
        the loss would change if we rescaled the logits --- if this
        is zero, the current representation is already optimal for
        a trivial linear classifier, which is the IRM invariance condition.

        Parameters
        ----------
        logits : (B, C)  model logits for this environment
        labels : (B,)    ground-truth labels

        Returns
        -------
        penalty : scalar tensor
        """
        # Scalar dummy classifier, requires grad for penalty computation
        w_dummy = torch.ones(1, requires_grad=True, device=self.device)

        # Scale logits by dummy scalar (w=1 leaves logits unchanged)
        scaled_logits = logits * w_dummy

        loss = self.criterion(scaled_logits, labels)

        # Gradient of loss w.r.t. w_dummy at w=1
        grad = torch.autograd.grad(loss, w_dummy,
                                    create_graph=True)[0]
        return grad.pow(2).sum()

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def _get_irm_weight(self, epoch: int) -> float:
        """
        Anneal IRM penalty weight from 1.0 to irm_lambda
        over the first anneal_epochs epochs.

        Before annealing: weight = 1.0 (pure ERM, stable early training)
        After annealing:  weight = irm_lambda (full IRM regularization)
        """
        if epoch < self.anneal_epochs:
            # Linear interpolation
            return 1.0 + (self.irm_lambda - 1.0) * (epoch / self.anneal_epochs)
        return self.irm_lambda

    def train_epoch(self,
                    loaders:    Dict[str, DataLoader],
                    epoch:      int) -> Dict[str, float]:
        """
        Run one epoch of IRM training across all environments.

        Strategy: for each mini-batch step, sample one batch from
        each environment, compute the per-environment ERM loss and
        IRM penalty, and minimize their weighted sum.

        Parameters
        ----------
        loaders : dict  {env_name: DataLoader}
        epoch   : int   current epoch (for lambda annealing)

        Returns
        -------
        metrics : dict  {'loss': ..., 'erm_loss': ..., 'irm_penalty': ...,
                         'acc': ...}
        """
        self.model.train()
        irm_weight = self._get_irm_weight(epoch)

        # Create iterators for all environments
        env_iters = {name: iter(loader)
                     for name, loader in loaders.items()}

        # Determine steps per epoch: minimum batches across envs
        steps = min(len(loader) for loader in loaders.values())

        total_loss    = 0.0
        total_erm     = 0.0
        total_penalty = 0.0
        total_correct = 0
        total_samples = 0

        for step in range(steps):
            self.optimizer.zero_grad()

            erm_loss_sum    = 0.0
            irm_penalty_sum = 0.0
            batch_correct   = 0
            batch_samples   = 0

            for env_name, it in env_iters.items():
                try:
                    x, y = next(it)
                except StopIteration:
                    env_iters[env_name] = iter(loaders[env_name])
                    x, y = next(env_iters[env_name])

                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)

                # Per-environment ERM loss
                erm_loss = self.criterion(logits, y)
                erm_loss_sum += erm_loss

                # IRM invariance penalty for this environment
                penalty = self._irm_penalty(logits.detach().requires_grad_(),
                                             y)
                irm_penalty_sum += penalty

                # Accuracy tracking
                preds = logits.argmax(dim=1)
                batch_correct += (preds == y).sum().item()
                batch_samples += y.size(0)

            K = len(loaders)
            loss = (erm_loss_sum / K
                    + irm_weight * irm_penalty_sum / K)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss    += loss.item()
            total_erm     += (erm_loss_sum / K).item()
            total_penalty += (irm_penalty_sum / K).item()
            total_correct += batch_correct
            total_samples += batch_samples

        return {
            'loss':        total_loss    / steps,
            'erm_loss':    total_erm     / steps,
            'irm_penalty': total_penalty / steps,
            'acc':         total_correct / total_samples,
            'irm_weight':  irm_weight,
        }

    def train(self,
              train_loaders: Dict[str, DataLoader],
              val_loaders:   Dict[str, DataLoader]) -> List[dict]:
        """
        Full training loop with validation and checkpointing.

        Parameters
        ----------
        train_loaders : dict {env_name: DataLoader}
        val_loaders   : dict {env_name: DataLoader}  (ID validation)

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        history = []
        best_val_acc = 0.0

        for epoch in range(self.epochs):
            t0 = time.time()

            # Training
            train_metrics = self.train_epoch(train_loaders, epoch)

            # Validation on each training environment
            val_acc = self._validate(val_loaders)

            epoch_metrics = {
                'epoch': epoch,
                **train_metrics,
                'val_acc': val_acc,
                'time_s':  time.time() - t0,
            }
            history.append(epoch_metrics)

            print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                  f"Loss={train_metrics['loss']:.4f} | "
                  f"ERM={train_metrics['erm_loss']:.4f} | "
                  f"IRM_pen={train_metrics['irm_penalty']:.4f} | "
                  f"λ={train_metrics['irm_weight']:.1f} | "
                  f"TrainAcc={train_metrics['acc']:.3f} | "
                  f"ValAcc={val_acc:.3f} | "
                  f"t={epoch_metrics['time_s']:.1f}s")

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint('best.pt')

        self._save_checkpoint('final.pt')
        print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
        return history

    @torch.no_grad()
    def _validate(self, loaders: Dict[str, DataLoader]) -> float:
        """Average accuracy across all validation environments."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        for loader in loaders.values():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
        self.model.train()
        return total_correct / total_samples

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config':               self.config,
        }, path)
        print(f"  → Checkpoint saved: {path}")
