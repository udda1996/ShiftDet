"""
trainers/erm_trainer.py
========================
Standard Empirical Risk Minimization (ERM) trainer.

This is the ERM-CNN baseline described in the paper.
It trains the same CNN backbone as ShiftDet but with a
standard single-environment cross-entropy loss, with NO
IRM invariance penalty and NO multi-environment objective.

The purpose of this trainer is to provide a fair comparison
against ShiftDet: same architecture, same data, same compute
budget — only the training objective differs.

Usage
-----
  python experiments/train.py --config configs/default.yaml --method erm

Author: ShiftDet Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List
import os
import time


class ERMTrainer:
    """
    Standard ERM trainer for the CNN baseline.

    All training environments are merged into a single dataset
    and the model is trained with standard cross-entropy loss.
    This is the simplest possible baseline and directly shows
    the performance gap that ShiftDet's IRM objective closes.

    Parameters
    ----------
    model  : ShiftDetModel  (same architecture as ShiftDet)
    config : dict from default.yaml
    device : torch.device
    """

    def __init__(self,
                 model:  nn.Module,
                 config: dict,
                 device: torch.device):
        self.model  = model.to(device)
        self.config = config
        self.device = device

        tcfg = config['training']
        self.epochs   = tcfg['epochs']
        self.ckpt_dir = config['paths']['checkpoint_dir']

        self.optimizer = optim.Adam(
            model.parameters(),
            lr           = tcfg['learning_rate'],
            weight_decay = tcfg['weight_decay'],
        )
        self.criterion = nn.CrossEntropyLoss()

        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train_epoch(self,
                    merged_loader: DataLoader) -> Dict[str, float]:
        """
        One epoch of standard ERM training on the merged dataset.

        Parameters
        ----------
        merged_loader : DataLoader over ConcatDataset of all envs

        Returns
        -------
        metrics : {'loss', 'acc'}
        """
        self.model.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in merged_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss   = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss    += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

        return {
            'loss': total_loss    / total_samples,
            'acc':  total_correct / total_samples,
        }

    def train(self,
              train_loaders: Dict[str, DataLoader],
              val_loaders:   Dict[str, DataLoader]) -> List[dict]:
        """
        Full ERM training loop.

        Merges all training environment datasets into one,
        then trains with standard cross-entropy.

        Parameters
        ----------
        train_loaders : {env_name: DataLoader}
        val_loaders   : {env_name: DataLoader}

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        # Merge all training environments into a single DataLoader
        all_datasets = [loader.dataset
                        for loader in train_loaders.values()]
        merged_ds = ConcatDataset(all_datasets)
        merged_loader = DataLoader(
            merged_ds,
            batch_size  = self.config['training']['batch_size'],
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True,
        )

        history      = []
        best_val_acc = 0.0

        for epoch in range(self.epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(merged_loader)
            val_acc       = self._validate(val_loaders)

            epoch_metrics = {
                'epoch':  epoch,
                **train_metrics,
                'val_acc': val_acc,
                'time_s':  time.time() - t0,
            }
            history.append(epoch_metrics)

            print(f"[ERM] Epoch {epoch+1:3d}/{self.epochs} | "
                  f"Loss={train_metrics['loss']:.4f} | "
                  f"TrainAcc={train_metrics['acc']:.3f} | "
                  f"ValAcc={val_acc:.3f} | "
                  f"t={epoch_metrics['time_s']:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint('erm_best.pt')

        self._save_checkpoint('erm_final.pt')
        print(f"\n[ERM] Training complete. "
              f"Best val accuracy: {best_val_acc:.4f}")
        return history

    @torch.no_grad()
    def _validate(self, loaders: Dict[str, DataLoader]) -> float:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        for loader in loaders.values():
            for x, y in loader:
                x, y  = x.to(self.device), y.to(self.device)
                preds  = self.model(x).argmax(dim=1)
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
        print(f"  → ERM checkpoint saved: {path}")
