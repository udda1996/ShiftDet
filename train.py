"""
experiments/train.py
====================
Main training entry point for ShiftDet.

Usage
-----
# Train ShiftDet (IRM backbone):
  python experiments/train.py --config configs/default.yaml --method irm

# Train ERM-CNN baseline:
  python experiments/train.py --config configs/default.yaml --method erm

# Override any config key inline:
  python experiments/train.py --config configs/default.yaml \
         --method irm --training.irm_lambda 200

Author: ShiftDet Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import torch
import numpy as np
import random

from data.channel_dataset   import build_dataloaders
from models.backbone         import build_model
from models.ood_monitor      import EnergyOODMonitor
from trainers.irm_trainer    import IRMTrainer


# ------------------------------------------------------------------ #
# Utility helpers                                                      #
# ------------------------------------------------------------------ #

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    return device


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Train ShiftDet')
    parser.add_argument('--config', type=str,
                        default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--method', type=str,
                        choices=['irm', 'erm'],
                        default='irm',
                        help="Training method: 'irm' (ShiftDet) or 'erm' (baseline)")
    args = parser.parse_args()

    # ---- Load config ----
    config = load_config(args.config)
    config['training']['method'] = args.method
    print(f"\n{'='*60}")
    print(f"  ShiftDet Training  |  Method: {args.method.upper()}")
    print(f"{'='*60}\n")

    # ---- Setup ----
    set_seed(config['training']['seed'])
    device = get_device()

    # ---- Build data loaders ----
    print("Building training environments...")
    train_loaders = build_dataloaders(config, split='train')
    print(f"  Training environments: {list(train_loaders.keys())}")

    # Use first training environment as validation set
    # (separate split drawn from same distribution)
    val_config = dict(config)
    val_config['data'] = dict(config['data'])
    val_config['data']['n_samples_per_env'] = 2000
    val_loaders = build_dataloaders(val_config, split='train')

    # ---- Build model ----
    print("\nBuilding ShiftDet model...")
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"  Total trainable parameters: {n_params:,}")

    # ---- Train ----
    if args.method == 'irm':
        print(f"\nStarting IRM training "
              f"(λ={config['training']['irm_lambda']}, "
              f"anneal_epochs={config['training']['irm_anneal_epochs']})...")
        trainer = IRMTrainer(model, config, device)
    else:
        # ERM baseline: use IRM trainer with lambda=0
        print("\nStarting ERM training (lambda=0)...")
        config['training']['irm_lambda'] = 0.0
        trainer = IRMTrainer(model, config, device)

    history = trainer.train(train_loaders, val_loaders)

    # ---- Calibrate OOD monitor ----
    print("\nCalibrating OOD monitor on ID validation data...")
    monitor = EnergyOODMonitor(
        target_fpr  = config['ood']['target_fpr'],
        score_type  = config['ood']['score'],
    )
    # Build a combined val loader (all train envs merged)
    from torch.utils.data import ConcatDataset, DataLoader
    all_val_ds = ConcatDataset(
        [loader.dataset for loader in val_loaders.values()])
    combined_val_loader = DataLoader(
        all_val_ds,
        batch_size  = config['training']['batch_size'],
        shuffle     = False,
        num_workers = 0,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(config['paths']['checkpoint_dir'], 'best.pt'),
            map_location=device
        )['model_state_dict']
    )
    model = model.to(device)
    monitor.calibrate(model, combined_val_loader, device)

    # Save threshold alongside the checkpoint
    import json
    threshold_path = os.path.join(config['paths']['checkpoint_dir'],
                                  'ood_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': monitor.threshold,
                   'score_type': monitor.score_type,
                   'target_fpr': monitor.target_fpr}, f, indent=2)
    print(f"  OOD threshold saved to: {threshold_path}")

    print("\nTraining pipeline complete.")
    print(f"Checkpoints saved in: {config['paths']['checkpoint_dir']}")


if __name__ == '__main__':
    main()
