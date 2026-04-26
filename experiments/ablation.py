"""
experiments/ablation.py
========================
Ablation study runner for ShiftDet.

Reproduces Figure 4 of the paper by sweeping three key
hyperparameters and measuring average P_D across unseen
test environments:

  Ablation A: IRM penalty weight  lambda_IRM  in {0,1,10,50,100,200,500}
  Ablation B: TTA gradient steps  T           in {1,5,10,20,50}
  Ablation C: OOD threshold FPR   target_fpr  in {0.01,0.05,0.10,0.20}

For Ablation A, a separate model must be trained for each
lambda value. This script supports a "fast" mode that trains
for a reduced number of epochs to speed up the sweep.

For Ablations B and C, a single trained ShiftDet checkpoint
is reused -- only inference-time parameters are swept.

Usage
-----
  # Full ablation (trains one model per lambda -- slow)
  python experiments/ablation.py --config configs/default.yaml \
         --ablation lambda --full

  # Fast ablation (fewer training epochs per lambda)
  python experiments/ablation.py --config configs/default.yaml \
         --ablation lambda

  # TTA steps ablation (uses existing best.pt checkpoint)
  python experiments/ablation.py --config configs/default.yaml \
         --ablation tta_steps

  # OOD threshold ablation
  python experiments/ablation.py --config configs/default.yaml \
         --ablation threshold

Author: ShiftDet Team
"""

import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import json
import copy
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.channel_dataset import build_dataloaders, ChannelDataset
from models.backbone       import build_model
from models.ood_monitor    import EnergyOODMonitor
from trainers.irm_trainer  import IRMTrainer
from utils.tta             import TTAAdapter
from utils.metrics         import compute_pd_at_pfa, summarize_results


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(config, ckpt_name, device):
    model = build_model(config)
    ckpt  = torch.load(
        os.path.join(config['paths']['checkpoint_dir'], ckpt_name),
        map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def build_test_loaders(config, device):
    """Build unseen test environment loaders."""
    return build_dataloaders(config, split='test_unseen')


def build_id_val_loader(config):
    """Build a small ID validation loader for OOD calibration."""
    ds = ChannelDataset(
        channel_type  = 'awgn',
        n_samples     = 2000,
        signal_length = config['data']['signal_length'],
        snr_range_db  = tuple(config['data']['snr_range_train']),
        seed          = 999,
    )
    return DataLoader(ds, batch_size=256, shuffle=False)


@torch.no_grad()
def evaluate_model(model, test_loaders, ood_monitor,
                   tta_steps, device, pfa_target=1e-3):
    """
    Evaluate a model across all unseen test environments.

    Returns average P_D across environments.
    """
    tta = TTAAdapter(model, lr=1e-4, steps=tta_steps)
    all_pd = []

    for env_name, loader in test_loaders.items():
        scores, labels = [], []

        for x, y in loader:
            x = x.to(device)
            with torch.no_grad():
                logits = model(x)
                energy = EnergyOODMonitor.energy_score(logits)

            if ood_monitor.batch_is_ood(energy):
                logits = tta.adapt(x, device)

            scores.append(logits[:, 1].cpu().numpy())
            labels.append(y.numpy())

        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        pd     = compute_pd_at_pfa(scores, labels, pfa_target)
        all_pd.append(pd)
        print(f"    {env_name:<20} PD = {pd:.4f}")

    return float(np.mean(all_pd))


# ------------------------------------------------------------------ #
# Ablation A: IRM lambda sweep                                        #
# ------------------------------------------------------------------ #

def ablation_lambda(config, device, fast=False):
    """
    Train one model per lambda value and evaluate on unseen envs.
    """
    lambdas     = [0, 1, 10, 50, 100, 200, 500]
    results     = {}
    pfa_target  = config['eval']['pfa_target']

    train_loaders = build_dataloaders(config, split='train')
    val_config    = copy.deepcopy(config)
    val_config['data']['n_samples_per_env'] = 1000
    val_loaders   = build_dataloaders(val_config, split='train')
    test_loaders  = build_test_loaders(config, device)
    id_val_loader = build_id_val_loader(config)

    for lam in lambdas:
        print(f"\n{'='*55}")
        print(f"  Ablation lambda = {lam}")
        print(f"{'='*55}")

        cfg = copy.deepcopy(config)
        cfg['training']['irm_lambda'] = lam
        if fast:
            cfg['training']['epochs']        = 20
            cfg['training']['irm_anneal_epochs'] = 5

        model   = build_model(cfg)
        trainer = IRMTrainer(model, cfg, device)
        trainer.train(train_loaders, val_loaders)

        # Load best checkpoint
        ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'],
                                  'best.pt')
        model = build_model(cfg)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device).eval()

        # Calibrate OOD monitor
        monitor = EnergyOODMonitor(
            target_fpr = cfg['ood']['target_fpr'])
        monitor.calibrate(model, id_val_loader, device)

        avg_pd = evaluate_model(
            model, test_loaders, monitor,
            tta_steps=cfg['tta']['steps'],
            device=device, pfa_target=pfa_target)

        results[lam] = avg_pd
        print(f"  lambda={lam}  →  Avg PD = {avg_pd:.4f}")

    # Save results
    out_path = os.path.join(config['paths']['results_dir'],
                             'ablation_lambda.json')
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'lambdas': lambdas,
                   'avg_pd':  [results[l] for l in lambdas]}, f,
                  indent=2)

    print(f"\nAblation A results saved to: {out_path}")
    print("\nlambda   | Avg PD")
    print("---------|-------")
    for l in lambdas:
        print(f"{l:<9}| {results[l]:.4f}")
    return results


# ------------------------------------------------------------------ #
# Ablation B: TTA steps sweep                                         #
# ------------------------------------------------------------------ #

def ablation_tta_steps(config, device):
    """
    Sweep TTA gradient steps T using the existing best checkpoint.
    """
    tta_steps_list = [0, 1, 5, 10, 20, 50]
    results        = {}
    pfa_target     = config['eval']['pfa_target']

    model         = load_model(config, 'best.pt', device)
    test_loaders  = build_test_loaders(config, device)
    id_val_loader = build_id_val_loader(config)

    monitor = EnergyOODMonitor(
        target_fpr = config['ood']['target_fpr'])
    monitor.calibrate(model, id_val_loader, device)

    print("\nAblation B: TTA gradient steps")
    for T in tta_steps_list:
        print(f"\n  T = {T} steps")
        avg_pd = evaluate_model(
            model, test_loaders, monitor,
            tta_steps=T, device=device,
            pfa_target=pfa_target)
        results[T] = avg_pd
        print(f"  T={T}  →  Avg PD = {avg_pd:.4f}")

    out_path = os.path.join(config['paths']['results_dir'],
                             'ablation_tta_steps.json')
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'tta_steps': tta_steps_list,
                   'avg_pd':    [results[T]
                                  for T in tta_steps_list]}, f,
                  indent=2)

    print(f"\nAblation B results saved to: {out_path}")
    print("\nT steps  | Avg PD")
    print("---------|-------")
    for T in tta_steps_list:
        print(f"{T:<9}| {results[T]:.4f}")
    return results


# ------------------------------------------------------------------ #
# Ablation C: OOD threshold FPR sweep                                 #
# ------------------------------------------------------------------ #

def ablation_threshold(config, device):
    """
    Sweep the OOD monitor calibration target FPR.
    Lower FPR = stricter threshold (fewer OOD flags, less TTA).
    Higher FPR = looser threshold (more OOD flags, more TTA).
    """
    fpr_values = [0.01, 0.05, 0.10, 0.20, 0.50]
    results    = {}
    pfa_target = config['eval']['pfa_target']

    model         = load_model(config, 'best.pt', device)
    test_loaders  = build_test_loaders(config, device)
    id_val_loader = build_id_val_loader(config)

    print("\nAblation C: OOD threshold target FPR")
    for fpr in fpr_values:
        print(f"\n  target_fpr = {fpr}")
        monitor = EnergyOODMonitor(target_fpr=fpr)
        monitor.calibrate(model, id_val_loader, device)

        avg_pd = evaluate_model(
            model, test_loaders, monitor,
            tta_steps=config['tta']['steps'],
            device=device, pfa_target=pfa_target)
        results[fpr] = avg_pd
        print(f"  FPR={fpr}  →  Avg PD = {avg_pd:.4f}")

    out_path = os.path.join(config['paths']['results_dir'],
                             'ablation_threshold.json')
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'target_fpr': fpr_values,
                   'avg_pd':     [results[f]
                                   for f in fpr_values]}, f,
                  indent=2)

    print(f"\nAblation C results saved to: {out_path}")
    print("\nFPR      | Avg PD")
    print("---------|-------")
    for fpr in fpr_values:
        print(f"{fpr:<9}| {results[fpr]:.4f}")
    return results


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/default.yaml')
    parser.add_argument('--ablation',
                        choices=['lambda', 'tta_steps', 'threshold'],
                        default='lambda',
                        help='Which hyperparameter to ablate')
    parser.add_argument('--full', action='store_true',
                        help='Full training epochs (slow). '
                             'Default: fast mode (20 epochs)')
    args   = parser.parse_args()
    config = load_config(args.config)
    device = get_device()

    print(f"\n{'='*55}")
    print(f"  ShiftDet Ablation Study: {args.ablation}")
    print(f"{'='*55}\n")

    if args.ablation == 'lambda':
        ablation_lambda(config, device, fast=not args.full)
    elif args.ablation == 'tta_steps':
        ablation_tta_steps(config, device)
    elif args.ablation == 'threshold':
        ablation_threshold(config, device)


if __name__ == '__main__':
    main()
