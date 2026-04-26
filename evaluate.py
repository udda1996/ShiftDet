"""
experiments/evaluate.py
========================
Evaluate ShiftDet and all baselines across unseen test environments.

Reproduces Table I from the paper:
  PD at PFA=1e-3 for {Rician, MIMO, HW-Impaired} environments.

Also computes:
  - OOD detection AUROC (energy score vs ground-truth ID/OOD label)
  - FPR@95TPR

Usage
-----
  python experiments/evaluate.py --config configs/default.yaml

Author: ShiftDet Team
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm

from data.channel_dataset import build_dataloaders, ChannelDataset
from models.backbone       import build_model
from models.ood_monitor    import EnergyOODMonitor
from utils.tta             import TTAAdapter
from utils.tta             import (compute_pd_at_pfa, compute_roc,
                                    compute_auroc, compute_fpr_at_tpr,
                                    summarize_results)
from torch.utils.data      import DataLoader


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config, checkpoint_name, device):
    model = build_model(config)
    ckpt  = torch.load(
        os.path.join(config['paths']['checkpoint_dir'], checkpoint_name),
        map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model


def load_ood_monitor(config) -> EnergyOODMonitor:
    thresh_path = os.path.join(config['paths']['checkpoint_dir'],
                               'ood_threshold.json')
    with open(thresh_path) as f:
        t = json.load(f)
    monitor           = EnergyOODMonitor(t['target_fpr'], t['score_type'])
    monitor.threshold = t['threshold']
    return monitor


@torch.no_grad()
def collect_scores_and_labels(model, loader, device):
    """
    Run the model over a DataLoader and collect:
      - detection scores (max logit value, used as H1 score)
      - ground-truth binary labels (0=H0, 1=H1)
      - energy scores (for OOD evaluation)
    """
    all_scores  = []
    all_labels  = []
    all_energy  = []

    model.eval()
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        logits = model(x)
        # Detection score: logit for H1 class
        scores = logits[:, 1].cpu().numpy()
        energy = EnergyOODMonitor.energy_score(logits).cpu().numpy()

        all_scores.append(scores)
        all_labels.append(y.numpy())
        all_energy.append(energy)

    return (np.concatenate(all_scores),
            np.concatenate(all_labels),
            np.concatenate(all_energy))


def evaluate_with_tta(model, loader, monitor, tta_adapter, device):
    """
    Evaluate ShiftDet (full) with OOD monitor + TTA.

    For each batch:
      1. Compute energy score.
      2. If batch is OOD → run TTA adaptation → get logits.
      3. Else → standard forward pass.
    """
    all_scores = []
    all_labels = []
    all_energy = []

    model.eval()
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            energy = EnergyOODMonitor.energy_score(logits)

        if monitor.batch_is_ood(energy):
            # TTA recalibration
            logits = tta_adapter.adapt(x, device)

        scores = logits[:, 1].detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(y.numpy())
        all_energy.append(energy.detach().cpu().numpy())

    return (np.concatenate(all_scores),
            np.concatenate(all_labels),
            np.concatenate(all_energy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  ShiftDet Evaluation on Unseen Test Environments")
    print(f"{'='*65}\n")

    # ---- Build test loaders ----
    test_loaders = build_dataloaders(config, split='test_unseen')

    # Also build ID test loader (first training env) for OOD baseline
    id_test_ds = ChannelDataset(
        channel_type  = 'awgn',
        n_samples     = config['data']['n_test_samples'],
        signal_length = config['data']['signal_length'],
        snr_range_db  = tuple(config['data']['snr_range_test']),
        seed          = config['training']['seed'] + 99,
    )
    id_loader = DataLoader(id_test_ds, batch_size=256,
                            shuffle=False, num_workers=0)

    # ---- Load models ----
    shiftdet_model   = load_model(config, 'best.pt', device)
    monitor          = load_ood_monitor(config)
    tta_cfg          = config['tta']
    tta_adapter      = TTAAdapter(shiftdet_model,
                                  lr=tta_cfg['lr'],
                                  steps=tta_cfg['steps'])

    # For ERM baseline, load a separately trained checkpoint
    erm_ckpt = os.path.join(config['paths']['checkpoint_dir'], 'erm_best.pt')
    has_erm  = os.path.exists(erm_ckpt)
    if has_erm:
        erm_model = load_model(config, 'erm_best.pt', device)

    # ---- Evaluate ----
    results = {}
    pfa_target = config['eval']['pfa_target']

    # Collect ID energy scores for OOD AUROC computation
    _, _, id_energy = collect_scores_and_labels(shiftdet_model,
                                                 id_loader, device)

    for env_name, loader in test_loaders.items():
        print(f"\nEvaluating on: {env_name}")

        # ShiftDet (full, with TTA)
        sd_scores, sd_labels, sd_energy = evaluate_with_tta(
            shiftdet_model, loader, monitor, tta_adapter, device)

        sd_pd = compute_pd_at_pfa(sd_scores, sd_labels, pfa_target)

        # OOD AUROC: compare ID energy vs OOD energy
        ood_energy  = sd_energy
        ood_labels  = np.ones(len(ood_energy))          # test env = OOD
        combined_e  = np.concatenate([id_energy, ood_energy])
        combined_l  = np.concatenate([np.zeros(len(id_energy)), ood_labels])
        sd_auroc    = compute_auroc(combined_e, combined_l)
        sd_fpr95    = compute_fpr_at_tpr(combined_e, combined_l, 0.95)

        entry = {
            'ShiftDet_full': {'pd': sd_pd,
                              'auroc': sd_auroc,
                              'fpr95': sd_fpr95},
        }

        # ShiftDet (no TTA) = standard ShiftDet forward pass
        scores_noTTA, labels_noTTA, _ = collect_scores_and_labels(
            shiftdet_model, loader, device)
        entry['ShiftDet_noTTA'] = {
            'pd': compute_pd_at_pfa(scores_noTTA, labels_noTTA, pfa_target)
        }

        # ERM-CNN baseline
        if has_erm:
            erm_scores, erm_labels, _ = collect_scores_and_labels(
                erm_model, loader, device)
            entry['ERM_CNN'] = {
                'pd': compute_pd_at_pfa(erm_scores, erm_labels, pfa_target)
            }

        results[env_name] = entry

        # Print per-environment results
        print(f"  ShiftDet (full)  PD={sd_pd:.4f}  "
              f"AUROC={sd_auroc:.4f}  FPR@95={sd_fpr95:.4f}")
        print(f"  ShiftDet (noTTA) PD="
              f"{entry['ShiftDet_noTTA']['pd']:.4f}")
        if has_erm:
            print(f"  ERM-CNN          PD={entry['ERM_CNN']['pd']:.4f}")

    # ---- Save results ----
    results_path = os.path.join(config['paths']['results_dir'],
                                'eval_results.json')
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
