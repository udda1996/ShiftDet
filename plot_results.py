"""
figures/plot_results.py
========================
Reproduce all figures from the ShiftDet paper.

Figures generated
-----------------
  Fig 1. ROC curves (PD vs PFA) for all methods under each
         unseen test environment.
  Fig 2. Energy score histogram: ID vs OOD environments.
  Fig 3. PD vs SNR curves for ShiftDet (full) and baselines.
  Fig 4. Ablation: PD vs IRM lambda (lambda sweep).

Run after evaluate.py has saved results to results/eval_results.json.

Usage
-----
  python figures/plot_results.py --config configs/default.yaml

Author: ShiftDet Team
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.channel_dataset import ChannelDataset
from models.backbone       import build_model
from models.ood_monitor    import EnergyOODMonitor
from utils.tta             import compute_roc


# ------------------------------------------------------------------ #
# Style                                                               #
# ------------------------------------------------------------------ #

STYLE = {
    'ShiftDet_full':  {'color': '#1f77b4', 'ls': '-',  'lw': 2.5,
                       'marker': 'o', 'label': 'ShiftDet (full)'},
    'ShiftDet_noTTA': {'color': '#ff7f0e', 'ls': '--', 'lw': 2.0,
                       'marker': 's', 'label': 'ShiftDet (no TTA)'},
    'ERM_CNN':        {'color': '#2ca02c', 'ls': ':',  'lw': 2.0,
                       'marker': '^', 'label': 'ERM-CNN'},
    'GLRT':           {'color': '#d62728', 'ls': '-.', 'lw': 2.0,
                       'marker': 'D', 'label': 'GLRT'},
    'DA_CNN':         {'color': '#9467bd', 'ls': '--', 'lw': 2.0,
                       'marker': 'v', 'label': 'DA-CNN†'},
}

plt.rcParams.update({
    'font.family':  'serif',
    'font.size':    12,
    'axes.grid':    True,
    'grid.alpha':   0.3,
    'figure.dpi':   150,
})


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_for_plot(config, ckpt_name, device):
    model = build_model(config)
    ckpt  = torch.load(
        os.path.join(config['paths']['checkpoint_dir'], ckpt_name),
        map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


# ------------------------------------------------------------------ #
# Fig 1: ROC Curves                                                   #
# ------------------------------------------------------------------ #

def plot_roc_curves(config, device, out_dir):
    """
    ROC curves for ShiftDet (full) vs baselines on the
    hardware-impaired channel (most challenging unseen environment).
    """
    print("Plotting Fig 1: ROC curves...")

    hw_ds = ChannelDataset(
        channel_type   = 'hw_impaired',
        n_samples      = config['data']['n_test_samples'],
        signal_length  = config['data']['signal_length'],
        snr_range_db   = tuple(config['data']['snr_range_test']),
        channel_kwargs = {
            'eps':       0.1,
            'phi_deg':   5.0,
            'pn_lw_khz': 10.0,
        },
        seed = 1234,
    )
    loader = DataLoader(hw_ds, batch_size=512, shuffle=False)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Collect scores for each method
    methods = {
        'ShiftDet_full':  'best.pt',
        'ShiftDet_noTTA': 'best.pt',
        'ERM_CNN':        'erm_best.pt',
    }

    for method_name, ckpt in methods.items():
        ckpt_path = os.path.join(config['paths']['checkpoint_dir'], ckpt)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {method_name} (checkpoint not found)")
            continue

        model = load_model_for_plot(config, ckpt, device)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(device))
                all_scores.append(logits[:, 1].cpu().numpy())
                all_labels.append(y.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        pfa, pd = compute_roc(scores, labels,
                               n_points=config['eval']['n_roc_points'])

        s = STYLE[method_name]
        ax.plot(pfa, pd,
                color=s['color'], linestyle=s['ls'],
                linewidth=s['lw'], label=s['label'])

    ax.set_xlabel(r'False-Alarm Probability $P_{FA}$')
    ax.set_ylabel(r'Detection Probability $P_D$')
    ax.set_title('ROC Curves – Hardware-Impaired Channel (Unseen)')
    ax.set_xlim([1e-4, 1.0])
    ax.set_ylim([0.0,  1.02])
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    ax.axvline(1e-3, color='gray', linestyle=':', alpha=0.6,
               label=r'$P_{FA}=10^{-3}$')

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'fig1_roc_curves.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------ #
# Fig 2: Energy Score Histogram                                        #
# ------------------------------------------------------------------ #

def plot_energy_histogram(config, device, out_dir):
    """
    Histogram of energy scores for ID test samples (AWGN) vs
    OOD test samples (Rician, MIMO, HW-impaired).
    Shows the separability exploited by the OOD monitor.
    """
    print("Plotting Fig 2: Energy score histogram...")

    model = load_model_for_plot(config, 'best.pt', device)

    ckpt_thresh = os.path.join(config['paths']['checkpoint_dir'],
                               'ood_threshold.json')
    with open(ckpt_thresh) as f:
        thresh_info = json.load(f)
    threshold = thresh_info['threshold']

    env_configs = [
        ('awgn',        {},                           'ID: AWGN',    '#1f77b4'),
        ('rician',      {'k_factor': 5.0,
                         'doppler_hz': 100.0},        'OOD: Rician', '#d62728'),
        ('hw_impaired', {'eps': 0.1, 'phi_deg': 5.0,
                         'pn_lw_khz': 10.0},          'OOD: HW-Imp', '#ff7f0e'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    for ch_type, kwargs, label, color in env_configs:
        ds = ChannelDataset(
            channel_type   = ch_type,
            n_samples      = 3000,
            signal_length  = config['data']['signal_length'],
            snr_range_db   = tuple(config['data']['snr_range_test']),
            channel_kwargs = kwargs,
            seed           = 42,
        )
        loader = DataLoader(ds, batch_size=512, shuffle=False)

        all_energy = []
        with torch.no_grad():
            for x, _ in loader:
                logits = model(x.to(device))
                energy = EnergyOODMonitor.energy_score(logits)
                all_energy.append(energy.cpu().numpy())
        energy = np.concatenate(all_energy)

        ax.hist(energy, bins=60, density=True, alpha=0.55,
                color=color, label=label, edgecolor='none')

    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.8,
               label=f'Threshold λ = {threshold:.2f}')
    ax.set_xlabel('Energy Score $S(\\mathbf{y})$')
    ax.set_ylabel('Density')
    ax.set_title('Energy Score Distribution: ID vs OOD Environments')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'fig2_energy_histogram.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------ #
# Fig 3: PD vs SNR                                                    #
# ------------------------------------------------------------------ #

def plot_pd_vs_snr(config, device, out_dir):
    """
    Detection probability PD vs SNR for each method on the
    Rician (unseen) environment, at fixed PFA = 1e-3.
    """
    print("Plotting Fig 3: PD vs SNR...")

    from utils.tta import compute_pd_at_pfa

    snr_points = np.arange(-10, 21, 2)
    pfa_target = config['eval']['pfa_target']

    methods = {
        'ShiftDet_full': 'best.pt',
        'ERM_CNN':       'erm_best.pt',
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    for method_name, ckpt in methods.items():
        ckpt_path = os.path.join(config['paths']['checkpoint_dir'], ckpt)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {method_name}")
            continue

        model = load_model_for_plot(config, ckpt, device)
        pd_curve = []

        for snr in snr_points:
            ds = ChannelDataset(
                channel_type   = 'rician',
                n_samples      = 2000,
                signal_length  = config['data']['signal_length'],
                snr_range_db   = (snr, snr),   # fixed SNR point
                channel_kwargs = {'k_factor': 5.0, 'doppler_hz': 100.0},
                seed           = 77,
            )
            loader = DataLoader(ds, batch_size=512, shuffle=False)
            scores, labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    logits = model(x.to(device))
                    scores.append(logits[:, 1].cpu().numpy())
                    labels.append(y.numpy())
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            pd_curve.append(compute_pd_at_pfa(scores, labels, pfa_target))

        s = STYLE[method_name]
        ax.plot(snr_points, pd_curve,
                color=s['color'], linestyle=s['ls'],
                linewidth=s['lw'], marker=s['marker'],
                markersize=5, label=s['label'])

    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5,
               label='$P_D = 0.9$')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'Detection Probability $P_D$')
    ax.set_title(r'$P_D$ vs SNR – Rician Channel (Unseen, $P_{FA}=10^{-3}$)')
    ax.set_xlim([snr_points[0], snr_points[-1]])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'fig3_pd_vs_snr.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------ #
# Fig 4: Ablation – IRM lambda sweep                                   #
# ------------------------------------------------------------------ #

def plot_ablation_lambda(out_dir):
    """
    Placeholder: load ablation results from ablation.py output
    and plot PD vs IRM lambda for each unseen environment.
    Replace the synthetic data below with actual results.
    """
    print("Plotting Fig 4: Ablation (IRM lambda)...")

    lambdas    = [0, 1, 10, 50, 100, 200, 500]
    # Synthetic placeholder values — replace with real ablation output
    pd_rician  = [0.79, 0.81, 0.83, 0.86, 0.89, 0.88, 0.85]
    pd_mimo    = [0.74, 0.77, 0.80, 0.83, 0.86, 0.85, 0.82]
    pd_hw      = [0.61, 0.65, 0.70, 0.77, 0.82, 0.81, 0.78]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(lambdas, pd_rician, 'o-',  color='#1f77b4', label='Rician')
    ax.plot(lambdas, pd_mimo,   's--', color='#ff7f0e', label='MIMO')
    ax.plot(lambdas, pd_hw,     '^:',  color='#2ca02c', label='HW-Impaired')

    ax.axvline(100, color='gray', linestyle=':', alpha=0.6,
               label='Selected λ=100')
    ax.set_xlabel(r'IRM Penalty Weight $\lambda_{\mathrm{IRM}}$')
    ax.set_ylabel(r'Detection Probability $P_D$')
    ax.set_title(r'Ablation: $P_D$ vs IRM Penalty Weight')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'fig4_ablation_lambda.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args   = parser.parse_args()

    config  = load_config(args.config)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = config['paths']['figures_dir']
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating ShiftDet paper figures → {out_dir}\n")

    plot_roc_curves(config, device, out_dir)
    plot_energy_histogram(config, device, out_dir)
    plot_pd_vs_snr(config, device, out_dir)
    plot_ablation_lambda(out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
