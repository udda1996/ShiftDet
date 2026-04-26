"""
utils/metrics.py
================
Evaluation metrics for ShiftDet.

All detection performance metrics used in the paper live here:

  - compute_pd_at_pfa   : PD at a fixed PFA (primary detection metric)
  - compute_roc         : Full ROC curve (PD vs PFA)
  - compute_auroc       : Area Under ROC (OOD detection quality)
  - compute_fpr_at_tpr  : FPR@95TPR (standard OOD benchmark)
  - summarize_results   : Pretty-print results table

Convention throughout:
  - Detection scores   : higher score = more likely H1 (signal present)
  - OOD scores         : higher score = more likely OOD
  - labels (detection) : 0 = H0 (noise only), 1 = H1 (signal present)
  - labels (OOD)       : 0 = ID (in-distribution), 1 = OOD

Author: ShiftDet Team
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict, Optional


def compute_pd_at_pfa(scores:     np.ndarray,
                      labels:     np.ndarray,
                      pfa_target: float = 1e-3) -> float:
    """
    Compute detection probability P_D at a fixed false-alarm
    rate P_FA.

    The detection threshold is set to the (1 - pfa_target)
    quantile of the H0 (noise-only) scores, so that exactly
    pfa_target fraction of H0 samples exceed the threshold.
    P_D is then the fraction of H1 (signal) samples that also
    exceed the threshold.

    Parameters
    ----------
    scores     : (N,)  detector output scores
    labels     : (N,)  ground truth {0=H0, 1=H1}
    pfa_target : float  desired P_FA (default 1e-3)

    Returns
    -------
    pd : float  detection probability at pfa_target
    """
    h0_scores = scores[labels == 0]
    h1_scores = scores[labels == 1]

    if len(h0_scores) == 0 or len(h1_scores) == 0:
        return float('nan')

    threshold = np.percentile(h0_scores,
                               100.0 * (1.0 - pfa_target))
    pd = float((h1_scores > threshold).mean())
    return pd


def compute_roc(scores:   np.ndarray,
                labels:   np.ndarray,
                n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Receiver Operating Characteristic (ROC) curve.

    Evaluates (P_FA, P_D) at n_points operating thresholds,
    evenly spaced in P_FA from 1e-4 to 1.0.

    Parameters
    ----------
    scores   : (N,)
    labels   : (N,)
    n_points : number of PFA operating points

    Returns
    -------
    pfa_arr : (n_points,)  false-alarm probabilities
    pd_arr  : (n_points,)  detection probabilities
    """
    h0_scores = scores[labels == 0]
    h1_scores = scores[labels == 1]

    pfa_arr = np.linspace(1e-4, 1.0, n_points)
    pd_arr  = np.zeros(n_points)

    for i, pfa in enumerate(pfa_arr):
        threshold  = np.percentile(h0_scores, 100.0 * (1.0 - pfa))
        pd_arr[i]  = (h1_scores > threshold).mean()

    return pfa_arr, pd_arr


def compute_auroc(ood_scores: np.ndarray,
                  ood_labels: np.ndarray) -> float:
    """
    Area Under the ROC curve for OOD detection.

    Parameters
    ----------
    ood_scores : (N,)  OOD scores (higher = more OOD)
    ood_labels : (N,)  ground truth {0=ID, 1=OOD}

    Returns
    -------
    auroc : float in [0, 1], higher is better
    """
    if len(np.unique(ood_labels)) < 2:
        return float('nan')
    return float(roc_auc_score(ood_labels, ood_scores))


def compute_fpr_at_tpr(ood_scores: np.ndarray,
                       ood_labels: np.ndarray,
                       tpr_target: float = 0.95) -> float:
    """
    False Positive Rate at a fixed True Positive Rate.

    FPR@95TPR is the standard benchmark metric in the OOD
    detection literature (lower is better).

    Parameters
    ----------
    ood_scores : (N,)  higher = more OOD
    ood_labels : (N,)  {0=ID, 1=OOD}
    tpr_target : target TPR (default 0.95 = 95%)

    Returns
    -------
    fpr : float  FPR at the specified TPR level
    """
    if len(np.unique(ood_labels)) < 2:
        return float('nan')

    fpr_arr, tpr_arr, _ = roc_curve(ood_labels, ood_scores)
    idx = np.searchsorted(tpr_arr, tpr_target)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])


def compute_detection_error(ood_scores: np.ndarray,
                             ood_labels: np.ndarray) -> float:
    """
    Detection Error: minimum over all thresholds of
    0.5 * (FPR + FNR), where FNR = 1 - TPR.

    Lower is better. Useful when OOD and ID sets are balanced.

    Parameters
    ----------
    ood_scores : (N,)
    ood_labels : (N,)  {0=ID, 1=OOD}

    Returns
    -------
    detection_error : float
    """
    if len(np.unique(ood_labels)) < 2:
        return float('nan')

    fpr_arr, tpr_arr, _ = roc_curve(ood_labels, ood_scores)
    fnr_arr = 1.0 - tpr_arr
    return float(np.min(0.5 * (fpr_arr + fnr_arr)))


def summarize_results(results:    Dict[str, dict],
                      pfa_target: float = 1e-3) -> str:
    """
    Format evaluation results as a printable ASCII table,
    matching the layout of Table I in the paper.

    Parameters
    ----------
    results    : {env_name: {'pd': float, 'auroc': float,
                              'fpr95': float}}
    pfa_target : PFA at which PD was computed (for header label)

    Returns
    -------
    table_str : formatted multi-line string
    """
    pfa_str = f"PD@PFA={pfa_target:.0e}"
    header  = (f"{'Environment':<22} {pfa_str:>12} "
               f"{'AUROC':>8} {'FPR@95':>8}")
    sep     = "─" * len(header)
    rows    = [sep, header, sep]

    for env, m in results.items():
        pd    = m.get('pd',    float('nan'))
        auroc = m.get('auroc', float('nan'))
        fpr95 = m.get('fpr95', float('nan'))
        rows.append(
            f"{env:<22} {pd:>12.4f} {auroc:>8.4f} {fpr95:>8.4f}")

    rows.append(sep)

    # Average row
    vals = {k: [m.get(k, float('nan'))
                for m in results.values()]
            for k in ('pd', 'auroc', 'fpr95')}
    avgs = {k: float(np.nanmean(v)) for k, v in vals.items()}
    rows.append(
        f"{'Average':<22} {avgs['pd']:>12.4f} "
        f"{avgs['auroc']:>8.4f} {avgs['fpr95']:>8.4f}")
    rows.append(sep)

    return "\n".join(rows)


def snr_curve_summary(snr_points:  np.ndarray,
                      pd_curves:   Dict[str, np.ndarray],
                      pfa_target:  float = 1e-3) -> str:
    """
    Print a compact SNR vs PD table for multiple methods.

    Parameters
    ----------
    snr_points : (K,) SNR values in dB
    pd_curves  : {method_name: (K,) PD values}
    pfa_target : PFA level used

    Returns
    -------
    table_str : formatted string
    """
    method_names = list(pd_curves.keys())
    col_w        = 12
    header = f"{'SNR (dB)':<10}" + "".join(
        f"{n:>{col_w}}" for n in method_names)
    sep    = "─" * len(header)
    rows   = [sep,
              f"PD at P_FA = {pfa_target:.0e}",
              header, sep]

    for i, snr in enumerate(snr_points):
        row = f"{snr:<10.1f}"
        for name in method_names:
            val  = pd_curves[name][i]
            row += f"{val:>{col_w}.4f}"
        rows.append(row)

    rows.append(sep)
    return "\n".join(rows)
