"""
data/channel_dataset.py
=======================
Synthetic wireless channel dataset generator for ShiftDet.

This module implements all channel environments used in the paper:
  - AWGN
  - Rayleigh flat fading (with configurable Doppler)
  - Rician fading  (unseen at training time)
  - 2x2 MIMO Rayleigh  (unseen at training time)
  - Hardware-impaired AWGN: IQ imbalance + phase noise  (unseen)

Each environment produces complex baseband I/Q samples for
binary signal detection:
  H0 : y = n                  (noise only)
  H1 : y = h * s + n          (signal present)

Outputs are returned as real-valued 2-channel tensors [I, Q]
of shape (N, 2, signal_length), compatible with the CNN backbone.

Author: ShiftDet Team
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple


# ------------------------------------------------------------------ #
#  Low-level channel simulation helpers                               #
# ------------------------------------------------------------------ #

def generate_bpsk(n: int, length: int) -> np.ndarray:
    """
    Generate n BPSK baseband symbols of block length `length`.
    Symbols are drawn uniformly from {-1, +1} and repeated
    to fill the block (constant envelope assumption).

    Returns complex array of shape (n, length).
    """
    bits = 2 * np.random.randint(0, 2, size=(n, 1)) - 1   # {-1, +1}
    return (bits * np.ones((n, length))).astype(complex)


def generate_qpsk(n: int, length: int) -> np.ndarray:
    """
    Generate n QPSK baseband symbols of block length `length`.
    Phase drawn uniformly from {π/4, 3π/4, 5π/4, 7π/4}.

    Returns complex array of shape (n, length).
    """
    phases = np.random.choice([np.pi/4, 3*np.pi/4,
                                5*np.pi/4, 7*np.pi/4], size=(n, 1))
    symbols = np.exp(1j * phases) * np.ones((n, length))
    return symbols


def awgn_channel(signals: np.ndarray, snr_db: np.ndarray) -> np.ndarray:
    """
    Pass signals through AWGN channel.

    Parameters
    ----------
    signals : complex (N, L)   clean transmitted signal blocks
    snr_db  : float array (N,) per-sample SNR in dB

    Returns
    -------
    received : complex (N, L)  noisy received signal
    """
    snr_linear = 10 ** (snr_db / 10.0)          # (N,)
    sig_power  = np.mean(np.abs(signals)**2, axis=1)   # (N,)
    noise_var  = sig_power / snr_linear          # (N,)
    noise_std  = np.sqrt(noise_var / 2)          # per real component

    noise = (noise_std[:, None] *
             (np.random.randn(*signals.shape) +
              1j * np.random.randn(*signals.shape)))
    return signals + noise


def rayleigh_channel(signals: np.ndarray,
                     snr_db:  np.ndarray,
                     doppler_hz: float = 0.0,
                     fs: float = 1e6) -> np.ndarray:
    """
    Flat Rayleigh fading channel with optional Doppler spread.

    A single complex fading coefficient h ~ CN(0,1) is drawn
    per sample block (block-fading model, coherence >> block length).
    If doppler_hz > 0, a time-varying phase rotation is added
    to simulate Doppler shift across the block.

    Parameters
    ----------
    signals    : complex (N, L)
    snr_db     : float array (N,)
    doppler_hz : max Doppler frequency in Hz
    fs         : sample rate in Hz (used for Doppler normalisation)

    Returns
    -------
    received : complex (N, L)
    """
    N, L = signals.shape

    # Block-fading Rayleigh coefficient: h ~ CN(0,1)
    h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)

    if doppler_hz > 0:
        # Add a random Doppler frequency uniformly in [0, doppler_hz]
        fd = np.random.uniform(0, doppler_hz, size=(N, 1))
        t  = np.arange(L)[None, :] / fs
        h  = h * np.exp(1j * 2 * np.pi * fd * t)

    received_clean = h * signals

    # Normalise so E[|h|^2] = 1 => received SNR = transmitted SNR
    received_clean = received_clean / np.sqrt(np.mean(np.abs(h)**2,
                                                       axis=1, keepdims=True))
    return awgn_channel(received_clean, snr_db)


def rician_channel(signals: np.ndarray,
                   snr_db:  np.ndarray,
                   k_factor: float = 5.0,
                   doppler_hz: float = 100.0,
                   fs: float = 1e6) -> np.ndarray:
    """
    Rician fading channel.

    The fading coefficient is h = sqrt(K/(K+1)) * h_LOS
                                + sqrt(1/(K+1)) * h_scatter
    where h_LOS = 1 (normalized LoS component)
    and   h_scatter ~ CN(0,1).

    This environment is UNSEEN during training.

    Parameters
    ----------
    k_factor   : Rician K-factor (ratio of LOS to scatter power)
    """
    N, L = signals.shape
    los_amp    = np.sqrt(k_factor / (k_factor + 1))
    scatter_amp = np.sqrt(1.0 / (k_factor + 1))

    h_scatter = (np.random.randn(N, 1) +
                 1j * np.random.randn(N, 1)) / np.sqrt(2)
    h = los_amp + scatter_amp * h_scatter        # (N, 1)

    if doppler_hz > 0:
        fd = np.random.uniform(0, doppler_hz, size=(N, 1))
        t  = np.arange(L)[None, :] / fs
        h  = h * np.exp(1j * 2 * np.pi * fd * t)

    received_clean = h * signals
    received_clean = received_clean / np.sqrt(
        np.mean(np.abs(h)**2, axis=1, keepdims=True))
    return awgn_channel(received_clean, snr_db)


def mimo_2x2_channel(signals: np.ndarray,
                     snr_db:  np.ndarray) -> np.ndarray:
    """
    2x2 MIMO Rayleigh channel with maximum-ratio combining (MRC).

    H ~ CN(0, I_{2x2}) drawn per block. The receiver applies
    matched filter (H^H y) and selects the first stream output.

    This environment is UNSEEN during training.

    Parameters
    ----------
    signals : complex (N, L)  single-stream input

    Returns
    -------
    received : complex (N, L) after MRC on first receive antenna
    """
    N, L = signals.shape
    n_tx, n_rx = 2, 2

    # Channel matrix H (N, n_rx, n_tx) ~ CN(0, 1/n_tx)
    H_real = np.random.randn(N, n_rx, n_tx)
    H_imag = np.random.randn(N, n_rx, n_tx)
    H = (H_real + 1j * H_imag) / np.sqrt(2 * n_tx)

    # Received signal per RX antenna (N, n_rx, L)
    # Broadcast signals over TX antennas (simplified: same signal on both TX)
    s_tx = signals[:, np.newaxis, :]          # (N, 1, L)
    received = np.sum(H[:, :, :, np.newaxis] *
                      s_tx[:, np.newaxis, :, :], axis=2)  # (N, n_rx, L)

    # MRC: combine across RX antennas weighted by conjugate channel
    # w = H[:, :, 0] / ||H[:, :, 0]||  (first TX column)
    h_col = H[:, :, 0]                        # (N, n_rx)
    h_norm = np.linalg.norm(h_col, axis=1, keepdims=True) + 1e-10
    w = np.conj(h_col) / h_norm               # (N, n_rx)

    # Combined output: (N, L)
    combined = np.einsum('ni,nil->nl', w, received)

    return awgn_channel(combined, snr_db)


def hw_impaired_channel(signals:   np.ndarray,
                        snr_db:    np.ndarray,
                        eps:       float = 0.1,
                        phi_deg:   float = 5.0,
                        pn_lw_khz: float = 10.0,
                        fs:        float = 1e6) -> np.ndarray:
    """
    Hardware-impaired AWGN channel.

    Models three common RF front-end impairments:
      1. IQ Imbalance: amplitude imbalance `eps` and phase
         imbalance `phi` distort the I and Q branches.
         y_iq = (1 + eps/2)*cos(phi/2)*I
              + j*(1 - eps/2)*sin(phi/2)*Q  + cross-terms
      2. Phase Noise: Wiener-process phase noise with linewidth
         `pn_lw_khz` kHz accumulated over the block.
      3. AWGN as usual.

    This environment is UNSEEN during training.
    """
    N, L = signals.shape
    phi_rad = np.deg2rad(phi_deg)

    # --- IQ Imbalance ---
    alpha = (1 + eps/2) * np.cos(phi_rad/2)
    beta  = (1 - eps/2) * np.sin(phi_rad/2)
    I_in  = np.real(signals)
    Q_in  = np.imag(signals)
    I_out = alpha * I_in - beta  * Q_in
    Q_out = beta  * I_in + alpha * Q_in
    signals_iq = I_out + 1j * Q_out

    # --- Phase Noise (Wiener process) ---
    # Variance per sample: sigma^2 = 2*pi*linewidth / fs
    pn_var_per_sample = 2 * np.pi * (pn_lw_khz * 1e3) / fs
    pn_increments = np.random.randn(N, L) * np.sqrt(pn_var_per_sample)
    phase_noise   = np.cumsum(pn_increments, axis=1)
    signals_pn    = signals_iq * np.exp(1j * phase_noise)

    return awgn_channel(signals_pn, snr_db)


# ------------------------------------------------------------------ #
#  Dataset class                                                      #
# ------------------------------------------------------------------ #

class ChannelDataset(Dataset):
    """
    PyTorch Dataset for one channel environment.

    Each sample is a tuple (x, label) where:
      x     : torch.FloatTensor of shape (2, signal_length)
               channel 0 = In-phase (I), channel 1 = Quadrature (Q)
      label : 0 (H0, noise only) or 1 (H1, signal present)

    Parameters
    ----------
    channel_type : str
        One of {'awgn', 'rayleigh', 'rician', 'mimo', 'hw_impaired'}
    n_samples    : int
        Total samples (equal split between H0 and H1)
    signal_length: int
        Number of complex samples per block
    snr_range_db : tuple (min_snr, max_snr)
        SNR drawn uniformly per sample from this range
    channel_kwargs : dict
        Extra keyword arguments forwarded to the channel function
        (e.g., doppler_hz=200, k_factor=5)
    modulation   : str
        'bpsk' or 'qpsk' for H1 signal generation
    """

    CHANNEL_FUNCTIONS = {
        'awgn':        awgn_channel,
        'rayleigh':    rayleigh_channel,
        'rician':      rician_channel,
        'mimo':        mimo_2x2_channel,
        'hw_impaired': hw_impaired_channel,
    }

    def __init__(self,
                 channel_type:   str,
                 n_samples:      int   = 10000,
                 signal_length:  int   = 128,
                 snr_range_db:   tuple = (-5, 20),
                 channel_kwargs: dict  = None,
                 modulation:     str   = 'bpsk',
                 seed:           Optional[int] = None):

        if seed is not None:
            np.random.seed(seed)

        assert channel_type in self.CHANNEL_FUNCTIONS, \
            f"Unknown channel type '{channel_type}'. " \
            f"Choose from {list(self.CHANNEL_FUNCTIONS.keys())}"

        self.channel_type   = channel_type
        self.signal_length  = signal_length
        self.snr_range_db   = snr_range_db
        self.channel_kwargs = channel_kwargs or {}

        channel_fn = self.CHANNEL_FUNCTIONS[channel_type]

        n_h0 = n_samples // 2
        n_h1 = n_samples - n_h0

        snr_h0 = np.random.uniform(*snr_range_db, size=n_h0)
        snr_h1 = np.random.uniform(*snr_range_db, size=n_h1)

        # H0: pure noise -- pass zeros through channel (only AWGN added)
        zeros_h0 = np.zeros((n_h0, signal_length), dtype=complex)
        rx_h0    = awgn_channel(zeros_h0, snr_h0)

        # H1: signal + channel + noise
        gen_fn = generate_bpsk if modulation == 'bpsk' else generate_qpsk
        tx_h1  = gen_fn(n_h1, signal_length)
        rx_h1  = channel_fn(tx_h1, snr_h1, **self.channel_kwargs)

        # Stack and create labels
        rx     = np.concatenate([rx_h0, rx_h1], axis=0)   # (N, L)
        labels = np.concatenate([np.zeros(n_h0, dtype=np.int64),
                                 np.ones(n_h1,  dtype=np.int64)])

        # Shuffle
        idx = np.random.permutation(n_samples)
        rx, labels = rx[idx], labels[idx]

        # Convert to 2-channel real tensor: (N, 2, L)
        self.x = torch.FloatTensor(
            np.stack([np.real(rx), np.imag(rx)], axis=1))
        self.y = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def build_dataloaders(config: dict,
                      split: str = 'train') -> Dict[str, DataLoader]:
    """
    Build DataLoaders for all environments specified in config.

    Parameters
    ----------
    config : dict   loaded from configs/default.yaml
    split  : str    'train' or 'test_unseen'

    Returns
    -------
    dict mapping environment name -> DataLoader
    """
    envs   = config['environments'][split]
    n_samp = (config['data']['n_samples_per_env'] if split == 'train'
              else config['data']['n_test_samples'])
    snr_range = (config['data']['snr_range_train'] if split == 'train'
                 else config['data']['snr_range_test'])

    loaders = {}
    for env_cfg in envs:
        name         = env_cfg['name']
        channel_type = env_cfg['type']
        kwargs       = {k: v for k, v in env_cfg.items()
                        if k not in ('name', 'type')}

        ds = ChannelDataset(
            channel_type   = channel_type,
            n_samples      = n_samp,
            signal_length  = config['data']['signal_length'],
            snr_range_db   = tuple(snr_range),
            channel_kwargs = kwargs,
            seed           = config['training']['seed'],
        )
        loaders[name] = DataLoader(
            ds,
            batch_size  = config['training']['batch_size'],
            shuffle     = (split == 'train'),
            num_workers = 0,
            pin_memory  = True,
        )

    return loaders
