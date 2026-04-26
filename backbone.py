"""
models/backbone.py
==================
CNN Invariant Feature Extractor for ShiftDet.

Architecture
------------
Input : (B, 2, L)  -- 2-channel (I/Q) complex signal blocks
Output: (B, feature_dim)  -- shift-invariant feature embedding
        (B, n_classes)    -- classification logits

The backbone is a 1-D convolutional network with three
convolutional blocks followed by global average pooling
and a fully-connected classifier head.

Batch Normalization layers are deliberately included so
that Test-Time Adaptation (TTA) can update their running
statistics and affine parameters at inference time without
touching the convolutional weights.

Design rationale
----------------
- 1-D convolutions operate along the time axis of the I/Q
  signal, capturing temporal correlations in the baseband
  waveform.
- Global Average Pooling makes the architecture
  length-agnostic (works for any signal_length >= kernel_size).
- The split into feature extractor + classifier head
  enables the IRM training objective, which requires
  separate access to the representation and the linear head.

Author: ShiftDet Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """
    One convolutional block: Conv1d → BatchNorm1d → ReLU → MaxPool1d.

    The BatchNorm layer is the target of TTA adaptation; its
    `weight` (gamma) and `bias` (beta) affine parameters are
    the only parameters updated during test-time adaptation.
    """

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 kernel_size:  int = 3,
                 pool_size:    int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=False)   # BN absorbs bias
        self.bn   = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SignalBackbone(nn.Module):
    """
    CNN feature extractor: maps raw I/Q blocks to a
    fixed-dimensional embedding vector.

    Parameters
    ----------
    in_channels       : number of input channels (2 for I/Q)
    backbone_channels : list of output channels per conv block
    kernel_size       : conv kernel size (same for all blocks)
    feature_dim       : dimensionality of the output embedding
    """

    def __init__(self,
                 in_channels:       int       = 2,
                 backbone_channels: list      = (32, 64, 128),
                 kernel_size:       int       = 3,
                 feature_dim:       int       = 256):
        super().__init__()

        # Build convolutional blocks dynamically from config
        blocks = []
        ch_in  = in_channels
        for ch_out in backbone_channels:
            blocks.append(ConvBlock(ch_in, ch_out, kernel_size))
            ch_in = ch_out
        self.conv_blocks = nn.Sequential(*blocks)

        # Global Average Pooling collapses the time dimension
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Projection to feature_dim embedding
        self.projector = nn.Sequential(
            nn.Linear(backbone_channels[-1], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, L)  raw I/Q signal

        Returns
        -------
        z : (B, feature_dim)  embedding
        """
        x = self.conv_blocks(x)     # (B, C_last, L')
        x = self.gap(x).squeeze(-1) # (B, C_last)
        z = self.projector(x)       # (B, feature_dim)
        return z


class ShiftDetModel(nn.Module):
    """
    Full ShiftDet model: backbone + linear classifier head.

    The classifier head `w` is intentionally kept as a simple
    linear layer. The IRM penalty requires that this head be
    simultaneously optimal across all training environments;
    a simple linear head makes the IRM constraint tractable
    and interpretable.

    Parameters
    ----------
    backbone   : SignalBackbone instance
    n_classes  : number of output classes (2 for binary detection)
    feature_dim: must match backbone output dimension
    """

    def __init__(self,
                 backbone:    SignalBackbone,
                 n_classes:   int = 2,
                 feature_dim: int = 256):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self,
                x: torch.Tensor,
                return_features: bool = False
                ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        x               : (B, 2, L)
        return_features : if True, also return the embedding z

        Returns
        -------
        logits : (B, n_classes)
        z      : (B, feature_dim)  [only if return_features=True]
        """
        z      = self.backbone(x)
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits

    def get_bn_params(self):
        """
        Return only the BatchNorm affine parameters (gamma, beta).
        Used by TTA to restrict adaptation to BN layers only.
        """
        bn_params = []
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                for p in module.parameters():
                    if p.requires_grad:
                        bn_params.append(p)
        return bn_params

    def freeze_except_bn(self):
        """
        Freeze all parameters except BatchNorm affine params.
        Called at the start of each TTA adaptation step.
        """
        for name, param in self.named_parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                for p in module.parameters():
                    p.requires_grad = True

    def unfreeze_all(self):
        """Restore full gradient computation after TTA."""
        for param in self.parameters():
            param.requires_grad = True


def build_model(config: dict) -> ShiftDetModel:
    """
    Instantiate ShiftDetModel from a config dictionary.

    Parameters
    ----------
    config : dict  loaded from configs/default.yaml

    Returns
    -------
    model : ShiftDetModel (on CPU; move to device in trainer)
    """
    mcfg = config['model']
    backbone = SignalBackbone(
        in_channels       = mcfg['in_channels'],
        backbone_channels = mcfg['backbone_channels'],
        kernel_size       = mcfg['kernel_size'],
        feature_dim       = mcfg['feature_dim'],
    )
    model = ShiftDetModel(
        backbone    = backbone,
        n_classes   = mcfg['n_classes'],
        feature_dim = mcfg['feature_dim'],
    )
    return model
