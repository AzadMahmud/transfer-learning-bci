"""EEG data augmentation.

EEGAugmenter
    Operates on raw EEG epochs (n_trials, n_channels, n_times) before CWT.
    Techniques: Gaussian noise injection, temporal cropping + zero-pad,
    channel dropout, amplitude scaling.

SpectrogramAugmenter
    Operates on spectrogram images (n_trials, n_channels, freq_bins, time_steps).
    Techniques: SpecAugment (frequency/time masking), patch dropout, mixup.

Usage::

    from bci.data.augmentation import EEGAugmenter, SpectrogramAugmenter
    from bci.utils.config import AugmentationConfig

    cfg = AugmentationConfig()
    eeg_aug = EEGAugmenter(cfg, seed=42)
    spec_aug = SpectrogramAugmenter(cfg, seed=42)

    X_aug = eeg_aug(X_train, training=True)
    spec_aug_train = spec_aug(spectrograms, training=True)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from bci.utils.config import AugmentationConfig

logger = logging.getLogger(__name__)


class EEGAugmenter:
    """Augmentations applied to EEG epochs before CWT spectrogram generation.

    All augmentations are applied independently per trial.

    Args:
        config: AugmentationConfig with enable flags and hyperparameters.
        seed: Base random seed. Each call draws fresh samples.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        X: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Apply augmentation pipeline to a batch of EEG trials.

        Args:
            X: EEG epochs of shape (n_trials, n_channels, n_times).
            training: If False, return X unchanged (inference mode).

        Returns:
            Augmented array, same shape as input.
        """
        if not training:
            return X

        X = X.copy()
        cfg = self.config

        if cfg.apply_amplitude_scale:
            X = self._amplitude_scale(X)

        if cfg.apply_gaussian_noise:
            X = self._gaussian_noise(X)

        if cfg.apply_channel_dropout:
            X = self._channel_dropout(X)

        if cfg.apply_temporal_crop:
            X = self._temporal_crop(X)

        return X

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------

    def _gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise scaled to per-channel std of each trial."""
        cfg = self.config
        # Per-trial, per-channel std: shape (n_trials, n_channels, 1)
        std = X.std(axis=2, keepdims=True)
        noise = self._rng.standard_normal(X.shape).astype(X.dtype)
        X_noisy = X + noise * std * cfg.gaussian_noise_std
        return X_noisy

    def _temporal_crop(self, X: np.ndarray) -> np.ndarray:
        """Randomly zero-pad both ends of each trial (simulate crop + re-pad).

        Zeroes up to ``temporal_crop_ratio`` of the time-points from the
        start and end of each trial independently.
        """
        cfg = self.config
        n_times = X.shape[2]
        max_drop = int(n_times * cfg.temporal_crop_ratio)
        if max_drop < 1:
            return X

        X_out = X.copy()
        for i in range(X.shape[0]):
            drop_start = int(self._rng.integers(0, max_drop + 1))
            drop_end = int(self._rng.integers(0, max_drop + 1))
            if drop_start > 0:
                X_out[i, :, :drop_start] = 0.0
            if drop_end > 0:
                X_out[i, :, n_times - drop_end :] = 0.0
        return X_out

    def _channel_dropout(self, X: np.ndarray) -> np.ndarray:
        """Zero out randomly selected channels per trial."""
        cfg = self.config
        n_channels = X.shape[1]
        X_out = X.copy()
        for i in range(X.shape[0]):
            mask = self._rng.random(n_channels) < cfg.channel_dropout_prob
            X_out[i, mask, :] = 0.0
        return X_out

    def _amplitude_scale(self, X: np.ndarray) -> np.ndarray:
        """Randomly scale the amplitude of each trial."""
        cfg = self.config
        low, high = cfg.amplitude_scale_range
        scales = self._rng.uniform(low, high, size=(X.shape[0], 1, 1)).astype(X.dtype)
        return X * scales


class SpectrogramAugmenter:
    """ViT-specific augmentations for spectrogram images.

    Implements augmentation techniques specifically designed for Vision Transformers
    operating on EEG spectrograms:

    1. **SpecAugment**: Frequency and time masking (from speech recognition)
    2. **Patch Dropout**: Randomly drop patches before transformer
    3. **Mixup**: Interpolate between samples for regularization

    All augmentations preserve the (n_channels, freq_bins, time_steps) structure.

    Args:
        config: AugmentationConfig with enable flags and hyperparameters.
        seed: Base random seed.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        training: bool = True,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Apply augmentation pipeline to spectrogram batch.

        Args:
            X: Spectrograms of shape (n_trials, n_channels, freq_bins, time_steps).
            y: Labels of shape (n_trials,). Required for mixup.
            training: If False, return X unchanged.

        Returns:
            If mixup is disabled or y is None:
                Augmented spectrograms of the same shape.
            If mixup is enabled and y is provided:
                Tuple of (X_mixed, y_a, y_b, lam) for mixup training.
        """
        if not training:
            return X

        X = X.copy()
        cfg = self.config

        # Apply SpecAugment-style frequency masking
        if cfg.apply_freq_mask:
            X = self._frequency_mask(X)

        # Apply SpecAugment-style time masking
        if cfg.apply_time_mask:
            X = self._time_mask(X)

        # Apply mixup if labels provided and enabled
        if hasattr(cfg, "apply_mixup") and cfg.apply_mixup and y is not None:
            return self._mixup(X, y)

        return X

    def _frequency_mask(self, X: np.ndarray) -> np.ndarray:
        """Apply frequency masking (SpecAugment-style).

        Randomly mask contiguous frequency bands with zeros.

        Args:
            X: Shape (n_trials, n_channels, freq_bins, time_steps)

        Returns:
            Masked spectrograms.
        """
        cfg = self.config
        n_trials, n_channels, n_freqs, n_times = X.shape
        max_width = min(cfg.freq_mask_max_width, n_freqs // 2)

        if max_width < 1:
            return X

        X_out = X.copy()
        for i in range(n_trials):
            # Random mask width
            width = self._rng.integers(1, max_width + 1)
            # Random start position
            start = self._rng.integers(0, n_freqs - width + 1)
            # Apply mask across all channels for this trial
            X_out[i, :, start : start + width, :] = 0.0

        return X_out

    def _time_mask(self, X: np.ndarray) -> np.ndarray:
        """Apply time masking (SpecAugment-style).

        Randomly mask contiguous time steps with zeros.

        Args:
            X: Shape (n_trials, n_channels, freq_bins, time_steps)

        Returns:
            Masked spectrograms.
        """
        cfg = self.config
        n_trials, n_channels, n_freqs, n_times = X.shape
        max_width = min(cfg.time_mask_max_width, n_times // 2)

        if max_width < 1:
            return X

        X_out = X.copy()
        for i in range(n_trials):
            # Random mask width
            width = self._rng.integers(1, max_width + 1)
            # Random start position
            start = self._rng.integers(0, n_times - width + 1)
            # Apply mask across all channels for this trial
            X_out[i, :, :, start : start + width] = 0.0

        return X_out

    def _mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Apply mixup augmentation.

        Mixes pairs of samples: X_new = lam*X1 + (1-lam)*X2

        Args:
            X: Spectrograms of shape (n_trials, n_channels, freq_bins, time_steps)
            y: Labels of shape (n_trials,)

        Returns:
            Tuple of (X_mixed, y_a, y_b, lam) where:
                X_mixed: Mixed spectrograms
                y_a: Original labels
                y_b: Shuffled labels for mixing
                lam: Mixing coefficient
        """
        cfg = self.config
        n_trials = X.shape[0]

        # Sample lambda from Beta distribution
        alpha = getattr(cfg, "mixup_alpha", 0.4)
        lam = self._rng.beta(alpha, alpha)

        # Shuffle indices for mixing pairs
        perm = self._rng.permutation(n_trials)

        # Mix samples
        X_mixed = lam * X + (1 - lam) * X[perm]

        return X_mixed, y, y[perm], lam

    def patch_dropout(
        self,
        patches: np.ndarray,
        drop_ratio: float = 0.1,
    ) -> np.ndarray:
        """Drop random patches (for use inside ViT forward pass).

        This is designed to be called on embedded patch sequences.

        Args:
            patches: Shape (batch, n_patches, embed_dim)
            drop_ratio: Fraction of patches to drop

        Returns:
            Patches with some set to zero.
        """
        n_patches = patches.shape[1]
        n_drop = int(n_patches * drop_ratio)

        if n_drop < 1:
            return patches

        patches_out = patches.copy()
        for i in range(patches.shape[0]):
            drop_idx = self._rng.choice(n_patches, size=n_drop, replace=False)
            patches_out[i, drop_idx, :] = 0.0

        return patches_out


def mixup_criterion(
    criterion,
    pred,
    y_a,
    y_b,
    lam: float,
):
    """Compute mixup loss.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
