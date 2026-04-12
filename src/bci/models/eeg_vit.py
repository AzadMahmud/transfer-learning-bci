"""EEG-specific Vision Transformer for Motor Imagery Classification.

This module implements a ViT architecture specifically designed for EEG signals,
addressing the key issues with treating EEG like natural images:

1. **Structured Input Representation**: Preserves channel x frequency x time structure
2. **Electrode Topology Encoding**: Maps electrodes to 2D spatial positions (10-20 system)
3. **Channel-Aware Patch Embedding**: Uses separate embeddings per EEG channel
4. **Temporal Positional Encoding**: Separate encoding for time dimension
5. **Smaller Patch Size**: patch8 or patch4 for fine-grained temporal info
6. **Lightweight Architecture**: Fewer layers (4-6) for EEG-scale data

Architecture:
    Input: (batch, n_channels, freq_bins, time_steps)
    -> Channel-wise patch embedding with electrode topology
    -> Add channel embedding + temporal positional encoding
    -> Transformer encoder (4-6 layers)
    -> Classification head
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from bci.models.temporal_encoder import TemporalEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 10-20 System Electrode Positions (normalized to 0-1 grid)
# These positions approximate the standard 10-20 electrode placement
# on a 2D projection of the scalp (top view, nose up)
# ---------------------------------------------------------------------------

ELECTRODE_POSITIONS_10_20 = {
    # Frontal central
    "Fz": (0.5, 0.3),
    "FC1": (0.4, 0.35),
    "FC2": (0.6, 0.35),
    "FC3": (0.3, 0.35),
    "FC4": (0.7, 0.35),
    "FC5": (0.2, 0.35),
    "FC6": (0.8, 0.35),
    # Central (motor cortex - most important for MI)
    "C1": (0.4, 0.5),
    "C2": (0.6, 0.5),
    "C3": (0.25, 0.5),
    "C4": (0.75, 0.5),
    "C5": (0.1, 0.5),
    "C6": (0.9, 0.5),
    "Cz": (0.5, 0.5),
    # Centro-parietal
    "CP1": (0.4, 0.65),
    "CP2": (0.6, 0.65),
    "CP3": (0.3, 0.65),
    "CP4": (0.7, 0.65),
    "CP5": (0.2, 0.65),
    "CP6": (0.8, 0.65),
    "CPz": (0.5, 0.65),
    # Parietal
    "Pz": (0.5, 0.7),
    "P1": (0.4, 0.7),
    "P2": (0.6, 0.7),
    "P3": (0.3, 0.7),
    "P4": (0.7, 0.7),
}


def get_electrode_grid_positions(
    channel_names: list[str],
    grid_size: int = 8,
) -> torch.Tensor:
    """Map electrode names to grid positions for spatial embedding.

    Args:
        channel_names: List of electrode names (e.g., ["C3", "Cz", "C4"])
        grid_size: Size of the spatial grid

    Returns:
        Tensor of shape (n_channels, 2) with (row, col) grid positions
    """
    positions = []
    for ch in channel_names:
        if ch in ELECTRODE_POSITIONS_10_20:
            x, y = ELECTRODE_POSITIONS_10_20[ch]
            row = int(y * (grid_size - 1))
            col = int(x * (grid_size - 1))
            positions.append([row, col])
        else:
            # Default to center if unknown
            positions.append([grid_size // 2, grid_size // 2])
            logger.warning("Unknown electrode %s, using center position", ch)
    return torch.tensor(positions, dtype=torch.long)


class ChannelPatchEmbedding(nn.Module):
    """Channel-aware patch embedding for EEG spectrograms.

    Instead of treating all channels as a single image, this embedding:
    1. Processes each EEG channel's spectrogram separately
    2. Uses smaller patches (4x4 or 8x8) to preserve fine-grained temporal info
    3. Adds channel-specific embeddings based on electrode positions

    Args:
        n_channels: Number of EEG channels
        freq_bins: Number of frequency bins in spectrogram
        time_steps: Number of time steps in spectrogram
        patch_size: Size of each patch (default: 4 for fine-grained)
        embed_dim: Embedding dimension
        channel_names: List of electrode names for topology encoding
    """

    def __init__(
        self,
        n_channels: int = 9,
        freq_bins: int = 64,
        time_steps: int = 64,
        patch_size: int = 4,
        embed_dim: int = 192,
        channel_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Number of patches per channel
        self.n_freq_patches = freq_bins // patch_size
        self.n_time_patches = time_steps // patch_size
        self.patches_per_channel = self.n_freq_patches * self.n_time_patches
        self.n_patches = n_channels * self.patches_per_channel

        # Patch embedding: each channel's patch gets embedded
        # Using 1 input channel since each EEG channel is processed separately
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Channel embedding (learnable, based on electrode identity)
        self.channel_embed = nn.Embedding(n_channels, embed_dim)

        # Spatial position embedding based on electrode topology
        if channel_names is not None:
            grid_positions = get_electrode_grid_positions(channel_names)
            self.register_buffer("electrode_positions", grid_positions)
            # Learnable spatial embedding (8x8 grid to cover 10-20 system)
            self.spatial_embed = nn.Embedding(64, embed_dim)  # 8x8 grid
        else:
            self.register_buffer("electrode_positions", None)
            self.spatial_embed = None

        # Temporal positional encoding (sinusoidal + learnable)
        self.time_pos_embed = nn.Parameter(torch.zeros(1, self.n_time_patches, embed_dim))
        # Frequency positional encoding
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, self.n_freq_patches, embed_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize positional embeddings with small values
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, n_channels, freq_bins, time_steps)

        Returns:
            Embedded patches of shape (batch, n_patches, embed_dim)
        """
        B, C, F, T = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"

        # Process each channel separately and collect patches
        all_patches = []

        for ch_idx in range(C):
            # Extract single channel: (B, 1, F, T)
            ch_data = x[:, ch_idx : ch_idx + 1, :, :]

            # Apply patch embedding: (B, embed_dim, n_freq_patches, n_time_patches)
            ch_patches = self.proj(ch_data)

            # Reshape to (B, n_freq_patches, n_time_patches, embed_dim)
            ch_patches = ch_patches.permute(0, 2, 3, 1)

            # Add frequency positional encoding
            ch_patches = ch_patches + self.freq_pos_embed.unsqueeze(2)

            # Add temporal positional encoding
            ch_patches = ch_patches + self.time_pos_embed.unsqueeze(1)

            # Add channel embedding
            ch_embed = self.channel_embed.weight[ch_idx]  # (embed_dim,)
            ch_patches = ch_patches + ch_embed

            # Add spatial (electrode topology) embedding if available
            if self.spatial_embed is not None and self.electrode_positions is not None:
                row, col = self.electrode_positions[ch_idx]
                spatial_idx = row * 8 + col  # 8x8 grid
                spatial_emb = self.spatial_embed.weight[spatial_idx]
                ch_patches = ch_patches + spatial_emb

            # Flatten to (B, patches_per_channel, embed_dim)
            ch_patches = ch_patches.reshape(B, -1, self.embed_dim)
            all_patches.append(ch_patches)

        # Concatenate all channel patches: (B, n_patches, embed_dim)
        patches = torch.cat(all_patches, dim=1)

        return patches


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))

        return x


class EEGViT(nn.Module):
    """EEG-specific Vision Transformer for Motor Imagery Classification.

    A lightweight ViT designed specifically for EEG spectrograms with:
    - Channel-aware patch embedding with electrode topology
    - Temporal and frequency positional encodings
    - Smaller patch sizes (4x4) for fine-grained features
    - Fewer transformer layers (4-6) suited for EEG data scale

    Args:
        n_channels: Number of EEG channels (default: 9)
        freq_bins: Frequency bins in spectrogram (default: 64)
        time_steps: Time steps in spectrogram (default: 64)
        patch_size: Patch size (default: 4 for fine-grained)
        embed_dim: Embedding dimension (default: 192)
        depth: Number of transformer blocks (default: 6)
        num_heads: Number of attention heads (default: 3)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        n_classes: Number of output classes (default: 2)
        drop_rate: Dropout rate (default: 0.1)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        channel_names: List of electrode names for topology encoding
        use_cls_token: Whether to use a CLS token (default: True)
    """

    # Feature dimension for external reference
    FEATURE_DIM = 192

    def __init__(
        self,
        n_channels: int = 9,
        freq_bins: int = 64,
        time_steps: int = 64,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        n_classes: int = 2,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        channel_names: list[str] | None = None,
        use_cls_token: bool = True,
        use_covariance_token: bool = False,
        use_temporal_encoder: bool = False,
        as_feature_extractor: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim  # For compatibility with existing code
        self.use_cls_token = use_cls_token
        self.use_covariance_token = use_covariance_token
        self.use_temporal_encoder = bool(use_temporal_encoder)
        self.n_channels = int(n_channels)
        self.as_feature_extractor = as_feature_extractor

        if self.use_temporal_encoder:
            self.temporal_encoder = TemporalEncoder(in_height=n_channels * freq_bins)
            self.temporal_to_spectrogram = nn.Linear(64, n_channels * freq_bins)
        else:
            self.temporal_encoder = None
            self.temporal_to_spectrogram = None

        # Default channel names for motor cortex
        if channel_names is None:
            channel_names = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"]
        self.channel_names = channel_names

        # Channel-aware patch embedding
        self.patch_embed = ChannelPatchEmbedding(
            n_channels=n_channels,
            freq_bins=freq_bins,
            time_steps=time_steps,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channel_names=channel_names,
        )

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Positional embedding for the full sequence (including optional tokens)
        n_patches = self.patch_embed.n_patches
        seq_len = n_patches + (1 if use_cls_token else 0) + (1 if use_covariance_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if use_covariance_token:
            self.cov_proj = nn.Sequential(
                nn.LayerNorm(self.n_channels * self.n_channels),
                nn.Linear(self.n_channels * self.n_channels, embed_dim),
            )
        else:
            self.cov_proj = None

        # Dropout after embedding
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        if not as_feature_extractor:
            self.head = nn.Linear(embed_dim, n_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = nn.Identity()

        self._log_architecture()

    def _log_architecture(self) -> None:
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "EEGViT: %d patches/channel, %d total patches, %d params (%d trainable)",
            self.patch_embed.patches_per_channel,
            self.patch_embed.n_patches,
            n_params,
            n_trainable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, n_channels, freq_bins, time_steps)
               OR (batch, n_channels, H, W) for spectrogram images

        Returns:
            Logits of shape (batch, n_classes) or features of shape (batch, embed_dim)
        """
        B = x.shape[0]

        if self.use_temporal_encoder and self.temporal_encoder is not None:
            x_tf = x.reshape(
                B, 1, self.n_channels * self.patch_embed.freq_bins, self.patch_embed.time_steps
            )
            x_seq = self.temporal_encoder(x_tf)
            x_seq = x_seq.transpose(1, 2)
            x_spec = self.temporal_to_spectrogram(x_seq)
            x = x_spec.transpose(1, 2).reshape(B, self.n_channels, self.patch_embed.freq_bins, -1)
            if x.shape[-1] != self.patch_embed.time_steps:
                x = F.interpolate(
                    x,
                    size=(self.patch_embed.freq_bins, self.patch_embed.time_steps),
                    mode="bilinear",
                    align_corners=False,
                )

        # Patch embedding: (B, n_patches, embed_dim)
        x_raw = x
        x = self.patch_embed(x)

        # Prepend CLS token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        if self.use_covariance_token and self.cov_proj is not None:
            x_flat = x_raw.flatten(start_dim=2)
            x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
            denom = max(x_centered.shape[-1] - 1, 1)
            cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / float(denom)
            cov_vec = cov.flatten(start_dim=1)
            cov_token = self.cov_proj(cov_vec).unsqueeze(1)
            x = torch.cat([x, cov_token], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Extract features (CLS token or mean pooling)
        if self.cls_token is not None:
            features = x[:, 0]  # CLS token
        else:
            features = x.mean(dim=1)  # Global average pooling

        # Classification head
        return self.head(features)

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Freeze backbone parameters for transfer learning."""
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks
        if unfreeze_last_n_blocks > 0:
            for block in self.blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze norm and head
        for param in self.norm.parameters():
            param.requires_grad = True
        if hasattr(self.head, "parameters"):
            for param in self.head.parameters():
                param.requires_grad = True

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        logger.info(
            "Frozen %d/%d parameters (unfroze last %d blocks + head)",
            frozen,
            total,
            unfreeze_last_n_blocks,
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class EEGViTWithMasking(EEGViT):
    """EEG-ViT with support for Masked Patch Prediction (self-supervised pretraining).

    Extends EEGViT with:
    - Random patch masking during forward pass
    - Reconstruction head for masked patch prediction
    - Contrastive learning support

    Args:
        mask_ratio: Fraction of patches to mask (default: 0.4)
        **kwargs: Arguments passed to EEGViT
    """

    def __init__(
        self,
        mask_ratio: float = 0.4,
        **kwargs,
    ) -> None:
        # Force feature extractor mode for pretraining
        kwargs["as_feature_extractor"] = True
        super().__init__(**kwargs)

        self.mask_ratio = mask_ratio

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Reconstruction head for masked patch prediction
        patch_dim = self.patch_embed.patch_size**2  # Pixels per patch
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, patch_dim),
        )

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patches for self-supervised learning.

        Args:
            x: Embedded patches of shape (B, N, D)
            mask_ratio: Override default mask ratio

        Returns:
            x_masked: Patches with mask tokens (B, N, D)
            mask: Boolean mask (B, N) - True for masked positions
            ids_restore: Indices to restore original order
        """
        B, N, D = x.shape
        mask_ratio = mask_ratio or self.mask_ratio

        # Number of patches to keep
        n_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Create mask: True for masked positions
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[:, :n_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Replace masked patches with mask token
        x_masked = x.clone()
        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked[mask] = mask_tokens[mask]

        return x_masked, mask, ids_restore

    def forward_masked(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with masking for pretraining.

        Args:
            x: Input tensor (B, C, F, T)
            mask_ratio: Override default mask ratio

        Returns:
            pred: Reconstructed patch pixels (B, n_masked, patch_dim)
            target: Original patch pixels (B, n_masked, patch_dim)
            mask: Boolean mask (B, N)
        """
        B = x.shape[0]

        # Get original patches before embedding for reconstruction target
        # Store original input for target computation
        original_x = x.clone()

        # Patch embedding
        patches = self.patch_embed(x)

        # Prepend CLS token (not masked)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)

        # Add positional embedding
        patches = patches + self.pos_embed

        # Random masking (skip CLS token)
        if self.cls_token is not None:
            cls_patch = patches[:, :1]
            patches_no_cls = patches[:, 1:]
            patches_masked, mask, ids_restore = self.random_masking(patches_no_cls, mask_ratio)
            patches = torch.cat([cls_patch, patches_masked], dim=1)
        else:
            patches, mask, ids_restore = self.random_masking(patches, mask_ratio)

        patches = self.pos_drop(patches)

        # Transformer blocks
        for block in self.blocks:
            patches = block(patches)

        patches = self.norm(patches)

        # Get features for masked positions
        if self.cls_token is not None:
            patches_out = patches[:, 1:]  # Remove CLS token
        else:
            patches_out = patches

        # Reconstruct only masked patches
        masked_features = patches_out[mask]  # (n_masked_total, D)
        pred = self.reconstruction_head(masked_features)

        # Compute target from original patches (mean pixel value per patch)
        # This is a simplified target - could use full patch pixels
        patch_size = self.patch_embed.patch_size
        n_channels = self.patch_embed.n_channels

        # Reshape original to get patch targets
        # For simplicity, use mean patch intensity as target
        with torch.no_grad():
            # Unfold original input to get patches
            # original_x: (B, C, F, T)
            patches_unfolded = F.unfold(
                original_x.reshape(B * n_channels, 1, original_x.shape[2], original_x.shape[3]),
                kernel_size=patch_size,
                stride=patch_size,
            )  # (B*C, patch_size^2, n_patches_per_channel)

            patches_unfolded = patches_unfolded.reshape(
                B, n_channels, patch_size**2, -1
            )  # (B, C, patch_dim, patches_per_channel)

            # Reshape to (B, total_patches, patch_dim)
            # Note: patches are ordered by channel, then by spatial position
            patches_target = patches_unfolded.permute(0, 1, 3, 2).reshape(B, -1, patch_size**2)

            # Get targets for masked positions
            target = patches_target[mask]

        return pred, target, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (no masking) - returns features."""
        return super().forward(x)


# Convenience factory function
def create_eeg_vit(
    n_channels: int = 9,
    freq_bins: int = 64,
    time_steps: int = 64,
    variant: Literal["tiny", "small", "base"] = "tiny",
    n_classes: int = 2,
    drop_rate: float = 0.1,
    channel_names: list[str] | None = None,
    as_feature_extractor: bool = False,
) -> EEGViT:
    """Create an EEG-ViT model with predefined configurations.

    Args:
        n_channels: Number of EEG channels
        freq_bins: Frequency bins in spectrogram
        time_steps: Time steps in spectrogram
        variant: Model size ("tiny", "small", "base")
        n_classes: Number of output classes
        drop_rate: Dropout rate
        channel_names: List of electrode names
        as_feature_extractor: Return features instead of logits

    Returns:
        EEGViT model

    Note:
        Patch sizes are chosen to balance between fine-grained features and
        memory efficiency. With 9 channels and 64x64 spectrograms:
        - patch_size=8: 64 patches/channel, 576 total patches (recommended)
        - patch_size=16: 16 patches/channel, 144 total patches (more efficient)
        - patch_size=4: 256 patches/channel, 2304 total patches (OOM on 8GB GPU)
    """
    # Use larger patch sizes to reduce memory usage while still capturing
    # fine-grained temporal/frequency information
    configs = {
        "tiny": {
            "patch_size": 8,  # 64 patches per channel, 576 total (memory efficient)
            "embed_dim": 192,
            "depth": 4,
            "num_heads": 3,
            "mlp_ratio": 4.0,
        },
        "small": {
            "patch_size": 8,
            "embed_dim": 256,
            "depth": 6,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        },
        "base": {
            "patch_size": 8,
            "embed_dim": 384,
            "depth": 8,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
    }

    cfg = configs[variant]

    return EEGViT(
        n_channels=n_channels,
        freq_bins=freq_bins,
        time_steps=time_steps,
        n_classes=n_classes,
        drop_rate=drop_rate,
        channel_names=channel_names,
        as_feature_extractor=as_feature_extractor,
        **cfg,
    )


# Export feature dimension for external reference
FEATURE_DIM = EEGViT.FEATURE_DIM
