"""Feature fusion strategies for combining ViT and Math branch outputs.

Supported fusion methods:
    - "concat": Simple concatenation of feature vectors
    - "attention": Learnable attention-based gating
    - "gated": Sigmoid gating mechanism
    - "late": Late fusion at classifier stage (features processed separately until final layer)
    - "attention_v2": Improved attention fusion with per-element gating

The attention-based fusion learns to weight the importance of each branch
dynamically, allowing the model to rely more on whichever branch provides
more discriminative features for a given input.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from bci.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion.

    Concatenates two feature vectors and optionally projects to a lower dimension.

    Args:
        dim_a: Dimension of branch A (ViT) features.
        dim_b: Dimension of branch B (Math) features.
        output_dim: Desired output dimension. If None, output_dim = dim_a + dim_b.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        combined_dim = dim_a + dim_b
        self.output_dim = output_dim or combined_dim

        if output_dim is not None and output_dim != combined_dim:
            self.projection = nn.Sequential(
                nn.Linear(combined_dim, output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Concatenate and optionally project.

        Args:
            feat_a: (batch, dim_a) from ViT branch.
            feat_b: (batch, dim_b) from Math branch.

        Returns:
            Fused features of shape (batch, output_dim).
        """
        combined = torch.cat([feat_a, feat_b], dim=-1)
        return self.projection(combined)


class AttentionFusion(nn.Module):
    """Attention-based feature fusion.

    Learns to weight the two branches using a shared attention mechanism.
    Each branch's features are first projected to a common dimension,
    then weighted by learned attention scores.

    Args:
        dim_a: Dimension of branch A (ViT) features.
        dim_b: Dimension of branch B (Math) features.
        output_dim: Dimension of fused output.
    """

    def __init__(self, dim_a: int, dim_b: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Project both branches to the same dimension
        self.proj_a = nn.Linear(dim_a, output_dim)
        self.proj_b = nn.Linear(dim_b, output_dim)

        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1),
        )

        logger.info(
            "AttentionFusion: dim_a=%d, dim_b=%d -> output_dim=%d",
            dim_a,
            dim_b,
            output_dim,
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Fuse features with learned attention weights.

        Args:
            feat_a: (batch, dim_a) from ViT branch.
            feat_b: (batch, dim_b) from Math branch.

        Returns:
            Fused features of shape (batch, output_dim).
        """
        proj_a = self.proj_a(feat_a)  # (batch, output_dim)
        proj_b = self.proj_b(feat_b)  # (batch, output_dim)

        # Compute attention weights
        combined = torch.cat([proj_a, proj_b], dim=-1)  # (batch, 2 * output_dim)
        weights = self.attention(combined)  # (batch, 2)

        # Weighted combination
        w_a = weights[:, 0:1]  # (batch, 1)
        w_b = weights[:, 1:2]  # (batch, 1)

        fused = w_a * proj_a + w_b * proj_b  # (batch, output_dim)
        return fused


class AttentionFusionV2(nn.Module):
    """Improved attention fusion with per-element gating.

    This implements the improved fusion from the MI-EEG improvement plan:

        w1 = sigmoid(W1 * f1)
        w2 = sigmoid(W2 * f2)
        output = w1 * f1 + w2 * f2

    Each element in the feature vector gets its own learned gate, allowing
    fine-grained control over which features to emphasize from each branch.

    Args:
        dim_a: Dimension of branch A (ViT) features.
        dim_b: Dimension of branch B (Math) features.
        output_dim: Dimension of fused output.
    """

    def __init__(self, dim_a: int, dim_b: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Project both branches to the same dimension
        self.proj_a = nn.Linear(dim_a, output_dim)
        self.proj_b = nn.Linear(dim_b, output_dim)

        # Per-element gating networks (sigmoid activation)
        # These learn which features to emphasize from each branch
        self.gate_a = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.gate_b = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )

        # Optional final projection to refine the fusion
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

        logger.info(
            "AttentionFusionV2: dim_a=%d, dim_b=%d -> output_dim=%d",
            dim_a,
            dim_b,
            output_dim,
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Fuse features with per-element gating.

        Args:
            feat_a: (batch, dim_a) from ViT branch.
            feat_b: (batch, dim_b) from Math branch.

        Returns:
            Fused features of shape (batch, output_dim).
        """
        proj_a = self.proj_a(feat_a)  # (batch, output_dim)
        proj_b = self.proj_b(feat_b)  # (batch, output_dim)

        # Per-element gating
        w_a = self.gate_a(proj_a)  # (batch, output_dim) in [0, 1]
        w_b = self.gate_b(proj_b)  # (batch, output_dim) in [0, 1]

        # Weighted combination with per-element gates
        fused = w_a * proj_a + w_b * proj_b  # (batch, output_dim)

        # Final projection
        fused = self.final_proj(fused)

        return fused


class GatedFusion(nn.Module):
    """Gated fusion using sigmoid gating.

    Uses a learned gate to control how much each branch contributes.

    Args:
        dim_a: Dimension of branch A features.
        dim_b: Dimension of branch B features.
        output_dim: Dimension of fused output.
    """

    def __init__(self, dim_a: int, dim_b: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.proj_a = nn.Linear(dim_a, output_dim)
        self.proj_b = nn.Linear(dim_b, output_dim)

        # Gate: takes concatenated features, outputs gate values in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(dim_a + dim_b, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Fuse with sigmoid gating.

        Args:
            feat_a: (batch, dim_a) from ViT branch.
            feat_b: (batch, dim_b) from Math branch.

        Returns:
            Fused features of shape (batch, output_dim).
        """
        proj_a = self.proj_a(feat_a)
        proj_b = self.proj_b(feat_b)

        gate_input = torch.cat([feat_a, feat_b], dim=-1)
        g = self.gate(gate_input)  # (batch, output_dim) in [0, 1]

        fused = g * proj_a + (1 - g) * proj_b
        return fused


class LateFusion(nn.Module):
    """Late fusion at classifier stage.

    Per the improvement plan: "Do NOT fuse too early."

    This module keeps features separate until the final classification layer,
    processing each branch through its own high-level feature extractor first.

    Architecture:
        ViT features -> high-level MLP -> vit_high
        CSP features -> high-level MLP -> csp_high
        -> Concatenate at classifier stage

    Args:
        dim_a: Dimension of branch A (ViT) features.
        dim_b: Dimension of branch B (Math) features.
        output_dim: Dimension of fused output.
        hidden_dim: Hidden dimension for each branch's MLP.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # High-level feature extraction for each branch (separate)
        self.vit_mlp = nn.Sequential(
            nn.Linear(dim_a, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.math_mlp = nn.Sequential(
            nn.Linear(dim_b, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Final projection after concatenation
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

        logger.info(
            "LateFusion: dim_a=%d, dim_b=%d -> hidden=%d -> output_dim=%d",
            dim_a,
            dim_b,
            hidden_dim,
            output_dim,
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Fuse at late stage.

        Args:
            feat_a: (batch, dim_a) from ViT branch.
            feat_b: (batch, dim_b) from Math branch.

        Returns:
            Fused features of shape (batch, output_dim).
        """
        # Process each branch to high-level features
        vit_high = self.vit_mlp(feat_a)  # (batch, hidden_dim)
        math_high = self.math_mlp(feat_b)  # (batch, hidden_dim)

        # Late concatenation and final projection
        combined = torch.cat([vit_high, math_high], dim=-1)  # (batch, hidden_dim * 2)
        fused = self.final_proj(combined)  # (batch, output_dim)

        return fused


def create_fusion(
    dim_a: int,
    dim_b: int,
    config: ModelConfig | None = None,
) -> nn.Module:
    """Factory function to create the appropriate fusion module.

    Args:
        dim_a: Dimension of ViT branch features.
        dim_b: Dimension of Math branch features.
        config: Model configuration.

    Returns:
        Fusion module.
    """
    config = config or ModelConfig()
    method = config.fusion_method
    output_dim = config.fused_dim

    if method == "concat":
        return ConcatFusion(dim_a, dim_b, output_dim)
    elif method == "attention":
        return AttentionFusion(dim_a, dim_b, output_dim)
    elif method == "attention_v2":
        return AttentionFusionV2(dim_a, dim_b, output_dim)
    elif method == "gated":
        return GatedFusion(dim_a, dim_b, output_dim)
    elif method == "late":
        return LateFusion(dim_a, dim_b, output_dim)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
