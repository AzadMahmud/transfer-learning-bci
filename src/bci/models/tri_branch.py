"""Tri-branch adaptive fusion model for MI-EEG.

This model fuses three expert branches:
    1) ViT branch on spectrograms
    2) CSP-feature branch
    3) Riemannian-feature branch

Fusion is adaptive per-sample: the model predicts branch weights and gives
higher weight to branches that are more confident for that sample.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from bci.models.vit_branch import ViTBranch
from bci.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class _FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, drop_rate: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriBranchAdaptiveModel(nn.Module):
    """ViT + CSP + Riemannian adaptive logit fusion."""

    def __init__(
        self,
        math_input_dim: int,
        csp_dim: int,
        config: ModelConfig | None = None,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        if csp_dim <= 0 or csp_dim >= math_input_dim:
            raise ValueError(f"Invalid csp_dim={csp_dim} for math_input_dim={math_input_dim}")

        self.csp_dim = csp_dim
        self.riemann_dim = math_input_dim - csp_dim

        self.vit_branch = ViTBranch(
            config=self.config, as_feature_extractor=True, img_size=img_size
        )
        self.vit_embed = nn.Sequential(
            nn.Linear(self.vit_branch.feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.csp_branch = _FeatureMLP(self.csp_dim, hidden_dim=64, drop_rate=0.2)
        self.riemann_branch = _FeatureMLP(self.riemann_dim, hidden_dim=64, drop_rate=0.2)

        n_classes = self.config.n_classes
        self.vit_head = nn.Linear(64, n_classes)
        self.csp_head = nn.Linear(64, n_classes)
        self.riemann_head = nn.Linear(64, n_classes)

        # Gating uses branch embeddings + confidence summary (max prob for each branch)
        self.gate = nn.Sequential(
            nn.Linear(64 * 3 + 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TriBranchAdaptiveModel: %d total params (%d trainable), csp_dim=%d, riemann_dim=%d",
            total_params,
            trainable_params,
            self.csp_dim,
            self.riemann_dim,
        )

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        csp = features[:, : self.csp_dim]
        riemann = features[:, self.csp_dim :]

        vit_feat = self.vit_embed(self.vit_branch(images))
        csp_feat = self.csp_branch(csp)
        riem_feat = self.riemann_branch(riemann)

        vit_logits = self.vit_head(vit_feat)
        csp_logits = self.csp_head(csp_feat)
        riem_logits = self.riemann_head(riem_feat)

        with torch.no_grad():
            vit_conf = vit_logits.softmax(dim=-1).amax(dim=-1, keepdim=True)
            csp_conf = csp_logits.softmax(dim=-1).amax(dim=-1, keepdim=True)
            riem_conf = riem_logits.softmax(dim=-1).amax(dim=-1, keepdim=True)

        gate_in = torch.cat([vit_feat, csp_feat, riem_feat, vit_conf, csp_conf, riem_conf], dim=-1)
        weights = self.gate(gate_in).softmax(dim=-1)

        fused_logits = (
            weights[:, 0:1] * vit_logits
            + weights[:, 1:2] * csp_logits
            + weights[:, 2:3] * riem_logits
        )
        return fused_logits

    def freeze_vit_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        self.vit_branch.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)
