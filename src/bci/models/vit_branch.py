"""EEG-ViT branch used across the full pipeline.

This module keeps the historical ``ViTBranch`` API used by training scripts,
but the implementation is now the improved EEG-specific ViT architecture.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from bci.models.eeg_vit import EEGViT
from bci.utils.config import DEFAULT_CLS_HIDDEN, DEFAULT_FUSED_DIM, ModelConfig

logger = logging.getLogger(__name__)

# Backward-compatible constants used by pipeline scripts.
BACKBONE_SHORT: str = "vit"
MODEL_NAME: str = "eeg_vit_tiny_patch8_64"
FEATURE_DIM: int = 192


class ViTBranch(nn.Module):
    """Compatibility wrapper around the improved EEGViT backbone.

    Args:
        config: Model configuration.
        as_feature_extractor: If True, returns features; else returns logits.
        img_size: Input spectrogram size. Defaults to 64.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        as_feature_extractor: bool = True,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.img_size = int(img_size or 64)

        self.backbone = EEGViT(
            n_channels=self.config.in_chans,
            freq_bins=self.img_size,
            time_steps=self.img_size,
            patch_size=8,
            embed_dim=FEATURE_DIM,
            depth=4,
            num_heads=3,
            mlp_ratio=4.0,
            n_classes=self.config.n_classes,
            drop_rate=self.config.vit_drop_rate,
            channel_names=["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"],
            as_feature_extractor=as_feature_extractor,
        )
        self.feature_dim = self.backbone.feature_dim

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (self.img_size, self.img_size):
            return x
        return torch.nn.functional.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_if_needed(x)
        return self.backbone(x)

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        self.backbone.freeze_backbone(unfreeze_last_n_blocks)

    def unfreeze_all(self) -> None:
        self.backbone.unfreeze_all()

    def get_num_params(self, trainable_only: bool = True) -> int:
        return self.backbone.get_num_params(trainable_only=trainable_only)

    def get_backbone_params(self) -> list[nn.Parameter]:
        head_ids = {id(p) for p in self.get_head_params()}
        return [p for p in self.backbone.parameters() if id(p) not in head_ids]

    def get_head_params(self) -> list[nn.Parameter]:
        if hasattr(self.backbone.head, "parameters"):
            return list(self.backbone.head.parameters())
        return []

    def get_layerwise_param_groups(self) -> list[tuple[str, list[nn.Parameter]]]:
        groups: list[tuple[str, list[nn.Parameter]]] = []
        groups.append(("patch_embed", list(self.backbone.patch_embed.parameters())))
        groups.append(("pos_embed", [self.backbone.pos_embed]))
        if self.backbone.cls_token is not None:
            groups.append(("cls_token", [self.backbone.cls_token]))
        for i, block in enumerate(self.backbone.blocks):
            groups.append((f"block_{i}", list(block.parameters())))
        groups.append(("norm", list(self.backbone.norm.parameters())))
        groups.append(("head", self.get_head_params()))
        return groups
