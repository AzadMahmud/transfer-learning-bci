"""EEG-ViT branch used across the full pipeline.

This module keeps the historical ``ViTBranch`` API used by training scripts,
but the implementation is now the improved EEG-specific ViT architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from bci.models.eeg_vit import EEGViT
from bci.utils.config import DEFAULT_CLS_HIDDEN, DEFAULT_FUSED_DIM, ModelConfig

logger = logging.getLogger(__name__)

# Backward-compatible constants used by pipeline scripts.
BACKBONE_SHORT: str = "vit"
MODEL_NAME: str = "eeg_vit_tiny_patch8_64"
FEATURE_DIM: int = 192


def load_backbone_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    min_match_ratio: float = 0.95,
    strict_min_match: bool = True,
) -> dict[str, float | int]:
    """Load a backbone checkpoint with explicit compatibility diagnostics.

    Returns a metrics dictionary containing tensor- and parameter-level match ratios.
    Raises ``RuntimeError`` when ``strict_min_match`` is enabled and the matched
    parameter ratio is lower than ``min_match_ratio``.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = {
        k: v for k, v in state.items() if not (k.startswith("head") or k.startswith("classifier"))
    }

    target_state = backbone.state_dict()

    # Handle optional extra token in the target positional embedding (e.g. cov token).
    if "pos_embed" in state and "pos_embed" in target_state:
        src_pe = state["pos_embed"]
        tgt_pe = target_state["pos_embed"]
        if src_pe.shape != tgt_pe.shape and src_pe.ndim == 3 and tgt_pe.ndim == 3:
            if src_pe.shape[0] == tgt_pe.shape[0] and src_pe.shape[2] == tgt_pe.shape[2]:
                if src_pe.shape[1] + 1 == tgt_pe.shape[1]:
                    pad = torch.zeros(src_pe.shape[0], 1, src_pe.shape[2], dtype=src_pe.dtype)
                    state["pos_embed"] = torch.cat([src_pe, pad], dim=1)
                elif src_pe.shape[1] - 1 == tgt_pe.shape[1]:
                    state["pos_embed"] = src_pe[:, : tgt_pe.shape[1], :]

    matched = {
        k: v for k, v in state.items() if k in target_state and target_state[k].shape == v.shape
    }
    missing, unexpected = backbone.load_state_dict(matched, strict=False)

    total_tensors = len(target_state)
    matched_tensors = len(matched)
    total_params = int(sum(v.numel() for v in target_state.values()))
    matched_params = int(sum(v.numel() for v in matched.values()))
    tensor_ratio = matched_tensors / max(1, total_tensors)
    param_ratio = matched_params / max(1, total_params)

    logger.info(
        "Checkpoint load diagnostics: matched_tensors=%d/%d (%.1f%%), matched_params=%d/%d (%.1f%%), missing=%d, unexpected=%d",
        matched_tensors,
        total_tensors,
        tensor_ratio * 100.0,
        matched_params,
        total_params,
        param_ratio * 100.0,
        len(missing),
        len(unexpected),
    )

    if strict_min_match and param_ratio < min_match_ratio:
        raise RuntimeError(
            "Checkpoint compatibility too low: "
            f"matched_params={param_ratio * 100.0:.1f}% < required {min_match_ratio * 100.0:.1f}%."
        )

    return {
        "matched_tensors": matched_tensors,
        "total_tensors": total_tensors,
        "matched_params": matched_params,
        "total_params": total_params,
        "tensor_ratio": float(tensor_ratio),
        "param_ratio": float(param_ratio),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


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
        embed_dim: int = FEATURE_DIM,
        depth: int = 4,
        num_heads: int = 3,
        use_covariance_token: bool = False,
        use_temporal_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.img_size = int(img_size or 64)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.use_covariance_token = bool(use_covariance_token)
        self.use_temporal_encoder = bool(use_temporal_encoder)

        self.backbone = EEGViT(
            n_channels=self.config.in_chans,
            freq_bins=self.img_size,
            time_steps=self.img_size,
            patch_size=8,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            n_classes=self.config.n_classes,
            drop_rate=self.config.vit_drop_rate,
            channel_names=["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"],
            use_covariance_token=self.use_covariance_token,
            use_temporal_encoder=self.use_temporal_encoder,
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
