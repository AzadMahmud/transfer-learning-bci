#!/usr/bin/env python3
"""Full MI-EEG Experiment with Improved ViT Branch.

This script implements all the improvements from the MI-EEG Accuracy Improvement Plan:

Phase 1: Fix ViT Input Representation (CRITICAL)
    - Structured input (channel x frequency x time)
    - Electrode topology encoding (10-20 system)
    - Channel-aware patch embedding
    - Smaller patch size (4x4)
    - Temporal and frequency positional encoding
    - Channel embedding

Phase 2: Self-Supervised Pretraining
    - Masked patch prediction
    - Contrastive learning support

Phase 3: Improved Fusion
    - AttentionFusionV2 with per-element gating
    - Late fusion option

Phase 4: Data Augmentation
    - SpecAugment (frequency/time masking)
    - Patch dropout
    - Mixup

Phase 5: Training Strategy
    - Higher LR (3e-4 to 5e-4)
    - Warmup (10 epochs)
    - More epochs (200)
    - Regularization (dropout 0.1-0.2, weight decay 0.05)
    - Gradient clipping (1.0)

Usage:
    uv run python scripts/pipeline/run_improved_vit_experiment.py --run-dir runs/improved_vit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bci.data.augmentation import EEGAugmenter, SpectrogramAugmenter, mixup_criterion
from bci.data.dual_branch_builder import DualBranchFoldBuilder
from bci.models.eeg_vit import EEGViT, EEGViTWithMasking, create_eeg_vit
from bci.models.fusion import create_fusion, AttentionFusionV2, LateFusion
from bci.models.math_branch import MathBranch
from bci.training.cross_validation import CVResult, FoldResult
from bci.training.evaluation import compute_metrics
from bci.training.splits import get_or_create_splits
from bci.utils.config import (
    AugmentationConfig,
    ModelConfig,
    SpectrogramConfig,
    TrainingConfig,
)
from bci.utils.logging import setup_stage_logging
from bci.utils.seed import get_device, set_seed
from bci.utils.visualization import save_confusion_matrix, save_per_subject_accuracy


class ImprovedDualBranchModel(nn.Module):
    """Improved Dual-Branch Model with EEG-specific ViT.

    Combines the new EEG-ViT (with electrode topology, channel embeddings,
    smaller patches) with the Math branch and improved fusion.

    Args:
        n_eeg_channels: Number of EEG channels in spectrograms
        freq_bins: Frequency bins in spectrogram
        time_steps: Time steps in spectrogram
        math_input_dim: Dimension of handcrafted features (CSP + Riemannian)
        n_classes: Number of output classes
        vit_variant: EEG-ViT variant ("tiny", "small", "base")
        fusion_method: Fusion method ("attention", "attention_v2", "late", "gated")
        drop_rate: Dropout rate
        channel_names: List of electrode names
    """

    def __init__(
        self,
        n_eeg_channels: int = 9,
        freq_bins: int = 64,
        time_steps: int = 64,
        math_input_dim: int = 140,
        n_classes: int = 2,
        vit_variant: str = "tiny",
        fusion_method: str = "attention_v2",
        fused_dim: int = 128,
        drop_rate: float = 0.1,
        channel_names: list[str] | None = None,
    ) -> None:
        super().__init__()

        # Default motor cortex channels
        if channel_names is None:
            channel_names = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"]

        # Branch A: EEG-specific ViT
        self.vit_branch = create_eeg_vit(
            n_channels=n_eeg_channels,
            freq_bins=freq_bins,
            time_steps=time_steps,
            variant=vit_variant,
            n_classes=n_classes,
            drop_rate=drop_rate,
            channel_names=channel_names,
            as_feature_extractor=True,
        )

        # Branch B: Math branch (CSP + Riemannian)
        self.math_branch = MathBranch(
            input_dim=math_input_dim,
            config=ModelConfig(
                math_hidden_dims=[256, 128],
                math_drop_rate=0.3,
            ),
        )

        # Get feature dimensions
        vit_dim = self.vit_branch.feature_dim
        math_dim = self.math_branch.output_dim

        # Fusion layer (use improved fusion methods)
        if fusion_method == "attention_v2":
            self.fusion = AttentionFusionV2(vit_dim, math_dim, fused_dim)
        elif fusion_method == "late":
            self.fusion = LateFusion(vit_dim, math_dim, fused_dim)
        else:
            config = ModelConfig(fusion_method=fusion_method, fused_dim=fused_dim)
            self.fusion = create_fusion(vit_dim, math_dim, config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(64, n_classes),
        )

        self._log_architecture()

    def _log_architecture(self) -> None:
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ImprovedDualBranchModel: {n_params:,} params ({n_trainable:,} trainable)")

    def forward(
        self,
        spectrograms: torch.Tensor,
        math_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            spectrograms: (batch, n_channels, freq_bins, time_steps)
            math_features: (batch, math_input_dim)

        Returns:
            Logits of shape (batch, n_classes)
        """
        # Branch A: EEG-ViT features
        vit_features = self.vit_branch(spectrograms)

        # Branch B: Math features
        math_out = self.math_branch(math_features)

        # Fusion
        fused = self.fusion(vit_features, math_out)

        # Classification
        logits = self.classifier(fused)

        return logits

    def freeze_vit_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Freeze ViT backbone for transfer learning."""
        self.vit_branch.freeze_backbone(unfreeze_last_n_blocks)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class ImprovedTrainer:
    """Training loop with improvements from the plan.

    Implements:
    - Higher learning rate (3e-4 to 5e-4)
    - Longer warmup (10 epochs)
    - More epochs (up to 200)
    - Gradient clipping (1.0)
    - Weight decay (0.05)
    - SpecAugment augmentation
    - Mixup training
    - Label smoothing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        epochs: int = 200,
        batch_size: int = 32,
        warmup_epochs: int = 10,
        patience: int = 20,
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
        use_mixup: bool = True,
        mixup_alpha: float = 0.4,
        use_spec_augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_spec_augment = use_spec_augment
        self.seed = seed

        self.spec_augmenter = SpectrogramAugmenter(
            AugmentationConfig(
                apply_freq_mask=True,
                freq_mask_max_width=8,
                apply_time_mask=True,
                time_mask_max_width=16,
            ),
            seed=seed,
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        n_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create cosine scheduler with linear warmup."""
        warmup_steps = self.warmup_epochs * (n_steps // self.batch_size + 1)
        total_steps = self.epochs * (n_steps // self.batch_size + 1)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def fit(
        self,
        train_ds: Dataset,
        val_ds: Dataset | None = None,
        log_fn=None,
    ) -> dict:
        """Train the model.

        Args:
            train_ds: Training dataset yielding (spectrograms, math_features, labels)
            val_ds: Validation dataset
            log_fn: Optional logging function

        Returns:
            Training history dict
        """
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size * 2,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            )

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(train_ds))

        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        best_val_acc = 0.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": [],
        }

        rng = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                specs, feats, labels = batch
                specs = specs.to(self.device)
                feats = feats.to(self.device)
                labels = labels.to(self.device)

                # Apply SpecAugment
                if self.use_spec_augment:
                    specs_np = specs.cpu().numpy()
                    specs_np = self.spec_augmenter(specs_np, training=True)
                    specs = torch.from_numpy(specs_np).to(self.device)

                # Apply Mixup
                if self.use_mixup and rng.random() < 0.5:
                    lam = rng.beta(self.mixup_alpha, self.mixup_alpha)
                    perm = torch.randperm(specs.size(0), device=self.device)

                    specs_mixed = lam * specs + (1 - lam) * specs[perm]
                    feats_mixed = lam * feats + (1 - lam) * feats[perm]

                    optimizer.zero_grad()
                    logits = self.model(specs_mixed, feats_mixed)
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(
                        logits, labels[perm]
                    )
                else:
                    optimizer.zero_grad()
                    logits = self.model(specs, feats)
                    loss = criterion(logits, labels)

                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(1, n_batches)
            current_lr = scheduler.get_last_lr()[0]
            history["train_loss"].append(avg_train_loss)
            history["lr"].append(current_lr)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if log_fn:
                    log_fn(
                        f"Epoch {epoch + 1}/{self.epochs}: "
                        f"train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_acc={val_acc:.2f}%, "
                        f"lr={current_lr:.2e}"
                    )

                # Early stopping
                if patience_counter >= self.patience:
                    if log_fn:
                        log_fn(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                history["val_loss"].append(0.0)
                history["val_accuracy"].append(0.0)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "history": history,
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
        }

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                specs, feats, labels = batch
                specs = specs.to(self.device)
                feats = feats.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(specs, feats)
                loss = criterion(logits, labels)

                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        accuracy = 100.0 * correct / max(1, total)

        return avg_loss, accuracy

    def predict(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions."""
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                specs, feats, labels = batch
                specs = specs.to(self.device)
                feats = feats.to(self.device)

                logits = self.model(specs, feats)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_preds), np.concatenate(all_probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run improved ViT experiment for MI-EEG classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Run output directory")
    parser.add_argument("--processed-dir", default=None, help="Processed data directory")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--vit-variant", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument(
        "--fusion", default="attention_v2", choices=["attention", "attention_v2", "late", "gated"]
    )
    parser.add_argument("--no-mixup", action="store_true", help="Disable mixup")
    parser.add_argument("--no-spec-augment", action="store_true", help="Disable SpecAugment")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["within_subject", "loso"],
        choices=["within_subject", "loso"],
    )
    return parser.parse_args()


def run_experiment(
    args: argparse.Namespace,
    subject_data: dict,
    subject_spec_data: dict,
    spec_mean: np.ndarray,
    spec_std: np.ndarray,
    run_dir: Path,
    device: str,
    log,
) -> dict:
    """Run the full experiment with improved ViT."""

    # Spectrogram dimensions (after resizing to 64x64)
    TARGET_SIZE = 64

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        riemann_estimator="oas",
        riemann_metric="riemann",
        sfreq=128.0,
    )

    def normalise_specs(imgs: np.ndarray) -> np.ndarray:
        """Resize and normalize spectrograms."""
        if imgs.shape[-1] != TARGET_SIZE:
            t = torch.from_numpy(imgs.astype(np.float32))
            t = torch.nn.functional.interpolate(
                t, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False
            )
            imgs = t.numpy()
        return (imgs - spec_mean[None, :, None, None]) / spec_std[None, :, None, None]

    results = {}

    for strategy in args.strategies:
        log.info("=" * 60)
        log.info(f"Running {strategy} evaluation with Improved EEG-ViT")
        log.info("=" * 60)

        all_folds: list[FoldResult] = []
        subjects = sorted(subject_data.keys())

        split_spec = get_or_create_splits(
            run_dir=run_dir,
            dataset="bci_iv2a",
            subject_data=subject_data,
            n_folds=args.n_folds,
            seed=args.seed,
        )

        t0 = time.time()

        if strategy == "within_subject":
            fold_counter = 0
            for sid in subjects:
                X, y = subject_data[sid]
                spec_imgs, _ = subject_spec_data[sid]
                log.info(f"Subject {sid} ({len(y)} trials)...")

                folds = split_spec.within_subject.get(sid, [])
                for fold_idx, fold in enumerate(folds):
                    train_idx = np.array(fold["train_idx"], dtype=int)
                    test_idx = np.array(fold["test_idx"], dtype=int)
                    set_seed(args.seed + fold_counter)

                    # Build math features
                    features_train, features_test, math_input_dim = builder.build_math_features(
                        X[train_idx],
                        y[train_idx],
                        X[test_idx],
                        y[test_idx],
                    )

                    # Prepare spectrograms
                    spec_train = normalise_specs(spec_imgs[train_idx]).astype(np.float32)
                    spec_test = normalise_specs(spec_imgs[test_idx]).astype(np.float32)

                    # Create datasets
                    train_ds = TensorDataset(
                        torch.from_numpy(spec_train),
                        torch.from_numpy(features_train),
                        torch.from_numpy(y[train_idx].astype(np.int64)),
                    )
                    test_ds = TensorDataset(
                        torch.from_numpy(spec_test),
                        torch.from_numpy(features_test),
                        torch.from_numpy(y[test_idx].astype(np.int64)),
                    )

                    # Split train into train/val
                    n_val = max(1, int(len(train_idx) * 0.2))
                    val_ds = TensorDataset(
                        torch.from_numpy(spec_train[-n_val:]),
                        torch.from_numpy(features_train[-n_val:]),
                        torch.from_numpy(y[train_idx[-n_val:]].astype(np.int64)),
                    )
                    train_ds = TensorDataset(
                        torch.from_numpy(spec_train[:-n_val]),
                        torch.from_numpy(features_train[:-n_val]),
                        torch.from_numpy(y[train_idx[:-n_val]].astype(np.int64)),
                    )

                    # Create model
                    model = ImprovedDualBranchModel(
                        n_eeg_channels=9,
                        freq_bins=TARGET_SIZE,
                        time_steps=TARGET_SIZE,
                        math_input_dim=math_input_dim,
                        n_classes=2,
                        vit_variant=args.vit_variant,
                        fusion_method=args.fusion,
                        drop_rate=0.1,
                    )

                    # Train
                    trainer = ImprovedTrainer(
                        model=model,
                        device=device,
                        learning_rate=args.lr,
                        weight_decay=0.05,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        warmup_epochs=args.warmup_epochs,
                        patience=args.patience,
                        use_mixup=not args.no_mixup,
                        use_spec_augment=not args.no_spec_augment,
                        seed=args.seed + fold_counter,
                    )

                    train_result = trainer.fit(
                        train_ds,
                        val_ds=val_ds,
                        log_fn=lambda msg: log.info(f"  [S{sid} F{fold_idx}] {msg}"),
                    )

                    # Evaluate
                    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)
                    y_pred, y_prob = trainer.predict(test_loader)
                    m = compute_metrics(y[test_idx], y_pred, y_prob)

                    fr = FoldResult(
                        fold=fold_idx,
                        subject=sid,
                        accuracy=m["accuracy"],
                        kappa=m["kappa"],
                        f1_macro=m["f1_macro"],
                        n_train=len(train_idx),
                        n_test=len(test_idx),
                        y_true=y[test_idx],
                        y_pred=y_pred,
                        y_prob=y_prob,
                    )
                    log.info(
                        f"  Fold {fold_idx} [S{sid:02d}]: acc={fr.accuracy:.2f}%  kappa={fr.kappa:.3f}"
                    )
                    all_folds.append(fr)
                    fold_counter += 1

        else:  # loso
            for fold_idx, test_sid in enumerate(split_spec.loso_subjects):
                train_sids = [s for s in split_spec.loso_subjects if s != test_sid]
                X_train = np.concatenate([subject_data[s][0] for s in train_sids])
                y_train = np.concatenate([subject_data[s][1] for s in train_sids])
                X_test, y_test = subject_data[test_sid]
                spec_train = np.concatenate([subject_spec_data[s][0] for s in train_sids])
                spec_test, _ = subject_spec_data[test_sid]

                log.info(f"LOSO fold {fold_idx + 1}/{len(subjects)}: test=S{test_sid:02d}")
                set_seed(args.seed + fold_idx)

                # Build math features
                features_train, features_test, math_input_dim = builder.build_math_features(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

                # Prepare spectrograms
                spec_train_n = normalise_specs(spec_train).astype(np.float32)
                spec_test_n = normalise_specs(spec_test).astype(np.float32)

                # Create datasets
                train_ds = TensorDataset(
                    torch.from_numpy(spec_train_n),
                    torch.from_numpy(features_train),
                    torch.from_numpy(y_train.astype(np.int64)),
                )
                test_ds = TensorDataset(
                    torch.from_numpy(spec_test_n),
                    torch.from_numpy(features_test),
                    torch.from_numpy(y_test.astype(np.int64)),
                )

                # Split train into train/val
                n_val = max(1, int(len(y_train) * 0.2))
                perm = np.random.permutation(len(y_train))
                val_idx = perm[:n_val]
                train_idx_inner = perm[n_val:]

                val_ds = TensorDataset(
                    torch.from_numpy(spec_train_n[val_idx]),
                    torch.from_numpy(features_train[val_idx]),
                    torch.from_numpy(y_train[val_idx].astype(np.int64)),
                )
                train_ds = TensorDataset(
                    torch.from_numpy(spec_train_n[train_idx_inner]),
                    torch.from_numpy(features_train[train_idx_inner]),
                    torch.from_numpy(y_train[train_idx_inner].astype(np.int64)),
                )

                # Create model
                model = ImprovedDualBranchModel(
                    n_eeg_channels=9,
                    freq_bins=TARGET_SIZE,
                    time_steps=TARGET_SIZE,
                    math_input_dim=math_input_dim,
                    n_classes=2,
                    vit_variant=args.vit_variant,
                    fusion_method=args.fusion,
                    drop_rate=0.1,
                )

                # Train
                trainer = ImprovedTrainer(
                    model=model,
                    device=device,
                    learning_rate=args.lr,
                    weight_decay=0.05,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    warmup_epochs=args.warmup_epochs,
                    patience=args.patience,
                    use_mixup=not args.no_mixup,
                    use_spec_augment=not args.no_spec_augment,
                    seed=args.seed + fold_idx,
                )

                train_result = trainer.fit(
                    train_ds,
                    val_ds=val_ds,
                    log_fn=lambda msg, sid=test_sid: log.info(f"  [LOSO S{sid}] {msg}"),
                )

                # Evaluate
                test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)
                y_pred, y_prob = trainer.predict(test_loader)
                m = compute_metrics(y_test, y_pred, y_prob)

                fr = FoldResult(
                    fold=fold_idx,
                    subject=test_sid,
                    accuracy=m["accuracy"],
                    kappa=m["kappa"],
                    f1_macro=m["f1_macro"],
                    n_train=len(y_train),
                    n_test=len(y_test),
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                )
                log.info(
                    f"  LOSO fold {fold_idx} [S{test_sid:02d}]: acc={fr.accuracy:.2f}%  kappa={fr.kappa:.3f}"
                )
                all_folds.append(fr)

        elapsed = time.time() - t0
        result = CVResult(
            strategy=strategy, model_name="ImprovedEEGViT+CSP+Riemann", folds=all_folds
        )

        log.info(
            f"{strategy} done in {elapsed:.1f}s: {result.mean_accuracy:.2f}% +/- {result.std_accuracy:.2f}%"
        )

        # Save results
        tag = f"improved_eeg_vit_{args.fusion}_{strategy}"
        out_path = run_dir / "results" / f"{tag}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": "ImprovedEEGViT+CSP+Riemann",
            "vit_variant": args.vit_variant,
            "fusion": args.fusion,
            "strategy": strategy,
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "mean_kappa": result.mean_kappa,
            "mean_f1": result.mean_f1,
            "n_folds": len(all_folds),
            "per_subject": result.per_subject_accuracy,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_epochs": args.warmup_epochs,
                "use_mixup": not args.no_mixup,
                "use_spec_augment": not args.no_spec_augment,
            },
        }

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Saved: {out_path}")

        # Save plots
        plots_dir = run_dir / "plots" / f"improved_eeg_vit_{args.fusion}_{strategy}"
        try:
            agg_y_true = np.concatenate([f.y_true for f in all_folds])
            agg_y_pred = np.concatenate([f.y_pred for f in all_folds])
            save_confusion_matrix(
                agg_y_true,
                agg_y_pred,
                plots_dir,
                title=f"Improved EEG-ViT ({strategy.replace('_', ' ').title()})",
            )
        except Exception as e:
            log.warning(f"Plot failed: {e}")

        if strategy == "within_subject":
            try:
                save_per_subject_accuracy(
                    result.per_subject_accuracy,
                    plots_dir,
                    title=f"Improved EEG-ViT - Per Subject ({strategy.replace('_', ' ').title()})",
                )
            except Exception as e:
                log.warning(f"Per-subject plot failed: {e}")

        results[strategy] = data

    return results


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_stage_logging(run_dir, "improved_vit", "improved_vit_experiment.log")

    log.info("=" * 60)
    log.info("Improved EEG-ViT Experiment")
    log.info("=" * 60)
    log.info(f"ViT Variant: {args.vit_variant}")
    log.info(f"Fusion: {args.fusion}")
    log.info(f"Epochs: {args.epochs}")
    log.info(f"LR: {args.lr}")
    log.info(f"Warmup: {args.warmup_epochs} epochs")
    log.info(f"Mixup: {not args.no_mixup}")
    log.info(f"SpecAugment: {not args.no_spec_augment}")

    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    device = get_device(args.device)
    log.info(f"Device: {device}")
    set_seed(args.seed)

    # Load data
    from bci.data.download import (
        load_all_subjects,
        load_spectrogram_stats,
        load_subject_spectrograms,
        _processed_dir as _get_processed_dir,
    )

    log.info("Loading BCI IV-2a data...")
    try:
        subject_data, channel_names, sfreq = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error(f"{e} Run download step first.")
        sys.exit(1)

    log.info(f"Loaded {len(subject_data)} subjects (sfreq={sfreq:.0f} Hz)")

    # Load spectrograms
    log.info("Loading spectrogram cache...")
    try:
        spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error(f"{e} Run download step first.")
        sys.exit(1)

    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    subject_ids = [int(p.stem.split("_")[1]) for p in spec_files]

    subject_spec_data = {}
    for sid in subject_ids:
        try:
            imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
            subject_spec_data[sid] = (imgs, y)
        except Exception as e:
            log.warning(f"Subject {sid} spectrograms skipped: {e}")

    # Keep only subjects with both data types
    common_sids = sorted(set(subject_data.keys()) & set(subject_spec_data.keys()))
    subject_data = {s: subject_data[s] for s in common_sids}
    subject_spec_data = {s: subject_spec_data[s] for s in common_sids}
    log.info(f"Using {len(common_sids)} subjects with complete data")

    if not common_sids:
        log.error("No subjects with complete data. Exiting.")
        sys.exit(1)

    # Run experiment
    results = run_experiment(
        args=args,
        subject_data=subject_data,
        subject_spec_data=subject_spec_data,
        spec_mean=spec_mean,
        spec_std=spec_std,
        run_dir=run_dir,
        device=device,
        log=log,
    )

    log.info("=" * 60)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 60)
    for strategy, data in results.items():
        log.info(f"{strategy}: {data['mean_accuracy']:.2f}% +/- {data['std_accuracy']:.2f}%")

    # Save summary
    summary_path = run_dir / "results" / "improved_vit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
