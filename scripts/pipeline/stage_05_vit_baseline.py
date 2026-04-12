"""EEG-ViT-only classifier on BCI IV-2a spectrograms.

Loads pre-cached 9-channel multichannel CWT spectrograms for BCI IV-2a
subjects and fine-tunes an EEG-ViT classifier.
Uses PhysioNet-pretrained backbone weights from pretraining.
Runs 5-fold within-subject CV and LOSO CV.

Prerequisite: Download and pretraining must have been run first.

Output::

    <run-dir>/results/real_baseline_c_vit.json
    <run-dir>/plots/vit_baseline/  (per-fold training curves + confusion matrices)

Usage::

    uv run python scripts/pipeline/stage_05_vit_baseline.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_05_vit_baseline.py \\
        --run-dir runs/my_run --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bci.data.augmentation import SpectrogramAugmenter
from bci.utils.logging import setup_stage_logging
from bci.utils.visualization import save_confusion_matrix, save_per_subject_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline C – ViT-only on cached spectrograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root of processed .npz cache (default: data/processed/)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.25,
        help="Backbone LR scale vs head LR (set <=0 to disable differential LR)",
    )
    p.add_argument(
        "--layer-lr-decay",
        type=float,
        default=None,
        help="Layer-wise LR decay (0-1); if set, overrides backbone-lr-scale",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--vit-drop-rate", type=float, default=0.15)
    p.add_argument("--vit-embed-dim", type=int, default=192)
    p.add_argument("--vit-depth", type=int, default=4)
    p.add_argument("--vit-num-heads", type=int, default=3)
    p.add_argument(
        "--use-covariance-token",
        action="store_true",
        help="Enable covariance token (must match pretraining for strict transfer)",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process)",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker when num_workers > 0",
    )
    p.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision on CUDA",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model (PyTorch 2.0+)",
    )
    p.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    p.add_argument("--no-mixup", action="store_true", help="Disable mixup")
    p.add_argument("--no-spec-augment", action="store_true", help="Disable SpecAugment")
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--use-temporal-encoder", action="store_true")
    p.add_argument("--debug-encoder-stats", action="store_true")
    p.add_argument("--scheduler", choices=["cosine", "onecycle"], default="onecycle")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--tta-runs", type=int, default=4)
    p.add_argument("--tta-noise-std", type=float, default=0.01)
    p.add_argument("--disable-differential-lr", action="store_true")
    p.add_argument("--subject-tune", action="store_true")
    p.add_argument("--tune-epochs", type=int, default=40)
    p.add_argument("--tune-lrs", nargs="+", type=float, default=[3e-4, 5e-4, 1e-3])
    p.add_argument("--tune-dropouts", nargs="+", type=float, default=[0.1, 0.2])
    p.add_argument("--freq-mask-max-width", type=int, default=8)
    p.add_argument("--time-mask-max-width", type=int, default=16)
    p.add_argument(
        "--class-weighting", action="store_true", help="Enable inverse-frequency class weights"
    )
    p.add_argument(
        "--entropy-reg", type=float, default=0.0, help="Entropy regularization coefficient"
    )
    p.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=4,
        help="Unfreeze only last N transformer blocks (0 = unfreeze all)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to PhysioNet-pretrained backbone checkpoint from pretraining "
        "(default: <run-dir>/checkpoints/vit_pretrained_physionet_eeg_vit.pt)",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained checkpoint (for debugging/sanity checks)",
    )
    p.add_argument(
        "--debug-sanity-check",
        action="store_true",
        help="Run label/input/gradient checks plus single-batch overfit before CV",
    )
    p.add_argument(
        "--debug-overfit-steps",
        type=int,
        default=300,
        help="Optimization steps for single-batch overfit sanity check",
    )
    p.add_argument(
        "--debug-feature-stats",
        action="store_true",
        help="Log feature mean/std during sanity checks",
    )
    p.add_argument("--full-subject-overfit-subject", type=int, default=None)
    p.add_argument("--full-subject-overfit-epochs", type=int, default=200)
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["within_subject", "loso"],
        choices=["within_subject", "loso"],
        help="CV strategies to run",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument(
        "--no-strict-ckpt-match",
        action="store_true",
        help="Allow low checkpoint-compatibility loads (not recommended)",
    )
    p.add_argument(
        "--min-ckpt-match-ratio",
        type=float,
        default=0.95,
        help="Minimum required matched-parameter ratio for checkpoint loading",
    )
    args, _ = p.parse_known_args()
    return args


def run_vit_cv(
    strategy: str,
    subject_spec_data: dict,
    spec_mean,
    spec_std,
    checkpoint_path: Path,
    run_dir: Path,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    log,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    use_amp: bool,
    compile_model: bool,
    accumulation_steps: int,
    use_mixup: bool,
    use_spec_augment: bool,
    mixup_alpha: float,
    mixup_prob: float,
    freq_mask_max_width: int,
    time_mask_max_width: int,
    learning_rate: float,
    scheduler_name: str,
    weight_decay: float,
    warmup_epochs: int,
    patience: int,
    label_smoothing: float,
    backbone_lr_scale: float,
    layer_lr_decay: float | None,
    val_fraction: float,
    vit_drop_rate: float,
    vit_embed_dim: int,
    vit_depth: int,
    vit_num_heads: int,
    use_covariance_token: bool,
    use_temporal_encoder: bool,
    class_weighting: bool,
    entropy_reg: float,
    unfreeze_last_n: int,
    use_pretrained: bool,
    strict_ckpt_match: bool,
    min_ckpt_match_ratio: float,
    debug_sanity_check: bool,
    debug_overfit_steps: int,
    debug_feature_stats: bool,
    debug_encoder_stats: bool,
    seeds: list[int],
    tta_runs: int,
    tta_noise_std: float,
    disable_differential_lr: bool,
    subject_tune: bool,
    tune_epochs: int,
    tune_lrs: list[float],
    tune_dropouts: list[float],
    full_subject_overfit_subject: int | None,
    full_subject_overfit_epochs: int,
    force: bool,
) -> Path:
    """Run ViT-only CV (within_subject or loso) and save results."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from bci.models.vit_branch import ViTBranch, load_backbone_checkpoint
    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.splits import get_or_create_splits
    from bci.training.trainer import Trainer
    from bci.utils.config import AugmentationConfig, ModelConfig
    from bci.utils.seed import set_seed

    MODEL_NAME = "CWT+EEG-ViT (PhysioNet pretrained)"
    BACKBONE = "eeg_vit_tiny_patch8_64"
    # Must match the img_size used in pretraining.
    TARGET_IMG_SIZE = 64

    tag_base = "real_baseline_c_vit"
    if strategy == "loso":
        tag_base += "_loso"
    out_path = run_dir / "results" / f"{tag_base}.json"
    plots_dir = run_dir / "plots" / f"vit_baseline_{strategy}"

    if out_path.exists() and not force:
        log.info("Already exists: %s – skipping.", out_path)
        return out_path

    if use_pretrained and use_covariance_token:
        log.warning(
            "Covariance token enabled while using pretrained checkpoint; this must match pretraining config."
        )
    if use_pretrained:
        log.info(
            "Checkpoint strict mode=%s min_match_ratio=%.2f",
            strict_ckpt_match,
            min_ckpt_match_ratio,
        )

    _device = torch.device(device)

    def build_model(vit_drop_override: float | None = None):
        cfg = ModelConfig(
            vit_model_name=BACKBONE,
            vit_pretrained=use_pretrained,
            vit_drop_rate=(vit_drop_override if vit_drop_override is not None else vit_drop_rate),
            in_chans=9,
            n_classes=2,
        )
        model = ViTBranch(
            config=cfg,
            as_feature_extractor=False,
            img_size=TARGET_IMG_SIZE,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            use_covariance_token=use_covariance_token,
            use_temporal_encoder=use_temporal_encoder,
        )
        if use_pretrained:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"PhysioNet checkpoint not found at {checkpoint_path}; run pretraining first."
                )
            load_backbone_checkpoint(
                model.backbone,
                checkpoint_path,
                min_match_ratio=min_ckpt_match_ratio,
                strict_min_match=strict_ckpt_match,
            )
        if not use_pretrained:
            model.unfreeze_all()
            log.info("No pretrained checkpoint: unfreezing full backbone for supervised training")
        elif unfreeze_last_n <= 0:
            model.unfreeze_all()
        else:
            model.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n)
        return model

    def select_subject_hparams(
        sid: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
    ) -> tuple[float, float]:
        if not subject_tune:
            return learning_rate, vit_drop_rate
        best_acc = -1.0
        best = (learning_rate, vit_drop_rate)
        stats_mean, stats_std = compute_channel_stats(train_x)
        ds = make_ds(train_x, train_y, stats_mean, stats_std)
        for lr_cand in tune_lrs:
            for drop_cand in tune_dropouts:
                set_seed(seed + sid + int(lr_cand * 1e6) + int(drop_cand * 100))
                model_c = build_model(vit_drop_override=float(drop_cand))

                def fwd_c(batch):
                    x, labels = batch
                    x = x.to(_device, non_blocking=True)
                    labels = labels.to(_device, non_blocking=True)
                    return model_c(x), labels

                trainer_c = Trainer(
                    model=model_c,
                    device=device,
                    learning_rate=float(lr_cand),
                    weight_decay=weight_decay,
                    epochs=tune_epochs,
                    batch_size=batch_size,
                    warmup_epochs=min(warmup_epochs, tune_epochs // 4),
                    patience=max(10, tune_epochs // 2),
                    label_smoothing=0.0,
                    val_fraction=val_fraction,
                    seed=seed,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    use_amp=use_amp,
                    compile_model=compile_model,
                    gradient_accumulation_steps=accumulation_steps,
                    backbone_lr_scale=(
                        None
                        if disable_differential_lr
                        else (backbone_lr_scale if backbone_lr_scale > 0 else None)
                    ),
                    layer_lr_decay=layer_lr_decay,
                    scheduler_name=scheduler_name,
                    class_weights=(class_weights_from_labels(train_y) if class_weighting else None),
                    entropy_reg=entropy_reg,
                )
                res = trainer_c.fit(ds, forward_fn=fwd_c, model_tag=f"vit_tune_s{sid:02d}")
                if res.best_val_accuracy > best_acc:
                    best_acc = res.best_val_accuracy
                    best = (float(lr_cand), float(drop_cand))
        log.info(
            "[S%02d tune] best_lr=%.2e best_drop=%.2f val=%.2f%%", sid, best[0], best[1], best_acc
        )
        return best

    def predict_with_tta(
        trainer: Trainer,
        dataset,
        model,
        sid: int,
        fold_idx: int,
    ):
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory=_device.type == "cuda",
        )
        if tta_runs <= 1:
            return trainer.predict(
                loader, forward_fn=lambda b: (model(b[0].to(_device)), b[1].to(_device))
            )

        probs_runs = []
        y_pred = None
        for tta_i in range(tta_runs):

            def fwd_tta(batch, _i=tta_i):
                x, labels = batch
                x = x.to(_device, non_blocking=True)
                labels = labels.to(_device, non_blocking=True)
                if _i > 0:
                    x = x + torch.randn_like(x) * float(tta_noise_std)
                return model(x), labels

            preds_i, probs_i = trainer.predict(loader, forward_fn=fwd_tta)
            probs_runs.append(probs_i)
            if y_pred is None:
                y_pred = preds_i

        probs_avg = np.mean(np.stack(probs_runs, axis=0), axis=0)
        y_pred = probs_avg.argmax(axis=1)
        return y_pred, probs_avg

    def run_sanity_checks(imgs: np.ndarray, y: np.ndarray, tag: str) -> None:
        """Run fast checks to detect collapsed predictions or broken gradients."""
        if not debug_sanity_check:
            return
        from torch.utils.data import DataLoader

        stats_mean, stats_std = compute_channel_stats(imgs)
        debug_ds = make_ds(imgs, y, stats_mean, stats_std)
        debug_loader = DataLoader(debug_ds, batch_size=min(batch_size, len(debug_ds)), shuffle=True)
        x0, y0 = next(iter(debug_loader))
        x0 = x0.to(_device, non_blocking=True)
        y0 = y0.to(_device, non_blocking=True)

        labels_np = y0.detach().cpu().numpy()
        label_hist = np.bincount(labels_np, minlength=2)
        log.info(
            "[SANITY %s] input mean=%.5f std=%.5f min=%.5f max=%.5f | labels=%s",
            tag,
            float(x0.mean().item()),
            float(x0.std().item()),
            float(x0.min().item()),
            float(x0.max().item()),
            label_hist.tolist(),
        )

        model_dbg = build_model().to(_device)
        model_dbg.train()
        trainable = [p for p in model_dbg.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=0.0)
        ce = torch.nn.CrossEntropyLoss()

        report_every = max(1, debug_overfit_steps // 6)
        for step in range(1, debug_overfit_steps + 1):
            opt.zero_grad(set_to_none=True)
            logits = model_dbg(x0)
            loss = ce(logits, y0)
            if entropy_reg > 0.0:
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                loss = loss - entropy_reg * entropy
            loss.backward()
            opt.step()

            if step == 1 or step % report_every == 0 or step == debug_overfit_steps:
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    acc = (pred == y0).float().mean().item() * 100.0
                    pred_hist = torch.bincount(pred, minlength=2).detach().cpu().tolist()
                    if debug_feature_stats:
                        feat = model_dbg.backbone.patch_embed(x0)
                        feat_mean = float(feat.mean().item())
                        feat_std = float(feat.std().item())
                    else:
                        feat_mean = 0.0
                        feat_std = 0.0
                    no_grad = sum(
                        1
                        for p in model_dbg.parameters()
                        if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)
                    )
                log.info(
                    "[SANITY %s] step=%d/%d loss=%.4f acc=%.2f%% pred_hist=%s no_grad=%d feat_mean=%.5f feat_std=%.5f",
                    tag,
                    step,
                    debug_overfit_steps,
                    float(loss.item()),
                    float(acc),
                    pred_hist,
                    no_grad,
                    feat_mean,
                    feat_std,
                )

        with torch.no_grad():
            final_pred = model_dbg(x0).argmax(dim=-1)
            final_acc = (final_pred == y0).float().mean().item() * 100.0
        if final_acc < 90.0:
            raise RuntimeError(
                f"Sanity overfit failed on {tag}: final single-batch acc={final_acc:.2f}%"
            )
        log.info("[SANITY %s] passed: single-batch acc=%.2f%%", tag, final_acc)

    def compute_channel_stats(imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = imgs.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
        std = imgs.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
        std = np.maximum(std, 1e-6)
        return mean, std

    def make_ds(imgs, y, stats_mean: np.ndarray, stats_std: np.ndarray):
        # Resize from on-disk 224×224 to TARGET_IMG_SIZE×TARGET_IMG_SIZE.
        # Reduces ViT forward-pass cost by ~12× with minimal accuracy impact.
        if imgs.shape[-1] != TARGET_IMG_SIZE:
            t = torch.from_numpy(imgs.astype(np.float32))
            t = torch.nn.functional.interpolate(
                t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode="bilinear", align_corners=False
            )
            imgs = t.numpy()
        # Normalise per-channel using fold-local training-set stats
        imgs_norm = (imgs - stats_mean[None, :, None, None]) / stats_std[None, :, None, None]
        return TensorDataset(
            torch.from_numpy(imgs_norm.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )

    def class_weights_from_labels(y_arr: np.ndarray) -> list[float]:
        counts = np.bincount(y_arr.astype(int), minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        return (inv / inv.sum() * len(inv)).tolist()

    def maybe_run_full_subject_overfit() -> bool:
        if full_subject_overfit_subject is None:
            return False
        sid = int(full_subject_overfit_subject)
        if sid not in subject_spec_data:
            raise ValueError(f"Subject S{sid:02d} not found")
        imgs, y = subject_spec_data[sid]
        stats_mean, stats_std = compute_channel_stats(imgs)
        ds = make_ds(imgs, y, stats_mean, stats_std)
        model = build_model()

        def fwd(batch):
            x, labels = batch
            x = x.to(_device, non_blocking=True)
            labels = labels.to(_device, non_blocking=True)
            return model(x), labels

        cw = class_weights_from_labels(y) if class_weighting else None
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=0.0,
            epochs=full_subject_overfit_epochs,
            batch_size=batch_size,
            warmup_epochs=0,
            patience=full_subject_overfit_epochs,
            label_smoothing=0.0,
            val_fraction=0.0,
            seed=seed,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            use_amp=use_amp,
            compile_model=compile_model,
            gradient_accumulation_steps=accumulation_steps,
            class_weights=cw,
            entropy_reg=entropy_reg,
        )
        res = trainer.fit(ds, forward_fn=fwd, model_tag=f"vit_overfit_s{sid:02d}", val_dataset=ds)
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False)
        y_pred, y_prob = trainer.predict(loader, forward_fn=fwd)
        m = compute_metrics(y, y_pred, y_prob)
        log.info(
            "[FULL-OVERFIT S%02d] acc=%.2f%% kappa=%.3f best_val=%.2f%% epoch=%d",
            sid,
            m["accuracy"],
            m["kappa"],
            res.best_val_accuracy,
            res.best_epoch,
        )
        out = run_dir / "results" / "real_vit_full_subject_overfit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "mode": "full_subject_overfit",
                    "subject": sid,
                    "accuracy": float(m["accuracy"]),
                    "kappa": float(m["kappa"]),
                    "best_val_accuracy": float(res.best_val_accuracy),
                    "best_epoch": int(res.best_epoch),
                },
                f,
                indent=2,
            )
        log.info("Saved: %s", out)
        return True

    t0 = time.time()
    all_folds: list[FoldResult] = []
    subjects = sorted(subject_spec_data.keys())
    if maybe_run_full_subject_overfit():
        return run_dir / "results" / "real_vit_full_subject_overfit.json"
    split_spec = get_or_create_splits(
        run_dir=run_dir,
        dataset="bci_iv2a",
        subject_data={
            sid: (subject_spec_data[sid][0], subject_spec_data[sid][1]) for sid in subjects
        },
        n_folds=n_folds,
        seed=seed,
    )

    if strategy == "within_subject":
        fold_counter = 0
        for sid in subjects:
            imgs, y = subject_spec_data[sid]
            if fold_counter == 0:
                run_sanity_checks(imgs, y, tag=f"within_S{sid:02d}")
            log.info("Subject %d (%d trials)...", sid, len(y))
            folds = split_spec.within_subject.get(sid, [])
            for fold_idx, fold in enumerate(folds):
                train_idx = np.array(fold["train_idx"], dtype=int)
                test_idx = np.array(fold["test_idx"], dtype=int)
                set_seed(seed + fold_counter)
                train_x = imgs[train_idx].astype(np.float32)
                test_x = imgs[test_idx].astype(np.float32)
                fold_mean, fold_std = compute_channel_stats(train_x)
                train_ds = make_ds(train_x, y[train_idx], fold_mean, fold_std)
                test_ds = make_ds(test_x, y[test_idx], fold_mean, fold_std)
                lr_sid, drop_sid = select_subject_hparams(sid, train_x, y[train_idx])
                per_seed_probs = []
                aug = SpectrogramAugmenter(
                    AugmentationConfig(
                        apply_freq_mask=True,
                        freq_mask_max_width=freq_mask_max_width,
                        apply_time_mask=True,
                        time_mask_max_width=time_mask_max_width,
                        apply_mixup=False,
                    ),
                    seed=seed + fold_counter,
                )

                for run_i, run_seed in enumerate(seeds):
                    set_seed(int(run_seed) + fold_counter)
                    model = build_model(vit_drop_override=drop_sid)

                    def fwd(batch):
                        x, labels = batch
                        x = x.to(_device, non_blocking=True)
                        labels = labels.to(_device, non_blocking=True)
                        if debug_encoder_stats and hasattr(model.backbone, "temporal_encoder"):
                            if model.backbone.temporal_encoder is not None:
                                with torch.no_grad():
                                    enc = model.backbone.temporal_encoder(
                                        x.reshape(
                                            x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3]
                                        )
                                    )
                                log.info(
                                    "[ENC within S%02d F%d R%d] mean=%.6f std=%.6f",
                                    sid,
                                    fold_idx,
                                    run_i,
                                    float(enc.mean().item()),
                                    float(enc.std().item()),
                                )
                        if model.training and use_spec_augment:
                            x_np = x.detach().cpu().numpy()
                            x = torch.from_numpy(aug(x_np, training=True)).to(
                                _device, non_blocking=True
                            )
                        return model(x), labels

                    trainer = Trainer(
                        model=model,
                        device=device,
                        learning_rate=lr_sid,
                        weight_decay=weight_decay,
                        epochs=epochs,
                        batch_size=batch_size,
                        warmup_epochs=warmup_epochs,
                        patience=patience,
                        label_smoothing=label_smoothing,
                        val_fraction=val_fraction,
                        seed=int(run_seed),
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers,
                        use_amp=use_amp,
                        compile_model=compile_model,
                        gradient_accumulation_steps=accumulation_steps,
                        backbone_lr_scale=(
                            None
                            if disable_differential_lr
                            else (backbone_lr_scale if backbone_lr_scale > 0 else None)
                        ),
                        layer_lr_decay=layer_lr_decay,
                        scheduler_name=scheduler_name,
                        class_weights=(
                            class_weights_from_labels(y[train_idx]) if class_weighting else None
                        ),
                        entropy_reg=entropy_reg,
                    )
                    trainer.fit(
                        train_ds, forward_fn=fwd, model_tag=f"vit_within_f{fold_counter}_s{run_i}"
                    )
                    y_pred_i, y_prob_i = predict_with_tta(trainer, test_ds, model, sid, fold_idx)
                    per_seed_probs.append(y_prob_i)

                y_prob = np.mean(np.stack(per_seed_probs, axis=0), axis=0)
                y_pred = y_prob.argmax(axis=1)
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
                    "  Fold %d [S%02d]: acc=%.2f%%  kappa=%.3f",
                    fold_idx,
                    sid,
                    fr.accuracy,
                    fr.kappa,
                )
                all_folds.append(fr)
                fold_counter += 1

    else:  # loso
        for fold_idx, test_sid in enumerate(split_spec.loso_subjects):
            train_sids = [s for s in split_spec.loso_subjects if s != test_sid]
            val_sid = train_sids[fold_idx % len(train_sids)]
            fit_sids = [s for s in train_sids if s != val_sid]
            train_imgs = np.concatenate([subject_spec_data[s][0] for s in fit_sids])
            train_y = np.concatenate([subject_spec_data[s][1] for s in fit_sids])
            val_imgs, val_y = subject_spec_data[val_sid]
            test_imgs, test_y = subject_spec_data[test_sid]
            if fold_idx == 0:
                run_sanity_checks(train_imgs, train_y, tag="loso_train")
            log.info(
                "LOSO fold %d/%d: test=S%02d, val=S%02d, train_subjects=%s",
                fold_idx + 1,
                len(subjects),
                test_sid,
                val_sid,
                [f"S{s:02d}" for s in fit_sids],
            )

            set_seed(seed + fold_idx)
            train_x = train_imgs.astype(np.float32)
            val_x = val_imgs.astype(np.float32)
            test_x = test_imgs.astype(np.float32)
            fold_mean, fold_std = compute_channel_stats(train_x)
            train_ds = make_ds(train_x, train_y, fold_mean, fold_std)
            val_ds = make_ds(val_x, val_y, fold_mean, fold_std)
            test_ds = make_ds(test_x, test_y, fold_mean, fold_std)
            model = build_model()
            aug = SpectrogramAugmenter(
                AugmentationConfig(
                    apply_freq_mask=True,
                    freq_mask_max_width=freq_mask_max_width,
                    apply_time_mask=True,
                    time_mask_max_width=time_mask_max_width,
                    apply_mixup=False,
                ),
                seed=seed + fold_idx,
            )
            rng = np.random.default_rng(seed + fold_idx)

            def fwd(batch):
                x, labels = batch
                x = x.to(_device, non_blocking=True)
                labels = labels.to(_device, non_blocking=True)
                if debug_encoder_stats and hasattr(model.backbone, "temporal_encoder"):
                    if model.backbone.temporal_encoder is not None:
                        with torch.no_grad():
                            enc = model.backbone.temporal_encoder(
                                x.reshape(x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3])
                            )
                        log.info(
                            "[ENC loso S%02d] mean=%.6f std=%.6f",
                            test_sid,
                            float(enc.mean().item()),
                            float(enc.std().item()),
                        )
                if model.training and use_spec_augment:
                    x_np = x.detach().cpu().numpy()
                    x = torch.from_numpy(aug(x_np, training=True)).to(_device, non_blocking=True)
                if model.training and use_mixup and x.shape[0] > 1 and rng.random() < mixup_prob:
                    lam = float(rng.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(x.shape[0], device=_device)
                    x_mix = lam * x + (1.0 - lam) * x[perm]
                    logits = model(x_mix)
                    return logits, (labels, labels[perm], lam)
                return model(x), labels

            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                warmup_epochs=warmup_epochs,
                patience=patience,
                label_smoothing=label_smoothing,
                val_fraction=val_fraction,
                seed=seed,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                use_amp=use_amp,
                compile_model=compile_model,
                gradient_accumulation_steps=accumulation_steps,
                backbone_lr_scale=backbone_lr_scale if backbone_lr_scale > 0 else None,
                layer_lr_decay=layer_lr_decay,
                class_weights=(class_weights_from_labels(train_y) if class_weighting else None),
                entropy_reg=entropy_reg,
            )
            train_result = trainer.fit(
                train_ds,
                forward_fn=fwd,
                model_tag=f"vit_loso_f{fold_idx}",
                val_dataset=val_ds,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                pin_memory=_device.type == "cuda",
            )
            y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
            m = compute_metrics(test_y, y_pred, y_prob)
            pred_hist = np.bincount(y_pred.astype(int), minlength=2)
            true_hist = np.bincount(test_y.astype(int), minlength=2)
            fr = FoldResult(
                fold=fold_idx,
                subject=test_sid,
                accuracy=m["accuracy"],
                kappa=m["kappa"],
                f1_macro=m["f1_macro"],
                n_train=len(train_y),
                n_test=len(test_y),
                y_true=test_y,
                y_pred=y_pred,
                y_prob=y_prob,
            )
            log.info(
                "  LOSO fold %d [S%02d]: acc=%.2f%%  kappa=%.3f  pred=%s true=%s",
                fold_idx,
                test_sid,
                fr.accuracy,
                fr.kappa,
                pred_hist.tolist(),
                true_hist.tolist(),
            )
            all_folds.append(fr)

    elapsed = time.time() - t0
    result = CVResult(strategy=strategy, model_name=MODEL_NAME, folds=all_folds)
    log.info(
        "%s done in %.1fs: %.2f%% ± %.2f%%",
        strategy,
        elapsed,
        result.mean_accuracy,
        result.std_accuracy,
    )

    # ── Summary plots ──────────────────────────────────────────────────────
    strategy_label = "Within-Subject CV" if strategy == "within_subject" else "LOSO CV"
    try:
        import numpy as _np

        agg_y_true = _np.concatenate([f.y_true for f in all_folds])
        agg_y_pred = _np.concatenate([f.y_pred for f in all_folds])
        save_confusion_matrix(
            agg_y_true,
            agg_y_pred,
            plots_dir,
            filename="confusion_matrix",
            title=f"ViT-Only Baseline ({strategy_label})",
        )
    except Exception as e:
        log.warning("Confusion matrix plot failed: %s", e)

    if strategy == "within_subject":
        try:
            save_per_subject_accuracy(
                result.per_subject_accuracy,
                plots_dir,
                filename="per_subject_accuracy",
                title=f"ViT-Only Baseline \u2013 Per-Subject Accuracy ({strategy_label})",
            )
        except Exception as e:
            log.warning("Per-subject plot failed: %s", e)

    # Build output dict keyed for compatibility with phase4_compile_results.py
    data: dict
    if strategy == "within_subject":
        data = {
            "model": MODEL_NAME,
            "backbone": BACKBONE,
            "within_subject": {
                "mean_accuracy": result.mean_accuracy,
                "std_accuracy": result.std_accuracy,
                "mean_kappa": result.mean_kappa,
                "mean_f1": result.mean_f1,
                "n_folds": len(all_folds),
                "per_subject": result.per_subject_accuracy,
            },
        }
    else:
        data = {
            "model": MODEL_NAME,
            "backbone": BACKBONE,
            "strategy": "loso",
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "mean_kappa": result.mean_kappa,
            "mean_f1": result.mean_f1,
            "n_folds": len(all_folds),
            "per_subject": result.per_subject_accuracy,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    try:
        from bci.utils.results_index import update_results_index, write_manifest

        outputs = {f"{strategy}": str(out_path)}
        update_results_index(run_dir, "vit_baseline", outputs)
        write_manifest(
            run_dir,
            "vit_baseline",
            outputs,
            meta={"strategy": strategy},
        )
    except Exception as e:
        log.warning("Failed to update results index: %s", e)
    log.info("Saved: %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_stage_logging(run_dir, "vit_baseline", "vit_baseline.log")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_eeg_vit.pt"
    )
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    import numpy as np

    from bci.data.download import load_subject_spectrograms
    from bci.data.download import _processed_dir as _get_processed_dir
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)
    set_seed(args.seed)

    # Fold-local train-set stats are computed inside CV; no global stats required.
    spec_mean = None
    spec_std = None

    # ── Discover available subjects ────────────────────────────────────────
    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    if not spec_files:
        log.error("No BCI IV-2a spectrogram files found in %s. Run the download step first.", pdir)
        sys.exit(1)
    subject_ids = [int(p.stem.split("_")[1]) for p in spec_files]
    log.info("Found %d BCI IV-2a subjects.", len(subject_ids))

    # ── Load all spectrogram data ──────────────────────────────────────────
    subject_spec_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in subject_ids:
        try:
            imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
            subject_spec_data[sid] = (imgs, y)
            log.info("  Subject %d: spectrograms=%s", sid, imgs.shape)
        except Exception as e:
            log.warning("  Subject %d skipped: %s", sid, e)

    if not subject_spec_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    if not args.no_pretrained and not checkpoint_path.exists():
        log.error("PhysioNet checkpoint not found at %s. Run pretraining first.", checkpoint_path)
        sys.exit(1)
    if args.no_pretrained:
        log.info(
            "Loaded %d subjects, running without pretrained checkpoint", len(subject_spec_data)
        )
    else:
        log.info("Loaded %d subjects, checkpoint=%s", len(subject_spec_data), checkpoint_path)
    log.info(
        "ViT config: embed_dim=%d depth=%d heads=%d cov_token=%s",
        args.vit_embed_dim,
        args.vit_depth,
        args.vit_num_heads,
        args.use_covariance_token,
    )

    for strategy in args.strategies:
        if strategy == "within_subject":
            log.info("Running ViT-only within-subject %d-fold CV...", args.n_folds)
        else:
            log.info("Running ViT-only LOSO CV...")
        run_vit_cv(
            strategy=strategy,
            subject_spec_data=subject_spec_data,
            spec_mean=spec_mean,
            spec_std=spec_std,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            n_folds=args.n_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=str(device),
            seed=args.seed,
            log=log,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            use_amp=args.amp,
            compile_model=args.compile,
            accumulation_steps=args.accumulation_steps,
            use_mixup=not args.no_mixup,
            use_spec_augment=not args.no_spec_augment,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            freq_mask_max_width=args.freq_mask_max_width,
            time_mask_max_width=args.time_mask_max_width,
            learning_rate=args.learning_rate,
            scheduler_name=args.scheduler,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            label_smoothing=args.label_smoothing,
            backbone_lr_scale=args.backbone_lr_scale,
            layer_lr_decay=args.layer_lr_decay,
            val_fraction=args.val_fraction,
            vit_drop_rate=args.vit_drop_rate,
            vit_embed_dim=args.vit_embed_dim,
            vit_depth=args.vit_depth,
            vit_num_heads=args.vit_num_heads,
            use_covariance_token=args.use_covariance_token,
            use_temporal_encoder=args.use_temporal_encoder,
            class_weighting=args.class_weighting,
            entropy_reg=args.entropy_reg,
            unfreeze_last_n=args.unfreeze_last_n,
            use_pretrained=not args.no_pretrained,
            strict_ckpt_match=not args.no_strict_ckpt_match,
            min_ckpt_match_ratio=args.min_ckpt_match_ratio,
            debug_sanity_check=args.debug_sanity_check,
            debug_overfit_steps=args.debug_overfit_steps,
            debug_feature_stats=args.debug_feature_stats,
            debug_encoder_stats=args.debug_encoder_stats,
            seeds=args.seeds,
            tta_runs=args.tta_runs,
            tta_noise_std=args.tta_noise_std,
            disable_differential_lr=args.disable_differential_lr,
            subject_tune=args.subject_tune,
            tune_epochs=args.tune_epochs,
            tune_lrs=args.tune_lrs,
            tune_dropouts=args.tune_dropouts,
            full_subject_overfit_subject=args.full_subject_overfit_subject,
            full_subject_overfit_epochs=args.full_subject_overfit_epochs,
            force=args.force,
        )

    log.info("ViT baseline complete.")


if __name__ == "__main__":
    main()
