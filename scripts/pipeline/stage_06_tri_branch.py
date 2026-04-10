"""Tri-branch adaptive fusion: ViT + CSP + Riemannian (within-subject + LOSO)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bci.data.augmentation import SpectrogramAugmenter
from bci.data.download import load_all_subjects, load_spectrogram_stats, load_subject_spectrograms
from bci.data.download import _processed_dir as _get_processed_dir
from bci.data.dual_branch_builder import DualBranchFoldBuilder
from bci.models.tri_branch import TriBranchAdaptiveModel
from bci.training.cross_validation import CVResult, FoldResult
from bci.training.evaluation import compute_metrics
from bci.training.splits import get_or_create_splits
from bci.training.trainer import Trainer
from bci.utils.config import AugmentationConfig, ModelConfig
from bci.utils.logging import setup_stage_logging
from bci.utils.seed import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tri-branch adaptive fusion (ViT + CSP + Riemannian).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--no-mixup", action="store_true", help="Disable mixup")
    p.add_argument("--no-spec-augment", action="store_true", help="Disable SpecAugment")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to PhysioNet-pretrained checkpoint (default: <run-dir>/checkpoints/vit_pretrained_physionet_eeg_vit.pt)",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["within_subject", "loso"],
        choices=["within_subject", "loso"],
    )
    return p.parse_args()


def _csp_dim_from_builder(builder: DualBranchFoldBuilder) -> int:
    n_bands = 6
    if builder.csp_k_best is not None:
        return int(builder.csp_k_best)
    return int(n_bands * 2 * builder.csp_n_components)


def run_strategy(
    strategy: str,
    run_dir: Path,
    subject_data: dict,
    subject_spec_data: dict,
    spec_mean: np.ndarray,
    spec_std: np.ndarray,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: str,
    log,
) -> Path:
    out_path = (
        run_dir
        / "results"
        / f"real_tri_branch_adaptive_vit{'_loso' if strategy == 'loso' else ''}.json"
    )
    if out_path.exists():
        log.info("Already exists: %s – skipping.", out_path)
        return out_path

    target_img_size = 64
    _device = torch.device(device)

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_k_best=12,
        riemann_estimator="oas",
        riemann_metric="riemann",
        sfreq=128.0,
    )
    csp_dim = _csp_dim_from_builder(builder)

    split_spec = get_or_create_splits(
        run_dir=run_dir,
        dataset="bci_iv2a",
        subject_data=subject_data,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    def normalise_specs(imgs: np.ndarray) -> np.ndarray:
        if imgs.shape[-1] != target_img_size:
            t = torch.from_numpy(imgs.astype(np.float32))
            t = torch.nn.functional.interpolate(
                t, size=(target_img_size, target_img_size), mode="bilinear", align_corners=False
            )
            imgs = t.numpy()
        return (imgs - spec_mean[None, :, None, None]) / spec_std[None, :, None, None]

    def build_model(math_input_dim: int) -> TriBranchAdaptiveModel:
        cfg = ModelConfig(
            vit_model_name="eeg_vit_tiny_patch8_64",
            vit_pretrained=True,
            vit_drop_rate=0.1,
            in_chans=9,
            n_classes=2,
        )
        model = TriBranchAdaptiveModel(
            math_input_dim=math_input_dim,
            csp_dim=csp_dim,
            config=cfg,
            img_size=target_img_size,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backbone_state = {
            k: v
            for k, v in ckpt.items()
            if not (k.startswith("head") or k.startswith("classifier"))
        }
        model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
        model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    def make_forward(model, seed_local: int):
        aug = SpectrogramAugmenter(
            AugmentationConfig(
                apply_freq_mask=True,
                freq_mask_max_width=8,
                apply_time_mask=True,
                time_mask_max_width=16,
                apply_mixup=False,
            ),
            seed=seed_local,
        )
        rng = np.random.default_rng(seed_local)

        def fwd(batch, _m=model):
            imgs, feats, labels = batch
            imgs = imgs.to(_device, non_blocking=True)
            feats = feats.to(_device, non_blocking=True)
            labels = labels.to(_device, non_blocking=True)
            if _m.training and not args.no_spec_augment:
                imgs_np = imgs.detach().cpu().numpy()
                imgs = torch.from_numpy(aug(imgs_np, training=True)).to(_device, non_blocking=True)
            if _m.training and not args.no_mixup and imgs.shape[0] > 1 and rng.random() < 0.5:
                lam = float(rng.beta(0.4, 0.4))
                perm = torch.randperm(imgs.shape[0], device=_device)
                imgs = lam * imgs + (1.0 - lam) * imgs[perm]
                feats = lam * feats + (1.0 - lam) * feats[perm]
                logits = _m(imgs, feats)
                return logits, (labels, labels[perm], lam)
            return _m(imgs, feats), labels

        return fwd

    all_folds: list[FoldResult] = []
    t0 = time.time()

    if strategy == "within_subject":
        fold_counter = 0
        for sid in sorted(subject_data.keys()):
            X, y = subject_data[sid]
            spec, _ = subject_spec_data[sid]
            for fold_idx, fold in enumerate(split_spec.within_subject.get(sid, [])):
                train_idx = np.array(fold["train_idx"], dtype=int)
                test_idx = np.array(fold["test_idx"], dtype=int)
                set_seed(args.seed + fold_counter)
                ft_tr, ft_te, math_input_dim = builder.build_math_features(
                    X[train_idx], y[train_idx], X[test_idx], y[test_idx], cache_path=None
                )
                tr_ds = TensorDataset(
                    torch.from_numpy(normalise_specs(spec[train_idx]).astype(np.float32)),
                    torch.from_numpy(ft_tr),
                    torch.from_numpy(y[train_idx].astype(np.int64)),
                )
                te_ds = TensorDataset(
                    torch.from_numpy(normalise_specs(spec[test_idx]).astype(np.float32)),
                    torch.from_numpy(ft_te),
                    torch.from_numpy(y[test_idx].astype(np.int64)),
                )

                model = build_model(math_input_dim)
                fwd = make_forward(model, args.seed + fold_counter)
                trainer = Trainer(
                    model=model,
                    device=device,
                    learning_rate=3e-4,
                    weight_decay=0.05,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    warmup_epochs=10,
                    patience=20,
                    label_smoothing=0.1,
                    val_fraction=0.2,
                    seed=args.seed,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=args.persistent_workers,
                    use_amp=args.amp,
                    compile_model=args.compile,
                    gradient_accumulation_steps=args.accumulation_steps,
                )
                trainer.fit(tr_ds, forward_fn=fwd, model_tag=f"tri_adapt_within_f{fold_counter}")

                te_loader = DataLoader(
                    te_ds,
                    batch_size=args.batch_size * 2,
                    shuffle=False,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                    persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
                    pin_memory=_device.type == "cuda",
                )
                y_pred, y_prob = trainer.predict(te_loader, forward_fn=fwd)
                m = compute_metrics(y[test_idx], y_pred, y_prob)
                all_folds.append(
                    FoldResult(
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
                )
                fold_counter += 1
    else:
        loso_subjects = split_spec.loso_subjects
        for fold_idx, test_sid in enumerate(loso_subjects):
            train_sids = [s for s in loso_subjects if s != test_sid]
            X_train = np.concatenate([subject_data[s][0] for s in train_sids])
            y_train = np.concatenate([subject_data[s][1] for s in train_sids])
            X_test, y_test = subject_data[test_sid]
            spec_train = np.concatenate([subject_spec_data[s][0] for s in train_sids])
            spec_test, _ = subject_spec_data[test_sid]

            set_seed(args.seed + fold_idx)
            ft_tr, ft_te, math_input_dim = builder.build_math_features(
                X_train, y_train, X_test, y_test, cache_path=None
            )
            tr_ds = TensorDataset(
                torch.from_numpy(normalise_specs(spec_train).astype(np.float32)),
                torch.from_numpy(ft_tr),
                torch.from_numpy(y_train.astype(np.int64)),
            )
            te_ds = TensorDataset(
                torch.from_numpy(normalise_specs(spec_test).astype(np.float32)),
                torch.from_numpy(ft_te),
                torch.from_numpy(y_test.astype(np.int64)),
            )

            model = build_model(math_input_dim)
            fwd = make_forward(model, args.seed + fold_idx)
            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=3e-4,
                weight_decay=0.05,
                epochs=args.epochs,
                batch_size=args.batch_size,
                warmup_epochs=10,
                patience=20,
                label_smoothing=0.1,
                val_fraction=0.2,
                seed=args.seed,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                use_amp=args.amp,
                compile_model=args.compile,
                gradient_accumulation_steps=args.accumulation_steps,
            )
            trainer.fit(tr_ds, forward_fn=fwd, model_tag=f"tri_adapt_loso_f{fold_idx}")

            te_loader = DataLoader(
                te_ds,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
                pin_memory=_device.type == "cuda",
            )
            y_pred, y_prob = trainer.predict(te_loader, forward_fn=fwd)
            m = compute_metrics(y_test, y_pred, y_prob)
            all_folds.append(
                FoldResult(
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
            )

    result = CVResult(strategy=strategy, model_name="TriBranchAdaptive", folds=all_folds)
    elapsed = time.time() - t0
    log.info(
        "Tri-branch %s done in %.1fs: %.2f%% ± %.2f%%",
        strategy,
        elapsed,
        result.mean_accuracy,
        result.std_accuracy,
    )

    data: dict
    if strategy == "within_subject":
        data = {
            "model": "TriBranchAdaptive-VIT+CSP+Riemann",
            "backbone": "eeg_vit_tiny_patch8_64",
            "fusion": "adaptive",
            "strategy": "within_subject",
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "mean_kappa": result.mean_kappa,
            "mean_f1": result.mean_f1,
            "n_folds": len(all_folds),
            "per_subject": result.per_subject_accuracy,
        }
    else:
        data = {
            "model": "TriBranchAdaptive-VIT+CSP+Riemann",
            "backbone": "eeg_vit_tiny_patch8_64",
            "fusion": "adaptive",
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

        key = f"tri_branch_adaptive_{strategy}"
        outputs = {key: str(out_path)}
        update_results_index(run_dir, "tri_branch", outputs)
        write_manifest(run_dir, "tri_branch", outputs, meta={"strategy": strategy})
    except Exception as e:
        log.warning("Failed to update results index: %s", e)

    log.info("Saved: %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_stage_logging(run_dir, "tri_branch", "tri_branch_adaptive_vit.log")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_eeg_vit.pt"
    )
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    device = str(get_device(args.device))
    set_seed(args.seed)
    log.info("Device: %s", device)

    try:
        subject_data, _, _ = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run the download step first.", e)
        sys.exit(1)

    try:
        spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run the download step first.", e)
        sys.exit(1)

    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    subject_spec_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sf in spec_files:
        sid = int(sf.stem.split("_")[1])
        try:
            imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
            subject_spec_data[sid] = (imgs, y)
        except Exception as e:
            log.warning("Subject %d spectrograms skipped: %s", sid, e)

    common = sorted(set(subject_data.keys()) & set(subject_spec_data.keys()))
    subject_data = {s: subject_data[s] for s in common}
    subject_spec_data = {s: subject_spec_data[s] for s in common}
    if not common:
        log.error("No subjects with complete data. Exiting.")
        sys.exit(1)

    if not checkpoint_path.exists():
        log.error("Checkpoint not found at %s. Run pretraining first.", checkpoint_path)
        sys.exit(1)

    for strategy in args.strategies:
        run_strategy(
            strategy=strategy,
            run_dir=run_dir,
            subject_data=subject_data,
            subject_spec_data=subject_spec_data,
            spec_mean=spec_mean,
            spec_std=spec_std,
            checkpoint_path=checkpoint_path,
            args=args,
            device=device,
            log=log,
        )

    log.info("Tri-branch adaptive stage complete.")


if __name__ == "__main__":
    main()
