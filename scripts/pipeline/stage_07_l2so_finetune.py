"""L2SO-style practical transfer: train on 7, fine-tune target subject, test target remainder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bci.data.download import load_spectrogram_stats, load_subject_spectrograms
from bci.data.download import _processed_dir as _get_processed_dir
from bci.models.vit_branch import ViTBranch
from bci.models.vit_branch import load_backbone_checkpoint
from bci.training.trainer import Trainer
from bci.utils.config import ModelConfig
from bci.utils.logging import setup_stage_logging
from bci.utils.seed import get_device, set_seed


class TransferViT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        img_size: int,
        use_cov_token: bool = True,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        self.vit = ViTBranch(
            config=config,
            as_feature_extractor=True,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            use_covariance_token=use_cov_token,
        )
        self.head = nn.Linear(self.vit.feature_dim, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        self.vit.freeze_backbone(unfreeze_last_n_blocks)

    def unfreeze_all(self) -> None:
        self.vit.unfreeze_all()
        for p in self.head.parameters():
            p.requires_grad = True

    def get_backbone_params(self):
        return self.vit.get_backbone_params()

    def get_head_params(self):
        return list(self.head.parameters())

    def get_layerwise_param_groups(self):
        groups = self.vit.get_layerwise_param_groups()
        groups.append(("task_head", list(self.head.parameters())))
        return groups


def _normalise(imgs: np.ndarray, mean: np.ndarray, std: np.ndarray, img_size: int) -> np.ndarray:
    if imgs.shape[-1] != img_size:
        t = torch.from_numpy(imgs.astype(np.float32))
        t = torch.nn.functional.interpolate(
            t, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
        imgs = t.numpy()
    return (imgs - mean[None, :, None, None]) / std[None, :, None, None]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    p = np.clip(y_prob, 1e-8, 1.0)
    ent = float((-(p * np.log(p)).sum(axis=1)).mean())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "entropy": ent,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2SO practical transfer fine-tuning")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--fine-tune-frac", type=float, default=0.2)
    p.add_argument("--pretrain-epochs", type=int, default=80)
    p.add_argument("--finetune-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--vit-embed-dim", type=int, default=192)
    p.add_argument("--vit-depth", type=int, default=4)
    p.add_argument("--vit-num-heads", type=int, default=3)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "results" / "real_l2so_finetune_vit.json"
    log = setup_stage_logging(run_dir, "l2so_finetune", "l2so_finetune.log")
    if out.exists() and not args.force:
        log.info("Already exists: %s", out)
        return

    set_seed(args.seed)
    device = str(get_device(args.device))
    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    subject_spec_data = {}
    for sf in sorted(pdir.glob("subject_[0-9]*_spectrograms.npz")):
        sid = int(sf.stem.split("_")[1])
        imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
        subject_spec_data[sid] = (imgs, y)

    sids = sorted(subject_spec_data.keys())
    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_eeg_vit.pt"
    )

    rows = []
    for target in sids:
        others = [s for s in sids if s != target]
        train7 = others[:7]

        X_train = np.concatenate([subject_spec_data[s][0] for s in train7])
        y_train = np.concatenate([subject_spec_data[s][1] for s in train7])
        X_t, y_t = subject_spec_data[target]

        rng = np.random.default_rng(args.seed + target)
        idx = np.arange(len(y_t))
        rng.shuffle(idx)
        n_ft = max(1, int(len(idx) * args.fine_tune_frac))
        ft_idx = idx[:n_ft]
        te_idx = idx[n_ft:]

        X_train_n = _normalise(X_train, spec_mean, spec_std, args.img_size).astype(np.float32)
        X_ft_n = _normalise(X_t[ft_idx], spec_mean, spec_std, args.img_size).astype(np.float32)
        X_te_n = _normalise(X_t[te_idx], spec_mean, spec_std, args.img_size).astype(np.float32)

        tr_ds = TensorDataset(
            torch.from_numpy(X_train_n), torch.from_numpy(y_train.astype(np.int64))
        )
        ft_ds = TensorDataset(
            torch.from_numpy(X_ft_n), torch.from_numpy(y_t[ft_idx].astype(np.int64))
        )
        te_ds = TensorDataset(
            torch.from_numpy(X_te_n), torch.from_numpy(y_t[te_idx].astype(np.int64))
        )

        cfg = ModelConfig(vit_drop_rate=0.1, in_chans=9, n_classes=2)
        model = TransferViT(
            cfg,
            args.img_size,
            use_cov_token=True,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
        )
        if ckpt.exists():
            load_backbone_checkpoint(
                model.vit.backbone,
                ckpt,
                min_match_ratio=0.95,
                strict_min_match=True,
            )

        def fwd(batch):
            x, yb = batch
            x = x.to(device)
            yb = yb.to(device)
            return model(x), yb

        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            warmup_epochs=10,
            patience=20,
            label_smoothing=0.1,
            val_fraction=0.2,
            seed=args.seed,
            entropy_reg=0.01,
            layer_lr_decay=0.8,
        )
        trainer.fit(tr_ds, forward_fn=fwd, model_tag=f"l2so_pretrain_s{target:02d}")

        model.unfreeze_all()
        trainer_ft = Trainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate * 0.5,
            weight_decay=args.weight_decay,
            epochs=args.finetune_epochs,
            batch_size=min(args.batch_size, len(ft_ds)),
            warmup_epochs=2,
            patience=max(8, args.finetune_epochs // 3),
            label_smoothing=0.05,
            val_fraction=0.2,
            seed=args.seed + 100,
            entropy_reg=0.005,
        )
        trainer_ft.fit(ft_ds, forward_fn=fwd, model_tag=f"l2so_finetune_s{target:02d}")

        te_loader = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False)
        y_pred, y_prob = trainer_ft.predict(te_loader, forward_fn=fwd)
        m = _metrics(y_t[te_idx], y_pred, y_prob)
        rows.append(
            {
                "subject": target,
                "train_subjects": train7,
                "fine_tune_frac": args.fine_tune_frac,
                "n_finetune": int(len(ft_idx)),
                "n_test": int(len(te_idx)),
                **m,
            }
        )
        log.info("L2SO target S%02d: acc=%.2f%% kappa=%.3f", target, m["accuracy"], m["kappa"])

    accs = [r["accuracy"] for r in rows]
    data = {
        "model": "L2SO+FineTune ViT(cov)",
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "per_subject": {str(r["subject"]): r["accuracy"] for r in rows},
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved: %s", out)


if __name__ == "__main__":
    main()
