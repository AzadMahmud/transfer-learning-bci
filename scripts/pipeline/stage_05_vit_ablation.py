"""Clean LOSO ablation for ViT invariance components.

Runs exactly five models:
1) baseline_vit
2) vit_plus_cov_token
3) vit_plus_grl
4) vit_plus_cov_grl
5) full_cov_grl_curriculum

Outputs:
  - results/vit_ablation_loso.json
  - results/vit_ablation_stats.json
  - results/vit_ablation_<model>.json
  - figures/ablation_embeddings_<model>_class.png
  - figures/ablation_embeddings_<model>_subject.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class AblationViT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        img_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        use_cov_token: bool,
        n_subjects: int,
        use_grl: bool,
        grl_lambda: float,
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
        self.task_head = nn.Linear(self.vit.feature_dim, config.n_classes)
        self.use_grl = bool(use_grl)
        self.grl_lambda = float(grl_lambda)
        self.subject_head: nn.Module | None = None
        if self.use_grl and n_subjects > 1:
            self.subject_head = nn.Sequential(
                nn.Linear(self.vit.feature_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_subjects),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vit(x)
        return self.task_head(feat)

    def forward_with_subject(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        feat = self.vit(x)
        task = self.task_head(feat)
        subj = None
        if self.subject_head is not None:
            rev = _GradReverse.apply(feat, self.grl_lambda)
            subj = self.subject_head(rev)
        return task, subj, feat

    def freeze_backbone(self, unfreeze_last_n_blocks: int) -> None:
        self.vit.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    def unfreeze_all(self) -> None:
        self.vit.unfreeze_all()
        for p in self.task_head.parameters():
            p.requires_grad = True
        if self.subject_head is not None:
            for p in self.subject_head.parameters():
                p.requires_grad = True

    def get_backbone_params(self) -> list[nn.Parameter]:
        return self.vit.get_backbone_params()

    def get_head_params(self) -> list[nn.Parameter]:
        params = list(self.task_head.parameters())
        if self.subject_head is not None:
            params.extend(list(self.subject_head.parameters()))
        return params

    def get_layerwise_param_groups(self) -> list[tuple[str, list[nn.Parameter]]]:
        groups = self.vit.get_layerwise_param_groups()
        groups.append(("task_head", list(self.task_head.parameters())))
        if self.subject_head is not None:
            groups.append(("subject_head", list(self.subject_head.parameters())))
        return groups


@dataclass
class AblationConfig:
    name: str
    use_cov_token: bool
    use_grl: bool
    curriculum: bool


def _entropy_from_probs(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-8, 1.0)
    return float((-(p * np.log(p)).sum(axis=1)).mean())


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "entropy": _entropy_from_probs(y_prob),
    }


def _wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import wilcoxon

        diffs = np.asarray(a) - np.asarray(b)
        diffs = diffs[np.abs(diffs) > 1e-12]
        if len(diffs) < 2:
            return float("nan"), float("nan")
        stat, p = wilcoxon(diffs)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _normalise(imgs: np.ndarray, mean: np.ndarray, std: np.ndarray, img_size: int) -> np.ndarray:
    if imgs.shape[-1] != img_size:
        t = torch.from_numpy(imgs.astype(np.float32))
        t = torch.nn.functional.interpolate(
            t, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
        imgs = t.numpy()
    return (imgs - mean[None, :, None, None]) / std[None, :, None, None]


def _plot_embeddings(
    out_dir: Path,
    model_name: str,
    feats: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    out_dir.mkdir(parents=True, exist_ok=True)
    z = PCA(n_components=2, random_state=42).fit_transform(feats)

    plt.figure(figsize=(7, 6))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(z[m, 0], z[m, 1], s=10, alpha=0.65, label=f"class {int(c)}")
    plt.title(f"{model_name} embeddings (PCA) - by class")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"ablation_embeddings_{model_name}_class.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 6))
    for s in np.unique(subjects):
        m = subjects == s
        plt.scatter(z[m, 0], z[m, 1], s=8, alpha=0.6, label=f"S{int(s):02d}")
    plt.title(f"{model_name} embeddings (PCA) - by subject")
    plt.legend(loc="best", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"ablation_embeddings_{model_name}_subject.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean 5-model ViT LOSO ablation")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--entropy-reg", type=float, default=0.01)
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--domain-loss-weight", type=float, default=0.2)
    p.add_argument("--grl-lambda", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--vit-embed-dim", type=int, default=96)
    p.add_argument("--vit-depth", type=int, default=4)
    p.add_argument("--vit-num-heads", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--unfreeze-last-n", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_stage_logging(run_dir, "vit_ablation", "vit_ablation.log")
    device = str(get_device(args.device))
    set_seed(args.seed)
    log.info("Device: %s", device)

    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    subject_spec_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sf in spec_files:
        sid = int(sf.stem.split("_")[1])
        imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
        subject_spec_data[sid] = (imgs, y)
    sids = sorted(subject_spec_data.keys())
    if not sids:
        raise RuntimeError("No subject spectrograms found")

    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_eeg_vit.pt"
    )

    variants = [
        AblationConfig("baseline_vit", use_cov_token=False, use_grl=False, curriculum=False),
        AblationConfig("vit_plus_cov_token", use_cov_token=True, use_grl=False, curriculum=False),
        AblationConfig("vit_plus_grl", use_cov_token=False, use_grl=True, curriculum=False),
        AblationConfig("vit_plus_cov_grl", use_cov_token=True, use_grl=True, curriculum=False),
        AblationConfig(
            "full_cov_grl_curriculum", use_cov_token=True, use_grl=True, curriculum=True
        ),
    ]

    all_results: dict[str, dict] = {}
    figures_dir = run_dir / "figures"
    for v in variants:
        out_file = run_dir / "results" / f"vit_ablation_{v.name}.json"
        if out_file.exists() and not args.force:
            with open(out_file) as f:
                all_results[v.name] = json.load(f)
            log.info("Skip existing: %s", out_file)
            continue

        fold_rows: list[dict] = []
        emb_feats: list[np.ndarray] = []
        emb_labels: list[np.ndarray] = []
        emb_subjects: list[np.ndarray] = []

        for fold_idx, test_sid in enumerate(sids):
            train_sids = [s for s in sids if s != test_sid]
            val_sid = train_sids[fold_idx % len(train_sids)]
            fit_sids = [s for s in train_sids if s != val_sid]

            cfg = ModelConfig(
                vit_model_name="eeg_vit_tiny_patch8_64",
                vit_pretrained=not args.no_pretrained,
                vit_drop_rate=0.0,
                in_chans=9,
                n_classes=2,
            )
            model = AblationViT(
                config=cfg,
                img_size=args.img_size,
                embed_dim=args.vit_embed_dim,
                depth=args.vit_depth,
                num_heads=args.vit_num_heads,
                use_cov_token=v.use_cov_token,
                n_subjects=len(sids),
                use_grl=v.use_grl,
                grl_lambda=args.grl_lambda,
            )
            if not args.no_pretrained:
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
                load_backbone_checkpoint(
                    model.vit.backbone,
                    checkpoint,
                    min_match_ratio=0.95,
                    strict_min_match=True,
                )
                model.freeze_backbone(args.unfreeze_last_n)
            else:
                model.unfreeze_all()

            X_train = np.concatenate([subject_spec_data[s][0] for s in fit_sids])
            y_train = np.concatenate([subject_spec_data[s][1] for s in fit_sids])
            sid_train = np.concatenate(
                [
                    np.full(subject_spec_data[s][1].shape[0], sid_to_idx[s], dtype=np.int64)
                    for s in fit_sids
                ]
            )
            X_val, y_val = subject_spec_data[val_sid]
            sid_val = np.full(y_val.shape[0], sid_to_idx[val_sid], dtype=np.int64)
            X_test, y_test = subject_spec_data[test_sid]
            sid_test = np.full(y_test.shape[0], sid_to_idx[test_sid], dtype=np.int64)

            X_train_n = _normalise(X_train, spec_mean, spec_std, args.img_size).astype(np.float32)
            X_val_n = _normalise(X_val, spec_mean, spec_std, args.img_size).astype(np.float32)
            X_test_n = _normalise(X_test, spec_mean, spec_std, args.img_size).astype(np.float32)

            tr_ds_all = TensorDataset(
                torch.from_numpy(X_train_n),
                torch.from_numpy(y_train.astype(np.int64)),
                torch.from_numpy(sid_train.astype(np.int64)),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_val_n),
                torch.from_numpy(y_val.astype(np.int64)),
                torch.from_numpy(sid_val.astype(np.int64)),
            )
            te_ds = TensorDataset(
                torch.from_numpy(X_test_n),
                torch.from_numpy(y_test.astype(np.int64)),
                torch.from_numpy(sid_test.astype(np.int64)),
            )

            rng = np.random.default_rng(args.seed + fold_idx)

            def fwd(batch):
                x, yb, sb = batch
                x = x.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                sb = sb.to(device, non_blocking=True)
                if model.training and x.shape[0] > 1 and rng.random() < args.mixup_prob:
                    lam = float(rng.beta(args.mixup_alpha, args.mixup_alpha))
                    perm = torch.randperm(x.shape[0], device=x.device)
                    x = lam * x + (1.0 - lam) * x[perm]
                    task_logits, subj_logits, _ = model.forward_with_subject(x)
                    aux_loss = None
                    if v.use_grl and subj_logits is not None:
                        ce = nn.CrossEntropyLoss()
                        aux_loss = args.domain_loss_weight * (
                            lam * ce(subj_logits, sb) + (1.0 - lam) * ce(subj_logits, sb[perm])
                        )
                    return task_logits, (yb, yb[perm], lam), aux_loss
                task_logits, subj_logits, _ = model.forward_with_subject(x)
                aux_loss = None
                if v.use_grl and subj_logits is not None:
                    ce = nn.CrossEntropyLoss()
                    aux_loss = args.domain_loss_weight * ce(subj_logits, sb)
                return task_logits, yb, aux_loss

            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                warmup_epochs=args.warmup_epochs,
                patience=args.patience,
                label_smoothing=args.label_smoothing,
                val_fraction=args.val_fraction,
                seed=args.seed + fold_idx,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                use_amp=args.amp,
                compile_model=args.compile,
                gradient_accumulation_steps=args.accumulation_steps,
                entropy_reg=args.entropy_reg,
                layer_lr_decay=0.8,
            )

            if v.curriculum and len(fit_sids) >= 4:
                curr_sets = [fit_sids[:2], fit_sids[:4], fit_sids]
                for i, sset in enumerate(curr_sets, start=1):
                    mask = np.isin(sid_train, [sid_to_idx[s] for s in sset])
                    ds_stage = TensorDataset(
                        torch.from_numpy(X_train_n[mask]),
                        torch.from_numpy(y_train[mask].astype(np.int64)),
                        torch.from_numpy(sid_train[mask].astype(np.int64)),
                    )
                    trainer.fit(
                        ds_stage,
                        forward_fn=fwd,
                        model_tag=f"ablation_{v.name}_f{fold_idx}_s{i}",
                        val_dataset=val_ds,
                    )
            else:
                trainer.fit(
                    tr_ds_all,
                    forward_fn=fwd,
                    model_tag=f"ablation_{v.name}_f{fold_idx}",
                    val_dataset=val_ds,
                )

            te_loader = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False)
            y_pred, y_prob = trainer.predict(te_loader, forward_fn=fwd)
            m = _compute_metrics(y_test, y_pred, y_prob)

            with torch.inference_mode():
                model.eval()
                feats_fold = []
                for xb, _, _ in te_loader:
                    xb = xb.to(device)
                    _, _, feat = model.forward_with_subject(xb)
                    feats_fold.append(feat.cpu().numpy())
                feats_fold_np = np.concatenate(feats_fold)

            emb_feats.append(feats_fold_np)
            emb_labels.append(y_test.astype(np.int64))
            emb_subjects.append(np.full_like(y_test.astype(np.int64), test_sid))

            pred_counts = np.bincount(y_pred.astype(np.int64), minlength=2).tolist()
            fold_rows.append(
                {
                    "subject": int(test_sid),
                    "accuracy": m["accuracy"],
                    "kappa": m["kappa"],
                    "f1": m["f1"],
                    "entropy": m["entropy"],
                    "pred_counts": pred_counts,
                }
            )
            log.info(
                "[%s] LOSO S%02d acc=%.2f entropy=%.4f pred=%s",
                v.name,
                test_sid,
                m["accuracy"],
                m["entropy"],
                pred_counts,
            )

        accs = [r["accuracy"] for r in fold_rows]
        ent = [r["entropy"] for r in fold_rows]
        per_subject = {str(r["subject"]): r["accuracy"] for r in fold_rows}
        per_subject_entropy = {str(r["subject"]): r["entropy"] for r in fold_rows}
        result = {
            "model": v.name,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "per_subject": per_subject,
            "per_subject_entropy": per_subject_entropy,
            "folds": fold_rows,
        }
        all_results[v.name] = result
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        _plot_embeddings(
            figures_dir,
            v.name,
            np.concatenate(emb_feats),
            np.concatenate(emb_labels),
            np.concatenate(emb_subjects),
        )

    # Stats: CSP vs baseline and CSP vs full
    csp_path = run_dir / "results" / "real_baseline_a_csp_lda_loso.json"
    csp = {}
    if csp_path.exists():
        with open(csp_path) as f:
            csp = json.load(f)
    csp_ps = csp.get("loso", csp).get("per_subject", {}) if csp else {}

    stats = {}
    if csp_ps:
        csp_vals = [csp_ps[str(s)] for s in sorted(int(k) for k in csp_ps.keys())]
        bsl = all_results.get("baseline_vit", {}).get("per_subject", {})
        full = all_results.get("full_cov_grl_curriculum", {}).get("per_subject", {})
        if bsl:
            bsl_vals = [bsl[str(s)] for s in sorted(int(k) for k in bsl.keys())]
            w, p = _wilcoxon(csp_vals, bsl_vals)
            stats["csp_vs_vit_baseline"] = {"wilcoxon_stat": w, "wilcoxon_p": p}
        if full:
            full_vals = [full[str(s)] for s in sorted(int(k) for k in full.keys())]
            w, p = _wilcoxon(csp_vals, full_vals)
            stats["csp_vs_full_model"] = {"wilcoxon_stat": w, "wilcoxon_p": p}

    # Final-model decision
    ranking = sorted(
        all_results.values(),
        key=lambda r: (-r["mean_accuracy"], r["std_accuracy"]),
    )
    final_choice = ranking[0]["model"] if ranking else None
    summary = {
        "models": all_results,
        "stats": stats,
        "final_model": final_choice,
        "ranking": [r["model"] for r in ranking],
    }
    with open(run_dir / "results" / "vit_ablation_loso.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "results" / "vit_ablation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Ablation complete. Final model: %s", final_choice)


if __name__ == "__main__":
    main()
