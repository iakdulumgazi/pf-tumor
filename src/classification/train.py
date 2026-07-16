"""Sınıflandırıcı eğitimi — 5-fold hasta-bazlı CV, early stopping, metrikler.

Yerelde küçük veriyle (veya sentetik) test edilebilir; Colab'da GPU ile gerçek
eğitim için notebooks/train_classifier.ipynb bunu çağırır.

    python -m src.classification.train --roi-dir data/roi --labels labels.csv \
        --out-dir runs/exp1 --folds 5 --epochs 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from .dataset import (
    ROIDataset, Sample, class_weights, read_labels, split_patient_level,
    train_transforms, val_transforms,
)
from .model import CLASSES, build_model, count_params


def pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, preds, ys = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(1))
        ys.append(y.numpy())
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)

    metrics = {
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "confusion": confusion_matrix(ys, preds, labels=list(range(len(CLASSES)))).tolist(),
    }
    # AUC (one-vs-rest) — yalnızca >=2 sınıf mevcutsa
    try:
        if len(np.unique(ys)) == len(CLASSES):
            metrics["auc_ovr"] = float(
                roc_auc_score(ys, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        pass
    return metrics


def train_one_fold(
    train_samples: list[Sample], val_samples: list[Sample],
    out_dir: Path, fold: int, epochs: int, lr: float, batch_size: int,
    device: torch.device, patience: int, dropout: float, pretrained: str | None,
) -> dict:
    tr_ds = ROIDataset(train_samples, train_transforms())
    va_ds = ROIDataset(val_samples, val_transforms())
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(len(CLASSES), in_channels=4, dropout=dropout,
                        pretrained_path=pretrained).to(device)
    w = class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_f1, best_epoch, no_improve = -1.0, -1, 0
    best_path = out_dir / f"fold{fold}_best.pth"
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in tr_ld:
            optim.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optim.step()
            losses.append(loss.item())
        sched.step()

        val = evaluate(model, va_ld, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
        print(f"  [fold {fold}] epoch {epoch:3d} loss={np.mean(losses):.3f} "
              f"val_acc={val['accuracy']:.3f} macroF1={val['macro_f1']:.3f}")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, no_improve = val["macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "metrics": val, "classes": CLASSES}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [fold {fold}] early stopping (epoch {epoch}, en iyi {best_epoch})")
                break

    return {"fold": fold, "best_epoch": best_epoch, "best_macro_f1": best_f1,
            "best_ckpt": str(best_path), "history": history}


def run(
    roi_dir: Path, labels_csv: Path, out_dir: Path, n_folds: int = 5,
    epochs: int = 100, lr: float = 1e-4, batch_size: int = 4,
    patience: int = 15, dropout: float = 0.2, seed: int = 42,
    device: str | None = None, pretrained: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    samples = read_labels(labels_csv, roi_dir)
    dist = np.bincount([s.label for s in samples], minlength=len(CLASSES))
    n_patients = len({s.patient_id for s in samples})
    print(f"{len(samples)} örnek, {n_patients} hasta | "
          f"dağılım: {dict(zip(CLASSES, dist.tolist()))} | device={dev}")
    print(f"model parametreleri: {count_params(build_model()):,}")

    folds = split_patient_level(samples, n_folds, seed)
    results = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr = [samples[i] for i in tr_idx]
        va = [samples[i] for i in va_idx]
        # leakage kontrolü: hasta kesişimi olmamalı
        assert not ({s.patient_id for s in tr} & {s.patient_id for s in va}), \
            "DATA LEAKAGE: hasta hem train hem val'de!"
        print(f"\n=== Fold {fold}: {len(tr)} train / {len(va)} val ===")
        results.append(train_one_fold(tr, va, out_dir, fold, epochs, lr,
                                      batch_size, dev, patience, dropout, pretrained))

    cv_f1 = [r["best_macro_f1"] for r in results]
    summary = {
        "n_samples": len(samples), "n_patients": n_patients,
        "class_distribution": dict(zip(CLASSES, dist.tolist())),
        "cv_macro_f1_mean": float(np.mean(cv_f1)),
        "cv_macro_f1_std": float(np.std(cv_f1)),
        "folds": results,
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== CV macro-F1: {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f} ===")
    print(f"özet: {out_dir / 'cv_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sınıflandırıcı eğitimi (5-fold hasta-bazlı CV)")
    p.add_argument("--roi-dir", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--pretrained", default=None, help="Ön-eğitilmiş ağırlık (.pth)")
    args = p.parse_args(argv)
    run(args.roi_dir, args.labels, args.out_dir, args.folds, args.epochs,
        args.lr, args.batch_size, args.patience, args.dropout,
        device=args.device, pretrained=args.pretrained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
