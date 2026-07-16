"""Sınıflandırıcı veri seti — ROI .npy dosyaları + etiket CSV'si.

Etiket CSV formatı (labels.csv):
    case_id,patient_id,label
    HASTA001_pre,HASTA001,MB
    HASTA002_pre,HASTA002,PA
    ...

Önemli: patient_id, case_id'den AYRI tutulur. Aynı hastanın birden çok
görüntüsü olabilir; hasta-bazlı bölme (data leakage'a karşı) patient_id'ye göre
yapılır — bkz. split_patient_level.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose, EnsureChannelFirst, NormalizeIntensity, RandAffine,
    RandFlip, RandGaussianNoise, RandScaleIntensity, ToTensor,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from .model import CLASSES

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class Sample:
    case_id: str
    patient_id: str
    label: int
    roi_path: Path


def read_labels(csv_path: Path, roi_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().upper()
            if label not in CLASS_TO_IDX:
                raise ValueError(f"Geçersiz etiket '{label}'. Geçerli: {CLASSES}")
            roi = roi_dir / f"{row['case_id']}_roi.npy"
            if not roi.exists():
                raise FileNotFoundError(f"ROI bulunamadı: {roi}")
            samples.append(Sample(row["case_id"], row["patient_id"],
                                  CLASS_TO_IDX[label], roi))
    return samples


def split_patient_level(
    samples: list[Sample], n_folds: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    """Hasta-bazlı stratified K-fold. Aynı hasta hem train hem val'de OLAMAZ.
    Dönüş: her fold için (train_idx, val_idx)."""
    y = [s.label for s in samples]
    groups = [s.patient_id for s in samples]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(samples)), y, groups))


def train_transforms() -> Compose:
    """Az veriye karşı agresif augmentation (context: overfit önlemi)."""
    return Compose([
        EnsureChannelFirst(channel_dim=0),   # (C,Z,Y,X) zaten kanal-önce
        NormalizeIntensity(nonzero=True, channel_wise=True),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandAffine(prob=0.5, rotate_range=(0.26, 0.26, 0.26),
                   scale_range=(0.1, 0.1, 0.1), padding_mode="zeros"),
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandGaussianNoise(prob=0.3, std=0.05),
        ToTensor(),
    ])


def val_transforms() -> Compose:
    return Compose([
        EnsureChannelFirst(channel_dim=0),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        ToTensor(),
    ])


class ROIDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        roi = np.load(s.roi_path).astype(np.float32)  # (C, Z, Y, X)
        x = self.transform(roi)
        return x, torch.tensor(s.label, dtype=torch.long)


def class_weights(samples: list[Sample]) -> torch.Tensor:
    """Sınıf dengesizliği için ters-frekans ağırlıkları."""
    counts = np.bincount([s.label for s in samples], minlength=len(CLASSES))
    counts = np.maximum(counts, 1)
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)
