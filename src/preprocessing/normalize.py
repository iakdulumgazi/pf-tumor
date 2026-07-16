"""Yoğunluk normalizasyonu (z-score, beyin maskesi içinde).

Skull stripping sonrası beyin dışı voksel 0'dır. Z-score yalnızca beyin
voksellerinden (arka plan hariç) hesaplanır — BraTS/nnU-Net standardı.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def zscore_normalize(
    image: sitk.Image,
    mask: sitk.Image | None = None,
    nonzero_as_mask: bool = True,
) -> sitk.Image:
    """Beyin voksellerini sıfır ortalama / birim varyansa getirir.

    mask verilirse onu kullanır; yoksa nonzero_as_mask=True ile sıfır-olmayan
    vokseller beyin kabul edilir (skull strip sonrası geçerli).
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    if mask is not None:
        m = sitk.GetArrayFromImage(mask).astype(bool)
    elif nonzero_as_mask:
        m = arr > 0
    else:
        m = np.ones_like(arr, dtype=bool)

    if m.sum() == 0:
        raise ValueError("Normalizasyon için maske boş (hiç beyin vokseli yok).")

    mean = arr[m].mean()
    std = arr[m].std()
    if std < 1e-8:
        std = 1.0
    out = np.zeros_like(arr)
    out[m] = (arr[m] - mean) / std

    result = sitk.GetImageFromArray(out)
    result.CopyInformation(image)
    return result


def normalize_case(
    inputs: dict[str, Path],
    out_dir: Path,
    case_id: str,
    brain_mask: Path | None = None,
) -> dict[str, Path]:
    """Bir vakanın tüm sekanslarını z-score normalize eder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_to_index = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    mask_img = sitk.ReadImage(str(brain_mask)) if brain_mask else None

    written: dict[str, Path] = {}
    for seq, path in inputs.items():
        idx = seq_to_index[seq]
        img = sitk.ReadImage(str(path), sitk.sitkFloat32)
        norm = zscore_normalize(img, mask_img)
        dst = out_dir / f"{case_id}_{idx:04d}.nii.gz"
        sitk.WriteImage(norm, str(dst))
        written[seq] = dst
    return written
