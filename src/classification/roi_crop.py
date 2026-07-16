"""Tümör ROI kırpma — segmentasyon → sınıflandırıcı köprüsü.

BraTS-PEDs maskesinden whole-tumor sınırlayıcı kutusunu (bounding box) bulur,
çevresine bir miktar bağlam (margin) ekler, 4 sekansı + maskeyi kırpar ve
sınıflandırıcı için sabit küp boyutuna (varsayılan 96³) getirir.

Bağlam neden korunur: peritümöral ödem ve tümörün konumu (4. ventrikül, serebellar
hemisfer vs.) MB/PA/EP ayrımında klinik olarak değerli — sadece tümörü maskeleyip
atmak bilgi kaybettirir. İsteğe bağlı 'mask_out' ile tümör-dışı sıfırlanabilir.

Çıktı:
    <case>_roi.npy        (C, Z, Y, X) float32 — sınıflandırıcıya doğrudan girdi
    <case>_roi_0000..0003.nii.gz  kırpılmış sekanslar (görselleştirme/denetim)
    <case>_roi_mask.nii.gz        kırpılmış tümör maskesi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

CHANNELS = {0: "FLAIR", 1: "T1", 2: "T1CE", 3: "T2"}


def tumor_bbox(
    mask: np.ndarray, margin_vox: int
) -> tuple[slice, slice, slice] | None:
    """Whole-tumor (mask>0) sınırlayıcı kutusu + margin. Tümör yoksa None."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    lo = np.maximum(lo - margin_vox, 0)
    hi = np.minimum(hi + margin_vox, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _resize(arr: np.ndarray, target: tuple[int, int, int], is_mask: bool) -> np.ndarray:
    """Kırpılmış hacmi sabit boyuta getirir (SimpleITK; mask için nearest)."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    orig_size = img.GetSize()  # (x, y, z)
    orig_spacing = img.GetSpacing()
    tgt_xyz = target[::-1]      # numpy (z,y,x) -> sitk (x,y,z)
    new_spacing = [os_ * osz / tsz for os_, osz, tsz in
                   zip(orig_spacing, orig_size, tgt_xyz)]
    rs = sitk.ResampleImageFilter()
    rs.SetSize([int(t) for t in tgt_xyz])
    rs.SetOutputSpacing(new_spacing)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return sitk.GetArrayFromImage(rs.Execute(img))


def crop_case(
    image_paths: dict[str, Path],
    mask_path: Path,
    out_dir: Path,
    case_id: str,
    margin_mm: int = 10,
    target_size: tuple[int, int, int] = (96, 96, 96),
    mask_out: bool = False,
    save_nifti: bool = True,
) -> Path:
    """Bir vakanın tümör ROI'sini kırpar ve sınıflandırıcı girdisi üretir.

    image_paths: {"FLAIR":.., "T1":.., "T1CE":.., "T2":..} (hizalı, 1mm).
    margin_mm: 1mm izotropik olduğundan margin_vox == margin_mm.
    mask_out: True ise tümör-dışı vokseller sıfırlanır (sadece tümör yoğunlukları).
    Dönüş: <case>_roi.npy yolu.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_img = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(mask_img)  # (z, y, x)

    box = tumor_bbox(mask, margin_mm)
    if box is None:
        raise ValueError(
            f"{case_id}: maskede tümör yok (whole-tumor boş). "
            "Segmentasyon başarısız olmuş olabilir."
        )

    mask_crop = mask[box]
    mask_rs = _resize(mask_crop, target_size, is_mask=True)

    channels = []
    for idx in range(4):
        seq = CHANNELS[idx]
        if seq not in image_paths:
            raise ValueError(f"{case_id}: '{seq}' sekansı eksik.")
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(image_paths[seq]), sitk.sitkFloat32))
        if arr.shape != mask.shape:
            raise ValueError(
                f"{case_id}: {seq} geometrisi maskeyle uyuşmuyor "
                f"({arr.shape} vs {mask.shape}). Aynı uzayda olmalılar."
            )
        crop = arr[box]
        if mask_out:
            crop = crop * (mask_crop > 0)
        channels.append(_resize(crop, target_size, is_mask=False))

    roi = np.stack(channels, axis=0).astype(np.float32)  # (C, Z, Y, X)
    npy_path = out_dir / f"{case_id}_roi.npy"
    np.save(npy_path, roi)

    if save_nifti:
        for idx in range(4):
            out = sitk.GetImageFromArray(roi[idx])
            sitk.WriteImage(out, str(out_dir / f"{case_id}_roi_{idx:04d}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(mask_rs.astype(np.uint8)),
                        str(out_dir / f"{case_id}_roi_mask.nii.gz"))

    box_size = tuple(s.stop - s.start for s in box)
    print(f"{case_id}: tümör bbox {box_size} (+{margin_mm}mm margin) "
          f"-> {target_size}, {roi.shape} kaydedildi")
    print(f"  npy: {npy_path.name}  | tümör voksel: {int((mask_crop>0).sum())}")
    return npy_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tümör ROI kırpma (segmentasyon -> sınıflandırıcı)")
    p.add_argument("--image-dir", required=True, type=Path,
                   help="<case>_0000..0003.nii.gz içeren klasör (hizalı sekanslar)")
    p.add_argument("--mask", required=True, type=Path, help="4-etiketli segmentasyon maskesi")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--case-id", required=True)
    p.add_argument("--margin-mm", type=int, default=10)
    p.add_argument("--size", type=int, nargs=3, default=[96, 96, 96], metavar=("Z", "Y", "X"))
    p.add_argument("--mask-out", action="store_true", help="Tümör-dışını sıfırla")
    args = p.parse_args(argv)

    image_paths = {seq: args.image_dir / f"{args.case_id}_{idx:04d}.nii.gz"
                   for idx, seq in CHANNELS.items()}
    crop_case(image_paths, args.mask, args.out_dir, args.case_id,
              args.margin_mm, tuple(args.size), args.mask_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
