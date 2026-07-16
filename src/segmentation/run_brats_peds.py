"""BraTS-PEDs segmentasyon inference sarmalayıcısı.

İki nnU-Net modelini (WT: Dataset106, 3L: Dataset107) çalıştırıp çıktıları
4-etiketli bir maskeye birleştirir (repo'daki conversion.py mantığı):

    nihai etiketler:  1=ET (kontrast tutan)  2=NCR  3=CC (kistik)  4=ED (ödem)
    whole tumor (WT) = tüm sıfır-olmayan bölge

Girdi: preprocessing pipeline'ının skull-stripped + ko-registre çıktısı
(nnU-Net formatı: <case>_0000=FL, _0001=T1, _0002=T1CE, _0003=T2). nnU-Net
resample + z-score'u kendi içinde yapar (plans: 1mm, maske-içi z-score).

CPU notu: 3D full-res nnU-Net CPU'da yavaştır. Hızlı doğrulama için tek fold
(folds=(0,)) kullan; klinik/üretimde 5-fold ensemble (folds=(0,1,2,3,4)).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2] / "models" / "braTS_peds_repo"
RESULTS = REPO / "data" / "nnUNet_results"
WT_MODEL = RESULTS / "Dataset106_WTPED24" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
TL_MODEL = RESULTS / "Dataset107_3LabelPED24" / "nnUNetTrainer__nnUNetPlans__3d_fullres"

LABELS = {1: "ET (kontrast tutan)", 2: "NCR (nekroz)", 3: "CC (kistik)", 4: "ED (ödem)"}


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    # nnU-Net'te MPS bazı op'ları desteklemeyebilir; CPU güvenli varsayılan.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_predictor(device: torch.device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    return nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,        # TTA kapalı — CPU'da hız için
        perform_everything_on_device=device.type in ("cuda", "mps"),
        device=device,
        verbose=False,
        allow_tqdm=True,
    )


def _predict(model_dir: Path, input_dir: Path, output_dir: Path,
             folds: tuple[int, ...], device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = _make_predictor(device)
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=folds, checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files(
        str(input_dir), str(output_dir),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
    )


def combine_masks(wt_path: Path, tl_path: Path, out_path: Path) -> dict[int, int]:
    """WT + 3L maskelerini 4-etiketli nihai maskeye birleştirir.
    Dönüş: {etiket: voksel_sayısı}."""
    wt = nib.load(str(wt_path))
    tl = nib.load(str(tl_path))
    dwt = wt.get_fdata()
    dtl = tl.get_fdata()

    new = np.zeros(dwt.shape, dtype=np.uint8)
    new[dwt == 1] = 2          # WT -> NCR taban
    new[dtl == 1] = 1          # ET
    new[dtl == 2] = 3          # CC
    new[dtl == 3] = 4          # ED

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(new, wt.affine), str(out_path))
    return {lab: int((new == lab).sum()) for lab in LABELS}


def segment_case(
    input_dir: Path,
    output_dir: Path,
    case_id: str,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    device: str | None = None,
) -> Path:
    """Bir vakayı segmente eder. input_dir: <case>_0000..0003.nii.gz içermeli.
    Dönüş: 4-etiketli nihai maske yolu."""
    if not WT_MODEL.exists() or not TL_MODEL.exists():
        raise FileNotFoundError(
            "BraTS-PEDs modelleri kurulu değil. Önce WT.zip/3L.zip indirilip\n"
            "nnUNetv2_install_pretrained_model_from_zip ile kurulmalı."
        )
    dev = _pick_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    wt_out = output_dir / "outputWT"
    tl_out = output_dir / "output3L"

    print(f"[WT] segmentasyon (Dataset106, folds={folds}, device={dev})")
    _predict(WT_MODEL, input_dir, wt_out, folds, dev)
    print(f"[3L] segmentasyon (Dataset107, folds={folds}, device={dev})")
    _predict(TL_MODEL, input_dir, tl_out, folds, dev)

    wt_file = wt_out / f"{case_id}.nii.gz"
    tl_file = tl_out / f"{case_id}.nii.gz"
    final = output_dir / f"{case_id}_seg.nii.gz"
    counts = combine_masks(wt_file, tl_file, final)

    total = sum(counts.values())
    print(f"\nNihai maske: {final.name}")
    for lab, n in counts.items():
        print(f"  {lab} {LABELS[lab]:22}: {n:>8} voksel")
    print(f"  whole tumor (toplam)   : {total:>8} voksel")
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BraTS-PEDs segmentasyon inference")
    p.add_argument("--input-dir", required=True, type=Path,
                   help="<case>_0000..0003.nii.gz içeren klasör (skull-stripped)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--case-id", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Kullanılacak fold'lar (hızlı test için tek fold: --folds 0)")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = p.parse_args(argv)
    segment_case(args.input_dir, args.output_dir, args.case_id,
                 tuple(args.folds), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
