"""Tam preprocessing pipeline'ı: DICOM -> nnU-Net'e hazır hacimler.

Adımlar:
    1. DICOM -> NIfTI + sekans tanıma           (dicom_to_nifti)
    2. Ko-registrasyon (T1CE ref) + 1mm resample (registration)
    3. Skull stripping (HD-BET) — tek maske, tüm sekanslara  (skull_strip)
    4. Z-score normalizasyon (beyin maskesi içinde)  (normalize)

Sıra notu: BraTS literatürü "önce skull strip" der; ancak tek beyin maskesini
tüm sekanslara uygulayabilmek için sekansların hizalı olması gerekir. Bu yüzden
ko-registrasyonu öne aldık (teknik olarak daha tutarlı, sonuç eşdeğer).

Kullanım:
    python -m src.preprocessing.pipeline \
        --dicom-dir data/raw/HASTA001 --case-id HASTA001 \
        --work-dir  data/preprocessed/HASTA001
    # HD-BET kurulu değilse:  --skip-skull-strip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dicom_to_nifti as d2n
from . import registration as reg
from . import normalize as norm
from . import skull_strip as ss


def run_pipeline(
    dicom_dir: Path,
    case_id: str,
    work_dir: Path,
    overrides: dict[str, str] | None = None,
    skip_skull_strip: bool = False,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str | None = None,
) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/4] DICOM -> NIfTI  ({case_id})")
    conv = d2n.convert_case(dicom_dir, work_dir / "01_nifti", case_id, overrides)
    d2n._print_report(conv)
    if conv.missing_sequences:
        sys.exit(
            f"\nDURDU: eksik sekans {conv.missing_sequences}. "
            "--override ile eşle ya da veriyi kontrol et."
        )

    print(f"\n[2/4] Ko-registrasyon (T1CE ref) + {spacing} mm resample")
    aligned = reg.coregister_case(conv.mapping, work_dir / "02_aligned", case_id, spacing)
    print("  hizalandı:", ", ".join(f"{k}" for k in aligned))

    if skip_skull_strip:
        print("\n[3/4] Skull stripping ATLANDI (--skip-skull-strip)")
        stripped, brain_mask = aligned, None
    else:
        print(f"\n[3/4] Skull stripping (HD-BET, device={device or ss._pick_device()})")
        stripped, brain_mask = ss.skull_strip_case(
            aligned, work_dir / "03_stripped", case_id, "T1CE", device
        )
        print("  maske:", brain_mask.name if brain_mask else "-")

    print("\n[4/4] Z-score normalizasyon")
    final = norm.normalize_case(stripped, work_dir / "04_final", case_id, brain_mask)
    print("  nnU-Net'e hazır:")
    for seq, path in sorted(final.items(), key=lambda kv: kv[1].name):
        print(f"    {seq:5} -> {path.name}")
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tam preprocessing pipeline (DICOM -> nnU-Net)")
    p.add_argument("--dicom-dir", required=True, type=Path)
    p.add_argument("--case-id", required=True)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--override", action="append", default=[], metavar="SEKANS=parça")
    p.add_argument("--skip-skull-strip", action="store_true",
                   help="HD-BET kurulu değilse skull strip'i atla")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = p.parse_args(argv)

    overrides = {}
    for item in args.override:
        seq, needle = item.split("=", 1)
        overrides[seq.upper()] = needle

    run_pipeline(
        args.dicom_dir, args.case_id, args.work_dir,
        overrides=overrides or None,
        skip_skull_strip=args.skip_skull_strip,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
