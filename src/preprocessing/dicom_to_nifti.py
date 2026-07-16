"""DICOM -> NIfTI dönüşümü ve sekans (T1 / T1CE / T2 / FLAIR) tanıma.

dcm2niix'i sarmalar. Bir hasta klasöründeki tüm DICOM serilerini NIfTI'ye
çevirir, ardından SeriesDescription / dosya adı üzerinden hangi serinin hangi
sekans olduğunu otomatik tahmin eder. Otomatik tahmin belirsizse kullanıcı
manuel eşleme (mapping) ile düzeltebilir.

Çıktı, nnU-Net / BraTS-PEDs konvansiyonuna göre adlandırılır:
    <case>_0000.nii.gz  -> FLAIR
    <case>_0001.nii.gz  -> T1
    <case>_0002.nii.gz  -> T1CE  (T1 kontrastlı)
    <case>_0003.nii.gz  -> T2

Kullanım:
    python -m src.preprocessing.dicom_to_nifti \
        --dicom-dir data/raw/HASTA001 \
        --out-dir   data/nifti/HASTA001 \
        --case-id   HASTA001
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# nnU-Net kanal sırası: index -> sekans
CHANNEL_ORDER = {0: "FLAIR", 1: "T1", 2: "T1CE", 3: "T2"}
SEQUENCES = list(CHANNEL_ORDER.values())

# SeriesDescription içinde aranacak anahtar kelimeler (küçük harfe çevrilmiş).
# Sıra önemli: T1CE, T1'den önce gelmeli; FLAIR, T2'den önce kontrol edilmeli.
_PATTERNS = [
    ("FLAIR", [r"flair", r"t2[\s_\-]*flair", r"dark[\s_\-]*fluid"]),
    ("T1CE", [r"t1.*(\+c|ce|gd|post|contrast|km|kontrast)", r"(\+c|gd|post).*t1",
              r"mprage.*(post|gd|\+c)", r"t1c\b"]),
    ("T1", [r"\bt1\b", r"mprage", r"t1[\s_\-]*w", r"spgr"]),
    ("T2", [r"\bt2\b", r"t2[\s_\-]*w", r"tse"]),
]


@dataclass
class ConvertedSeries:
    nifti_path: Path
    json_path: Path | None
    series_description: str
    guessed_sequence: str | None = None

    @property
    def label(self) -> str:
        return self.guessed_sequence or "?"


@dataclass
class ConversionResult:
    case_id: str
    series: list[ConvertedSeries] = field(default_factory=list)
    # sekans -> seçilen NIfTI yolu (nnU-Net'e hazır)
    mapping: dict[str, Path] = field(default_factory=dict)

    @property
    def missing_sequences(self) -> list[str]:
        return [s for s in SEQUENCES if s not in self.mapping]


def _check_dcm2niix() -> None:
    if shutil.which("dcm2niix") is None:
        sys.exit(
            "HATA: 'dcm2niix' bulunamadı. Kurulum:  brew install dcm2niix"
        )


def run_dcm2niix(dicom_dir: Path, out_dir: Path) -> None:
    """Bir DICOM klasörünü NIfTI'ye çevirir (her seri ayrı dosya)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z", "y",            # gzip (.nii.gz)
        "-b", "y",            # BIDS JSON sidecar üret
        "-f", "%s_%d",        # dosya adı: seriNo_seriAçıklaması
        "-o", str(out_dir),
        str(dicom_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"dcm2niix başarısız (kod {proc.returncode})")


def _read_series_description(nifti_path: Path, json_path: Path | None) -> str:
    """Önce JSON sidecar'dan SeriesDescription, yoksa dosya adından çıkar."""
    if json_path and json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
            desc = meta.get("SeriesDescription") or meta.get("ProtocolName")
            if desc:
                return str(desc)
        except (json.JSONDecodeError, OSError):
            pass
    # dosya adı:  "5_T1_MPRAGE.nii.gz" -> "T1_MPRAGE"
    stem = nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
    return re.sub(r"^\d+_", "", stem)


def guess_sequence(description: str) -> str | None:
    """SeriesDescription metninden sekans tipini tahmin eder."""
    text = description.lower()
    for seq, patterns in _PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                return seq
    return None


def collect_series(out_dir: Path) -> list[ConvertedSeries]:
    series: list[ConvertedSeries] = []
    for nii in sorted(out_dir.glob("*.nii*")):
        if nii.name.endswith(".json"):
            continue
        json_path = nii.with_suffix("").with_suffix(".json")
        if not json_path.exists():
            json_path = Path(str(nii).replace(".nii.gz", ".json"))
        desc = _read_series_description(nii, json_path if json_path.exists() else None)
        series.append(
            ConvertedSeries(
                nifti_path=nii,
                json_path=json_path if json_path.exists() else None,
                series_description=desc,
                guessed_sequence=guess_sequence(desc),
            )
        )
    return series


def build_mapping(series: list[ConvertedSeries]) -> dict[str, Path]:
    """Tahminlerden sekans->dosya eşlemesi. Çakışma olursa ilkini alır."""
    mapping: dict[str, Path] = {}
    for s in series:
        if s.guessed_sequence and s.guessed_sequence not in mapping:
            mapping[s.guessed_sequence] = s.nifti_path
    return mapping


def write_nnunet_inputs(
    mapping: dict[str, Path], dest_dir: Path, case_id: str
) -> dict[str, Path]:
    """Eşlemeyi nnU-Net adlandırmasıyla (<case>_000X.nii.gz) kopyalar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    seq_to_index = {v: k for k, v in CHANNEL_ORDER.items()}
    for seq, src in mapping.items():
        idx = seq_to_index[seq]
        dst = dest_dir / f"{case_id}_{idx:04d}.nii.gz"
        shutil.copy2(src, dst)
        written[seq] = dst
    return written


def convert_case(
    dicom_dir: Path,
    out_dir: Path,
    case_id: str,
    overrides: dict[str, str] | None = None,
) -> ConversionResult:
    """Tek bir hastayı uçtan uca çevirir.

    overrides: {sekans: dosya_adı_parçası} -- otomatik tahmin yanlışsa
    elle düzeltmek için (ör. {"T1CE": "5_T1_post"}).
    """
    _check_dcm2niix()
    run_dcm2niix(dicom_dir, out_dir)
    series = collect_series(out_dir)
    if not series:
        raise RuntimeError(f"NIfTI üretilmedi: {dicom_dir} içinde DICOM serisi yok?")

    mapping = build_mapping(series)

    if overrides:
        for seq, needle in overrides.items():
            match = next(
                (s.nifti_path for s in series if needle.lower() in s.nifti_path.name.lower()),
                None,
            )
            if match is None:
                raise ValueError(f"override '{needle}' hiçbir seriyle eşleşmedi ({seq})")
            mapping[seq] = match

    result = ConversionResult(case_id=case_id, series=series, mapping=mapping)
    return result


def _print_report(result: ConversionResult) -> None:
    print(f"\n=== {result.case_id} — bulunan seriler ===")
    for s in result.series:
        print(f"  [{s.label:5}] {s.series_description}  ->  {s.nifti_path.name}")
    print("\n=== sekans eşlemesi ===")
    for seq in SEQUENCES:
        path = result.mapping.get(seq)
        status = path.name if path else "!! EKSİK !!"
        print(f"  {seq:5} : {status}")
    if result.missing_sequences:
        print(
            f"\nUYARI: eksik sekans(lar): {', '.join(result.missing_sequences)}.\n"
            "  --override SEKANS=dosya_parçası ile elle eşleyebilirsin."
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DICOM klasörünü NIfTI'ye çevir ve sekansları tanı")
    p.add_argument("--dicom-dir", required=True, type=Path, help="Hastanın DICOM klasörü")
    p.add_argument("--out-dir", required=True, type=Path, help="NIfTI çıktı klasörü")
    p.add_argument("--case-id", required=True, help="Hasta/vaka kimliği (anonim)")
    p.add_argument(
        "--override", action="append", default=[],
        metavar="SEKANS=parça",
        help="Otomatik tahmini ez, ör: --override T1CE=5_T1_post",
    )
    p.add_argument(
        "--nnunet-dir", type=Path, default=None,
        help="Verilirse eşlemeyi nnU-Net adlandırmasıyla buraya kopyalar",
    )
    args = p.parse_args(argv)

    overrides = {}
    for item in args.override:
        if "=" not in item:
            p.error(f"--override formatı SEKANS=parça olmalı: '{item}'")
        seq, needle = item.split("=", 1)
        seq = seq.upper()
        if seq not in SEQUENCES:
            p.error(f"geçersiz sekans '{seq}'. Geçerli: {SEQUENCES}")
        overrides[seq] = needle

    result = convert_case(args.dicom_dir, args.out_dir, args.case_id, overrides)
    _print_report(result)

    if args.nnunet_dir:
        written = write_nnunet_inputs(result.mapping, args.nnunet_dir, args.case_id)
        print(f"\nnnU-Net girdileri yazıldı ({len(written)}/4):")
        for seq, path in written.items():
            print(f"  {seq:5} -> {path.name}")

    return 0 if not result.missing_sequences else 2


if __name__ == "__main__":
    raise SystemExit(main())
