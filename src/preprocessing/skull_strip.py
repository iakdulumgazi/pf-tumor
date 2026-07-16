"""Skull stripping (beyin maskeleme) — HD-BET sarmalayıcı (v2 CLI).

HD-BET (MIC-DKFZ) standart araçtır. v2, Apple Silicon'da 'mps' cihazını native
destekler (v1'de yalnızca cuda/cpu vardı).

Kurulum:
    pip install HD-BET

Klinik sırada skull stripping ko-registrasyondan ÖNCE her sekansa ayrı uygulanır;
pratikte genelde T1CE üzerinde maske çıkarılıp diğerlerine uygulanır. Burada
basitlik için referans (T1CE) üzerinde maske üretip tüm sekanslara uyguluyoruz.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _pick_device() -> str:
    """CUDA varsa CUDA (Windows/Linux GPU), yoksa CPU.

    NOT: HD-BET v2 nominal olarak MPS destekler ama uygulamada bazı işlemleri
    CPU'ya düşürüyor (nnU-Net perform_everything_on_device uyarısı) ve Mac'te
    zaten `--disable_tta` ile CPU tercih ediliyor. MPS bilinçli olarak
    devre dışı — kararlılık için."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _hdbet_exe() -> str | None:
    """hd-bet çalıştırılabilirini bul: önce PATH, sonra Python yorumlayıcısının
    yanındaki bin/Scripts klasörü (venv activate edilmeden çalışsın diye).
    Windows'ta 'hd-bet.exe', Unix'te 'hd-bet' aranır."""
    exe = shutil.which("hd-bet")
    if exe:
        return exe
    bin_dir = Path(sys.executable).parent  # .venv/bin (Unix) veya .venv/Scripts (Windows)
    for name in ("hd-bet", "hd-bet.exe"):
        candidate = bin_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def hdbet_available() -> bool:
    return _hdbet_exe() is not None


def run_hdbet(
    input_path: Path,
    output_path: Path,
    device: str | None = None,
    log_fn=None,
) -> Path:
    """HD-BET'i bir görüntüye uygular. Beyin maskesini (_bet_mask) döndürür.

    log_fn: verilirse HD-BET'in stdout/stderr satırlarını canlı olarak alır
    (kullanıcıya ilerleme göstermek için). Yoksa satırlar bastırılır.
    """
    exe = _hdbet_exe()
    if exe is None:
        sys.exit(
            "HATA: 'hd-bet' bulunamadı.\n"
            "  Kurulum:  pip install HD-BET\n"
            "  (PyTorch gerektirir; CUDA yoksa CPU'da çalışır.)"
        )
    device = device or _pick_device()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "-i", str(input_path), "-o", str(output_path),
           "-device", device, "--save_bet_mask"]
    # CPU'da TTA (test-time augmentation) kapatmak hızı belirgin artırır.
    if device == "cpu":
        cmd += ["--disable_tta"]

    # ÖNEMLİ: HD-BET v2 içinde nnU-Net multiprocessing worker'ları başlatır.
    # Bu worker'lar stdout PIPE'ının yazma ucunu miras alır; asıl süreç bitse bile
    # pipe açık kaldığı için read() EOF alamaz -> sonsuz deadlock. Bu yüzden çıktıyı
    # bir PIPE yerine geçici DOSYAYA yönlendiriyoruz (okumuyoruz), süreç bitişini
    # wait(timeout) ile yokluyoruz. Canlı log dosyanın son satırından okunur.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_file = tempfile.NamedTemporaryFile(
        mode="w+", suffix="_hdbet.log", delete=False, dir=str(output_path.parent)
    )
    log_path = Path(log_file.name)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    start = time.time()
    while True:
        try:
            ret = proc.wait(timeout=2.0)
            break
        except subprocess.TimeoutExpired:
            if log_fn is not None:
                elapsed = int(time.time() - start)
                mm, ss = divmod(elapsed, 60)
                tail = "çalışıyor..."
                try:
                    lines = log_path.read_text(errors="replace").strip().splitlines()
                    if lines:
                        tail = lines[-1].strip()[-100:] or tail
                except OSError:
                    pass
                try:
                    log_fn(f"[{mm:02d}:{ss:02d}] {tail}")
                except Exception:
                    pass

    log_file.close()
    if ret != 0:
        try:
            sys.stderr.write(log_path.read_text(errors="replace") + "\n")
        except OSError:
            pass
        raise RuntimeError(f"hd-bet başarısız (kod {ret})")

    # --save_bet_mask ile maske <output>_mask.nii.gz olarak yazılır
    mask = Path(str(output_path).replace(".nii.gz", "_mask.nii.gz"))
    if mask.exists():
        return mask
    # bulunamazsa stripped çıktıdan maske türet
    return derive_mask_from_stripped(output_path)


def derive_mask_from_stripped(stripped_path: Path) -> Path:
    """Skull-stripped görüntüden (beyin>0) ikili maske üretir."""
    img = sitk.ReadImage(str(stripped_path))
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(img)
    mask_path = stripped_path.with_name(stripped_path.name.replace(".nii.gz", "_mask.nii.gz"))
    sitk.WriteImage(mask, str(mask_path))
    return mask_path


def apply_mask(image_path: Path, mask_path: Path, out_path: Path) -> Path:
    """Bir beyin maskesini bir görüntüye uygular (maske dışı = 0)."""
    img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path))
    mask = sitk.Resample(mask, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    masked = sitk.Mask(img, sitk.Cast(mask, sitk.sitkUInt8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(masked, str(out_path))
    return out_path


def skull_strip_case(
    inputs: dict[str, Path],
    out_dir: Path,
    case_id: str,
    reference_seq: str = "T1CE",
    device: str | None = None,
    log_fn=None,
) -> tuple[dict[str, Path], Path]:
    """Referans sekansta maske üretir, tüm sekanslara uygular.

    Dönüş: (stripped_inputs, brain_mask_path)
    NOT: Bu fonksiyon, sekansların zaten hizalı (ko-registre) olduğunu varsayar.
    Tek maske tüm sekanslara uygulanır.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if reference_seq not in inputs:
        raise ValueError(f"Referans sekans '{reference_seq}' eşlemede yok.")

    ref_stripped = out_dir / f"{case_id}_{reference_seq}_stripped.nii.gz"
    mask_path = run_hdbet(inputs[reference_seq], ref_stripped, device, log_fn=log_fn)

    seq_to_index = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    stripped: dict[str, Path] = {}
    for seq, path in inputs.items():
        idx = seq_to_index[seq]
        dst = out_dir / f"{case_id}_{idx:04d}.nii.gz"
        apply_mask(path, mask_path, dst)
        stripped[seq] = dst
    return stripped, mask_path
