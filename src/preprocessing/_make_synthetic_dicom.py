"""Test amaçlı sentetik DICOM üretici.

Gerçek hasta verisi olmadan pipeline'ı denemek için 4 sekanslı (FLAIR/T1/T1CE/T2)
küçük bir 3D hacim seti üretir. Sadece geliştirme/test için — klinik değeri yoktur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage


def _make_series(out_dir: Path, series_desc: str, series_num: int,
                 shape=(16, 64, 64), seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vol = (rng.random(shape) * 400).astype(np.uint16)
    # ortaya parlak bir "lezyon" koy
    z, y, x = shape
    vol[z//2-2:z//2+2, y//2-6:y//2+6, x//2-6:x//2+6] += 500
    vol = np.clip(vol, 0, 4095).astype(np.uint16)

    series_uid = generate_uid()
    study_uid = generate_uid()
    for i in range(shape[0]):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "TEST^SENTETIK"
        ds.PatientID = "SYN001"
        ds.Modality = "MR"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = MRImageStorage
        ds.SeriesNumber = series_num
        ds.SeriesDescription = series_desc
        ds.InstanceNumber = i + 1
        ds.Rows, ds.Columns = shape[1], shape[2]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0.0, 0.0, float(i)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PixelData = vol[i].tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(out_dir / f"{series_desc.replace(' ', '_')}_{i:03d}.dcm")


def main(dest: Path) -> None:
    series = [
        ("Ax T2 FLAIR", 1),
        ("Sag 3D T1 MPRAGE", 2),
        ("Ax T1 post Gd", 3),
        ("Ax T2 TSE", 4),
    ]
    for seed, (desc, num) in enumerate(series):
        _make_series(dest / f"series{num}", desc, num, seed=seed)
    print(f"Sentetik DICOM üretildi: {dest}  ({len(series)} seri)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/SYN001")
    main(out)
