"""Ko-registrasyon ve resampling (SimpleITK).

Tüm sekanslar T1CE referansına hizalanır (rigid), ardından 1×1×1 mm izotropik
çözünürlüğe getirilir. BraTS standardı.
"""
from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


def resample_to_spacing(
    image: sitk.Image,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False,
) -> sitk.Image:
    """Görüntüyü verilen izotropik spacing'e yeniden örnekler."""
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(orig_size, orig_spacing, spacing)
    ]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(spacing)
    rs.SetSize(new_size)
    rs.SetOutputDirection(image.GetDirection())
    rs.SetOutputOrigin(image.GetOrigin())
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(interp)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(image)


def register_rigid(
    moving: sitk.Image,
    fixed: sitk.Image,
) -> tuple[sitk.Image, sitk.Transform]:
    """moving görüntüyü fixed referansına rigid (6-DOF) hizalar."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)

    transform = reg.Execute(fixed_f, moving_f)

    resampled = sitk.Resample(
        moving_f, fixed_f, transform, sitk.sitkLinear, 0.0, moving_f.GetPixelID()
    )
    return resampled, transform


def coregister_case(
    nnunet_inputs: dict[str, Path],
    out_dir: Path,
    case_id: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Path]:
    """Bir vakanın 4 sekansını T1CE'ye hizalar + izotropik resample yapar.

    nnunet_inputs: {"FLAIR":path, "T1":path, "T1CE":path, "T2":path}
    Dönüş: işlenmiş dosya yolları (aynı sözlük yapısı).
    """
    if "T1CE" not in nnunet_inputs:
        raise ValueError("Referans olarak T1CE gerekli ama eşlemede yok.")

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_to_index = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    # Referansı önce resample et
    ref = sitk.ReadImage(str(nnunet_inputs["T1CE"]), sitk.sitkFloat32)
    ref = resample_to_spacing(ref, spacing)

    written: dict[str, Path] = {}
    for seq, path in nnunet_inputs.items():
        idx = seq_to_index[seq]
        dst = out_dir / f"{case_id}_{idx:04d}.nii.gz"
        if seq == "T1CE":
            sitk.WriteImage(ref, str(dst))
        else:
            img = sitk.ReadImage(str(path), sitk.sitkFloat32)
            aligned, _ = register_rigid(img, ref)
            sitk.WriteImage(aligned, str(dst))
        written[seq] = dst
    return written
