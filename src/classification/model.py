"""Sınıflandırıcı model fabrikası — MONAI 3D DenseNet121.

Girdi: 4 kanallı 3D ROI (FLAIR/T1/T1CE/T2). Çıkış: 3 sınıf (MB/PA/EP).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

CLASSES = ["MB", "PA", "EP"]  # Medulloblastom, Pilositik Astrositom, Ependimom


def build_model(
    num_classes: int = 3,
    in_channels: int = 4,
    dropout: float = 0.2,
    pretrained_path: str | None = None,
) -> nn.Module:
    """4-kanal girdili 3D DenseNet121 döndürür.

    pretrained_path verilirse önceden eğitilmiş ağırlıkları yükler (ör. CBTN veya
    kendi ön-eğitimimiz). Şekil uymayan katmanlar atlanır (transfer learning).
    """
    model = DenseNet121(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        dropout_prob=dropout,
    )
    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        own = model.state_dict()
        loaded = {k: v for k, v in state.items()
                  if k in own and v.shape == own[k].shape}
        own.update(loaded)
        model.load_state_dict(own)
        print(f"pretrained: {len(loaded)}/{len(own)} katman yüklendi ({pretrained_path})")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
