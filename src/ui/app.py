"""Gradio arayüzü — DICOM yükle -> NIfTI -> preprocessing -> BraTS-PEDs segmentasyon.

Sınıflandırma (MB/PA/EP) henüz YOK — gerçek hasta verisiyle eğitilmiş bir model
gelene kadar arayüzden çıkarıldı (kullanıcı kararı). Bu sürüm yalnızca
segmentasyon sonucunu (tümör bölgesi + alt-etiketler) gösterir.

Çalıştırma:
    .venv/bin/python src/ui/app.py
"""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import dicom_to_nifti as d2n
from src.preprocessing import normalize as norm
from src.preprocessing import registration as reg
from src.preprocessing import skull_strip as ss
from src.segmentation.run_brats_peds import LABELS, segment_case

WORKDIR = Path(tempfile.gettempdir()) / "pf_tumor_ui"
WORKDIR.mkdir(exist_ok=True)
SEQS = ["FLAIR", "T1", "T1CE", "T2"]


def step1_convert(files):
    """DICOM klasörü (sürükle-bırak) veya .zip -> NIfTI + otomatik sekans tanıma.

    Gradio file_count='directory' bir klasör sürüklendiğinde her dosya için
    ayrı bir geçici yol içeren bir liste döndürür (alt klasör bilgisi kaybolur,
    ama dcm2niix serileri DICOM etiketlerinden (SeriesInstanceUID) tanır,
    dizin yapısından değil — bu yüzden düz kopyalama sorun yaratmaz).
    """
    if not files:
        return "Önce bir DICOM klasörü sürükleyin (veya .zip yükleyin).", None, None

    session_dir = Path(tempfile.mkdtemp(prefix="case_", dir=WORKDIR))
    dicom_dir = session_dir / "dicom"
    dicom_dir.mkdir()

    paths = [Path(f) if isinstance(f, str) else Path(f.name) for f in files]

    if len(paths) == 1 and paths[0].suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(paths[0]) as zf:
                zf.extractall(dicom_dir)
        except zipfile.BadZipFile:
            return "HATA: geçerli bir .zip dosyası değil.", None, None
    else:
        for i, p in enumerate(paths):
            if not p.is_file():
                continue
            dest = dicom_dir / f"{i:05d}_{p.name}"
            dest.write_bytes(p.read_bytes())
        if not any(dicom_dir.iterdir()):
            return "HATA: klasörde dosya bulunamadı.", None, None

    case_id = "CASE"
    try:
        result = d2n.convert_case(dicom_dir, session_dir / "01_nifti", case_id)
    except Exception as e:
        return f"HATA (DICOM dönüşümü): {e}", None, None

    rows = [[s.series_description, s.guessed_sequence or "—", s.nifti_path.name]
            for s in result.series]
    status = f"{len(result.series)} seri bulundu."
    if result.missing_sequences:
        status += f"  ⚠️ Otomatik eksik: {', '.join(result.missing_sequences)} — aşağıdan elle seçin."
    else:
        status += "  ✅ FLAIR/T1/T1CE/T2 tümü tanındı — kontrol edip devam edin."

    state = {
        "session_dir": str(session_dir),
        "case_id": case_id,
        "series": [{"desc": s.series_description, "path": str(s.nifti_path),
                    "guess": s.guessed_sequence} for s in result.series],
    }
    return status, rows, state


def _series_choices(state) -> list[tuple[str, str]]:
    """Dropdown seçenekleri: (görünen etiket, benzersiz değer=dosya yolu).

    Aynı SeriesDescription iki farklı seride bulunabilir (ör. T1 + T1CE aynı
    isim). Görünen etikete dosya adı da eklenir, backend değer olarak yolu alır."""
    if not state:
        return []
    return [(f"{s['desc']}  •  {Path(s['path']).name}", s["path"])
            for s in state["series"]]


def _guess_for(state, seq: str) -> str | None:
    """Otomatik tahmin edilen sekans için o serinin dosya yolunu (dropdown değeri) döndür."""
    if not state:
        return None
    for s in state["series"]:
        if s["guess"] == seq:
            return s["path"]
    return None


def on_convert(zip_file):
    status, rows, state = step1_convert(zip_file)
    choices = _series_choices(state)
    return (
        status, rows, state,
        gr.update(choices=choices, value=_guess_for(state, "FLAIR")),
        gr.update(choices=choices, value=_guess_for(state, "T1")),
        gr.update(choices=choices, value=_guess_for(state, "T1CE")),
        gr.update(choices=choices, value=_guess_for(state, "T2")),
    )


PLANE_AXIS = {"Aksiyel": 0, "Koronal": 1, "Sagital": 2}
# Etiket renkleri (RGB): 1=ET kırmızı, 2=NCR sarı, 3=CC camgöbeği, 4=ED lime
LABEL_COLORS = {
    1: (255, 60, 60), 2: (255, 220, 40), 3: (40, 220, 255), 4: (120, 255, 80),
}

# Segmentasyon hacimleri gr.State içinde (deepcopy pahalı) değil, oturum bazlı
# bellek cache'inde tutulur; State yalnızca session_dir anahtarını taşır.
_SEG_CACHE: dict[str, dict] = {}


def _take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = int(idx)
    return np.rot90(vol[tuple(sl)])


def _to_gray_u8(sl: np.ndarray) -> np.ndarray:
    """Kesiti 1-99 persentil ile pencereleyip 0-255 gri tonlamaya çevir."""
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return np.zeros(sl.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    return (g * 255).astype(np.uint8)


def render_slice_rgb(vol: np.ndarray, mask: np.ndarray, plane: str,
                     idx: int, alpha: float = 0.45) -> np.ndarray:
    """PACS benzeri tek-kesit: gri taban + renkli segmentasyon overlay (RGB uint8).

    Saf numpy — matplotlib.pyplot'un thread-güvensiz global state'ini kullanmaz,
    bu yüzden hızlı kaydırıcı hareketlerinde çökmez."""
    axis = PLANE_AXIS[plane]
    idx = max(0, min(int(idx), vol.shape[axis] - 1))
    base = _take_slice(vol, axis, idx)
    m = _take_slice(mask, axis, idx)

    g = _to_gray_u8(base)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
    for lab, color in LABEL_COLORS.items():
        sel = m == lab
        if sel.any():
            rgb[sel] = (1 - alpha) * rgb[sel] + alpha * np.array(color, dtype=np.float32)
    return rgb.astype(np.uint8)


def _slice_info(vol: np.ndarray, plane: str, idx: int) -> str:
    n = vol.shape[PLANE_AXIS[plane]]
    idx = max(0, min(int(idx), n - 1))
    return f"**{plane}** — kesit {idx + 1} / {n}"


def on_plane_change(seg_key, plane):
    """Düzlem değişince kaydırıcı aralığını (kesit sayısı) güncelle + orta kesit."""
    entry = _SEG_CACHE.get(seg_key) if seg_key else None
    if not entry:
        return gr.update(), None, ""
    vol, mask = entry["vol"], entry["mask"]
    n = vol.shape[PLANE_AXIS[plane]]
    mid_idx = n // 2
    img = render_slice_rgb(vol, mask, plane, mid_idx)
    return (gr.update(minimum=0, maximum=n - 1, value=mid_idx, step=1),
            img, _slice_info(vol, plane, mid_idx))


def on_slice_change(seg_key, plane, idx):
    entry = _SEG_CACHE.get(seg_key) if seg_key else None
    if not entry:
        return None, ""
    vol, mask = entry["vol"], entry["mask"]
    img = render_slice_rgb(vol, mask, plane, idx)
    return img, _slice_info(vol, plane, idx)


def run_pipeline(state, ov_flair, ov_t1, ov_t1ce, ov_t2, folds_choice,
                 skip_strip, progress=gr.Progress()):
    if not state:
        return ("Önce 1. adımı (yükle ve dönüştür) tamamlayın.",
                None, None, gr.update(), None, "")

    session_dir = Path(state["session_dir"])
    case_id = state["case_id"]
    known_paths = {s["path"] for s in state["series"]}
    chosen = {"FLAIR": ov_flair, "T1": ov_t1, "T1CE": ov_t1ce, "T2": ov_t2}

    mapping: dict[str, Path] = {}
    for seq, path in chosen.items():
        if path and path in known_paths:
            mapping[seq] = Path(path)
    missing = [s for s in SEQS if s not in mapping]
    if missing:
        return (f"HATA: eksik sekans eşlemesi: {missing}. Yukarıdan tüm sekansları seçin.",
                None, None, gr.update(), None, "")

    try:
        progress(0.05, desc="Ko-registrasyon (T1CE referans)...")
        aligned = reg.coregister_case(mapping, session_dir / "02_aligned", case_id)

        if skip_strip:
            progress(0.4, desc="Skull stripping ATLANDI (hızlı mod) — segmentasyona geçiliyor...")
            stripped, brain_mask = aligned, None
        else:
            progress(0.25, desc="Skull stripping (HD-BET) — Mac CPU'da yavaş, birkaç-15 dk sürebilir...")

            def _hdbet_log(line: str) -> None:
                line = line.strip()
                if line:
                    progress(0.25, desc=f"HD-BET {line[:110]}")

            stripped, brain_mask = ss.skull_strip_case(
                aligned, session_dir / "03_stripped", case_id, "T1CE",
                device=None,  # otomatik: CUDA varsa (Windows/Linux GPU), yoksa CPU
                log_fn=_hdbet_log,
            )

        progress(0.5, desc="Yoğunluk normalizasyonu...")
        final = norm.normalize_case(stripped, session_dir / "04_final", case_id, brain_mask)

        folds = (0,) if folds_choice.startswith("Hızlı") else (0, 1, 2, 3, 4)
        progress(0.6, desc=f"BraTS-PEDs segmentasyon ({len(folds)} kat) — birkaç dakika sürebilir...")
        seg_path = segment_case(
            session_dir / "04_final", session_dir / "05_seg", case_id,
            folds=folds, device=None,  # otomatik: CUDA > MPS > CPU
        )
    except Exception as e:
        return f"HATA: {e}", None, None, gr.update(), None, ""

    progress(0.9, desc="Görselleştirme...")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(final["T1CE"])))

    stats, total = [], 0
    for lab, name in LABELS.items():
        n = int((arr == lab).sum())
        total += n
        stats.append([name, n, round(n / 1000, 2)])  # 1mm izotropik -> mm3/1000 = cm3
    stats.append(["TOPLAM (whole tumor)", total, round(total / 1000, 2)])

    seg_key = str(session_dir)
    _SEG_CACHE[seg_key] = {"vol": vol, "mask": arr}
    slider_update, slice_img, info = on_plane_change(seg_key, "Aksiyel")

    progress(1.0, desc="Tamamlandı")
    return ("✅ Segmentasyon tamamlandı.", stats, seg_key,
            slider_update, slice_img, info)


with gr.Blocks(title="Posterior Fossa Tümörü — Segmentasyon") as demo:
    gr.Markdown(
        "# Pediatrik Posterior Fossa Tümörü — DICOM Segmentasyon\n"
        "DICOM → NIfTI → preprocessing (registrasyon + skull strip + normalizasyon) → "
        "BraTS-PEDs segmentasyonu.\n\n"
        "⚠️ **Sınıflandırma (MB/PA/EP) henüz yok** — model gerçek hasta verisiyle "
        "eğitilmeyi bekliyor. Bu sürüm yalnızca tümör segmentasyonunu gösterir."
    )
    state = gr.State(None)

    with gr.Tab("1. Yükle ve Sekansları Kontrol Et"):
        zip_input = gr.File(
            label="DICOM klasörünü sürükle-bırak (T1, T1CE, T2, FLAIR sekansları) — veya .zip yükle",
            file_count="directory",
        )
        convert_btn = gr.Button("Dönüştür", variant="primary")
        convert_status = gr.Textbox(label="Durum", interactive=False)
        series_table = gr.Dataframe(
            headers=["Seri Açıklaması", "Tahmin Edilen Sekans", "NIfTI Dosyası"],
            interactive=False, label="Bulunan Seriler",
        )
        gr.Markdown("### Sekans eşlemesini kontrol et / düzelt")
        with gr.Row():
            dd_flair = gr.Dropdown(label="FLAIR", choices=[])
            dd_t1 = gr.Dropdown(label="T1", choices=[])
            dd_t1ce = gr.Dropdown(label="T1CE", choices=[])
            dd_t2 = gr.Dropdown(label="T2", choices=[])

        convert_btn.click(
            on_convert, inputs=[zip_input],
            outputs=[convert_status, series_table, state, dd_flair, dd_t1, dd_t1ce, dd_t2],
        )

    with gr.Tab("2. Segmentasyon"):
        seg_state = gr.State(None)
        folds_choice = gr.Radio(
            ["Hızlı (1 kat)", "Tam ensemble (5 kat, daha yavaş)"],
            value="Hızlı (1 kat)", label="Model modu",
        )
        skip_strip_cb = gr.Checkbox(
            value=False,
            label="Skull stripping'i atla (HD-BET) — çok daha hızlı, kaba sonuç",
            info="HD-BET Mac CPU'da 10-25 dk sürebilir. Sadece test/deneme için atla; "
                 "gerçek klinik sonuç için işaretleme (skull strip segmentasyon kalitesini artırır).",
        )
        run_btn = gr.Button("Segmentasyonu Çalıştır", variant="primary")
        run_status = gr.Textbox(label="Durum", interactive=False)
        stats_table = gr.Dataframe(headers=["Etiket", "Voksel", "Hacim (cm³)"], interactive=False)

        gr.Markdown(
            "### Kesit kesit incele (PACS benzeri)\n"
            "🟥 ET (kontrast tutan) · 🟨 NCR (nekroz) · 🟦 CC (kistik) · 🟩 ED (ödem)"
        )
        with gr.Row():
            plane_dd = gr.Radio(list(PLANE_AXIS), value="Aksiyel", label="Düzlem")
        slice_info = gr.Markdown("")
        slice_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="Kesit (kaydır)")
        slice_img = gr.Image(label="Kesit görüntüsü", height=460)

        run_btn.click(
            run_pipeline,
            inputs=[state, dd_flair, dd_t1, dd_t1ce, dd_t2, folds_choice, skip_strip_cb],
            outputs=[run_status, stats_table, seg_state, slice_slider, slice_img, slice_info],
        )
        plane_dd.change(
            on_plane_change, inputs=[seg_state, plane_dd],
            outputs=[slice_slider, slice_img, slice_info],
        )
        slice_slider.change(
            on_slice_change, inputs=[seg_state, plane_dd, slice_slider],
            outputs=[slice_img, slice_info],
        )

    with gr.Tab("3. Sınıflandırma (yakında)"):
        gr.Markdown(
            "MB / PA / EP sınıflandırma modeli henüz eğitilmedi — gerçek hasta verisi "
            "toplanıp Colab'da eğitim yapılması bekleniyor. Model hazır olunca bu "
            "sekme aktif hale gelecek."
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
