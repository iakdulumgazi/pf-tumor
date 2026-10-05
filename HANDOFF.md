# pf-tumor — Proje Handoff (Windows'a Geçiş)

> Pediatrik posterior fossa tümörü (MB / PA / EP) preoperatif MR sınıflandırma projesi.
> Bu dosya, projeyi **başka bir bilgisayarda (Windows + RTX 5070)** kaldığı yerden
> sürdürmek için her şeyi tek yerde toplar: durum, strateji, kod, kurulum, bekleyen işler.
> Son güncelleme: 2026-10-05 · GitHub: **github.com/iakdulumgazi/pf-tumor** (public, sadece kod)

---

## 0. TL;DR — Şu an neredeyiz

- **Kod tarafı bitti ve doğrulandı** (sentetik/örnek veriyle): DICOM→NIfTI→preprocessing→
  BraTS-PEDs segmentasyon→ROI kırpma→3D DenseNet sınıflandırıcı→Gradio arayüz→Colab notebook.
- **Gerçek veri: 41 hasta** (MB 18, PA 13, EP 10). Deep learning için **çok az → pilot.**
- **CBTN engellendi:** başvuru onaylandı ama DUA kurumsal imza + tazminat istiyor; Gazi sıcak
  bakmıyor → CBTN'siz ilerleme planı.
- **Yeni ana strateji:** sıfırdan eğitim yerine, **yayınlanmış bir modelin ağırlıklarını alıp**
  (Tampu veya Quon) kendi 41 hastamızla **dış validasyon + fine-tune** (domain-matched transfer
  learning — 41 hasta için en güçlü yol). Ağırlıklar için yazarlara mail atılacak.
- **Kritik bulgu (Tampu):** en iyi ayrım **ADC + yaş** ile (%87, AUC 0.94); T2 tek %73.
  → ADC (DWI) kritik, yaş özellik olarak eklenecek.
- **Bekleyen:** Tampu'ya mail (taslak hazır), Quon'a mail (taslak yapılacak), etik kapsam
  kontrolü (taslak hazır), 41 hastanın ADC'si var mı teyidi.

---

## 1. Bilimsel hedef ve tasarım

- **Amaç:** Medulloblastom (MB), Pilositik Astrositom (PA), Ependimom (EP) preoperatif MR ayrımı.
- **Tip:** Tek merkez (Gazi), retrospektif, girişimsel olmayan, etik onaylı.
- **Hariç:** DIPG/beyin sapı gliomu, ATRT, metastaz (bilinçli dışlandı).
- **Zorunlu sekanslar:** T1, T1CE, T2, FLAIR (BraTS standardı). **ADC (DWI) kritik** (Tampu bulgusu).
- **Raporlama:** CLAIM checklist. Radyolog vs model karşılaştırması planlı.
- **Bağlam dosyası:** `PROJECT_CONTEXT.md` (orijinal planlama; bu handoff onu günceller).

## 2. Gerçek veri durumu

| Tip | Hasta |
|-----|-------|
| Medulloblastom (MB) | 18 |
| Pilositik Astrositom (PA) | 13 |
| Ependimom (EP) | 10 |
| **Toplam** | **41** |

- 41, sıfırdan 3D deep learning için **çok az** → overfit riski yüksek → **pilot çalışma.**
- EP (10) üç-sınıf hasta-bazlı CV için **tehlikeli az.** → Başta **2 sınıf (MB vs PA, 31 hasta)**
  ile başlamak, EP'yi sonra eklemek öneriliyor.
- **AÇIK SORU:** 41 hastanın **ADC (DWI)** verisi var mı? Hangi model yolunu seçeceğimizi bu belirler.

## 3. Strateji — CBTN engeli ve çıkış yolu

**Durum:** CBTN başvurusu içerik olarak onaylandı, ama veri erişimi için **DUA** gerekiyor;
DUA Gazi'nin kurumsal imzasını + tazminat (indemnification) taahhüdünü istiyor ("as-is",
değiştirilemez). Hastane bu tür anlaşmalara sıcak bakmıyor → **CBTN büyük ihtimalle kapalı.**

**Çıkış yolu (en güçlüden):**

1. **Yayınlanmış modelin ağırlığını al → fine-tune.** CBTN'le eğitilmiş bir MB/PA/EP modelinin
   ağırlıklarını yazarlardan alırsak, 41 hastayla fine-tune **sıfırdan eğitimden çok daha iyi**
   (domain-matched). Üstelik **ham veri değil model aldığımız için CBTN DUA'sına gerek kalmaz.**
2. **Kendi veri + ImageNet/MedicalNet transfer + 2D kesit-bazlı + ADC+yaş.** Ağırlık gelmezse.
3. **Türkiye içi çok-merkez** (KVKK, aynı ülke → ABD tazminat maddesinden kolay).

**Neden 41 için 2D + transfer:** 2D kesit-bazlı her hastadan onlarca örnek çıkarır (41 hasta →
yüzlerce kesit); transfer learning küçük-veri sorununu çözer. Multi-parametrik fusion (ResSwinT
tarzı) büyük veride iyi ama 41 hastada overfit'i artırır. Detay: **`docs/veri-stratejisi.md`**.

### Aday modeller (ağırlık için mail atılacak)

| Model | Veri | Sekans | Perf. | Kod | Ağırlık |
|-------|------|--------|-------|-----|---------|
| **Tampu** (Linköping, Neuro-Oncol Adv 2025) | CBTN ~244 | **ADC+yaş** (en iyi); T2/T1CE de | %87, AUC 0.94 | ✅ açık (CC BY-NC-SA 4.0) | ❌ paylaşılmadı → mail |
| **Quon** (AJNR 2020, Stanford) | 617, 5 kurum | **T2-only** | ~%92 | ❌ | ❌ → mail |

- **Tampu kodu:** github.com/IulianEmilTampu/PediatricBrainTumorClassification-MRI
  (CC BY-NC-SA: akademik kullanım serbest, atıf + türevi aynı lisansla aç; **ticari YASAK** —
  SaMD/ürünleştirme düşünülmüyor, sorun değil). Lisans ağırlık paylaşımını **engellemez**.
- **Quon avantajı:** T2-only → ADC'n yoksa her hastada çalışır. Dezavantaj: kod/ağırlık açık değil,
  paylaşıma daha az yatkın olabilir.
- **Öneri:** İkisine de paralel mail at; ADC yoksa Quon öncelik, varsa Tampu.

## 4. Bekleyen aksiyonlar (sıradaki işler)

1. **Tampu'ya mail** — taslak hazır: `~/Downloads/mail-tampu-model-talebi.md`.
   Adres: **iulian.emil.tampu@liu.se** (CC: Neda Haj Hosseini, Peter Lundberg — LiU).
   Çerçeve: dış validasyon + fine-tune + olası ortak yazarlık (katkıyla orantılı).
2. **Quon'a mail** — taslak HENÜZ YAPILMADI. Corresponding author muhtemelen **Kristen W. Yeom
   (Stanford)**. Tampu mailini T2/multi-institutional vurgusuyla uyarla.
3. **Etik kapsam kontrolü** — mevcut onay dış veri (CBTN) + dış model + AI'yı kapsıyor mu?
   Kapsamıyorsa ek/revizyon başvurusu. Özet taslağı: `~/Downloads/etik-ozet-taslak.md`.
4. **ADC teyidi** — 41 hastanın DWI/ADC'si var mı? (model seçimini belirler).
5. **Sınıf kararı** — 2 sınıf (MB vs PA) ile başla, sonra EP ekle.

> ⚠️ `~/Downloads/` altındaki 3 dosya (mail-tampu-model-talebi.md, etik-ozet-taslak.md,
> veri-stratejisi.md kopyası) repoda **değil** — Windows'a ayrıca kopyala.
> (`docs/veri-stratejisi.md` repoda var.)

## 5. Kurulan kod — ne çalışıyor

Hepsi sentetik/örnek veriyle **uçtan uca doğrulandı**; gerçek hasta verisi bekliyor.

| Aşama | Dosya | Durum |
|-------|-------|-------|
| DICOM→NIfTI + sekans tanıma (TR/EN PACS) | `src/preprocessing/dicom_to_nifti.py` | ✅ |
| Ko-registrasyon (T1CE ref) + 1mm resample | `src/preprocessing/registration.py` | ✅ |
| Skull strip (HD-BET) | `src/preprocessing/skull_strip.py` | ✅ |
| Z-score normalizasyon | `src/preprocessing/normalize.py` | ✅ |
| Tam preprocessing orkestrasyon | `src/preprocessing/pipeline.py` | ✅ |
| BraTS-PEDs segmentasyon (nnU-Net WT+3L → 4-etiket) | `src/segmentation/run_brats_peds.py` | ✅ |
| Tümör ROI kırpma (maske→bbox→sabit küp .npy) | `src/classification/roi_crop.py` | ✅ |
| Sınıflandırıcı (MONAI 3D DenseNet121) | `src/classification/{model,dataset,train}.py` | ✅ kod |
| Gradio arayüz (3 sekme, PACS kesit görüntüleyici) | `src/ui/app.py` | ✅ |
| Colab eğitim notebook (GitHub'dan clone, demo+gerçek) | `notebooks/train_classifier.ipynb` | ✅ T4'te test edildi |

**Önemli mimari notu:** BraTS-PEDs **sadece segmentasyon** yapar (tümörü bulur), **tip söylemez.**
Rolü: tümörü lokalize edip ROI kırpmak. Asıl sınıflandırıcı ayrı. Not: 41 hasta + CBTN engeli
nedeniyle **asıl plan artık 3D DenseNet'i sıfırdan eğitmek DEĞİL**, Tampu/Quon ağırlığından
transfer (muhtemelen 2D). Mevcut 3D DenseNet kodu **fallback** olarak duruyor.

### Sınıflandırıcı özellikleri (mevcut kod)
- Hasta-bazlı `StratifiedGroupKFold` (data-leakage assert'li), sınıf-ağırlıklı loss,
  agresif augmentation (flip/affine/noise), early stopping (macro-F1).
- Metrikler: accuracy, macro-F1, AUC-ovr, confusion matrix. `--pretrained <ckpt>` ile transfer.
- `labels.csv` formatı: `case_id,patient_id,label` (label ∈ MB/PA/EP).

## 6. Repo ve dosya haritası

```
pf-tumor/
├── README.md                 # kurulum + kullanım (Windows bölümü dahil)
├── HANDOFF.md                # BU DOSYA
├── requirements.txt
├── PROJECT_CONTEXT.md        # orijinal planlama (bağlam)
├── docs/veri-stratejisi.md   # CBTN'siz strateji (5 çıkış yolu)
├── src/
│   ├── preprocessing/        # dicom_to_nifti, registration, skull_strip, normalize, pipeline
│   ├── segmentation/         # run_brats_peds
│   ├── classification/       # model, dataset, train, roi_crop
│   └── ui/                   # app.py (Gradio)
├── notebooks/                # train_classifier.ipynb (Colab)
├── data/      (gitignore — hasta verisi, ASLA commit edilmez)
└── models/    (gitignore — 2.2GB BraTS-PEDs ağırlıkları, Windows'ta tekrar indirilecek)
```

- `data/`, `models/`, `.venv/` **git'e girmez** (`.gitignore`). Repoda sadece kod (~21 dosya).
- **Transfer yöntemi:** Windows'ta `git clone https://github.com/iakdulumgazi/pf-tumor` → kod gelir.
  Modeller `gdown` ile tekrar indirilir (aşağıda). Hasta verisi ayrı/güvenli taşınır.

## 7. Ortam

**Mac (mevcut):** `.venv` Python 3.11 · torch 2.12 · nnunetv2 2.5.2 (bundled fork, `--no-deps`) ·
HD-BET 2.0.1 · SimpleITK · MONAI 1.6 · gradio 6 · gdown · dcm2niix (brew).

**Windows (hedef) + RTX 5070:**
- ⚠️ **RTX 50 serisi = Blackwell (sm_120) → PyTorch cu128 ŞART:**
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (Eski cu121 "no kernel image available" hatası verir. RTX 40/30 için cu121 de olur.)
- RTX 5070 12GB VRAM → inference **ve** eğitim için fazlasıyla yeter. GPU'da HD-BET saniyeler,
  segmentasyon dakikalar (Mac CPU'daki 10-25 dk çilesi biter).
- Cihaz seçimi otomatik (`cuda` varsa cuda, yoksa cpu). MPS Mac'e özgü, Windows'ta rol oynamaz.

### Windows kurulum özeti (tam adımlar README "Windows'ta kurulum" bölümünde)
1. Python 3.11, Git for Windows, **dcm2niix.exe** (PATH'e), güncel NVIDIA driver.
2. `git clone` → `py -3.11 -m venv .venv` → aktive et.
3. `pip install nibabel numpy pydicom SimpleITK matplotlib gradio HD-BET gdown` + nnU-Net deps
   (acvl-utils, dynamic-network-architectures, batchgenerators(v2), scikit-*, blosc2, einops...).
4. **torch cu128** (yukarıda).
5. `pip install -e .\models\braTS_peds_repo\nnUNet --no-deps` (bundled fork).
6. BraTS-PEDs ağırlıkları: `gdown --folder "https://drive.google.com/drive/folders/1dNwbkVnyTGoGZfr4tD10Mibv5jCopp2d"`
   → `nnUNetv2_install_pretrained_model_from_zip WT.zip` ve `3L.zip` (nnUNet_results env var'ları set edilmiş halde).
7. `python src\ui\app.py` → http://127.0.0.1:7860

## 8. Önemli teknik notlar / tuzaklar

- **Python 3.9 KULLANMA** — 3.9'da blosc2/numpy ABI çakışması çıkmıştı. **3.11** kullan.
- **nnU-Net bundled fork** `--no-deps` ile kurulur (`dicom2nifti`→`python-gdcm` derlenmiyor; gerekmez).
- **HD-BET v2 CLI** v1'den farklı (`--disable_tta`, `--save_bet_mask`, `-device cuda/cpu/mps`).
  pip sürüm pinlenmedi → ileride CLI yine değişebilir, `skull_strip.py`'yi kontrol et.
- **HD-BET deadlock düzeltmesi (UI):** nnU-Net multiprocessing worker'ları PIPE'ı miras alıp EOF'u
  engelliyordu → çıktı geçici dosyaya yönlendiriliyor, canlı log oradan okunuyor. (`run_hdbet`)
- **Gradio kesit render'ı SAF NUMPY RGB** (`render_slice_rgb`), matplotlib.pyplot DEĞİL —
  pyplot global state'i thread havuzunda çöküyordu. Hacimler `_SEG_CACHE`'te (gr.State değil).
- **BraTS-PEDs girdisi:** skull-stripped, z-score'SUZ veriliyor (nnU-Net içeride 1mm resample +
  maske-içi z-score yapıyor). Format: `_0000=FLAIR, _0001=T1, _0002=T1CE, _0003=T2`.
- **Gradio arayüzü:** "Skull stripping'i atla" kutucuğu var (CPU'da HD-BET çok yavaş olduğunda
  test/demo için). Gerçek klinik sonuçta işaretlenmeli.
- **Veri gizliliği:** Hasta DICOM'ları başka makineye kopyalamadan önce anonimleştirme + kurumsal
  politika kontrolü. CBTN/dış veri Colab'a KONMAZ.

## 9. Dış kaynaklar

- **BraTS-PEDs seg modeli:** github.com/NUBagciLab/Pediatric-Brain-Tumor-Segmentation-Model
  (nnU-Net; ağırlık Google Drive folder 1dNwbkVnyTGoGZfr4tD10Mibv5jCopp2d; Dataset106 WT + 107 3L, 5 fold).
- **Tampu (sınıflandırma, ADC+yaş):** Neuro-Oncology Advances 2025;7(1):vdae205 ·
  DOI 10.1093/noa/vdae205 · kod: github.com/IulianEmilTampu/PediatricBrainTumorClassification-MRI
- **Quon (sınıflandırma, T2):** AJNR 2020;41(9):1718 · DOI 10.3174/ajnr.A6704
- **CBTN/Cavatica:** cbtn.org · Cavatica (genomik platform — hazır imaging modeli YOK).
- **Colab notebook (GitHub'dan):** colab.research.google.com/github/iakdulumgazi/pf-tumor/blob/main/notebooks/train_classifier.ipynb

## 10. Sıradaki oturumda nereden devam

1. **ADC var mı?** → model yolunu seç (Tampu ADC / Quon T2).
2. **Mailleri gönder** (Tampu hazır; Quon taslağı yapılacak).
3. **Etik kapsamı** netleştir.
4. Ağırlık gelince: **dış validasyon → fine-tune** (2 sınıfla başla). Gelmezse: 2D + ImageNet transfer.
5. Gerçek veriyle pipeline'ı koştur → ROI üret → eğit → CLAIM raporla.

---
*Bu proje Claude Code oturumlarıyla geliştirildi. Kod tamamen çalışır; darboğaz veri/erişim (CBTN
DUA) ve yayınlanmış model ağırlıkları. Mac → Windows geçişinde yeni repo açma; mevcut public repoyu
klonla, yolları Windows'a uyarla.*
