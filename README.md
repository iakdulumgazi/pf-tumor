# Pediatrik Posterior Fossa Tümörü Sınıflandırma — Pipeline

Pediatrik posterior fossa tümörlerinin (Medulloblastom / Pilositik Astrositom /
Ependimom) preoperatif MR ile ayrımı. Bkz. `../Downloads/PROJECT_CONTEXT.md`.

## Mimari (iki aşama)

```
DICOM  →  NIfTI  →  preprocessing  →  [1] SEGMENTASYON  →  tümör maskesi
                                          (BraTS-PEDs, hazır)        │
                                                                     ▼
                                      [2] SINIFLANDIRMA  ←  tümör bölgesini kırp
                                      (kendi verimizle eğitilir)
                                          → MB / PA / EP
```

- **BraTS-PEDs** yalnızca *segmentasyon* yapar (tümörü bulur). Tümör tipini söylemez.
- **Sınıflandırıcı** ayrı bir 3D CNN'dir; kendi verimizle (Gazi, 50-100 hasta)
  transfer learning ile fine-tune edilir. Eğitim Colab/TRUBA'da, inference yerelde.

## Durum

| Aşama | Durum |
|-------|-------|
| DICOM → NIfTI + sekans tanıma (T1/T1CE/T2/FLAIR) | ✅ çalışıyor |
| Ko-registrasyon (T1CE ref) + resample 1mm | ✅ çalışıyor |
| Skull stripping (HD-BET v1, CPU) | ✅ çalışıyor (~7 dk/vaka CPU) |
| Z-score normalizasyon | ✅ çalışıyor |
| BraTS-PEDs segmentasyon (inference) | ✅ çalışıyor (model kurulu; CPU yavaş → GPU önerilir) |
| Tümör ROI kırpma (segmentasyon → sınıflandırıcı girdisi) | ✅ çalışıyor |
| Sınıflandırıcı eğitimi (Colab notebook) | ✅ hazır, gerçek veri bekliyor |
| Arayüz (Gradio) | ✅ çalışıyor — yalnızca segmentasyon (sınıflandırma henüz yok) |

### ROI kırpma kullanımı
```bash
python -m src.classification.roi_crop \
    --image-dir <hizalı _0000.._0003> --mask <seg_maskesi> \
    --out-dir <OUT> --case-id <ID> --margin-mm 10 --size 96 96 96
```
Çıktı: `<case>_roi.npy` (4, 96, 96, 96) — sınıflandırıcıya doğrudan girdi.

### Sınıflandırıcı (MONAI 3D DenseNet121)
```bash
python -m src.classification.train \
    --roi-dir <ROI klasörü> --labels labels.csv --out-dir runs/exp1 \
    --folds 5 --epochs 100 --device mps   # GPU: cuda, Colab'da da aynı kod
```
`labels.csv` formatı: `case_id,patient_id,label` (label ∈ MB/PA/EP).
5-fold **hasta-bazlı** stratified CV (leakage koruması), sınıf-ağırlıklı loss,
agresif augmentation, early stopping (macro-F1), çıktı `cv_summary.json` +
her fold için `foldN_best.pth`. `--pretrained <ckpt>` ile transfer learning.

Sentetik veriyle (18 sahte hasta, ayırt edici sinyal) uçtan uca doğrulandı:
kod çalışıyor, hasta-bazlı bölme leakage'a izin vermiyor, model gerçekten
öğrenebiliyor (bir fold'da macroF1=1.0'a ulaştı). Gerçek performans gerçek
veriyle (Colab, GPU) ölçülecek.

### Colab'da eğitim
`notebooks/train_classifier.ipynb` — yereldeki `src/classification/` kodunun
aynısını GPU üzerinde çalıştırır. Preprocessing/segmentasyon/ROI kırpma
**yerelde** yapılır; Colab yalnızca eğitim için kullanılır. Kullanım:
1. `src/classification/`, `data/roi/*.npy` ve `labels.csv`'yi Google Drive'da
   `MyDrive/pf-tumor/` altına yükle (yapı notebook'ta anlatılıyor).
2. Notebook'u Colab'da aç, Runtime → GPU seç, hücreleri sırayla çalıştır.
3. Checkpoint'ler (`foldN_best.pth`) Drive'a kaydedilir; inference için yerele indir.

⚠️ ROI'ler hasta verisinden türediği için Drive'a yüklemeden önce
anonimleştirme + kurumsal onay kontrolü yapılmalı.

## Windows'ta kurulum (başka bilgisayarda deneme)

Uzun ama tek-seferlik. NVIDIA GPU'lu bir Windows makine varsa **çok daha hızlı** (HD-BET
saniyeler, segmentasyon dakikalar); CPU-only Windows'ta Mac'teki gibi yavaş olur —
o durumda arayüzdeki **"Skull stripping'i atla"** kutucuğunu işaretle.

### Ön gereksinimler
1. **Python 3.11** — [python.org](https://www.python.org/downloads/windows/) (kurulumda "Add python.exe to PATH" işaretle)
2. **Git for Windows** — [git-scm.com](https://git-scm.com/download/win)
3. **dcm2niix (DICOM→NIfTI çevirici)** — https://github.com/rordenlab/dcm2niix/releases sayfasından `dcm2niix_win.zip` indir, `dcm2niix.exe`'yi PATH'e eklediğin bir klasöre koy (ör. `C:\Tools\`). Denetle: `dcm2niix --version`
4. **NVIDIA GPU (opsiyonel ama önerilir):** Güncel NVIDIA driver yeterli — PyTorch kendi CUDA runtime'ını getirir.
   - ⚠️ **RTX 50 serisi (5070/5080/5090):** Blackwell mimarisi (sm_120). PyTorch'un **CUDA 12.8 (cu128)** build'i ŞART; eski cu121 build'i "no kernel image available" hatası verir. Aşağıdaki kurulumda cu128 kullanılıyor.
   - RTX 40/30 serisi: cu121 veya cu128, ikisi de çalışır.
   - GPU yoksa CPU'da çalışır (yavaş).

### Projeyi taşı
Bu Mac'te git repo yok, o yüzden en pratik: `~/pf-tumor` klasörünü zip'leyip Windows'a
kopyala. **Bunları hariç tut** (büyük ve gerektiğinde yeniden üretilebilir):
- `.venv/` (Windows'ta yeniden kurulacak)
- `data/` (hasta verisi — kesinlikle gitmez)
- `models/braTS_peds_repo/data/` (2.2 GB pretrained ağırlıklar — Windows'ta tekrar indireceğiz)
- `models/braTS_peds_repo/pretrained/` (2.2 GB indirilen zip'ler)

Kalan kod ~50-100 MB olur.

### Windows'ta ortam kur (PowerShell)
Kopyaladığın klasöre gir, sonra sırayla:

```powershell
cd C:\Users\<sen>\pf-tumor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Ana bağımlılıklar
pip install nibabel numpy pydicom SimpleITK matplotlib gradio HD-BET gdown
pip install "acvl-utils>=0.2,<0.3" "dynamic-network-architectures>=0.3.1,<0.4" `
    tqdm scipy "batchgenerators>=0.25" scikit-learn "scikit-image>=0.19.3" `
    pandas tifffile requests seaborn imagecodecs yacs `
    "batchgeneratorsv2>=0.2" einops graphviz blosc2

# PyTorch — GPU için (ekran kartına göre CUDA sürümü DEĞİŞİR, aşağıya bak):
#   RTX 50 serisi (5070/5080/5090, Blackwell/sm_120) -> CUDA 12.8 ŞART:
pip install torch --index-url https://download.pytorch.org/whl/cu128
#   RTX 40/30 serisi (daha eski) için cu121 de yeter:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   CPU-only (GPU yok) için sadece:
#   pip install torch

# nnU-Net fork (bundled, deps hariç)
pip install -e .\models\braTS_peds_repo\nnUNet --no-deps
```

### BraTS-PEDs ağırlıklarını indir
Google Drive'dan iki zip (~1.1 GB her biri):

```powershell
mkdir models\braTS_peds_repo\pretrained
cd models\braTS_peds_repo
gdown --folder "https://drive.google.com/drive/folders/1dNwbkVnyTGoGZfr4tD10Mibv5jCopp2d" -O .\pretrained
mkdir data\nnUNet_raw, data\nnUNet_preprocessed, data\nnUNet_results
$env:nnUNet_raw = "$PWD\data\nnUNet_raw"
$env:nnUNet_preprocessed = "$PWD\data\nnUNet_preprocessed"
$env:nnUNet_results = "$PWD\data\nnUNet_results"
nnUNetv2_install_pretrained_model_from_zip .\pretrained\WT.zip
nnUNetv2_install_pretrained_model_from_zip .\pretrained\3L.zip
cd ..\..
```

### Arayüzü başlat
```powershell
python src\ui\app.py
```
Tarayıcıda **http://127.0.0.1:7860** aç.

### Notlar
- **GPU otomatik seçilir** — cihaz seçimini kod yapıyor (CUDA varsa `cuda`, yoksa `cpu`).
  MPS Mac'e özgü, Windows'ta rol oynamaz.
- **Ağdan uzaktan erişim:** Sadece bu bilgisayardan değil, aynı ağdaki başka
  cihazlardan da açmak istersen `src\ui\app.py`'nin en son satırında
  `server_name="127.0.0.1"` → `server_name="0.0.0.0"` yap ve Windows Güvenlik
  Duvarı'nda 7860 portunu aç.
- **Hasta verisi:** Windows'a hasta DICOM'ları kopyalamadan önce mutlaka
  anonimleştirme + kurumsal veri politikası kontrolü.

## Arayüz (Gradio)

```bash
.venv/bin/python src/ui/app.py     # http://127.0.0.1:7860
```

3 sekme: **(1)** DICOM klasörünü sürükle-bırak (veya .zip yükle) → otomatik
sekans tanıma, elle düzeltme dropdown'ları · **(2)** preprocessing + BraTS-PEDs
segmentasyonu çalıştır → **PACS benzeri kesit kesit görüntüleyici** (düzlem
seç: Aksiyel/Koronal/Sagital + kaydırıcıyla kesit kesit gez, renkli overlay)
+ etiket bazlı hacim tablosu (cm³) · **(3)** sınıflandırma — henüz devre dışı,
gerçek model eğitilene kadar bilgilendirme mesajı gösteriyor.

Yükleme hem klasör sürükle-bırak hem `.zip` kabul eder (dcm2niix serileri
DICOM etiketlerinden tanır, klasör yapısından değil — düz kopyalama sorun
çıkarmaz).

"Hızlı (1 kat)" / "Tam ensemble (5 kat)" seçeneği var; tam ensemble daha
doğru ama çok daha yavaş (Mac'te dakikalar sürer). Sentetik DICOM ile uçtan
uca doğrulandı (yükleme→dönüşüm→segmentasyon→kesit görüntüleyici, ~50-90 sn
küçük hacimde).

### Segmentasyon kullanımı
```bash
python -m src.segmentation.run_brats_peds \
    --input-dir <skull-stripped _0000.._0003> --output-dir <OUT> \
    --case-id <ID> --folds 0 1 2 3 4 --device cpu   # GPU varsa --device cuda
```
Modeller: `models/braTS_peds_repo/data/nnUNet_results/` (Dataset106 WT + Dataset107 3L, 5 fold).
Nihai maske etiketleri: 1=ET, 2=NCR, 3=CC, 4=ED. **Not:** 5-fold×2-model ensemble
Mac CPU'da çok yavaş (>10 dk/vaka); batch için GPU (TRUBA/Colab) gerekir.

Tam pipeline: `python -m src.preprocessing.pipeline --dicom-dir <DICOM> --case-id <ID> --work-dir <OUT>`
(HD-BET kurulu değilse `--skip-skull-strip`).

## Kurulum

```bash
brew install dcm2niix
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Kullanım — DICOM → NIfTI

```bash
python -m src.preprocessing.dicom_to_nifti \
    --dicom-dir data/raw/HASTA001 \
    --out-dir   data/nifti/HASTA001 \
    --case-id   HASTA001 \
    --nnunet-dir data/preprocessed/HASTA001_nnunet
```

Sekans otomatik tanınır (Türkçe/İngilizce PACS açıklamaları desteklenir).
Yanlış tanınırsa elle düzelt:

```bash
    --override T1CE=3_T1_post --override FLAIR=1_dark_fluid
```

Çıktı nnU-Net konvansiyonu: `_0000`=FLAIR, `_0001`=T1, `_0002`=T1CE, `_0003`=T2.

## Test (sentetik veri)

Gerçek hasta verisi olmadan denemek için:

```bash
python -m src.preprocessing._make_synthetic_dicom data/raw/SYN001
python -m src.preprocessing.dicom_to_nifti --dicom-dir data/raw/SYN001 \
    --out-dir data/nifti/SYN001 --case-id SYN001 \
    --nnunet-dir data/preprocessed/SYN001_nnunet
```

## Dizin yapısı

```
src/preprocessing/   DICOM→NIfTI, skull strip, registrasyon, normalizasyon
src/segmentation/    BraTS-PEDs inference sarmalayıcısı
src/classification/  sınıflandırıcı (eğitim Colab'da, inference burada)
data/raw/            DICOM (anonim) — git'e girmez
data/nifti/          ham NIfTI
data/preprocessed/   nnU-Net'e hazır + işlenmiş hacimler
data/masks/          segmentasyon maskeleri
models/              indirilen/eğitilen ağırlıklar — git'e girmez
notebooks/           Colab eğitim notebook'ları
```

## Notlar

- **Veri gizliliği:** `data/` ve `models/` git'e girmez (`.gitignore`). DICOM'lar
  anonimleştirilmiş olmalı.
- **Donanım:** inference Mac (Apple Silicon, CPU/MPS) üzerinde çalışır; eğitim için
  GPU (Colab/TRUBA) gerekir. AMD RX 6650 XT CUDA-DL için uygun değil.
