# Biwi Dataset - Kurulum ve Dosya Yapısı

## Dataset'i Nereye Yüklemeli?

Biwi dataset'ini **nereye indirdiğiniz önemli değil**. Önemli olan, script'in doğru path'i bulabilmesi.

**Önerilen yapı:**

```
face_reconstruction/              # Proje ana dizini
├── data/
│   ├── biwi_raw/                # ⬅️ Dataset'i buraya kopyalayın (opsiyonel)
│   │   └── [biwi dataset files]
│   ├── biwi_test/               # ⬅️ Dönüştürülmüş veriler buraya gelecek
│   ├── model/                   # PCA model dosyaları
│   └── test/                    # Diğer test verileri
└── ...
```

**Alternatif:** Dataset'i başka bir yere koyabilirsiniz, sadece `--input` parametresinde path'i belirtmeniz yeterli.

## Hangi Dosyalar Gerekli?

Biwi dataset'inin farklı versiyonları olabilir. İhtiyacınız olan **asgari dosyalar**:

### ✅ Zorunlu Dosyalar

1. **RGB görüntüler**
   - Format: PNG, JPEG
   - Çözünürlük: Genellikle 640x480
   - İsim formatı: `*.png`, `rgb*.png`, `*_rgb.png`, vs.

2. **Depth görüntüler**
   - Format: PNG (16-bit), binary, veya compressed binary
   - Çözünürlük: Genellikle 640x480
   - İsim formatı: `depth*.png`, `*_depth.png`, vs.

### 📋 Opsiyonel Dosyalar

3. **Camera intrinsics** (opsiyonel - script otomatik ayarlar)
4. **Pose annotations** (opsiyonel - şimdilik kullanmıyoruz)
5. **README/readme.txt** (dataset hakkında bilgi)

## Dataset Yapısı - Örnekler

### Yapı 1: Basit Klasör Yapısı

```
biwi_dataset/
├── rgb/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── depth/
    ├── 0000.png
    ├── 0001.png
    └── ...
```

**Kullanım:**
```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi_dataset \
    --output data/biwi_test
```

### Yapı 2: Kişi Bazlı Klasörler

```
biwi_dataset/
├── person01/
│   ├── rgb_0000.png
│   ├── depth_0000.png
│   ├── rgb_0001.png
│   ├── depth_0001.png
│   └── ...
├── person02/
│   └── ...
└── person03/
    └── ...
```

**Kullanım (bir kişi için):**
```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi_dataset/person01 \
    --output data/biwi_test
```

### Yapı 3: Karma Dosyalar

```
biwi_dataset/
├── frame_0000_rgb.png
├── frame_0000_depth.png
├── frame_0001_rgb.png
├── frame_0001_depth.png
└── ...
```

**Kullanım:**
```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi_dataset \
    --output data/biwi_test
```

## Adım Adım Kurulum

### 1. Dataset'i İndirin

Biwi dataset'ini resmi kaynaktan indirin. Genellikle şu formatta gelir:
- ZIP/TAR dosyası
- İçinde RGB ve depth klasörleri veya dosyaları

### 2. Dataset'i Açın

**Önemli:** Dataset'i **nereye koyduğunuz önemli değil**, önemli olan path'i doğru belirtmek.

**Önerilen:**
```bash
# Dataset'i proje klasörüne kopyalayın (opsiyonel)
cd /path/to/face_reconstruction
mkdir -p data/biwi_raw
# Dataset'i data/biwi_raw/ klasörüne çıkarın
```

**VEYA** dataset'i başka bir yere koyabilirsiniz (örn: `~/Downloads/biwi_dataset`)

### 3. Dataset Yapısını Kontrol Edin

**Otomatik kontrol (önerilen):**
```bash
python scripts/check_biwi_dataset.py /path/to/biwi_dataset
```

Bu script size şunları söyleyecek:
- ✅ Kaç RGB dosyası bulundu
- ✅ Kaç depth dosyası bulundu
- ✅ Hangi klasörlerde bulundu
- ✅ Hangi komutu kullanmanız gerektiği

**Manuel kontrol:**
```bash
# Dataset'inizin içinde ne var bakalım
ls -la /path/to/biwi_dataset/
# veya
ls -la data/biwi_raw/
```

**Beklenen çıktı örnekleri:**
```bash
# Yapı 1: RGB ve depth klasörleri görürsünüz
rgb/  depth/  README.txt

# Yapı 2: Kişi klasörleri görürsünüz
person01/  person02/  person03/  ...

# Yapı 3: Karma dosyalar görürsünüz
frame_0000_rgb.png  frame_0000_depth.png  ...
```

### 4. Test: RGB ve Depth Dosyalarını Kontrol Edin

```bash
# RGB dosyalarını listeleyin
find /path/to/biwi_dataset -name "*rgb*" -o -name "*RGB*" | head -5

# Depth dosyalarını listeleyin
find /path/to/biwi_dataset -name "*depth*" -o -name "*Depth*" | head -5
```

### 5. Dönüştürme Script'ini Çalıştırın

```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi_dataset \
    --output data/biwi_test \
    --max-frames 5  # İlk 5 frame ile test edin
```

## Hangi Dosyaları Yüklemeliyim? - Özet

### ✅ Yüklemeniz Gerekenler:

1. **RGB görüntü dosyaları** (PNG/JPEG)
2. **Depth görüntü dosyaları** (PNG/binary)

### ❌ Yüklememeniz Gerekenler (opsiyonel):

- README dosyaları (bilgi amaçlı)
- Pose annotation dosyaları (şimdilik gerekli değil)
- Source code dosyaları
- Diğer metadata dosyaları

## Örnek Komutlar

### Senaryo 1: Dataset Downloads klasöründe

```bash
# Dataset Downloads klasöründe olsun
python scripts/convert_biwi_dataset.py \
    --input ~/Downloads/biwi_dataset \
    --output data/biwi_test
```

### Senaryo 2: Dataset proje içinde

```bash
# Dataset'i proje içine kopyaladıysanız
python scripts/convert_biwi_dataset.py \
    --input data/biwi_raw \
    --output data/biwi_test
```

### Senaryo 3: Belirli bir kişi/sequence

```bash
# Sadece bir kişiyi işlemek istiyorsanız
python scripts/convert_biwi_dataset.py \
    --input data/biwi_raw/person01 \
    --output data/biwi_test_person01
```

## Sorun Giderme

### "No RGB-depth pairs found" Hatası

**Kontrol edin:**
1. Path doğru mu?
   ```bash
   ls /path/to/biwi_dataset/
   ```
2. RGB dosyaları var mı?
   ```bash
   find /path/to/biwi_dataset -name "*.png" | grep -i rgb
   ```
3. Depth dosyaları var mı?
   ```bash
   find /path/to/biwi_dataset -name "*.png" | grep -i depth
   ```

**Çözüm:** Farklı bir alt klasörü deneyin veya dataset yapısını manuel kontrol edin.

### Dataset Formatı Hakkında Bilgi

Eğer dataset'inizin yapısını bilmiyorsanız:

```bash
# Dataset yapısını görmek için
tree -L 2 /path/to/biwi_dataset  # tree yoksa:
find /path/to/biwi_dataset -type f | head -20
find /path/to/biwi_dataset -type d
```

Bu bilgileri paylaşırsanız, size özel komut hazırlayabilirim!
