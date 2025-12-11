# Biwi Dataset - Çalışan Kılavuz ✅

Dataset'iniz `/Users/atakansucu/Downloads/archive` klasöründe. İşte çalışan adımlar:

## ✅ Çalışan Adımlar

### 1. Dataset'i Dönüştürün

```bash
cd /path/to/face_reconstruction

# Bir kişiyi dönüştür (örnek: kişi 01)
python scripts/convert_biwi_dataset.py \
    --input /Users/atakansucu/Downloads/archive/faces_0/01 \
    --output data/biwi_person01 \
    --max-frames 10
```

**Sonuç:**
- ✅ RGB dosyaları: `data/biwi_person01/rgb/frame_XXXXX.png`
- ✅ Camera intrinsics: `data/biwi_person01/intrinsics.txt` (otomatik oluşturuldu)
- ⚠️ Depth dosyaları: Henüz desteklenmiyor (binary format)

### 2. Model ile Test Edin

```bash
cd build
./bin/test_real_data \
    --rgb ../data/biwi_person01/rgb/frame_00000.png \
    --intrinsics ../data/biwi_person01/intrinsics.txt \
    --model-dir ../data/model \
    --output-mesh ../data/biwi_person01/reconstructed_00000.ply
```

Bu komut mean shape (ortalama yüz) mesh'ini oluşturur.

### 3. Landmarks Ekleyin (Opsiyonel - MediaPipe Gerekli)

```bash
# Önce MediaPipe'ı kurun
pip install mediapipe opencv-python

# Landmark detection
python scripts/detect_landmarks.py \
    --image data/biwi_person01/rgb/frame_00000.png \
    --method mediapipe \
    --output data/biwi_person01/landmarks_00000.txt \
    --visualize

# Landmarks ile test
cd build
./bin/test_real_data \
    --rgb ../data/biwi_person01/rgb/frame_00000.png \
    --intrinsics ../data/biwi_person01/intrinsics.txt \
    --model-dir ../data/model \
    --landmarks ../data/biwi_person01/landmarks_00000.txt \
    --output-mesh ../data/biwi_person01/reconstructed_00000.ply
```

## 📊 Dataset Bilgileri

- **Toplam kişi sayısı**: 20+ (01, 02, 03, ...)
- **Her kişi için frame sayısı**: ~500-1500
- **Camera intrinsics**: Otomatik okunuyor (fx=575.816, fy=575.816, cx=320, cy=240)
- **Resolution**: 640x480

## 🔄 Farklı Kişileri Test Etmek

```bash
# Kişi 02
python scripts/convert_biwi_dataset.py \
    --input /Users/atakansucu/Downloads/archive/faces_0/02 \
    --output data/biwi_person02 \
    --max-frames 10

# Kişi 03
python scripts/convert_biwi_dataset.py \
    --input /Users/atakansucu/Downloads/archive/faces_0/03 \
    --output data/biwi_person03 \
    --max-frames 10
```

## ⚠️ Bilinen Sınırlamalar

1. **Depth Dosyaları**: Binary format henüz tam desteklenmiyor
   - RGB + landmarks ile test edebilirsiniz
   - Depth desteği yakında eklenecek

2. **Landmark Detection**: MediaPipe veya dlib kurulu olmalı
   ```bash
   pip install mediapipe opencv-python
   # veya
   pip install dlib opencv-python
   ```

## 📁 Oluşturulan Dosyalar

```
data/biwi_person01/
├── intrinsics.txt           # Camera parametreleri (otomatik)
├── rgb/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   └── ...
└── reconstructed_00000.ply  # Output mesh
```

## 🎯 Sonraki Adımlar

1. ✅ RGB + Model: Çalışıyor
2. 🔄 RGB + Landmarks + Model: MediaPipe kurulumu gerekli
3. ⏳ RGB + Depth + Model: Depth format çözülmesi bekleniyor
4. ⏳ Full Pipeline: Week 2+ (optimization)

## 💡 İpuçları

- Önce birkaç frame ile test edin (`--max-frames 5`)
- Farklı kişileri deneyin (farklı açılar, ışık koşulları)
- Mesh'leri MeshLab'da görüntüleyin
- Landmarks eklemek daha iyi sonuçlar verecektir
