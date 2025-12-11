# Biwi Dataset - Hızlı Başlangıç (Türkçe)

Dataset'iniz `/Users/atakansucu/Downloads/archive` klasöründe. Hemen başlayalım!

## 1. Dataset Yapısını Kontrol Edin

```bash
python scripts/check_biwi_dataset.py /Users/atakansucu/Downloads/archive
```

Bu size dataset'te ne olduğunu gösterir.

## 2. Bir Kişiyi Dönüştürün

Dataset'te birçok kişi var (01, 02, 03, vb.). Önce bir kişiyi test edin:

```bash
# Kişi 01'i dönüştür (ilk 5 frame ile test)
python scripts/convert_biwi_dataset.py \
    --input /Users/atakansucu/Downloads/archive/faces_0/01 \
    --output data/biwi_person01 \
    --max-frames 5
```

**Not:** Depth dosyaları özel bir formatta, şu an için RGB + landmarks ile test edebilirsiniz.

## 3. RGB + Landmarks ile Test Edin

Depth dosyaları henüz tam olarak desteklenmese de, RGB + landmarks ile test edebilirsiniz:

```bash
# Landmark detection
python scripts/detect_landmarks.py \
    --image data/biwi_person01/rgb/frame_00000.png \
    --method mediapipe \
    --output data/biwi_person01/landmarks_00000.txt \
    --visualize

# Test (RGB olmadan depth)
cd build
./bin/test_real_data \
    --rgb ../data/biwi_person01/rgb/frame_00000.png \
    --intrinsics ../data/biwi_person01/intrinsics.txt \
    --model-dir ../data/model \
    --landmarks ../data/biwi_person01/landmarks_00000.txt \
    --output-mesh ../data/biwi_person01/reconstructed_00000.ply
```

## 4. Depth Desteği (Deneysel)

Depth dosyalarını okumak için:

```bash
# Depth dosyasını PNG'ye çevirmeyi dene
python scripts/read_biwi_depth.py \
    /Users/atakansucu/Downloads/archive/faces_0/01/frame_00003_depth.bin \
    /tmp/depth_test.png

# Eğer çalışırsa, bunu conversion script'e ekleyebiliriz
```

## Önemli Notlar

1. **Camera Intrinsics**: Dataset'ten otomatik olarak `depth.cal` dosyasından okunuyor (fx=575.816, fy=575.816, cx=320, cy=240)

2. **RGB Dosyaları**: PNG formatında, direkt kullanılabilir ✅

3. **Depth Dosyaları**: Binary format, özel decompression gerekiyor ⚠️
   - Şu an için RGB-only test yapabilirsiniz
   - Depth desteği yakında eklenecek

4. **Kişiler**: Dataset'te 20+ kişi var, her birini ayrı test edebilirsiniz

## Sonraki Adımlar

- Depth format'ını tam olarak çözdükten sonra full RGB-D desteği eklenecek
- Şimdilik RGB + landmarks ile yüz rekonstrüksiyonu test edilebilir
