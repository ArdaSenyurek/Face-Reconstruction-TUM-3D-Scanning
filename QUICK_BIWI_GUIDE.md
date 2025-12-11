# Biwi Dataset - Hızlı Başlangıç

Biwi Kinect Head Pose Dataset'i kullanmak için hızlı rehber.

> 💡 **İlk kez kurulum mu?** Önce [BLWI_DATASET_SETUP.md](BLWI_DATASET_SETUP.md) dosyasına bakın - hangi dosyaların gerekli olduğunu açıklar.

## 1. Dataset'i Dönüştür

```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi/dataset \
    --output data/biwi_test \
    --kinect-version v1 \
    --max-frames 10
```

**Önemli:** Dataset'inizin yapısına göre `--input` path'ini ayarlayın:
- Eğer tüm dataset bir klasördeyse: `/path/to/biwi`
- Eğer her kişi ayrı klasördeyse: `/path/to/biwi/person01`

## 2. Tek Bir Frame Test Et

```bash
cd build
./bin/test_real_data \
    --rgb ../data/biwi_test/rgb/frame_00000.png \
    --depth ../data/biwi_test/depth/frame_00000.png \
    --intrinsics ../data/biwi_test/intrinsics.txt \
    --model-dir ../data/model \
    --output-mesh ../data/biwi_test/reconstructed_00000.ply
```

## 3. Sonucu Görüntüle

```bash
meshlab data/biwi_test/reconstructed_00000.ply
```

## 4. Birden Fazla Frame İşle (Opsiyonel)

```bash
./scripts/process_biwi_frames.sh data/biwi_test data/model data/biwi_test/output 10
```

Bu komut ilk 10 frame'i işler ve `data/biwi_test/output/` klasörüne kaydeder.

## Sorun Giderme

### "No RGB-depth pairs found" hatası

Dataset yapısını kontrol edin:
```bash
ls -la /path/to/biwi/
```

Eğer RGB ve depth farklı klasörlerdeyse:
```bash
# Örnek: rgb ve depth ayrı klasörlerde
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi  # rgb/ ve depth/ klasörlerini içeren klasör
    --output data/biwi_test
```

### Depth okuma hatası

Biwi dataset bazı versiyonlarda compressed binary format kullanır. Eğer hata alırsanız:

1. Dataset versiyonunu kontrol edin
2. Depth dosyalarının formatını kontrol edin:
   ```bash
   file data/biwi_test/depth/*.png | head -1
   ```

Eğer binary format ise, dataset'in kendi okuma scriptini kullanmanız gerekebilir.

## Detaylı Bilgi

Daha fazla bilgi için: [BLWI_DATASET_GUIDE.md](BLWI_DATASET_GUIDE.md)
