# Quick Start Guide - Using Dummy Model

Bu kısa rehber, dummy model ile hızlıca test yapmanızı sağlar.

## 1. Dummy Model Oluşturma

```bash
python3 scripts/prepare_model.py --generate-dummy --output data/model
```

Bu komut test için bir model oluşturur:
- 53,490 vertex
- 199 identity component
- 100 expression component

## 2. Model Yükleme ve Mesh Oluşturma

```bash
cd build
./bin/test_real_data --model-dir ../data/model --output-mesh ../data/reconstructed_face.ply
```

Bu komut:
1. Modeli yükler
2. Mean shape'i (sıfır katsayılarla) rekonstrükte eder
3. PLY formatında mesh kaydeder

## 3. Mesh Görüntüleme

Oluşturulan mesh'i görüntülemek için:

### MeshLab (Önerilen)
```bash
# macOS
brew install meshlab
meshlab data/reconstructed_face.ply

# Linux
sudo apt-get install meshlab
meshlab data/reconstructed_face.ply
```

### Blender
1. Blender'ı açın
2. File → Import → Stanford (.ply)
3. `data/reconstructed_face.ply` dosyasını seçin

### Online Viewer
- [3D Viewer Online](https://3dviewer.net/) - PLY dosyalarını tarayıcıda görüntüleyebilirsiniz

## Örnek Kullanım Senaryoları

### Sadece Model ile Mesh Oluşturma
```bash
./bin/test_real_data --model-dir ../data/model --output-mesh ../data/mean_face.ply
```

### RGB-D Verisi ile Test
```bash
./bin/test_real_data \
    --rgb data/rgb.png \
    --depth data/depth.png \
    --intrinsics data/intrinsics.txt \
    --model-dir ../data/model \
    --output-mesh ../data/reconstructed.ply
```

### Landmarks ile Test
```bash
./bin/test_real_data \
    --model-dir ../data/model \
    --landmarks data/landmarks.txt \
    --output-mesh ../data/reconstructed.ply
```

## Model Parametrelerini Özelleştirme

Dummy model oluştururken parametreleri değiştirebilirsiniz:

```bash
python3 scripts/prepare_model.py --generate-dummy \
    --output data/model_small \
    --num-vertices 10000 \
    --num-identity 50 \
    --num-expression 25
```

## Sonraki Adımlar

1. **Gerçek RGB-D verisi** ile test edin
2. **Landmark detection** ekleyin (Python script ile)
3. **Week 2**: Gauss-Newton optimizasyonu ile katsayıları optimize edin

## Sorun Giderme

### Model yüklenemiyor
- Dosya yollarını kontrol edin
- Binary dosyaların doğru formatta olduğundan emin olun
- `DATA_FORMAT.md` dosyasını kontrol edin

### Mesh görüntülenemiyor
- PLY dosyasının oluştuğunu kontrol edin: `ls -lh data/*.ply`
- Dosya boyutunun 0 olmadığından emin olun
- Farklı bir viewer deneyin

### Hata mesajları
- Build dizininde olduğunuzdan emin olun
- Executable'ın derlendiğini kontrol edin: `ls -lh bin/test_real_data`
