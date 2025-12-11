# Dataset Kontrol Listesi

## 1. Dataset'in Nerede Olduğunu Bulun

```bash
# Mac'te Downloads klasöründe olabilir
ls ~/Downloads/ | grep -i biwi

# Veya Desktop'ta
ls ~/Desktop/ | grep -i biwi

# Veya indirdiğiniz yeri hatırlayın
find ~ -name "*biwi*" -type d 2>/dev/null | head -5
```

## 2. Dataset Yapısını İnceleyin

Dataset'i bulduktan sonra:

```bash
cd /path/to/biwi_dataset
ls -la
```

**Ne arıyoruz:**
- ✅ `rgb/` klasörü veya `*rgb*.png` dosyaları
- ✅ `depth/` klasörü veya `*depth*.png` dosyaları
- ✅ Veya `person01/`, `person02/` gibi klasörler

## 3. Hızlı Kontrol

```bash
# RGB dosyaları var mı?
find /path/to/biwi_dataset -name "*rgb*" -o -name "*RGB*" | wc -l

# Depth dosyaları var mı?
find /path/to/biwi_dataset -name "*depth*" -o -name "*Depth*" | wc -l

# Toplam dosya sayısı
find /path/to/biwi_dataset -type f | wc -l
```

Bu sayıları görürseniz, dataset hazır! 🎉
