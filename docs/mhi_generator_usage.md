# MHI Generator Kullanım Kılavuzu

## Güncellenmiş Özellikler

Artık MHI generator hem normal hem de augment edilmiş verileri işleyebilir. Augment edilmiş veriler `aug_{i}_{original_name}.jpg` formatında olmalıdır.

## Kullanım Seçenekleri

### 1. Sadece Normal Veriler İçin MHI Üretimi
```bash
python mhi_generator.py --input_dir "path/to/images" --output_dir "path/to/mhi_output"
```

### 2. Normal + Augment Edilmiş Veriler İçin MHI Üretimi
```bash
python mhi_generator.py --input_dir "path/to/images" --output_dir "path/to/mhi_output" --include_augmented
```

### 3. Sadece Augment Edilmiş Veriler İçin MHI Üretimi
```bash
python mhi_generator.py --input_dir "path/to/images" --output_dir "path/to/mhi_output" --augmented_only
```

### 4. Özel Sequence Uzunluğu ile
```bash
python mhi_generator.py --input_dir "path/to/images" --output_dir "path/to/mhi_output" --include_augmented --sequence_length 7
```

## Parametreler

- `--input_dir`: Görsellerin bulunduğu dizin
- `--output_dir`: MHI dosyalarının kaydedileceği dizin
- `--sequence_length`: MHI için kullanılacak görüntü sayısı (varsayılan: 5)
- `--include_augmented`: Augment edilmiş verileri de dahil et
- `--augmented_only`: Sadece augment edilmiş veriler için MHI üret

## Gruplandırma Mantığı

### Normal Veriler
- `WEB` ile başlayan görseller → WEB grubu
- `AoF` ile başlayan görseller → AoF grubu
- `PublicDataset` ile başlayan görseller → PublicDataset grubu
- `ck0q` ile başlayan görseller → ck0q grubu

### Augment Edilmiş Veriler
- `aug_0_AoF00001.jpg` → AoF grubu (AoF00001'e ait)
- `aug_1_WEB001.jpg` → WEB grubu (WEB001'e ait)
- Her augment edilmiş görsel orijinal görsel adına göre gruplandırılır

## Çıktı Formatı

### Normal MHI Dosyaları
- `AoF00001_to_AoF00005_mhi.jpg`
- `WEB001_to_WEB005_mhi.jpg`

### Augment Edilmiş MHI Dosyaları
- `aug_0_AoF00001_to_aug_4_AoF00001_mhi.jpg`
- `aug_0_WEB001_to_aug_4_WEB001_mhi.jpg`

## Örnek Kullanım Senaryoları

### Senaryo 1: Tüm Veriler İçin MHI
```bash
# Train klasörü için
python mhi_generator.py --input_dir "C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\images" --output_dir "C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\mhi2" --include_augmented

# Test klasörü için
python mhi_generator.py --input_dir "D-Fire/test/images" --output_dir "D-Fire/test/mhi" --include_augmented
```

### Senaryo 2: Sadece Augment Edilmiş Veriler
```bash
python mhi_generator.py --input_dir "C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\images" --output_dir "C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\mhi2" --augmented_only
```

### Senaryo 3: Farklı Sequence Uzunluğu
```bash
python mhi_generator.py --input_dir "D-Fire/train/images" --output_dir "D-Fire/train/mhi_7frame" --include_augmented --sequence_length 7
```

## Notlar

1. Augment edilmiş veriler orijinal görsel adına göre gruplandırılır
2. Her grup için yeterli ardışık görsel yoksa MHI üretilmez
3. Çıktı dosyaları JPG formatında kaydedilir
4. MHI hesaplaması için threshold değeri 30 olarak ayarlanmıştır 