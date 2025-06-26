# label_extractor.py Kullanım Kılavuzu

Bu script, ardışık görsellerin etiketlerini birleştirerek RGB+MHI dosyaları için toplu etiket dosyaları oluşturur.

## Amaç
- 5'li (veya belirlediğiniz uzunlukta) ardışık görsel grubunun etiketlerini birleştirip, her grup için tek bir etiket dosyası üretmek
- Hem orijinal hem de augment edilmiş görsellerle çalışabilir

## Kullanım

```bash
python label_extractor.py --images_dir <görsellerin olduğu klasör> --labels_dir <etiketlerin olduğu klasör> --output_dir <etiketlerin kaydedileceği klasör> [--sequence_length 5] [--include_augmented] [--augmented_only]
```

### Argümanlar
- `--images_dir` : Görsellerin bulunduğu dizin (örn: `.../train/images`)
- `--labels_dir` : Etiketlerin bulunduğu dizin (örn: `.../train/labels`)
- `--output_dir` : Etiket dosyalarının kaydedileceği dizin (örn: `.../labels_rgbmhi`)
- `--sequence_length` : Kaç ardışık görselin birleştirileceği (varsayılan: 5)
- `--include_augmented` : Augment edilmiş verileri de dahil et (opsiyonel)
- `--augmented_only` : Sadece augment edilmiş veriler için işlem yap (opsiyonel)

## Çıktı
- Her ardışık grup için, örn: `WEB00000_to_WEB00004_rgbmhi.txt` gibi bir dosya oluşturulur.
- Dosya başında grup sınıfı bilgisi (`# class_info: ...`) ve ardından bounding box etiketleri yer alır.
- Her dosya, `output_dir` altında kaydedilir.

## Notlar
- Sınıf etiketi otomatik olarak belirlenir: `fire`, `smoke`, `smoke+fire`, `none`.
- Eksik veya okunamayan etiket dosyalarında uyarı verilir ve ilgili grup atlanır.
- Çıktı dizini yoksa otomatik olarak oluşturulur.

## Örnek
```bash
python label_extractor.py \
  --images_dir /path/to/train/images \
  --labels_dir /path/to/train/labels \
  --output_dir /path/to/labels_rgbmhi \
  --sequence_length 5 \
  --include_augmented
```

---
Herhangi bir hata veya öneri için kodun başındaki iletişim adresini kullanabilirsiniz. 