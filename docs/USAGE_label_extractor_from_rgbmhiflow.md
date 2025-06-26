# label_extractor_from_rgbmhiflow.py Kullanım Kılavuzu

Bu script, rgbmhi/rgbmhiflow klasöründeki .npy dosyalarını okuyarak, her dosya için aynı isimli etiket dosyası oluşturur.

## Amaç
- RGBMHI/RGBMHIFLOW .npy dosyalarının isimlerini kullanarak, %100 uyumlu etiket dosyaları oluşturmak
- Sadece var olan .npy dosyaları için etiket üretir, eksik dosya sorunu olmaz

## Kullanım

```bash
python label_extractor_from_rgbmhiflow.py --rgbmhiflow_dir <npy dosyalarının olduğu klasör> --labels_dir <orijinal etiketlerin olduğu klasör> --output_dir <etiketlerin kaydedileceği klasör>
```

### Argümanlar
- `--rgbmhiflow_dir` : RGBMHI/RGBMHIFLOW .npy dosyalarının bulunduğu dizin (örn: `.../rgbmhiflow`)
- `--labels_dir` : Orijinal etiketlerin bulunduğu dizin (örn: `.../train/labels`)
- `--output_dir` : Etiket dosyalarının kaydedileceği dizin (örn: `.../labels_rgbmhiflow`)

## Çıktı
- Her .npy dosyası için, aynı isimli .txt dosyası oluşturulur:
  - `AoF00001_to_AoF00005_rgbmhiflow.npy` → `AoF00001_to_AoF00005_rgbmhiflow.txt`
  - `aug_0_AoF00001_to_aug_4_AoF00001_rgbmhiflow.npy` → `aug_0_AoF00001_to_aug_4_AoF00001_rgbmhiflow.txt`
- Dosya başında grup sınıfı bilgisi (`# class_info: ...`) ve ardından bounding box etiketleri yer alır.

## Avantajları
✅ **%100 Uyumluluk:** Sadece var olan .npy dosyaları için etiket oluşturur  
✅ **Hata Yok:** Eksik dosya sorunu olmaz  
✅ **Basit:** Tek kaynak (rgbmhiflow klasörü)  
✅ **Güvenli:** Sadece gerçekten üretilmiş veriler için etiket  

## Örnek
```bash
python label_extractor_from_rgbmhiflow.py \
  --rgbmhiflow_dir /path/to/rgbmhiflow \
  --labels_dir /path/to/train/labels \
  --output_dir /path/to/labels_rgbmhiflow
```

## Notlar
- Sınıf etiketi otomatik olarak belirlenir: `fire`, `smoke`, `smoke+fire`, `none`.
- Eksik veya okunamayan etiket dosyalarında uyarı verilir ve ilgili grup atlanır.
- Çıktı dizini yoksa otomatik olarak oluşturulur.
- İşlem sırasında ilerleme durumu için progress bar (tqdm) kullanılır.
- İşlem sonunda başarılı/başarısız dosya sayıları raporlanır.

---
Herhangi bir hata veya öneri için kodun başındaki iletişim adresini kullanabilirsiniz. 