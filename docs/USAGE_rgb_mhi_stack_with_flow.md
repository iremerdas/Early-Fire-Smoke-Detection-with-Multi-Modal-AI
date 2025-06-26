# rgb_mhi_stack_with_flow.py Kullanım Kılavuzu

Bu script, her MHI dosyası için ilgili 5'li ardışık görsel grubunu bulur, optical flow magnitude kanalını hesaplar ve (H, W, 5) shape'li `.npy` dosyası olarak kaydeder.

## Amaç
- RGB (3 kanal) + MHI (1 kanal) + Optical Flow Magnitude (1 kanal) = (H, W, 5) shape'li tensor üretmek
- Çıktı dosyaları: `_rgbmhiflow.npy` uzantılı

## Kullanım

```bash
python rgb_mhi_stack_with_flow.py --rgb_dir <images klasörü> --mhi_dir <mhi2 klasörü> --output_dir <çıktı klasörü>
```

### Argümanlar
- `--rgb_dir` : RGB görüntülerinin bulunduğu dizin (örn: `.../images`)
- `--mhi_dir` : MHI dosyalarının bulunduğu dizin (örn: `.../mhi2`)
- `--output_dir` : Çıktıların kaydedileceği dizin (örn: `.../rgbmhiflow`)

## Çıktı
- Her MHI dosyası için, aynı isimli ve `_rgbmhiflow.npy` uzantılı dosya oluşturulur.
- Her çıktı dosyası `(H, W, 5)` shape'li numpy array içerir:
    - İlk 3 kanal: RGB
    - 4. kanal: MHI
    - 5. kanal: Optical Flow Magnitude (normalize edilmiş)
- Kaydedilen her dosya ismi ekrana yazdırılır: `[✓] Kaydedildi: <dosya_adı>`

## Notlar
- Optical flow hesaplanırken, 5'li ardışık görsel grubundaki 4 çift (0→1, 1→2, 2→3, 3→4) kullanılır.
- Eksik veya okunamayan görsellerde uyarı verilir ve ilgili dosya atlanır.
- Çıktı dizini yoksa otomatik olarak oluşturulur.
- İşlem sırasında ilerleme durumu için progress bar (tqdm) kullanılır.

## Örnek
```bash
python rgb_mhi_stack_with_flow.py \
  --rgb_dir /path/to/images \
  --mhi_dir /path/to/mhi2 \
  --output_dir /path/to/rgbmhiflow
```

---
Herhangi bir hata veya öneri için kodun başındaki iletişim adresini kullanabilirsiniz. 