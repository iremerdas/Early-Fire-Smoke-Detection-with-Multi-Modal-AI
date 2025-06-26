# Live Detection Pipeline Kullanım Kılavuzu

Bu pipeline, gerçek zamanlı duman/ateş tespiti ve nesne tespiti yapar. Önce CNN ile sahne sınıflandırması yapar, pozitif sahnelerde YOLO ile nesne tespiti gerçekleştirir.

## 🎯 Amaç
- Video akışından 5'li frame grupları alır
- MHI (Motion History Image) ve Optical Flow hesaplar
- CNN ile sahne sınıflandırması yapar (smoke/fire/none)
- Pozitif sahnelerde YOLO ile nesne tespiti yapar
- Gerçek zamanlı görselleştirme sağlar

## 📦 Gereksinimler

### Model Dosyaları
- `runs/exp_scene_split/best_model.pth` - CNN model ağırlıkları
- `yolov8n.pt` - YOLO model ağırlıkları (otomatik indirilir)

### Python Paketleri
```bash
pip install torch torchvision opencv-python ultralytics numpy
```

## 🚀 Kullanım

### 1. Kamera ile Çalıştırma
```bash
python live_detection_pipeline.py
```

### 2. Video Dosyası ile Çalıştırma
```bash
python live_detection_pipeline.py --video "video.mp4"
```

### 3. Çıktı Video Kaydetme
```bash
python live_detection_pipeline.py --video "input.mp4" --output "output.mp4"
```

### 4. Özel Model Dosyaları
```bash
python live_detection_pipeline.py \
  --cnn_model "path/to/custom_model.pth" \
  --yolo_model "path/to/custom_yolo.pt"
```

### 5. Parametre Ayarlama
```bash
python live_detection_pipeline.py \
  --threshold 0.7 \
  --sequence_length 7 \
  --no_logging
```

## ⚙️ Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--video` | `0` | Video dosyası veya kamera (0) |
| `--output` | `None` | Çıktı video dosyası |
| `--cnn_model` | `runs/exp_scene_split/best_model.pth` | CNN model dosyası |
| `--yolo_model` | `yolov8n.pt` | YOLO model dosyası |
| `--threshold` | `0.5` | Pozitif sahne eşiği |
| `--sequence_length` | `5` | MHI için frame sayısı |
| `--no_logging` | `False` | Logging sistemini devre dışı bırak |

## 🔄 Pipeline Adımları

### 1. Frame Yakalama
- Video akışından ardışık 5 frame alınır
- Frame buffer'da saklanır

### 2. MHI Hesaplama
```python
# 5 frame'den Motion History Image üretilir
mhi = compute_mhi(frames_list)
```

### 3. Optical Flow Hesaplama
```python
# Ardışık frame çiftlerinden optical flow magnitude
optical_flow = compute_optical_flow(frames_list)
```

### 4. Input Tensor Oluşturma
```python
# 5-kanal tensor: [R, G, B, MHI, Flow]
input_tensor = create_input_tensor(rgb_frame, mhi, optical_flow)
# Shape: (1, 5, 224, 224)
```

### 5. CNN Sınıflandırması
```python
# Multi-label çıktı: [smoke_prob, fire_prob, none_prob]
scene_prediction = predict_scene(input_tensor)
```

### 6. YOLO Nesne Tespiti
```python
# Sadece pozitif sahnelerde
if scene_prediction['is_positive']:
    detections = detect_objects(frame)
```

## 📊 Çıktı Formatları

### Sahne Sınıflandırması
```python
{
    'smoke_prob': 0.85,    # Duman olasılığı
    'fire_prob': 0.12,     # Ateş olasılığı  
    'none_prob': 0.03,     # Hiçbiri olasılığı
    'is_positive': True,   # Pozitif sahne mi?
    'predictions': [0.85, 0.12, 0.03]  # Ham çıktılar
}
```

### Nesne Tespiti
```python
[
    {
        'bbox': [x1, y1, x2, y2],  # Bounding box koordinatları
        'confidence': 0.92,        # Güven skoru
        'class': 'fire',           # Sınıf adı
        'class_id': 0              # Sınıf ID'si
    }
]
```

## 🎨 Görselleştirme

### Frame Üzerine Çizilen Bilgiler
- **Sahne Olasılıkları**: Sol üst köşede smoke/fire olasılıkları
- **Pozitif Sahne Uyarısı**: Kırmızı çerçeve + "ALERT" yazısı
- **YOLO Tespitleri**: Yeşil bounding box'lar + sınıf etiketleri
- **FPS Bilgisi**: Sol alt köşede gerçek zamanlı FPS
- **İstatistikler**: Pozitif sahne ve tespit sayıları

## 📝 Logging

### Log Dosyası
- `detection_log.txt` dosyasına kaydedilir
- Her frame için sahne olasılıkları ve tespitler
- `--no_logging` ile devre dışı bırakılabilir

## 📈 Performans İstatistikleri

### Gerçek Zamanlı Metrikler
- **FPS**: Saniyedeki frame sayısı
- **Pozitif Sahne Sayısı**: Tespit edilen pozitif sahneler
- **Toplam Tespit**: YOLO ile tespit edilen nesne sayısı

### Final Rapor
```
Final İstatistikler:
Toplam Frame: 1500
Pozitif Sahne: 45
Toplam Tespit: 23
Ortalama FPS: 25.3
```

## 🔧 Özelleştirme

### Model Değiştirme
```python
# Farklı CNN modeli
pipeline = LiveDetectionPipeline(
    cnn_model_path="custom_model.pth",
    yolo_model_path="yolov8s.pt"  # Daha büyük model
)
```

### Eşik Ayarlama
```python
# Daha hassas tespit için
pipeline = LiveDetectionPipeline(threshold=0.3)

# Daha seçici tespit için  
pipeline = LiveDetectionPipeline(threshold=0.7)
```

### Sequence Uzunluğu
```python
# Daha uzun hareket geçmişi
pipeline = LiveDetectionPipeline(sequence_length=7)
```

## ⚠️ Önemli Notlar

### Donanım Gereksinimleri
- **GPU**: CUDA destekli GPU önerilir (CNN + YOLO)
- **RAM**: En az 4GB RAM
- **CPU**: Multi-core CPU (optical flow hesaplama)

### Performans Optimizasyonu
- Daha küçük YOLO modeli kullanın (`yolov8n.pt`)
- Frame boyutunu küçültün
- Sequence length'i azaltın

### Hata Durumları
- **Model bulunamadı**: Otomatik YOLO indirme
- **Kamera erişimi**: USB kamera bağlantısını kontrol edin
- **CUDA hatası**: CPU moduna geçer

## 🎮 Kontroller

### Klavye Kısayolları
- **Q**: Programdan çık
- **ESC**: Programdan çık

### Çıkış
```bash
# Ctrl+C ile güvenli çıkış
# Final istatistikler otomatik gösterilir
```

## 📝 Örnek Kullanım Senaryoları

### 1. Güvenlik Kamerası
```bash
python live_detection_pipeline.py --video 0 --output security_feed.mp4
```

### 2. Video Analizi
```bash
python live_detection_pipeline.py --video test_video.mp4 --threshold 0.6
```

### 3. Sessiz Mod
```bash
python live_detection_pipeline.py --no_logging
```

### 4. Yüksek Hassasiyet
```bash
python live_detection_pipeline.py --threshold 0.3 --sequence_length 7
```

---

Bu pipeline, eğittiğiniz CNN modelini gerçek zamanlı olarak kullanarak hem sahne sınıflandırması hem de nesne tespiti yapar. Herhangi bir sorun yaşarsanız log dosyalarını kontrol edin. 