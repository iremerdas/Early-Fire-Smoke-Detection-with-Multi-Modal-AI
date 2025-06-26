# YOLO Eğitimi Kullanım Kılavuzu

Bu script, fire/smoke tespiti için YOLO modelini eğitir.

## Gereksinimler

```bash
pip install ultralytics pandas pyyaml
```

## Dataset Yapısı

YOLO formatında hazır dataset'iniz olmalı:

```
data_split/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── val/
    ├── images/
    │   ├── image3.jpg
    │   ├── image4.jpg
    │   └── ...
    └── labels/
        ├── image3.txt
        ├── image4.txt
        └── ...
```

## Kullanım

### 1. Temel Eğitim

```bash
python yolo_training.py \
  --data_dir data_split \
  --output_dir yolo_training \
  --epochs 100 \
  --batch_size 16
```

### 2. Gelişmiş Parametreler

```bash
python yolo_training.py \
  --data_dir data_split \
  --output_dir yolo_training \
  --epochs 200 \
  --batch_size 32 \
  --imgsz 640 \
  --model_size m
```

### 3. Sadece Doğrulama

```bash
python yolo_training.py \
  --data_dir data_split \
  --output_dir yolo_training \
  --validate_only
```

## Parametreler

- `--data_dir`: YOLO formatında dataset dizini (train/val altında images/labels)
- `--output_dir`: Çıktı dizini (varsayılan: yolo_training)
- `--epochs`: Eğitim epoch sayısı (varsayılan: 100)
- `--batch_size`: Batch boyutu (varsayılan: 16)
- `--imgsz`: Görsel boyutu (varsayılan: 640)
- `--model_size`: YOLO model boyutu (n/s/m/l/x, varsayılan: n)
- `--validate_only`: Sadece doğrulama yap

## YOLO Label Formatı

Her .txt dosyası şu formatta olmalı:
```
class_id x_center y_center width height
```

### Sınıflar
- `0: smoke`
- `1: fire`
- `2: none`

## Çıktı

```
yolo_training/
├── dataset.yaml
├── fire_smoke_detection/
│   ├── weights/
│   │   ├── best.pt          # En iyi model
│   │   └── last.pt          # Son model
│   ├── results.png
│   ├── confusion_matrix.png
│   └── ...
└── images/
    ├── train/
    └── val/
```

## Pipeline ile Entegrasyon

Eğitilen modeli `live_detection_pipeline.py`'de kullanmak için:

```python
# live_detection_pipeline.py içinde
yolo_model_path = "yolo_training/fire_smoke_detection/weights/best.pt"
```

## Örnek Kullanım

### 1. Küçük Model (Hızlı)
```bash
python yolo_training.py \
  --data_dir data_split \
  --output_dir yolo_training_small \
  --model_size n \
  --epochs 50
```

### 2. Orta Model (Dengeli)
```bash
python yolo_training.py \
  --data_dir data_split \
  --output_dir yolo_training_medium \
  --model_size m \
  --epochs 100
```

### 3. Büyük Model (Yüksek Doğruluk)
```bash
python yolo_training.py \
  --data_dir data_split \
  --output_dir yolo_training_large \
  --model_size l \
  --epochs 200
```

## Notlar

- GPU varsa otomatik olarak kullanılır
- Early stopping ile overfitting önlenir
- Model boyutu büyüdükçe doğruluk artar ama hız azalır
- Eğitim sonuçları `runs/` klasöründe saklanır
- En iyi model `best.pt` olarak kaydedilir

## Sorun Giderme

### CUDA Hatası
```bash
# CPU kullan
export CUDA_VISIBLE_DEVICES=""
```

### Bellek Hatası
```bash
# Batch size'ı küçült
--batch_size 8
```

### Dataset Hatası
- Dataset yapısını kontrol edin
- Label dosyalarının doğru formatta olduğundan emin olun
- Sınıf ID'lerinin doğru olduğunu kontrol edin 