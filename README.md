# Early-Fire-Smoke-Detection-with-Multi-Modal-AI
5-Channel Early Wildfire Detection: RGB-MHI-Flow + CNN-YOLO Pipeline
 ---

 # Fire/Smoke Detection System 

Bu proje, RGB + MHI + Optical Flow verilerini kullanarak **duman ve ateş tespiti** yapan, hem sahne sınıflandırması hem de nesne tespiti (YOLO) destekli, gerçek zamanlı ve offline çalışabilen bir derin öğrenme sistemidir.

---

## 🚩 Proje Özeti
- **Multi-label CNN**: EfficientNet-B4+CBAM, ResNet50+SE, SwinTransformer+CBAM ile sahne sınıflandırması
- **5-kanal Input**: RGB (3) + MHI (1) + Optical Flow (1)
- **Gerçek Zamanlı Tespit**: Live detection pipeline ile video/kamera akışı
- **Nesne Tespiti**: YOLOv8 ile pozitif sahnelerde obje tespiti
- **Esnek Eğitim/Test**: Komut satırından backbone, loss, scheduler, batch, threshold seçimi
- **Görselleştirme**: FPS, istatistik, uyarı, confusion matrix, ROC, eğitim geçmişi

---

## 📁 Proje Klasör Yapısı
```
Fire-Smoke-Detect/
├── README.md
├── requirements.txt
├── config.yaml
│
├── dataset_tools/         # Veri hazırlama ve etiketleme araçları
│   ├── cascade_masking_full.py
│   ├── rgb_mhi_stack_with_flow.py
│   ├── label_extractor_from_rgbmhiflow.py
│   ├── feature_extraction.py
│   └── ...
│
├── models/               # Eğitim, test, çıkarım ve model tanım dosyaları
│   ├── train_multi_label.py
│   ├── test_model.py
│   ├── inference.py
│   ├── multi_label_classifier.py
│   ├── multi_label_dataset.py
│   ├── training_utils.py
│   ├── cbam_module.py
│   ├── evaluation.py
│   └── ...
│
├── live/                 # Gerçek zamanlı çalışma modülleri
│   ├── live_detection_pipeline.py
│   ├── test_live_pipeline.py
│   └── USAGE_live_detection_pipeline.md
│
├── yolo_training/        # YOLO eğitim ve çıktı dosyaları
│   ├── dataset.yaml
│   ├── fire_smoke_detection/
│   │   ├── weights/best.pt
│   │   └── ...
│   └── ...
│
├── runs/                 # Eğitim çıktı klasörü (model.pth, loglar, görseller)
│   └── ...
│
├── docs/                 # Kullanım kılavuzları ve dökümantasyon
│   ├── KULLANIM_KILAVUZU.md
│   └── ...
│
├── examples/             # Demo videoları ve örnek çıkışlar
│   └── ...
│
├── ozellikler/           # Özellik çıkarım sonuçları (csv)
│   └── ...
│
├── gorsel/               # Görsel örnekler ve maskeler
│   └── ...
│
└── .gitignore
```

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Veri Hazırlama (Özet)
- **MHI ve Optical Flow üretimi:**
  ```bash
  python dataset_tools/mhi_generator.py
  python dataset_tools/rgb_mhi_stack_with_flow.py
  ```
- **Etiket çıkarımı:**
  ```bash
  python dataset_tools/label_extractor_from_rgbmhiflow.py
  ```

### 3. CNN Model Eğitimi
```bash
python models/train_multi_label.py \
  --train_npy_dir scene_split_output_v4/train/flow \
  --train_labels_dir scene_split_output_v4/train/flow_labels \
  --val_npy_dir scene_split_output_v4/val/flow_val \
  --val_labels_dir scene_split_output_v4/val/flow_val_labels \
  --output_dir runs/efficientnet_b4_cbam \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --model_name efficientnet_b4_cbam \
  --scheduler cosine \
  --loss focal \
  --patience 5 \
  --checkpoint_interval 5
```
- **Alternatif backbone:**
  - `--model_name resnet50_se` veya `--model_name swintransformer_cbam`
- **Loss seçenekleri:**
  - `--loss bce`, `--loss focal`, `--loss smooth`
- **Scheduler seçenekleri:**
  - `--scheduler step`, `--scheduler cosine`

### 4. Model Testi ve Değerlendirme
```bash
python models/evaluation.py \
  --model_path runs/efficientnet_b4_cbam/best_model.pth \
  --test_npy_dir scene_split_output_v4/test/flow_test \
  --test_labels_dir scene_split_output_v4/test/flow_test_labels \
  --output_dir runs/efficientnet_b4_cbam/evaluation_results_test \
  --batch_size 32 \
  --num_workers 4 \
  --threshold 0.5
```

### 5. Gerçek Zamanlı Tespit
```bash
python live/live_detection_pipeline.py \
  --cnn_model runs/efficientnet_b4_cbam/best_model.pth \
  --yolo_model yolo_training/fire_smoke_detection/weights/best.pt \
  --video video1.mp4 \
  --output cikti.mp4
```
- **Kamera ile:** `--video 0`
- **Çıktı videosu:** `--output cikti.mp4`

---

## ⚙️ Parametreler ve Seçenekler

### CNN Model Eğitimi
- `--model_name` : efficientnet_b4_cbam, resnet50_se, swintransformer_cbam
- `--loss` : bce, focal, smooth
- `--scheduler` : step, cosine
- `--batch_size`, `--epochs`, `--lr`, `--patience`, `--checkpoint_interval`

### YOLO Eğitimi
- `--data_dir`, `--annotation_file`, `--output_dir`, `--epochs`, `--batch_size`, `--imgsz`, `--model_size`, `--train_split`, `--validate_only`

### Live Detection Pipeline
- `--cnn_model` : Eğitilmiş CNN model dosyası
- `--yolo_model` : YOLO model dosyası
- `--video` : Video dosyası veya kamera (0)
- `--output` : Çıktı video dosyası
- `--threshold` : Pozitif sahne eşiği (varsayılan 0.5)
- `--sequence_length` : MHI için frame sayısı (varsayılan 5)
- `--no_logging` : Logging sistemini kapat

---

## 📊 Model Performansı ve Görselleştirme
- **Accuracy, F1, Recall, Exact Match**: CNN test ve validasyon sonuçları
- **Confusion Matrix, ROC, Loss/Accuracy Curves**: Eğitim ve test görselleri
- **YOLO mAP, Precision/Recall**: Nesne tespiti sonuçları
- **Gerçek Zamanlı Çıktı**: FPS, istatistik, uyarı, bounding box, sahne olasılıkları

---

## 📚 Dökümantasyon ve Ekstra
- **KULLANIM_KILAVUZU.md**: Detaylı kullanım ve pipeline akışı
- **USAGE_live_detection_pipeline.md**: Canlı tespit pipeline'ı için özel kılavuz
- **USAGE_label_extractor.md**: Özel görüntü ve hareket verilerinden etiket çıkarımı için kılavuz
- **USAGE_rgb_mhi_stack_with_flow.md**: 5 kanallı .npy dosyaları oluşturmak için kılavuz
- **USAGE_yolo_training.md**: YOLO eğitimi için kılavuzu

---

## 📝 Notlar
- Tüm veri yolları ve parametreler komut satırından esnek şekilde ayarlanabilir.
- 5-kanal giriş (RGB+MHI+Flow) ve CBAM/SE attention tüm modellerde desteklenir.
- Geçici, debug veya eski dosyalar ana akıştan çıkarılmıştır.
- Proje modüler ve genişletilebilir yapıdadır.

---

Her türlü soru ve katkı için iletişime geçebilirsiniz! 
