# Yangın ve Duman Çoklu Etiket Sınıflandırma Sistemi Kullanım Kılavuzu (KOD2)

## İçindekiler
- [1. Proje Özeti](#1-proje-özeti)
- [2. Klasör ve Dosya Yapısı (Özet)](#2-klasör-ve-dosya-yapısı-özet)
- [3. Hızlı Başlangıç](#3-hızlı-başlangıç)
  - [3.1 Gereksinimleri Yükleyin](#31-gereksinimleri-yükleyin)
  - [3.2 Veri Hazırlama](#32-veri-hazırlama)
  - [3.3 CNN Model Eğitimi](#33-cnn-model-eğitimi)
  - [3.4 Model Testi ve Değerlendirme](#34-model-testi-ve-değerlendirme)
  - [3.5 Gerçek Zamanlı Tespit](#35-gerçek-zamanlı-tespit)
- [4. Parametreler ve Seçenekler](#4-parametreler-ve-seçenekler)
- [5. Model Performansı ve Görselleştirme](#5-model-performansı-ve-görselleştirme)
- [6. Dökümantasyon ve Ekstra](#6-dökümantasyon-ve-ekstra)
- [7. Notlar](#7-notlar)

---

## 1. Proje Özeti
- EfficientNet-B4+CBAM, ResNet50+SE, SwinTransformer+CBAM ile multi-label sahne sınıflandırması
- 5-kanal giriş (RGB+MHI+Flow)
- YOLOv8 ile nesne tespiti (pozitif sahnelerde)
- Gerçek zamanlı ve offline çalışma
- Esnek eğitim/test parametreleri
- FPS, istatistik, confusion matrix, ROC, eğitim geçmişi görselleştirme

---

## 2. Klasör ve Dosya Yapısı (Özet)
```
KOD2/
├── requirements.txt
├── dataset_tools/         # Veri hazırlama scriptleri
├── models/               # Eğitim, test, çıkarım, model tanımı
├── live/                 # Gerçek zamanlı pipeline
├── yolo_training/        # YOLO eğitim ve çıktı dosyaları
├── runs/                 # Eğitim çıktıları
├── docs/                 # Kılavuzlar
├── examples/, ozellikler/, gorsel/
└── .gitignore
```

---

## 3. Hızlı Başlangıç

### 3.1 Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3.2 Veri Hazırlama
- **MHI ve Optical Flow üretimi:**
  ```bash
  python dataset_tools/mhi_generator.py
  python dataset_tools/rgb_mhi_stack_with_flow.py
  ```
- **Etiket çıkarımı:**
  ```bash
  python dataset_tools/label_extractor_from_rgbmhiflow.py
  ```

### 3.3 CNN Model Eğitimi
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

### 3.4 Model Testi ve Değerlendirme
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

### 3.5 Gerçek Zamanlı Tespit
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

## 4. Parametreler ve Seçenekler

### CNN Model Eğitimi
- `--model_name` : efficientnet_b4_cbam, resnet50_se, swintransformer_cbam
- `--loss` : bce, focal, smooth
- `--scheduler` : step, cosine
- `--batch_size`, `--epochs`, `--lr`, `--patience`, `--checkpoint_interval`

### Live Detection Pipeline
- `--cnn_model` : Eğitilmiş CNN model dosyası
- `--yolo_model` : YOLO model dosyası
- `--video` : Video dosyası veya kamera (0)
- `--output` : Çıktı video dosyası
- `--threshold` : Pozitif sahne eşiği (varsayılan 0.5)
- `--sequence_length` : MHI için frame sayısı (varsayılan 5)
- `--no_logging` : Logging sistemini kapat

### YOLO Eğitimi
- `--data_dir`, `--annotation_file`, `--output_dir`, `--epochs`, `--batch_size`, `--imgsz`, `--model_size`, `--train_split`, `--validate_only`

---

## 5. Model Performansı ve Görselleştirme
- **Accuracy, F1, Recall, Exact Match**: CNN test ve validasyon sonuçları
- **Confusion Matrix, ROC, Loss/Accuracy Curves**: Eğitim ve test görselleri
- **YOLO mAP, Precision/Recall**: Nesne tespiti sonuçları
- **Gerçek Zamanlı Çıktı**: FPS, istatistik, uyarı, bounding box, sahne olasılıkları

---

## 6. Dökümantasyon ve Ekstra
- **KULLANIM_KILAVUZU.md**: Detaylı kullanım ve pipeline akışı
- **USAGE_live_detection_pipeline.md**: Canlı tespit pipeline'ı için özel kılavuz
- **README_pipeline_akisi.md**: Akış diyagramları ve pipeline açıklamaları

---

## 7. Notlar
- Tüm veri yolları ve parametreler komut satırından esnek şekilde ayarlanabilir.
- 5-kanal giriş (RGB+MHI+Flow) ve CBAM/SE attention tüm modellerde desteklenir.
- Geçici, debug veya eski dosyalar ana akıştan çıkarılmıştır.
- Proje modüler ve genişletilebilir yapıdadır.

---

Her türlü soru ve katkı için iletişime geçebilirsiniz!

## 0. Çalıştırma Sırası

### 0.1 Sistem Kurulumu ve Hazırlık
1. **Veri Hazırlama**: Veri setini uygun formatta organize et
2. **Etiket Dosyası**: labels.csv dosyasını oluştur
3. **Bağımlılıklar**: Gerekli Python paketlerini yükle

### 0.2 Model Eğitimi
1. **Eğitim**: `train_multi_label.py` ile model eğitimi
2. **İzleme**: TensorBoard ile eğitim sürecini takip et
3. **Değerlendirme**: `test_model.py` ile model performansını test et

### 0.3 Çıkarım ve Kullanım
1. **Tek Görüntü**: `inference.py` ile tek görüntü tahmini
2. **Toplu İşlem**: `batch_inference.py` ile toplu tahmin
3. **Sonuç Analizi**: Çıktıları analiz et ve raporla

## 0.5 Kod Dosyaları Çalıştırma Sırası

### 0.5.1 Zorunlu Çalıştırma Sırası

#### 1. Adım: Veri Hazırlama
```bash
# 1. Veri setini organize et (manuel)
mkdir -p dataset/{train,val,test}/{images,mhi,flow}

# 2. Etiket dosyasını oluştur (eğer yoksa)
python create_labels.py --data_dir dataset/train --output labels.csv
```

#### 2. Adım: Model Eğitimi
```bash
# 3. Model eğitimi (ZORUNLU)
python train_multi_label.py \
    --data_dir dataset/train \
    --image_dir dataset/train/images \
    --mhi_dir dataset/train/mhi \
    --flow_dir dataset/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir runs/experiment1
```

#### 3. Adım: Model Testi
```bash
# 4. Model performansını test et (ZORUNLU)
python test_model.py \
    --model_path runs/experiment1/best_model.pth \
    --test_dir dataset/test \
    --image_dir dataset/test/images \
    --mhi_dir dataset/test/mhi \
    --flow_dir dataset/test/flow \
    --output_dir test_results/experiment1
```

#### 4. Adım: Çıkarım
```bash
# 5. Tek görüntü tahmini (İSTEĞE BAĞLI)
python inference.py \
    --model_path runs/experiment1/best_model.pth \
    --image_path test_image.jpg \
    --mhi_path test_mhi.npy \
    --flow_path test_flow.npy \
    --threshold 0.5 \
    --output_dir predictions/single_test

# 6. Toplu tahmin (İSTEĞE BAĞLI)
python batch_inference.py \
    --model_path runs/experiment1/best_model.pth \
    --input_dir dataset/test \
    --output_dir predictions/batch_test
```

### 0.5.2 Dosya Bağımlılıkları

#### Zorunlu Dosyalar (Sırayla)
1. **`multi_label_dataset.py`** - Veri yükleme ve ön işleme (otomatik import)
2. **`multi_label_classifier.py`** - Model mimarisi (otomatik import)
3. **`training_utils.py`** - Eğitim yardımcı fonksiyonları (otomatik import)
4. **`train_multi_label.py`** - Eğitim scripti (1. çalıştırılacak)
5. **`test_model.py`** - Test scripti (2. çalıştırılacak)
6. **`inference.py`** - Çıkarım scripti (3. çalıştırılacak)

#### İsteğe Bağlı Dosyalar
- **`batch_inference.py`** - Toplu çıkarım (4. çalıştırılacak)
- **`create_labels.py`** - Etiket oluşturma (veri hazırlama)

### 0.5.3 Dosya Açıklamaları

#### Ana Dosyalar
- **`multi_label_dataset.py`**: 
  - Veri yükleme ve ön işleme
  - 5 kanallı veri formatı (RGB + MHI + Flow)
  - Otomatik resize (224x224) ve normalizasyon
  - Çoklu etiket formatı dönüşümü
  - İki mod: birleşik .npy dosyaları ve ayrı dosyalar

- **`multi_label_classifier.py`**:
  - EfficientNet-B0 + CBAM attention modeli
  - Çoklu etiket sınıflandırma için özel tasarım
  - Sigmoid aktivasyon ve BCEWithLogitsLoss
  - 3 sınıf çıkışı: [smoke, fire, none]

- **`training_utils.py`**:
  - Eğitim metrikleri hesaplama
  - ROC eğrileri ve AUC skorları
  - Confusion matrix oluşturma
  - CBAM attention görselleştirme
  - Eğitim grafikleri ve loglama

- **`train_multi_label.py`**:
  - Ana eğitim scripti
  - Command line argument parsing
  - Early stopping ve checkpoint kaydetme
  - TensorBoard loglama
  - Eğitim/validasyon döngüsü

- **`test_model.py`**:
  - Model performans değerlendirme
  - Test metrikleri hesaplama
  - Görselleştirme oluşturma
  - Sonuç raporlama

- **`inference.py`**:
  - Tek görüntü çıkarımı
  - Veri ön işleme
  - Tahmin sonuçları görselleştirme
  - JSON formatında sonuç kaydetme

#### Yardımcı Dosyalar
- **`batch_inference.py`**:
  - Toplu çıkarım işlemi
  - Batch processing optimizasyonu
  - Toplu sonuç raporlama

- **`create_labels.py`**:
  - Etiket dosyası oluşturma
  - Veri seti etiketleme yardımcısı

### 0.5.4 Dosya İlişkileri

```
multi_label_dataset.py
    ↓ (import)
train_multi_label.py
    ↓ (import)
training_utils.py
    ↓ (import)
multi_label_classifier.py
    ↓ (kullanır)
test_model.py
    ↓ (kullanır)
inference.py
    ↓ (kullanır)
batch_inference.py
```

### 0.5.5 Dosya Geliştirme Sırası

#### 1. Temel Bileşenler (Önce geliştirilir)
1. **`multi_label_dataset.py`** - Veri yapısı
2. **`multi_label_classifier.py`** - Model mimarisi
3. **`training_utils.py`** - Yardımcı fonksiyonlar

#### 2. Ana Scriptler (Sonra geliştirilir)
4. **`train_multi_label.py`** - Eğitim
5. **`test_model.py`** - Test
6. **`inference.py`** - Çıkarım

#### 3. Ek Bileşenler (En son geliştirilir)
7. **`batch_inference.py`** - Toplu işlem
8. **`create_labels.py`** - Veri hazırlama

### 0.5.6 Minimum Çalıştırma Sırası

#### Sadece Temel İşlevsellik İçin:
```bash
# 1. Eğitim
python train_multi_label.py --data_dir dataset/train --output_dir runs/basic

# 2. Test
python test_model.py --model_path runs/basic/best_model.pth --test_dir dataset/test

# 3. Çıkarım
python inference.py --model_path runs/basic/best_model.pth --input_path data.npy
```

### 0.5.7 Tam Çalıştırma Sırası

#### Detaylı İşlevsellik İçin:
```bash
# 1. Veri hazırlama
python create_labels.py --data_dir dataset/train --output labels.csv

# 2. Eğitim (ayrı dosyalar ile)
python train_multi_label.py \
    --data_dir dataset/train \
    --image_dir dataset/train/images \
    --mhi_dir dataset/train/mhi \
    --flow_dir dataset/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 10 \
    --output_dir runs/full_experiment

# 3. Eğitim izleme (ayrı terminal)
tensorboard --logdir runs/full_experiment

# 4. Detaylı test
python test_model.py \
    --model_path runs/full_experiment/best_model.pth \
    --test_dir dataset/test \
    --image_dir dataset/test/images \
    --mhi_dir dataset/test/mhi \
    --flow_dir dataset/test/flow \
    --output_dir test_results/full_experiment

# 5. Tek görüntü çıkarımı
python inference.py \
    --model_path runs/full_experiment/best_model.pth \
    --image_path test_image.jpg \
    --mhi_path test_mhi.npy \
    --flow_path test_flow.npy \
    --threshold 0.5 \
    --output_dir predictions/single_test

# 6. Toplu çıkarım
python batch_inference.py \
    --model_path runs/full_experiment/best_model.pth \
    --input_dir dataset/test \
    --output_dir predictions/batch_test
```

### 0.5.8 Hata Durumunda Kontrol Sırası

#### Sorun Giderme:
```bash
# 1. Veri format kontrolü
python -c "
import numpy as np
data = np.load('dataset/train/sample.npy')
print('Veri boyutu:', data.shape)
"

# 2. Dataset testi
python -c "
from multi_label_dataset import MultiLabelDataset
dataset = MultiLabelDataset('dataset/train')
print('Dataset boyutu:', len(dataset))
sample = dataset[0]
print('Örnek boyutu:', sample[0].shape)
print('Etiket:', sample[1])
"

# 3. Model yükleme testi
python -c "
import torch
from multi_label_classifier import MultiLabelFireSmokeClassifier
model = MultiLabelFireSmokeClassifier(num_classes=3, pretrained=False)
print('Model oluşturuldu!')
"

# 4. Training utils testi
python -c "
from training_utils import get_metrics
import torch
outputs = torch.randn(10, 3)
targets = torch.randint(0, 2, (10, 3)).float()
metrics = get_metrics(outputs, targets)
print('Metrikler:', metrics)
"
```

### 0.5.9 Çalıştırma Öncesi Kontrol Listesi

#### Eğitim Öncesi:
- [ ] Veri seti organize edildi mi?
- [ ] Etiket dosyası oluşturuldu mu?
- [ ] `multi_label_dataset.py` çalışıyor mu?
- [ ] `multi_label_classifier.py` import edilebiliyor mu?
- [ ] `training_utils.py` fonksiyonları çalışıyor mu?
- [ ] GPU/CPU uyumluluğu kontrol edildi mi?
- [ ] Yeterli disk alanı var mı?

#### Test Öncesi:
- [ ] Eğitim tamamlandı mı?
- [ ] Model dosyası oluştu mu?
- [ ] Test veri seti hazır mı?
- [ ] `test_model.py` çalışıyor mu?

#### Çıkarım Öncesi:
- [ ] Model dosyası mevcut mu?
- [ ] Giriş verisi doğru formatta mı?
- [ ] `inference.py` çalışıyor mu?
- [ ] Çıktı dizini oluşturuldu mu?

## 1. Sistem Mimarisi

### 1.1 Model Yapısı
- **Temel Model**: EfficientNet-B0
- **Attention Mekanizması**: CBAM (Convolutional Block Attention Module)
- **Giriş**: 5 kanal (RGB + MHI + Optical Flow)
- **Çıkış**: 3 sınıf için çoklu etiket [smoke, fire, none]

### 1.2 Etiket Formatı
Sistem çoklu etiket sınıflandırma yapar. Her örnek için etiket vektörü şu şekilde kodlanır:
- Sadece duman → [1, 0, 0]
- Sadece yangın → [0, 1, 0]
- Hem duman hem yangın → [1, 1, 0]
- Hiçbiri (boş) → [0, 0, 1]

### 1.3 CBAM Attention Mekanizması
- **Channel Attention**: Kanal bazında önemli özellikleri vurgular
- **Spatial Attention**: Uzamsal olarak önemli bölgeleri vurgular
- **Kombinasyon**: İki attention mekanizmasının birleşimi ile daha etkili özellik çıkarımı

### 1.4 Model Parametreleri
- **Backbone**: EfficientNet-B0 (pretrained)
- **Giriş Boyutu**: 224x224x5 (sabit)
- **Attention Boyutu**: Backbone çıktısı ile aynı
- **Çıkış Katmanı**: 3 nöronlu tam bağlantılı katman + Sigmoid

## 2. Veri Hazırlama

### 2.1 Veri Formatı
- **Görüntüler**: RGB formatta, 224x224 boyutuna otomatik resize
- **MHI**: Tek kanallı hareket geçmişi görüntüsü
- **Optical Flow**: Tek kanallı optik akış görüntüsü
- **Etiketler**: 3 boyutlu binary vektör [smoke, fire, none]

### 2.2 Veri Organizasyonu
```
dataset/
├── train/
│   ├── images/          # RGB görüntüler
│   ├── mhi/            # MHI görüntüleri
│   ├── flow/           # Optical Flow görüntüleri
│   └── labels.csv      # Etiket dosyası
├── val/
│   ├── images/
│   ├── mhi/
│   ├── flow/
│   └── labels.csv
└── test/
    ├── images/
    ├── mhi/
    ├── flow/
    └── labels.csv
```

### 2.3 Etiket Dosyası Formatı
labels.csv dosyası aşağıdaki formatta olmalıdır:
```
image_name,smoke,fire,none
image1.jpg,1,0,0
image2.jpg,0,1,0
image3.jpg,1,1,0
image4.jpg,0,0,1
```

### 2.4 Veri Ön İşleme
Sistem otomatik olarak şu işlemleri yapar:
- **Resize**: Tüm görüntüler 224x224 boyutuna getirilir
- **Normalizasyon**: RGB için ImageNet mean/std, MHI/Flow için 0.5/0.5
- **Kanal Birleştirme**: RGB(3) + MHI(1) + Flow(1) = 5 kanal
- **Veri Tipi Dönüşümü**: float32 tensör formatına çevrilir

## 3. Model Eğitimi

### 3.1 Eğitim Parametreleri
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Loss Function**: BCEWithLogitsLoss
- **Early Stopping**: 10 epoch sabır
- **LR Scheduler**: ReduceLROnPlateau

### 3.2 Eğitim Komutları

#### Birleşik .npy Dosyaları ile Eğitim
```bash
python train_multi_label.py \
    --data_dir path/to/dataset \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir path/to/save
```

**Örnek Kullanım:**
```bash
python train_multi_label.py \
    --data_dir KOD/D-Fire/train \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 10 \
    --output_dir runs/multi_label_experiment1
```

#### Ayrı Dosyalar ile Eğitim
```bash
python train_multi_label.py \
    --data_dir path/to/dataset \
    --image_dir path/to/images \
    --mhi_dir path/to/mhi \
    --flow_dir path/to/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir path/to/save
```

**Örnek Kullanım:**
```bash
python train_multi_label.py \
    --data_dir KOD/D-Fire/train \
    --image_dir KOD/D-Fire/train/images \
    --mhi_dir KOD/D-Fire/train/mhi \
    --flow_dir KOD/D-Fire/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 10 \
    --output_dir runs/multi_label_experiment2
```

### 3.3 Eğitim İzleme
```bash
# TensorBoard ile eğitim sürecini izleme
tensorboard --logdir runs/multi_label_experiment1

# Eğitim metriklerini kontrol etme
python -c "
import json
with open('runs/multi_label_experiment1/training_history.json', 'r') as f:
    history = json.load(f)
print('Son epoch metrikleri:', history['val_metrics'][-1])
"
```

### 3.4 Eğitim Çıktıları
- Model checkpoint dosyaları
- Eğitim/validasyon metrikleri grafikleri
- ROC eğrileri
- Confusion matrix
- CBAM attention görselleştirmeleri

## 4. Model Değerlendirme

### 4.1 Test Komutları
```bash
python test_model.py \
    --model_path path/to/model.pth \
    --test_dir path/to/test \
    --output_dir path/to/results
```

**Örnek Kullanım:**
```bash
python test_model.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --test_dir KOD/D-Fire/test \
    --output_dir test_results/experiment1
```

### 4.2 Detaylı Test (Ayrı Dosyalar ile)
```bash
python test_model.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --test_dir KOD/D-Fire/test \
    --image_dir KOD/D-Fire/test/images \
    --mhi_dir KOD/D-Fire/test/mhi \
    --flow_dir KOD/D-Fire/test/flow \
    --output_dir test_results/experiment1_detailed
```

### 4.3 Değerlendirme Metrikleri
- Per-class precision/recall
- Per-class F1 score
- Micro/Macro averages
- ROC AUC scores
- Confusion matrices

### 4.4 Görselleştirmeler
- CBAM attention haritaları
- Yanlış sınıflandırılan örnekler
- ROC eğrileri
- Precision-Recall eğrileri

## 5. Çıkarım (Inference)

### 5.1 Tek Görüntü Çıkarımı

#### Birleşik .npy Dosyası ile
```bash
python inference.py \
    --model_path path/to/model.pth \
    --input_path path/to/data.npy \
    --threshold 0.5 \
    --output_dir path/to/output
```

**Örnek Kullanım:**
```bash
python inference.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --input_path KOD/D-Fire/test/sample_data.npy \
    --threshold 0.5 \
    --output_dir predictions/single_test
```

#### Ayrı Dosyalar ile
```bash
python inference.py \
    --model_path path/to/model.pth \
    --image_path path/to/image.jpg \
    --mhi_path path/to/mhi.npy \
    --flow_path path/to/flow.npy \
    --threshold 0.5 \
    --output_dir path/to/output
```

**Örnek Kullanım:**
```bash
python inference.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --image_path KOD/D-Fire/test/images/AoF06723.jpg \
    --mhi_path KOD/D-Fire/test/mhi/AoF06723.npy \
    --flow_path KOD/D-Fire/test/flow/AoF06723.npy \
    --threshold 0.5 \
    --output_dir predictions/single_test_separate
```

### 5.2 Batch Çıkarımı
```bash
python batch_inference.py \
    --model_path path/to/model.pth \
    --input_dir path/to/input \
    --output_dir path/to/output
```

**Örnek Kullanım:**
```bash
python batch_inference.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --input_dir KOD/D-Fire/test \
    --output_dir predictions/batch_test
```

### 5.3 Farklı Eşik Değerleri ile Test
```bash
# Düşük eşik (daha hassas)
python inference.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --input_path KOD/D-Fire/test/sample_data.npy \
    --threshold 0.3 \
    --output_dir predictions/low_threshold

# Yüksek eşik (daha seçici)
python inference.py \
    --model_path runs/multi_label_experiment1/best_model.pth \
    --input_path KOD/D-Fire/test/sample_data.npy \
    --threshold 0.7 \
    --output_dir predictions/high_threshold
```

### 5.4 Çıkarım Eşikleri
- Smoke threshold: 0.5
- Fire threshold: 0.5
- None threshold: 0.5

### 5.5 Çıkarım Çıktıları
- Tahmin sonuçları (JSON formatında)
- Görselleştirmeler (PNG formatında)
- Güven skorları
- Sınıf etiketleri

## 6. Hata Ayıklama

### 6.1 Yaygın Hatalar
- CUDA out of memory
- Veri yükleme hataları
- Model kaydetme/yükleme hataları
- Veri format uyumsuzlukları

### 6.2 Çözüm Önerileri
- Batch size düşürme
- Veri yollarını kontrol etme
- GPU bellek optimizasyonu
- Checkpoint doğrulama
- Veri format kontrolü

### 6.3 Hata Ayıklama Komutları
```bash
# GPU bellek kontrolü
nvidia-smi

# Veri format kontrolü
python -c "
import numpy as np
data = np.load('KOD/D-Fire/test/sample_data.npy')
print('Veri boyutu:', data.shape)
print('Veri tipi:', data.dtype)
print('Min/Max değerler:', data.min(), data.max())
"

# Model yükleme testi
python -c "
import torch
from multi_label_classifier import MultiLabelFireSmokeClassifier
model = MultiLabelFireSmokeClassifier(num_classes=3, pretrained=False)
checkpoint = torch.load('runs/multi_label_experiment1/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print('Model başarıyla yüklendi!')
"
```

## 7. Performans İyileştirme

### 7.1 Veri Artırma (Augmentation)
- Rastgele döndürme
- Rastgele kırpma
- Parlaklık/kontrast ayarı
- Gürültü ekleme

### 7.2 Model Optimizasyonu
- Gradient clipping
- Weight decay
- Learning rate scheduling
- Model pruning

### 7.3 Çıkarım Optimizasyonu
- Model quantization
- TensorRT dönüşümü
- Batch inference
- CPU/GPU optimizasyonu

## 8. Kod Yapısı

### 8.1 Ana Dosyalar
- `multi_label_dataset.py`: Veri yükleme ve ön işleme
- `multi_label_classifier.py`: Model mimarisi
- `train_multi_label.py`: Eğitim scripti
- `inference.py`: Çıkarım scripti
- `test_model.py`: Test ve değerlendirme

### 8.2 Dataset Sınıfları
- `MultiLabelDataset`: Birleşik .npy dosyaları için
- `MultiLabelDatasetWithImages`: Ayrı dosyalar için

### 8.3 Veri Ön İşleme
- Otomatik resize (224x224)
- Normalizasyon (RGB: ImageNet, MHI/Flow: 0.5/0.5)
- Kanal birleştirme (5 kanal)
- Veri tipi dönüşümü

## 9. Örnek Kullanım Senaryoları

### 9.1 Yeni Veri Seti ile Eğitim
1. Veriyi uygun formatta organize et
2. Etiket dosyasını hazırla
3. Eğitim komutunu çalıştır
4. Sonuçları değerlendir

**Örnek Komutlar:**
```bash
# 1. Veri organizasyonu
mkdir -p dataset/train/{images,mhi,flow}
mkdir -p dataset/val/{images,mhi,flow}
mkdir -p dataset/test/{images,mhi,flow}

# 2. Etiket dosyası oluşturma
python create_labels.py --data_dir dataset/train --output labels.csv

# 3. Eğitim
python train_multi_label.py \
    --data_dir dataset/train \
    --image_dir dataset/train/images \
    --mhi_dir dataset/train/mhi \
    --flow_dir dataset/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --output_dir runs/new_experiment

# 4. Test
python test_model.py \
    --model_path runs/new_experiment/best_model.pth \
    --test_dir dataset/test \
    --output_dir test_results/new_experiment
```

### 9.2 Mevcut Model ile Tahmin
1. Model dosyasını yükle
2. Giriş verisini hazırla
3. Çıkarım komutunu çalıştır
4. Sonuçları analiz et

**Örnek Komutlar:**
```bash
# 1. Model kontrolü
python -c "
import torch
checkpoint = torch.load('runs/experiment1/best_model.pth', map_location='cpu')
print('Model yüklendi, epoch:', checkpoint.get('epoch', 'N/A'))
"

# 2. Tek görüntü tahmini
python inference.py \
    --model_path runs/experiment1/best_model.pth \
    --image_path test_image.jpg \
    --mhi_path test_mhi.npy \
    --flow_path test_flow.npy \
    --threshold 0.5 \
    --output_dir predictions/test

# 3. Toplu tahmin
python batch_inference.py \
    --model_path runs/experiment1/best_model.pth \
    --input_dir test_dataset \
    --output_dir predictions/batch_test

# 4. Sonuç analizi
python analyze_results.py --results_dir predictions/batch_test
```

### 9.3 Model Performans Değerlendirmesi
1. Test veri setini hazırla
2. Test komutunu çalıştır
3. Metrikleri incele
4. Görselleştirmeleri analiz et

**Örnek Komutlar:**
```bash
# 1. Test veri seti hazırlama
python prepare_test_data.py \
    --input_dir raw_test_data \
    --output_dir processed_test_data

# 2. Model testi
python test_model.py \
    --model_path runs/experiment1/best_model.pth \
    --test_dir processed_test_data \
    --output_dir test_results/experiment1

# 3. Metrik analizi
python analyze_metrics.py \
    --results_dir test_results/experiment1 \
    --output_dir analysis/experiment1

# 4. Görselleştirme
python create_visualizations.py \
    --results_dir test_results/experiment1 \
    --output_dir visualizations/experiment1
```

## 10. Hızlı Başlangıç

### 10.1 Minimum Kurulum
```bash
# 1. Bağımlılıkları yükle
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm

# 2. Veri setini hazırla
python prepare_dataset.py --input_dir raw_data --output_dir dataset

# 3. Model eğitimi
python train_multi_label.py \
    --data_dir dataset/train \
    --batch_size 16 \
    --epochs 50 \
    --output_dir runs/quick_start

# 4. Test
python test_model.py \
    --model_path runs/quick_start/best_model.pth \
    --test_dir dataset/test \
    --output_dir test_results/quick_start
```

### 10.2 Tam Kurulum
```bash
# 1. Tüm bağımlılıkları yükle
pip install -r requirements.txt

# 2. Veri setini hazırla
python prepare_dataset.py --input_dir raw_data --output_dir dataset --full_preprocessing

# 3. Detaylı eğitim
python train_multi_label.py \
    --data_dir dataset/train \
    --image_dir dataset/train/images \
    --mhi_dir dataset/train/mhi \
    --flow_dir dataset/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 10 \
    --output_dir runs/full_experiment

# 4. Kapsamlı test
python test_model.py \
    --model_path runs/full_experiment/best_model.pth \
    --test_dir dataset/test \
    --image_dir dataset/test/images \
    --mhi_dir dataset/test/mhi \
    --flow_dir dataset/test/flow \
    --output_dir test_results/full_experiment
```

## 11. Özet ve Hızlı Referans

### 11.1 Tüm Komutların Özeti

#### Veri Hazırlama
```bash
# Veri organizasyonu
mkdir -p dataset/{train,val,test}/{images,mhi,flow}

# Etiket dosyası oluşturma
python create_labels.py --data_dir dataset/train --output labels.csv
```

#### Model Eğitimi
```bash
# Birleşik .npy dosyaları ile
python train_multi_label.py \
    --data_dir dataset/train \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir runs/experiment1

# Ayrı dosyalar ile
python train_multi_label.py \
    --data_dir dataset/train \
    --image_dir dataset/train/images \
    --mhi_dir dataset/train/mhi \
    --flow_dir dataset/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir runs/experiment2
```

#### Model Testi
```bash
# Temel test
python test_model.py \
    --model_path runs/experiment1/best_model.pth \
    --test_dir dataset/test \
    --output_dir test_results

# Detaylı test
python test_model.py \
    --model_path runs/experiment1/best_model.pth \
    --test_dir dataset/test \
    --image_dir dataset/test/images \
    --mhi_dir dataset/test/mhi \
    --flow_dir dataset/test/flow \
    --output_dir test_results_detailed
```

#### Çıkarım (Inference)
```bash
# Birleşik .npy dosyası ile
python inference.py \
    --model_path runs/experiment1/best_model.pth \
    --input_path data.npy \
    --threshold 0.5 \
    --output_dir predictions

# Ayrı dosyalar ile
python inference.py \
    --model_path runs/experiment1/best_model.pth \
    --image_path image.jpg \
    --mhi_path mhi.npy \
    --flow_path flow.npy \
    --threshold 0.5 \
    --output_dir predictions

# Toplu çıkarım
python batch_inference.py \
    --model_path runs/experiment1/best_model.pth \
    --input_dir dataset/test \
    --output_dir batch_predictions
```

### 11.2 Tam Çalıştırma Sırası

#### Adım 1: Sistem Kurulumu
```bash
# 1. Bağımlılıkları yükle
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm

# 2. Veri setini hazırla
python prepare_dataset.py --input_dir raw_data --output_dir dataset

# 3. Etiket dosyasını oluştur
python create_labels.py --data_dir dataset/train --output labels.csv
```

#### Adım 2: Model Eğitimi
```bash
# 4. Model eğitimi
python train_multi_label.py \
    --data_dir dataset/train \
    --image_dir dataset/train/images \
    --mhi_dir dataset/train/mhi \
    --flow_dir dataset/train/flow \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 10 \
    --output_dir runs/final_experiment

# 5. Eğitim sürecini izle
tensorboard --logdir runs/final_experiment
```

#### Adım 3: Model Değerlendirme
```bash
# 6. Model testi
python test_model.py \
    --model_path runs/final_experiment/best_model.pth \
    --test_dir dataset/test \
    --image_dir dataset/test/images \
    --mhi_dir dataset/test/mhi \
    --flow_dir dataset/test/flow \
    --output_dir test_results/final_experiment

# 7. Sonuçları analiz et
python analyze_results.py --results_dir test_results/final_experiment
```

#### Adım 4: Çıkarım ve Kullanım
```bash
# 8. Tek görüntü tahmini
python inference.py \
    --model_path runs/final_experiment/best_model.pth \
    --image_path test_image.jpg \
    --mhi_path test_mhi.npy \
    --flow_path test_flow.npy \
    --threshold 0.5 \
    --output_dir predictions/single_test

# 9. Toplu tahmin
python batch_inference.py \
    --model_path runs/final_experiment/best_model.pth \
    --input_dir dataset/test \
    --output_dir predictions/batch_test

# 10. Sonuç raporu oluştur
python generate_report.py --results_dir predictions/batch_test
```

### 11.3 Önemli Parametreler

#### Eğitim Parametreleri
- `--batch_size`: 16-64 arası (GPU belleğine göre)
- `--epochs`: 50-200 arası (veri setine göre)
- `--lr`: 1e-4 ile 1e-3 arası
- `--patience`: 5-15 arası (early stopping için)

#### Çıkarım Parametreleri
- `--threshold`: 0.3-0.7 arası (hassasiyet için)
- `--target_size`: [224, 224] (sabit boyut)

#### Veri Formatları
- **Birleşik**: .npy dosyası (5, H, W) formatında
- **Ayrı**: RGB (.jpg), MHI (.npy), Flow (.npy)

### 11.4 Hata Ayıklama Komutları

#### GPU Kontrolü
```bash
nvidia-smi
```

#### Veri Format Kontrolü
```bash
python -c "
import numpy as np
data = np.load('data.npy')
print('Boyut:', data.shape)
print('Tip:', data.dtype)
print('Min/Max:', data.min(), data.max())
"
```

#### Model Yükleme Testi
```bash
python -c "
import torch
from multi_label_classifier import MultiLabelFireSmokeClassifier
model = MultiLabelFireSmokeClassifier(num_classes=3, pretrained=False)
checkpoint = torch.load('model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print('Model başarıyla yüklendi!')
"
```

### 11.5 Çıktı Dosyaları

#### Eğitim Çıktıları
- `best_model.pth`: En iyi model
- `training_history.json`: Eğitim metrikleri
- `loss_curves.png`: Kayıp eğrileri
- `accuracy_curves.png`: Doğruluk eğrileri

#### Test Çıktıları
- `metrics.json`: Test metrikleri
- `roc_curves.png`: ROC eğrileri
- `confusion_matrices.png`: Karışıklık matrisleri
- `attention_maps.png`: CBAM attention haritaları

#### Çıkarım Çıktıları
- `prediction_TIMESTAMP.json`: Tahmin sonuçları
- `prediction_TIMESTAMP.png`: Görselleştirmeler
- `batch_predictions_TIMESTAMP.json`: Toplu sonuçlar
- `batch_report_TIMESTAMP.txt`: Detaylı rapor

### 11.6 Performans İpuçları

#### Eğitim Optimizasyonu
- GPU bellek yetersizse batch_size düşür
- Eğitim yavaşsa num_workers artır
- Overfitting varsa patience düşür
- Underfitting varsa epochs artır

#### Çıkarım Optimizasyonu
- Hızlı çıkarım için batch mode kullan
- Hassas tahmin için threshold ayarla
- Bellek tasarrufu için CPU kullan

#### Veri Optimizasyonu
- Veri artırma için augmentation kullan
- Sınıf dengesizliği varsa weighting uygula
- Veri kalitesi için preprocessing kontrol et

---

**Not**: Bu kılavuz, yangın ve duman çoklu etiket sınıflandırma sisteminin tam kullanımını kapsar. Herhangi bir sorun yaşarsanız, hata ayıklama bölümünü kontrol edin veya sistem loglarını inceleyin. 

## 1. Model Eğitimi (Güncel)

Artık model eğitimi sırasında farklı backbone, loss ve scheduler seçenekleri kullanılabilir.

### Desteklenen Model Mimarileri (Backbone)
- EfficientNet-B4 + CBAM: `--model_name efficientnet_b4_cbam`
- ResNet50 + SE: `--model_name resnet50_se`
- SwinTransformer + CBAM: `--model_name swintransformer_cbam`

### Desteklenen Loss Fonksiyonları
- BCEWithLogitsLoss: `--loss bce`
- Focal Loss: `--loss focal`
- Label Smoothing Loss: `--loss smooth`

### Desteklenen Scheduler Seçenekleri
- StepLR: `--scheduler step`
- CosineAnnealingLR: `--scheduler cosine`

### Ensemble Desteği
- Tüm modelleri birlikte eğitmek/test etmek için: `--ensemble` (geliştirilebilir)

### Örnek Eğitim Komutları

#### EfficientNet-B4 + CBAM
```bash
python train_multi_label.py \
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

#### ResNet50 + SE
```bash
python train_multi_label.py \
  --train_npy_dir scene_split_output_v4/train/flow \
  --train_labels_dir scene_split_output_v4/train/flow_labels \
  --val_npy_dir scene_split_output_v4/val/flow_val \
  --val_labels_dir scene_split_output_v4/val/flow_val_labels \
  --output_dir runs/resnet50_se \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --model_name resnet50_se \
  --scheduler step \
  --loss bce \
  --patience 5 \
  --checkpoint_interval 5
```

#### SwinTransformer + CBAM
```bash
python train_multi_label.py \
  --train_npy_dir scene_split_output_v4/train/flow \
  --train_labels_dir scene_split_output_v4/train/flow_labels \
  --val_npy_dir scene_split_output_v4/val/flow_val \
  --val_labels_dir scene_split_output_v4/val/flow_val_labels \
  --output_dir runs/swintransformer_cbam \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --model_name swintransformer_cbam \
  --scheduler cosine \
  --loss smooth \
  --patience 5 \
  --checkpoint_interval 5
```

### Parametre Açıklamaları
- `--model_name`: Model mimarisi seçimi (efficientnet_b4_cbam, resnet50_se, swintransformer_cbam)
- `--loss`: Kayıp fonksiyonu seçimi (bce, focal, smooth)
- `--scheduler`: Öğrenme oranı zamanlayıcı tipi (step, cosine)
- `--ensemble`: (isteğe bağlı) Ensemble eğitim/test
- `--patience`: Early stopping için sabır
- `--checkpoint_interval`: Kaç epoch'ta bir checkpoint kaydedilsin

### Notlar
- 5-kanal giriş (RGB + MHI + Flow) tüm modellerde desteklenir.
- CBAM ve SE attention modülleri otomatik entegre edilir.
- Tüm loss ve scheduler seçenekleri komut satırından değiştirilebilir.
- Test ve inference için de benzer parametreler kullanılabilir. 
