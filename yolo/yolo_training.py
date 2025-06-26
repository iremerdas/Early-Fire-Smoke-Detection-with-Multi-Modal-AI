#!/usr/bin/env python3
"""
YOLO Eğitimi Script'i
Fire/Smoke tespiti için YOLO modelini eğitir
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import argparse

def create_yolo_dataset_structure(base_dir, output_dir):
    """
    YOLO formatında dataset yapısı oluştur
    """
    print("YOLO dataset yapısı oluşturuluyor...")
    
    # YOLO dataset yapısı
    yolo_dir = Path(output_dir)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # Alt dizinler
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    return yolo_dir

def convert_annotations_to_yolo_format(annotation_file, output_dir, class_mapping):
    """
    Mevcut etiketleri YOLO formatına çevir
    """
    print(f"Etiketler YOLO formatına çevriliyor: {annotation_file}")
    
    # CSV dosyasını oku
    import pandas as pd
    df = pd.read_csv(annotation_file)
    
    yolo_annotations = []
    
    for _, row in df.iterrows():
        filename = row['filename']
        class_name = row['class']
        
        # Sınıf ID'sini al
        if class_name in class_mapping:
            class_id = class_mapping[class_name]
        else:
            continue  # Bilinmeyen sınıfı atla
        
        # Bounding box koordinatları (normalize edilmiş)
        x_center = row.get('x_center', 0.5)
        y_center = row.get('y_center', 0.5)
        width = row.get('width', 0.1)
        height = row.get('height', 0.1)
        
        # YOLO formatı: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_annotations.append((filename, yolo_line))
    
    return yolo_annotations

def prepare_yolo_dataset(base_dir, output_dir, train_split=0.8):
    """
    Mevcut YOLO formatındaki dataset'i kullan
    """
    print("Mevcut YOLO dataset yapısı kullanılıyor...")
    
    base_dir = Path(base_dir)
    
    # Mevcut yapıyı kontrol et
    train_images = base_dir / "train" / "images"
    train_labels = base_dir / "train" / "labels"
    val_images = base_dir / "val" / "images"
    val_labels = base_dir / "val" / "labels"
    
    if not train_images.exists():
        print(f"❌ Train images dizini bulunamadı: {train_images}")
        return None
    
    if not train_labels.exists():
        print(f"❌ Train labels dizini bulunamadı: {train_labels}")
        return None
    
    if not val_images.exists():
        print(f"❌ Val images dizini bulunamadı: {val_images}")
        return None
    
    if not val_labels.exists():
        print(f"❌ Val labels dizini bulunamadı: {val_labels}")
        return None
    
    print(f"✅ Dataset yapısı doğrulandı:")
    print(f"  Train images: {train_images}")
    print(f"  Train labels: {train_labels}")
    print(f"  Val images: {val_images}")
    print(f"  Val labels: {val_labels}")
    
    # YOLO dataset yapısı oluştur
    yolo_dir = create_yolo_dataset_structure(base_dir, output_dir)
    
    # Dosyaları kopyala
    print("Dosyalar kopyalanıyor...")
    
    # Train dosyalarını kopyala
    for img_file in train_images.glob("*"):
        if img_file.is_file():
            dst_img = yolo_dir / "images" / "train" / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Etiket dosyasını da kopyala
            label_file = train_labels / (img_file.stem + ".txt")
            if label_file.exists():
                dst_label = yolo_dir / "labels" / "train" / label_file.name
                shutil.copy2(label_file, dst_label)
    
    # Val dosyalarını kopyala
    for img_file in val_images.glob("*"):
        if img_file.is_file():
            dst_img = yolo_dir / "images" / "val" / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Etiket dosyasını da kopyala
            label_file = val_labels / (img_file.stem + ".txt")
            if label_file.exists():
                dst_label = yolo_dir / "labels" / "val" / label_file.name
                shutil.copy2(label_file, dst_label)
    
    # Dosya sayılarını say
    train_count = len(list((yolo_dir / "images" / "train").glob("*")))
    val_count = len(list((yolo_dir / "images" / "val").glob("*")))
    
    print(f"Dataset hazırlandı:")
    print(f"  Train: {train_count} örnek")
    print(f"  Val: {val_count} örnek")
    
    return yolo_dir

def create_yolo_config(dataset_dir, output_dir, model_size='n'):
    """
    YOLO config dosyası oluştur
    """
    config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'smoke',
            1: 'fire',
            2: 'none'
        },
        'nc': 3  # sınıf sayısı
    }
    
    config_path = Path(output_dir) / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO config dosyası oluşturuldu: {config_path}")
    return config_path

def train_yolo_model(config_path, output_dir, model_to_load, epochs=100, batch_size=16, imgsz=640, patience=20):
    """
    YOLO modelini eğit
    """
    print("YOLO modeli eğitiliyor...")
    
    # CUDA kontrolü
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'  # GPU device 0 kullan
        print(f"✅ GPU kullanılıyor (Cihaz: {torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        print("⚠️ GPU bulunamadı veya erişilemiyor, CPU kullanılıyor")
        print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    # Model oluştur (ya sıfırdan ya da checkpoint'ten)
    model = YOLO(model_to_load)
    
    # Eğitim parametreleri
    results = model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=output_dir,
        name='fire_smoke_detection',
        patience=patience,  # early stopping
        save=True,
        device=device,
        exist_ok=True  # Var olan projenin üzerine yazmaya izin ver
    )
    
    print("Eğitim tamamlandı!")
    return results

def validate_yolo_model(model_path, config_path):
    """
    Eğitilmiş modeli doğrula
    """
    print("Model doğrulanıyor...")
    
    model = YOLO(model_path)
    results = model.val(data=str(config_path))
    
    print(f"Validation sonuçları:")
    print(f"  mAP50: {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLO Fire/Smoke Detection Training')
    parser.add_argument('--data_dir', type=str,
                       help='YOLO formatında dataset dizini (train/val altında images/labels)')
    parser.add_argument('--output_dir', type=str, default='yolo_training',
                       help='Çıktı dizini')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch boyutu')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Görsel boyutu')
    parser.add_argument('--model_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model boyutu')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (epoch sayısı)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/val split oranı (mevcut yapı kullanıldığında kullanılmaz)')
    parser.add_argument('--validate_only', action='store_true',
                       help='Sadece doğrulama yap')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Eğitime devam etmek için checkpoint dosyası yolu (örn: yolo_training/fire_smoke_detection/weights/last.pt)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / 'dataset.yaml'
    
    if args.resume_from:
        # Checkpoint'ten devam et
        model_to_load = args.resume_from
        if not Path(model_to_load).exists():
            print(f"❌ Checkpoint dosyası bulunamadı: {model_to_load}")
            return
        if not config_path.exists():
            print(f"❌ 'dataset.yaml' config dosyası bulunamadı: {config_path}")
            return
        print(f"Eğitime checkpoint'ten devam ediliyor: {model_to_load}")

    else:
        # Yeni eğitim başlat
        if not args.data_dir:
            print("❌ Yeni bir eğitim için --data_dir parametresi gereklidir.")
            return
            
        print("Yeni eğitim başlatılıyor, dataset hazırlanıyor...")
        dataset_dir = prepare_yolo_dataset(
            args.data_dir, 
            output_dir, 
            args.train_split
        )
        if dataset_dir is None:
            print("❌ Dataset hazırlanamadı!")
            return
        
        config_path = create_yolo_config(dataset_dir, output_dir, args.model_size)
        model_to_load = f'yolov8{args.model_size}.pt'

    # Modeli eğit
    train_yolo_model(
        config_path, 
        output_dir, 
        model_to_load,
        args.epochs, 
        args.batch_size, 
        args.imgsz,
        args.patience
    )
    
    # Doğrulama adımı
    # Ultralytics, aynı isimde yeni klasörler oluşturur (fire_smoke_detection, fire_smoke_detection2, vb.)
    # Bu yüzden en son değiştirilen klasörü bularak doğru modeli doğrulayalım.
    project_dir = Path(args.output_dir)
    try:
        # Proje dizinindeki tüm alt klasörleri al
        subdirs = [d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith('fire_smoke_detection')]
        if not subdirs:
            raise FileNotFoundError

        # En son değiştirilen klasörü bul
        latest_run_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
        best_model_path = latest_run_dir / 'weights' / 'best.pt'

        if best_model_path.exists():
            print(f"\n✅ Eğitim başarıyla tamamlandı (veya Early Stopping ile durduruldu).")
            print("En iyi modelin performansı doğrulanıyor...")
            validate_yolo_model(best_model_path, config_path)
            print(f"\nEn iyi model burada kaydedildi: {best_model_path}")
            print(f"Bu modeli live_detection_pipeline.py'de kullanabilirsiniz.")
        else:
            print(f"❌ Eğitim sonrası 'best.pt' modeli bulunamadı: {best_model_path}")

    except (ValueError, FileNotFoundError):
        print("❌ Eğitim sonrası çıktı klasörü bulunamadı veya beklenen yapıda değil.")

if __name__ == "__main__":
    main() 