#!/usr/bin/env python3
"""
YOLO Eğitimi Test Script'i
"""

import os
import sys
from pathlib import Path

def test_yolo_imports():
    """YOLO eğitimi için gerekli paketleri test et"""
    print("🔍 YOLO eğitimi paketleri test ediliyor...")
    
    try:
        import ultralytics
        print(f"✅ ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ ultralytics import hatası: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas import hatası: {e}")
        return False
    
    try:
        import yaml
        print("✅ pyyaml")
    except ImportError as e:
        print(f"❌ pyyaml import hatası: {e}")
        return False
    
    return True

def test_dataset_structure():
    """Dataset yapısını test et"""
    print("\n🔍 Dataset yapısı test ediliyor...")
    
    # Gerekli dosyaları kontrol et
    required_dirs = [
        "data_split/train/images",
        "data_split/train/labels", 
        "data_split/val/images",
        "data_split/val/labels"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} bulunamadı")
            return False
    
    return True

def test_yolo_script():
    """YOLO eğitimi script'ini test et"""
    print("\n🔍 YOLO eğitimi script'i test ediliyor...")
    
    if not os.path.exists("yolo_training.py"):
        print("❌ yolo_training.py bulunamadı")
        return False
    
    try:
        # Script'i import et
        import yolo_training
        print("✅ yolo_training.py import başarılı")
        
        # Fonksiyonları kontrol et
        required_functions = [
            'create_yolo_dataset_structure',
            'convert_annotations_to_yolo_format',
            'prepare_yolo_dataset',
            'create_yolo_config',
            'train_yolo_model',
            'validate_yolo_model'
        ]
        
        for func_name in required_functions:
            if hasattr(yolo_training, func_name):
                print(f"✅ {func_name} fonksiyonu mevcut")
            else:
                print(f"❌ {func_name} fonksiyonu bulunamadı")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ yolo_training.py import hatası: {e}")
        return False

def test_yolo_model_creation():
    """YOLO model oluşturmayı test et"""
    print("\n🔍 YOLO model oluşturma test ediliyor...")
    
    try:
        from ultralytics import YOLO
        
        # Küçük model oluştur
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model oluşturma başarılı")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model oluşturma hatası: {e}")
        return False

def test_sample_training():
    """Örnek eğitim test et (küçük dataset ile)"""
    print("\n🔍 Örnek eğitim test ediliyor...")
    
    # Test için küçük bir dataset oluştur
    test_dir = Path("test_yolo_dataset")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Test config oluştur
        import yaml
        
        test_config = {
            'path': str(test_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'fire',
                1: 'smoke',
                2: 'smoke+fire'
            },
            'nc': 3
        }
        
        config_path = test_dir / 'test_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
        
        print("✅ Test config oluşturuldu")
        
        # Test dizinini temizle
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Test eğitim hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 YOLO Eğitimi Test Başlatılıyor...\n")
    
    tests = [
        ("Paket Importları", test_yolo_imports),
        ("Dataset Yapısı", test_dataset_structure),
        ("YOLO Script", test_yolo_script),
        ("Model Oluşturma", test_yolo_model_creation),
        ("Örnek Eğitim", test_sample_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name} test ediliyor...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} başarılı\n")
        else:
            print(f"❌ {test_name} başarısız\n")
    
    print(f"📊 Test Sonuçları: {passed}/{total} başarılı")
    
    if passed == total:
        print("\n🎉 Tüm testler başarılı! YOLO eğitimi hazır.")
        print("\n📝 Kullanım örneği:")
        print("python yolo_training.py \\")
        print("  --data_dir data_split \\")
        print("  --output_dir yolo_training \\")
        print("  --epochs 50")
    else:
        print(f"\n⚠️ {total - passed} test başarısız. Lütfen hataları düzeltin.")
        print("\n🔧 Öneriler:")
        print("1. requirements.txt'yi güncelleyin")
        print("2. Dataset dosyalarını kontrol edin")
        print("3. YOLO script'ini kontrol edin")

if __name__ == "__main__":
    main() 