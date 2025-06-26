#!/usr/bin/env python3
"""
Live Detection Pipeline Test Script
Bu script pipeline'ın temel işlevlerini test eder
"""

import cv2
import numpy as np
import torch
import time
from live_detection_pipeline import LiveDetectionPipeline

def create_test_video():
    """Test için basit video oluştur"""
    print("Test video oluşturuluyor...")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (640, 480))
    
    # Basit animasyonlu video oluştur
    for i in range(100):  # 5 saniye (20fps)
        # Arka plan
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Hareketli daire (simüle edilmiş hareket)
        x = int(320 + 200 * np.sin(i * 0.1))
        y = int(240 + 100 * np.cos(i * 0.15))
        
        # Farklı renkler (bazen kırmızı/turuncu - ateş simülasyonu)
        if i % 30 < 15:
            color = (0, 0, 255)  # Kırmızı
        else:
            color = (0, 255, 255)  # Sarı
        
        cv2.circle(frame, (x, y), 30, color, -1)
        
        # Metin ekle
        cv2.putText(frame, f"Frame: {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("Test video oluşturuldu: test_video.mp4")

def test_pipeline_components():
    """Pipeline bileşenlerini test et"""
    print("\n=== Pipeline Bileşenleri Testi ===")
    
    # Pipeline oluştur
    pipeline = LiveDetectionPipeline(
        cnn_model_path="runs/exp_scene_split/best_model.pth",
        yolo_model_path="yolov8n.pt",
        device="cpu",  # Test için CPU kullan
        enable_logging=False
    )
    
    # Test frame oluştur
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Frame buffer'ı doldur
    for _ in range(5):
        pipeline.frame_buffer.append(test_frame.copy())
    
    print(f"Frame buffer boyutu: {len(pipeline.frame_buffer)}")
    
    # MHI test
    frames_list = list(pipeline.frame_buffer)
    mhi = pipeline.compute_mhi(frames_list)
    if mhi is not None:
        print(f"MHI shape: {mhi.shape}")
        print(f"MHI range: [{mhi.min():.3f}, {mhi.max():.3f}]")
    else:
        print("MHI hesaplanamadı")
    
    # Optical Flow test
    optical_flow = pipeline.compute_optical_flow(frames_list)
    if optical_flow is not None:
        print(f"Optical Flow shape: {optical_flow.shape}")
        print(f"Optical Flow range: [{optical_flow.min():.3f}, {optical_flow.max():.3f}]")
    else:
        print("Optical Flow hesaplanamadı")
    
    # Input tensor test
    if mhi is not None and optical_flow is not None:
        input_tensor = pipeline.create_input_tensor(test_frame, mhi, optical_flow)
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        # CNN test (eğer model yüklendiyse)
        try:
            scene_pred = pipeline.predict_scene(input_tensor)
            print(f"Scene prediction: {scene_pred}")
        except Exception as e:
            print(f"CNN test hatası: {e}")
    
    print("Pipeline bileşenleri testi tamamlandı")

def test_with_video():
    """Video ile test et"""
    print("\n=== Video Testi ===")
    
    # Test video oluştur
    create_test_video()
    
    # Pipeline oluştur
    pipeline = LiveDetectionPipeline(
        cnn_model_path="runs/exp_scene_split/best_model.pth",
        yolo_model_path="yolov8n.pt",
        device="cpu",
        enable_logging=True,
        threshold=0.3  # Daha hassas test için
    )
    
    # Video ile test et
    print("Video testi başlıyor...")
    print("Çıkmak için 'q' tuşuna basın")
    
    try:
        pipeline.run_video(video_source="test_video.mp4", output_path="test_output.mp4")  # type: ignore
    except Exception as e:
        print(f"Video testi hatası: {e}")

def test_camera():
    """Kamera ile test et"""
    print("\n=== Kamera Testi ===")
    
    # Kamera erişimini test et
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera erişilemiyor, test atlanıyor")
        return
    
    # Birkaç frame al
    print("Kamera testi - 5 saniye...")
    start_time = time.time()
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Kamera testi tamamlandı")

def main():
    """Ana test fonksiyonu"""
    print("=== Live Detection Pipeline Test ===")
    print("Bu script pipeline'ın temel işlevlerini test eder")
    
    # Test seçenekleri
    print("\nTest seçenekleri:")
    print("1. Pipeline bileşenleri testi")
    print("2. Video ile test")
    print("3. Kamera testi")
    print("4. Tüm testler")
    print("5. Çıkış")
    
    choice = input("\nSeçiminizi yapın (1-5): ").strip()
    
    if choice == "1":
        test_pipeline_components()
    elif choice == "2":
        test_with_video()
    elif choice == "3":
        test_camera()
    elif choice == "4":
        test_pipeline_components()
        test_with_video()
        test_camera()
    elif choice == "5":
        print("Test sonlandırılıyor...")
        return
    else:
        print("Geçersiz seçim")
    
    print("\nTest tamamlandı!")

if __name__ == "__main__":
    main() 