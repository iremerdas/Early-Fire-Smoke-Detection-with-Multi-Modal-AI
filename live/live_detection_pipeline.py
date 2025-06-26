import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
import argparse
import os
from pathlib import Path
import logging
from ultralytics import YOLO
from typing import Union

# Mevcut model sınıflarını import et
from multi_label_classifier import MultiLabelClassifier
from cbam_module import CBAM2D

class LiveDetectionPipeline:
    """
    Gerçek zamanlı duman/ateş tespiti ve nesne tespiti pipeline'ı
    """
    
    def __init__(self, 
                 cnn_model_path="runs/exp_scene_split/best_model.pth",
                 yolo_model_path="yolov8n.pt",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 sequence_length=5,
                 threshold=0.5,
                 enable_logging=True):
        
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.device = device
        self.enable_logging = enable_logging
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Model yükleme
        self.load_models(cnn_model_path, yolo_model_path)
        
        # Logging
        if enable_logging:
            self.setup_logging()
        
        # İstatistikler
        self.stats = {
            'total_frames': 0,
            'positive_scenes': 0,
            'detections': 0,
            'fps': 0.0
        }
        
        print(f"Pipeline başlatıldı - Device: {device}")
        print(f"CNN Model: {cnn_model_path}")
        print(f"YOLO Model: {yolo_model_path}")
    
    def load_models(self, cnn_model_path, yolo_model_path):
        """CNN ve YOLO modellerini yükle"""
        print("Modeller yükleniyor...")
        
        # CNN Model (Multi-label classifier)
        self.cnn_model = MultiLabelClassifier(num_classes=3, pretrained=False, model_name='efficientnet_b4_cbam')
        
        if os.path.exists(cnn_model_path):
            checkpoint = torch.load(cnn_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.cnn_model.load_state_dict(checkpoint)
            print(f"CNN model yüklendi: {cnn_model_path}")
        else:
            print(f"Uyarı: CNN model bulunamadı: {cnn_model_path}")
        
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # YOLO Model
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"YOLO model yüklendi: {yolo_model_path}")
        except Exception as e:
            print(f"YOLO model yüklenemedi: {e}")
            self.yolo_model = None
    
    def setup_logging(self):
        """Logging sistemi kurulumu"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('detection_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def compute_mhi(self, frames):
        """Motion History Image hesapla"""
        if len(frames) < 2:
            return None
        
        height, width = frames[0].shape[:2]
        mhi = np.zeros((height, width), dtype=np.float32)
        tau = 30  # MHI parametresi
        
        for i in range(len(frames) - 1):
            # Gri tonlamaya çevir
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            
            # Fark görüntüsü
            diff = cv2.absdiff(gray1, gray2)
            
            # Eşikleme
            _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.float32)
            
            # MHI güncelle
            mhi = cv2.addWeighted(mhi, 1.0, motion_mask, 1.0, 0)
            mhi = np.clip(mhi, 0, tau)
        
        # Normalize et
        mhi = mhi / tau
        return mhi
    
    def compute_optical_flow(self, frames):
        """Optical Flow magnitude hesapla"""
        if len(frames) < 2:
            return None
        
        flows = []
        target_shape = None
        
        for i in range(len(frames) - 1):
            # Gri tonlamaya çevir
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            
            # Optical flow hesapla
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )  # type: ignore
            
            # Magnitude hesapla
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            if target_shape is None:
                target_shape = mag.shape
            else:
                if mag.shape != target_shape:
                    mag = cv2.resize(mag, (target_shape[1], target_shape[0]))
            
            flows.append(mag)
        
        # Ortalama magnitude
        avg_mag = np.mean(flows, axis=0)
        
        # Normalize et
        if np.max(avg_mag) > 0:
            avg_mag = avg_mag / np.max(avg_mag)
        
        return avg_mag.astype(np.float32)
    
    def create_input_tensor(self, rgb_frame, mhi, optical_flow):
        """CNN için 5-kanal input tensor oluştur"""
        # RGB frame'i normalize et
        rgb_norm = rgb_frame.astype(np.float32) / 255.0
        
        # Resize to 224x224
        rgb_norm = cv2.resize(rgb_norm, (224, 224))
        
        # MHI'yi resize et ve normalize et
        mhi_resized = cv2.resize(mhi, (224, 224))
        
        # Optical flow'u resize et
        flow_resized = cv2.resize(optical_flow, (224, 224))
        
        # 5-kanal tensor oluştur: [R, G, B, MHI, Flow]
        combined = np.concatenate([rgb_norm, mhi_resized[..., np.newaxis], flow_resized[..., np.newaxis]], axis=-1)
        
        # PyTorch formatına çevir: (C, H, W)
        combined = np.transpose(combined, (2, 0, 1))
        
        # Batch dimension ekle: (1, C, H, W)
        combined = np.expand_dims(combined, axis=0)
        
        return torch.from_numpy(combined).float()
    
    def predict_scene(self, input_tensor):
        """CNN ile sahne sınıflandırması"""
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.cnn_model(input_tensor)
            probabilities = torch.sigmoid(outputs)
            
            # Multi-label çıktı: [smoke, fire, none]
            smoke_prob = probabilities[0, 0].item()
            fire_prob = probabilities[0, 1].item()
            none_prob = probabilities[0, 2].item()
            
            # Pozitif sahne kontrolü
            is_positive = (smoke_prob - 0.4 > 1e-8) or (fire_prob - 0.4 > 1e-8)
            
            return {
                'smoke_prob': smoke_prob,
                'fire_prob': fire_prob,
                'none_prob': none_prob,
                'is_positive': is_positive,
                'predictions': probabilities.cpu().numpy()[0]
            }
    
    def detect_objects(self, frame):
        """YOLO ile nesne tespiti"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Sadece fire ve smoke sınıflarını al
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # YOLO sınıf isimlerini kontrol et (fire, smoke olabilir)
                        class_names = result.names
                        class_name = class_names.get(cls, f"class_{cls}")
                        
                        # Fire/smoke ile ilgili sınıfları filtrele
                        if any(keyword in class_name.lower() for keyword in ['fire', 'smoke', 'flame']):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class': class_name,
                                'class_id': cls
                            })
            
            return detections
        except Exception as e:
            print(f"YOLO tespit hatası: {e}")
            return []
    
    def draw_detections(self, frame, scene_pred, detections):
        """Tespitleri frame üzerine çiz"""
        # Yazı ve kutu boyutlarını büyüt
        font_scale = 0.5
        thickness = 1
        scene_text = ""
        smoke_pos = scene_pred['smoke_prob'] > self.threshold if scene_pred else False
        fire_pos = scene_pred['fire_prob'] > self.threshold if scene_pred else False
        if scene_pred:
            if smoke_pos:
                scene_text += f"Smoke: {scene_pred['smoke_prob']:.4f} "
            if fire_pos:
                scene_text += f"Fire: {scene_pred['fire_prob']:.4f}"
        if scene_text:
            cv2.putText(frame, scene_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        if smoke_pos and fire_pos:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 255), thickness)
            cv2.putText(frame, "ALERT: FIRE & SMOKE DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)
        elif fire_pos:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness)
            cv2.putText(frame, "ALERT: FIRE DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        elif smoke_pos:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), thickness)
            cv2.putText(frame, "ALERT: SMOKE DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (0, 255, 0), thickness)
        return frame
    
    def log_detection(self, scene_pred, detections):
        """Tespit bilgilerini logla"""
        if self.enable_logging:
            log_msg = f"Scene: Smoke={scene_pred['smoke_prob']:.3f}, Fire={scene_pred['fire_prob']:.3f}, Positive={scene_pred['is_positive']}"
            if detections:
                log_msg += f", Objects: {len(detections)}"
            self.logger.info(log_msg)
    
    def process_frame(self, frame):
        """Tek frame işle"""
        # Frame'i buffer'a ekle
        self.frame_buffer.append(frame.copy())
        
        # Yeterli frame yoksa bekle
        if len(self.frame_buffer) < self.sequence_length:
            return frame, None, []
        
        # MHI ve Optical Flow hesapla
        frames_list = list(self.frame_buffer)
        mhi = self.compute_mhi(frames_list)
        optical_flow = self.compute_optical_flow(frames_list)
        
        if mhi is None or optical_flow is None:
            return frame, None, []
        
        # Ortadaki frame'i al
        middle_frame = frames_list[self.sequence_length // 2]
        
        # Input tensor oluştur
        input_tensor = self.create_input_tensor(middle_frame, mhi, optical_flow)
        
        # CNN ile sahne sınıflandırması
        scene_prediction = self.predict_scene(input_tensor)
        
        # İstatistikleri güncelle
        self.stats['total_frames'] += 1
        if scene_prediction['is_positive']:
            self.stats['positive_scenes'] += 1
        
        # Pozitif sahne ise YOLO tespiti yap
        detections = []
        # FPS'i artırmak için YOLO'yu her 3 frame'de bir çalıştır
        if scene_prediction['is_positive'] and (self.stats['total_frames'] % 3 == 0):
            detections = self.detect_objects(middle_frame)
            self.stats['detections'] += len(detections)
        
        # Log
        self.log_detection(scene_prediction, detections)
        
        # Frame'i çiz
        output_frame = self.draw_detections(middle_frame, scene_prediction, detections)
        
        return output_frame, scene_prediction, detections
    
    def run_video(self, video_source: Union[int, str] = 0, output_path=None):
        """Video akışını işle"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Video açılamadı: {video_source}")
            return
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width, height = 224, 224
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Video işleme başladı. Çıkmak için 'q' tuşuna basın.")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # FPS için frame'i küçült
                frame = cv2.resize(frame, (224, 224))
                # Frame işle
                output_frame, scene_pred, detections = self.process_frame(frame)
                
                # FPS hesapla
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    self.stats['fps'] = frame_count / elapsed_time
                
                # FPS bilgisini çiz
                cv2.putText(output_frame, f"FPS: {self.stats['fps']:.1f}", 
                           (10, output_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # İstatistikleri çiz
                stats_text = f"Positive: {self.stats['positive_scenes']}, Detections: {self.stats['detections']}"
                cv2.putText(output_frame, stats_text, 
                           (10, output_frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Frame'i göster
                cv2.imshow('Live Detection Pipeline', output_frame)
                
                # Video'ya kaydet
                if writer:
                    # Emin olmak için frame'i uint8 ve BGR olarak kaydet
                    out_frame = output_frame
                    if out_frame.dtype != np.uint8:
                        out_frame = (np.clip(out_frame, 0, 255)).astype(np.uint8)
                    if out_frame.shape[0:2] != (224, 224):
                        out_frame = cv2.resize(out_frame, (224, 224))
                    writer.write(out_frame)
                
                # Çıkış kontrolü
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nKullanıcı tarafından durduruldu.")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Final istatistikler
            print(f"\nFinal İstatistikler:")
            print(f"Toplam Frame: {self.stats['total_frames']}")
            print(f"Pozitif Sahne: {self.stats['positive_scenes']}")
            print(f"Toplam Tespit: {self.stats['detections']}")
            print(f"Ortalama FPS: {self.stats['fps']:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Live Fire/Smoke Detection Pipeline')
    parser.add_argument('--video', type=str, default='0', help='Video dosyası veya kamera (0)')
    parser.add_argument('--output', type=str, help='Çıktı video dosyası')
    parser.add_argument('--cnn_model', type=str, default='runs/exp_scene_split/best_model.pth', 
                       help='CNN model dosyası')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', 
                       help='YOLO model dosyası')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Pozitif sahne eşiği')
    parser.add_argument('--sequence_length', type=int, default=5, 
                       help='MHI için frame sayısı')
    parser.add_argument('--no_logging', action='store_true', 
                       help='Logging sistemini devre dışı bırak')
    
    args = parser.parse_args()
    
    # Video source belirle
    video_source = 0 if args.video == '0' else args.video
    
    # Pipeline oluştur
    pipeline = LiveDetectionPipeline(
        cnn_model_path=args.cnn_model,
        yolo_model_path=args.yolo_model,
        threshold=args.threshold,
        sequence_length=args.sequence_length,
        enable_logging=not args.no_logging
    )
    
    # Pipeline'ı çalıştır
    pipeline.run_video(video_source=video_source, output_path=args.output)

if __name__ == "__main__":
    main() 