import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from pathlib import Path
from fire_smoke_classifier import create_model
from train_classifier import NormalizeTransform
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from multi_label_classifier import MultiLabelFireSmokeClassifier
import torchvision.transforms as transforms
from PIL import Image

class MultiLabelPredictor:
    """Multi-label Fire/Smoke predictor class"""
    def __init__(self, model_path, threshold=0.5, device='auto', target_size=(224, 224)):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.threshold = threshold
        self.target_size = target_size
        self.class_names = ['smoke', 'fire', 'none']
        
        # Load model
        self.model = MultiLabelFireSmokeClassifier(num_classes=3, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {model_path}")
        print(f"Device: {self.device}")
        print(f"Threshold: {self.threshold}")
        print(f"Target size: {self.target_size}")
        
    def preprocess_data(self, data):
        """
        Preprocess input data to match model expectations
        Args:
            data: Input data (numpy array or tensor)
        Returns:
            processed_data: Preprocessed tensor ready for model
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Ensure correct shape (5, H, W)
        if len(data.shape) == 3:
            if data.shape[0] == 5:
                # Already in correct format (5, H, W)
                pass
            elif data.shape[2] == 5:
                # Convert from (H, W, 5) to (5, H, W)
                data = data.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
        else:
            raise ValueError(f"Expected 3D array, got shape: {data.shape}")
        
        # Resize to target size
        data = torch.nn.functional.interpolate(
            data.unsqueeze(0), 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Normalize
        data = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5, 0.5],
            std=[0.229, 0.224, 0.225, 0.5, 0.5]
        )(data)
        
        return data
    
    def load_separate_files(self, image_path, mhi_path, flow_path):
        """
        Load and combine RGB, MHI, and Flow from separate files
        Args:
            image_path: Path to RGB image
            mhi_path: Path to MHI file
            flow_path: Path to Flow file
        Returns:
            combined_data: Combined 5-channel tensor
        """
        # Load RGB image
        rgb_img = Image.open(image_path).convert('RGB')
        rgb_tensor = transforms.ToTensor()(rgb_img)
        
        # Load MHI
        mhi_data = np.load(str(mhi_path))
        if len(mhi_data.shape) == 2:
            mhi_data = mhi_data[np.newaxis, :, :]  # Add channel dimension
        mhi_tensor = torch.from_numpy(mhi_data).float()
        
        # Load Flow
        flow_data = np.load(str(flow_path))
        if len(flow_data.shape) == 2:
            flow_data = flow_data[np.newaxis, :, :]  # Add channel dimension
        flow_tensor = torch.from_numpy(flow_data).float()
        
        # Concatenate channels: RGB(3) + MHI(1) + Flow(1) = 5 channels
        combined = torch.cat([rgb_tensor, mhi_tensor, flow_tensor], dim=0)
        
        # Preprocess
        combined = self.preprocess_data(combined)
        
        return combined
        
    def predict(self, data):
        """
        Predict multi-label classes for input data
        Args:
            data: Input tensor of shape (C, H, W) or (B, C, H, W)
        Returns:
            predictions: Binary predictions for each class
            probabilities: Raw probability scores
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            
            # Move to device
            data = data.to(self.device)
            
            # Get model predictions
            outputs = self.model(data)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # Apply threshold
            predictions = (probabilities >= self.threshold).astype(int)
            
            return predictions, probabilities
    
    def predict_and_visualize(self, data, save_path=None):
        """
        Predict and visualize results
        Args:
            data: Input tensor
            save_path: Optional path to save visualization
        """
        predictions, probabilities = self.predict(data)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot probabilities
        plt.subplot(1, 3, 1)
        plt.bar(self.class_names, probabilities[0])
        plt.title('Class Probabilities')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Plot predictions
        plt.subplot(1, 3, 2)
        plt.bar(self.class_names, predictions[0])
        plt.title('Binary Predictions')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Plot threshold line
        plt.subplot(1, 3, 3)
        plt.bar(self.class_names, probabilities[0])
        plt.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold})')
        plt.title('Probabilities with Threshold')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Print results
        print("\nPrediction Results:")
        for i, class_name in enumerate(self.class_names):
            status = "✓" if predictions[0][i] == 1 else "✗"
            print(f"{class_name}: {probabilities[0][i]:.3f} -> {status}")
        
        return predictions, probabilities

def save_batch_results(results, output_dir):
    """Toplu tahmin sonuçlarını kaydet"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Başarılı tahminleri filtrele
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    # İstatistikler
    class_counts = {}
    for class_name in ['smoke', 'fire', 'none']:
        class_counts[class_name] = sum(1 for r in successful_results 
                                     if r['predicted_class'] == class_name)
    
    # JSON formatında sonuçları kaydet
    batch_results = {
        'timestamp': timestamp,
        'total_files': len(results),
        'successful_predictions': len(successful_results),
        'failed_predictions': len(failed_results),
        'class_distribution': class_counts,
        'predictions': successful_results,
        'errors': failed_results
    }
    
    with open(output_dir / f'batch_predictions_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=4, ensure_ascii=False)
    
    # Detaylı metin raporu
    with open(output_dir / f'batch_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("FIRE/SMOKE CLASSIFIER TOPLU TAHMİN RAPORU\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Tarih: {timestamp}\n")
        f.write(f"Toplam Dosya: {len(results)}\n")
        f.write(f"Başarılı Tahmin: {len(successful_results)}\n")
        f.write(f"Başarısız Tahmin: {len(failed_results)}\n\n")
        
        f.write("SINIF DAĞILIMI\n")
        f.write("-" * 15 + "\n")
        for class_name, count in class_counts.items():
            percentage = 100 * count / len(successful_results) if successful_results else 0
            f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nDETAYLI TAHMİNLER\n")
        f.write("-" * 20 + "\n")
        for result in successful_results:
            f.write(f"{result['file_path']}: {result['predicted_class']} "
                   f"(Güven: {result['confidence']:.3f})\n")
        
        if failed_results:
            f.write("\nHATALAR\n")
            f.write("-" * 8 + "\n")
            for result in failed_results:
                f.write(f"{result['file_path']}: {result['error']}\n")
    
    print(f"Toplu tahmin sonuçları kaydedildi:")
    print(f"  - JSON: batch_predictions_{timestamp}.json")
    print(f"  - Rapor: batch_report_{timestamp}.txt")

def main():
    parser = argparse.ArgumentParser(description='Multi-label Fire/Smoke Prediction')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_path', type=str, help='Path to input .npy file')
    parser.add_argument('--image_path', type=str, help='Path to RGB image file')
    parser.add_argument('--mhi_path', type=str, help='Path to MHI file')
    parser.add_argument('--flow_path', type=str, help='Path to Flow file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224], help='Target image size (H W)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = MultiLabelPredictor(
        args.model_path, 
        threshold=args.threshold,
        target_size=tuple(args.target_size)
    )
    
    # Load data based on input type
    if args.input_path:
        # Load from combined .npy file
        print(f"Loading data from: {args.input_path}")
        data = torch.from_numpy(np.load(args.input_path)).float()
        data = predictor.preprocess_data(data)
    elif args.image_path and args.mhi_path and args.flow_path:
        # Load from separate files
        print(f"Loading separate files:")
        print(f"  RGB: {args.image_path}")
        print(f"  MHI: {args.mhi_path}")
        print(f"  Flow: {args.flow_path}")
        data = predictor.load_separate_files(args.image_path, args.mhi_path, args.flow_path)
    else:
        raise ValueError("Either --input_path or all of --image_path, --mhi_path, --flow_path must be provided")
    
    # Run prediction with visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"prediction_{timestamp}.png"
    predictions, probabilities = predictor.predict_and_visualize(data, save_path=save_path)
    
    # Save results
    results = {
        'input_path': args.input_path or f"{args.image_path},{args.mhi_path},{args.flow_path}",
        'probabilities': probabilities.tolist(),
        'predictions': predictions.tolist(),
        'threshold': args.threshold,
        'class_names': predictor.class_names,
        'target_size': args.target_size
    }
    
    with open(output_dir / f"prediction_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main() 