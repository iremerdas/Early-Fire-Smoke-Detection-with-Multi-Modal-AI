import torch
import numpy as np
import matplotlib.pyplot as plt
from fire_smoke_classifier import create_model, count_parameters
from train_classifier import NormalizeTransform
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
from datetime import datetime
import json

from multi_label_dataset import MultiLabelDataset
from multi_label_classifier import MultiLabelFireSmokeClassifier

def test_model_creation():
    """Model oluşturma testi"""
    print("=== Model Oluşturma Testi ===")
    
    # Model oluştur
    model = create_model(num_classes=3, pretrained=False)
    
    # Parametre sayısını yazdır
    param_count = count_parameters(model)
    print(f"Model parametre sayısı: {param_count:,}")
    
    # Model yapısını yazdır
    print("\nModel yapısı:")
    print(model)
    
    return model

def test_forward_pass(model):
    """Forward pass testi"""
    print("\n=== Forward Pass Testi ===")
    
    # Test girdisi oluştur (batch_size=2, channels=5, height=224, width=224)
    batch_size = 2
    test_input = torch.randn(batch_size, 5, 224, 224)
    
    print(f"Girdi şekli: {test_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Çıktı şekli: {output.shape}")
    print(f"Çıktı değerleri:\n{output}")
    
    # Softmax uygula
    probabilities = F.softmax(output, dim=1)
    print(f"Olasılıklar:\n{probabilities}")
    
    # Sınıf tahminleri
    predicted_classes = torch.argmax(probabilities, dim=1)
    print(f"Tahmin edilen sınıflar: {predicted_classes}")
    
    return output, probabilities

def test_with_normalization():
    """Normalizasyon ile test"""
    print("\n=== Normalizasyon Testi ===")
    
    model = create_model(num_classes=3, pretrained=False)
    transform = NormalizeTransform()
    
    # Test girdisi
    test_input = torch.randn(1, 5, 224, 224)
    print(f"Orijinal girdi aralığı: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # Normalizasyon uygula
    normalized_input = transform(test_input)
    print(f"Normalize edilmiş girdi aralığı: [{normalized_input.min():.3f}, {normalized_input.max():.3f}]")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(normalized_input)
    
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
    # Güncellenmiş sınıf isimleri: 0:smoke, 1:fire, 2:none
    class_names = ['smoke', 'fire', 'none']
    print(f"Tahmin edilen sınıf: {class_names[predicted_class.item()]}")
    print(f"Olasılıklar: {probabilities[0]}")

def test_attention_modules():
    """CBAM modüllerini test et"""
    print("\n=== CBAM Modül Testi ===")
    
    from fire_smoke_classifier import CBAM
    
    # Test CBAM modülü
    cbam = CBAM(in_channels=64, reduction_ratio=16)
    test_input = torch.randn(1, 64, 32, 32)
    
    print(f"CBAM girdi şekli: {test_input.shape}")
    
    output = cbam(test_input)
    print(f"CBAM çıktı şekli: {output.shape}")
    print(f"CBAM çıktı aralığı: [{output.min():.3f}, {output.max():.3f}]")

def test_five_channel_conv():
    """5-kanal konvolüsyon testi"""
    print("\n=== 5-Kanal Konvolüsyon Testi ===")
    
    from fire_smoke_classifier import FiveChannelConv
    
    # Test 5-kanal konvolüsyon
    conv = FiveChannelConv(in_channels=5, out_channels=3)
    test_input = torch.randn(1, 5, 224, 224)
    
    print(f"5-kanal girdi şekli: {test_input.shape}")
    
    output = conv(test_input)
    print(f"3-kanal çıktı şekli: {output.shape}")
    print(f"Çıktı aralığı: [{output.min():.3f}, {output.max():.3f}]")

def create_sample_data():
    """Örnek 5-kanal veri oluştur"""
    print("\n=== Örnek Veri Oluşturma ===")
    
    # RGB kanalları (0-1 aralığında)
    rgb = np.random.rand(224, 224, 3).astype(np.float32)
    
    # MHI kanalı (0-1 aralığında)
    mhi = np.random.rand(224, 224).astype(np.float32)
    
    # Optical Flow kanalı (0-1 aralığında)
    optical_flow = np.random.rand(224, 224).astype(np.float32)
    
    # 5-kanal veri oluştur: [R, G, B, MHI, OpticalFlow]
    data = np.concatenate([rgb, mhi[..., np.newaxis], optical_flow[..., np.newaxis]], axis=2)
    
    print(f"5-kanal veri şekli: {data.shape}")
    print(f"Veri aralıkları:")
    print(f"  RGB: [{data[:,:,:3].min():.3f}, {data[:,:,:3].max():.3f}]")
    print(f"  MHI: [{data[:,:,3].min():.3f}, {data[:,:,3].max():.3f}]")
    print(f"  Optical Flow: [{data[:,:,4].min():.3f}, {data[:,:,4].max():.3f}]")
    
    return data

def test_with_realistic_data():
    """Gerçekçi veri ile test"""
    print("\n=== Gerçekçi Veri Testi ===")
    
    model = create_model(num_classes=3, pretrained=False)
    transform = NormalizeTransform()
    
    # Gerçekçi 5-kanal veri oluştur
    data = create_sample_data()
    
    # (H, W, 5) -> (5, H, W) formatına çevir
    data = np.transpose(data, (2, 0, 1))
    
    # Tensor'a çevir ve batch dimension ekle
    data = torch.FloatTensor(data).unsqueeze(0)
    
    print(f"Model girdi şekli: {data.shape}")
    
    # Normalizasyon uygula
    normalized_data = transform(data)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(normalized_data)
    
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
    # Güncellenmiş sınıf isimleri: 0:smoke, 1:fire, 2:none
    class_names = ['smoke', 'fire', 'none']
    print(f"Tahmin edilen sınıf: {class_names[predicted_class.item()]}")
    print(f"Olasılıklar:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
        print(f"  {class_name}: {prob:.3f}")

def visualize_model_components():
    """Model bileşenlerini görselleştir"""
    print("\n=== Model Bileşenleri Görselleştirme ===")
    
    # Örnek veri oluştur
    data = create_sample_data()
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # RGB kanalları
    axes[0].imshow(data[:, :, :3])
    axes[0].set_title('RGB Channels', fontweight='bold')
    axes[0].axis('off')
    
    # MHI kanalı
    axes[1].imshow(data[:, :, 3], cmap='gray')
    axes[1].set_title('MHI Channel', fontweight='bold')
    axes[1].axis('off')
    
    # Optical Flow kanalı
    axes[2].imshow(data[:, :, 4], cmap='hot')
    axes[2].set_title('Optical Flow Channel', fontweight='bold')
    axes[2].axis('off')
    
    # 5-kanal veri (ilk 3 kanal)
    axes[3].imshow(data[:, :, :3])
    axes[3].set_title('Combined (RGB)', fontweight='bold')
    axes[3].axis('off')
    
    # 5-kanal veri (son 2 kanal)
    combined_motion = np.stack([data[:, :, 3], data[:, :, 4], np.zeros_like(data[:, :, 3])], axis=2)
    axes[4].imshow(combined_motion)
    axes[4].set_title('Combined (MHI + Flow)', fontweight='bold')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_components_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Görselleştirme kaydedildi: model_components_visualization.png")

def test_class_mapping():
    """Sınıf eşleştirme testi"""
    print("\n=== Sınıf Eşleştirme Testi ===")
    
    # Güncellenmiş sınıf isimleri: 0:smoke, 1:fire, 2:none
    class_names = ['smoke', 'fire', 'none']
    
    print("Sınıf eşleştirmesi:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")
    
    # Test tahminleri
    test_predictions = [0, 1, 2, 0, 1, 2]
    print(f"\nTest tahminleri: {test_predictions}")
    print("Tahmin edilen sınıflar:")
    for pred in test_predictions:
        print(f"  {pred} -> {class_names[pred]}")

class ModelTester:
    def __init__(self, model_path, threshold=0.5, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.threshold = threshold
        self.class_names = ['smoke', 'fire', 'none']
        
        # Load model
        self.model = MultiLabelFireSmokeClassifier(num_classes=3, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def test(self, test_loader):
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        for data, targets in test_loader:
            # Move to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            probabilities = outputs.cpu().numpy()
            predictions = (probabilities >= self.threshold).astype(int)
            
            # Store results
            all_predictions.append(predictions)
            all_targets.append(targets.cpu().numpy())
            all_probabilities.append(probabilities)
        
        # Concatenate results
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_probabilities = np.vstack(all_probabilities)
        
        return all_predictions, all_targets, all_probabilities
    
    def calculate_metrics(self, predictions, targets):
        """Calculate detailed metrics for each class"""
        # Per-class metrics
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = classification_report(
            targets, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Per-class precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions,
            average=None,
            labels=range(len(self.class_names))
        )
        
        # Store per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        return metrics
    
    def visualize_results(self, predictions, targets, probabilities, output_dir):
        """Create and save various visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix per class
        plt.figure(figsize=(15, 5))
        for i, class_name in enumerate(self.class_names):
            plt.subplot(1, 3, i+1)
            cm = pd.crosstab(
                pd.Series(targets[:, i], name='Actual'),
                pd.Series(predictions[:, i], name='Predicted')
            )
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{class_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices.png')
        plt.close()
        
        # 2. ROC curves
        from sklearn.metrics import roc_curve, auc
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(output_dir / 'roc_curves.png')
        plt.close()
        
        # 3. Precision-Recall curves
        from sklearn.metrics import precision_recall_curve, average_precision_score
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(targets[:, i], probabilities[:, i])
            ap = average_precision_score(targets[:, i], probabilities[:, i])
            plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.savefig(output_dir / 'precision_recall_curves.png')
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Multi-label Fire/Smoke Classifier')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir / f"test_results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test dataset and loader
    test_dataset = MultiLabelDataset(args.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize tester and run tests
    tester = ModelTester(args.model_path, threshold=args.threshold)
    predictions, targets, probabilities = tester.test(test_loader)
    
    # Calculate metrics
    metrics = tester.calculate_metrics(predictions, targets)
    
    # Save metrics
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create visualizations
    tester.visualize_results(predictions, targets, probabilities, results_dir)
    
    # Print summary
    print("\nTest Results Summary:")
    print("-" * 80)
    print("Per-class metrics:")
    for class_name in tester.class_names:
        metrics_dict = metrics[class_name]
        print(f"\n{class_name}:")
        print(f"Precision: {metrics_dict['precision']:.3f}")
        print(f"Recall: {metrics_dict['recall']:.3f}")
        print(f"F1-score: {metrics_dict['f1']:.3f}")
        print(f"Support: {metrics_dict['support']}")
    
    print("\nDetailed classification report saved to:", results_dir / 'metrics.json')
    print("Visualizations saved to:", results_dir)

if __name__ == '__main__':
    main() 