import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from sklearn.model_selection import train_test_split

from multi_label_dataset import MultiLabelDataset, MultiLabelDatasetWithImages
from multi_label_classifier import MultiLabelClassifier, get_metrics, FocalLoss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = MultiLabelClassifier(
            num_classes=3,
            model_name=config['model']['model_name'],
            pretrained=config['model']['pretrained'],
            image_size=config['data']['target_size'][0]
        ).to(self.device)
        
        # Loss function and optimizer
        loss_type = config['training']['loss']
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=[0.33,0.33,0.34], gamma=2.5)
        elif loss_type == 'smooth':
            self.criterion = nn.BCEWithLogitsLoss()  # Placeholder, will add smoothing below
        else:
            raise ValueError(f'Unknown loss: {loss_type}')
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler_type = config['training']['scheduler']
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        else:
            raise ValueError(f'Unknown scheduler: {scheduler_type}')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        with tqdm(train_loader, desc='Training') as pbar:
            for data, targets in pbar:
                # Move to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Squeeze the singleton dimension: (B, 1, C, H, W) -> (B, C, H, W)
                if data.dim() == 5:
                    data = data.squeeze(1)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                all_outputs.append(outputs.detach())
                all_targets.append(targets)
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        metrics = get_metrics(all_outputs, all_targets)
        
        return total_loss / len(train_loader), metrics
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        for data, targets in val_loader:
            # Move to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Squeeze the singleton dimension: (B, 1, C, H, W) -> (B, C, H, W)
            if data.dim() == 5:
                data = data.squeeze(1)

            # Forward pass
            outputs = self.model(data)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Update statistics
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(targets)
        
        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        metrics = get_metrics(all_outputs, all_targets)
        
        return total_loss / len(val_loader), metrics
    
    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model and history
        self.save_checkpoint('final_model.pth')
        self.save_history()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        # Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'loss_curves.png')
        plt.close()
        
        # Accuracy curves
        plt.figure(figsize=(10, 5))
        train_acc = [m['accuracy'] for m in self.history['train_metrics']]
        val_acc = [m['accuracy'] for m in self.history['val_metrics']]
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.output_dir / 'accuracy_curves.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train multi-label fire/smoke classifier')
    parser.add_argument('--train_npy_dir', type=str, required=True, help='Path to training .npy files directory')
    parser.add_argument('--train_labels_dir', type=str, required=True, help='Path to training label files directory')
    parser.add_argument('--val_npy_dir', type=str, required=True, help='Path to validation .npy files directory')
    parser.add_argument('--val_labels_dir', type=str, required=True, help='Path to validation label files directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Checkpoint save interval')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224], help='Target image size (H W)')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4_cbam', choices=['efficientnet_b4_cbam','resnet50_se','swintransformer_cbam'], help='Model backbone type')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step','cosine'], help='Scheduler type')
    parser.add_argument('--loss', type=str, default='focal', choices=['bce','focal','smooth'], help='Loss function')
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble training/testing')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data': {
            'train_npy_dir': args.train_npy_dir,
            'train_labels_dir': args.train_labels_dir,
            'val_npy_dir': args.val_npy_dir,
            'val_labels_dir': args.val_labels_dir,
            'target_size': tuple(args.target_size)
        },
        'model': {
            'pretrained': True,
            'model_name': args.model_name
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'patience': args.patience,
            'checkpoint_interval': args.checkpoint_interval,
            'weight_decay': 1e-4,
            'scheduler': args.scheduler,
            'loss': args.loss
        },
        'output_dir': args.output_dir or f'runs/multi_label_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print(f"Configuration:")
    print(f"  Train NPY directory: {config['data']['train_npy_dir']}")
    print(f"  Train Labels directory: {config['data']['train_labels_dir']}")
    print(f"  Val NPY directory: {config['data']['val_npy_dir']}")
    print(f"  Val Labels directory: {config['data']['val_labels_dir']}")
    print(f"  Target size: {config['data']['target_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training'].get('weight_decay', 'None')}")
    print(f"  Output directory: {config['output_dir']}")
    
    # Create datasets
    train_dataset = MultiLabelDataset(
        npy_dir=config['data']['train_npy_dir'],
        labels_dir=config['data']['train_labels_dir'],
        target_size=config['data']['target_size']
    )
    
    val_dataset = MultiLabelDataset(
        npy_dir=config['data']['val_npy_dir'],
        labels_dir=config['data']['val_labels_dir'],
        target_size=config['data']['target_size']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    if len(val_dataset) == 0:
        print("Validation dataset is empty. Please check the validation data paths and content.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main() 