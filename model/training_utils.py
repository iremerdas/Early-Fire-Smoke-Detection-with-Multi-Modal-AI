import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import pandas as pd
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0,
            'epoch': 0
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics, learning_rate):
        """Log metrics for one epoch"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rates'].append(learning_rate)
        
        # Update best metrics
        if val_loss < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = val_loss
            self.best_metrics['val_accuracy'] = val_metrics['accuracy']
            self.best_metrics['epoch'] = epoch
        
        # Save current history
        self.save_history()
        
        # Create and save plots
        self.create_training_plots()
    
    def save_history(self):
        """Save training history to JSON"""
        history_file = self.output_dir / 'training_history.json'
        
        # Add best metrics to history
        history_dict = {
            'history': self.history,
            'best_metrics': self.best_metrics
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=4)
    
    def create_training_plots(self):
        """Create and save training visualization plots"""
        # 1. Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'loss_curves.png')
        plt.close()
        
        # 2. Accuracy curves
        plt.figure(figsize=(10, 5))
        train_acc = [m['accuracy'] for m in self.history['train_metrics']]
        val_acc = [m['accuracy'] for m in self.history['val_metrics']]
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'accuracy_curves.png')
        plt.close()
        
        # 3. Per-class metrics (only if metrics structure is correct)
        if self.history['val_metrics'] and 'smoke' in self.history['val_metrics'][0]:
            class_names = ['smoke', 'fire', 'none']
            metrics_to_plot = ['precision', 'recall', 'f1']
            
            for metric in metrics_to_plot:
                plt.figure(figsize=(10, 5))
                for i, class_name in enumerate(class_names):
                    try:
                        values = [m[class_name][metric] for m in self.history['val_metrics']]
                        plt.plot(values, label=f'{class_name}')
                    except (KeyError, TypeError):
                        continue
                plt.title(f'Validation {metric.capitalize()} per Class')
                plt.xlabel('Epoch')
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)
                plt.savefig(self.output_dir / f'{metric}_curves.png')
                plt.close()
        
        # 4. Learning rate curve
        if self.history['learning_rates']:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['learning_rates'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(self.output_dir / 'learning_rate.png')
            plt.close()
    
    def create_confusion_matrices(self, predictions, targets, epoch):
        """Create and save confusion matrices for each class"""
        plt.figure(figsize=(15, 5))
        for i, class_name in enumerate(['smoke', 'fire', 'none']):
            plt.subplot(1, 3, i+1)
            try:
                cm = pd.crosstab(
                    pd.Series(targets[:, i], name='Actual'),
                    pd.Series(predictions[:, i], name='Predicted')
                )
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{class_name} Confusion Matrix')
            except (IndexError, ValueError) as e:
                plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                plt.title(f'{class_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrices_epoch_{epoch}.png')
        plt.close()
    
    def log_final_metrics(self, test_predictions, test_targets, test_probabilities):
        """Log and visualize final test metrics"""
        # For multi-label classification, we need to handle each class separately
        class_names = ['smoke', 'fire', 'none']
        final_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'support': []
        }
        
        # Calculate metrics for each class
        for i in range(test_targets.shape[1]):
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    test_targets[:, i], test_predictions[:, i],
                    average='binary',
                    zero_division='warn'
                )
                final_metrics['precision'].append(precision)
                final_metrics['recall'].append(recall)
                final_metrics['f1'].append(f1)
                final_metrics['support'].append(support)
            except Exception as e:
                print(f"Error calculating metrics for class {i}: {e}")
                final_metrics['precision'].append(0.0)
                final_metrics['recall'].append(0.0)
                final_metrics['f1'].append(0.0)
                final_metrics['support'].append(0)
        
        # Save metrics
        with open(self.output_dir / 'final_test_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        # Create ROC curves
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(test_targets[:, i], test_probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Error creating ROC curve for {class_name}: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'final_roc_curves.png')
        plt.close()
        
        # Create confusion matrices
        self.create_confusion_matrices(test_predictions, test_targets, 'final')
        
        return final_metrics

def plot_attention_maps(attention_maps, output_dir, epoch):
    """Plot attention maps from CBAM module"""
    output_dir = Path(output_dir)
    plt.figure(figsize=(15, 5))
    
    # Channel attention
    plt.subplot(1, 3, 1)
    sns.heatmap(attention_maps['channel_attention'], cmap='viridis')
    plt.title('Channel Attention')
    
    # Spatial attention
    plt.subplot(1, 3, 2)
    sns.heatmap(attention_maps['spatial_attention'], cmap='viridis')
    plt.title('Spatial Attention')
    
    # Combined attention
    plt.subplot(1, 3, 3)
    sns.heatmap(attention_maps['combined_attention'], cmap='viridis')
    plt.title('Combined Attention')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'attention_maps_epoch_{epoch}.png')
    plt.close() 