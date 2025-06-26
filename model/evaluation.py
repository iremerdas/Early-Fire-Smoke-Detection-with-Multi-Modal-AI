import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from multi_label_dataset import MultiLabelDataset
from multi_label_classifier import MultiLabelClassifier
from training_utils import TrainingLogger

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run model inference on the test set and collect results."""
    model.eval()
    all_targets = []
    all_probs = []

    for data, targets in tqdm(dataloader, desc="Evaluating"):
        data = data.to(device)
        targets = targets.to(device)
        
        if data.dim() == 5:
            data = data.squeeze(1)

        # Get model outputs (logits)
        outputs = model(data)
        
        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(outputs)

        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    return all_targets, all_probs

def main():
    parser = argparse.ArgumentParser(description='Evaluate a multi-label fire/smoke classifier.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('--test_npy_dir', type=str, required=True, help='Path to the test .npy files directory.')
    parser.add_argument('--test_labels_dir', type=str, required=True, help='Path to the test label files directory.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers.')
    parser.add_argument('--threshold', type=float, nargs='+', default=[0.5], help='Prediction threshold(s). Can be one value for all classes or one value per class.')
    
    args = parser.parse_args()

    # If a single threshold is provided, use it for all classes.
    # Otherwise, expect a threshold for each class.
    thresholds = args.threshold
    if len(thresholds) == 1:
        thresholds = thresholds[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {}) # Get config from checkpoint if it exists

    # Determine image size from config or default
    image_size = config.get('data', {}).get('target_size', (224, 224))[0]

    # Initialize model
    model_name = config.get('model', {}).get('model_name', 'efficientnet_b4_cbam')
    model = MultiLabelClassifier(
        num_classes=3,
        pretrained=False, # Pretrained weights are already loaded
        image_size=image_size,
        model_name=model_name
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("Creating test dataset...")
    test_dataset = MultiLabelDataset(
        npy_dir=args.test_npy_dir,
        labels_dir=args.test_labels_dir,
        target_size=(image_size, image_size)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Run evaluation
    targets, probabilities = evaluate(model, test_loader, device)
    
    # Get binary predictions based on threshold
    if isinstance(thresholds, list):
        predictions = (probabilities >= np.array(thresholds)).astype(int)
    else:
        predictions = (probabilities >= thresholds).astype(int)

    # Initialize logger and save final metrics and plots
    logger = TrainingLogger(output_dir=str(output_path))
    final_metrics = logger.log_final_metrics(predictions, targets, probabilities)
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {output_path.resolve()}")
    print("\nFinal Metrics:")
    print(json.dumps(final_metrics, indent=4))

if __name__ == '__main__':
    main() 