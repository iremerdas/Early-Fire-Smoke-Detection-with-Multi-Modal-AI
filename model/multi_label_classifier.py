import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from cbam_module import CBAM2D
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from typing import Union, List
from torchvision import models
try:
    from torchvision.models import swin_t, Swin_T_Weights
    SWIN_DEFAULT_WEIGHTS = getattr(Swin_T_Weights, 'IMAGENET1K_V1', None)
except ImportError:
    swin_t = None
    SWIN_DEFAULT_WEIGHTS = None

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    """
    def __init__(self, alpha: Union[float, List[float]] = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Move alpha to the same device as inputs
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(inputs.device)
            
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name='efficientnet_b4_cbam', pretrained=True, image_size=380):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.image_size = image_size
        if self.model_name == 'efficientnet_b4_cbam':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b4', image_size=image_size) if pretrained else EfficientNet.from_name('efficientnet-b4', image_size=image_size)
            # 5-kanal giriş
            original_conv = self.base_model._conv_stem
            self.base_model._conv_stem = Conv2dStaticSamePadding(
                in_channels=5,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size[0],
                stride=original_conv.stride[0],
                image_size=image_size,
                bias=original_conv.bias is not None
            )
            num_features = self.base_model._conv_head.out_channels
            self.attn = CBAM2D(num_features)
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.model_name == 'resnet50_se':
            self.base_model = models.resnet50(pretrained=pretrained)
            # 5-kanal giriş
            self.base_model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.base_model.fc.in_features
            self.attn = SEModule(num_features)
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.model_name == 'swintransformer_cbam':
            assert swin_t is not None, 'torchvision >= 0.12 gerekli'
            self.base_model = swin_t(weights=SWIN_DEFAULT_WEIGHTS if pretrained else None)
            # SwinTransformer ilk patch_embed katmanını 5-kanal yap
            self.base_model.features[0][0] = nn.Conv2d(5, 96, kernel_size=4, stride=4)
            num_features = self.base_model.head.in_features
            self.attn = CBAM2D(num_features)
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f'Unknown model_name: {model_name}')
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        if 'efficientnet' in self.model_name:
            features = self.base_model.extract_features(x)
            features = self.attn(features)
            x = self.pool(features)
            x = x.flatten(start_dim=1)
        elif 'resnet' in self.model_name:
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            features = self.attn(x)
            x = self.pool(features)
            x = x.flatten(start_dim=1)
        elif 'swintransformer' in self.model_name:
            x = self.base_model.features(x)
            features = self.attn(x)
            x = self.pool(features)
            x = x.flatten(start_dim=1)
        else:
            raise ValueError(f'Unknown model_name: {self.model_name}')
        output = self.classifier(x)
        return output

def get_loss_fn():
    """Returns FocalLoss for multi-label classification to handle class imbalance and overfitting."""
    return FocalLoss(alpha=[0.33, 0.33, 0.34], gamma=2.5)

def get_metrics(outputs, targets, threshold=0.5):
    """Calculate metrics for multi-label classification.
    
    Args:
        outputs (torch.Tensor): Model predictions (logits, before sigmoid)
        targets (torch.Tensor): Ground truth labels
        threshold (float or list/tuple): Decision threshold(s) for positive prediction.
                                         Can be a single float or per-class thresholds.
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert predictions (logits) to binary using threshold(s)
    probabilities = torch.sigmoid(outputs)
    
    if isinstance(threshold, (list, tuple)):
        # Per-class thresholds
        threshold = torch.tensor(threshold, device=probabilities.device).view(1, -1)
        predictions = (probabilities >= threshold).float()
    else:
        # Single threshold for all classes
        predictions = (probabilities >= threshold).float()
    
    # Calculate metrics
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    
    # Per-class metrics
    class_correct = correct.sum(dim=0)
    class_total = torch.ones_like(class_correct) * targets.size(0)
    class_accuracy = (class_correct / class_total).cpu().numpy()
    
    # Calculate exact match ratio (all classes predicted correctly)
    exact_match = (correct.sum(dim=1) == targets.size(1)).float().mean().item()
    
    metrics = {
        'accuracy': accuracy,
        'exact_match': exact_match,
        'class_accuracy': class_accuracy.tolist(),
        'smoke_accuracy': float(class_accuracy[0]),
        'fire_accuracy': float(class_accuracy[1]),
        'none_accuracy': float(class_accuracy[2])
    }
    
    return metrics 