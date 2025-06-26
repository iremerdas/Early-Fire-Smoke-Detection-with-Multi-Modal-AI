import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# 3D CNN için CBAM (isteğe bağlı)
class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        # x: (B, C, T, H, W) -> (B*T, C, H, W)
        B, C, T, H, W = x.shape
        x_2d = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        out = x_2d * self.ca(x_2d)
        out = out * self.sa(out)
        out = out.reshape(B, T, C, H, W).permute(0,2,1,3,4)
        return out

class CBAM2D(CBAM):
    """2D Convolutional Block Attention Module (for (B, C, H, W) tensors)"""
    pass

if __name__ == "__main__":
    # 2D örnek
    x = torch.randn(8, 32, 64, 64)
    cbam = CBAM(32)
    y = cbam(x)
    print('CBAM 2D çıktı:', y.shape)
    # 3D örnek
    x3d = torch.randn(4, 32, 5, 32, 32)
    cbam3d = CBAM3D(32)
    y3d = cbam3d(x3d)
    print('CBAM 3D çıktı:', y3d.shape) 