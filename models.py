import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Sequence
import math
import numpy as np
from monai.networks.nets import SwinUNETR as SegResNet
import logging

logger = logging.getLogger(__name__)

class ResidualBlock3D(nn.Module):
    """3D Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SegResNet(nn.Module):
    """간단하고 안정적인 SegResNet 구현"""
    def __init__(self, 
                 in_channels=1, 
                 out_channels=2, 
                 init_filters=32, 
                 dropout_prob=0.2):
        super().__init__()
        
        # Encoder
        self.conv1 = nn.Conv3d(in_channels, init_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(init_filters)
        
        self.conv2 = nn.Conv3d(init_filters, init_filters*2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(init_filters*2)
        
        self.conv3 = nn.Conv3d(init_filters*2, init_filters*4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(init_filters*4)
        
        self.conv4 = nn.Conv3d(init_filters*4, init_filters*8, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(init_filters*8)
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(init_filters*8, init_filters*4, 2, stride=2)
        self.conv_up1 = nn.Conv3d(init_filters*8, init_filters*4, 3, padding=1)
        self.bn_up1 = nn.BatchNorm3d(init_filters*4)
        
        self.up2 = nn.ConvTranspose3d(init_filters*4, init_filters*2, 2, stride=2)
        self.conv_up2 = nn.Conv3d(init_filters*4, init_filters*2, 3, padding=1)
        self.bn_up2 = nn.BatchNorm3d(init_filters*2)
        
        self.up3 = nn.ConvTranspose3d(init_filters*2, init_filters, 2, stride=2)
        self.conv_up3 = nn.Conv3d(init_filters*2, init_filters, 3, padding=1)
        self.bn_up3 = nn.BatchNorm3d(init_filters)
        
        # Final layer
        self.conv_final = nn.Conv3d(init_filters, out_channels, 1)
        self.dropout = nn.Dropout3d(dropout_prob)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))      # 32
        x2 = F.relu(self.bn2(self.conv2(x1)))     # 64, /2
        x3 = F.relu(self.bn3(self.conv3(x2)))     # 128, /4
        x4 = F.relu(self.bn4(self.conv4(x3)))     # 256, /8
        
        # Decoder with skip connections
        up1 = self.up1(x4)                        # 128
        # 크기 맞추기
        if up1.shape[2:] != x3.shape[2:]:
            up1 = F.interpolate(up1, size=x3.shape[2:], mode='trilinear', align_corners=True)
        up1 = torch.cat([up1, x3], dim=1)         # 256
        up1 = F.relu(self.bn_up1(self.conv_up1(up1)))  # 128
        
        up2 = self.up2(up1)                       # 64
        # 크기 맞추기
        if up2.shape[2:] != x2.shape[2:]:
            up2 = F.interpolate(up2, size=x2.shape[2:], mode='trilinear', align_corners=True)
        up2 = torch.cat([up2, x2], dim=1)         # 128
        up2 = F.relu(self.bn_up2(self.conv_up2(up2)))  # 64
        
        up3 = self.up3(up2)                       # 32
        # 크기 맞추기
        if up3.shape[2:] != x1.shape[2:]:
            up3 = F.interpolate(up3, size=x1.shape[2:], mode='trilinear', align_corners=True)
        up3 = torch.cat([up3, x1], dim=1)         # 64
        up3 = F.relu(self.bn_up3(self.conv_up3(up3)))  # 32
        
        # Final
        x = self.dropout(up3)
        x = self.conv_final(x)
        return x


def get_model(model_name: str, **kwargs) -> nn.Module:
    """모델 팩토리 함수 - SegResNet만 지원"""
    models = {
        'segresnet': SegResNet,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    # SegResNet 파라미터 처리
    if model_name.lower() == 'segresnet':
        # SegResNet: in_channels, out_channels 사용
        filtered_kwargs = {}
        if 'in_channels' in kwargs:
            filtered_kwargs['in_channels'] = kwargs['in_channels']
        if 'out_channels' in kwargs:
            filtered_kwargs['out_channels'] = kwargs['out_channels']
        # SegResNet이 지원하는 다른 파라미터들
        for param in ['init_filters', 'blocks_down', 'blocks_up', 'dropout_prob']:
            if param in kwargs:
                filtered_kwargs[param] = kwargs[param]
        kwargs = filtered_kwargs
    
    return models[model_name.lower()](**kwargs)


if __name__ == "__main__":
    # SegResNet 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 테스트 입력
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    
    # SegResNet 테스트
    print("Testing SegResNet model...")
    model = SegResNet(in_channels=1, out_channels=2)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(x)
        print(f"SegResNet: Input {x.shape} -> Output {output.shape}")
        
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SegResNet: Total params: {total_params:,}, Trainable: {trainable_params:,}")
    print("-" * 50)
    
    # get_model 함수 테스트
    print("Testing get_model function...")
    model2 = get_model('segresnet', in_channels=1, out_channels=2)
    model2 = model2.to(device)
    model2.eval()
    
    with torch.no_grad():
        output2 = model2(x)
        print(f"get_model('segresnet'): Input {x.shape} -> Output {output2.shape}")
    
    print("✅ All tests passed!")