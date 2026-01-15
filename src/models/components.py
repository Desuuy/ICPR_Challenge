"""Reusable model components for multi-frame OCR."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Attention-based fusion module for combining multi-frame features.
    
    Computes attention weights across frames and performs weighted fusion.
    """
    
    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input feature channels.
        """
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B*num_frames, C, H, W].
        
        Returns:
            Fused tensor of shape [B, C, H, W].
        """
        b_frames, c, h, w = x.size()
        num_frames = 5
        b_size = b_frames // num_frames
        
        # Reshape for attention computation
        x_view = x.view(b_size, num_frames, c, h, w)
        scores = self.score_net(x).view(b_size, num_frames, 1, h, w)
        
        weights = F.softmax(scores, dim=1)
        return torch.sum(x_view * weights, dim=1)  # [B, C, H, W]


class CNNBackbone(nn.Module):
    """CNN feature extractor backbone for OCR."""
    
    def __init__(self, out_channels: int = 512):
        """
        Args:
            out_channels: Number of output channels (default 512).
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.cnn(x)


class SequenceModeler(nn.Module):
    """Bidirectional LSTM for sequence modeling."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.25):
        """
        Args:
            input_size: Input feature dimension.
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence through LSTM."""
        output, _ = self.rnn(x)
        return output


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor optimized for OCR.
    
    Uses modified strides to preserve width (sequence length) while reducing height.
    Includes adaptive pooling to ensure final height is 1.
    """
    
    def __init__(self, layers: int = 18):
        """
        Args:
            layers: ResNet variant (18 or 34).
        """
        super().__init__()
        if layers == 18:
            num_blocks = [2, 2, 2, 2]
        elif layers == 34:
            num_blocks = [3, 4, 6, 3]
        else:
            raise ValueError(f"Unsupported ResNet layers: {layers}. Use 18 or 34.")
        
        self.in_channels = 64
        
        # Initial conv: stride (2, 2) reduces both H and W
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=(2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers with asymmetric strides to preserve width
        self.layer1 = self._make_layer(64, num_blocks[0], stride=(1, 1))
        self.layer2 = self._make_layer(128, num_blocks[1], stride=(2, 1))  # H/2, W same
        self.layer3 = self._make_layer(256, num_blocks[2], stride=(2, 1))  # H/2, W same
        self.layer4 = self._make_layer(512, num_blocks[3], stride=(2, 1))  # H/2, W same
        
        self._init_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: tuple) -> nn.Sequential:
        strides = [stride] + [(1, 1)] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W].
        
        Returns:
            Feature tensor [B, 512, 1, W'].
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Force Height to 1 regardless of input height (Robust for H=48 or H=32)
        x = F.adaptive_avg_pool2d(x, (1, None))
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, seq_len, d_model].
        
        Returns:
            Position-encoded tensor [B, seq_len, d_model].
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSequenceModeler(nn.Module):
    """Transformer encoder for sequence modeling with positional encoding."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of encoder layers.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, seq_len, d_model].
        
        Returns:
            Encoded tensor [B, seq_len, d_model].
        """
        x = self.pos_encoder(x)
        return self.transformer(x)


class STNBlock(nn.Module):
    """Spatial Transformer Network for input image alignment.
    
    Learns an affine transformation to spatially align input images.
    Can handle single images (C=3) or batches.
    """
    
    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: Number of input channels (typically 3 for RGB).
        """
        super().__init__()
        
        # Lightweight localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) # Fixed size for FC input
        )
        
        # Regressor for 2x3 affine transformation matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        
        # Initialize to identity transformation for stable training
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W].
        
        Returns:
            Aligned image [B, C, H, W].
        """
        # Predict affine transformation parameters
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Generate sampling grid and apply transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        aligned = F.grid_sample(x, grid, align_corners=False)
        
        return aligned