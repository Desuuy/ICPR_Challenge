"""ResTranOCR: ResNet + Transformer architecture for multi-frame OCR."""
import torch
import torch.nn as nn

from src.models.components import (
    AttentionFusion,
    ResNetFeatureExtractor,
    TransformerSequenceModeler
)


class ResTranOCR(nn.Module):
    """ResNet-Transformer OCR model with multi-frame attention fusion.
    
    Architecture: ResNet Backbone -> Attention Fusion -> Transformer -> FC -> CTC
    """
    
    def __init__(
        self,
        num_classes: int,
        resnet_layers: int = 18,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            num_classes: Number of output classes (including blank for CTC).
            resnet_layers: ResNet variant (18 or 34).
            transformer_heads: Number of attention heads.
            transformer_layers: Number of transformer encoder layers.
            transformer_ff_dim: Feedforward dimension in transformer.
            dropout: Dropout rate.
        """
        super().__init__()
        self.cnn_channels = 512  # ResNet output channels
        
        # Backbone: ResNet feature extractor
        self.backbone = ResNetFeatureExtractor(layers=resnet_layers)
        
        # Fusion: Multi-frame attention
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # For H=32 input -> ResNet output H'=2, so d_model = 512 * 2 = 1024
        self.feature_height = 2
        self.d_model = self.cnn_channels * self.feature_height
        
        # Neck: Transformer sequence modeler
        self.neck = TransformerSequenceModeler(
            d_model=self.d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout
        )
        
        # Head: Classification layer
        self.fc = nn.Linear(self.d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C, H, W] where T=5 frames.
        
        Returns:
            Log-softmax output [B, W', num_classes] for CTC loss.
        """
        b, t, c, h, w = x.size()
        
        # Process all frames through CNN backbone
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)  # [B*T, 512, H', W']
        
        # Fuse multi-frame features
        fused = self.fusion(feat)  # [B, 512, H', W']
        
        # Reshape for transformer: [B, C, H', W'] -> [B, W', C*H']
        b_out, c_out, h_f, w_f = fused.size()
        # Permute to [B, W', C, H'] then flatten last two dims
        seq_input = fused.permute(0, 3, 1, 2).reshape(b_out, w_f, c_out * h_f)
        
        # Transformer sequence modeling
        seq_out = self.neck(seq_input)  # [B, W', d_model]
        
        # Classification
        out = self.fc(seq_out)  # [B, W', num_classes]
        
        return out.log_softmax(2)
