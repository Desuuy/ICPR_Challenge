"""ResTranOCR: ResNet + Transformer architecture for multi-frame OCR."""
import torch
import torch.nn as nn

from src.models.components import (
    AttentionFusion,
    ResNetFeatureExtractor,
    STNBlock,
    TransformerSequenceModeler
)


class ResTranOCR(nn.Module):
    """ResNet-Transformer OCR model with multi-frame attention fusion and input STN.
    
    Architecture: 
    Input -> Shared STN -> ResNet Backbone -> Attention Fusion -> Transformer -> FC -> CTC
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
        
        # STN: Spatial Transformer Network for INPUT ALIGNMENT
        # Operates on RGB images (3 channels)
        self.stn = STNBlock(in_channels=3)
        
        # Backbone: ResNet feature extractor
        self.backbone = ResNetFeatureExtractor(layers=resnet_layers)
        
        # Fusion: Multi-frame attention
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # ResNetFeatureExtractor now uses AdaptiveAvgPool to force H=1
        self.feature_height = 1 
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
            x: Input tensor [B, T, 3, H, W] where T=5 frames.
        
        Returns:
            Log-softmax output [B, W', num_classes] for CTC loss.
        """
        b, t, c, h, w = x.size()
        
        # --- Shared STN Logic ---
        # 1. Compute temporal mean of frames to find the "average" pose
        # Shape: [B, 3, H, W]
        x_mean = torch.mean(x, dim=1)
        
        # 2. Predict affine transformation parameters based on mean image
        xs = self.stn.localization(x_mean)
        theta = self.stn.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # 3. Apply the SAME transformation to ALL frames
        # Reshape x to [B*T, 3, H, W]
        x_flat = x.view(b * t, c, h, w)
        
        # Repeat theta for each frame: [B, 2, 3] -> [B, T, 2, 3] -> [B*T, 2, 3]
        theta_repeated = theta.unsqueeze(1).repeat(1, t, 1, 1).view(b * t, 2, 3)
        
        # Generate grid and warp
        grid = torch.nn.functional.affine_grid(theta_repeated, x_flat.size(), align_corners=False)
        aligned_images = torch.nn.functional.grid_sample(x_flat, grid, align_corners=False)
        
        # --- Backbone ---
        feat = self.backbone(aligned_images)  # [B*T, 512, 1, W']
        
        # --- Fusion ---
        fused = self.fusion(feat)  # [B, 512, 1, W']
        
        # --- Transformer ---
        # Reshape: [B, C, 1, W'] -> [B, W', C]
        b_out, c_out, h_f, w_f = fused.size()
        seq_input = fused.squeeze(2).permute(0, 2, 1) # [B, W', C]
        
        seq_out = self.neck(seq_input)  # [B, W', d_model]
        
        # --- Head ---
        out = self.fc(seq_out)  # [B, W', num_classes]
        
        return out.log_softmax(2)