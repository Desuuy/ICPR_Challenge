import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Thêm đường dẫn để đảm bảo import được các module bên trong openrec
current_dir = os.path.dirname(__file__)
openrec_path = os.path.join(current_dir, 'openrec')
if openrec_path not in sys.path:
    sys.path.append(openrec_path)

# Import các thành phần từ pipeline cũ của bạn
from src.models.components import STNBlock, AttentionFusion

# Import SVTRv2 chuẩn từ folder openrec đã tải
# Lưu ý: Tên class có thể là SVTRv2 hoặc SVTRv2Encoder tùy phiên bản bạn tải
from .openrec.modeling.backbones.rec_svtr_v2 import SVTRv2 # Kiểm tra lại đường dẫn chính xác trong folder của bạn
from .openrec.modeling.heads.rec_ctc_head import CTCHead



class MultiFrameSVTRv2(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        embed_dim: int = 192,  # Thông số bản SVTRv2-Small
        use_stn: bool = True
    ):
        super().__init__()
        self.use_stn = use_stn
        self.embed_dim = embed_dim
        
        # 1. Spatial Transformer Network (STN)
        # Giữ nguyên để nắn thẳng biển số trước khi vào SVTRv2
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)
        
        # 2. SVTRv2 Backbone (Encoder)
        # Đây là "linh hồn" thay thế cho ResNet34
        # Các tham số này nên khớp với cấu hình UniRec40M bạn định dùng
        self.backbone = SVTRv2(
            img_size=(32, 128),
            in_channels=3,
            embed_dim=self.embed_dim,
            depth=[3, 6, 3],        # Cấu hình các khối Mixing
            num_heads=[3, 6, 9],
            mixer_types=['Local', 'Global'] # Kết hợp học đặc trưng và ngữ cảnh
        )
        
        # 3. Attention Fusion
        # Gộp đặc trưng từ 5 frames lại thành 1 đại diện tốt nhất
        self.fusion = AttentionFusion(channels=self.embed_dim)
        
        # 4. CTC Prediction Head
        # Thay thế Head cũ bằng CTC Head chuẩn của SVTRv2
        self.head = CTCHead(
            in_channels=self.embed_dim, 
            out_channels=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames(5), 3, H, W]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)  # Gom Batch và Frames lại để xử lý song song
        
        # --- Giai đoạn 1: STN Alignment ---
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        # --- Giai đoạn 2: SVTRv2 Feature Extraction ---
        # Chạy SVTRv2 cho từng frame. Output: [B*F, Embed_Dim, H', W']
        features = self.backbone(x_aligned)
        
        # --- Giai đoạn 3: Multi-frame Fusion ---
        # Gộp thông tin từ 5 frames. Output: [B, Embed_Dim, H', W']
        fused = self.fusion(features)
        
        # --- Giai đoạn 4: Sequence Preparation & Head ---
        # SVTRv2 thường ép Height về 1 để tạo chuỗi ký tự theo chiều ngang
        fused_seq = F.adaptive_avg_pool2d(fused, (1, None)) 
        
        # CTC Head dự đoán xác suất ký tự
        logits = self.head(fused_seq) # [B, Seq_Len, Num_Classes]
        
        return logits.log_softmax(2)