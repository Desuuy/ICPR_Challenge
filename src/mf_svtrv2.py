import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Thêm đường dẫn để đảm bảo import được openrec (from openrec.xxx)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 1. Import các thành phần nội bộ của bạn
from src.models.components import STNBlock, AttentionFusion

# 2. Import SVTRv2 chuẩn từ thư mục modeling/encoders
# Chúng ta lấy 'svtrv2.py' vì đây là kiến trúc gốc mạnh nhất của dự án
# Sửa lại dòng này cho đúng với file trong ảnh image_28ea4e.png
from .openrec.modeling.encoders.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33

# 3. Import CTC Head chuẩn từ thư mục modeling/decoders
# SVTRv2 đi kèm với cơ chế CTC để đạt tốc độ cao
from .openrec.modeling.decoders.rctc_decoder import RCTCDecoder


class MultiFrameSVTRv2(nn.Module):
    def __init__(self, num_classes, use_stn=True):
        super().__init__()
        self.use_stn = use_stn
        # 1. STN và Fusion giữ nguyên, nhưng Fusion channels phải là 384
        self.stn = STNBlock(in_channels=3)
        
        # 2. Cập nhật Encoder 
        self.backbone = SVTRv2LNConvTwo33(
            img_size=(32, 128),
            dims=[128, 256, 384], # Khớp dims
            depths=[6, 6, 6],     # Khớp depths
            num_heads=[4, 8, 12], # Khớp num_heads
            mixer=[
                ['Conv'] * 6,
                ['Conv', 'Conv', 'FGlobal', 'Global', 'Global', 'Global'],
                ['Global'] * 6
            ],
            local_k=[[5, 5], [5, 5], [-1, -1]],
            sub_k=[[1, 1], [2, 1], [-1, -1]],
            feat2d=True
        )
        
        # 3. Fusion nhận 384 channels từ stage cuối của Encoder
        self.fusion = AttentionFusion(channels=384)
        
        # 4. Head nhận 384 channels
        self.head = RCTCDecoder(in_channels=384, out_channels=num_classes)

    def load_unirec_weights(model, weight_path):
        """Hàm nạp trọng số thông minh tránh lỗi lệch tên Layer."""
        if not os.path.exists(weight_path):
            print(f"⚠️ Không tìm thấy file tại: {weight_path}")
            return model

        checkpoint = torch.load(weight_path, map_location='cpu')
        # Nếu file .pth chứa dict 'state_dict' thì lấy ra, nếu không lấy trực tiếp
        state_dict = checkpoint.get('state_dict', checkpoint) 
        
        model_dict = model.state_dict()
        
        # Ánh xạ trọng số: Chỉ lấy những layer khớp tên và kích thước
        # Đặc biệt quan trọng vì bạn có STN và Fusion mà bản gốc không có
        filtered_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and v.size() == model_dict[k].size()
        }
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        print(f"✅ Đã nạp thành công {len(filtered_dict)} layers từ UniRec40M.")
        return model
    # Load weights
    def load_weights(self, weight_path):
        self.load_unirec_weights(self, weight_path)

    # Save weights
    def save_weights(self, weight_path):
        torch.save(self.state_dict(), weight_path)


    # Forward pass
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
        fused_seq = F.adaptive_avg_pool2d(fused, (1, fused.size(-1))) 
        
        # CTC Head dự đoán xác suất ký tự
        logits = self.head(fused_seq) # [B, Seq_Len, Num_Classes]
        
        return logits.log_softmax(2)