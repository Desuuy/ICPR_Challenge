


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Thêm đường dẫn để đảm bảo import được openrec (from openrec.xxx)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)



from src.models.components import STNBlock, AttentionFusion
from .openrec.modeling.encoders.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
from .openrec.modeling.decoders.rctc_decoder import RCTCDecoder
class MultiFrameSVTRv2(nn.Module):
    def __init__(self, num_classes, use_stn=True, dropout=0.0):
        super().__init__()
        self.use_stn = use_stn
        # 1. STN để nắn thẳng biển số trước khi vào backbone
        self.stn = STNBlock(in_channels=3, dropout=dropout)

        # 2. SVTRv2 Encoder (SVTRv2LNConvTwo33)
        # Cấu hình này được lấy TRỰC TIẾP từ weights/config.yml:
        #   dims:      [128, 256, 384]
        #   depths:    [6, 6, 6]
        #   num_heads: [4, 8, 12]
        #   mixer:     giống hệt trong config.yml
        #   local_k, sub_k, feat2d: khớp config.yml
        self.backbone = SVTRv2LNConvTwo33(
            max_sz=[32, 128],
            dims=[128, 256, 384],
            depths=[6, 6, 6],
            num_heads=[4, 8, 12],
            mixer=[
                ['Conv'] * 6,
                ['Conv', 'Conv', 'FGlobal', 'Global', 'Global', 'Global'],
                ['Global'] * 6
            ],
            local_k=[[5, 5], [5, 5], [-1, -1]],
            sub_k=[[1, 1], [2, 1], [-1, -1]],
            feat2d=True,
        )

        # 3. Fusion nhận 384 channels từ stage cuối của Encoder
        # (dims[-1] = 384 trong config.yml)
        self.fusion = AttentionFusion(channels=384, dropout=dropout)

        # 4. Head (RCTCDecoder) cũng nhận 384 channels
        self.head = RCTCDecoder(in_channels=384, out_channels=num_classes)

    def load_unirec_weights(self, weight_path: str):
        """Nạp trọng số UniRec/checkpoint vào model (lọc theo tên + shape, hỗ trợ remap key)."""
        if not os.path.exists(weight_path):
            print(f"⚠️ Không tìm thấy file tại: {weight_path}")
            return

        try:
            checkpoint = torch.load(
                weight_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_dict = self.state_dict()

        # Thử load trực tiếp trước
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }

        # Nếu 0 layer khớp, thử remap key (UniRec/openrec dùng "encoder.", DDP dùng "module.")
        if len(filtered_dict) == 0 and len(state_dict) > 0:
            def remap_key(key: str) -> str:
                key = key.replace("module.", "")
                key = key.replace("encoder.", "backbone.")
                return key
            state_dict_remap = {remap_key(k): v for k, v in state_dict.items()}
            filtered_dict = {
                k: v for k, v in state_dict_remap.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
        if len(filtered_dict) == 0 and len(state_dict) > 0:
            # In vài key mẫu để debug
            ck_keys = list(state_dict.keys())[:5]
            md_keys = list(model_dict.keys())[:5]
            print(f"   (Checkpoint keys mẫu: {ck_keys})")
            print(f"   (Model keys mẫu: {md_keys})")

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        print(
            f"✅ Đã nạp thành công {len(filtered_dict)} layers từ checkpoint.")

    # Load weights
    def load_weights(self, weight_path: str):
        """Wrapper tiện dụng để nạp trọng số từ đường dẫn."""
        self.load_unirec_weights(weight_path)

    # Save weights
    def save_weights(self, weight_path):
        torch.save(self.state_dict(), weight_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass với SVTRv2 + STN.

        Args:
            x: [Batch, Frames(5), 3, H, W]
        Returns:
            logits: [Batch, Seq_Len, Num_Classes] (log_softmax)
        """
        b, f, c, h, w = x.size()
        # Gom Batch và Frames lại để xử lý song song
        x_flat = x.view(b * f, c, h, w)

        # --- Giai đoạn 1: STN Alignment ---
        if self.use_stn:
            theta = self.stn(x_flat)  # STN: Spatial Transformer Network
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        # --- Giai đoạn 2: SVTRv2 Feature Extraction ---
        # SVTRv2LNConvTwo33: Encoder backbone
        # Output: [B*F, C, H', W'] với C=384 (dims[-1])
        features = self.backbone(x_aligned)

        # --- Giai đoạn 3: Multi-frame Fusion ---
        # AttentionFusion: Gộp thông tin từ 5 frames
        # Output: [B, C, H', W']
        fused = self.fusion(features)

        # --- Giai đoạn 4: Sequence Preparation & Head ---
        # Ép Height về 1 để tạo chuỗi ký tự theo chiều ngang
        fused_seq = F.adaptive_avg_pool2d(fused, (1, fused.size(-1)))

        # RCTCDecoder: CTC Head dự đoán xác suất ký tự
        # Output: [B, Seq_Len, Num_Classes]
        logits = self.head(fused_seq)

        return logits.log_softmax(2)

    def verify_architecture(self) -> dict:
        """Kiểm tra và trả về thông tin về architecture của model."""
        info = {
            'has_stn': hasattr(self, 'stn') and self.use_stn,
            'has_backbone': hasattr(self, 'backbone'),
            'backbone_type': type(self.backbone).__name__ if hasattr(self, 'backbone') else None,
            'has_fusion': hasattr(self, 'fusion'),
            'fusion_type': type(self.fusion).__name__ if hasattr(self, 'fusion') else None,
            'has_head': hasattr(self, 'head'),
            'head_type': type(self.head).__name__ if hasattr(self, 'head') else None,
        }
        return info
