


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Thêm đường dẫn để đảm bảo import được openrec (from openrec.xxx)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)



from src.models.components import STNBlock, AttentionFusion, TemperatureScaling
from .openrec.modeling.encoders.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
from .openrec.modeling.decoders.rctc_decoder import RCTCDecoder


class MultiFrameSVTRv2(nn.Module):
    def __init__(self, num_classes, use_stn=True, dropout=0.1, use_temp_scaling=True):
        super().__init__()
        self.use_stn = use_stn
        # 1. STN để nắn thẳng biển số trước khi vào backbone
        self.stn = STNBlock(in_channels=3, dropout=dropout)

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

        # Thêm Country Embedding
        self.country_emb = nn.Embedding(2, 64) # 2 quốc gia, vector 64 chiều

        # 3. Fusion nhận 384 channels từ stage cuối của Encoder
        # (dims[-1] = 384 trong config.yml) + tăng dropout để tránh overfit HQ
        self.fusion = AttentionFusion(channels=384, dropout=0.3)

        # 4. Context Projection: nối đặc trưng ảnh (384) + quốc gia (64)
        self.decoder_proj = nn.Sequential(
            nn.Linear(384 + 64, 384),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 5. Head (RCTCDecoder) cũng nhận 384 channels
        self.head = RCTCDecoder(in_channels=384, out_channels=num_classes)

        # Temperature scaling for confidence calibration
        self.use_temp_scaling = use_temp_scaling
        if self.use_temp_scaling:
            self.temp_scaling = TemperatureScaling()


    def load_unirec_weights(self, weight_path: str):
        """Nạp trọng số UniRec/GTC checkpoint vào model.

        Checkpoint từ weights/config.yml (GTCDecoder) có cấu trúc:
        - encoder.* -> backbone.*
        - decoder.ctc_decoder.* -> head.*
        """
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

        def remap_key(key: str) -> str:
            key = key.replace("module.", "")
            key = key.replace("encoder.", "backbone.")
            # GTCDecoder: decoder.ctc_decoder.* -> head.*
            key = key.replace("decoder.ctc_decoder.", "head.")
            return key

        # Remap toàn bộ checkpoint trước
        state_dict_remap = {remap_key(k): v for k, v in state_dict.items()}

        filtered_dict = {}
        for k, v in state_dict_remap.items():
            if k not in model_dict:
                continue
            md_shape = model_dict[k].shape
            if v.shape == md_shape:
                filtered_dict[k] = v
            elif k == "head.fc.weight" and v.dim() == 2:
                # Checkpoint có vocab lớn (6625), model có 37 classes -> slice N class đầu
                if v.shape[1] == md_shape[1] and v.shape[0] >= md_shape[0]:
                    filtered_dict[k] = v[:md_shape[0], :].clone()
                elif v.shape == md_shape:
                    filtered_dict[k] = v
            elif k == "head.fc.bias" and v.dim() == 1:
                if v.shape[0] >= md_shape[0]:
                    filtered_dict[k] = v[:md_shape[0]].clone()
                elif v.shape == md_shape:
                    filtered_dict[k] = v

        if len(filtered_dict) == 0 and len(state_dict) > 0:
            ck_keys = list(state_dict.keys())[:8]
            md_keys = list(model_dict.keys())[:8]
            print(f"   (Checkpoint keys mẫu: {ck_keys})")
            print(f"   (Model keys mẫu: {md_keys})")

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)

        loaded_head = sum(1 for k in filtered_dict if k.startswith("head."))
        loaded_backbone = sum(1 for k in filtered_dict if k.startswith("backbone."))
        print(
            f"✅ Đã nạp thành công {len(filtered_dict)} layers từ checkpoint "
            f"(backbone: {loaded_backbone}, head: {loaded_head})")

    # Load weights
    def load_weights(self, weight_path: str):
        """Wrapper tiện dụng để nạp trọng số từ đường dẫn."""
        self.load_unirec_weights(weight_path)

    # Save weights
    def save_weights(self, weight_path):
        torch.save(self.state_dict(), weight_path)

    def forward(self, x: torch.Tensor, country_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass với SVTRv2 + STN + country embedding.

        Args:
            x: [Batch, Frames, 3, H, W]
            country_ids: [Batch] (mỗi phần tử là ID quốc gia)
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

        # --- Giai đoạn 3: Multi-frame Fusion (5 LQ + 5 HQ) ---
        # AttentionFusion nhận all-frames và trả về 1 feature map / sample
        # Input: [B*F, C, H', W'], Output: [B, C, H', W']
        fused = self.fusion(features)

        # --- Giai đoạn 4: Nhúng và kết hợp thông tin quốc gia ---
        # Chuyển H' về 1 để tạo chuỗi theo chiều ngang
        b_f, c_f, h_f, w_f = fused.size()
        fused_seq = F.adaptive_avg_pool2d(fused, (1, w_f)).squeeze(2)  # [B, C, W']
        fused_seq = fused_seq.permute(0, 2, 1)  # [B, W', C]

        # country_emb: [B, 64] -> broadcast theo chiều time (W')
        c_vec = self.country_emb(country_ids)  # [B, 64]
        c_vec = c_vec.unsqueeze(1).expand(-1, w_f, -1)  # [B, W', 64]

        # Kết hợp đặc trưng hình ảnh và embedding quốc gia
        combined = torch.cat([fused_seq, c_vec], dim=-1)  # [B, W', 384+64]
        combined = self.decoder_proj(combined)  # [B, W', 384]

        # RCTCDecoder mong đợi input có chiều Height = 1
        logits = self.head(combined.permute(0, 2, 1).unsqueeze(2))

        # Apply temperature scaling
        if self.use_temp_scaling and hasattr(self, 'temp_scaling'):
            logits = self.temp_scaling(logits)

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
