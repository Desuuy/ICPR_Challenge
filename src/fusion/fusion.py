import torch.nn as nn
import torch
import torch.nn.functional as F


class GatedAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        Gated Attention Fusion từ paper "Attention-based Deep Multiple Instance Learning".
        Công thức: a_k = softmax(w^T * (tanh(V * h_k) . sigmoid(U * h_k)))

        Args:
            input_dim: Kích thước embed_dim từ ViT (thường là 384)
            hidden_dim: Kích thước lớp ẩn L (paper dùng 128)
        """
        super().__init__()
        self.L = hidden_dim

        # 1. Feature Transformation V (Tanh branch)
        self.attn_v = nn.Sequential(
            nn.Linear(input_dim, self.L),
            nn.Tanh()
        )

        # 2. Gating Mechanism U (Sigmoid branch)
        self.attn_u = nn.Sequential(
            nn.Linear(input_dim, self.L),
            nn.Sigmoid()
        )

        # 3. Attention Weights w
        self.attn_w = nn.Linear(self.L, 1)

    def forward(self, x):
        # Input x từ Encoder: [Batch * 5, Seq_Len, Embed_Dim]
        bs_5, seq_len, embed_dim = x.shape
        num_frames = 5
        bs = bs_5 // num_frames

        # 1. Reshape để tách biệt chiều Frame
        # Shape: [Batch, 5, Seq_Len, Embed_Dim]
        x = x.view(bs, num_frames, seq_len, embed_dim)

        # 2. Tính Attention Scores (Gated Mechanism)
        # Áp dụng Linear layers lên toàn bộ tensor (PyTorch tự handle dims cuối)

        # [Batch, 5, Seq_Len, L]
        v_out = self.attn_v(x)
        u_out = self.attn_u(x)

        # Element-wise multiplication (Hadamard product) -> Gating
        # [Batch, 5, Seq_Len, L]
        gated_feat = v_out * u_out

        # Tính điểm số thô (Logits)
        # [Batch, 5, Seq_Len, 1]
        scores = self.attn_w(gated_feat)

        # 3. Softmax trên chiều Frame (dim=1) để chuẩn hóa trọng số
        # [Batch, 5, Seq_Len, 1]
        attn_weights = F.softmax(scores, dim=1)

        # 4. Cộng gộp có trọng số (Weighted Sum)
        # x: [Batch, 5, Seq_Len, Dim]
        # weights: [Batch, 5, Seq_Len, 1] (Broadcasting)
        # Output: [Batch, Seq_Len, Dim]
        fused_x = torch.sum(x * attn_weights, dim=1)

        return fused_x