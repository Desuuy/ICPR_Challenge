"""Adapter/wrapper cho mô hình super-resolution MF-LPR (LP-Diff).

Mục tiêu:
    - Load mô hình SR từ thư mục `sr_model/` với checkpoint do bạn cung cấp.
    - Cung cấp API đơn giản để:
        + Nâng SR cho bộ 3 frame (LR1, LR2, LR3) -> 1 frame SR.
        + Nâng SR cho cả sequence nhiều frame (dùng cửa sổ 3 frame trượt).

Lưu ý:
    - Module này dùng cho INFERENCE (eval) trong pipeline OCR, không dùng train lại SR.
    - Đầu vào/đầu ra đều là tensor đã normalize theo chuẩn [-1, 1]
      (mean=0.5, std=0.5) giống pipeline đang dùng trong OCR.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F


class MF_LPR_SR:
    """Wrapper tiện dụng quanh mô hình LP-Diff trong `sr_model/`.

    Ví dụ dùng cơ bản:

    ```python
    from src.sr.mf_lpr_sr import MF_LPR_SR
    sr = MF_LPR_SR(
        checkpoint_path="path/to/Ixxxx_Exxx_gen_best_psnr.pth",
        config_path="sr_model/config/LP-Diff.json",
        device=config.DEVICE,
    )

    # lr_seq: Tensor (T, C, H, W), đã Normalize [-1, 1]
    sr_seq = sr.enhance_sequence(lr_seq)
    ```
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "sr_model/config/LP-Diff.json",
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            checkpoint_path: Đường dẫn tới file checkpoint GEN của LP-Diff,
                ví dụ: `.../I10000_E233_gen_best_psnr.pth` hoặc `..._gen.pth`.
            config_path: Đường dẫn tới file JSON cấu hình LP-Diff
                (mặc định dùng `sr_model/config/LP-Diff.json` trong repo).
            device: Thiết bị để chạy mô hình (torch.device).
        """
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        project_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))
        sr_model_root = os.path.join(project_root, "sr_model")

        # Đảm bảo folder `sr_model/` nằm trong sys.path để import đúng kiểu của code gốc
        if os.path.isdir(sr_model_root) and sr_model_root not in sys.path:
            sys.path.insert(0, sr_model_root)

        # Import theo phong cách nội bộ của `sr_model`
        try:
            from model import networks as sr_networks  # type: ignore
        except ImportError as e:  # pragma: no cover - environment-specific
            raise ImportError(
                f"Không thể import 'model.networks' từ thư mục sr_model. "
                f"Đã tìm ở: {sr_model_root}. Chi tiết: {e}"
            ) from e

        # Load cấu hình LP-Diff
        cfg_path_resolved = config_path
        if not os.path.isabs(cfg_path_resolved):
            cfg_path_resolved = os.path.join(project_root, config_path)

        if not os.path.exists(cfg_path_resolved):
            raise FileNotFoundError(
                f"Không tìm thấy file config SR: {cfg_path_resolved}"
            )

        with open(cfg_path_resolved, "r") as f:
            opt = json.load(f)

        # Bổ sung/override một số field cần thiết cho inference độc lập
        opt["phase"] = "val"
        # Không dùng distributed trong pipeline OCR hiện tại
        opt.setdefault("distributed", False)
        # Nếu không dùng DataParallel, để gpu_ids rỗng cho an toàn
        opt.setdefault("gpu_ids", [])

        self.opt = opt

        # Khởi tạo mạng GaussianDiffusion + MTA từ config
        netG = sr_networks.define_G(opt)
        netG = netG.to(self.device)

        # Thiết lập noise schedule cho pha inference (val)
        if "model" in opt and "beta_schedule" in opt["model"]:
            schedule_opt = opt["model"]["beta_schedule"].get(
                "val") or opt["model"]["beta_schedule"].get("train")
            if schedule_opt is None:
                raise ValueError(
                    "Cấu hình LP-Diff thiếu 'beta_schedule' cho 'train'/'val'.")
            # GaussianDiffusion.set_new_noise_schedule
            netG.set_new_noise_schedule(schedule_opt, device=self.device)

        # Load checkpoint GEN trực tiếp vào netG
        ckpt_path_resolved = checkpoint_path
        if not os.path.isabs(ckpt_path_resolved):
            ckpt_path_resolved = os.path.join(project_root, checkpoint_path)

        if not os.path.exists(ckpt_path_resolved):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint SR: {ckpt_path_resolved}"
            )

        state = torch.load(ckpt_path_resolved, map_location=self.device)
        # Hỗ trợ cả dict thuần và dict có 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Bỏ qua các key diffusion schedule nếu shape không khớp (n_timestep khác:
        # checkpoint 1000 vs config 100). Chúng sẽ được tính lại bởi set_new_noise_schedule.
        model_dict = netG.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        for k, v in state.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped_mismatch.append(k)
            # unexpected keys: bỏ qua

        if skipped_mismatch:
            print(
                f"[MF_LPR_SR] Bỏ qua {len(skipped_mismatch)} key (shape mismatch, n_timestep khác) - sẽ dùng schedule từ config."
            )

        missing, unexpected = netG.load_state_dict(filtered_state, strict=False)
        if missing:
            print(
                f"[MF_LPR_SR] Cảnh báo: thiếu {len(missing)} key khi load checkpoint SR.")
        if unexpected:
            print(
                f"[MF_LPR_SR] Cảnh báo: thừa {len(unexpected)} key khi load checkpoint SR.")

        netG.eval()
        self.netG = netG

    @torch.no_grad()
    def enhance_triplet(
        self,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        lr3: torch.Tensor,
        continous: bool = False,
        resize_to: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Nâng SR cho 3 frame LR (1 bộ) -> 1 frame SR.

        Args:
            lr1, lr2, lr3:
                Tensor dạng (C, H, W), giá trị đã normalize [-1, 1].
            continous:
                Nếu True, trả về SR theo dạng "continuous" như code gốc (ít dùng).
            resize_to:
                Nếu được set (H, W), sẽ resize output SR về đúng kích thước này
                (ví dụ về lại (32, 128) để feed vào OCR).

        Returns:
            Tensor (C, H_out, W_out), normalized [-1, 1].
        """
        # Chuẩn hóa shape + device
        lr1 = lr1.unsqueeze(0).to(
            self.device, non_blocking=True)  # (1, C, H, W)
        lr2 = lr2.unsqueeze(0).to(self.device, non_blocking=True)
        lr3 = lr3.unsqueeze(0).to(self.device, non_blocking=True)

        # MTA: Multi-temporal attention fusion để tạo điều kiện cho diffusion
        condition = self.netG.MTA(lr1, lr2, lr3)  # (1, C, H, W)

        # Diffusion-based super-resolution (residual trên condition)
        sr = self.netG.super_resolution(
            condition, continous=continous)  # (1, C, H, W)

        if resize_to is not None:
            # resize_to: (H, W)
            sr = F.interpolate(sr, size=resize_to,
                               mode="bilinear", align_corners=False)

        return sr.squeeze(0).cpu()

    @torch.no_grad()
    def enhance_sequence(
        self,
        frames: torch.Tensor,
        continous: bool = False,
        resize_to: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Nâng SR cho cả sequence nhiều frame của một sample.

        Dùng cửa sổ 3 frame trượt: với frame i sẽ lấy:
            - LR1 = frame i-1 (hoặc i nếu i=0)
            - LR2 = frame i
            - LR3 = frame i+1 (hoặc i nếu là frame cuối)

        Args:
            frames:
                Tensor (T, C, H, W), normalized [-1, 1].
            continous:
                Tham số truyền vào diffusion.super_resolution.
            resize_to:
                Nếu set, resize tất cả frame SR về (H, W) này.

        Returns:
            Tensor (T, C, H_out, W_out), cùng số frame như input.
        """
        if frames.dim() != 4:
            raise ValueError(
                f"'frames' phải có shape (T, C, H, W), nhận được {tuple(frames.shape)}"
            )

        T, C, H, W = frames.shape
        enhanced_list = []

        for i in range(T):
            i0 = max(i - 1, 0)
            i1 = i
            i2 = min(i + 1, T - 1)

            sr_frame = self.enhance_triplet(
                frames[i0],
                frames[i1],
                frames[i2],
                continous=continous,
                resize_to=resize_to,
            )
            enhanced_list.append(sr_frame)

        enhanced = torch.stack(enhanced_list, dim=0)  # (T, C, H_out, W_out)
        return enhanced


__all__ = ["MF_LPR_SR"]
