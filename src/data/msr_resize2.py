"""
MSR resizing utilities for license plate recognition.

Contains:
- LicensePlateResize: single-frame dynamic-width resize with padding.
- MultiFrameLicensePlateResize: multi-frame (sequence) resize ensuring consistent width.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class LicensePlateResize:
    """
    Resize license plate images với dynamic width.

    Features:
    - Fixed height (mặc định 32px) cho consistency
    - Dynamic width dựa trên aspect ratio (min=64, max=256)
    - Padding để batch processing
    - Giữ aspect ratio để tránh distortion
    """

    def __init__(
        self,
        img_height: int = 32,
        img_width_min: int = 64,
        img_width_max: int = 256,
        padding: bool = True,
        **kwargs,
    ):
        self.img_height = img_height
        self.img_width_min = img_width_min
        self.img_width_max = img_width_max
        self.padding = padding

    def __call__(self, data: Dict) -> Dict:
        """
        Args:
            data: dict với key 'image' (H, W, C) uint8

        Returns:
            data: dict với:
                - 'image': (C, H, W_max) float32, chuẩn hóa [-1, 1] nếu padding=True
                - 'valid_ratio': tỉ lệ width thực / width_max
        """
        img = data["image"]  # [H, W, C]
        h, w = img.shape[:2]

        # Tính target width dựa trên aspect ratio
        ratio = w / float(h)
        target_width = int(self.img_height * ratio)

        # Clamp width trong khoảng [min, max]
        target_width = max(self.img_width_min, target_width)
        target_width = min(self.img_width_max, target_width)

        # Resize
        resized_image = cv2.resize(
            img,
            (target_width, self.img_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize: [0, 255] → [-1, 1]
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image = (resized_image - 0.5) / 0.5

        if self.padding:
            # Padding to max width
            padding_im = np.zeros(
                (3, self.img_height, self.img_width_max), dtype=np.float32
            )
            padding_im[:, :, :target_width] = resized_image
            data["image"] = padding_im
            data["valid_ratio"] = target_width / self.img_width_max
        else:
            data["image"] = resized_image
            data["valid_ratio"] = 1.0

        return data


class MultiFrameLicensePlateResize:
    """
    Resize cho multi-frame (ví dụ 5 LR frames per track).

    IMPORTANT: Tất cả frames trong 1 track phải có CÙNG width
    để có thể stack thành tensor [T, C, H, W].
    """

    def __init__(
        self,
        img_height: int = 32,
        img_width_min: int = 64,
        img_width_max: int = 256,
        padding: bool = True,
        **kwargs,
    ):
        self.img_height = img_height
        self.img_width_min = img_width_min
        self.img_width_max = img_width_max
        self.padding = padding

    def resize_frame(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """Resize single frame HWC → resized HWC và trả về (resized, width)."""
        h, w = img.shape[:2]
        ratio = w / float(h)
        target_width = int(self.img_height * ratio)
        target_width = max(self.img_width_min, target_width)
        target_width = min(self.img_width_max, target_width)

        resized = cv2.resize(
            img,
            (target_width, self.img_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, target_width

    def __call__(self, frames_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Args:
            frames_list: List các ảnh HWC (uint8)

        Returns:
            stacked: (T, C, H, W_max) float32
            valid_ratio: max_width / self.img_width_max
        """
        resized_frames: List[np.ndarray] = []
        widths: List[int] = []

        for img in frames_list:
            resized, width = self.resize_frame(img)
            resized_frames.append(resized)
            widths.append(width)

        max_width = max(widths)

        normalized_frames: List[np.ndarray] = []
        for resized in resized_frames:
            # Normalize
            norm = resized.astype("float32")
            norm = norm.transpose((2, 0, 1)) / 255.0
            norm = (norm - 0.5) / 0.5

            if self.padding:
                padded = np.zeros(
                    (3, self.img_height, self.img_width_max), dtype=np.float32
                )
                padded[:, :, : norm.shape[2]] = norm
                normalized_frames.append(padded)
            else:
                normalized_frames.append(norm)

        stacked = np.stack(normalized_frames, axis=0)  # [T, 3, H, W_max]
        valid_ratio = max_width / self.img_width_max

        return stacked, valid_ratio

