"""
STRATEGY 1: DYNAMIC WIDTH RESIZE
Tối ưu cho License Plate Recognition với multi-frame input
"""

import cv2
import numpy as np
import math


class LicensePlateResize:
    """
    Resize license plate images với dynamic width.
    
    Features:
    - Fixed height (32px) cho consistency
    - Dynamic width dựa trên aspect ratio (min=64, max=256)
    - Padding để batch processing
    - Giữ aspect ratio để tránh distortion
    """
    
    def __init__(
        self,
        img_height=32,
        img_width_min=64,
        img_width_max=256,
        padding=True,
        **kwargs
    ):
        self.img_height = img_height
        self.img_width_min = img_width_min
        self.img_width_max = img_width_max
        self.padding = padding
        
    def __call__(self, data):
        """
        Args:
            data: dict with 'image' key
        Returns:
            data: dict with resized 'image' and 'valid_ratio'
        """
        img = data['image']  # [H, W, C]
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
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize: [0, 255] → [-1, 1]
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        
        if self.padding:
            # Padding to max width
            padding_im = np.zeros(
                (3, self.img_height, self.img_width_max), 
                dtype=np.float32
            )
            padding_im[:, :, :target_width] = resized_image
            data['image'] = padding_im
            data['valid_ratio'] = target_width / self.img_width_max
        else:
            data['image'] = resized_image
            data['valid_ratio'] = 1.0
            
        return data


class MultiFrameLicensePlateResize:
    """
    Resize cho multi-frame (5 LR + 5 HR frames).
    
    IMPORTANT: Tất cả 5 frames trong 1 track phải có CÙNG width
    để có thể stack thành tensor [5, 3, H, W]
    """
    
    def __init__(
        self,
        img_height=32,
        img_width_min=64,
        img_width_max=256,
        padding=True,
        **kwargs
    ):
        self.img_height = img_height
        self.img_width_min = img_width_min
        self.img_width_max = img_width_max
        self.padding = padding
        
    def resize_frame(self, img):
        """Resize single frame."""
        h, w = img.shape[:2]
        ratio = w / float(h)
        target_width = int(self.img_height * ratio)
        target_width = max(self.img_width_min, target_width)
        target_width = min(self.img_width_max, target_width)
        
        resized = cv2.resize(
            img,
            (target_width, self.img_height),
            interpolation=cv2.INTER_LINEAR
        )
        return resized, target_width
    
    def __call__(self, frames_list):
        """
        Args:
            frames_list: List of 5 images [H, W, C]
        Returns:
            stacked: [5, 3, H, W_padded]
            valid_ratio: float
        """
        # Resize tất cả frames và tìm max width
        resized_frames = []
        widths = []
        
        for img in frames_list:
            resized, width = self.resize_frame(img)
            resized_frames.append(resized)
            widths.append(width)
        
        # Dùng max width để padding đồng nhất
        max_width = max(widths)
        
        # Normalize và padding
        normalized_frames = []
        for resized in resized_frames:
            # Normalize
            norm = resized.astype('float32')
            norm = norm.transpose((2, 0, 1)) / 255.0
            norm = (norm - 0.5) / 0.5
            
            # Padding
            if self.padding:
                padded = np.zeros(
                    (3, self.img_height, self.img_width_max),
                    dtype=np.float32
                )
                padded[:, :, :norm.shape[2]] = norm
                normalized_frames.append(padded)
            else:
                normalized_frames.append(norm)
        
        stacked = np.stack(normalized_frames, axis=0)  # [5, 3, H, W]
        valid_ratio = max_width / self.img_width_max
        
        return stacked, valid_ratio


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    import torch
    
    # Single frame example
    print("="*60)
    print("TEST 1: Single Frame Resize")
    print("="*60)
    
    single_resizer = LicensePlateResize(
        img_height=32,
        img_width_min=64,
        img_width_max=256,
        padding=True
    )
    
    # Simulate license plate image (short)
    img_short = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    data_short = {'image': img_short}
    result_short = single_resizer(data_short)
    print(f"Short plate (40x120) → {result_short['image'].shape}")
    print(f"Valid ratio: {result_short['valid_ratio']:.3f}")
    
    # Simulate license plate image (long)
    img_long = np.random.randint(0, 255, (40, 320, 3), dtype=np.uint8)
    data_long = {'image': img_long}
    result_long = single_resizer(data_long)
    print(f"Long plate (40x320) → {result_long['image'].shape}")
    print(f"Valid ratio: {result_long['valid_ratio']:.3f}")
    
    # Multi-frame example
    print("\n" + "="*60)
    print("TEST 2: Multi-Frame Resize")
    print("="*60)
    
    multi_resizer = MultiFrameLicensePlateResize(
        img_height=32,
        img_width_min=64,
        img_width_max=256,
        padding=True
    )
    
    # Simulate 5 frames
    frames = [
        np.random.randint(0, 255, (40, 120 + i*10, 3), dtype=np.uint8)
        for i in range(5)
    ]
    
    stacked, valid_ratio = multi_resizer(frames)
    print(f"5 frames → {stacked.shape}")
    print(f"Valid ratio: {valid_ratio:.3f}")
    
    # Convert to torch for model
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # [1, 5, 3, 32, 256]
    print(f"Torch tensor: {tensor.shape}")