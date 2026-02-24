"""
Multi-Scale Resize (MSR) cho License Plate Recognition.
FIXED VERSION: Luôn trả về MAX_WIDTH với padding để tương thích với batching.
"""

import cv2
import numpy as np
import math


class LicensePlateResize:
    """
    Resize license plate images với dynamic width + padding to max.
    
    CRITICAL: Luôn trả về shape [3, H, MAX_WIDTH] để có thể stack thành batch!
    
    Features:
    - Fixed height (32px)
    - Dynamic width based on aspect ratio
    - ALWAYS pad to max_width để batching works
    """
    
    def __init__(
        self,
        img_height=32,
        img_width_min=64,
        img_width_max=256,
        padding=True,
        **kwargs
    ):
        print(">>> USING LicensePlateResize FROM msr_resize.py")

        self.img_height = img_height
        self.img_width_min = img_width_min
        self.img_width_max = img_width_max
        self.padding = padding
        
        
        print(f"📐 [MSR] Init: H={img_height}, W=[{img_width_min}, {img_width_max}], padding={padding}")
        
    def __call__(self, data):
        """
        Args:
            data: dict with 'image' key [H, W, C]
        Returns:
            data: dict with:
                - 'image': [3, H, MAX_WIDTH] (ALWAYS MAX_WIDTH!)
                - 'valid_ratio': actual_width / max_width
        """
        img = data['image']  # [H, W, C]
        h, w = img.shape[:2]
        
        # Tính target width dựa trên aspect ratio
        ratio = w / float(h)
        target_width = int(self.img_height * ratio)
        
        # Clamp width trong khoảng [min, max]
        target_width = max(self.img_width_min, target_width)
        target_width = min(self.img_width_max, target_width)
        
        # Resize to target size
        resized_image = cv2.resize(
            img, 
            (target_width, self.img_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize: [0, 255] → [-1, 1]
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        
        # CRITICAL: ALWAYS pad to MAX_WIDTH để batch có thể stack!
        padding_im = np.zeros(
            (3, self.img_height, self.img_width_max), 
            dtype=np.float32
        )
        padding_im[:, :, :target_width] = resized_image
        
        data['image'] = padding_im  # [3, H, MAX_WIDTH]
        data['valid_ratio'] = target_width / self.img_width_max
        
        return data


class MultiFrameLicensePlateResize:
    """
    Resize cho multi-frame (5 frames).
    
    CRITICAL: Tất cả frames trong 1 track phải có CÙNG width (MAX_WIDTH)
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
        """Resize single frame to target width."""
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
            stacked: [5, 3, H, MAX_WIDTH] (ALWAYS MAX_WIDTH!)
            valid_ratio: float
        """
        # Resize tất cả frames
        resized_frames = []
        widths = []
        
        for img in frames_list:
            resized, width = self.resize_frame(img)
            resized_frames.append(resized)
            widths.append(width)
        
        # Use max width từ 5 frames
        max_width = max(widths)
        
        # Normalize và padding ALL frames to MAX_WIDTH
        normalized_frames = []
        for resized in resized_frames:
            # Normalize
            norm = resized.astype('float32')
            norm = norm.transpose((2, 0, 1)) / 255.0
            norm = (norm - 0.5) / 0.5
            
            # CRITICAL: Pad to MAX_WIDTH
            padded = np.zeros(
                (3, self.img_height, self.img_width_max),
                dtype=np.float32
            )
            padded[:, :, :norm.shape[2]] = norm
            normalized_frames.append(padded)
        
        stacked = np.stack(normalized_frames, axis=0)  # [5, 3, H, MAX_WIDTH]
        valid_ratio = max_width / self.img_width_max
        
        return stacked, valid_ratio


# ============================================================
# USAGE TEST
# ============================================================

if __name__ == "__main__":
    import torch
    
    print("="*60)
    print("TEST: MSR with FIXED MAX WIDTH")
    print("="*60)
    
    resizer = LicensePlateResize(
        img_height=32,
        img_width_min=64,
        img_width_max=256,
        padding=True
    )
    
    # Test với 3 images có aspect ratio khác nhau
    test_images = [
        ("Short", (40, 80, 3)),    # Ratio 2:1
        ("Medium", (40, 160, 3)),  # Ratio 4:1
        ("Long", (40, 320, 3)),    # Ratio 8:1
    ]
    
    outputs = []
    for name, shape in test_images:
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
        data = {'image': img}
        result = resizer(data)
        
        print(f"\n{name} plate {shape[:2]}:")
        print(f"  Output shape: {result['image'].shape}")
        print(f"  Valid ratio: {result['valid_ratio']:.3f}")
        print(f"  Actual width: {int(result['valid_ratio'] * 256)}px")
        
        outputs.append(result['image'])
    
    # CRITICAL TEST: Có thể stack không?
    try:
        stacked = np.stack(outputs, axis=0)
        print(f"\n✅ CAN STACK: {stacked.shape}")
        print("✅ MSR COMPATIBLE WITH BATCHING!")
    except Exception as e:
        print(f"\n❌ CANNOT STACK: {e}")
        print("❌ MSR NOT COMPATIBLE!")