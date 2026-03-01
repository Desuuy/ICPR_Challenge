"""
STRATEGY 2: MULTI-SCALE BUCKETS
Phức tạp hơn nhưng performance tốt hơn cho varied aspect ratios
"""

import cv2
import numpy as np
import math


class MultiScaleBucketResize:
    """
    Resize license plate vào các "buckets" theo aspect ratio.
    
    Inspiration: PaddleOCR's LongResize
    
    Buckets:
    - Ratio 2:1 → 64x32
    - Ratio 3:1 → 96x32
    - Ratio 4:1 → 128x32
    - Ratio 5:1 → 160x32
    - Ratio 6:1 → 192x32
    - Ratio 7:1 → 224x32
    - Ratio 8:1 → 256x32
    """
    
    def __init__(
        self,
        base_height=32,
        base_shapes=[
            [64, 32],   # Ratio 2:1
            [96, 32],   # Ratio 3:1
            [128, 32],  # Ratio 4:1
            [160, 32],  # Ratio 5:1
            [192, 32],  # Ratio 6:1
            [224, 32],  # Ratio 7:1
            [256, 32],  # Ratio 8:1
        ],
        max_ratio=8,
        padding=True,
        **kwargs
    ):
        self.base_height = base_height
        self.base_shapes = base_shapes
        self.max_ratio = max_ratio
        self.padding = padding
        
    def __call__(self, data):
        """
        Args:
            data: dict with 'image' key
        Returns:
            data: dict with resized 'image', 'valid_ratio', 'bucket_id'
        """
        img = data['image']
        h, w = img.shape[:2]
        
        # Tính aspect ratio
        ratio = w / float(h)
        gen_ratio = max(1, round(ratio))  # Round to nearest integer
        gen_ratio = min(gen_ratio, self.max_ratio)
        
        # Chọn bucket shape
        if gen_ratio <= len(self.base_shapes):
            target_w, target_h = self.base_shapes[gen_ratio - 1]
        else:
            target_w = self.base_height * gen_ratio
            target_h = self.base_height
            
        # Resize với padding
        if not self.padding:
            resized = cv2.resize(img, (target_w, target_h))
            resized_w = target_w
        else:
            # Keep aspect ratio
            if math.ceil(target_h * ratio) > target_w:
                resized_w = target_w
            else:
                resized_w = int(math.ceil(target_h * ratio))
                resized_w = min(target_w, resized_w)
            
            resized = cv2.resize(img, (resized_w, target_h))
        
        # Normalize
        resized = resized.astype('float32')
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5
        
        # Padding
        padding_im = np.zeros((3, target_h, target_w), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized
        
        data['image'] = padding_im
        data['valid_ratio'] = resized_w / target_w
        data['bucket_id'] = gen_ratio
        data['target_shape'] = (target_h, target_w)
        
        return data


class AdaptiveMultiScaleResize:
    """
    Adaptive MSR: Chọn scale tốt nhất cho mỗi track.
    
    Ưu điểm:
    - Tự động chọn scale phù hợp
    - Giảm padding waste
    - Tăng độ rõ nét cho text nhỏ
    """
    
    def __init__(
        self,
        base_height=32,
        min_width=64,
        max_width=256,
        width_step=32,  # Bước nhảy width
        padding=True,
        **kwargs
    ):
        self.base_height = base_height
        self.min_width = min_width
        self.max_width = max_width
        self.width_step = width_step
        self.padding = padding
        
        # Generate available widths
        self.available_widths = list(
            range(min_width, max_width + width_step, width_step)
        )
        
    def __call__(self, data):
        img = data['image']
        h, w = img.shape[:2]
        
        # Tính target width dựa trên aspect ratio
        ratio = w / float(h)
        ideal_width = int(self.base_height * ratio)
        
        # Chọn width gần nhất trong available_widths
        target_width = min(
            self.available_widths,
            key=lambda x: abs(x - ideal_width)
        )
        
        # Resize
        if not self.padding:
            resized = cv2.resize(img, (target_width, self.base_height))
            resized_w = target_width
        else:
            if math.ceil(self.base_height * ratio) > target_width:
                resized_w = target_width
            else:
                resized_w = int(math.ceil(self.base_height * ratio))
                resized_w = min(target_width, resized_w)
            
            resized = cv2.resize(img, (resized_w, self.base_height))
        
        # Normalize
        resized = resized.astype('float32')
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5
        
        # Padding
        padding_im = np.zeros(
            (3, self.base_height, target_width),
            dtype=np.float32
        )
        padding_im[:, :, :resized_w] = resized
        
        data['image'] = padding_im
        data['valid_ratio'] = resized_w / target_width
        data['selected_width'] = target_width
        
        return data


# ============================================================
# COMPARISON TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("COMPARING MSR STRATEGIES")
    print("="*60)
    
    # Test images với aspect ratios khác nhau
    test_images = [
        ("Short plate", (40, 80, 3)),   # Ratio ~2:1
        ("Medium plate", (40, 160, 3)), # Ratio ~4:1
        ("Long plate", (40, 320, 3)),   # Ratio ~8:1
    ]
    
    # Strategy 1: Multi-Scale Buckets
    print("\nSTRATEGY 1: Multi-Scale Buckets")
    print("-" * 60)
    bucket_resizer = MultiScaleBucketResize()
    
    for name, shape in test_images:
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
        data = {'image': img}
        result = bucket_resizer(data)
        print(f"{name} {shape[:2]} → {result['image'].shape} "
              f"(bucket {result['bucket_id']}, "
              f"ratio {result['valid_ratio']:.2f})")
    
    # Strategy 2: Adaptive
    print("\nSTRATEGY 2: Adaptive MSR")
    print("-" * 60)
    adaptive_resizer = AdaptiveMultiScaleResize(
        width_step=32
    )
    
    for name, shape in test_images:
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
        data = {'image': img}
        result = adaptive_resizer(data)
        print(f"{name} {shape[:2]} → {result['image'].shape} "
              f"(width {result['selected_width']}, "
              f"ratio {result['valid_ratio']:.2f})")