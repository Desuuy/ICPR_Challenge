"""
PATCH DATASET.PY - TÍCH HỢP MSR

Thay đổi trong MultiFrameDataset.__init__() và __getitem__()
"""

# ============================================================
# THÊM VÀO ĐẦU FILE dataset.py
# ============================================================

from src.data.msr_resize import LicensePlateResize  # Import MSR class


# ============================================================
# TRONG __init__() - Thêm MSR transform
# ============================================================

class MultiFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        # ... existing params ...
        use_msr: bool = False,  # ← THÊM PARAM MỚI
        msr_width_min: int = 64,
        msr_width_max: int = 256,
    ):
        # ... existing code ...
        
        self.use_msr = use_msr
        
        # Initialize transform
        if mode == 'train':
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
        else:
            self.transform = get_val_transforms(img_height, img_width)
        
        # THÊM MSR transform (nếu enabled)
        if self.use_msr:
            self.msr_transform = LicensePlateResize(
                img_height=img_height,
                img_width_min=msr_width_min,
                img_width_max=msr_width_max,
                padding=True
            )
        else:
            self.msr_transform = None


# ============================================================
# TRONG __getitem__() - Áp dụng MSR
# ============================================================

def __getitem__(self, idx: int):
    item = self.samples[idx]
    img_paths = item['paths']
    # ... existing code ...
    
    images_list = []
    
    for p in img_paths:
        image = cv2.imread(p, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply degradation first (if synthetic)
        if is_synthetic and self.degrade:
            image = self.degrade(image=image)['image']
        
        # THAY ĐỔI: Áp dụng MSR hoặc transform thông thường
        if self.msr_transform is not None:
            # MSR path
            data = {'image': image}
            data = self.msr_transform(data)
            image_tensor = torch.from_numpy(data['image'])  # [3, H, W]
        else:
            # Original path (fixed size)
            if sample_seed is not None:
                random.seed(sample_seed)
                np.random.seed(sample_seed)
            image = self.transform(image=image)['image']
            image_tensor = image
        
        images_list.append(image_tensor)
    
    images_tensor = torch.stack(images_list, dim=0)
    
    # ... rest of code ...


# ============================================================
# SỬA TRONG train.py - PASS use_msr argument
# ============================================================

# Common dataset parameters
common_ds_params = {
    'split_ratio': config.SPLIT_RATIO,
    'img_height': config.IMG_HEIGHT,
    'img_width': config.IMG_WIDTH,
    'char2idx': config.CHAR2IDX,
    'val_split_file': config.VAL_SPLIT_FILE,
    'seed': config.SEED,
    'augmentation_level': config.AUGMENTATION_LEVEL,
    'same_aug_per_sample': getattr(config, 'SAME_AUG_PER_SAMPLE', True),
    'use_msr': getattr(config, 'USE_MSR', False),  # ← THÊM
    'msr_width_min': getattr(config, 'MSR_WIDTH_MIN', 64),  # ← THÊM
    'msr_width_max': getattr(config, 'MSR_WIDTH_MAX', 256),  # ← THÊM
}


# ============================================================
# THÊM VÀO configs/config.py
# ============================================================

@dataclass
class Config:
    # ... existing fields ...
    
    # Multi-Scale Resize (MSR)
    USE_MSR: bool = False  # Set True to enable
    MSR_WIDTH_MIN: int = 64
    MSR_WIDTH_MAX: int = 256
    
    # ... rest of config ...