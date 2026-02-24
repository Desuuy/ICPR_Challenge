"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.data.msr_resize2 import MultiFrameLicensePlateResize, LicensePlateResize

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
    get_msr_svtv2_transforms,
)

from src.openrec.preprocess.resize import resize_norm_img_long
# MSR (Multi-Size Resizing) chuẩn SVTRv2 từ openrec: base_shape [W, H].
# Dùng H=32 để khớp backbone max_sz=[32, 128]; width 64,96,112,128.
MSR_BASE_SHAPE_H32: List[List[int]] = [
    [64, 32],
    [96, 32],
    [112, 32],
    [128, 32],
]


def _msr_from_openrec_to_albumentations(data: Dict[str, Any]) -> np.ndarray:
    """
    Chuyển ảnh sau resize_norm_img_long (CHW, float [-1,1]) sang HWC uint8 [0,255]
    để đưa vào pipeline Albumentations.
    """
    x = data["image"]  # (3, H, W), giá trị [-1, 1]
    x = (x + 1.0) * 0.5 * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x.transpose(1, 2, 0)  # HWC


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.

    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
        same_aug_per_sample: bool = True,
        sr_enhancer: Any = None,
        use_msr: bool = False,  # ← THÊM
        msr_width_min: int = 64,  # ← THÊM
        msr_width_max: int = 256,  # ← THÊM
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            augmentation_level: 'full' or 'light' augmentation for training.
            is_test: If True, load test data without labels (for submission).
            full_train: If True, use all tracks for training (no val split).
            same_aug_per_sample: If True, use same augmentation for all 5 frames (recommended).
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.same_aug_per_sample = same_aug_per_sample
        # Optional super-resolution enhancer (MF_LPR_SR hoặc tương tự)
        self.sr_enhancer = sr_enhancer
        self.use_msr = use_msr 
        self.msr_width_min = msr_width_min 
        self.msr_width_max = msr_width_max 
        # MSR (Multi-Size Resizing) theo SVTRv2 cho training
        self.use_msr_svtv2 = mode == 'train' and augmentation_level == "msr_svtv2"

        if mode == 'train':
            # Training: apply augmentation on the fly
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)       
            #elif augmentation_level == "msr_svtv2":
                # MSR: resize + padding đa tỉ lệ sẽ làm thủ công trong __getitem__
                # Transform chỉ còn augment + normalize + ToTensor.
            #    self.transform = get_msr_svtv2_transforms()
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            # Validation or test: only resize and normalize
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        # Initialize MSR transform (dynamic-width)
        self.use_msr = use_msr
        if self.use_msr:
            self.msr_transform = LicensePlateResize(
                img_height=self.img_height,
                img_width_min=self.msr_width_min,
                img_width_max=self.msr_width_max,
                padding=True
            )
            print(f"📐 MSR enabled: width range [{self.msr_width_min}, {self.msr_width_max}]")
        else:
            self.msr_transform = None
        

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        # Handle test mode differently
        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
            print(f"-> Total: {len(self.samples)} test samples.")
        else:
            train_tracks, val_tracks = self._load_or_create_split(
                all_tracks, split_ratio)

            selected_tracks = train_tracks if mode == 'train' else val_tracks
            print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")

            self._index_samples(selected_tracks)
            print(f"-> Total: {len(self.samples)} samples.")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority."""
        # If full_train mode, return all tracks as training
        if self.full_train:
            print(
                "📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []

        train_tracks, val_tracks = [], []

        # 1. Load split file if exists
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)

            # Check consistency: If val empty or no Scenario-B, recreate
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []  # Reset to trigger new split logic

        # 2. Create new split if needed
        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")

            # Filter Scenario-B tracks
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]

            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks

            # Val size = (1 - split_ratio) * total_scenario_b
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))

            # Shuffle and take from beginning as val
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]

            # Train = (All) - (Val)
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]

            # Save track IDs (folder names)
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t)
                          for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks with LR/HQ pairs and country info."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]

                # Label
                label = data.get(
                    'plate_text',
                    data.get('license_plate', data.get('text', ''))
                )
                if not label:
                    continue

                # Country: ưu tiên field 'country', fallback theo Scenario-A/B trong path
                country_name = data.get(
                    'country',
                    'Scenario-B' if 'Scenario-B' in track_path else 'Scenario-A'
                )
                country_id = 1 if 'Scenario-B' in country_name else 0

                track_id = os.path.basename(track_path)

                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )

                # Chỉ nhận sample nếu đủ 5 LR và 5 HR
                if len(lr_files) == 5 and len(hr_files) == 5:
                    # Chuẩn hoá cấu trúc sample để __getitem__ dùng chung với test:
                    # - 'paths': danh sách đường dẫn ảnh đầu vào (ở đây là LR thật)
                    # - 'is_synthetic': đánh dấu có phải LR giả lập từ HR hay không
                    # Giữ lại 'lr_paths' và 'hr_paths' để sau này có thể dùng cho SR / debug nếu cần.
                    self.samples.append({
                        'paths': lr_files,          # dùng trong __getitem__
                        'is_synthetic': False,      # LR thật, không phải degrade từ HR
                        'lr_paths': lr_files,
                        'hr_paths': hr_files,
                        'label': label,
                        'country_id': country_id,
                        'track_id': track_id,
                    })
            except Exception:
                pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without labels."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            # Country: dựa vào tên folder Scenario-A/B (giống train/val)
            country_name = 'Scenario-B' if 'Scenario-B' in track_path else 'Scenario-A'
            country_id = 1 if 'Scenario-B' in country_name else 0

            # Load all LR images (sorted by frame number)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )

            if lr_files:
                self.samples.append({
                    'paths': lr_files,
                    'label': '',  # No label for test data
                    'is_synthetic': False,
                    'country_id': country_id,
                    'track_id': track_id
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str, List[str], int]:
        """Load exactly 5 frames (guaranteed by dataset structure).

        For training: applies degradation (if synthetic) then augmentation.
        For validation: applies degradation (if synthetic) then clean transform.
        For test: only applies clean transform, returns dummy targets.
        """
        item = self.samples[idx]
        img_paths = item['paths']
        label = item['label']
        is_synthetic = item['is_synthetic']
        track_id = item['track_id']
        # Country id (0/1) cho country embedding; fallback 0 nếu thiếu
        country_id = item.get('country_id', 0)

        images_list = []
        # Same augmentation for all 5 frames: fix RNG seed per sample so transform params match
        sample_seed = (self.seed + idx) % (2 **
                                           32) if self.same_aug_per_sample and self.mode == 'train' else None

        for p in img_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply degradation trước (nếu synthetic)
            if is_synthetic and self.degrade:
                image = self.degrade(image=image)['image']

            # Áp dụng MSR (dynamic-width) hoặc transform thường CHO TỪNG FRAME
            if self.msr_transform is not None:
                data = {'image': image}
                data = self.msr_transform(data)
                image_tensor = torch.from_numpy(data['image'])
            else:
                if sample_seed is not None:
                    random.seed(sample_seed)
                    np.random.seed(sample_seed)
                image_tensor = self.transform(image=image)['image']

            images_list.append(image_tensor)


        images_tensor = torch.stack(images_list, dim=0)

        # Optional: apply super-resolution enhancement per-sample
        if self.sr_enhancer is not None:
            # frames: (T, C, H, W) -> (T, C, H, W) sau SR
            images_tensor = self.sr_enhancer.enhance_sequence(
                images_tensor,
                resize_to=(self.img_height, self.img_width),
            )

        # Handle test mode (no labels)
        if self.is_test:
            target = [0]  # Dummy target
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)

        return (
            images_tensor,
            torch.tensor(target, dtype=torch.long),
            target_len,
            label,
            track_id,
            img_paths,
            country_id,
        )

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[tuple, ...],
        torch.Tensor,
    ]:
        """Custom collate function for DataLoader."""
        images, targets, target_lengths, labels_text, track_ids, img_paths, country_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        country_ids = torch.tensor(country_ids, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids, img_paths, country_ids
