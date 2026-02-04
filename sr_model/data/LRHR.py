import os
from glob import glob
from glog import logger
from torch.utils.data import Dataset
from data import aug
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import re
import json
import random

class LRHRDataset(Dataset):
    def __init__(self, opt, phase, test_split=0.2, random_seed=42):
        """
        PlateDataset with baseline-compatible train/val split - 3 LR inputs
        
        Args:
            opt: Configuration dictionary with keys:
                - dataroot: Root directory containing Scenario-A, Scenario-B folders
                - height: Target height for images
                - width: Target width for images
                - val_split_file: Path to val_tracks.json (optional)
                - test_split_file: Path to test_tracks.json (optional)
                - val_size: Number of validation tracks (if files don't exist)
                - test_size: Number of test tracks (if files don't exist)
            phase: 'train' or 'val' or 'test'
            test_split: Fallback fraction for testing (only used if files don't exist and val_size not set)
            random_seed: Random seed for consistent splits (default: 42)
        """
        super(LRHRDataset, self).__init__()
        self.opt = opt
        height = opt['height']
        width = opt['width']
        self.phase = phase
        self.test_split = test_split
        self.random_seed = random_seed

        self.dataroot = opt['dataroot']
        
        # Get split file paths from config or use defaults
        self.val_split_file = opt.get('val_split_file', 'val_tracks.json')
        self.test_split_file = opt.get('test_split_file', 'test_tracks.json')
        
        # Get split sizes from config or calculate from test_split
        self.val_size = opt.get('val_size', None)
        self.test_size = opt.get('test_size', 0)  # Default: no test set for SR
        
        # Collect all tracks first
        all_tracks = []
        
        # Navigate through: Scenario-A or B/Brazilian or Mercosur/track_*/
        for scenario in ['Scenario-A', 'Scenario-B']:
            scenario_path = os.path.join(self.dataroot, scenario)
            if not os.path.exists(scenario_path):
                continue
            
            # Iterate through plate types
            for plate_type in ['Brazilian', 'Mercosur']:
                plate_path = os.path.join(scenario_path, plate_type)
                if not os.path.exists(plate_path):
                    continue
                
                # Get all track folders
                track_folders = sorted([f for f in os.listdir(plate_path) 
                                      if f.startswith('track_') and os.path.isdir(os.path.join(plate_path, f))])
                
                for track_folder in track_folders:
                    track_path = os.path.join(plate_path, track_folder)
                    
                    # Get LR and HR files
                    lr_files = sorted(glob(os.path.join(track_path, 'lr-*.png')) + 
                                    glob(os.path.join(track_path, 'lr-*.jpg')), 
                                    key=self.extract_number)
                    hr_files = sorted(glob(os.path.join(track_path, 'hr-*.png')) +
                                    glob(os.path.join(track_path, 'hr-*.jpg')), 
                                    key=self.extract_number)
                    
                    if len(lr_files) > 0 and len(hr_files) > 0:
                        # Store track info with its full path for unique identification
                        track_info = {
                            'track_path': track_path,
                            'track_name': track_folder,
                            'lr_files': lr_files,
                            'hr_files': hr_files,
                            'scenario': scenario,
                            'plate_type': plate_type
                        }
                        all_tracks.append(track_info)
        
        # Split tracks using baseline-compatible logic
        train_tracks, val_tracks, test_tracks = self._split_tracks(all_tracks)
        
        # Select tracks based on phase
        if phase == 'train':
            selected_tracks = train_tracks
        elif phase == 'val':
            selected_tracks = val_tracks
        elif phase == 'test':
            selected_tracks = test_tracks
        else:
            raise ValueError(f"Phase must be 'train', 'val', or 'test', got '{phase}'")
        
        # Build lr and hr lists from selected tracks
        self.lr = []
        self.hr = []
        
        for track_info in selected_tracks:
            self.lr.append(track_info['lr_files'])
            self.hr.append(track_info['hr_files'])
            # lr_files = track_info['lr_files']
            # hr_files = track_info['hr_files']
            
            # # Create one training sample for each HR image in the track
            # for hr_file in hr_files:
            #     self.lr.append(lr_files)  # All LR frames for this track
            #     self.hr.append(hr_file)   # One specific HR target
        
        assert len(self.lr) == len(self.hr), f"Mismatch: {len(self.lr)} LR tracks vs {len(self.hr)} HR images"

        # Transforms - same as original
        self.transform_fn1 = aug.get_transforms(size=(height, width))
        self.transform_fn2 = aug.get_transforms(size=(height, width))
        self.transform_fn3 = aug.get_transforms(size=(height, width))
        self.transform_fn = aug.get_transforms(size=(height, width))

        self.normalize_fn = aug.get_normalize()
        
        logger.info(f'{phase.upper()} dataset created with {len(selected_tracks)} tracks and {len(self.lr)} samples')
    
    def _split_tracks(self, all_tracks):
        """
        Split tracks using baseline OCR logic:
        1. Try to load from JSON files (val_tracks.json, test_tracks.json)
        2. If files exist, use them
        3. If not, create split using random shuffle with seed
        
        This ensures SR training uses EXACT same split as baseline OCR
        """
        total_tracks = len(all_tracks)
        
        # Try to load existing split files
        val_exists = os.path.exists(self.val_split_file)
        test_exists = os.path.exists(self.test_split_file)
        
        val_ids, test_ids = set(), set()
        
        if val_exists and test_exists:
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
                    logger.info(f"Loaded {len(val_ids)} validation track IDs from {self.val_split_file}")
                
                with open(self.test_split_file, 'r') as f:
                    test_ids = set(json.load(f))
                    logger.info(f"Loaded {len(test_ids)} test track IDs from {self.test_split_file}")
            except Exception as e:
                logger.warning(f"Error loading split files: {e}. Will create new split.")
                val_ids, test_ids = set(), set()
        
        # Assign tracks based on loaded IDs
        train_tracks, val_tracks, test_tracks = [], [], []
        
        for track_info in all_tracks:
            track_name = track_info['track_name']  # e.g., "track_02876"
            
            if track_name in val_ids:
                val_tracks.append(track_info)
            elif track_name in test_ids:
                test_tracks.append(track_info)
            else:
                train_tracks.append(track_info)
        
        # If split files didn't exist or were invalid, create new split
        if not val_ids or not test_ids:
            logger.info("Creating new train/val/test split...")
            
            # Use val_size from config, or calculate from test_split
            if self.val_size is None:
                val_size = int(total_tracks * self.test_split)
            else:
                val_size = self.val_size
            
            test_size = self.test_size
            
            # Ensure we don't exceed total tracks
            if val_size + test_size >= total_tracks:
                logger.warning(f"val_size ({val_size}) + test_size ({test_size}) >= total_tracks ({total_tracks})")
                logger.warning(f"Adjusting split sizes...")
                val_size = max(1, int(total_tracks * 0.2))
                test_size = 0
            
            # Shuffle with seed
            random.Random(self.random_seed).shuffle(all_tracks)
            
            # Split
            val_tracks = all_tracks[:val_size]
            test_tracks = all_tracks[val_size:val_size + test_size]
            train_tracks = all_tracks[val_size + test_size:]
            
            # Save split to JSON files for future consistency
            val_track_names = [t['track_name'] for t in val_tracks]
            test_track_names = [t['track_name'] for t in test_tracks]
            
            try:
                with open(self.val_split_file, 'w') as f:
                    json.dump(val_track_names, f, indent=2)
                logger.info(f"Saved validation split to {self.val_split_file}")
                
                if test_size > 0:
                    with open(self.test_split_file, 'w') as f:
                        json.dump(test_track_names, f, indent=2)
                    logger.info(f"Saved test split to {self.test_split_file}")
            except Exception as e:
                logger.warning(f"Could not save split files: {e}")
        
        # Log split statistics
        logger.info(f"Total tracks: {total_tracks}")
        logger.info(f"Train tracks: {len(train_tracks)} ({len(train_tracks)/total_tracks*100:.1f}%)")
        logger.info(f"Val tracks: {len(val_tracks)} ({len(val_tracks)/total_tracks*100:.1f}%)")
        if len(test_tracks) > 0:
            logger.info(f"Test tracks: {len(test_tracks)} ({len(test_tracks)/total_tracks*100:.1f}%)")
        
        return train_tracks, val_tracks, test_tracks
        
    def extract_number(self, file_path):
        match = re.search(r'[lr|hr]-(\d+)\.(png|jpg)', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        else:
            logger.warning(f'Sort Error at: {file_path}')
            return -1

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        assert len(self.lr[idx]) != 0, f'Not enough LR images for index {idx}: {self.lr[idx]} found, expected at least 1.'
        
        # Handle cases with fewer than 3 LR images
        if len(self.lr[idx]) < 3:
            if len(self.lr[idx]) == 1:
                sample_id1, sample_id2, sample_id3 = 0, 0, 0
            elif len(self.lr[idx]) == 2:
                sample_id1, sample_id2, sample_id3 = 0, 1, 1
        else:
            # Use first 3 LR images
            sample_id1, sample_id2, sample_id3 = 0, 1, 2
            
        # Load images
        lr_image_1 = Image.open(self.lr[idx][sample_id1])
        lr_image_2 = Image.open(self.lr[idx][sample_id2])
        lr_image_3 = Image.open(self.lr[idx][sample_id3])
        hr_image = Image.open(self.hr[idx][4])
        
        # Convert to numpy
        lr_image_1 = np.array(lr_image_1)
        lr_image_2 = np.array(lr_image_2)
        lr_image_3 = np.array(lr_image_3)
        hr_image = np.array(hr_image)

        # Apply transforms
        lr_image_1 = self.transform_fn1(lr_image_1)
        lr_image_2 = self.transform_fn2(lr_image_2)
        lr_image_3 = self.transform_fn3(lr_image_3)
        hr_image = self.transform_fn(hr_image)
        
        # Normalize
        lr_image_1 = self.normalize_fn(lr_image_1)
        lr_image_2 = self.normalize_fn(lr_image_2)
        lr_image_3 = self.normalize_fn(lr_image_3)
        hr_image = self.normalize_fn(hr_image)

        # Convert to tensors
        lr_image_1 = transforms.ToTensor()(lr_image_1)
        lr_image_2 = transforms.ToTensor()(lr_image_2)
        lr_image_3 = transforms.ToTensor()(lr_image_3)
        hr_image = transforms.ToTensor()(hr_image)

        return {
            'LR1': lr_image_1, 
            'LR2': lr_image_2, 
            'LR3': lr_image_3, 
            'HR': hr_image, 
            'path': self.hr[idx]
        }
