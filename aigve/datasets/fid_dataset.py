# Copyright (c) IFM Lab. All rights reserved.

import os, sys
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from functools import lru_cache
from core.registry import DATASETS

# Lazy import to avoid circular import
@lru_cache(maxsize=1)
def lazy_import():
    from metrics.video_quality_assessment.nn_based.lightvqa_plus.Light_VQA_plus.extract_temporal_features import slowfast, pack_pathway_output
    return slowfast, pack_pathway_output


@DATASETS.register_module()
class FidDataset(Dataset):
    """
    Dataset for Fréchet Inception Distance (FID) evaluation.

    For each sample, this dataset:
        - Loads both the ground-truth (real) and generated (predicted) videos.
        - Converts each video into a tensor of shape [T, C, H, W] using OpenCV.
        - **Resizes the generated video to match the reference video's resolution.**
        - Optionally pads or truncates videos to a fixed number of frames.

    Args:
        video_dir (str): Directory containing video files.
        prompt_dir (str): Path to JSON file that lists ground-truth and generated video filenames.
        max_len (int): Maximum number of frames to load per video. Default: 500.
        if_pad (bool): Whether to pad videos to exactly `max_len` frames. If False, videos can have variable lengths.
        match_resolution (bool): If True, resize generated video to match reference resolution. Default: True.
    """

    def __init__(self, video_dir, prompt_dir, max_len=500, if_pad=False, match_resolution=True):
        super(FidDataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.max_len = max_len
        self.if_pad = if_pad
        self.match_resolution = match_resolution

        self.gt_video_names, self.gen_video_names = self._read_video_names()

    def _read_video_names(self):
        """Reads video names from the dataset JSON file."""
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)
            gt_video_names = [item['video_path_gt'].strip() for item in read_data["data_list"]]
            gen_video_names = [item['video_path_pd'].strip() for item in read_data["data_list"]]
        return gt_video_names, gen_video_names
    
    def _resize_video_tensor(self, video_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resize a video tensor to target resolution in batches to save memory.
        
        Args:
            video_tensor (torch.Tensor): Video tensor of shape [T, C, H, W].
            target_h (int): Target height.
            target_w (int): Target width.
            
        Returns:
            torch.Tensor: Resized video tensor of shape [T, C, target_h, target_w].
        """
        T, C, H, W = video_tensor.shape
        
        if H == target_h and W == target_w:
            return video_tensor
        
        # Process in batches to save memory (16 frames at a time)
        batch_size = 16
        resized_frames = []
        
        for i in range(0, T, batch_size):
            batch = video_tensor[i:i+batch_size]
            resized_batch = F.interpolate(
                batch, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            resized_frames.append(resized_batch)
        
        # Concatenate and free intermediate tensors
        result = torch.cat(resized_frames, dim=0)
        del resized_frames
        return result
    
    def _load_video_tensor(self, video_path: str, target_size: tuple = None) -> torch.Tensor:
        """
        Load a video and return its tensor of shape [T, C, H, W].
        
        Args:
            video_path: Path to video file.
            target_size: Optional (H, W) tuple to resize frames during loading (more memory efficient).
        """
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        cap = cv2.VideoCapture(video_path)
        input_frames = []
        frame_count = 0
        while cap.isOpened() and frame_count < self.max_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize during loading if target_size specified (more memory efficient)
            if target_size is not None:
                target_h, target_w = target_size
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            input_frames.append(torch.tensor(frame).float())
            frame_count += 1

        cap.release()

        if len(input_frames) == 0:
            raise RuntimeError(f"No valid frames found in {video_path}")

        if self.if_pad:
            num_frames = len(input_frames)
            if num_frames < self.max_len:
                pad_frames = torch.zeros((self.max_len - num_frames, *input_frames[0].shape))
                video_tensor = torch.cat((torch.stack(input_frames), pad_frames), dim=0)
            else:
                video_tensor = torch.stack(input_frames[:self.max_len])
        else:
            video_tensor = torch.stack(input_frames)

        # Convert from [T, H, W, C] to [T, C, H, W]
        return video_tensor.permute(0, 3, 1, 2)
    
    def _get_video_resolution(self, video_path: str) -> tuple:
        """Get the resolution (H, W) of a video without loading all frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return height, width
    
    def __len__(self):
        return len(self.gt_video_names)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor, str, str]: 
                - Ground-truth (Real) video tensor of shape [T, C, H, W].
                - Generated video tensor of shape [T, C, H, W] (resized to match GT if match_resolution=True).
                - Ground-truth video name.
                - Generated video name.
        """
        gt_video_name = self.gt_video_names[index]
        gt_video_path = os.path.join(self.video_dir, gt_video_name)
        gen_video_name = self.gen_video_names[index]
        gen_video_path = os.path.join(self.video_dir, gen_video_name) 

        # Memory-efficient approach: get reference resolution first, then resize generated during loading
        if self.match_resolution:
            # Get reference video resolution without loading full video
            gt_h, gt_w = self._get_video_resolution(gt_video_path)
            gen_h, gen_w = self._get_video_resolution(gen_video_path)
            
            # Load reference video
            gt_video_tensor = self._load_video_tensor(gt_video_path)
            
            # Load generated video, resizing during load if needed (saves memory)
            if gt_h != gen_h or gt_w != gen_w:
                print(f"[FID] Resizing generated video from {gen_h}x{gen_w} to {gt_h}x{gt_w} during load")
                gen_video_tensor = self._load_video_tensor(gen_video_path, target_size=(gt_h, gt_w))
            else:
                gen_video_tensor = self._load_video_tensor(gen_video_path)
        else:
            gt_video_tensor = self._load_video_tensor(gt_video_path)
            gen_video_tensor = self._load_video_tensor(gen_video_path)

        return gt_video_tensor, gen_video_tensor, gt_video_name, gen_video_name
   
