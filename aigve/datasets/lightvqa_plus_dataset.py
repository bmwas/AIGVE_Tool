# Copyright (c) IFM Lab. All rights reserved.

import os, sys
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import clip
import torch.nn as nn
from functools import lru_cache
from core.registry import DATASETS
from utils import add_git_submodule, submodule_exists
# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)

# Lazy import to avoid circular import
@lru_cache(maxsize=1)
def lazy_import():
    from metrics.video_quality_assessment.nn_based.lightvqa_plus.Light_VQA_plus.extract_temporal_features import slowfast, pack_pathway_output
    return slowfast, pack_pathway_output


@DATASETS.register_module()
class LightVQAPlusDataset(Dataset):
    """
    Dataset for LightVQA+.
    Extracts:
        - spatial_features (torch.Tensor): Extracted key frames.
        - temporal_features (torch.Tensor): SlowFast motion features.
        - BNS_features (torch.Tensor): Brightness & Noise features.
        - BC_features (torch.Tensor): Temporal CLIP-based brightness contrast features.
        - video_name (str): Video filename.
    """

    def __init__(self, video_dir, prompt_dir, min_video_seconds=8):
        super(LightVQAPlusDataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.min_video_seconds = min_video_seconds

        self.video_names = self._read_video_names()

        # Load CLIP model for BNS and BC features
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.preprocess = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        self.to_tensor = transforms.ToTensor()

        # CLIP text prompts
        self.text_B = clip.tokenize([  # brightness (B)
            "an underexposed photo", "a slightly underexposed photo",
            "a well-exposed photo", "a slightly overexposed photo", "an overexposed photo"
        ])

        self.text_N = clip.tokenize([ # noise (N)
            "a photo with no noise", "a photo with little noise",
            "a photo with considerable noise", "a photo with serious noise", "a photo with extreme noise"
        ])

        self.submodel_path = os.path.join(os.getcwd(), 'metrics/video_quality_assessment/nn_based/lightvqa_plus')
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/SaMMyCHoo/Light-VQA-plus.git', 
                submodule_path=self.submodel_path
            )
        # original_path = os.path.join(self.submodel_path, "Light-VQA-plus")
        lightvqa_path = os.path.join(self.submodel_path, "Light_VQA_plus")
        # if os.path.exists(original_path) and not os.path.exists(lightvqa_path):
        #     os.rename(original_path, lightvqa_path)
        if lightvqa_path not in sys.path:
            sys.path.insert(0, lightvqa_path)
        # print(sys.path)

        # Load SlowFast model
        slowfast, _ = lazy_import()
        self.slowfast_model = slowfast()

    def _read_video_names(self):
        """Reads video names from the dataset JSON file."""
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)
        return [item['video_path_pd'].strip() for item in read_data["data_list"]]
    
    def __len__(self):
        return len(self.video_names)

    def extract_key_frames(self, video_path):
        """
        Extracts 8 evenly spaced key frames across the entire video duration.
        
        Args:
            video_path (str): Path to the video file.

        Returns:
            spatial_features (torch.Tensor): Shape [8, 3, 672, 1120] containing 8 key frames.
        """
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).split('.')[0]

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_length >= 8:
            # Select 8 unique frame indices evenly spaced across the entire video
            frame_indices = np.round(np.linspace(0, video_length - 1, num=8)).astype(int)
        else:
            # Select all available frames and repeat the last one to reach 8
            frame_indices = list(range(video_length)) + [video_length - 1] * (8 - video_length)

        spatial_features = torch.zeros([8, 3, 672, 1120])  # Ensure exactly 8 frames
        transform = transforms.Compose([
            transforms.Resize([672, 1120]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        last_valid_frame = None
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                spatial_features[idx] = transform(frame)
                last_valid_frame = spatial_features[idx]
            elif last_valid_frame is not None:  # If total frames are less than 8, repeat the last valid frame
                spatial_features[idx] = last_valid_frame

        cap.release()
        # print('spatial_features: ', spatial_features.shape) # torch.Size([8, 3, 672, 1120])
        return spatial_features

    def get_global_sf(self, video_path) -> torch.Tensor:
        """Extracts global brightness & noise features across full video.

        Args:
            video_path (str): Path to video file.

        Returns:
            torch.Tensor: Extracted global features (Shape: [8, 150]).
        """
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print('video_length: ', video_length)  # 16

        frames = []
        for _ in range(video_length):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1120, 672))
                frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"Failed to extract frames from {video_path}")

        res = []
        length = len(frames)
        now = 0
        interval = 10  # Process 10 frames at a time
        while now + interval - 1 < length:
            final = [self.to_tensor(Image.fromarray(cv2.cvtColor(frames[i + now], cv2.COLOR_BGR2RGB)))
                    for i in range(interval)]
            
            # Step 1: Convert to tensor batch
            images = torch.stack(final, dim=0)  # Shape: [10, 3, 672, 1120]

            # Step 2: Unfold into patches (Strictly following GET_SF)
            images = images.unfold(2, 224, 224).unfold(3, 224, 224)  # Shape: [10, 3, 3, 5, 224, 224]
            images = images.permute(0, 3, 2, 1, 4, 5).contiguous()  # Shape: [10, 5, 3, 3, 224, 224]
            images = images.reshape(-1, 15, 3, 224, 224)  # Shape: [10, 15, 3, 224, 224]
            images = images.view(-1, 3, 224, 224)  # Shape: [150, 3, 224, 224]
            images = self.preprocess(images)  # Normalize for CLIP
            # print('images get_global_sf: ', images.shape) # torch.Size([10*15, 3, 224, 224])
            
            # Step 3: Extract features using CLIP
            with torch.no_grad():
                logits_N, _ = self.clip_model(images, self.text_N)
                logits_B, _ = self.clip_model(images, self.text_B)

            tmp_N = logits_N.softmax(dim=-1).view(interval, -1) * 10
            tmp_B = logits_B.softmax(dim=-1).view(interval, -1) * 10
            # print('tmp_N get_global_sf', tmp_N.shape) # torch.Size([10, 75])
            # print('tmp_B get_global_sf', tmp_B.shape) # torch.Size([10, 75])
            res.append(torch.cat([tmp_N, tmp_B], dim=1))
            now += interval

        # Handle remaining frames
        if length > now:
            final = [self.to_tensor(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)))
                    for i in range(now, length)]
            
            images = torch.stack(final, dim=0)  # Shape: [remaining(6), 3, 672, 1120]
            images = images.unfold(2, 224, 224).unfold(3, 224, 224)  # Shape: [remaining, 3, 3, 5, 224, 224]
            images = images.permute(0, 3, 2, 1, 4, 5).contiguous()  # Shape: [remaining, 5, 3, 3, 224, 224]
            images = images.reshape(-1, 15, 3, 224, 224)  # Shape: [remaining, 15, 3, 224, 224]
            images = images.view(-1, 3, 224, 224)  # Shape: [remaining*15, 3, 224, 224]
            images = self.preprocess(images)

            with torch.no_grad():
                logits_N, _ = self.clip_model(images, self.text_N) # Shape: [remaining, 5(num_text_prompts)]
                logits_B, _ = self.clip_model(images, self.text_B) # Shape: [remaining, 5]
                # print('logits_N last get_global_sf', logits_N.shape) # torch.Size([6*15, 5])
                # print('logits_B last get_global_sf', logits_B.shape) #torch.Size([6*15, 5])

            tmp_N = logits_N.softmax(dim=-1).view(length - now, -1) * 10 # Shape: [remaining, 75]
            tmp_B = logits_B.softmax(dim=-1).view(length - now, -1) * 10 # Shape: [remaining, 75]
            # print('tmp_N last get_global_sf', tmp_N.shape)  # torch.Size([6, 75])
            # print('tmp_B last get_global_sf', tmp_B.shape)  # torch.Size([6, 75])

            res.append(torch.cat([tmp_N, tmp_B], dim=1))

        res = torch.cat(res, dim=0)  # Shape: [length, 150]
        # print('res: ', res.shape)  # torch.Size([16, 150]) for toy dataset

        # Step 4: Aggregate into 8 time slots
        chunk_size = length // 8
        final_res = [
            torch.mean(res[i * chunk_size: (i + 1) * chunk_size], dim=0) if i < 7 else torch.mean(res[7 * chunk_size:], dim=0)
            for i in range(8)
        ]

        return torch.stack(final_res, dim=0)  # Shape: [8, 150]

    def extract_bns_features(self, video_path):
        """Extracts Brightness & Noise Sensitivity (BNS) features using CLIP.
        Local Feature Extraction (res1) → Uses 8 key frames
        Global Feature Extraction (res2) → Uses all frames

        Args:
            video_path (str): Path to the video file.

        Returns:
            spatial_features (torch.Tensor): Extracted 8 evenly spaced key frames across the entire video duration.
                Shape [8, 3, 672, 1120] containing 8 key frames.
            final_res (torch.Tensor): Extracted BNS feature (Shape: [8, 300]).
        """
        # Local Feature Extraction Step 1: Extract key frames
        spatial_features = self.extract_key_frames(video_path) # Shape: [8, 3, 672, 1120]

        # Step 2: Apply unfolding transformation (Strictly following GET_S_F)
        images = spatial_features.unfold(2, 224, 224).unfold(3, 224, 224)  # Break into patches. Shape: [8, 3, 3, 5, 224, 224]
        images = images.permute(0, 3, 2, 1, 4, 5).contiguous()  # Shape: [8, 5, 3, 3, 224, 224]
        images = images.reshape(-1, 15, 3, 224, 224)  # Shape: [8, 15, 3, 224, 224]
        images = images.view(-1, 3, 224, 224)  # Shape: [120, 3, 224, 224]
        images = self.preprocess(images)  # Normalize for CLIP
        # print('images: ', images.shape) # torch.Size([120, 3, 224, 224])
        # print(images.device)
        # print(self.text_N.device)

        # Step 3: Pass through CLIP
        with torch.no_grad():
            logits_N, _ = self.clip_model(images, self.text_N)
            logits_B, _ = self.clip_model(images, self.text_B)

        res_N = logits_N.softmax(dim=-1).view(8, -1) * 10
        # print('res_N: ', res_N.shape) # torch.Size([8, 75])
        res_B = logits_B.softmax(dim=-1).view(8, -1) * 10
        # print('res_B: ', res_N.shape) # torch.Size([8, 75])
        res1 = torch.cat((res_N, res_B), dim=1)
        # print('res1: ', res1.shape) # torch.Size([8, 150])

        # Global Feature Extraction (GET_SF Equivalent)
        res2 = self.get_global_sf(video_path)
        # print('res2: ', res2.shape) # res2:  torch.Size([8, 150])

        # Split & Combine Features
        Nl, Bl = torch.split(res1, 75, dim=1)
        Ng, Bg = torch.split(res2, 75, dim=1)
        final_res = torch.cat([Nl, Ng, Bl, Bg], dim=1)
        # print('final_res: ', final_res.shape)

        return spatial_features, final_res  # Shape: [8, 300]

    def extract_bc_features(self, video_path) -> torch.Tensor:
        """
        Extracts Brightness Consistency features using CLIP-based temporal processing.
        
        Returns:
            torch.Tensor: Extracted BC feature (Shape: [8, final_dim]).
        """

        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for _ in range(video_length):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1120, 672))
                frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"Failed to extract frames from {video_path}")

        res = []
        now = 0
        interval = 10  # Process 10 frames at a time
        length = len(frames)

        # Step 1: Extract CLIP Features at Fixed Intervals
        while now + interval - 1 < length:
            batch = [self.to_tensor(Image.fromarray(cv2.cvtColor(frames[i + now], cv2.COLOR_BGR2RGB)))
                    for i in range(interval)]
            images = torch.stack(batch, dim=0)
            images = images.unfold(2, 224, 224).unfold(3, 224, 224)  # Shape: [10, 3, 3, 5, 224, 224]
            images = images.permute(0, 3, 2, 1, 4, 5).contiguous()  # Shape: [10, 5, 3, 3, 224, 224]
            images = images.reshape(-1, 15, 3, 224, 224)  # Shape: [10, 15, 3, 224, 224]
            images = images.view(-1, 3, 224, 224)  # Shape: [10*15, 3, 224, 224]
            images = self.preprocess(images)
            # print('images extract_bc_features', images.shape) # torch.Size([150, 3, 224, 224])

            with torch.no_grad():
                logits, _ = self.clip_model(images, self.text_B)

            tmp = logits.softmax(dim=-1) * 10
            res.append(tmp)
            now += interval

        # Handle Remaining Frames
        if length > now:
            batch = [self.to_tensor(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)))
                    for i in range(now, length)]
            images = torch.stack(batch, dim=0)
            images = images.unfold(2, 224, 224).unfold(3, 224, 224)  # Shape: [remaining(6), 3, 3, 5, 224, 224]
            images = images.permute(0, 3, 2, 1, 4, 5).contiguous()  # Shape: [remaining, 5, 3, 3, 224, 224]
            images = images.reshape(-1, 15, 3, 224, 224)  # Shape: [remaining, 15, 3, 224, 224]
            images = images.view(-1, 3, 224, 224)  # Shape: [remaining, 15, 3, 224, 224]
            images = self.preprocess(images)
            # print('images: ', images.shape) #  torch.Size([6*15, 3, 224, 224])

            with torch.no_grad():
                logits, _ = self.clip_model(images, self.text_B)

            tmp = logits.softmax(dim=-1) * 10
            res.append(tmp)

        res = torch.cat(res, dim=0)  # Shape: [length, 5]
        # print('res extract_bc_features: ', res.shape) # torch.Size([150+90, 5])

        # Step 2: Multi-Scale Variance Computation: downsample frames steps
        # smaller step: Captures fast, fine-grained changes.
        # larger step:  Captures slow, long-term trends.
        final_res = []
        for step in [1, 2, 4, 8]:  # Multi-scale temporal steps
            chunk_number = 8 // step
            chunk_size = length // chunk_number
            chunks = []
            for i in range(chunk_number):
                if i < chunk_number - 1:
                    chunk = res[i * chunk_size : (i + 1) * chunk_size, :]
                else:
                    chunk = res[(chunk_number - 1) * chunk_size:, :]  # Handle remaining frames
                tmp = []
                for j in range(step):
                    temp = chunk[j::step, :]  
                    tmp.append(torch.var(temp.float(), dim=0))  # Variance computation
                chunks.append(tmp)  # final chunks len: 8; 4; 2; 1 
            final_res.append(chunks) # final final_res len: 4

        # Step 3: Aggregate Multi-Scale Features
        temp = []
        for i in range(8):  # Aggregate temporal information across 8 time slots
            temp.append(torch.cat(final_res[0][i]                                                # variance for step size = 1
                                + [torch.mean(torch.stack(final_res[1][i // 2], dim=0), dim=0)]  # for step size = 2
                                + [torch.mean(torch.stack(final_res[2][i // 4], dim=0), dim=0)]  # Every 4 slots share the same value.
                                + [torch.mean(torch.stack(final_res[3][i // 8], dim=0), dim=0)]  # for step size = 8
                                , dim=0))

        final_res = torch.stack(temp, dim=0)  # Shape: [8, final_dim]  
        # print('final_res extract_bc_featuresx: ', final_res.shape) # torch.Size([8, 20])
        
        return final_res

    def extract_temporal_features(self, video_path) -> torch.Tensor:
        """Extracts SlowFast motion features on the entire video segment.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: Extracted motion features (Shape: [1, feature_dim(2304)]).
        """
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.round(np.linspace(0, video_length - 1, num=8)).astype(int)

        transform = transforms.Compose([
            transforms.Resize([224, 224]),  # Match SlowFast input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # Original normalization
        ])

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(transform(frame))  # Resize & normalize
        cap.release()

        if len(frames) < 8:
            raise ValueError(f"Insufficient frames in {video_path}, expected 8.")

        video_tensor = torch.stack(frames, dim=0)  # Shape: [8, 3, 224, 224]
        
        # Prepare for SlowFast input
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: [1, 8, 3, 224, 224]
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # Shape: [1, 3, 8, 224, 224]

        # Pack pathways for SlowFast model
        _, pack_pathway_output = lazy_import()
        inputs = pack_pathway_output(video_tensor, device='cpu')
        # print('inputs len: ', len(inputs))
        # print('inputs[0]: ', inputs[0].shape) # torch.Size([1, 3, 2, 224, 224])
        # print('inputs[1]: ', inputs[1].shape) # torch.Size([1, 3, 8, 224, 224])

        # Extract features using SlowFast
        with torch.no_grad():
            slow_feature, fast_feature = self.slowfast_model(inputs)

        # print('slow_feature extract_temporal_features: ', slow_feature.shape) # torch.Size([1, 2048, 1, 1, 1])
        # print('fast_feature extract_temporal_features: ', fast_feature.shape) # torch.Size([1, 256, 1, 1, 1])

        # Concatenate slow and fast features
        features = torch.cat([slow_feature, fast_feature], dim=1).squeeze(-1).squeeze(-1).squeeze(-1)
        # print('features extract_temporal_features: ', features.shape) # torch.Size([1, 2304])

        return features

    def __getitem__(self, index):
        """
        Returns:
            spatial_features (torch.Tensor): Spatial features. Shape: [8, 3, 672, 1120].
            bns_features (torch.Tensor): Brightness & Noise features. Shape: [8, 300].
            (bc_features (torch.Tensor): Temporal brightness contrast features. Shape: [8, final_dim].)
            temporal_features (torch.Tensor): SlowFast motion features. Shape: [1, feature_dim(2304)]
            video_name (str): Video filename.
        """
        video_name = self.video_names[index]
        video_path = os.path.join(self.video_dir, video_name)

        spatial_features, bns_features = self.extract_bns_features(video_path)
        bc_features = self.extract_bc_features(video_path)
        temporal_features = self.extract_temporal_features(video_path)

        return spatial_features, temporal_features, bns_features, bc_features, video_name

