# Copyright (c) IFM Lab. All rights reserved.

import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from core.registry import DATASETS
# import math

@DATASETS.register_module()
class SimpleVQADataset(Dataset):
    """
    Dataset for SimpleVQA.
    Each sample returns:
        - spatial_features (torch.Tensor): Extracted spatial frames.
        - motion_features (torch.Tensor): Extracted motion-based clips.
        - video_name (str): Video filename.
    """

    def __init__(self, video_dir, prompt_dir, min_video_seconds=8):
        super(SimpleVQADataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.min_video_seconds = min_video_seconds

        self.prompts, self.video_names = self._read_prompt_videoname()
    
    def _read_prompt_videoname(self):
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)

        prompt_data_list, video_name_list = [], []
        for item in read_data["data_list"]:
            prompt = item['prompt_gt'].strip()
            video_name = item['video_path_pd'].strip()
            prompt_data_list.append(prompt)
            video_name_list.append(video_name)

        return prompt_data_list, video_name_list
    
    def __len__(self):
        return len(self.prompts)
    
    def video_processing_spatial(self, video_path):
        """
        Extracts spatial frames with proper resizing and normalization.
            - Key frame extraction: It selects 1 frame per second.
            - Standard input size: It resizes frames to 448 * 448 (after an initial resize to 520).
        Return:
            transformed_video (torch.Tensor): shape[video_length_read, 3, 448, 448]. 
                `video_length_read` is total seconds of the video (though 2 for toy dataset) with minium 8 (i.e. min_video_seconds).
            video_name (str)
        """
        video_capture = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))

        # Compute the number of total seconds of the video
        video_length_read = int(video_length/video_frame_rate) # math.ceil()
        # print('video_length_read (s): ', video_length_read)
        transformations = transforms.Compose([
            transforms.Resize(520),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
        ])
        transformed_video = torch.zeros([max(video_length_read, self.min_video_seconds), 3, 448, 448])

        video_read_index = 0
        frame_idx = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # Key frames extraction
                if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):
                    read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    read_frame = transformations(read_frame)
                    transformed_video[video_read_index] = read_frame
                    video_read_index += 1
                frame_idx += 1

        # Pads remaining frames by repeating the last available frame.
        if video_read_index < self.min_video_seconds:
            for i in range(video_read_index, self.min_video_seconds):
                transformed_video[i] = transformed_video[video_read_index - 1]

        video_capture.release()
        return transformed_video, video_name

    def video_processing_motion(self, video_path):
        """
        Extracts motion-based clips suitable for SlowFast.
            - Standard input size: It resizes frames to 224 * 224.
            - Motion-based clips: Processes at leaset 8-second clips, select 32 consecutive frames from each second.
        Return:
            transformed_video_all (List[torch.Tensor]): Tensor shape[video_length_clip(32), 3, 224, 224]. 
                Len(List) is total seconds of the video, with minium 8.
            video_name (str)
        """
        video_capture = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))

        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]) # General purpose
        ])
        transformed_frame_all = torch.zeros([video_length, 3, 224, 224])
        video_read_index = 0
        for i in range(video_length): # All frames extraction
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        # Pads remaining frames by repeating the last available frame.
        if video_read_index < video_length: 
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
        
        video_capture.release()

        # Compute the number of total seconds of the video
        video_clip = int(video_length/video_frame_rate)
        # print('video_clip (s): ', video_clip)
        video_length_clip = 32
        transformed_video_all = []

        # Extract motion-based clips: select 32 consecutive frames from each second
        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, 3, 224, 224])
            if (i * video_frame_rate + video_length_clip) <= video_length: # if the clip can be fully extracted, select 32 consecutive frames starting at i*video_frame_rate
                transformed_video = transformed_frame_all[i * video_frame_rate:(i * video_frame_rate + video_length_clip)]
            else: # Copy all rest available frames. Pads remaining frames by repeating the last available frame.
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < self.min_video_seconds:
            for i in range(video_clip, self.min_video_seconds):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
        
        return transformed_video_all, video_name

    def __getitem__(self, index):
        """
        Returns:
            spatial_features (torch.Tensor): Shape [v_len_second, 3, 448, 448]
                `v_len_second` is total seconds of the video (though 2 for toy dataset) with minium 8 (i.e. min_video_seconds).
            motion_features (List[torch.Tensor]): List of motion feature tensors.
                Each tensor has shape [32, 3, 224, 224].
                Len(List) is total seconds of the video (i.e. v_len_second), with minium 8 (i.e. min_video_seconds).
            video_name (str): Video filename
        """
        video_name = self.video_names[index]
        video_path = os.path.join(self.video_dir, video_name)

        spatial_features, video_name = self.video_processing_spatial(video_path)
        motion_features, video_name = self.video_processing_motion(video_path)
        # print('spatial_features: ', spatial_features.shape) # torch.Size([8, 3, 448, 448]) for toy dataset
        # print('motion_features len: ', len(motion_features)) # 8
        # print('motion_features[0]: ', motion_features[0].shape) # torch.Size([32, 3, 224, 224])

        return spatial_features, motion_features, video_name
