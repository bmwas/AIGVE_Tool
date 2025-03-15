# encoding = utf-8

import os
import cv2
import json
import torch
from transformers import AutoProcessor
from torch.utils.data import Dataset
from core.registry import DATASETS

@DATASETS.register_module()
class CLIPTempDataset(Dataset):
    def __init__(self, processor_name, prompt_dir, video_dir):
        super(CLIPTempDataset, self).__init__()
        self.prompt_dir = prompt_dir
        self.video_dir = video_dir
        self.processor_name = processor_name

        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        self.video_names = self._read_videoname()

    def _read_videoname(self):
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)
        
        video_name_list = []
        for item in read_data["datset_list"]:
            video_name = item['video_path_pd'].strip()
            video_name_list.append(video_name)

        return video_name_list
    
    def __len__(self):
        return len(self.video_names)-1
    
    def __getitem__(self, index):
        '''return video frame pairs
        '''
        video_name = self.video_names[index]
        video_path = self.video_dir + video_name
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            frames.append(resized_frame)

        input_frame_tensor = self.processor(
            images=frames,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )['pixel_values']

        return input_frame_tensor


# DATASETS.register_module(module=CLIPTempDataset, force=True)
# if __name__ == '__main__':
#     video_dir = 'AIGVE_Tool/data/toy/evaluate/'
#     clip_dataset = CLIPTempDataset(video_dir=video_dir)
    
#     for index, data in enumerate(clip_dataset):
#         print(index, data)

    


