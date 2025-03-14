# encoding = utf-8

import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, CLIPModel
from typing import Sequence
from core.registry import DATASETS

@DATASETS.register_module()
class CLIPSimDataset(Dataset):
    def __init__(self, processor_name, video_dir, prompt_dir):
        super(CLIPSimDataset, self).__init__()
        self.processor_name = processor_name
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir

        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        self.prompts, self.video_names = self._read_prompt_videoname()

    def _read_prompt_videoname(self):
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)
        
        prompt_data_list, video_name_list = [], []
        for item in read_data["datset_list"]:
            prompt = item['prompt_gt'].strip()
            video_name = item['video_path_pd'].strip()
            prompt_data_list.append(prompt)
            video_name_list.append(video_name)

        return prompt_data_list, video_name_list


    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        prompt, video_name = self.prompts[index], self.video_names[index]
        video_path = self.video_dir + video_name
        input_frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            input_frames.append(resized_frame)
        input_frame_tensor= self.processor(
            images=input_frames,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )['pixel_values']
            
        input_prompt = self.processor(
            text=[prompt],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )['input_ids']

        return input_prompt, input_frame_tensor


DATASETS.register_module(module=CLIPSimDataset, force=True)


# if __name__ == '__main__':
#     processor_name = 'openai/clip-vit-base-patch32'
#     clip_model_name = 'openai/clip-vit-base-patch32'
#     prompt_dir = 'AIGVE_Tool/data/toy/annotations/evaluate.json'
#     video_dir = 'AIGVE_Tool/data/toy/evaluate/'

#     clip_dataset = CLIPSimDataset(processor_name=processor_name,
#                                   video_dir=video_dir,
#                                   prompt_dir=prompt_dir)
#     clip_dataset.__getitem__(0)
#     clip_model = CLIPModel.from_pretrained(clip_model_name).to("cuda")
