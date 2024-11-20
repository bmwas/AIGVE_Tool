# encoding = utf-8

import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from core.registry import DATASETS

@DATASETS.register_module()
class CLIPSimDataset(Dataset):
    def __init__(self, tokenizer_name, video_dir, prompt_dir):
        super(CLIPSimDataset, self).__init__()
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            frames.append(resized_frame)

        # Convert numpy arrays to tensors, change dtype to float, and resize frames
        tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]

        # Tokenize the prompt
        text_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)

        # Convert the tokenized text to a tensor and move it to the device
        prompt_input = text_tokens["input_ids"]

        return prompt_input, tensor_frames

DATASETS.register_module(module=CLIPSimDataset, force=True)

if __name__ == '__main__':
    tokenizer_name = 'openai/clip-vit-base-patch32'
    prompt_dir = '/storage/drive_1/zizhong/vqa_toolkit/VQA_Toolkit-main/data/toy/annotations/evaluate.json'
    video_dir = '/storage/drive_1/zizhong/vqa_toolkit/VQA_Toolkit-main/data/toy/evaluate/'

    clip_dataset = CLIPSimDataset(tokenizer_name=tokenizer_name,
                               video_dir=video_dir,
                               prompt_dir=prompt_dir)
    
    for index, data in enumerate(clip_dataset):
        print(index, data)

    


