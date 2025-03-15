# encoding = utf-8

import os
import sys
sys.path.append('..')
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Sequence
from transformers import AutoProcessor, BlipForImageTextRetrieval
from core.registry import DATASETS

@DATASETS.register_module()
class BLIPSimDataset(Dataset):
    def __init__(self, processor_name, video_dir, prompt_dir):
        super(BLIPSimDataset, self).__init__()
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
            # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            # preprocess text-image pairs
            input_frames.append(frame)
        input_frame_tensor = self.processor(
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
        #     input_prompt_frame_pair = self.processor(
        #         images=resized_frame, #channle, hight, width
        #         text=prompt,
        #         padding=True,
        #         truncation=True,
        #         max_length=77,
        #         return_tensors="pt",
        #     )
        #     input_prompt_frame_pairs.append(input_prompt_frame_pair)

        # return input_prompt_frame_pairs

DATASETS.register_module(module=BLIPSimDataset, force=True)

def process(blip_model, data_samples: Sequence) -> None:
    """BLIPSimScore process
    Process one batch of data samples and predictions. The processed
    results should be stored in ``self.results``, which will be used to
    compute the metrics when all batches have been processed.

    Args:
        data_batch (Sequence): A batch of data from the dataloader.
        data_samples (Sequence): A batch of data samples that
            contain annotations and predictions.
    """

    result = dict()

    input_prompt_frame_pairs = data_samples  

    # Initialize an empty tensor to store the concatenated features
    blip_scores = []
    with torch.no_grad():
        for input_prompt_frame_pair in input_prompt_frame_pairs:

            # If frame is a tuple, extract the tensor. Assume tensor is the first element.
            if isinstance(input_prompt_frame_pair, tuple):
                input_prompt_frame_pair = input_prompt_frame_pair[0]

            input_prompt_frame_pair = input_prompt_frame_pair.to("cuda")  # Add batch dimension and move the frame to the device
            blip_cosine_sim_score = blip_model(**input_prompt_frame_pair, use_itm_head=False)[0].item()
            print('blip_score', blip_cosine_sim_score)
            blip_scores.append(blip_cosine_sim_score)

    # Calculate the average BLIP score across all frames
    blip_score_frames_avg = sum(blip_scores)/len(blip_scores)
    
    result['blip_sim_score'] = blip_score_frames_avg

    return result



# if __name__ == '__main__':
#     prompt_dir = 'AIGVE_Tool/data/toy/annotations/evaluate.json'
#     video_dir = 'AIGVE_Tool/data/toy/evaluate/'
#     processor_name = 'Salesforce/blip-itm-base-coco'
#     blip_model_name = 'Salesforce/blip-itm-base-coco'

#     blip_model = BlipForImageTextRetrieval.from_pretrained(blip_model_name).to("cuda")
#     blip_dataset = BLIPSimDataset(processor_name=processor_name,
#                                   video_dir=video_dir, prompt_dir=prompt_dir)
#     blip_dataloader = DataLoader(blip_dataset, batch_size=4, shuffle=False)
    
#     for index, data in enumerate(blip_dataloader):
#         result = process(blip_model=blip_model, 
#                         data_samples=data)

  
