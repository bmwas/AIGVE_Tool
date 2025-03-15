# encoding = utf-8

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(LAST_SCRIPT_DIR))

os.environ["OPENAI_API_KEY"] = ''

import cv2
import json
from torch.utils.data import Dataset
from core.registry import DATASETS

@DATASETS.register_module()
class VIEDataset(Dataset):
    def __init__(self, video_dir, prompt_dir):
        super(VIEDataset, self).__init__()
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir

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
        video_sub_dir = video_path.split('.')[0]
        if not os.path.exists(video_sub_dir):
            os.mkdir(video_sub_dir)
        input_frames_path = []
        cap = cv2.VideoCapture(video_path)
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path_name = video_path.split('.')[0] + '/' + str(cnt) + '.jpg'
            cv2.imwrite(frame_path_name, frame)
            # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            # frames.append(resized_frame)
            input_frames_path.append(frame_path_name)
            cnt += 1
    
        return prompt, input_frames_path

    
    
