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
import torch
import openai
from torch.utils.data import Dataset
from transformers import AutoProcessor
from metrics.text_video_alignment.gpt_based.dsg.DSG.dsg.openai_utils import openai_completion
from metrics.text_video_alignment.gpt_based.TIFA.tifa.tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
from core.registry import DATASETS

@DATASETS.register_module()
class TIFADataset(Dataset):
    def __init__(self, video_dir, prompt_dir):
        super(TIFADataset, self).__init__()
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
    

# def openai_setup(openai_key, llm_model):
#     print('set up openai client')
#     openai.api_key = openai_key
#     assert openai.api_key is not None
#     test_prompt_string = 'hello, how are you doing?'
#     print('test prompt: ', test_prompt_string)
#     response = openai_completion(
#         test_prompt_string,
#         model=llm_model,
#     )
#     print('test response: ', response)

# def process(input_prompt, input_frames, unifiedqa_model, vqa_model):
#     print(input_prompt)
#     for index, frame_path in enumerate(input_frames):
#         if index > 0:
#             break
#         # Generate questions with GPT-3.5-turbo
#         gpt3_questions = get_question_and_answers(input_prompt)
#         # print(gpt3_questions)
            
#         # Filter questions with UnifiedQA
#         filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
            
#         # See the questions
#         print('filtered_questions: ', filtered_questions)

#         # calucluate TIFA score
#         result = tifa_score_single(vqa_model, filtered_questions, frame_path)
#         print(f"TIFA score is {result['tifa_score']}")   # 0.33
#         print(result)
    

# # DATASETS.register_module(module=PickScoreDataset, force=True)

# if __name__ == '__main__':
#     openai_key = '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'
#     prompt_dir = '/home/exouser/VQA_tool/VQA_Toolkit/data/toy/annotations/evaluate.json'
#     video_dir = '/home/exouser/VQA_tool/VQA_Toolkit/data/toy/evaluate/'

#     vie_dataset = TIFADataset(video_dir=video_dir,
#                               prompt_dir=prompt_dir)
#     unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
#     vqa_model = VQAModel("mplug-large")

#     input_prompt, input_frames = vie_dataset.__getitem__(0)
#     openai_setup(openai_key=openai_key, llm_model='gpt-3.5-turbo')
#     process(input_prompt=input_prompt, input_frames=input_frames, 
#             unifiedqa_model=unifiedqa_model, vqa_model=vqa_model)
    
    
