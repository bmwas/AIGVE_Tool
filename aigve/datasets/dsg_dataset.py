# encoding = utf-8
import os
import sys
import cv2
import json
import openai

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(LAST_SCRIPT_DIR))

from copy import deepcopy
from torch.utils.data import Dataset
from core.registry import DATASETS
from functools import lru_cache
from utils import add_git_submodule, submodule_exists

# Lazy import to avoid circular import
@lru_cache(maxsize=1)
def lazy_import():
    from metrics.text_video_alignment.gpt_based.dsg.DSG.dsg.openai_utils import openai_completion
    from metrics.text_video_alignment.gpt_based.dsg.DSG.dsg.query_utils import generate_dsg
    from metrics.text_video_alignment.gpt_based.dsg.DSG.dsg.parse_utils import parse_tuple_output, parse_question_output, parse_dependency_output
    return openai_completion, generate_dsg, parse_tuple_output, parse_question_output, parse_dependency_output


@DATASETS.register_module()
class DSGDataset(Dataset):
    def __init__(self, openai_key, video_dir, prompt_dir, verbose=False, llm_model='gpt-3.5-turbo'):
        super().__init__()
        self.openai_key = openai_key
        self.llm_model = llm_model
        self._openai_setup()
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.verbose = verbose

        self.prompts, self.video_names = self._read_prompt_videoname()
        if self.verbose:
            print("#"*50)
            print("1) Generate DSG from text with LLM")
            print("#"*50)
        
        self.submodel_path = os.path.join(os.getcwd(), 'metrics/text_video_alignment/gpt_based/dsg')
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/j-min/DSG.git', 
                submodule_path=self.submodel_path
            )
        dsg_path = os.path.join(self.submodel_path, "DSG")
        if dsg_path not in sys.path:
            sys.path.insert(0, dsg_path)
        self.openai_completion, self.generate_dsg, self.parse_tuple_output, self.parse_question_output, self.parse_dependency_output = lazy_import()

        
    def _openai_setup(self):
        print('set up openai client')
        openai.api_key = self.openai_key
        assert openai.api_key is not None
        test_prompt_string = 'hello, how are you doing?'
        print('test prompt: ', test_prompt_string)
        response = openai_completion(
            test_prompt_string,
            model=self.llm_model,
        )
        print('test response: ', response)


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
    

    def _generated_gsg_with_llm(self, prompt_item):
        id2prompts = {
            'custom_0': {
            'input': prompt_item,
            }
        }   
        id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
                                                                        id2prompts,
                                                                        generate_fn=openai_completion)
        qid2tuple = parse_tuple_output(id2tuple_outputs['custom_0']['output'])
        qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
        qid2question = parse_question_output(id2question_outputs['custom_0']['output'])

        return [qid2tuple, qid2dependency, qid2question]
            

    def __len__(self):
        return len(self.prompts)
        # return 2
    
    def __getitem__(self, index):
        prompt, video_name = self.prompts[index], self.video_names[index]
        input_qid_list = self._generated_gsg_with_llm(prompt)
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

        return input_qid_list, input_frames

# DATASETS.register_module(module=DSGDataset, force=True)

# def evaluate_image_dsg(qid_list, frame_index, frame, VQA, verbose=False):
#     '''evaluate a generated image with DSG
#     '''
#     if verbose:
#         print("#"*50)
#         print("2) Answer questions given the generated image, with VQA")
#         print("#"*50)

#     # 2) answer questions with the generated image
#     qid2answer = {}
#     qid2scores = {}

#     qid2tuple, qid2dependency, qid2question = qid_list
#     for id, question in qid2question.items():
#         answer = VQA.vqa(image=frame, question=question)
#         print(answer)
#         qid2answer[id] = answer
#         qid2scores[id] = float('yes' in answer)
            
#     average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
#     print(average_score_without_dep, qid2answer, qid2scores)
        
#     if verbose:
#         print("#"*50)
#         print("3) Zero-out scores from invalid questions")
#         print("#"*50)
        
#     # 3) zero-out scores from invalid questions 
#     qid2validity = {}
#     qid2scores_after_filtering = deepcopy(qid2scores)

#     for id, parent_ids in qid2dependency.items():
#         # zero-out scores if parent questions are answered 'no'
#         any_parent_answered_no = False
#         for parent_id in parent_ids:
#             if parent_id == 0:
#                 continue
#             if qid2scores[parent_id] == 0:
#                 any_parent_answered_no = True
#                 break
#         if any_parent_answered_no:
#             qid2scores_after_filtering[id] = 0.0
#             qid2validity[id] = False
#         else:
#             qid2validity[id] = True
            
#     if verbose:
#         print("Per-quesiton eval results (after using dependency)")
#         for id in qid2question:
#             print("ID", id)
#             print("question", qid2question[id])
#             print("answer", qid2answer[id])
#             print("validity", qid2validity[id])
#             print("score (before filtering)", qid2scores[id])
#             print("score (after filtering)", qid2scores_after_filtering[id])
#             print()

#     if verbose:
#         print("#"*50)
#         print("4) Calculate the final score by averaging")
#         print("#"*50)

#     average_score_with_dep = sum(qid2scores_after_filtering.values()) / len(qid2scores)
        
#     return {
#         'frame_index': frame_index,
#         'qid2tuple': qid2tuple,
#         'qid2dependency': qid2dependency,
#         'qid2question': qid2question,
#         'qid2answer': qid2answer,
#         'qid2scores': qid2scores,
#         'qid2validity': qid2validity,
#         'average_score_with_dependency': average_score_with_dep,
#         'average_score_without_dependency': average_score_without_dep
#     }


# def evaluate_video_dsg(qid_list, video, VQA, verbose=False):
#     evaluate_dict_list = []
#     dep_score, wo_dep_score = [], []
#     for index, frame in enumerate(video):
#         if index > 0:
#             break
#         evaluate_dict = evaluate_image_dsg(qid_list=qid_list, 
#                                            frame_index=index, 
#                                            frame=frame, 
#                                            VQA=VQA,
#                                            verbose=verbose)
#         evaluate_dict_list.append(evaluate_dict)
#         frame_average_score_with_dependency = evaluate_dict['average_score_with_dependency']
#         dep_score.append(frame_average_score_with_dependency)
#         frame_average_score_without_dependency = evaluate_dict['average_score_without_dependency']
#         wo_dep_score.append(frame_average_score_without_dependency)
#     avg_dep_score, avg_wo_dep_score = sum(dep_score)/len(dep_score), sum(wo_dep_score)/len(dep_score)

#     print('avg_dep_score', avg_dep_score)
#     print('avg_wo_dep_score', avg_wo_dep_score)


# if __name__ == '__main__':
#     openai_key = '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'
#     prompt_dir = 'AIGVE_Tool/data/toy/annotations/evaluate.json'
#     video_dir = 'AIGVE_Tool/data/toy/evaluate/'

#     dsg_dataset = DSGDataset(openai_key=openai_key,
#                              video_dir=video_dir,
#                              prompt_dir=prompt_dir)
#     input_qid_list, input_frames = dsg_dataset.__getitem__(0)
#     vqa_model = InstructBLIP()
#     # vqa_model = MPLUG()
#     evaluate_video_dsg(qid_list=input_qid_list, video=input_frames, VQA=vqa_model, verbose=True)
    
    

    


