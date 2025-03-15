# encoding = utf-8
# code based on: https://github.com/j-min/DSG/tree/main

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(LAST_SCRIPT_DIR))

import torch
import cv2
import time
import logging
import numpy as np

from core.registry import METRICS
from copy import deepcopy
from typing import Dict, Optional, Sequence, Union, List
from transformers import CLIPProcessor, CLIPModel

import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm
from utils import add_git_submodule, submodule_exists

@METRICS.register_module()
class DSGScore(BaseMetric):
    """ Initialize the ``DSGScore`` evaluator.
    
    Args:
        vqa_model_name (str): The name of the VQA model used in the DSGScore evaluator. Defaults to ``InstructBLIP``, you can also choose the "MPLUG" as the VQA model.
        verbose (bool): Whether the intermediate output processes is required. Defaults to False.
    """
    def __init__(self, 
                 vqa_model_name: str = "InstructBLIP",
                 verbose: bool = False):
        super().__init__()
        
        self.submodel_path = 'metrics/text_video_alignment/gpt_based/dsg'
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/j-min/DSG.git', 
                submodule_path=self.submodel_path
            )     
        from .DSG.dsg.vqa_utils import MPLUG, InstructBLIP

        self.vqa_model_name = vqa_model_name
        assert self.vqa_model_name in ["InstructBLIP", "MPLUG"]
        if self.vqa_model_name == 'InstructBLIP':
            self.vqa_model = InstructBLIP()
        else:
            self.vqa_model = MPLUG()

        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def evaluate_image_dsg(self, qid_list, frame_index, frame) -> Dict[str, Union[int, dict, float]]:
        """ Evaluate a generated image with DSG evaluator; this is the intermediate process of the ``process`` function. 
    
        Args:
            qid_list (List[str]): The list of DSG parse question generation results.
            frame_index (int): The index number of the currently evaluated frame.
            frame (List[List[float]]): The current evaluated frame.
    
        Returns:
            Dict[str, Union[int, dict, float]]: A dictionary containing evaluation results with the following keys:
                - 'frame_index' (int): The index of the evaluated frame.
                - 'qid2tuple' (dict): Mapping of question IDs to tuples.
                - 'qid2dependency' (dict): Mapping of question IDs to dependencies.
                - 'qid2question' (dict): Mapping of question IDs to actual questions.
                - 'qid2answer' (dict): Mapping of question IDs to predicted answers.
                - 'qid2scores' (dict): Mapping of question IDs to scores before dependency filtering.
                - 'qid2validity' (dict): Mapping of question IDs to boolean validity after dependency filtering.
                - 'average_score_with_dependency' (float): Average score considering dependency filtering.
                - 'average_score_without_dependency' (float): Average score before dependency filtering.
        """
        if self.verbose:
            print("#"*50)
            print("2) Answer questions given the generated image, with VQA")
            print("#"*50)

        # 2) answer questions with the generated image
        qid2answer = {}
        qid2scores = {}

        qid2tuple, qid2dependency, qid2question = qid_list
        for id, question in qid2question.items():
            answer = self.vqa_model.vqa(image=frame, question=question)
            print(answer)
            qid2answer[id] = answer
            qid2scores[id] = float('yes' in answer)
                
        average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
        print(average_score_without_dep, qid2answer, qid2scores)
            
        if self.verbose:
            print("#"*50)
            print("3) Zero-out scores from invalid questions")
            print("#"*50)
            
        # 3) zero-out scores from invalid questions 
        qid2validity = {}
        qid2scores_after_filtering = deepcopy(qid2scores)

        # print('qid2scores', qid2scores)
        # print('qid2dependency', qid2dependency)
        for id, parent_ids in qid2dependency.items():
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                parent_id = list(parent_id)[0]
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores_after_filtering[id] = 0.0
                qid2validity[id] = False
            else:
                qid2validity[id] = True
                
        if self.verbose:
            print("Per-quesiton eval results (after using dependency)")
            for id in qid2question:
                print("ID", id)
                print("question", qid2question[id])
                print("answer", qid2answer[id])
                print("validity", qid2validity[id])
                print("score (before filtering)", qid2scores[id])
                print("score (after filtering)", qid2scores_after_filtering[id])
                print()

        if self.verbose:
            print("#"*50)
            print("4) Calculate the final score by averaging")
            print("#"*50)

        average_score_with_dep = sum(qid2scores_after_filtering.values()) / len(qid2scores)
            
        return {
            'frame_index': frame_index,
            'qid2tuple': qid2tuple,
            'qid2dependency': qid2dependency,
            'qid2question': qid2question,
            'qid2answer': qid2answer,
            'qid2scores': qid2scores,
            'qid2validity': qid2validity,
            'average_score_with_dependency': average_score_with_dep,
            'average_score_without_dependency': average_score_without_dep
        }

    
    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """DSGScore process
        
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence): A batch of data from the dataloader.
            data_samples (Sequence): A batch of data samples that
                contain annotations and predictions.
        """

        result = dict()

        input_qid_lists, input_videos = data_samples
        bsz = len(input_qid_lists)
        # print('input_qid_lists: ', input_qid_lists)

        # Ensure prompt_input is a tensor
        if isinstance(input_qid_lists, tuple):
            input_qid_lists = list(input_qid_lists)
        
        if isinstance(input_videos, tuple):
            input_videos = list(input_videos)
        
        average_dep_score_list, average_wo_dep_score_list = [], []
        for input_qid_list, input_video in zip([input_qid_lists], input_videos):
            evaluate_dict_list = []
            dep_score, wo_dep_score = [], []
            for index, frame in enumerate(input_video):
                # print('input_qid_list: ', input_qid_list)
                evaluate_dict = self.evaluate_image_dsg(qid_list=input_qid_list, 
                                                        frame_index=index, 
                                                        frame=frame)
                evaluate_dict_list.append(evaluate_dict)
                frame_average_score_with_dependency = evaluate_dict['average_score_with_dependency']
                dep_score.append(frame_average_score_with_dependency)
                frame_average_score_without_dependency = evaluate_dict['average_score_without_dependency']
                wo_dep_score.append(frame_average_score_without_dependency)
            avg_dep_score, avg_wo_dep_score = sum(dep_score)/len(dep_score), sum(wo_dep_score)/len(dep_score)
            average_dep_score_list.append(avg_dep_score)
            average_wo_dep_score_list.append(avg_wo_dep_score)

    
        result['average_dep_dgs_score'] = sum(average_dep_score_list)/len(average_dep_score_list)
        result['average_wo_dep_dgs_score'] = sum(average_wo_dep_score_list)/len(average_wo_dep_score_list)

        self.results.append(result)


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        dep_dsg_score_np = np.zeros(len(results))
        wo_dep_dsg_score_np = np.zeros(len(results))
        for i, result in enumerate(results):
            dep_dsg_score_np[i] = result['average_dep_dgs_score']
            wo_dep_dsg_score_np[i] = result['average_wo_dep_dgs_score']
        
        dep_dsg_score_np_mean = np.mean(dep_dsg_score_np) 
        wo_dep_dsg_score_np_mean = np.mean(wo_dep_dsg_score_np)

        print("Test results: dsg score with dependency={:.4f}"
              .format(dep_dsg_score_np_mean))
        print("Test results: dsg score without dependency={:.4f}"
              .format(wo_dep_dsg_score_np_mean))

        return result
