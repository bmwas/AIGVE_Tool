# encoding = utf-8
# code based on: https://github.com/j-min/DSG/tree/main

import os
os.environ["OPENAI_API_KEY"] = ''
import torch
import cv2
import time
import logging
import openai
import numpy as np

from core.registry import METRICS
from copy import deepcopy
from typing import Dict, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from functools import lru_cache

# Lazy import to avoid circular import
@lru_cache(maxsize=1)
def lazy_import():
    from utils import add_git_submodule, submodule_exists
    submodel_path = 'metrics/text_video_alignment/gpt_based/dsg'
    if not submodule_exists(submodel_path):
        add_git_submodule(
            repo_url='https://github.com/j-min/DSG.git', 
            submodule_path=submodel_path
        )  
    submodel_path = 'metrics/text_video_alignment/gpt_based/TIFA'
    if not submodule_exists(submodel_path):
        add_git_submodule(
            repo_url='https://github.com/Yushi-Hu/tifa.git', 
            submodule_path=submodel_path
        )   
    from ..dsg.DSG.dsg.openai_utils import openai_completion
    from .tifa.tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
    
    return openai_completion, get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel


@METRICS.register_module()
class TIFAScore(BaseMetric):
    """ Initialize the ``TIFAScore`` evaluator.
    
    Args:   
        openai_key (str): The user's api key of the LLM models openai provides.
        llm_model (str): The name of the LLM model used in the TIFAScore evaluator. Defaults to ``gpt-3.5-turbo``.
        unifiedqa_model_name (str): The name of the ``UnifiedQAModel`` used in TIFAScore evaluator. Defaults to ``allenai/unifiedqa-v2-t5-large-1363200``.
        vqa_model_name (str): The name of the ``VQAModel used`` in TIFAScore evaluator. Defaults to ``mplug-large``.
    """
    def __init__(self, 
                 openai_key,
                 llm_model: str = 'gpt-3.5-turbo',
                 unifiedqa_model_name: str = 'allenai/unifiedqa-v2-t5-large-1363200',
                 vqa_model_name: str = 'mplug-large'):
        super().__init__()
        
        self.openai_key = openai_key
        self.llm_model = llm_model
        self.unifiedqa_model_name = unifiedqa_model_name
        self.openai_completion, self.get_question_and_answers, self.filter_question_and_answers, self.unifiedqa_model, self.tifa_score_single, self.vqa_model = lazy_import()
        self.unifiedqa_model = self.UnifiedQAModel(self.unifiedqa_model_name)
        self.vqa_model_name = vqa_model_name
        self.vqa_model = self.VQAModel(self.vqa_model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.openai_setup()
    
    def openai_setup(self):
        print('set up openai client')
        openai.api_key = self.openai_key
        assert openai.api_key is not None
        test_prompt_string = 'hello, how are you doing?'
        print('test prompt: ', test_prompt_string)
        response = self.openai_completion(
            test_prompt_string,
            model=self.llm_model,
        )
        print('test response: ', response)

    
    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """ TIFAScore process
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence): A batch of data from the dataloader.
            data_samples (Sequence): A batch of data samples that
                contain annotations and predictions.
        """

        result = dict()

        input_prompts, input_videos = data_samples
        bsz = len(input_prompts)

        # Ensure prompt_input is a tensor
        if isinstance(input_prompts, tuple):
            input_prompts = list(input_prompts)
        
        if isinstance(input_videos, tuple):
            input_videos = list(input_videos)
        
        average_tifa_score_list = []
        for input_prompt, input_video in zip(input_prompts, input_videos):
            tifa_score = []
            # Generate questions with GPT-3.5-turbo
            gpt3_questions = self.get_question_and_answers(input_prompt)
            # print(gpt3_questions)
            # Filter questions with UnifiedQA
            filtered_questions = self.filter_question_and_answers(self.unifiedqa_model, gpt3_questions)
            for index, frame_path in enumerate(input_video):
                # calucluate TIFA score
                result = self.tifa_score_single(self.vqa_model, filtered_questions, frame_path)
                # print(result)
                tifa_score.append(result['tifa_score'])
            average_tifa_score = sum(tifa_score)/len(tifa_score)
            average_tifa_score_list.append(average_tifa_score)
    
        result['tifa_score'] = sum(average_tifa_score_list)/len(average_tifa_score_list)

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

        tifa_score_np = np.zeros(len(results))
        for i, result in enumerate(results):
            tifa_score_np[i] = result['tifa_score']
        
        tifa_score_np_mean = np.mean(tifa_score_np) 

        print("Test results: tifa score={:.4f}"
              .format(tifa_score_np_mean))

        return result
