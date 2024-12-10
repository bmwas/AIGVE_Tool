
from typing import Dict, List, Optional, Sequence, Union, Any

from mantis.models.idefics2 import Idefics2ForSequenceClassification
from mmengine.evaluator import BaseMetric
from sympy.logic.inference import entails
from transformers import LlamaTokenizer

from core.registry import METRICS
from mmengine.logging import MMLogger
import torch
from .videoscore_utils import _read_video_pyav
import torch.nn as nn
import numpy as np

@METRICS.register_module()
class VideoScore(BaseMetric):
    def __init__(self,
                collect_device: Optional[Union[str, torch.device]] = None,
                prefix: Optional[str] = None,
                metric_path: str = None,
                model_path: str = 'TIGER-Lab/VideoScore-v1.1',
                datainfo_path: str = None,
                test_index: int = None,
                regression_query_prompt: str = None,
                max_num_frames: int = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        # self.train_index = train_index
        # TODO: ARE THERE PARAMETERS REQUIRED FOR THIS METRIC?
        self.metric_path = metric_path
        self.model_path = model_path
        self.datainfo_path = datainfo_path
        self.test_index = test_index

        if regression_query_prompt is not None:
            self.regression_query_prompt = regression_query_prompt
        else:
            self.regression_query_prompt = '''
                Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
                please watch the following frames of a given video and see the text prompt for generating the video,
                then give scores from 5 different dimensions:
                (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
                (2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
                (3) dynamic degree, the degree of dynamic changes
                (4) text-to-video alignment, the alignment between the text prompt and the video content
                (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge
                for each dimension, output a float number from 1.0 to 4.0,
                the higher the number is, the better the video performs in that sub-score, 
                the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
                Here is an output example:
                visual quality: 3.2
                temporal consistency: 2.7
                dynamic degree: 4.0
                text-to-video alignment: 2.3
                factual consistency: 1.8
                For this video, the text prompt is "{text_prompt}",
                all the frames of video are as follows:
            '''
        if max_num_frames is not None:
            self.max_num_frames = max_num_frames
        else:
            self.max_num_frames = 48

        self.model = Idefics2ForSequenceClassification.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.results = []