from typing import Union, List

from mmengine.dataset import BaseDataset
import json
import re
import torch
from transformers import AutoProcessor

from core.registry import DATASETS
import os
import av
import numpy as np

from PIL import Image

from metrics.multi_aspect_metrics.videoscore.videoscore_utils import _read_video_pyav


@DATASETS.register_module()
class VideoScoreDataset(BaseDataset):
    def __init__(self, ann_file='', metainfo=None, data_root='', data_prefix={'video_path_pd': ''}, filter_cfg=None, indices=None,
                 serialize_data=True, pipeline=[], test_mode=False, lazy_init=False, max_refetch=1000, model_name = None, regression_query_prompt: str = None,
                max_num_frames: int = None):
        """
        Args:
            ann_file (str): annotation file path
            metainfo (dict): meta information about the dataset
            data_root (str): the root path of the data
            data_prefix (dict): the prefix of the data, for example, the prefix of the image path
            filter_cfg (dict): the filter configuration
            indices (list): the indices of the data
            serialize_data (bool): whether to serialize the data
            pipeline (list): the pipeline of the data
            test_mode (bool): whether in test mode
            lazy_init (bool): whether to lazy initialize the dataset
            max_refetch (int): the maximum number of refetching data
            model_name (str): the name of the model
            regression_query_prompt (str): the prompt for the regression query
            max_num_frames (int): the maximum number of frames
        """
        super(VideoScoreDataset, self).__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)
        if model_name is None:
            self.model_name = 'TIGER-Lab/VideoScore-v1.1'
        else:
            self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(self.model_name,torch_dtype=torch.bfloat16)

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

    def __len__(self) -> int:
        """
        Returns:
            int: the length of the dataset
        """
        return self.metainfo['length']


    def __getitem__(self, idx):
        """
        Args:
            idx (int): the index of the data
        """
        anno_info = self.get_data_info(idx)
        video_path = os.path.join(self.data_root, anno_info['video_path_pd'])

        container = av.open(video_path)

        total_frames = container.streams.video[0].frames
        if total_frames > self.max_num_frames:
            indices = np.arange(0, total_frames, total_frames / self.max_num_frames).astype(int)
        else:
            indices = np.arange(total_frames)

        frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
        eval_prompt = self.regression_query_prompt.format(text_prompt=anno_info['prompt_gt'])
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < len(frames):
            eval_prompt += "<image> " * (len(frames) - num_image_token)

        flatten_images = []
        for x in [frames]:
            if isinstance(x, list):
                flatten_images.extend(x)
            else:
                flatten_images.append(x)
        flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
        inputs = self.processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
        return inputs









