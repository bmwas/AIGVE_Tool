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

@DATASETS.register_module()
class VbenchDataset(BaseDataset):
    def __init__(self, ann_file='', metainfo=None, data_root='', data_prefix={'video_path_pd': ''}, filter_cfg=None, indices=None,
                 serialize_data=True, pipeline=[], test_mode=False, lazy_init=False, max_refetch=1000,
                 ):
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

        """
        super(VbenchDataset, self).__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


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
        prompt = anno_info['prompt_gt']

        inputs = {
            'video_path': video_path,
            'prompt': prompt,
        }
        return inputs









