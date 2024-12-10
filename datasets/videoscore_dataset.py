from typing import Union, List

from mmengine.dataset import BaseDataset
import json
import re
import torch
from transformers import AutoProcessor

from core.registry import DATASETS
import av

@DATASETS.register_module()
class VideoScoreDataset(BaseDataset):
    def __init__(self, ann_file='', metainfo=None, data_root='', data_prefix={'img_path': ''}, filter_cfg=None, indices=None,
                 serialize_data=True, pipeline=[], test_mode=False, lazy_init=False, max_refetch=1000, model_name = None):
        super(VideoScoreDataset, self).__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)
        if model_name is None:
            self.model_name = 'TIGER-Lab/VideoScore-v1.1'
        else:
            self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(self.model_name,torch_dtype=torch.bfloat16)

    def __len__(self):
        return self.metainfo['length']

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        pass
    def __getitem__(self, idx):
        item = self.get_data_info(idx)






