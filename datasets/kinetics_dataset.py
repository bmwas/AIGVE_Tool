# Copyright (c) IFM Lab. All rights reserved.


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import os
from os import path as osp

from core.registry import DATASETS
from mmengine.dataset import BaseDataset


from metrics.video_quality_assessment.nn_based.starvqa import Kinetics, get_cfg
# DATASETS.register_module(module=Test_VQADataset, force=True)

@DATASETS.register_module()
class KineticsDataset(Dataset):
    """Dataset used in StarVQA
    Datails:
    dataloader: https://github.com/GZHU-DVL/StarVQA/blob/main/lib/datasets/loader.py#L82
    dataset: https://github.com/GZHU-DVL/StarVQA/blob/main/lib/datasets/kinetics.py#L44
    config: https://github.com/GZHU-DVL/StarVQA/blob/main/configs/Kinetics/TimeSformer_divST_8x32_224.yaml

    Args:
        cfg_path (string): configs path. The config is a CfgNode.
        mode (string): Options includes `train`, `val`, or `test` mode.
            For the train and val mode, the data loader will take data
            from the train or val set, and sample one clip per video.
            For the test mode, the data loader will take data from test set,
            and sample multiple clips per video.
        num_retries (int): number of retries.
    """

    def __init__(self,
                 cfg_path:str='',
                 mode='test',
                 num_retries=10):
        super().__init__()
        self.cfg_path = os.getcwd() + '/metrics/video_quality_assessment/nn_based/starvqa/' + cfg_path
        self.cfg = get_cfg() # Get default config. See details in https://github.com/GZHU-DVL/StarVQA/blob/main/lib/config/defaults.py
        if self.cfg_path is not None:
            self.cfg.merge_from_file(self.cfg_path) # Merge from config file. See details in https://github.com/GZHU-DVL/StarVQA/blob/main/configs/Kinetics/TimeSformer_divST_8x32_224.yaml
        self.mode = mode
        self.num_retries = num_retries

        # See details in https://github.com/GZHU-DVL/StarVQA/blob/main/lib/datasets/kinetics.py#L44
        self.dataset = Kinetics( 
            cfg=self.cfg,
            mode=self.mode,
            num_retries=self.num_retries,
        )

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx) -> Any:
        return self.dataset.__getitem__(idx)