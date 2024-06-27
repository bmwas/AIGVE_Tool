# Copyright (c) IFM Lab. All rights reserved.

from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
from os import path as osp

from core.registry import DATASETS
from mmengine.dataset import BaseDataset



@DATASETS.register_module()
class ToyDataset(BaseDataset):
    """ToyDataset for testing.

    Args:
        data
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = None,
                 pipeline: List[Union[Callable, dict]] = [],
                 modality: dict = dict(use_video=True, use_text=True),
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            **kwargs
        )
        self.modality = modality
        assert self.modality['use_video'] or self.modality['use_text'], (
            'Please specify the `modality` (`use_video` '
            f', `use_text`) for {self.__class__.__name__}')
        
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw data info."""
        info = {}
        if self.modality['use_text']:
            info['prompt_gt'] = osp.join(self.data_prefix.get('video', ''), 
                                         raw_data_info['prompt_gt'])

        if self.modality['use_video']:
            info['video_path_pd'] = osp.join(self.data_prefix.get('video', ''), 
                                     raw_data_info['video_path_pd'])
                                     
        return info

