# Copyright (c) IFM Lab. All rights reserved.

from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
from os import path as osp

from core.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class ToyDataset(BaseDataset):
    """ToyDataset for testing.

    Args:
        data_root (str, optional): Root directory for data.
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Metadata information.
        data_prefix (dict): Prefix paths for different modalities.
        pipeline (List[Union[Callable, dict]]): Data transformation pipeline.
        modality (dict): Specifies which modalities are used (video, text, image).
        image_frame (int, optional): Number of frames for images.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = None,
                 pipeline: List[Union[Callable, dict]] = [],
                 modality: dict = dict(use_video=True, use_text=True, use_image=False),
                 image_frame: int = None,
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
        self.image_frame = image_frame
        assert self.modality['use_video'] or self.modality['use_text'], (
            'Please specify the `modality` (`use_video` '
            f', `use_text`) for {self.__class__.__name__}')
        
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw data info."""
        info = {}
        info['img_frame'] = None
        if self.modality['use_text']:
            info['prompt_gt'] = osp.join(self.data_prefix.get('video', ''), 
                                         raw_data_info['prompt_gt'])

        if self.modality['use_video'] or self.modality['use_image']:
            info['video_path_pd'] = osp.join(self.data_prefix.get('video', ''), 
                                     raw_data_info['video_path_pd'])
            if self.modality['use_image']:
                info['img_frame'] = self.image_frame
                                     
        return info

