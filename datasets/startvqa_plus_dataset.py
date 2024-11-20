# Copyright (c) IFM Lab. All rights reserved.


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
from os import path as osp

from core.registry import DATASETS
from mmengine.dataset import BaseDataset


from metrics.video_quality_assessment.nn_based.gstvqa import Test_VQADataset


DATASETS.register_module(module=Test_VQADataset, force=True)