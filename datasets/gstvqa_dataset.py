# Copyright (c) IFM Lab. All rights reserved.


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
from os import path as osp

from core.registry import DATASETS
from mmengine.dataset import BaseDataset


from .TCSVT_Release.GVQA_Release.GVQA_Cross.cross_test import Test_VQADataset


DATASETS.register_module(module=Test_VQADataset, force=True)


# @DATASETS.register_module()
# class GSTVQADataset(BaseDataset):
#     """Dataset used in GSTVQA
#     Datails in: https://github.com/Baoliang93/GSTVQA

#     Args:
#         BaseDataset (_type_): _description_
#     """

#     def __init__(self,
#                  )