# Copyright (c) IFM Lab. All rights reserved.


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
from os import path as osp

from core.registry import DATASETS
# Importfrom mmengine.dataset import BaseDataset
from torch.utils.data import Dataset


# from metrics.video_quality_assessment.nn_based.gstvqa import Test_VQADataset
import h5py


@DATASETS.register_module()
class GSTVQADatasetCrossData(Dataset):
    """Dataset used in GSTVQA
    Datails in: https://github.com/Baoliang93/GSTVQA

    Args:
        feature_dir (str): Path to the feature directory.
        index (Optional): Index for dataset filtering.
        max_len (int): Maximum sequence length. Default is 500.
        feat_dim (int): Feature dimension. Default is 2944.
        datainfo_path (str): Path to dataset information file.
    """

    def __init__(self,
                 feature_dir, 
                 index=None, 
                 max_len=500, 
                 feat_dim=2944,
                 datainfo_path=''):
        super().__init__()
        self.feature_dir = feature_dir
        self.index = index
        self.max_len = max_len
        self.feat_dim = feat_dim

        self.datainfo_path = datainfo_path
        Info = h5py.File(name=datainfo_path, mode='r') # Runtime Error otherwise
        self.scale = Info['scores'][0, :].max() 

        # See details in https://github.com/Baoliang93/GSTVQA/blob/8463c9c3e5720349606d8efae7a5aa274bf69e7c/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py#L30
        self.dataset = Test_VQADataset( 
            features_dir=self.feature_dir, 
            index=self.index, 
            max_len=self.max_len, 
            feat_dim=self.feat_dim, 
            scale=self.scale
        )

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx) -> Any:
        return self.dataset.__getitem__(idx)
        