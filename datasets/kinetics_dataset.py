# Copyright (c) IFM Lab. All rights reserved.


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import os
from os import path as osp

from core.registry import DATASETS
from mmengine.dataset import BaseDataset
from torch.utils.data import Dataset


from metrics.video_quality_assessment.nn_based.starvqa_plus import Kinetics, get_cfg
from fvcore.common.file_io import PathManager
# DATASETS.register_module(module=Test_VQADataset, force=True)

@DATASETS.register_module()
class KineticsDataset(Kinetics):
    """Dataset used in StarVQA. Size: train:400Gb, test:60Gb
    Datails:
    dataloader: https://github.com/GZHU-DVL/CoSTA/blob/main/build/lib/lib/datasets/loader.py#L82
    dataset: https://github.com/GZHU-DVL/CoSTA/blob/main/build/lib/lib/datasets/kinetics.py#L40
    config: https://github.com/GZHU-DVL/CoSTA/blob/main/configs/Kinetics/TimeSformer_divST_8x32_224.yaml
    download: https://github.com/cvdfoundation/kinetics-dataset

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
                 cfg_path:str='StarVQA_PLUS/configs/Kinetics/TimeSformer_divST_8x32_224.yaml',
                 mode='test',
                 num_retries:int=10,
                 data_dir:str='StarVQA_PLUS/dataset_csv/LSVQcsv/',
                ):
        # super().__init__()
        
        self.cfg_path = os.getcwd() + '/metrics/video_quality_assessment/nn_based/starvqa_plus/' + cfg_path
        self.cfg = get_cfg() # Get default config. See details in https://github.com/GZHU-DVL/CoSTA/blob/main/build/lib/lib/config/defaults.py
        if self.cfg_path is not None:
            self.cfg.merge_from_file(self.cfg_path) # Merge from config file. See details in https://github.com/GZHU-DVL/CoSTA/blob/main/configs/Kinetics/TimeSformer_divST_8x32_224.yaml
        
        assert mode in ['train', 'val', 'test'], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.num_retries = num_retries
        self.data_dir = data_dir

        self.path_to_file = os.getcwd() + '/metrics/video_quality_assessment/nn_based/starvqa_plus/' + self.data_dir + '{}.csv'.format(self.mode)
        
        assert PathManager.exists(self.path_to_file), "{} dir not found".format(
            self.path_to_file
        )

        with PathManager.open(self.path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                path, label = path_label.split(',')
                path=(f'{DATA_DIR}{path}.mp4')
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    # labels=One_Hot(np.float(label),6)
                    labels=Pro_Lab(np.float(label),5)

                    self._labels.append((labels))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        # See details in https://github.com/GZHU-DVL/CoSTA/blob/main/build/lib/lib/datasets/kinetics.py#L40
        self.dataset = Kinetics( 
            cfg=self.cfg,
            mode=self.mode,
            num_retries=self.num_retries,
        )

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx) -> Any:
        return super().__getitem__(idx)