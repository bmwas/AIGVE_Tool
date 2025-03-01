# Copyright (c) IFM Lab. All rights reserved.

from .toy_dataset import ToyDataset

from .gstvqa_dataset import GSTVQADataset
from .gstvqa_crossdata_dataset import GSTVQADatasetCrossData
from .simplevqa_dataset import SimpleVQADataset
# from .konvid_1k_dataset import KONVID1KDataset_ModularBVQA

from .clipsim_dataset import CLIPSimDataset
from .cliptemp_dataset import CLIPTempDataset
from .blipsim_dataset import BLIPSimDataset
from .pickscore_dataset import PickScoreDataset
# from .kinetics_dataset import KineticsDataset

from .videophy_dataset import VideoPhyDataset

__all__ = ['ToyDataset', 
           'GSTVQADataset', 'GSTVQADatasetCrossData', 'SimpleVQADataset', 'KONVID1KDataset_ModularBVQA',
           'CLIPSimDataset', 'KineticsDataset', 'CLIPTempDataset', 'BLIPSimDataset', 'PickScoreDataset',
           'VideoPhyDataset']

