# Copyright (c) IFM Lab. All rights reserved.

__all__ = []

# Core/lightweight datasets
from .toy_dataset import ToyDataset
from .fid_dataset import FidDataset
from .gstvqa_dataset import GSTVQADataset
from .simplevqa_dataset import SimpleVQADataset
from .lightvqa_plus_dataset import LightVQAPlusDataset
from .clipsim_dataset import CLIPSimDataset
from .cliptemp_dataset import CLIPTempDataset
from .blipsim_dataset import BLIPSimDataset
from .pickscore_dataset import PickScoreDataset
from .vie_dataset import VIEDataset
from .tifa_dataset import TIFADataset
from .dsg_dataset import DSGDataset

__all__ += ['ToyDataset', 'FidDataset',
            'GSTVQADataset', 'SimpleVQADataset', 'LightVQAPlusDataset',
            'CLIPSimDataset', 'CLIPTempDataset', 'BLIPSimDataset', 'PickScoreDataset',
            'VIEDataset', 'TIFADataset', 'DSGDataset']

# Optional/heavy datasets guarded (may require extra deps like av/mantis/vbench)
try:
    from .videophy_dataset import VideoPhyDataset
    __all__ += ['VideoPhyDataset']
except Exception:
    pass

try:
    from .videoscore_dataset import VideoScoreDataset
    __all__ += ['VideoScoreDataset']
except Exception:
    pass

try:
    from .vbench_dataset import VbenchDataset
    __all__ += ['VbenchDataset']
except Exception:
    pass

