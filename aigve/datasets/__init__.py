# Copyright (c) IFM Lab. All rights reserved.

__all__ = []

# Core/lightweight datasets that shouldn't pull optional deps
from .toy_dataset import ToyDataset
from .fid_dataset import FidDataset
__all__ += ['ToyDataset', 'FidDataset']

# NN-based video datasets (depend on torch, which is part of the env)
try:
    from .gstvqa_dataset import GSTVQADataset
    __all__ += ['GSTVQADataset']
except Exception:
    pass

try:
    from .simplevqa_dataset import SimpleVQADataset
    __all__ += ['SimpleVQADataset']
except Exception:
    pass

try:
    from .lightvqa_plus_dataset import LightVQAPlusDataset
    __all__ += ['LightVQAPlusDataset']
except Exception:
    pass

# Text-video alignment datasets (may require transformers/tokenizers, etc.)
for _mod, _name in [
    ('.clipsim_dataset', 'CLIPSimDataset'),
    ('.cliptemp_dataset', 'CLIPTempDataset'),
    ('.blipsim_dataset', 'BLIPSimDataset'),
    ('.pickscore_dataset', 'PickScoreDataset'),
    ('.vie_dataset', 'VIEDataset'),
    ('.tifa_dataset', 'TIFADataset'),
    ('.dsg_dataset', 'DSGDataset'),
]:
    try:
        _m = __import__(__name__ + _mod, fromlist=[_name])
        globals()[_name] = getattr(_m, _name)
        __all__ += [_name]
    except Exception:
        # Optional deps not available; keep package importable for other use-cases
        pass

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

