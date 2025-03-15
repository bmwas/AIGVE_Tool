# Copyright (c) IFM Lab. All rights reserved.

r"""
This aigve library provides a **comprehensive** and **structured** evaluation framework 
for assessing AI-generated video quality. It integrates multiple evaluation metrics, 
covering diverse aspects of video evaluation, including neural-network-based assessment, 
distribution comparison, vision-language alignment, and multi-faceted analysis.
"""


__version__ = '0.0.1'

# from . import model, zootopia
# from . import module, head, layer, config
# from . import expansion, compression, transformation, reconciliation, remainder, interdependence, fusion
# from . import koala
# from . import data, output
from .datasets import ToyDataset, \
    GSTVQADataset, SimpleVQADataset, LightVQAPlusDataset, \
    CLIPSimDataset, CLIPTempDataset, BLIPSimDataset, PickScoreDataset, \
    VIEDataset, TIFADataset, DSGDataset, \
    VideoPhyDataset, VideoScoreDataset
from .metrics import VideoPhy, VideoScore, \
    DSGScore, TIFAScore, VIEEvalScore, CLIPSimScore, CLIPTempScore, PickScore, BlipSimScore, \
    FIDScore, FVDScore, \
    GstVqa, SimpleVqa, LightVQAPlus
# from . import visual, util


__all__ = [
    # ---- models and applications ----
    # ---- modules ----
    # ---- component functions ----
    # ---- other libraries ----
    # ---- dataloaders ----
    'ToyDataset', 
    'GSTVQADataset', 'SimpleVQADataset', 'LightVQAPlusDataset',
    'CLIPSimDataset', 'CLIPTempDataset', 'BLIPSimDataset', 'PickScoreDataset',
    'VIEDataset', 'TIFADataset', 'DSGDataset',
    'VideoPhyDataset', 'VideoScoreDataset'
    # ---- metrics ----
    'FIDScore',
    'FVDScore',

    'GstVqa', 
    'SimpleVqa',
    'LightVQAPlus',
    
    'CLIPSimScore', 
    'CLIPTempScore', 
    'BlipSimScore'
    'PickScore', 

    'VIEEvalScore',
    'TIFAScore', 
    'DSGScore', 
    
    'VideoPhy', 
    'VideoScore',

    # ---- visualization and utility ----
]