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
from metrics import VideoPhy, VideoScore, \
    DSGScore, TIFAScore, VIEEvalScore, \
    CLIPSimScore, CLIPTempScore, PickScore, BlipSimScore
# from . import visual, util


__all__ = [
    # ---- models and applications ----
    # ---- modules ----
    # ---- component functions ----
    # ---- other libraries ----
    # ---- input and output ----
    # ---- metrics ----
    'VideoPhy', 
    'VideoScore',
    'DSGScore', 
    'TIFAScore', 
    'VIEEvalScore',
    'CLIPSimScore', 
    'CLIPTempScore', 
    'PickScore', 
    'BlipSimScore'
    # ---- visualization and utility ----
]