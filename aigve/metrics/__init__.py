# Copyright (c) IFM Lab. All rights reserved.

"""
This module provides the videos evaluation metrics that can be used within the AIGVE toolkit.
"""

from .multi_aspect_metrics import VideoPhy, VideoScore
from .text_video_alignment import DSGScore, TIFAScore, VIEEvalScore, \
        CLIPSimScore, CLIPTempScore, PickScore, BlipSimScore
from .video_quality_assessment import FIDScore, FVDScore, ISScore, \
        GstVqa, SimpleVqa, LightVQAPlus

__all__ = [
    # ---- multi_aspect_metrics ----
    'VideoPhy', 'VideoScore',
    # ---- text_video_alignment ----
    'DSGScore', 'TIFAScore', 'VIEEvalScore', 
    'CLIPSimScore', 'CLIPTempScore', 'PickScore', 'BlipSimScore',
    # ---- video_quality_assessment ----
    'FIDScore', 'FVDScore', 'ISScore',
    'GstVqa', 'SimpleVqa', 'LightVQAPlus'
    ]
