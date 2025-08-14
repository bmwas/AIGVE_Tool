# Copyright (c) IFM Lab. All rights reserved.

"""
This module provides the videos evaluation metrics that can be used within the AIGVE toolkit.
"""

__all__ = []

# Import distribution-based and NN-based VQA metrics (no optional mantis/vbench deps)
try:
    from .video_quality_assessment import FIDScore, FVDScore, ISScore, \
            GstVqa, SimpleVqa, LightVQAPlus
    __all__ += ['FIDScore', 'FVDScore', 'ISScore', 'GstVqa', 'SimpleVqa', 'LightVQAPlus']
except Exception:
    pass

# Import text-video alignment metrics if available
try:
    from .text_video_alignment import DSGScore, TIFAScore, VIEEvalScore, \
            CLIPSimScore, CLIPTempScore, PickScore, BlipSimScore
    __all__ += ['DSGScore', 'TIFAScore', 'VIEEvalScore', 'CLIPSimScore', 'CLIPTempScore', 'PickScore', 'BlipSimScore']
except Exception:
    pass

# Do not import multi-aspect metrics by default to avoid triggering optional
# dependencies (e.g., flash-attn) at import time. These can be imported
# explicitly by their module paths when needed, e.g.:
#   from aigve.metrics.multi_aspect_metrics.videophy.videophy_metric import VideoPhy
