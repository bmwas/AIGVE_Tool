# Copyright (c) IFM Lab. All rights reserved.
from .distribution_based import FIDScore, FVDScore, ISScore
from .nn_based import GstVqa, SimpleVqa, LightVQAPlus

__all__ = ['FIDScore', 'FVDScore', 'ISScore',
           'GstVqa', 
           'SimpleVqa',
           'LightVQAPlus']