# Copyright (c) IFM Lab. All rights reserved.
from .gstvqa import GstVqa
from .simplevqa import SimpleVqa
from .lightvqa_plus import LightVQAPlus
# from .starvqa_plus import StarVQAplus, Kinetics
# from .modular_bvqa import ModularBVQA

__all__ = ['GstVqa', 
           'SimpleVqa', 
           'LightVQAPlus']