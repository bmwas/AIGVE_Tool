# Copyright (c) IFM Lab. All rights reserved.
from .gstvqa import GSTVQA, GSTVQACrossData
from .simplevqa import SimpleVQA
# from .starvqa_plus import StarVQAplus, Kinetics
# from .modular_bvqa import ModularBVQA

__all__ = ['GSTVQA', 'GSTVQACrossData', 
           'SimpleVQA',
           'StarVQAplus', 'Kinetics', 'ModularBVQA']