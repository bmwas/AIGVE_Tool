# Copyright (c) IFM Lab. All rights reserved.

from .starvqa_metric import StarVQA
from .StarVQA.lib.datasets import Kinetics
from .StarVQA.lib.config.defaults import get_cfg

__all__ = ['StarVQA', 'Kinetics', 'get_cfg']