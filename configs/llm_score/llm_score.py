# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base


with read_base():
    from .._base_.datasets.toy_dataset import *
    from .._base_.default import *

val_evaluator = dict(
    type= 
)