# Copyright (c) IFM Lab. All rights reserved.

from mmengine.model import BaseModel
from .registry import MODELS
import math

@MODELS.register_module()
class AIGVEModel(BaseModel):
    
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
    
    def forward(self, x):
        return x
