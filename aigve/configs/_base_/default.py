# Copyright (c) IFM Lab. All rights reserved.
from core import AIGVELoop, AIGVEModel

default_scope = None

log_level = 'INFO'

model = dict(type=AIGVEModel)

default_hooks = None # Execute default hook actions as https://github.com/open-mmlab/mmengine/blob/85c83ba61689907fb1775713622b1b146d82277b/mmengine/runner/runner.py#L1896

val_cfg = dict(type=AIGVELoop)

