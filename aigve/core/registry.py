# Copyright (c) IFM Lab. All rights reserved.

from mmengine.registry import Registry
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import MODELS as MMENGINE_MODELS

LOOPS = Registry('loop', parent=MMENGINE_LOOPS, locations=['core.loops'])
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['datasets'])
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS, locations=['transforms'])
METRICS = Registry('metric', parent=MMENGINE_METRICS, locations=['metrics'])
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['core.models'])

