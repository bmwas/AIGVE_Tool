# Copyright (c) IFM Lab. All rights reserved.

from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.distribution_based import Toy
from datasets import ToyDataset

with read_base():
    # from ._base_.datasets.gstvqa_dataset import *
    from ._base_.default import *


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ToyDataset,
    )
)

val_evaluator = dict(
    type=Toy,
)
