# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from datasets.videoscore_dataset import VideoScoreDataset
from metrics.aigve.videoscore.videoscore_metric import VideoScore
import torch

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=1,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VideoScoreDataset,
        ann_file = 'AIGVE_Tool/data/toy/annotations/evaluate.json',
        data_root='AIGVE_Tool/data/toy/evaluate'
    ),
)

val_evaluator = dict(
    type=VideoScore,
    collect_device='cpu',
    prefix='videoscore',
)