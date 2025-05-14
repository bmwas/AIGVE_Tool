# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from datasets.vbench_dataset import VbenchDataset
from metrics.multi_aspect_metrics.vbench.vbench_metric import VbenchMetric
import torch

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=1,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VbenchDataset,
        ann_file = 'AIGVE_Tool/data/toy/annotations/evaluate.json',
        data_root='AIGVE_Tool/data/toy/evaluate',

    ),
)

val_evaluator = dict(
    type=VbenchMetric,
    collect_device='cpu',
    prefix='videoscore',
    eval_aspects=['background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'], # 'subject_consistency',
    eval_mode='custom_input',
)