# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based import StarVQA
from datasets import KineticsDataset

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=KineticsDataset,
        cfg_path='StarVQA/configs/Kinetics/TimeSformer_divST_8x32_224.yaml',
        mode='test',
        num_retries=10
    )
)

val_evaluator = dict(
    type=StarVQA,
    cfg_path='StarVQA/configs/Kinetics/TimeSformer_divST_8x32_224.yaml',
    model_name='vit_base_patch16_224',
)
