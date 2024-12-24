# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.gpt_based import VIEEvalScore

with read_base():
    from ._base_.datasets.viescore_dataset import *
    from ._base_.default import *

val_evaluator = dict(
    type=VIEEvalScore,
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VIEDataset,
        video_dir='/home/exouser/VQA_tool/VQA_Toolkit/data/toy/evaluate/',
        prompt_dir='/home/exouser/VQA_tool/VQA_Toolkit/data/toy/annotations/evaluate.json',
    )
)
