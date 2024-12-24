# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.gpt_based import TIFAScore

with read_base():
    from ._base_.datasets.tifa_dataset import *
    from ._base_.default import *

openai_key = ''

val_evaluator = dict(
    type=TIFAScore,
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=TIFADataset,
        video_dir='/home/exouser/VQA_tool/VQA_Toolkit/data/toy/evaluate/',
        prompt_dir='/home/exouser/VQA_tool/VQA_Toolkit/data/toy/annotations/evaluate.json',
    )
)
