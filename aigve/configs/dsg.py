# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.gpt_based import DSGScore

with read_base():
    from ._base_.datasets.dsg_dataset import *
    from ._base_.default import *

openai_key = ''

val_evaluator = dict(
    type=DSGScore,
    vqa_model_name = "InstructBLIP",
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=DSGDataset,
        openai_key=openai_key,
        video_dir='AIGVE_Tool/data/toy/evaluate/',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
    )
)
