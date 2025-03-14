# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.similarity_based import CLIPSimScore

with read_base():
    from ._base_.datasets.clipsim_dataset import *
    from ._base_.default import *


val_evaluator = dict(
    type=CLIPSimScore,
    model_name='openai/clip-vit-base-patch32',
    logit_scale=False,
)

val_dataloader = dict(
    batch_size=2, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CLIPSimDataset,
        processor_name='openai/clip-vit-base-patch32',
        video_dir='AIGVE_Tool/data/toy/evaluate/',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
    )
)
