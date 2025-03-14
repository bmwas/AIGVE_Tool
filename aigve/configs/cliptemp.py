# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.similarity_based import CLIPTempScore

with read_base():
    from ._base_.datasets.cliptemp_dataset import *
    from ._base_.default import *


val_evaluator = dict(
    type=CLIPTempScore,
    model_name='openai/clip-vit-base-patch32',
)

val_dataloader = dict(
    batch_size=4, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CLIPTempDataset,
        processor_name='openai/clip-vit-base-patch32',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
        video_dir='AIGVE_Tool/data/toy/evaluate/',
    )
)
