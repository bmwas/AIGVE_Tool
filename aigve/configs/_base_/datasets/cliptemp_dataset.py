# Copyright (c) IFM Lab. All rights reserved.

from mmengine.dataset.sampler import DefaultSampler
from datasets import CLIPTempDataset


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
