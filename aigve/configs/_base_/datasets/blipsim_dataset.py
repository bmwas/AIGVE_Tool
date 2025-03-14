# Copyright (c) IFM Lab. All rights reserved.

from mmengine.dataset.sampler import DefaultSampler
from datasets import BLIPSimDataset


val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BLIPSimDataset,
        video_dir='AIGVE_Tool/data/toy/evaluate/',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
    )
)
