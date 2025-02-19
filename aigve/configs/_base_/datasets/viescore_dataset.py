# Copyright (c) IFM Lab. All rights reserved.
from mmengine.dataset.sampler import DefaultSampler
from vqa_datasets import VIEDataset

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VIEDataset,
        prompt_dir='/home/exouser/VQA_tool/VQA_Toolkit/data/toy/annotations/evaluate.json',
        video_dir='/home/exouser/VQA_tool/VQA_Toolkit/data/toy/evaluate/',
    )
)
