# Copyright (c) IFM Lab. All rights reserved.

from mmengine.dataset.sampler import DefaultSampler
from datasets import Test_VQADataset


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=Test_VQADataset,
        featureaa_dir = '',
        index=None,
        max_len=500,
        feat_dim=2944,
        scale=1,
    )
)
