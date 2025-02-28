# Copyright (c) IFM Lab. All rights reserved.
from utils import LoadVideoFromFile
from mmengine.dataset.sampler import DefaultSampler
from aigve.datasets import ToyDataset


eva_pipeline = [
    dict(type=LoadVideoFromFile, height = -1, width = -1)
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ToyDataset,
        data_root='data/toy/',
        ann_file='annotations/evaluate.json',
        data_prefix=dict(video='evaluate'),
        pipeline=eva_pipeline,
        backend_args=None,
        modality=dict(use_video=True, use_text=True, use_image=False),
        image_frame=None,
        test_mode=True,
    )
)
