# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from datasets.videophy_dataset import VideoPhyDataset
from metrics.aigve.videophy.videophy_metric import VideoPhy
import torch
from metrics.aigve.videophy.vidophy_utils import batchify

with read_base():
    from ._base_.default import *

hf_token = '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'



val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VideoPhyDataset,
        data_path = 'AIGVE_Tool/data/toy/annotations/evaluate.json',
        max_length = 256,
        hf_token = hf_token,
        video_root_path='AIGVE_Tool/data/toy/evaluate'
    ),
    collate_fn=batchify
)

val_evaluator = dict(
    type=VideoPhy,
    collect_device='cpu',
    hf_token=hf_token,
    prefix='video_phy',
)